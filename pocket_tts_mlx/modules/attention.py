"""Streaming attention modules for MLX with KV caches and RoPE."""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Tuple

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.modules.rope import RotaryEmbedding
from pocket_tts_mlx.modules.stateful_module import StatefulModule


class KVCacheResult(NamedTuple):
    """Container for cached keys/values and their positions."""
    keys: mx.array
    values: mx.array
    positions: mx.array

    @staticmethod
    def from_kv(keys: mx.array, values: mx.array) -> "KVCacheResult":
        B, H, T, D = keys.shape
        assert values.shape[:-1] == (B, H, T)
        positions = mx.arange(T, dtype=mx.int64)
        positions = mx.broadcast_to(positions, (B, T))
        return KVCacheResult(keys, values, positions)


def materialize_causal_mask(shape: Tuple[int, int]) -> mx.array:
    """Create additive causal mask with -1e9 for masked positions."""
    num_queries, num_keys = shape
    shift = num_keys - num_queries
    mask = mx.tril(mx.ones((num_queries, num_keys), dtype=mx.float32), k=shift)
    mask = mx.where(
        mask > 0,
        mx.zeros(mask.shape, dtype=mx.float32),
        mx.full(mask.shape, -1e9, dtype=mx.float32),
    )
    return mask


def complete_kv(cache: mx.array, current_end: mx.array, k: mx.array, v: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
    """Append new k/v to cache for autoregressive decoding."""
    current_pos = current_end.shape[0]
    t_new = k.shape[1]

    # cache: [2, B, T_max, H, D], k/v: [B, T, H, D]
    k_update = k[None, ...]
    v_update = v[None, ...]
    cache = mx.slice_update(
        cache,
        k_update,
        mx.array([0, 0, current_pos, 0, 0]),
        axes=[0, 1, 2, 3, 4],
    )
    cache = mx.slice_update(
        cache,
        v_update,
        mx.array([1, 0, current_pos, 0, 0]),
        axes=[0, 1, 2, 3, 4],
    )

    valid = cache[:, :, : current_pos + t_new]
    return cache, valid[0], valid[1]


def complete_mimi_kv(cache: mx.array, end_offset: mx.array, k: mx.array, v: mx.array) -> KVCacheResult:
    """Ring-buffer KV cache for Mimi streaming attention."""
    capacity = cache.shape[3]
    B, H, T, D = k.shape
    indexes = mx.arange(T, dtype=end_offset.dtype)
    indexes = mx.broadcast_to(indexes, (B, T)) + end_offset.reshape(B, 1)
    indexes = indexes % capacity

    cache_keys = cache[0]
    cache_values = cache[1]

    # MLX has no scatter; update with small loops over the streaming window.
    for b in range(B):
        for t in range(T):
            idx = int(indexes[b, t])
            k_update = k[b, :, t, :].reshape(1, H, 1, D)
            v_update = v[b, :, t, :].reshape(1, H, 1, D)
            start = mx.array([b, 0, idx, 0])
            cache_keys = mx.slice_update(cache_keys, k_update, start, axes=[0, 1, 2, 3])
            cache_values = mx.slice_update(cache_values, v_update, start, axes=[0, 1, 2, 3])

    keys = cache_keys
    values = cache_values

    full_indexes = mx.arange(capacity, dtype=mx.int64).reshape(1, capacity)
    last_offset = end_offset + T - 1
    end_index = last_offset % capacity
    delta = full_indexes - end_index.reshape(B, 1)
    positions = mx.where(
        delta <= 0,
        last_offset.reshape(B, 1) + delta,
        last_offset.reshape(B, 1) + delta - capacity,
    )

    new_end_offset = end_offset + T
    invalid = full_indexes >= new_end_offset.reshape(B, 1)
    positions = mx.where(invalid, mx.full(positions.shape, -1, dtype=positions.dtype), positions)

    return KVCacheResult(keys, values, positions), new_end_offset


class StreamingMultiheadAttention(StatefulModule):
    """Streaming multi-head attention with causal masking and RoPE."""
    def __init__(self, embed_dim: int, num_heads: int, rope: RotaryEmbedding):
        super().__init__()
        self.embed_dim = embed_dim
        self.rope = rope
        self.num_heads = num_heads

        out_dim = embed_dim
        num_kv = num_heads
        kv_dim = (embed_dim // num_heads) * num_kv
        out_dim += 2 * kv_dim

        self.in_proj = nn.Linear(embed_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def init_state(self, batch_size: int, sequence_length: int) -> Dict[str, mx.array]:
        """Initialize empty KV cache for streaming decoding."""
        dim_per_head = self.embed_dim // self.num_heads
        initial_current_end = mx.zeros((0,), dtype=mx.int64)
        cache = mx.full(
            (2, batch_size, sequence_length, self.num_heads, dim_per_head),
            mx.nan,
        )
        return {"current_end": initial_current_end, "cache": cache}

    def increment_step(self, state: Dict[str, mx.array], increment: int = 1) -> None:
        """Advance streaming position for causal masking."""
        new_size = state["current_end"].shape[0] + increment
        state["current_end"] = mx.zeros((new_size,), dtype=mx.int64)

    def _complete_kv(self, k: mx.array, v: mx.array, state: Dict[str, mx.array]):
        """Update cache and return full k/v."""
        cache, k_full, v_full = complete_kv(state["cache"], state["current_end"], k, v)
        state["cache"] = cache
        return k_full, v_full

    def _apply_rope(self, q: mx.array, k: mx.array, state: Dict[str, mx.array]):
        """Apply RoPE using the current streaming offset."""
        streaming_offset = state["current_end"].shape[0]
        return self.rope(q, k, offset=streaming_offset)

    def __call__(self, query: mx.array, model_state: Dict[str, Dict[str, mx.array]] | None):
        """Compute attention for the current chunk and update cache."""
        if model_state is None:
            raise ValueError("model_state must be provided")
        state = self.get_state(model_state)

        projected = self.in_proj(query)
        b, t, _ = projected.shape
        d = self.embed_dim // self.num_heads
        packed = projected.reshape(b, t, 3, self.num_heads, d)
        q, k, v = mx.split(packed, 3, axis=2)
        q = q.squeeze(2)
        k = k.squeeze(2)
        v = v.squeeze(2)
        q, k = self._apply_rope(q, k, state)
        k, v = self._complete_kv(k, v, state)

        mask_shape = (query.shape[1], query.shape[1] + state["current_end"].shape[0])
        attn_mask = materialize_causal_mask(mask_shape)

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        scale = 1.0 / mx.sqrt(mx.array(d, dtype=mx.float32))
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale
        scores = scores + attn_mask[None, None, :, :]
        weights = mx.softmax(scores, axis=-1)
        x = mx.matmul(weights, v)

        x = mx.transpose(x, (0, 2, 1, 3))
        x = x.reshape(b, t, self.num_heads * d)
        return self.out_proj(x)


class MimiStreamingMultiheadAttention(StatefulModule):
    """Streaming attention with fixed context window for Mimi."""
    def __init__(self, embed_dim: int, num_heads: int, context: int, rope: RotaryEmbedding):
        super().__init__()
        self.embed_dim = embed_dim
        self.context = context
        self.rope = rope
        self.num_heads = num_heads
        out_dim = 3 * embed_dim
        self.in_proj = nn.Linear(embed_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def init_state(self, batch_size: int, sequence_length: int) -> Dict[str, mx.array]:
        """Initialize ring-buffer cache and offsets."""
        dim_per_head = self.embed_dim // self.num_heads
        state = {}
        state["offset"] = mx.zeros(batch_size, dtype=mx.int64)
        state["cache"] = mx.zeros((2, batch_size, self.num_heads, sequence_length, dim_per_head))
        state["end_offset"] = mx.zeros(batch_size, dtype=mx.int64)
        return state

    def increment_step(self, state: Dict[str, mx.array], increment: int = 1) -> None:
        """Advance absolute offset for streaming windows."""
        state["offset"] = state["offset"] + increment

    def _complete_kv(self, k: mx.array, v: mx.array, model_state: dict | None):
        """Update ring-buffer cache or return full k/v in non-streaming mode."""
        if model_state is None:
            return KVCacheResult.from_kv(k, v)
        layer_state = self.get_state(model_state)
        result, new_end_offset = complete_mimi_kv(layer_state["cache"], layer_state["end_offset"], k, v)
        layer_state["cache"] = mx.stack([result.keys, result.values], axis=0)
        layer_state["end_offset"] = new_end_offset
        return result

    def __call__(self, query: mx.array, model_state: dict | None) -> mx.array:
        B, T = query.shape[:2]
        if model_state is None:
            offset = mx.zeros((B,), dtype=mx.int64)
        else:
            offset = self.get_state(model_state)["offset"]

        projected = self.in_proj(query)
        d = self.embed_dim // self.num_heads
        packed = projected.reshape(B, T, 3, self.num_heads, d)
        q, k, v = mx.split(packed, 3, axis=2)
        q = q.squeeze(2)
        k = k.squeeze(2)
        v = v.squeeze(2)

        # RoPE operates on [B, T, H, D].
        q, k = self.rope(q, k, offset)

        # Transpose to [B, H, T, D] for attention.
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        kv = self._complete_kv(k, v, model_state)
        pos_k = kv.positions[:, None]
        pos_q = offset.reshape(-1, 1, 1) + mx.arange(T, dtype=mx.int64).reshape(1, -1, 1)
        delta = pos_q - pos_k
        # Build causal mask with fixed context window.
        attn_bias = (pos_k >= 0) & (delta >= 0) & (delta < self.context)
        attn_bias = attn_bias[:, None]
        attn_mask = mx.where(
            attn_bias,
            mx.zeros(attn_bias.shape, dtype=mx.float32),
            mx.full(attn_bias.shape, -1e9, dtype=mx.float32),
        )

        scale = 1.0 / mx.sqrt(mx.array(d, dtype=mx.float32))
        scores = mx.matmul(q, mx.transpose(kv.keys, (0, 1, 3, 2))) * scale
        scores = scores + attn_mask
        weights = mx.softmax(scores, axis=-1)
        x = mx.matmul(weights, kv.values)

        x = mx.transpose(x, (0, 2, 1, 3))
        x = x.reshape(B, T, self.num_heads * d)
        return self.out_proj(x)
