"""Streaming multi-head attention modules for MLX.

This module implements streaming attention with KV cache for efficient autoregressive
generation. The implementations mirror the PyTorch versions but use MLX operations.

Key features:
- KV cache for storing past keys and values
- Causal masking for autoregressive generation
- Rotary positional embeddings
- Efficient state management for streaming inference
"""

from typing import Any, Dict, NamedTuple, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.modules.rope import RotaryEmbedding
from pocket_tts_mlx.modules.stateful_module import StatefulModule


class KVCacheResult(NamedTuple):
    """Result of KV cache completion operation."""

    keys: mx.array
    values: mx.array
    positions: mx.array

    @staticmethod
    def from_kv(keys: mx.array, values: mx.array) -> "KVCacheResult":
        """Create KVCacheResult from key and value tensors.

        Args:
            keys: Keys tensor of shape [B, H, T, D].
            values: Values tensor of shape [B, H, T, D].

        Returns:
            KVCacheResult with position indices.
        """
        B, H, T, D = keys.shape
        assert values.shape[:-1] == (B, H, T), f"Shape mismatch: keys={keys.shape}, values={values.shape}"
        positions = mx.arange(T, dtype=mx.int64)
        positions = mx.broadcast_to(positions, (B, -1))
        return KVCacheResult(keys, values, positions)


def complete_kv(
    cache: mx.array, current_end: mx.array, k: mx.array, v: mx.array
) -> Tuple[mx.array, mx.array, mx.array]:
    """Complete KV cache for FlowLM-style streaming attention.

    This function adds new keys and values to the cache at the current position,
    and returns the updated cache along with the complete keys and values.

    Args:
        cache: Cache tensor of shape [2, B, T_max, H, D] where index 0 is keys, 1 is values.
        current_end: Current end position tensor (scalar, used to determine write position).
        k: New keys of shape [B, T, H, D] (channels-last format from MLX conv).
        v: New values of shape [B, T, H, D] (channels-last format from MLX conv).

    Returns:
        Tuple of (updated_cache, complete_keys, complete_values).
        updated_cache: Updated cache tensor of shape [2, B, T_max, H, D].
        complete_keys: Complete keys with shape [B, T, H, D].
        complete_values: Complete values with shape [B, T, H, D].
    """
    # Get the current write position
    current_pos = current_end.shape[0]
    t_new = k.shape[1]  # T dimension (channels-last format: [B, L, C])

    # cache has shape [2, B, T_max, H, D]
    # k has shape [B, T, H, D] (channels-last format)
    B, T, H, D = k.shape
    T_max = cache.shape[2]

    # Update the cache with new keys and values
    # cache[0, :, current_end : current_end + t_new] = k
    # cache[1, :, current_end : current_end + t_new] = v
    # MLX doesn't support slice assignment, so we use slice_update

    # Prepare the update data - need to match cache dimensions [2, B, T_max, H, D]
    # k has shape [B, T, H, D], need to reshape to [1, B, T, H, D] for slice_update
    k_update = k[None, :, :, :, :]  # [1, B, T, H, D]
    v_update = v[None, :, :, :, :]  # [1, B, T, H, D]

    # Update keys cache (index 0) - update full cache with 5D start position
    write_start_k = mx.array([0, 0, current_pos, 0, 0])  # [cache_idx, batch, time, head, dim]
    cache = mx.slice_update(cache, k_update, write_start_k, axes=[0, 1, 2, 3, 4])

    # Update values cache (index 1)
    write_start_v = mx.array([1, 0, current_pos, 0, 0])  # [cache_idx, batch, time, head, dim]
    cache = mx.slice_update(cache, v_update, write_start_v, axes=[0, 1, 2, 3, 4])

    # Now read back the complete cache (for returning)
    cache_k = cache[0]  # [B, T_max, H, D]
    cache_v = cache[1]  # [B, T_max, H, D]

    # Transpose k from [B, T, H, D] to [B, H, T_new, D]
    k_transposed = mx.transpose(k, (0, 2, 1, 3))  # [B, H, T_new, D]
    v_transposed = mx.transpose(v, (0, 2, 1, 3))  # [B, H, T_new, D]

    # Transpose cache to [B, H, T_max, D] for easier manipulation
    cache_k_transposed = mx.transpose(cache_k, (0, 2, 1, 3))  # [B, H, T_max, D]
    cache_v_transposed = mx.transpose(cache_v, (0, 2, 1, 3))  # [B, H, T_max, D]

    # Concatenate along sequence dimension
    # Get existing cache content up to current_pos
    k_existing = cache_k_transposed[:, :, :current_pos, :]  # [B, H, current_pos, D]
    v_existing = cache_v_transposed[:, :, :current_pos, :]  # [B, H, current_pos, D]

    # Concatenate with new tokens
    k_cache = mx.concatenate([k_existing, k_transposed], axis=2)  # [B, H, current_pos + T_new, D]
    v_cache = mx.concatenate([v_existing, v_transposed], axis=2)  # [B, H, current_pos + T_new, D]

    # Transpose back to [B, T, H, D] format and return
    k_cache = mx.transpose(k_cache, (0, 2, 1, 3))  # [B, T, H, D]
    v_cache = mx.transpose(v_cache, (0, 2, 1, 3))  # [B, T, H, D]

    return cache, k_cache, v_cache


def materialize_causal_mask(
    shape: Tuple[int, int], shift: int, dtype: mx.Dtype = mx.float32
) -> mx.array:
    """Create a causal attention mask.

    Args:
        shape: Tuple of (num_queries, num_keys).
        shift: Shift for the causal mask (typically num_keys - num_queries).
        dtype: Data type for the mask.

    Returns:
        Causal mask tensor where valid positions are 0 and masked positions are -inf.
    """
    num_queries, num_keys = shape

    # Create lower triangular matrix
    mask = mx.tril(mx.ones((num_queries, num_keys), dtype=dtype), k=shift)
    # Log to convert to additive mask (0 -> 0, 1 -> -inf)
    mask = mx.log(mask + 1e-10)  # Small epsilon to avoid log(0)
    return mask.astype(dtype)


class StreamingMultiheadAttention(StatefulModule):
    """Streaming multi-head attention with KV cache.

    Similar to nn.MultiheadAttention but with support for streaming inference
    through KV caching.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads.
        rope: RotaryEmbedding module for positional encoding.
    """

    def __init__(self, embed_dim: int, num_heads: int, rope: RotaryEmbedding):
        super().__init__()

        self.embed_dim = embed_dim
        self.rope = rope
        self.num_heads = num_heads

        # Calculate dimensions
        out_dim = embed_dim
        num_kv = num_heads
        kv_dim = (embed_dim // num_heads) * num_kv
        out_dim += 2 * kv_dim

        # Input projection: Q, K, V combined
        self.in_proj = nn.Linear(embed_dim, out_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _get_mask(self, shape: Tuple[int, int], shift: int) -> mx.array:
        """Get causal mask for attention."""
        return materialize_causal_mask(shape, shift=shift)

    def init_state(self, batch_size: int, sequence_length: int) -> Dict[str, mx.array]:
        """Initialize state for streaming inference.

        Args:
            batch_size: Batch size.
            sequence_length: Maximum sequence length for KV cache.

        Returns:
            State dictionary with 'current_end' and 'cache' entries.
        """
        dim_per_head = self.embed_dim // self.num_heads
        initial_current_end = mx.zeros((0,), dtype=mx.int64)

        # Initialize cache with zeros (not NaN - NaN propagates through attention)
        cache = mx.zeros(
            (2, batch_size, sequence_length, self.num_heads, dim_per_head)
        )

        return {"current_end": initial_current_end, "cache": cache}

    def increment_step(self, state: Dict[str, mx.array], increment: int = 1) -> None:
        """Increment the streaming step counter.

        Args:
            state: Module state dictionary.
            increment: Number of steps to increment.
        """
        new_size = state["current_end"].shape[0] + increment
        state["current_end"] = mx.zeros((new_size,), dtype=mx.int64)

    def _complete_kv(
        self, k: mx.array, v: mx.array, state: Optional[Dict[str, mx.array]]
    ) -> Tuple[mx.array, mx.array]:
        """Complete KV cache with new keys and values.

        Args:
            k: New keys of shape [B, T, H, D].
            v: New values of shape [B, T, H, D].
            state: Module state dictionary.

        Returns:
            Tuple of (complete_keys, complete_values) with cached values prepended.
        """
        updated_cache, k, v = complete_kv(state["cache"], state["current_end"], k, v)
        # Update the state with the new cache
        state["cache"] = updated_cache
        return k, v

    def _apply_rope(
        self, query: mx.array, key: mx.array, state: Optional[Dict[str, mx.array]]
    ) -> Tuple[mx.array, mx.array]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            query: Query tensor of shape [B, T, H, D].
            key: Key tensor of shape [B, T, H, D].
            state: Module state dictionary.

        Returns:
            Tuple of (rotated_query, rotated_key).
        """
        streaming_offset = self._streaming_offset(state)
        return self.rope(query, key, offset=streaming_offset)

    def _streaming_offset(self, state: Optional[Dict[str, mx.array]]) -> int:
        """Get the current streaming offset from state.

        Args:
            state: Module state dictionary.

        Returns:
            Current position in the sequence.
        """
        return state["current_end"].shape[0]

    def check_model_state(self, model_state: Any) -> Dict[str, mx.array]:
        """Validate and extract module state.

        Args:
            model_state: Full model state dictionary (nested) or direct module state.

        Returns:
            This module's state dictionary.

        Raises:
            ValueError: If model_state is None.
        """
        if model_state is None:
            raise ValueError("model_state must be provided")

        # Check if this is a direct state dict (has 'current_end' key)
        # or a nested model state dict
        if "current_end" in model_state:
            # Direct state dict - return as-is for standalone usage
            return model_state
        else:
            # Nested model state dict - use get_state
            return self.get_state(model_state)

    def __call__(
        self, query: mx.array, model_state: Any
    ) -> mx.array:
        """Forward pass with streaming support.

        Args:
            query: Input tensor of shape [B, T, D].
            model_state: Model state dictionary for streaming.

        Returns:
            Output tensor of shape [B, T, D].
        """
        state = self.check_model_state(model_state)

        # Project input to Q, K, V
        projected = self.in_proj(query)

        # Reshape from [B, T, 3*H*D] to [B, T, 3, H, D]
        B, T, _ = projected.shape
        d = self.embed_dim // self.num_heads
        packed = mx.reshape(projected, (B, T, 3, self.num_heads, d))

        # Split into Q, K, V
        q, k, v = packed[..., 0, :, :], packed[..., 1, :, :], packed[..., 2, :, :]

        # Apply rotary embeddings
        q, k = self._apply_rope(q, k, state)

        # Complete KV cache
        k, v = self._complete_kv(k, v, state)

        # Create causal mask
        cache_len = state["current_end"].shape[0]
        mask_shape = (T, T + cache_len)
        shift = cache_len
        attn_mask = self._get_mask(mask_shape, shift=shift)

        # Scaled dot-product attention
        # Permute to [B, H, T, D] for attention
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # Attention scores
        attn = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) / (d**0.5)

        # Add causal mask
        attn = attn + attn_mask[None, None, :, :]

        # Softmax
        attn = mx.softmax(attn, axis=-1)

        # Apply attention to values
        x = mx.matmul(attn, v)

        # Permute back to [B, T, H, D]
        x = mx.transpose(x, (0, 2, 1, 3))

        # Reshape to [B, T, H*D]
        x = mx.reshape(x, (B, T, self.num_heads * d))

        # Output projection
        x = self.out_proj(x)

        return x


def complete_mimi_kv(
    cache: mx.array,
    end_offset: mx.array,
    k: mx.array,
    v: mx.array,
) -> "KVCacheResult":
    """Complete KV cache for Mimi-style streaming attention with circular buffer.

    This uses a circular buffer scheme where positions wrap around modulo capacity.

    Args:
        cache: Cache tensor of shape [2, B, H, T_max, D] where index 0 is keys, 1 is values.
        end_offset: Current write offset tensor of shape [B].
        k: New keys of shape [B, H, T_new, D].
        v: New values of shape [B, H, T_new, D].

    Returns:
        KVCacheResult with keys, values, and position indices.
    """
    capacity = cache.shape[3]
    B, H, T, D = k.shape

    assert T > 0, "New tokens must be positive"
    assert v.shape[:-1] == (B, H, T), f"Shape mismatch: k={k.shape}, v={v.shape}"

    # Create write indices with circular wrapping
    indexes = mx.arange(T, dtype=mx.int64)
    indexes = indexes + mx.expand_dims(end_offset, axis=-1)
    indexes = indexes % capacity  # Circular wrap

    # Expand indexes for all batches, heads, and dimensions
    # indexes: [B, T] -> [B, 1, T, 1] -> [B, H, T, D]
    indexes = mx.reshape(indexes, (B, 1, T, 1))
    indexes = mx.broadcast_to(indexes, (B, H, T, D))

    # Scatter add keys and values to cache
    # Note: MLX doesn't have scatter_, so we use a different approach
    # We'll need to iterate or use a more complex operation

    # For now, let's use a simpler approach that works with MLX
    # We'll gather the existing cache and update it

    cache_keys = cache[0]
    cache_values = cache[1]

    # Since MLX doesn't have a direct scatter operation, we need to
    # implement this differently. We'll use slice_update for each update.

    # Update keys and values using mx.slice_update
    for b in range(B):
        for t in range(T):
            idx = int(indexes[b, 0, t, 0])
            # Prepare update values: k[b, :, t, :] has shape (H, D)
            # We need to add batch and time dimensions
            k_update = mx.reshape(k[b, :, t, :], (1, H, 1, D))
            v_update = mx.reshape(v[b, :, t, :], (1, H, 1, D))
            # Start position for the update
            start = mx.array([b, 0, idx, 0])
            # Update cache_keys and cache_values
            cache_keys = mx.slice_update(cache_keys, k_update, start, axes=[0, 1, 2, 3])
            cache_values = mx.slice_update(cache_values, v_update, start, axes=[0, 1, 2, 3])

    keys = cache_keys
    values = cache_values

    # Calculate position indices
    full_indexes = mx.arange(capacity, dtype=mx.int64)

    # Calculate end position
    last_offset = end_offset + T - 1
    end_index = last_offset % capacity

    # Position calculation with wrap-around
    # Need to handle per-batch offsets: full_indexes (capacity), end_index (B)
    # Expand dimensions for broadcasting
    full_indexes_expanded = mx.reshape(full_indexes, (1, capacity))  # (1, capacity)
    end_index_expanded = mx.reshape(end_index, (B, 1))  # (B, 1)
    last_offset_expanded = mx.reshape(last_offset, (B, 1))  # (B, 1)

    delta = full_indexes_expanded - end_index_expanded  # (B, capacity)
    positions = mx.where(
        delta <= 0,
        last_offset_expanded + delta,
        last_offset_expanded + delta - capacity
    )  # (B, capacity)

    # Update end offset
    new_end_offset = end_offset + T

    # Mark invalid positions
    new_end_offset_expanded = mx.reshape(new_end_offset, (B, 1))  # (B, 1)
    invalid = full_indexes_expanded >= new_end_offset_expanded  # (B, capacity)
    positions = mx.where(invalid, mx.full(positions.shape, -1, dtype=positions.dtype), positions)

    return KVCacheResult(keys, values, positions)


class MimiStreamingMultiheadAttention(StatefulModule):
    """Mimi-style streaming multi-head attention with circular KV cache.

    This attention module uses a circular buffer for KV cache, allowing efficient
    streaming with a fixed context window.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads.
        context: Context window size (number of past steps to attend to).
        rope: RotaryEmbedding module for positional encoding.
    """

    def __init__(self, embed_dim: int, num_heads: int, context: int, rope: RotaryEmbedding):
        super().__init__()

        self.embed_dim = embed_dim
        self.context = context
        self.rope = rope
        self.num_heads = num_heads

        # Input projection: Q, K, V combined
        out_dim = 3 * embed_dim
        self.in_proj = nn.Linear(embed_dim, out_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def init_state(self, batch_size: int, sequence_length: int) -> Dict[str, mx.array]:
        """Initialize state for streaming inference.

        Args:
            batch_size: Batch size.
            sequence_length: Maximum sequence length for KV cache.

        Returns:
            State dictionary with 'offset', 'cache', and 'end_offset' entries.
        """
        dim_per_head = self.embed_dim // self.num_heads

        state = {}
        state["offset"] = mx.zeros(batch_size, dtype=mx.int64)
        state["cache"] = mx.zeros(
            (2, batch_size, self.num_heads, sequence_length, dim_per_head)
        )
        state["end_offset"] = mx.zeros(batch_size, dtype=mx.int64)

        return state

    def increment_step(self, state: Dict[str, mx.array], increment: int = 1) -> None:
        """Increment the streaming step counter.

        Args:
            state: Module state dictionary.
            increment: Number of steps to increment.
        """
        state["offset"] = state["offset"] + increment

    def _complete_kv(
        self, k: mx.array, v: mx.array, model_state: Any
    ) -> KVCacheResult:
        """Complete KV cache with new keys and values.

        Args:
            k: New keys of shape [B, H, T, D].
            v: New values of shape [B, H, T, D].
            model_state: Model state dictionary (nested or direct).

        Returns:
            KVCacheResult with keys, values, and position indices.
        """
        if model_state is None:
            return KVCacheResult.from_kv(k, v)
        else:
            layer_state = self._check_model_state(model_state)
            return complete_mimi_kv(layer_state["cache"], layer_state["end_offset"], k, v)

    def _check_model_state(self, model_state: Any) -> Dict[str, mx.array]:
        """Validate and extract module state.

        Args:
            model_state: Full model state dictionary (nested) or direct module state.

        Returns:
            This module's state dictionary.
        """
        if model_state is None:
            raise ValueError("model_state must be provided")

        # Check if this is a direct state dict (has 'offset' key)
        # or a nested model state dict
        if "offset" in model_state:
            # Direct state dict - return as-is for standalone usage
            return model_state
        else:
            # Nested model state dict - use get_state
            return self.get_state(model_state)

    def __call__(
        self, query: mx.array, model_state: Any
    ) -> mx.array:
        """Forward pass with streaming support.

        Args:
            query: Input tensor of shape [B, T, D].
            model_state: Model state dictionary for streaming.

        Returns:
            Output tensor of shape [B, T, D].
        """
        B, T = query.shape[:2]

        # Get offset for RoPE
        if model_state is None:
            offset = mx.zeros(B, dtype=mx.int64)
        else:
            state = self._check_model_state(model_state)
            offset = state["offset"]

        # Project input to Q, K, V
        projected = self.in_proj(query)

        # Reshape from [B, T, 3*H*D] to [3, B, H, T, D]
        qkv = mx.reshape(projected, (B, T, 3, self.num_heads, -1))
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))  # [3, B, H, T, D]

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Permute for RoPE: [B, H, T, D] -> [B, T, H, D]
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))

        # Apply rotary embeddings
        q, k = self.rope(q, k, offset)

        # Permute back: [B, T, H, D] -> [B, H, T, D]
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))

        # Complete KV cache
        k, v, pos_k = self._complete_kv(k, v, model_state)

        # Position calculations for attention mask
        pos_k = mx.expand_dims(pos_k, axis=1)  # [B, 1, T_cache]
        pos_q = mx.expand_dims(offset, axis=(-1,)) + mx.arange(T, dtype=mx.int64)
        pos_q = mx.reshape(pos_q, (B, -1, 1))  # [B, T, 1]

        # Calculate delta: pos_q - pos_k
        delta = pos_q - pos_k

        # Create attention bias
        attn_bias = (pos_k >= 0) & (delta >= 0)
        attn_bias = attn_bias & (delta < self.context)
        attn_bias = mx.expand_dims(attn_bias, axis=1)  # [B, 1, T, T_cache]

        # Scaled dot-product attention with mask
        d = self.embed_dim // self.num_heads

        # Attention scores
        attn = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) / (d**0.5)

        # Apply mask (set masked positions to -inf)
        attn = mx.where(attn_bias, attn, mx.full(attn.shape, -1e9, dtype=attn.dtype))

        # Softmax
        attn = mx.softmax(attn, axis=-1)

        # Apply attention to values
        x = mx.matmul(attn, v)

        # Reshape: [B, H, T, D] -> [B, T, H*D]
        x = mx.transpose(x, (0, 2, 1, 3))
        x = mx.reshape(x, (B, T, self.num_heads * d))

        # Output projection
        x = self.out_proj(x)

        return x
