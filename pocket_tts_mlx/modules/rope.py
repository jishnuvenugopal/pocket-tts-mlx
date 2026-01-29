"""Rotary Positional Embeddings (RoPE) for MLX.

This module implements RoPE from "RoFormer: Enhanced Transformer with Rotary Position Embedding"
(https://arxiv.org/abs/2104.09864).

The implementation mirrors the PyTorch version but uses MLX operations.
"""

import math
from typing import Union

import mlx.core as mx


def apply_rope(
    q: mx.array,
    k: mx.array,
    offset: Union[int, mx.array] = 0,
    max_period: Union[int, float] = 10_000,
) -> tuple[mx.array, mx.array]:
    """Apply rotary positional embeddings to query and key tensors.

    Args:
        q: Queries tensor of shape [B, T, H, D].
        k: Keys tensor of shape [B, T, H, D].
        offset: Current offset for streaming, can be an integer or tensor.
        max_period: Maximum period for the cos and sin.

    Returns:
        Tuple of (rotated_queries, rotated_keys) with shapes [B, T, H, D].
    """
    B, T, H, D = q.shape
    Bk, Tk, Hk, Dk = k.shape

    assert (B, T, D) == (Bk, Tk, Dk), f"Shape mismatch: q={(B, T, H, D)}, k={(Bk, Tk, Hk, Dk)}"
    assert D > 0, "Embedding dimension must be positive"
    assert D % 2 == 0, f"Embedding dimension must be even, got {D}"
    assert max_period > 0, "max_period must be positive"

    # Create frequency bands
    ds = mx.arange(D // 2, dtype=mx.float32)
    freqs = mx.exp(ds * (-math.log(max_period) * 2 / D))

    # Create time steps with offset
    ts = mx.arange(T, dtype=mx.float32)

    # Handle offset: can be scalar (int/float) or per-batch tensor
    if isinstance(offset, mx.array):
        # Per-batch offset: ts shape (T,), offset shape (B,)
        # Reshape ts to (1, T) and offset to (B, 1) for broadcasting
        ts = mx.reshape(ts, (1, -1))  # (1, T)
        ts = ts + mx.reshape(offset, (-1, 1))  # (B, T)
        ts = mx.reshape(ts, (B, -1, 1, 1))  # (B, T, 1, 1)
    else:
        # Scalar offset
        ts += offset
        ts = mx.reshape(ts, (-1, 1, 1))  # (T, 1, 1)
        # Broadcast to (B, T, 1, 1) for consistency
        ts = mx.broadcast_to(ts, (B, T, 1, 1))

    # Reshape for complex number representation
    # q, k: [B, T, H, D] -> [B, T, H, D//2, 2]
    q = mx.reshape(q, (B, T, H, D // 2, 2))
    k = mx.reshape(k, (B, T, Hk, D // 2, 2))

    # Extract real and imaginary parts
    # convention: 'r' suffix is real part, 'i' is imaginary
    qr = q[..., 0].astype(mx.float32)
    qi = q[..., 1].astype(mx.float32)

    kr = k[..., 0].astype(mx.float32)
    ki = k[..., 1].astype(mx.float32)

    # Compute rotation: cos(freqs * ts) + i * sin(freqs * ts)
    rotr = mx.cos(freqs * ts)
    roti = mx.sin(freqs * ts)

    # Apply rotation: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr

    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    # Stack real and imaginary parts back together
    dtype = q.dtype
    qo = mx.stack([qor.astype(dtype), qoi.astype(dtype)], axis=-1)
    ko = mx.stack([kor.astype(dtype), koi.astype(dtype)], axis=-1)

    # Reshape back to [B, T, H, D]
    qo = mx.reshape(qo, (B, T, H, D))
    ko = mx.reshape(ko, (B, T, Hk, D))

    return qo, ko


class RotaryEmbedding:
    """Rotary positional embedding (RoPE) module.

    From Su et al 2022 (https://arxiv.org/abs/2104.09864).

    This module applies rotary positional embeddings to query and key tensors
    for efficient position-aware attention without positional biases.

    Args:
        max_period: Maximum period of the rotation frequencies. Defaults to 10000.0.
    """

    def __init__(self, max_period: Union[float, int] = 10000.0):
        self.max_period = float(max_period)

    def __call__(
        self, q: mx.array, k: mx.array, offset: Union[int, mx.array] = 0
    ) -> tuple[mx.array, mx.array]:
        """Apply rope rotation to query and key tensors.

        Args:
            q: Queries tensor of shape [B, T, H, D].
            k: Keys tensor of shape [B, T, H, D].
            offset: Current offset for streaming inference.

        Returns:
            Tuple of (rotated_queries, rotated_keys).
        """
        return apply_rope(q, k, offset, self.max_period)
