"""Rotary position embedding (RoPE) utilities for MLX."""

import math

import mlx.core as mx
import mlx.nn as nn


def apply_rope(q: mx.array, k: mx.array, offset: int | mx.array = 0, max_period: int | float = 10_000):
    """Apply RoPE to query and key tensors."""
    B, T, H, D = q.shape
    Bk, Tk, Hk, Dk = k.shape
    assert (B, T, D) == (Bk, Tk, Dk)
    assert D % 2 == 0

    # Precompute frequencies for even/odd pairs.
    ds = mx.arange(D // 2, dtype=mx.float32)
    freqs = mx.exp(ds * (-math.log(max_period) * 2 / D))

    ts = mx.arange(T, dtype=mx.float32)
    ts = ts + offset
    ts = ts.reshape(-1, 1, 1)

    q = q.reshape(B, T, H, D // 2, 2)
    k = k.reshape(B, T, Hk, D // 2, 2)

    qr = q[..., 0].astype(mx.float32)
    qi = q[..., 1].astype(mx.float32)
    kr = k[..., 0].astype(mx.float32)
    ki = k[..., 1].astype(mx.float32)

    rotr = mx.cos(freqs * ts)
    roti = mx.sin(freqs * ts)
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    dtype = q.dtype
    qo = mx.stack([qor.astype(dtype), qoi.astype(dtype)], axis=-1)
    ko = mx.stack([kor.astype(dtype), koi.astype(dtype)], axis=-1)
    return qo.reshape(B, T, H, D), ko.reshape(B, T, Hk, D)


class RotaryEmbedding(nn.Module):
    """Callable RoPE wrapper with stored max_period."""
    def __init__(self, max_period: float | int = 10000.0):
        super().__init__()
        self.max_period = max_period

    def __call__(self, q: mx.array, k: mx.array, offset: int | mx.array):
        return apply_rope(q, k, offset, self.max_period)
