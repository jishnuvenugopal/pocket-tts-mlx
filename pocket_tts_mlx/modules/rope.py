"""Rotary position embedding (RoPE) utilities for MLX."""

import mlx.core as mx
import mlx.nn as nn


def apply_rope(q: mx.array, k: mx.array, offset: int | mx.array = 0, max_period: int | float = 10_000):
    """Apply fused traditional RoPE to query and key tensors."""
    B, T, H, D = q.shape
    Bk, Tk, Hk, Dk = k.shape
    assert (B, T, D) == (Bk, Tk, Dk)
    assert H == Hk
    assert D % 2 == 0

    # mx.fast.rope expects time on the penultimate axis. Stack q/k so one
    # fused invocation handles both tensors while preserving consecutive-pair
    # rotation semantics from the original implementation.
    qk = mx.stack(
        [mx.transpose(q, (0, 2, 1, 3)), mx.transpose(k, (0, 2, 1, 3))],
        axis=1,
    )
    qk = mx.fast.rope(
        qk,
        D,
        traditional=True,
        base=float(max_period),
        scale=1.0,
        offset=offset,
    )
    qo, ko = qk[:, 0], qk[:, 1]
    return mx.transpose(qo, (0, 2, 1, 3)), mx.transpose(ko, (0, 2, 1, 3))


class RotaryEmbedding(nn.Module):
    """Callable RoPE wrapper with stored max_period."""
    def __init__(self, max_period: float | int = 10000.0):
        super().__init__()
        self.max_period = max_period

    def __call__(self, q: mx.array, k: mx.array, offset: int | mx.array):
        return apply_rope(q, k, offset, self.max_period)
