"""Dummy quantizer that projects encoder output to codebook dimension."""

import mlx.core as mx
import mlx.nn as nn


class DummyQuantizer(nn.Module):
    """Linear projection used in Mimi for non-VQ path."""
    def __init__(self, dimension: int, output_dimension: int):
        super().__init__()
        self.dimension = dimension
        self.output_dimension = output_dimension
        self.output_proj = nn.Conv1d(self.dimension, self.output_dimension, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, T] -> MLX Conv1d expects [B, T, C].
        x_cl = mx.transpose(x, (0, 2, 1))
        y = self.output_proj(x_cl)
        return mx.transpose(y, (0, 2, 1))
