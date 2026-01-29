"""Layer scale module for MLX.

This module implements layer scale for training stability in deep networks.
"""

import mlx.core as mx
import mlx.nn as nn


class LayerScale(nn.Module):
    """Layer scale for gradient flow in deep networks.

    Args:
        channels: Number of channels (dimension).
        init: Initial value for the scale parameter.
    """

    def __init__(self, channels: int, init: float):
        super().__init__()
        self.scale = mx.full((channels,), init)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply layer scale.

        Args:
            x: Input tensor.

        Returns:
            Scaled tensor.
        """
        return self.scale * x
