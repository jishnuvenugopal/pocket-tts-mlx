"""Dummy quantizer module for MLX.

This is a simplified quantizer that only provides output projection for TTS.
It removes all unnecessary quantization logic since we don't use actual quantization.
"""

import mlx.core as mx
import mlx.nn as nn


class DummyQuantizer(nn.Module):
    """Simplified quantizer that only provides output projection for TTS.

    This removes all unnecessary quantization logic since we don't use actual quantization.

    Args:
        dimension: Input dimension.
        output_dimension: Output dimension.
    """

    def __init__(self, dimension: int, output_dimension: int):
        super().__init__()
        self.dimension = dimension
        self.output_dimension = output_dimension
        self.output_proj = nn.Conv1d(self.dimension, self.output_dimension, kernel_size=1, bias=False)

    def __call__(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape [B, T, C_in] (channels-last format).

        Returns:
            Output tensor of shape [B, C_out, T] (channels-first format).
        """
        # MLX Conv1d uses channels-last format (B, L, C)
        # Apply convolution in channels-last format
        y = self.output_proj(x)  # [B, T, C_out]
        # Transpose to channels-first format for downstream modules
        return mx.transpose(y, (0, 2, 1))  # [B, T, C_out] -> [B, C_out, T]
