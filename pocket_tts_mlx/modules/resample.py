"""Resampling modules for MLX.

This module implements downsampling and upsampling using convolutions.
"""

from typing import Optional, Any

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.modules.conv import StreamingConv1d, StreamingConvTranspose1d, DepthwiseConvTranspose1d


class ConvDownsample1d(nn.Module):
    """Downsampling by an integer amount using convolutions.

    Args:
        stride: Downsampling stride factor.
        dimension: Dimension of the input/output.
    """

    def __init__(self, stride: int, dimension: int):
        super().__init__()
        self.conv = StreamingConv1d(
            in_channels=dimension,
            out_channels=dimension,
            kernel_size=2 * stride,
            stride=stride,
            groups=1,
            bias=False,
            pad_mode="replicate",
        )

    def __call__(self, x, model_state: Any) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor.
            model_state: Model state for streaming.

        Returns:
            Downsampled tensor.
        """
        return self.conv(x, model_state)


class ConvTrUpsample1d(nn.Module):
    """Upsampling by an integer amount using transposed convolutions.

    Uses depthwise convolution (groups=in_channels) for efficiency.

    Args:
        stride: Upsampling stride factor.
        dimension: Dimension of the input/output.
    """

    def __init__(self, stride: int, dimension: int):
        super().__init__()
        # Use depthwise ConvTranspose1d for upsampling
        # This implements the same behavior as PyTorch with groups=dimension
        self.convtr = DepthwiseConvTranspose1d(
            in_channels=dimension,
            out_channels=dimension,
            kernel_size=2 * stride,
            stride=stride,
            bias=False,
        )

        self._stride = stride
        self._kernel_size = 2 * stride

    @property
    def _stride(self) -> int:
        return self.__dict__.get('_stride_value', self.convtr.stride)

    @_stride.setter
    def _stride(self, value: int):
        self.__dict__['_stride_value'] = value

    @property
    def _kernel_size(self) -> int:
        return self.__dict__.get('_kernel_size_value', self.convtr.kernel_size)

    @_kernel_size.setter
    def _kernel_size(self, value: int):
        self.__dict__['_kernel_size_value'] = value

    def __call__(self, x, model_state: Any = None) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, T] (channels-first).
            model_state: Model state for streaming.

        Returns:
            Upsampled tensor of shape [B, C, T'].
        """
        # DepthwiseConvTranspose1d expects channels-last format
        # Convert from channels-first (B, C, T) to channels-last (B, T, C)
        x = mx.transpose(x, (0, 2, 1))  # (B, C, T) -> (B, T, C)

        # Apply depthwise transposed convolution with streaming state
        y = self.convtr(x, model_state)  # (B, T', C)

        # Convert back to channels-first format
        y = mx.transpose(y, (0, 2, 1))  # (B, T', C) -> (B, C, T')

        return y
