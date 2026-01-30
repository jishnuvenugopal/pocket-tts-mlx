"""Streaming 1D convolution modules for MLX.

This module implements streaming 1D convolutions with support for asymmetric
padding and causal convolutions. The implementations mirror the PyTorch versions
but use MLX operations.

Key features:
- Streaming mode with stateful padding
- Support for both regular and transposed convolutions
- Custom padding for exact output size
"""

import math
import warnings
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.modules.stateful_module import StatefulModule


def get_extra_padding_for_conv1d(
    x: mx.array, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """Calculate extra padding needed for conv1d to ensure exact output size.

    Args:
        x: Input tensor.
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution.
        padding_total: Total padding already applied.

    Returns:
        Extra padding needed at the end.
    """
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return int(ideal_length - length)


def pad_for_conv1d(
    x: mx.array, kernel_size: int, stride: int, padding_total: int = 0
) -> mx.array:
    """Pad input for conv1d to make sure the last window is full.

    Extra padding is added at the end. This is required to ensure that we can
    rebuild an output of the same length.

    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution)
        0 0 1 2 3 4 5 0     # (output of tr. conv.)
            1 2 3 4         # once you removed padding, we're missing one time step!

    Args:
        x: Input tensor of shape [B, C, T].
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution.
        padding_total: Total padding already applied.

    Returns:
        Padded tensor with extra padding at the end.
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    if extra_padding > 0:
        # Pad with zeros at the end
        pad_width = [(0, 0), (0, 0), (0, extra_padding)]
        x = mx.pad(x, pad_width, mode="constant", constant_values=0)
    return x


class StreamingConv1d(StatefulModule):
    """Conv1d with built-in handling of asymmetric or causal padding.

    This module implements streaming convolutions with stateful padding management.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution. Defaults to 1.
        dilation: Dilation factor. Defaults to 1.
        groups: Number of groups for grouped convolution. Defaults to 1.
        bias: Whether to include bias term. Defaults to True.
        pad_mode: Padding mode ("constant" or "replicate"). Defaults to "constant".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "constant",
    ):
        super().__init__()

        assert pad_mode in ["constant", "replicate"], f"pad_mode must be 'constant' or 'replicate', got {pad_mode}"
        self.pad_mode = pad_mode

        # Warn about unusual stride/dilation combination
        if stride > 1 and dilation > 1:
            warnings.warn(
                f"StreamingConv1d initialized with stride > 1 and dilation > 1 "
                f"(kernel_size={kernel_size}, stride={stride}, dilation={dilation})"
            )

        # MLX Conv1d: (in_channels, out_channels, kernel_size)
        # Note: MLX uses different parameter order than PyTorch
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # Match PyTorch default - streaming handles padding via 'previous' state
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self._stride = stride
        self._kernel_size = kernel_size
        self._dilation = dilation
        self._in_channels = in_channels
        self._out_channels = out_channels

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def kernel_size(self) -> int:
        return self._kernel_size

    @property
    def effective_kernel_size(self) -> int:
        """Effective kernel size accounting for dilation."""
        return (self._kernel_size - 1) * self._dilation + 1

    def _check_model_state(self, model_state: Any) -> Dict[str, mx.array]:
        """Validate and extract module state.

        Args:
            model_state: Full model state dictionary (nested) or direct module state.

        Returns:
            This module's state dictionary.
        """
        if model_state is None:
            raise ValueError("model_state must be provided when not None")

        # Check if this is a direct state dict (has 'previous' key)
        # or a nested model state dict
        if "previous" in model_state:
            # Direct state dict - return as-is for standalone usage
            return model_state
        else:
            # Nested model state dict - use get_state
            return self.get_state(model_state)

    def init_state(self, batch_size: int, sequence_length: int) -> Dict[str, mx.array]:
        """Initialize state for streaming inference.

        Args:
            batch_size: Batch size.
            sequence_length: Maximum sequence length (not directly used here).

        Returns:
            State dictionary with 'previous' and 'first' entries.
        """
        S = self._stride
        kernel = self.effective_kernel_size

        # Previous samples to prepend
        previous = mx.zeros((batch_size, self._in_channels, kernel - S))

        # Track if this is the first step (for replicate padding)
        first = mx.ones((batch_size,), dtype=mx.bool_)

        return {"previous": previous, "first": first}

    def __call__(self, x: mx.array, model_state: Any) -> mx.array:
        """Forward pass with streaming support.

        Args:
            x: Input tensor of shape [B, C, T].
            model_state: Model state dictionary (nested or direct).

        Returns:
            Output tensor of shape [B, C_out, T'].
        """
        B, C, T = x.shape
        S = self._stride

        # Initialize state if not provided
        if model_state is None:
            state = self.init_state(B, 0)
            # For non-streaming mode, pad input to be compatible with stride
            # after prepending previous samples
            TP = state["previous"].shape[-1]
            T_total = T + TP
            pad_needed = (S - T_total % S) % S
            if pad_needed > 0:
                x = mx.pad(x, [(0, 0), (0, 0), (0, pad_needed)])
                T = x.shape[-1]
        else:
            state = self._check_model_state(model_state)
            # For streaming mode, require strict alignment
            assert T > 0 and T % S == 0, f"Steps must be multiple of stride, got T={T}, stride={S}"

        TP = state["previous"].shape[-1]

        # Handle replicate padding for first step
        if TP and self.pad_mode == "replicate":
            assert T >= TP, "Not enough content to pad streaming."
            init = x[..., :1]
            # Update previous with first sample if this is first step
            state["previous"] = mx.where(
                mx.expand_dims(mx.expand_dims(state["first"], axis=-1), axis=-1),
                init,
                state["previous"],
            )

        # Prepend previous samples
        if TP > 0:
            x = mx.concatenate([state["previous"], x], axis=-1)

        # Save the channels-first version for state update
        x_cf = x  # Keep reference to channels-first version

        # MLX Conv1d expects channels-last format (N, L, C)
        # Convert from channels-first (N, C, L) to channels-last (N, L, C)
        x = mx.transpose(x, (0, 2, 1))  # (N, C, L) -> (N, L, C)

        # Apply convolution
        y = self.conv(x)  # (N, L', C_out)

        # Convert back to channels-first format
        y = mx.transpose(y, (0, 2, 1))  # (N, L', C_out) -> (N, C_out, L')

        # Update previous samples for next step
        # state["previous"] should be in channels-first format
        if TP > 0:
            state["previous"] = x_cf[..., -TP:]
            if self.pad_mode == "replicate":
                state["first"] = mx.zeros_like(state["first"])

        return y


class StreamingConvTranspose1d(StatefulModule):
    """ConvTranspose1d with built-in handling of asymmetric or causal padding.

    This module implements streaming transposed convolutions with stateful
    partial overlap management.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution. Defaults to 1.
        groups: Number of groups for grouped convolution. Defaults to 1.
        bias: Whether to include bias term. Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        # MLX ConvTranspose1d (doesn't have groups parameter)
        # Note: PyTorch defaults to padding=0, MLX must match this for compatibility
        self.convtr = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # Match PyTorch default (padding=0)
            bias=bias,
        )

        self._stride = stride
        self._kernel_size = kernel_size
        self._in_channels = in_channels
        self._out_channels = out_channels

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def kernel_size(self) -> int:
        return self._kernel_size

    def _check_model_state(self, model_state: Any) -> Dict[str, mx.array]:
        """Validate and extract module state.

        Args:
            model_state: Full model state dictionary (nested) or direct module state.

        Returns:
            This module's state dictionary.
        """
        if model_state is None:
            raise ValueError("model_state must be provided when not None")

        # Check if this is a direct state dict (has 'partial' key)
        # or a nested model state dict
        if "partial" in model_state:
            # Direct state dict - return as-is for standalone usage
            return model_state
        else:
            # Nested model state dict - use get_state
            return self.get_state(model_state)

    def init_state(self, batch_size: int, sequence_length: int) -> Dict[str, mx.array]:
        """Initialize state for streaming inference.

        Args:
            batch_size: Batch size.
            sequence_length: Maximum sequence length.

        Returns:
            State dictionary with 'partial' entry for overlap.
        """
        K = self._kernel_size
        S = self._stride

        # Partial output to carry over to next step
        partial = mx.zeros((batch_size, self._out_channels, K - S))

        return {"partial": partial}

    def __call__(self, x: mx.array, mimi_state: Any) -> mx.array:
        """Forward pass with streaming support.

        Args:
            x: Input tensor of shape [B, C, T] (channels-first).
            mimi_state: Model state dictionary (nested or direct).

        Returns:
            Output tensor of shape [B, C_out, T'].
        """
        state = self._check_model_state(mimi_state)
        layer_state = state["partial"]

        # MLX ConvTranspose1d expects channels-last format (N, L, C)
        # Convert from channels-first (N, C, L) to channels-last (N, L, C)
        x = mx.transpose(x, (0, 2, 1))  # (N, C, L) -> (N, L, C)

        # Apply transposed convolution
        y = self.convtr(x)  # (N, L', C_out)

        # Convert back to channels-first format
        y = mx.transpose(y, (0, 2, 1))  # (N, L', C_out) -> (N, C_out, L')

        PT = layer_state.shape[-1]

        if PT > 0:
            # Add partial from previous step
            # Concatenate the partial portion with the rest
            y_with_partial = mx.concatenate([y[..., :PT] + layer_state, y[..., PT:]], axis=-1)
            y = y_with_partial

            # Compute new partial for next step
            # Remove bias contribution from partial
            bias = getattr(self.convtr, 'bias', None)
            for_partial = y[..., -PT:]
            if bias is not None:
                for_partial = for_partial - bias[:, None]

            layer_state = for_partial

            # Remove partial portion from output
            y = y[..., :-PT]

            # Update state
            state["partial"] = layer_state

        return y


class _DepthwiseConvWeightHolder(nn.Module):
    """Inner module to hold weights for DepthwiseConvTranspose1d.

    This creates a nested structure to match PyTorch's weight path:
    PyTorch: mimi.upsample.convtr.convtr.weight
    MLX: mimi.upsample.convtr.convtr.weight (via this holder)
    """
    def __init__(self, out_channels: int, kernel_size: int, bias: bool):
        super().__init__()
        # Weight shape: [C, K, 1] where C is channels, K is kernel_size
        # PyTorch shape is [C, 1, K], will be transposed during loading
        self.weight = mx.zeros((out_channels, kernel_size, 1))
        if bias:
            self.bias = mx.zeros(out_channels)
        else:
            self.bias = None


class DepthwiseConvTranspose1d(StatefulModule):
    """Depthwise ConvTranspose1d for MLX with streaming support.

    MLX doesn't support groups parameter in ConvTranspose1d. This class
    implements depthwise convolution with proper streaming support using
    the partial overlap-add mechanism like StreamingConvTranspose1d.

    Args:
        in_channels: Number of input channels (must equal out_channels for depthwise).
        out_channels: Number of output channels (must equal in_channels for depthwise).
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution.
        bias: Whether to include bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        assert in_channels == out_channels, (
            f"Depthwise convolution requires in_channels == out_channels, "
            f"got in_channels={in_channels}, out_channels={out_channels}"
        )

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._has_bias = bias

        # Use nested convtr to match PyTorch's weight path structure
        # PyTorch: StreamingConvTranspose1d.convtr.weight
        # MLX: DepthwiseConvTranspose1d.convtr.weight
        self.convtr = _DepthwiseConvWeightHolder(out_channels, kernel_size, bias)

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def kernel_size(self) -> int:
        return self._kernel_size

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def weight(self) -> mx.array:
        return self.convtr.weight

    @property
    def bias(self):
        return self.convtr.bias

    def init_state(self, batch_size: int, sequence_length: int) -> Dict[str, mx.array]:
        """Initialize state for streaming inference.

        Args:
            batch_size: Batch size.
            sequence_length: Maximum sequence length.

        Returns:
            State dictionary with 'partial' entry for overlap.
        """
        K = self._kernel_size
        S = self._stride
        # Partial output to carry over to next step (channels-last format)
        partial = mx.zeros((batch_size, K - S, self._out_channels))
        return {"partial": partial}

    def _raw_conv_transpose(self, x: mx.array) -> mx.array:
        """Raw transposed convolution without streaming logic.

        Args:
            x: Input tensor of shape [B, L, C] (channels-last format).

        Returns:
            Output tensor of shape [B, (L-1)*S + K, C] (channels-last format).
        """
        B, L, C = x.shape
        K = self._kernel_size
        S = self._stride

        # Output length for transposed convolution: (L - 1) * S + K
        out_L = (L - 1) * S + K

        # Weight shape is [C, K, 1] - squeeze the last dimension
        weight = self.weight[:, :, 0]  # [C, K]

        # Build output by accumulating contributions from each input position
        contributions = []

        for i in range(L):
            start = i * S
            x_i = x[:, i, :]  # [B, C]
            w_t = mx.transpose(weight, (1, 0))  # [K, C]
            contrib = x_i[:, None, :] * w_t[None, :, :]  # [B, K, C]

            pad_right = out_L - start - K
            if pad_right < 0:
                contrib = contrib[:, :out_L - start, :]
                pad_right = 0

            padded = mx.pad(contrib, [(0, 0), (start, pad_right), (0, 0)])
            contributions.append(padded)

        y = mx.stack(contributions, axis=0).sum(axis=0)  # [B, out_L, C]

        # Add bias if present
        if self.bias is not None:
            y = y + self.bias[None, None, :]

        return y

    def _check_model_state(self, model_state: Any) -> Dict[str, mx.array]:
        """Validate and extract module state."""
        if model_state is None:
            raise ValueError("model_state must be provided when not None")
        if "partial" in model_state:
            return model_state
        else:
            return self.get_state(model_state)

    def __call__(self, x: mx.array, model_state: Any = None) -> mx.array:
        """Forward pass for depthwise transposed convolution with streaming.

        Args:
            x: Input tensor of shape [B, L, C] (channels-last format).
            model_state: Model state dictionary for streaming.

        Returns:
            Output tensor of shape [B, L*S, C] (channels-last format).
        """
        B, L, C = x.shape
        K = self._kernel_size
        S = self._stride
        PT = K - S  # Partial size

        # Get state
        if model_state is None:
            state = self.init_state(B, 0)
        else:
            state = self._check_model_state(model_state)

        layer_state = state["partial"]

        # Raw transposed convolution
        y = self._raw_conv_transpose(x)  # [B, (L-1)*S + K, C]

        if PT > 0:
            # Add partial from previous step to beginning
            y_start = y[:, :PT, :] + layer_state
            y = mx.concatenate([y_start, y[:, PT:, :]], axis=1)

            # Compute new partial for next step (last PT samples)
            # Remove bias contribution from partial
            for_partial = y[:, -PT:, :]
            if self.bias is not None:
                for_partial = for_partial - self.bias[None, None, :]

            # Update state
            state["partial"] = for_partial

            # Remove partial portion from output (trim last PT samples)
            y = y[:, :-PT, :]

        return y
