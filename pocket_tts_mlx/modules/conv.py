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
            padding=dilation * (kernel_size - 1) // 2,  # Default symmetric padding
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

        assert T > 0 and T % S == 0, f"Steps must be multiple of stride, got T={T}, stride={S}"

        # Initialize state if not provided
        if model_state is None:
            state = self.init_state(B, 0)
        else:
            state = self._check_model_state(model_state)

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
        self.convtr = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size - stride,  # Default padding
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


class DepthwiseConvTranspose1d(StatefulModule):
    """Depthwise ConvTranspose1d for MLX with streaming support.

    MLX doesn't support groups parameter in ConvTranspose1d. This class
    implements depthwise convolution by processing each channel independently.

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

        # For depthwise convolution, each channel uses its own kernel
        # Weight shape: [C, K, 1] where C is channels, K is kernel_size
        self.weight = mx.zeros((out_channels, kernel_size, 1))
        if bias:
            self.bias = mx.zeros(out_channels)
        else:
            self.bias = None

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

    def init_state(self, batch_size: int, sequence_length: int) -> dict:
        """Initialize state for streaming inference.

        Args:
            batch_size: Batch size.
            sequence_length: Maximum sequence length (not used for this module).

        Returns:
            State dictionary with 'partial' buffer for overlap-add.
        """
        # For transposed convolution, we need to store partial output for overlap
        # The kernel size - 1 determines the overlap
        PT = self._kernel_size - 1  # Partial tail length
        partial = mx.zeros((batch_size, PT, self._out_channels))
        return {"partial": partial}

    def __call__(self, x: mx.array, model_state: dict = None) -> mx.array:
        """Forward pass for depthwise transposed convolution.

        Args:
            x: Input tensor of shape [B, L, C] (channels-last format).
            model_state: Model state for streaming (optional).

        Returns:
            Output tensor of shape [B, L', C] (channels-last format).
        """
        B, L, C = x.shape
        K = self._kernel_size
        S = self._stride

        # Calculate expected output length for ConvTranspose1d
        # out_L = (L - 1) * S + K
        expected_out_L = (L - 1) * S + K

        # Use nearest neighbor upsampling by stride S
        # This is simpler and more predictable than conv_transpose1d
        # Upsample by repeating each element S times
        x_expanded = mx.expand_dims(x, axis=2)  # [B, L, 1, C]
        x_tiled = mx.repeat(x_expanded, repeats=S, axis=2)  # [B, L, S, C]
        y = mx.reshape(x_tiled, (B, L * S, C))  # [B, L*S, C]

        # Pad or trim to expected output length
        current_L = y.shape[1]
        if current_L < expected_out_L:
            # Pad at the end
            pad_length = expected_out_L - current_L
            y = mx.pad(y, [(0, 0), (0, pad_length), (0, 0)])
        elif current_L > expected_out_L:
            # Trim from the end
            y = y[:, :expected_out_L, :]

        # Handle streaming state for overlap-add
        if model_state is not None:
            # Check if model_state is a direct state dict or nested
            if "partial" in model_state:
                # Direct state dict
                state = model_state
            else:
                # Nested model state dict - use get_state
                state = self.get_state(model_state)
            partial = state.get("partial", None)

            if partial is not None:
                # Prepend partial from previous chunk
                PT = partial.shape[1]  # Partial tail length
                y = mx.concatenate([partial, y], axis=1)

                # Store new partial (last PT elements)
                new_partial = y[:, -PT:, :]
                state["partial"] = new_partial

                # Remove partial portion from output
                y = y[:, :-PT, :]

        return y
