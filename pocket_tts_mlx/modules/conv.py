"""Streaming 1D convolutions for MLX with cache-aware padding."""

import math
import warnings

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.modules.stateful_module import StatefulModule


def get_extra_padding_for_conv1d(x: mx.array, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    """Compute padding needed to align stride boundaries."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return int(ideal_length - length)


def pad_for_conv1d(x: mx.array, kernel_size: int, stride: int, padding_total: int = 0):
    """Pad signal end so convolution produces full frames."""
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    if extra_padding > 0:
        pad_width = [(0, 0), (0, 0), (0, extra_padding)]
        x = mx.pad(x, pad_width, mode="constant", constant_values=0)
    return x


class ConvTranspose1dWithGroups(nn.Module):
    """ConvTranspose1d wrapper using mx.conv_transpose1d with group support."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        weight_shape = (out_channels, kernel_size, in_channels // groups)
        self.weight = mx.zeros(weight_shape)
        self.bias = mx.zeros((out_channels,)) if bias else None
        self.is_conv_transpose = True

    def __call__(self, x: mx.array) -> mx.array:
        """Apply transposed convolution with optional bias."""
        y = mx.conv_transpose1d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            output_padding=self.output_padding,
            groups=self.groups,
        )
        if self.bias is not None:
            y = y + self.bias
        return y


class StreamingConv1d(StatefulModule):
    """Streaming Conv1d with cached overlap for chunked inference."""
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
        assert pad_mode in ["constant", "replicate"]
        self.pad_mode = pad_mode
        if stride > 1 and dilation > 1:
            warnings.warn(
                "StreamingConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self._stride = stride
        self._kernel_size = kernel_size
        self._dilation = dilation
        self._in_channels = in_channels

    @property
    def effective_kernel_size(self) -> int:
        return (self._kernel_size - 1) * self._dilation + 1

    def init_state(self, batch_size: int, sequence_length: int):
        """Initialize overlap buffer for streaming convolution."""
        stride = self._stride
        kernel = self.effective_kernel_size
        previous = mx.zeros((batch_size, self._in_channels, kernel - stride))
        first = mx.ones((batch_size,), dtype=mx.bool_)
        return {"previous": previous, "first": first}

    def __call__(self, x: mx.array, model_state: dict | None):
        B, C, T = x.shape
        S = self._stride
        assert T > 0 and T % S == 0, "Steps must be multiple of stride"
        if model_state is None:
            state = self.init_state(B, 0)
        else:
            state = self.get_state(model_state)
        TP = state["previous"].shape[-1]
        if TP and self.pad_mode == "replicate":
            assert T >= TP, "Not enough content to pad streaming."
            init = x[..., :1]
            state["previous"] = mx.where(
                state["first"].reshape(-1, 1, 1),
                init,
                state["previous"],
            )
        if TP:
            x = mx.concatenate([state["previous"], x], axis=-1)

        # MLX Conv1d expects channels-last (N, L, C).
        x_cl = mx.transpose(x, (0, 2, 1))
        y = self.conv(x_cl)
        y = mx.transpose(y, (0, 2, 1))

        if TP:
            state["previous"] = x[..., -TP:]
            if self.pad_mode == "replicate":
                state["first"] = mx.zeros_like(state["first"])
        return y


class StreamingConvTranspose1d(StatefulModule):
    """Streaming ConvTranspose1d with overlap-add accumulation."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.convtr = ConvTranspose1dWithGroups(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
        )
        self._kernel_size = kernel_size
        self._stride = stride

    def init_state(self, batch_size: int, sequence_length: int):
        """Initialize partial overlap buffer."""
        K = self._kernel_size
        S = self._stride
        return {"partial": mx.zeros((batch_size, self.convtr.out_channels, K - S))}

    def __call__(self, x: mx.array, mimi_state: dict):
        layer_state = self.get_state(mimi_state)["partial"]
        x_cl = mx.transpose(x, (0, 2, 1))
        y = self.convtr(x_cl)
        y = mx.transpose(y, (0, 2, 1))

        PT = layer_state.shape[-1]
        if PT > 0:
            # Add overlap from previous chunk, then update partial buffer.
            y_head = y[..., :PT] + layer_state
            y_tail = y[..., PT:]
            y = mx.concatenate([y_head, y_tail], axis=-1)
            bias = self.convtr.bias
            for_partial = y[..., -PT:]
            if bias is not None:
                for_partial = for_partial - bias[:, None]
            self.get_state(mimi_state)["partial"] = for_partial
            y = y[..., :-PT]
        return y
