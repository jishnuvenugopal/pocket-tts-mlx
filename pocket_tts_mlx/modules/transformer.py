"""Streaming transformer modules for MLX.

This module implements streaming transformer layers and blocks for MLX.
The implementations mirror the PyTorch versions but use MLX operations.

Key features:
- Streaming multi-head attention with KV cache
- Layer normalization
- Feed-forward networks with GELU activation
- Layer scale for training stability
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.modules.attention import (
    MimiStreamingMultiheadAttention,
    StreamingMultiheadAttention,
)
from pocket_tts_mlx.modules.layer_scale import LayerScale
from pocket_tts_mlx.modules.rope import RotaryEmbedding
from pocket_tts_mlx.modules.stateful_module import StatefulModule


class LayerNorm(nn.Module):
    """Layer normalization module.

    Reimplementation of LayerNorm compatible with MLX.

    Args:
        channels: Number of channels (dimension).
        eps: Small value to avoid division by zero. Defaults to 1e-6.
        elementwise_affine: Whether to use learnable affine parameters.
    """

    def __init__(self, channels: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.channels = channels

        if elementwise_affine:
            self.weight = mx.ones((channels,))
            self.bias = mx.zeros((channels,))
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """Apply layer normalization.

        Args:
            x: Input tensor of shape [..., C].

        Returns:
            Normalized tensor of same shape.
        """
        # Compute mean and variance along last dimension
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)

        # Normalize
        x = (x - mean) / mx.sqrt(var + self.eps)

        # Apply affine transform if enabled
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias

        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer normalization.

    Args:
        dim: Dimension to normalize.
        eps: Small value to avoid division by zero. Defaults to 1e-5.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape [..., D].

        Returns:
            Normalized tensor of same shape.
        """
        var = mx.var(x, axis=-1, keepdims=True)
        y = x * (self.alpha / mx.sqrt(var + self.eps))
        return y


class StreamingTransformerLayer(nn.Module):
    """Streaming transformer layer with self-attention and feed-forward.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        dim_feedforward: Dimension of feed-forward network.
        context: Context window size (for Mimi attention). None means unlimited.
        rope: RotaryEmbedding module.
        layer_scale: Layer scale value, or None to disable.
        attention_kind: Type of attention ("mimi" or "flow_lm").
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        context: Optional[int],
        rope: RotaryEmbedding,
        layer_scale: Optional[float] = None,
        attention_kind: str = "mimi",
    ):
        super().__init__()

        # Select attention type
        if attention_kind == "mimi":
            self.self_attn = MimiStreamingMultiheadAttention(
                context=context, rope=rope, embed_dim=d_model, num_heads=num_heads
            )
        else:  # "flow_lm"
            self.self_attn = StreamingMultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, rope=rope
            )

        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)

        if layer_scale is None:
            self.layer_scale_1 = None
            self.layer_scale_2 = None
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale)
            self.layer_scale_2 = LayerScale(d_model, layer_scale)

    def _ff_block(self, x: mx.array) -> mx.array:
        """Apply feed-forward block.

        Args:
            x: Input tensor of shape [B, T, D].

        Returns:
            Output tensor of shape [B, T, D].
        """
        x_orig = x
        x = self.norm2(x)
        update = self.linear2(nn.gelu(self.linear1(x)))
        if self.layer_scale_2 is not None:
            update = self.layer_scale_2(update)
        return x_orig + update

    def _sa_block(
        self, x: mx.array, model_state: Any
    ) -> mx.array:
        """Apply self-attention block.

        Args:
            x: Input tensor of shape [B, T, D].
            model_state: Model state for streaming (nested dict).

        Returns:
            Output tensor of shape [B, T, D].
        """
        x_orig = x
        x = self.norm1(x)
        update = self.self_attn(x, model_state)
        if self.layer_scale_1 is not None:
            update = self.layer_scale_1(update)
        return x_orig + update

    def __call__(
        self, x: mx.array, model_state: Any
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, T, D].
            model_state: Model state for streaming (nested dict).

        Returns:
            Output tensor of shape [B, T, D].
        """
        x = self._sa_block(x, model_state)
        x = self._ff_block(x)
        return x


class StreamingTransformer(nn.Module):
    """Streaming transformer with multiple layers.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        layer_scale: Layer scale value, or None to disable.
        dim_feedforward: Dimension of feed-forward network (can be list for per-layer config).
        context: Context window size for attention (None means unlimited).
        max_period: Maximum period for rotary embeddings.
        kind: Type of attention ("mimi" or "flow_lm").
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        layer_scale: Optional[float] = None,
        dim_feedforward: Union[int, list] = 2048,
        context: Optional[int] = None,
        max_period: float = 10_000.0,
        kind: str = "mimi",
    ):
        super().__init__()

        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.max_period = max_period
        self.rope = RotaryEmbedding(max_period=max_period)

        self.layers = []
        for i in range(num_layers):
            # Get feedforward dimension for this layer
            if isinstance(dim_feedforward, list):
                d_ff = dim_feedforward[i]
            else:
                d_ff = dim_feedforward

            layer = StreamingTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=d_ff,
                context=context,
                rope=self.rope,
                layer_scale=layer_scale,
                attention_kind=kind,
            )
            self.layers.append(layer)

    def __call__(self, x: mx.array, model_state: Any) -> mx.array:
        """Forward pass through all layers.

        Args:
            x: Input tensor of shape [B, T, D].
            model_state: Model state for streaming (nested dict).

        Returns:
            Output tensor of shape [B, T, D].
        """
        for layer in self.layers:
            x = layer(x, model_state)
        return x


class ProjectedTransformer(nn.Module):
    """Transformer with input and output projections.

    This wraps a StreamingTransformer with linear projections to handle
    different input/output dimensions.

    Args:
        input_dimension: Input dimension.
        output_dimensions: Tuple of output dimensions.
        d_model: Model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        layer_scale: Layer scale value.
        context: Context window size.
        max_period: Maximum period for rotary embeddings.
        dim_feedforward: Dimension of feed-forward network.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimensions: Tuple[int, ...],
        d_model: int,
        num_heads: int,
        num_layers: int,
        layer_scale: float,
        context: int,
        max_period: float,
        dim_feedforward: int,
    ):
        super().__init__()

        self.transformer = StreamingTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            layer_scale=layer_scale,
            context=context,
            max_period=max_period,
            dim_feedforward=dim_feedforward,
        )

        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions

        # Input projection if needed
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)
        else:
            self.input_proj = None

        # Output projections for each output dimension
        self.output_projs = []
        for output_dimension in output_dimensions:
            if d_model == output_dimension:
                self.output_projs.append(None)
            else:
                self.output_projs.append(nn.Linear(d_model, output_dimension, bias=False))

    def __call__(
        self, x: mx.array, model_state: Any
    ) -> List[mx.array]:
        """Forward pass with projections.

        Args:
            x: Input tensor of shape [B, C, T] (channels first).
            model_state: Model state for streaming (nested dict).

        Returns:
            List of output tensors, one for each output dimension.
        """
        # Transpose to [B, T, C]
        x = mx.transpose(x, (0, 2, 1))

        # Apply input projection if needed
        if self.input_proj is not None:
            x = self.input_proj(x)

        # Apply transformer
        z = self.transformer(x, model_state)

        # Apply output projections
        ys = []
        for i, output_proj in enumerate(self.output_projs):
            y = z
            if output_proj is not None:
                y = output_proj(y)
            # Transpose back to [B, C, T]
            y = mx.transpose(y, (0, 2, 1))
            ys.append(y)

        return ys
