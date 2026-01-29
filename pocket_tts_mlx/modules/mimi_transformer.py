"""Mimi transformer modules for MLX.

This module implements transformer layers for the Mimi audio codec.
It's a separate implementation from the FlowLM transformer with different
attention patterns for audio processing.
"""

from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.modules.attention import MimiStreamingMultiheadAttention
from pocket_tts_mlx.modules.layer_scale import LayerScale
from pocket_tts_mlx.modules.rope import RotaryEmbedding
from pocket_tts_mlx.modules.stateful_module import StatefulModule
from pocket_tts_mlx.modules.transformer import LayerNorm
from pocket_tts_mlx.utils.config import FlowLMTransformerConfig


class StreamingTransformerLayer(nn.Module):
    """Streaming transformer layer for Mimi.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        dim_feedforward: Dimension of feed-forward network.
        context: Context window size (None means unlimited).
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

        # Use Mimi-style attention
        if attention_kind == "mimi":
            self.self_attn = MimiStreamingMultiheadAttention(
                context=context, rope=rope, embed_dim=d_model, num_heads=num_heads
            )
        else:
            from pocket_tts_mlx.modules.attention import StreamingMultiheadAttention
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
        """Apply feed-forward block."""
        x_orig = x
        x = self.norm2(x)
        update = self.linear2(nn.gelu(self.linear1(x)))
        if self.layer_scale_2 is not None:
            update = self.layer_scale_2(update)
        return x_orig + update

    def _sa_block(
        self, x: mx.array, model_state: Any
    ) -> mx.array:
        """Apply self-attention block."""
        x_orig = x
        x = self.norm1(x)
        # Debug: check what self_attn.get_state will try to access
        if hasattr(self.self_attn, '_module_absolute_name'):
            key = self.self_attn._module_absolute_name
        else:
            key = "NO_MODULE_NAME"
        update = self.self_attn(x, model_state)
        if self.layer_scale_1 is not None:
            update = self.layer_scale_1(update)
        return x_orig + update

    def __call__(
        self, x: mx.array, model_state: Any
    ) -> mx.array:
        """Forward pass."""
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
        dim_feedforward: Dimension of feed-forward network.
        context: Context window size for attention.
        max_period: Maximum period for rotary embeddings.
        kind: Type of attention ("mimi" or "flow_lm").
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        layer_scale: Optional[float] = None,
        dim_feedforward: int = 2048,
        context: Optional[int] = None,
        max_period: float = 10_000.0,
        kind: str = "mimi",
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.max_period = max_period
        self.rope = RotaryEmbedding(max_period=max_period)

        self.layers = []
        for _ in range(num_layers):
            layer = StreamingTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                context=context,
                rope=self.rope,
                layer_scale=layer_scale,
                attention_kind=kind,
            )
            self.layers.append(layer)

    @classmethod
    def from_pydantic_config(cls, config: FlowLMTransformerConfig):
        """Create StreamingTransformer from pydantic config.

        Args:
            config: FlowLM transformer configuration.

        Returns:
            StreamingTransformer instance.
        """
        dim_feedforward = int(config.d_model * config.hidden_scale)
        return cls(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dim_feedforward=dim_feedforward,
            max_period=float(config.max_period),
            kind="flow_lm",
        )

    def __call__(self, x: mx.array, model_state: Any) -> mx.array:
        """Forward pass through all layers."""
        # Extract only transformer's sub-state for efficiency
        # Filter model_state to only include keys starting with our prefix
        if model_state is not None:
            # Try to find what prefix to use by checking the first layer's attention module
            if hasattr(self.layers[0], 'self_attn') and hasattr(self.layers[0].self_attn, '_module_absolute_name'):
                prefix = self.layers[0].self_attn._module_absolute_name
                # Extract everything before 'layers' to get the transformer's prefix
                parts = prefix.split('.layers.')
                transformer_prefix = parts[0] if len(parts) > 1 else prefix
                # Filter state to only include keys under this prefix
                filtered_state = {}
                for key, value in model_state.items():
                    if key.startswith(transformer_prefix):
                        filtered_state[key] = value
                model_state = filtered_state

        for layer in self.layers:
            x = layer(x, model_state)
        return x


class ProjectedTransformer(nn.Module):
    """Transformer with input and output projections.

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
            model_state: Model state for streaming.

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
