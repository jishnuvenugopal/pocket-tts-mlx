"""Streaming transformer blocks for Mimi and FlowLM backbones."""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.modules.attention import MimiStreamingMultiheadAttention, StreamingMultiheadAttention
from pocket_tts_mlx.modules.layer_scale import LayerScale
from pocket_tts_mlx.modules.rope import RotaryEmbedding
from pocket_tts_mlx.modules.stateful_module import StatefulModule
from pocket_tts_mlx.utils.config import FlowLMTransformerConfig


class StreamingTransformerLayer(nn.Module):
    """Transformer layer with streaming attention and feed-forward blocks."""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        context: int | None,
        rope: RotaryEmbedding,
        layer_scale: float | None = None,
        attention_kind: str = "mimi",
    ):
        super().__init__()
        if attention_kind == "mimi":
            self.self_attn = MimiStreamingMultiheadAttention(
                context=context, rope=rope, embed_dim=d_model, num_heads=num_heads
            )
        else:
            self.self_attn = StreamingMultiheadAttention(
                rope=rope, embed_dim=d_model, num_heads=num_heads
            )

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)

        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale)
            self.layer_scale_2 = LayerScale(d_model, layer_scale)

    def _ff_block(self, x: mx.array) -> mx.array:
        """Feed-forward block with pre-norm and residual."""
        x_orig = x
        x = self.norm2(x)
        update = self.linear2(nn.gelu(self.linear1(x)))
        return x_orig + self.layer_scale_2(update)

    def _sa_block(self, x: mx.array, model_state: dict | None) -> mx.array:
        """Self-attention block with pre-norm and residual."""
        x_orig = x
        x = self.norm1(x)
        update = self.self_attn(x, model_state)
        return x_orig + self.layer_scale_1(update)

    def __call__(self, x: mx.array, model_state: dict | None) -> mx.array:
        x = self._sa_block(x, model_state)
        x = self._ff_block(x)
        return x


class StreamingTransformer(nn.Module):
    """Stack of streaming transformer layers with shared RoPE."""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        layer_scale: float | None = None,
        dim_feedforward: int | list[int] = 2048,
        context: int | None = None,
        max_period: float = 10_000.0,
        kind: str = "mimi",
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.max_period = max_period
        self.rope = RotaryEmbedding(max_period=max_period)

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                StreamingTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    context=context,
                    rope=self.rope,
                    layer_scale=layer_scale,
                    attention_kind=kind,
                )
            )

    @classmethod
    def from_pydantic_config(cls, config: FlowLMTransformerConfig):
        dim_feedforward = int(config.d_model * config.hidden_scale)
        return cls(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dim_feedforward=dim_feedforward,
            max_period=float(config.max_period),
            kind="flow_lm",
        )

    def __call__(self, x: mx.array, model_state: dict | None):
        """Apply transformer layers sequentially."""
        for layer in self.layers:
            x = layer(x, model_state)
        return x


class ProjectedTransformer(nn.Module):
    """Transformer with input/output projections for Mimi encoder/decoder."""
    def __init__(
        self,
        input_dimension: int,
        output_dimensions: tuple[int, ...],
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
        self.input_proj = None
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)

        self.output_projs = []
        for output_dimension in output_dimensions:
            if d_model == output_dimension:
                self.output_projs.append(nn.Identity())
            else:
                self.output_projs.append(nn.Linear(d_model, output_dimension, bias=False))

    def __call__(self, x: mx.array, model_state: dict | None):
        """Project inputs, run transformer, then project outputs."""
        x = mx.transpose(x, (0, 2, 1))
        if self.input_proj is not None:
            x = self.input_proj(x)
        z = self.transformer(x, model_state)
        ys = []
        for output_proj in self.output_projs:
            y = output_proj(z)
            y = mx.transpose(y, (0, 2, 1))
            ys.append(y)
        return ys
