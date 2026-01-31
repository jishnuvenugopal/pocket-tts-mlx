"""MLP modules with adaptive layer normalization for MLX.

This module implements MLP layers with adaptive layer normalization (AdaLN)
for the flow matching model. The implementations mirror the PyTorch versions
but use MLX operations.

Key features:
- RMSNorm and LayerNorm implementations
- Timestep embedding with sinusoidal encoding
- Residual blocks with adaptive modulation
- SimpleMLPAdaLN for diffusion-style flow matching
"""

import math
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.utils.config import FlowLMConfig


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    """Apply adaptive modulation to input tensor.

    Args:
        x: Input tensor.
        shift: Shift parameter.
        scale: Scale parameter.

    Returns:
        Modulated tensor: x * (1 + scale) + shift
    """
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

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
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)

        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias

        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.

    Uses sinusoidal position encoding similar to Transformer.

    Args:
        hidden_size: Output hidden size.
        frequency_embedding_size: Dimension of frequency embedding. Defaults to 256.
        max_period: Maximum period for sinusoidal encoding. Defaults to 10000.
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
    ):
        super().__init__()

        # MLP to process sinusoidal embeddings
        # MLX Sequential doesn't support append, so create with all layers at once
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            RMSNorm(hidden_size),
        )

        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2

        # Precompute frequency bands
        self.freqs = mx.exp(
            -math.log(max_period) * mx.arange(start=0, stop=half) / half
        )

    def __call__(self, t: mx.array) -> mx.array:
        """Embed timesteps.

        Args:
            t: Timestep tensor of shape [B] or [B, 1].

        Returns:
            Embedding tensor of shape [B, hidden_size].
        """
        # Ensure t is 1D
        if t.ndim == 2:
            t = mx.squeeze(t, axis=-1)

        # Compute sinusoidal embeddings
        args = t[:, None] * self.freqs[None, :]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

        # Process through MLP
        t_emb = self.mlp(embedding)
        return t_emb


class ResBlock(nn.Module):
    """Residual block with adaptive layer normalization.

    Args:
        channels: Number of channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        self.in_ln = LayerNorm(channels, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )

    def __call__(self, x: mx.array, y: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, T, C].
            y: Conditioning tensor of shape [B, T, C].

        Returns:
            Output tensor of shape [B, T, C].
        """
        # Get adaptive modulation parameters
        modulation = self.adaLN_modulation(y)
        shift_mlp, scale_mlp, gate_mlp = mx.split(modulation, 3, axis=-1)

        # Apply adaptive normalization
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)

        # Apply MLP
        h = self.mlp(h)

        # Residual connection with gated output
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """Final layer of the flow matching model.

    Similar to DiT (Diffusion Transformer) final layer.

    Args:
        model_channels: Model dimension.
        out_channels: Output dimension.
    """

    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True),
        )

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, T, model_channels].
            c: Conditioning tensor of shape [B, T, model_channels].

        Returns:
            Output tensor of shape [B, T, out_channels].
        """
        # Get adaptive modulation parameters
        shift, scale = mx.split(self.adaLN_modulation(c), 2, axis=-1)

        # Apply adaptive normalization
        x = modulate(self.norm_final(x), shift, scale)

        # Final projection
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """MLP with Adaptive Layer Normalization for Diffusion Loss.

    Taken from https://arxiv.org/abs/2406.11838.

    Args:
        in_channels: Channels in the input tensor.
        model_channels: Base channel count for the model.
        out_channels: Channels in the output tensor.
        cond_channels: Channels in the condition.
        num_res_blocks: Number of residual blocks.
        num_time_conds: Number of time conditions (typically 2 for s and t).
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        cond_channels: int,
        num_res_blocks: int,
        num_time_conds: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.num_time_conds = num_time_conds

        assert num_time_conds != 1, "num_time_conds should not be 1"

        # Time embedders
        self.time_embed = []
        for _ in range(num_time_conds):
            self.time_embed.append(TimestepEmbedder(model_channels))

        # Condition embedder
        self.cond_embed = nn.Linear(cond_channels, model_channels)

        # Input projection
        self.input_proj = nn.Linear(in_channels, model_channels)

        # Residual blocks
        self.res_blocks = []
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResBlock(model_channels))

        # Final layer
        self.final_layer = FinalLayer(model_channels, out_channels)

    @classmethod
    def from_pydantic_config(cls, cfg: FlowLMConfig, latent_dim: int, cond_dim: int):
        """Create SimpleMLPAdaLN from pydantic config.

        Args:
            cfg: FlowLM configuration.
            latent_dim: Latent dimension.
            cond_dim: Conditioning dimension.

        Returns:
            SimpleMLPAdaLN instance.
        """
        config = cfg.flow

        flow_dim = config.dim
        flow_depth = config.depth
        num_time_conds = 2
        return cls(
            latent_dim, flow_dim, latent_dim, cond_dim, flow_depth, num_time_conds=num_time_conds
        )

    def __call__(
        self,
        c: mx.array,
        s: mx.array,
        t: mx.array,
        x: mx.array,
    ) -> mx.array:
        """Forward pass.

        Args:
            c: Conditioning from AR transformer, shape [B, T, cond_channels].
            s: Start time tensor, shape [B] or [B, 1].
            t: Target time tensor, shape [B] or [B, 1].
            x: Input tensor, shape [B, T, in_channels].

        Returns:
            Output tensor of shape [B, T, out_channels].
        """
        # Ensure s and t are 1D
        if s.ndim == 2:
            s = mx.squeeze(s, axis=-1)
        if t.ndim == 2:
            t = mx.squeeze(t, axis=-1)

        # Combine time conditions
        t_combined = mx.stack([self.time_embed[i](ti) for i, ti in enumerate([s, t])], axis=0)
        t_combined = mx.mean(t_combined, axis=0)

        # Add condition
        c_emb = self.cond_embed(c)
        y = t_combined + c_emb

        # Project input
        x = self.input_proj(x)

        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x, y)

        # Final layer
        return self.final_layer(x, y)
