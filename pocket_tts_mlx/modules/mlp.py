"""MLP modules for FlowLM (MLX port)."""

import math

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.utils.config import FlowLMConfig


def modulate(x, shift, scale):
    """Apply adaptive shift and scale."""
    return x * (1 + scale) + shift


def _rms_norm(x: mx.array, alpha: mx.array, eps: float):
    """RMS normalization with variance computed using ddof=1."""
    x_dtype = x.dtype
    var = eps + mx.var(x, axis=-1, keepdims=True, ddof=1)
    y = (x * (alpha.astype(var.dtype) * mx.rsqrt(var))).astype(x_dtype)
    return y


class RMSNorm(nn.Module):
    """RMS normalization with learned scale."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = mx.full((dim,), 1.0)

    def __call__(self, x: mx.array):
        return _rms_norm(x, self.alpha, self.eps)


class LayerNorm(nn.Module):
    """LayerNorm with optional affine parameters."""
    def __init__(self, channels, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = mx.ones((channels,))
            self.bias = mx.zeros((channels,))

    def __call__(self, x: mx.array):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        if hasattr(self, "weight"):
            x = x * self.weight + self.bias
        return x


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding with MLP projection."""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, max_period: int = 10000):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        self.freqs = mx.exp(-math.log(max_period) * mx.arange(half) / half)
        # Mirror PyTorch structure for weight loading: mlp.{0,1,2,3}
        self.mlp = [
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            RMSNorm(hidden_size),
        ]

    def __call__(self, t: mx.array):
        args = t * self.freqs.astype(t.dtype)
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        x = self.mlp[0](embedding)
        x = self.mlp[1](x)
        x = self.mlp[2](x)
        return self.mlp[3](x)


class ResBlock(nn.Module):
    """Residual block with AdaLN modulation."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.in_ln = LayerNorm(channels, eps=1e-6)
        # Mirror PyTorch structure for weight loading: mlp.{0,1,2}
        self.mlp = [
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        ]
        # Mirror PyTorch structure for weight loading: adaLN_modulation.{0,1}
        self.adaLN_modulation = [
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        ]

    def __call__(self, x: mx.array, y: mx.array):
        ada = self.adaLN_modulation[1](self.adaLN_modulation[0](y))
        shift_mlp, scale_mlp, gate_mlp = mx.split(ada, 3, axis=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp[2](self.mlp[1](self.mlp[0](h)))
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """Final AdaLN + linear projection."""
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        # Mirror PyTorch structure for weight loading: adaLN_modulation.{0,1}
        self.adaLN_modulation = [
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True),
        ]

    def __call__(self, x: mx.array, c: mx.array):
        ada = self.adaLN_modulation[1](self.adaLN_modulation[0](c))
        shift, scale = mx.split(ada, 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class SimpleMLPAdaLN(nn.Module):
    """AdaLN-conditioned MLP used for FlowLM flow prediction."""
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        cond_channels,
        num_res_blocks,
        num_time_conds=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.num_time_conds = num_time_conds

        assert num_time_conds != 1
        self.time_embed = [TimestepEmbedder(model_channels) for _ in range(num_time_conds)]
        self.cond_embed = nn.Linear(cond_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = [ResBlock(model_channels) for _ in range(num_res_blocks)]
        self.final_layer = FinalLayer(model_channels, out_channels)

    @classmethod
    def from_pydantic_config(cls, cfg: FlowLMConfig, latent_dim: int, cond_dim: int):
        config = cfg.flow
        flow_dim = config.dim
        flow_depth = config.depth
        num_time_conds = 2
        return SimpleMLPAdaLN(
            latent_dim, flow_dim, latent_dim, cond_dim, flow_depth, num_time_conds=num_time_conds
        )

    def __call__(self, c: mx.array, s: mx.array, t: mx.array, x: mx.array) -> mx.array:
        ts = [s, t]
        x = self.input_proj(x)
        t_combined = (
            sum(self.time_embed[i](ts[i]) for i in range(self.num_time_conds)) / self.num_time_conds
        )
        c = self.cond_embed(c)
        y = t_combined + c
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)
