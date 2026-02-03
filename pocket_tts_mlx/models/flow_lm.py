"""Flow Language Model for MLX."""

import logging
from functools import partial
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.conditioners.text import LUTConditioner
from pocket_tts_mlx.modules.mimi_transformer import StreamingTransformer
from pocket_tts_mlx.modules.mlp import SimpleMLPAdaLN
from pocket_tts_mlx.utils.config import FlowLMConfig

logger = logging.getLogger(__name__)


def lsd_decode(v_t, x_0: mx.array, num_steps: int = 1) -> mx.array:
    """Low-step ODE solver for flow matching decoding."""
    current = x_0
    for i in range(num_steps):
        s = i / num_steps
        t = (i + 1) / num_steps
        s_tensor = mx.full((current.shape[0], 1), s)
        t_tensor = mx.full((current.shape[0], 1), t)
        flow_dir = v_t(s_tensor, t_tensor, current)
        current = current + flow_dir / num_steps
    return current


class FlowLMModel(nn.Module):
    """Flow language model with transformer backbone and flow MLP."""
    def __init__(
        self,
        conditioner: LUTConditioner,
        flow_net: SimpleMLPAdaLN,
        transformer: StreamingTransformer,
        dim: int = 128,
        ldim: int = 64,
        stats_ema_decay: float = 0.999,
        text_padding_weight: float = 1.0,
        dtype=None,
    ):
        super().__init__()
        self.conditioner = conditioner
        self.ldim = ldim
        self.stats_ema_decay = stats_ema_decay
        self.dim = dim
        self.text_padding_weight = text_padding_weight
        self.dtype = dtype

        self.flow_net = flow_net
        self.emb_std = mx.ones((ldim,), dtype=dtype)
        self.emb_mean = mx.zeros((ldim,), dtype=dtype)
        self.bos_emb = mx.random.normal(shape=(ldim,)).astype(dtype)

        self.input_linear = nn.Linear(ldim, dim, bias=False)
        self.transformer = transformer
        self.out_norm = nn.LayerNorm(dim, eps=1e-5)
        self.out_eos = nn.Linear(dim, 1)

    @classmethod
    def from_pydantic_config(cls, config: FlowLMConfig, latent_dim: int):
        d_model = config.transformer.d_model
        flow_mlp = SimpleMLPAdaLN.from_pydantic_config(config, latent_dim, d_model)
        conditioner = LUTConditioner(
            n_bins=config.lookup_table.n_bins,
            tokenizer_path=str(config.lookup_table.tokenizer_path),
            dim=config.lookup_table.dim,
            output_dim=d_model,
        )
        transformer = StreamingTransformer.from_pydantic_config(config.transformer)
        return cls(
            flow_net=flow_mlp,
            transformer=transformer,
            dim=d_model,
            conditioner=conditioner,
            ldim=latent_dim,
            dtype=getattr(mx, config.dtype),
        )

    def __call__(
        self,
        sequence: mx.array,
        text_embeddings: mx.array,
        model_state: Dict,
        lsd_decode_steps: int,
        temp: float,
        noise_clamp: Optional[float],
        eos_threshold: float,
    ) -> Tuple[mx.array, mx.array]:
        """Sample next latent and end-of-sequence flag."""
        sequence = mx.where(mx.isnan(sequence), self.bos_emb, sequence)
        input_ = self.input_linear(sequence)
        transformer_out = self.backbone(input_, text_embeddings, sequence, model_state)
        transformer_out = transformer_out.astype(mx.float32)
        assert lsd_decode_steps > 0

        transformer_out = transformer_out[:, -1]
        eos_logits = self.out_eos(transformer_out)
        is_eos = eos_logits > eos_threshold

        noise_shape = list(transformer_out.shape[:-1]) + [self.ldim]
        std = temp**0.5
        if noise_clamp is None:
            noise = mx.random.normal(noise_shape) * std
        else:
            noise = mx.random.normal(noise_shape) * std
            noise = mx.clip(noise, -noise_clamp, noise_clamp)

        # Condition flow MLP on transformer output.
        conditioned_flow = partial(self.flow_net, transformer_out)
        output = lsd_decode(conditioned_flow, noise, lsd_decode_steps)
        return output, is_eos

    def backbone(self, input_: mx.array, text_embeddings: mx.array, sequence: mx.array, model_state: Dict):
        """Run streaming transformer over text + latent sequence."""
        input_ = mx.concatenate([text_embeddings, input_], axis=1)
        transformer_out = self.transformer(input_, model_state)
        transformer_out = self.out_norm(transformer_out)
        transformer_out = transformer_out[:, -sequence.shape[1]:]
        return transformer_out

    def _sample_next_latent(
        self,
        sequence: mx.array,
        text_embeddings: mx.array,
        model_state: Dict,
        lsd_decode_steps: int,
        temp: float,
        noise_clamp: Optional[float],
        eos_threshold: float,
    ) -> Tuple[mx.array, mx.array]:
        return self(
            sequence=sequence,
            text_embeddings=text_embeddings,
            lsd_decode_steps=lsd_decode_steps,
            temp=temp,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
            model_state=model_state,
        )
