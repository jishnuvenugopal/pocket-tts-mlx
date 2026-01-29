"""Flow Language Model for MLX.

This module implements the FlowLM model with LSD (Lagrangian Self Distillation)
decoding for text-to-speech. The implementation mirrors the PyTorch version but
uses MLX operations.

FlowLM generates latents autoregressively conditioned on text embeddings.
"""

import logging
from functools import partial
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.conditioners.text import LUTConditioner
from pocket_tts_mlx.modules.mimi_transformer import StreamingTransformer
from pocket_tts_mlx.modules.mlp import SimpleMLPAdaLN
from pocket_tts_mlx.modules.stateful_module import StatefulModule, init_states, increment_steps
from pocket_tts_mlx.utils.config import FlowLMConfig

logger = logging.getLogger(__name__)


def lsd_decode(v_t, x_0: mx.array, num_steps: int = 1) -> mx.array:
    """Rebuild data sample from starting point x_0 using LSD.

    Lagrangian Self Distillation (https://arxiv.org/pdf/2505.18825)

    Args:
        v_t: Function taking (s, t, x_t) and returning the flow direction.
            s is start time (0 to 1), t is target time, x_t is current state.
        x_0: Starting point from the known distribution, shape [B, D].
        num_steps: Number of steps to take.

    Returns:
        x_1_hat: Reconstructed data sample of shape [B, D].
    """
    current = x_0
    for i in range(num_steps):
        s = i / num_steps
        t = (i + 1) / num_steps

        # Create time tensors
        s_tensor = mx.full((current.shape[0], 1), s)
        t_tensor = mx.full((current.shape[0], 1), t)

        # Get flow direction
        flow_dir = v_t(s_tensor, t_tensor, current)

        # Update current state
        current = current + flow_dir / num_steps

    return current


class FlowLMModel(nn.Module):
    """Transformer-based flow language model on latent streams.

    Args:
        conditioner: Text conditioner for processing text inputs.
        flow_net: Trainable function (cond, t, x_t) -> u_t.
        transformer: Streaming transformer for processing sequences.
        dim: Dimension of the transformer encoder.
        ldim: Latent dimension.
        stats_ema_decay: Decay for EMA of latent statistics.
        text_padding_weight: Weight for text padding tokens.
        dtype: Data type for parameters.
    """

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

    @property
    def device(self) -> str:
        """Get device type (MLX uses unified memory)."""
        return "cpu"  # MLX uses unified memory

    @classmethod
    def from_pydantic_config(cls, config: FlowLMConfig, latent_dim: int):
        """Create FlowLMModel from pydantic config.

        Args:
            config: FlowLM configuration.
            latent_dim: Latent dimension.

        Returns:
            FlowLMModel instance.
        """
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
        """Apply language model on sequence and conditions.

        Given a tensor of sequence of shape [B, S, ldim], returns the loss in
        training mode or the reconstructed latent in generation mode.

        Args:
            sequence: Latents to model, shape [B, S, ldim].
            text_embeddings: Pre-computed conditioning tensor.
            model_state: Model state for streaming.
            lsd_decode_steps: Number of LSD decode steps for generation.
            temp: Sampling temperature.
            noise_clamp: Maximum value for noise sampling (None = no clamp).
            eos_threshold: Threshold for end-of-sequence detection.

        Returns:
            Tuple of (output_latents, is_eos) where is_eos indicates EOS tokens.
        """
        # Replace NaN values with BOS embedding
        sequence = mx.where(mx.isnan(sequence), self.bos_emb, sequence)

        # Project input
        input_ = self.input_linear(sequence)

        # Apply backbone
        transformer_out = self.backbone(input_, text_embeddings, sequence, model_state)

        # Convert to float32
        transformer_out = transformer_out.astype(mx.float32)

        assert lsd_decode_steps > 0, "lsd_decode_steps must be positive for generation"

        # Get last timestep output
        transformer_out = transformer_out[:, -1]

        # EOS prediction
        eos_logits = self.out_eos(transformer_out)
        is_eos = eos_logits > eos_threshold

        # Sample noise for LSD
        noise_shape = list(transformer_out.shape[:-1]) + [self.ldim]
        std = temp**0.5

        if noise_clamp is None:
            noise = mx.random.normal(noise_shape) * std
        else:
            # Truncated normal
            noise = mx.random.normal(noise_shape) * std
            noise = mx.clip(noise, -noise_clamp, noise_clamp)

        # LSD decoding
        conditioned_flow = partial(self.flow_net, transformer_out)
        output = lsd_decode(conditioned_flow, noise, lsd_decode_steps)

        return output, is_eos

    def backbone(
        self,
        input_: mx.array,
        text_embeddings: mx.array,
        sequence: mx.array,
        model_state: Dict,
    ) -> mx.array:
        """Apply backbone transformer.

        Args:
            input_: Projected input latents.
            text_embeddings: Text conditioning embeddings.
            sequence: Original sequence for alignment.
            model_state: Model state for streaming.

        Returns:
            Transformer output with text prefix removed.
        """
        # Concatenate text and input
        # Note: one of these is typically empty
        input_ = mx.concatenate([text_embeddings, input_], axis=1)

        # Apply transformer
        transformer_out = self.transformer(input_, model_state)

        # Apply output norm
        transformer_out = self.out_norm(transformer_out)

        # Remove text prefix (keep only sequence portion)
        transformer_out = transformer_out[:, -sequence.shape[1] :]

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
        """Sample next latent from the model.

        Args:
            sequence: Current sequence of shape [B, K, S] where K is number of
                codebooks and S is sequence steps. S = 1 in streaming mode.
            text_embeddings: Condition tensor.
            model_state: Model state for streaming.
            lsd_decode_steps: Number of flow steps for LSD decoding.
            temp: Sampling temperature.
            noise_clamp: Maximum value for noise sampling.
            eos_threshold: EOS detection threshold.

        Returns:
            Tuple of (next_latent, is_eos) where next_latent is [B, 1, ldim]
            and is_eos is [B, 1] indicating EOS positions.
        """
        result = self(
            sequence=sequence,
            text_embeddings=text_embeddings,
            lsd_decode_steps=lsd_decode_steps,
            temp=temp,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
            model_state=model_state,
        )

        return result
