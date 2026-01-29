"""Mimi audio codec model for MLX.

This module implements the Mimi audio compression model using MLX.
Mimi encodes audio into latent representations and decodes back to audio.

The implementation mirrors the PyTorch version but uses MLX operations.
"""

from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.modules.conv import pad_for_conv1d
from pocket_tts_mlx.modules.dummy_quantizer import DummyQuantizer
from pocket_tts_mlx.modules.mimi_transformer import ProjectedTransformer
from pocket_tts_mlx.modules.resample import ConvDownsample1d, ConvTrUpsample1d
from pocket_tts_mlx.modules.seanet import SEANetDecoder, SEANetEncoder
from pocket_tts_mlx.modules.stateful_module import StatefulModule, init_states


class MimiModel(nn.Module):
    """Mimi audio compression model.

    This model encodes audio to latent representations and decodes back to audio.
    It includes encoder/decoder SEANet networks and transformer layers.

    Args:
        encoder: SEANet encoder.
        decoder: SEANet decoder.
        quantizer: Dummy quantizer (projection only).
        frame_rate: Output frame rate.
        encoder_frame_rate: Encoder output frame rate.
        sample_rate: Audio sample rate.
        channels: Number of audio channels.
        encoder_transformer: Encoder transformer.
        decoder_transformer: Decoder transformer.
    """

    def __init__(
        self,
        encoder: SEANetEncoder,
        decoder: SEANetDecoder,
        quantizer: DummyQuantizer,
        frame_rate: float,
        encoder_frame_rate: float,
        sample_rate: int,
        channels: int,
        encoder_transformer: ProjectedTransformer,
        decoder_transformer: ProjectedTransformer,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.encoder_transformer = encoder_transformer
        self.decoder_transformer = decoder_transformer
        self.quantizer = quantizer
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.encoder_frame_rate = encoder_frame_rate

        # Get dimension from encoder
        dimension = encoder.dimension
        assert isinstance(dimension, int), f"Dimension should be int, got {dimension} of type {type(dimension)}"
        self.dimension = dimension

        # Create resampling layers if frame rates differ
        if encoder_frame_rate != frame_rate:
            assert encoder_frame_rate > frame_rate, "Cannot upsample with conv."
            downsample_stride = encoder_frame_rate / frame_rate
            assert downsample_stride == int(downsample_stride), f"Only integer strides are supported, got {downsample_stride}"
            self.downsample = ConvDownsample1d(int(downsample_stride), dimension=dimension)
            self.upsample = ConvTrUpsample1d(int(downsample_stride), dimension=dimension)
        else:
            self.downsample = None
            self.upsample = None

    @property
    def frame_size(self) -> int:
        """Get frame size in samples."""
        return int(self.sample_rate / self.frame_rate)

    def _to_framerate(self, x: mx.array) -> mx.array:
        """Convert from encoder frame rate to overall frame rate.

        Args:
            x: Input tensor at encoder frame rate.

        Returns:
            Resampled tensor at target frame rate.
        """
        if self.encoder_frame_rate == self.frame_rate:
            return x
        return self.downsample(x, model_state=None)

    def _to_encoder_framerate(self, x: mx.array, mimi_state: Optional[Dict]) -> mx.array:
        """Convert from overall frame rate to encoder frame rate.

        Args:
            x: Input tensor at target frame rate.
            mimi_state: Model state for streaming.

        Returns:
            Resampled tensor at encoder frame rate.
        """
        if self.encoder_frame_rate == self.frame_rate:
            return x
        return self.upsample(x, mimi_state)

    def decode_from_latent(self, latent: mx.array, mimi_state: Optional[Dict]) -> mx.array:
        """Decode audio from latent representation.

        Args:
            latent: Latent tensor of shape [B, D, T].
            mimi_state: Model state for streaming.

        Returns:
            Audio tensor of shape [B, C, T'].
        """
        emb = self._to_encoder_framerate(latent, mimi_state)
        (emb,) = self.decoder_transformer(emb, mimi_state)
        out = self.decoder(emb, mimi_state)
        return out

    def encode_to_latent(self, x: mx.array) -> mx.array:
        """Encode audio to latent representation.

        Args:
            x: Audio waveform tensor of shape [B, C, T].

        Returns:
            Latent tensor of shape [B, D, T'].
        """
        assert x.ndim == 3, f"Expected audio of shape [B, C, T] but got {x.shape}"

        frame_size = self.frame_size

        # Pad input to exact multiple of frame size
        x = pad_for_conv1d(x, frame_size, frame_size)

        # Encode with SEANet encoder
        emb = self.encoder(x, model_state=None)

        # Apply encoder transformer
        (emb,) = self.encoder_transformer(emb, model_state=None)

        # Convert to target frame rate
        emb = self._to_framerate(emb)

        return emb
