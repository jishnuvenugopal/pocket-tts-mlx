"""Mimi encoder/decoder wrapper with streaming resampling."""

import logging

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.modules.conv import pad_for_conv1d
from pocket_tts_mlx.modules.dummy_quantizer import DummyQuantizer
from pocket_tts_mlx.modules.mimi_transformer import ProjectedTransformer
from pocket_tts_mlx.modules.resample import ConvDownsample1d, ConvTrUpsample1d
from pocket_tts_mlx.modules.seanet import SEANetDecoder, SEANetEncoder

logger = logging.getLogger()


class MimiModel(nn.Module):
    """Audio model that maps waveforms to/from latent frames."""
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

        dimension = encoder.dimension
        assert isinstance(dimension, int)
        self.dimension = dimension

        if encoder_frame_rate != frame_rate:
            # Build streaming resampler between encoder and model frame rates.
            assert self.encoder_frame_rate > self.frame_rate, "Cannot upsample with conv."
            downsample_stride = self.encoder_frame_rate / self.frame_rate
            assert downsample_stride == int(downsample_stride)
            self.downsample = ConvDownsample1d(int(downsample_stride), dimension=dimension)
            self.upsample = ConvTrUpsample1d(int(downsample_stride), dimension=dimension)

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate / self.frame_rate)

    def _to_framerate(self, x: mx.array):
        """Downsample encoder rate to model frame rate."""
        if self.encoder_frame_rate == self.frame_rate:
            return x
        return self.downsample(x, model_state=None)

    def _to_encoder_framerate(self, x: mx.array, mimi_state):
        """Upsample model rate to encoder frame rate."""
        if self.encoder_frame_rate == self.frame_rate:
            return x
        return self.upsample(x, mimi_state)

    def decode_from_latent(self, latent: mx.array, mimi_state):
        """Decode latent frames into waveform."""
        emb = self._to_encoder_framerate(latent, mimi_state)
        (emb,) = self.decoder_transformer(emb, mimi_state)
        out = self.decoder(emb, mimi_state)
        return out

    def encode_to_latent(self, x: mx.array) -> mx.array:
        """Encode waveform into latent frames."""
        assert x.ndim == 3, f"Expected audio of shape [B, C, T] but got {x.shape}"
        frame_size = self.frame_size
        x = pad_for_conv1d(x, frame_size, frame_size)
        emb = self.encoder(x, model_state=None)
        (emb,) = self.encoder_transformer(emb, model_state=None)
        emb = self._to_framerate(emb)
        return emb
