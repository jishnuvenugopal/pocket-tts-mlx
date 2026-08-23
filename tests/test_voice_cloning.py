from types import SimpleNamespace
import unittest

import mlx.core as mx
import numpy as np

from pocket_tts_mlx.models.tts_model import TTSModel


class _MimiEncoder:
    def encode_to_latent(self, audio):
        del audio
        return mx.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])


class _VoiceCloningHarness:
    _encode_audio = TTSModel._encode_audio

    def __init__(self):
        self.mimi = _MimiEncoder()
        self.flow_lm = SimpleNamespace(speaker_proj_weight=mx.eye(2))


class VoiceCloningTests(unittest.TestCase):
    def test_encode_audio_swaps_latent_time_and_channel_axes(self):
        conditioning = _VoiceCloningHarness()._encode_audio(mx.zeros((1, 1, 4)))

        np.testing.assert_array_equal(
            np.asarray(conditioning),
            np.array([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
