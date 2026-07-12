from types import SimpleNamespace
import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from pocket_tts_mlx.models.tts_model import TTSModel


class _Conditioner:
    def prepare(self, text):
        del text
        return SimpleNamespace(tokens=mx.zeros((1, 1), dtype=mx.int64))


class _FlowLM:
    def __init__(self):
        self.conditioner = _Conditioner()
        self.ldim = 1
        self.dtype = mx.float32
        self.emb_std = mx.ones((1,))
        self.emb_mean = mx.zeros((1,))


class _Mimi(nn.Module):
    def quantizer(self, value):
        return value

    def decode_from_latent(self, latent, state):
        del state
        return mx.broadcast_to(latent[:, :1, :1], (1, 1, 4))


class _SamplingHarness:
    _generate_audio_stream_short_text = TTSModel._generate_audio_stream_short_text

    def __init__(self):
        self.flow_lm = _FlowLM()
        self.mimi = _Mimi()
        self.config = SimpleNamespace(
            mimi=SimpleNamespace(
                transformer=SimpleNamespace(context=8),
                sample_rate=4,
            )
        )
        self.generation_call = 0
        self.temperatures = []

    def _estimate_max_gen_len(self, token_count):
        del token_count
        return 5

    def _flow_lm_current_end(self, model_state):
        del model_state
        return 0

    def _expand_kv_cache(self, model_state, sequence_length):
        del model_state, sequence_length

    def _warmup_mimi_decoder(self, mimi_state, warmup_frames):
        del mimi_state, warmup_frames

    def _run_flow_lm_and_increment_step(
        self,
        model_state,
        text_tokens=None,
        backbone_input_latents=None,
        audio_conditioning=None,
        temperature=None,
        sampling_key=None,
    ):
        del model_state, backbone_input_latents, audio_conditioning
        self.temperatures.append(temperature)
        if text_tokens is not None:
            return mx.zeros((1, 1, 1)), mx.array([[False]])

        eos_values = (False, True, True)
        eos = eos_values[self.generation_call]
        self.generation_call += 1
        latent = mx.random.normal((1, 1, 1), key=sampling_key)
        return latent, mx.array([[eos]])


def _generate(seed, temperature=0.5):
    harness = _SamplingHarness()
    chunks = list(
        harness._generate_audio_stream_short_text(
            model_state={},
            text_to_generate="Hello.",
            frames_after_eos=1,
            copy_state=False,
            warmup_frames=0,
            temperature=temperature,
            sampling_key=mx.random.key(seed),
        )
    )
    return np.concatenate([np.asarray(chunk) for chunk in chunks]), harness.temperatures


class SamplingControlTests(unittest.TestCase):
    def test_same_seed_reproduces_audio_without_global_rng_reset(self):
        first, _ = _generate(seed=42)
        mx.random.normal((100,))  # Perturb global state between requests.
        second, _ = _generate(seed=42)

        np.testing.assert_array_equal(first, second)

    def test_different_seeds_produce_different_audio(self):
        first, _ = _generate(seed=42)
        second, _ = _generate(seed=43)

        self.assertFalse(np.array_equal(first, second))

    def test_temperature_override_reaches_every_flow_step(self):
        _, temperatures = _generate(seed=42, temperature=0.35)

        self.assertTrue(temperatures)
        self.assertTrue(all(value == 0.35 for value in temperatures))


if __name__ == "__main__":
    unittest.main()
