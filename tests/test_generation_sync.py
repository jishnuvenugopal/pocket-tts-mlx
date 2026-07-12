from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn

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


class _GenerationHarness:
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
        self._generation_call = 0

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
    ):
        del model_state, audio_conditioning
        if text_tokens is not None:
            return mx.zeros((1, 1, 1)), mx.array([[False]])

        del backbone_input_latents
        eos_values = (False, True, True)
        eos = eos_values[self._generation_call]
        self._generation_call += 1
        return mx.ones((1, 1, 1)) * self._generation_call, mx.array([[eos]])


class GenerationSynchronizationTests(unittest.TestCase):
    def test_eos_and_audio_share_one_eval_per_generation_step(self):
        harness = _GenerationHarness()
        real_eval = mx.eval

        with patch("pocket_tts_mlx.models.tts_model.mx.eval", wraps=real_eval) as evaluate:
            chunks = list(
                harness._generate_audio_stream_short_text(
                    model_state={},
                    text_to_generate="Hello.",
                    frames_after_eos=1,
                    copy_state=False,
                    warmup_frames=0,
                )
            )

        # EOS occurs at step 1. Step 2 is evaluated but discarded, matching
        # the existing frames_after_eos behavior while using one sync per step.
        self.assertEqual(len(chunks), 2)
        self.assertEqual(evaluate.call_count, 3)
        self.assertTrue(all(len(call.args) == 2 for call in evaluate.call_args_list))


if __name__ == "__main__":
    unittest.main()
