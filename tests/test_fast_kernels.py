import math
import unittest

import mlx.core as mx
import numpy as np

from pocket_tts_mlx.modules.attention import materialize_causal_mask
from pocket_tts_mlx.modules.rope import apply_rope


def _reference_attention(q, k, v, mask):
    scale = q.shape[-1] ** -0.5
    scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale
    if mask.dtype == mx.bool_:
        scores = mx.where(mask, scores, mx.full(scores.shape, -1e9))
    else:
        scores = scores + mask
    return mx.matmul(mx.softmax(scores, axis=-1), v)


def _reference_rope(q, k, offset=0, max_period=10_000):
    batch, steps, heads, dimension = q.shape
    frequencies = mx.exp(
        mx.arange(dimension // 2, dtype=mx.float32)
        * (-math.log(max_period) * 2 / dimension)
    )
    positions = (mx.arange(steps, dtype=mx.float32) + offset).reshape(-1, 1, 1)

    def rotate(x):
        dtype = x.dtype
        pairs = x.reshape(batch, steps, heads, dimension // 2, 2)
        real = pairs[..., 0].astype(mx.float32)
        imaginary = pairs[..., 1].astype(mx.float32)
        cosine = mx.cos(frequencies * positions)
        sine = mx.sin(frequencies * positions)
        output = mx.stack(
            [real * cosine - imaginary * sine, real * sine + imaginary * cosine],
            axis=-1,
        )
        return output.astype(dtype).reshape(batch, steps, heads, dimension)

    return rotate(q), rotate(k)


class FastRoPETests(unittest.TestCase):
    def test_matches_traditional_reference_at_streaming_offset(self):
        q = mx.random.normal((1, 16, 8, 64))
        k = mx.random.normal((1, 16, 8, 64))

        expected_q, expected_k = _reference_rope(q, k, offset=128)
        actual_q, actual_k = apply_rope(q, k, offset=128)
        mx.eval(expected_q, expected_k, actual_q, actual_k)

        np.testing.assert_allclose(actual_q, expected_q, atol=5e-5, rtol=1e-5)
        np.testing.assert_allclose(actual_k, expected_k, atol=5e-5, rtol=1e-5)


class FastAttentionTests(unittest.TestCase):
    def test_flow_prefill_causal_mask_matches_reference(self):
        q = mx.random.normal((1, 16, 4, 64))
        k = mx.random.normal((1, 16, 9, 64))
        v = mx.random.normal((1, 16, 9, 64))
        additive_mask = materialize_causal_mask((4, 9))[None, None]

        expected = _reference_attention(q, k, v, additive_mask)
        actual = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=64**-0.5, mask="causal"
        )
        mx.eval(expected, actual)

        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_flow_single_step_causal_mask_matches_reference(self):
        q = mx.random.normal((1, 16, 1, 64))
        k = mx.random.normal((1, 16, 256, 64))
        v = mx.random.normal((1, 16, 256, 64))
        additive_mask = materialize_causal_mask((1, 256))[None, None]

        expected = _reference_attention(q, k, v, additive_mask)
        actual = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=64**-0.5, mask="causal"
        )
        mx.eval(expected, actual)

        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_mimi_boolean_window_mask_matches_reference(self):
        q = mx.random.normal((1, 8, 16, 64))
        k = mx.random.normal((1, 8, 250, 64))
        v = mx.random.normal((1, 8, 250, 64))
        query_positions = mx.arange(16).reshape(1, 1, 16, 1) + 200
        key_positions = mx.arange(250).reshape(1, 1, 1, 250)
        mask = (key_positions <= query_positions) & (key_positions > query_positions - 250)

        expected = _reference_attention(q, k, v, mask)
        actual = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=64**-0.5, mask=mask
        )
        mx.eval(expected, actual)

        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
