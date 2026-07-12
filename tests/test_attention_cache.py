import unittest

import mlx.core as mx
import numpy as np

from pocket_tts_mlx.modules.attention import complete_mimi_kv


class MimiCacheTests(unittest.TestCase):
    def test_vectorized_update_matches_ring_buffer_positions(self):
        batch, heads, steps, dimension, capacity = 2, 2, 4, 3, 8
        cache = mx.zeros((2, batch, heads, capacity, dimension))
        keys = mx.arange(batch * heads * steps * dimension).reshape(
            batch, heads, steps, dimension
        )
        values = keys + 1000
        offsets = mx.array([2, 7])

        result, new_offsets = complete_mimi_kv(cache, offsets, keys, values)
        mx.eval(result.keys, result.values, result.positions, new_offsets)

        expected_keys = np.zeros((batch, heads, capacity, dimension), dtype=np.float32)
        expected_values = np.zeros_like(expected_keys)
        keys_np = np.asarray(keys)
        values_np = np.asarray(values)
        for b, offset in enumerate((2, 7)):
            for step in range(steps):
                index = (offset + step) % capacity
                expected_keys[b, :, index, :] = keys_np[b, :, step, :]
                expected_values[b, :, index, :] = values_np[b, :, step, :]

        np.testing.assert_array_equal(np.asarray(result.keys), expected_keys)
        np.testing.assert_array_equal(np.asarray(result.values), expected_values)
        np.testing.assert_array_equal(np.asarray(new_offsets), np.array([6, 11]))

    def test_rejects_update_larger_than_cache(self):
        cache = mx.zeros((2, 1, 1, 2, 1))
        update = mx.zeros((1, 1, 3, 1))

        with self.assertRaisesRegex(ValueError, "cannot exceed cache capacity"):
            complete_mimi_kv(cache, mx.array([0]), update, update)


if __name__ == "__main__":
    unittest.main()
