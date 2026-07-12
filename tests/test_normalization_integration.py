from dataclasses import dataclass
import unittest

from pocket_tts_mlx.models.tts_model import split_into_best_sentences
from pocket_tts_mlx.text_normalization import UserDictionary


@dataclass
class _Tokens:
    tokens: object


class _TokenRow(list):
    def tolist(self):
        return list(self)


class _Tokenizer:
    def __init__(self):
        self.sp = self
        self._decoded = {}

    def __call__(self, text):
        ids = [999]
        for char in text:
            token = ord(char)
            ids.append(token)
            self._decoded[token] = char
        return _Tokens([_TokenRow(ids)])

    def decode(self, tokens):
        return "".join(self._decoded.get(token, "") for token in tokens if token != 999)


class NormalizationIntegrationTests(unittest.TestCase):
    def test_decimal_is_normalized_before_sentence_splitting(self):
        chunks = split_into_best_sentences(_Tokenizer(), "Pi is 3.14.", max_tokens=50)

        self.assertEqual(chunks, ["Pi is 3 point 14."])

    def test_dictionary_is_applied_before_tokenization(self):
        dictionary = UserDictionary.from_dict(
            {"english": [{"match": "MLX", "replace": "em el ex"}]}
        )

        chunks = split_into_best_sentences(
            _Tokenizer(),
            "MLX is fast.",
            max_tokens=50,
            dictionary=dictionary,
        )

        self.assertEqual(chunks, ["em el ex is fast."])


if __name__ == "__main__":
    unittest.main()
