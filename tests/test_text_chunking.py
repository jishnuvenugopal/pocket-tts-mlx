from dataclasses import dataclass
import unittest

from pocket_tts_mlx.models.tts_model import split_into_best_sentences


@dataclass
class _Tokens:
    tokens: object


class _TokenRow(list):
    def tolist(self):
        return list(self)


class _TokenMatrix(list):
    pass


class _CharacterTokenizer:
    """Small tokenizer double that assigns stable IDs to each character."""

    _special = {".": 1001, "!": 1002, "?": 1003, ",": 1004, ";": 1005, ":": 1006}

    def __init__(self):
        self.sp = self
        self._decoded = {}

    def __call__(self, text):
        # The production SentencePiece tokenizer emits a leading whitespace
        # piece; chunking deliberately discards it when extracting delimiters.
        ids = [999]
        for char in text:
            token = self._special.get(char, ord(char))
            ids.append(token)
            self._decoded[token] = char
        return _Tokens(_TokenMatrix([_TokenRow(ids)]))

    def decode(self, tokens):
        return "".join(self._decoded.get(token, "") for token in tokens if token != 999)


class TextChunkingTests(unittest.TestCase):
    def test_oversized_sentence_splits_at_clause_boundaries(self):
        tokenizer = _CharacterTokenizer()
        text = "Alpha clause, beta clause; gamma clause: delta clause."

        chunks = split_into_best_sentences(tokenizer, text, max_tokens=25)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(chunks).replace("  ", " "), text)
        self.assertTrue(
            all(len(tokenizer(chunk).tokens[0].tolist()) <= 25 for chunk in chunks)
        )

    def test_short_sentence_is_not_split_at_commas(self):
        tokenizer = _CharacterTokenizer()
        text = "Short, complete."

        self.assertEqual(split_into_best_sentences(tokenizer, text, max_tokens=50), [text])


if __name__ == "__main__":
    unittest.main()
