import json
from pathlib import Path
import tempfile
import unittest

from pocket_tts_mlx.text_normalization import (
    DictionaryEntry,
    UserDictionary,
    normalize_text,
)


class DictionaryEntryTests(unittest.TestCase):
    def test_literal_matches_whole_word_only(self):
        entry = DictionaryEntry(match="API", replace="ay pee eye")
        pattern = entry.compile()

        self.assertEqual(
            pattern.sub(entry.replace, "The API grew RAPIDLY."),
            "The ay pee eye grew RAPIDLY.",
        )

    def test_case_insensitive_and_regex_entries(self):
        dictionary = UserDictionary.from_dict(
            {
                "english": [
                    {
                        "match": "mlx",
                        "replace": "em el ex",
                        "case_insensitive": True,
                    },
                    {
                        "match": r"Mr\.?\s+",
                        "replace": "Mister ",
                        "regex": True,
                    },
                ]
            }
        )

        self.assertEqual(
            dictionary.apply("Mr. MLX", "english"),
            "Mister em el ex",
        )

    def test_invalid_regex_is_rejected_eagerly(self):
        with self.assertRaisesRegex(ValueError, "Invalid regex"):
            UserDictionary.from_dict(
                {"english": [{"match": "(open", "replace": "x", "regex": True}]}
            )


class UserDictionaryTests(unittest.TestCase):
    def test_language_entries_run_before_common_entries(self):
        dictionary = UserDictionary.from_dict(
            {
                "english": [{"match": "Bob", "replace": "Robert"}],
                "common": [{"match": "Robert", "replace": "Bobby"}],
            }
        )

        self.assertEqual(dictionary.apply("Hello Bob.", "english"), "Hello Bobby.")

    def test_merge_preserves_order_without_mutating_inputs(self):
        base = UserDictionary.from_dict(
            {"english": [{"match": "API", "replace": "ay pee eye"}]}
        )
        persona = UserDictionary.from_dict(
            {"english": [{"match": "Hormozi", "replace": "hor moh zee"}]}
        )

        merged = base.merge(persona)

        self.assertEqual(
            merged.apply("Hormozi API", "english"),
            "hor moh zee ay pee eye",
        )
        self.assertEqual(base.apply("Hormozi API", "english"), "Hormozi ay pee eye")

    def test_json_file_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dictionary.json"
            path.write_text(
                json.dumps(
                    {"english": [{"match": "MLX", "replace": "em el ex"}]}
                ),
                encoding="utf-8",
            )

            dictionary = UserDictionary.from_file(path)

        self.assertEqual(dictionary.apply("MLX", "english"), "em el ex")


class NormalizeTextTests(unittest.TestCase):
    def test_money_runs_before_decimal_normalization(self):
        self.assertEqual(
            normalize_text("It costs $3.02."),
            "It costs 3 dollars and 2 cents.",
        )

    def test_decimal_is_language_aware(self):
        self.assertEqual(normalize_text("37.0", language="german"), "37 Komma 0")

    def test_dictionary_runs_after_builtin_normalizers(self):
        dictionary = UserDictionary.from_dict(
            {"english": [{"match": "dollars", "replace": "bucks"}]}
        )

        self.assertEqual(
            normalize_text("It costs $3.02.", dictionary=dictionary),
            "It costs 3 bucks and 2 cents.",
        )


if __name__ == "__main__":
    unittest.main()
