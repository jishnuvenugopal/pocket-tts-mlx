"""Text normalization registry for TTS preprocessing.

Each registered normalizer rewrites a specific surface pattern (decimals,
currency, ...) into a spoken form *before* tokenisation, preventing the
sentence splitter in :mod:`pocket_tts.models.tts_model` from breaking on
punctuation that is structural rather than prosodic.

The registry is ordered: normalizers run sequentially via
:func:`normalize_text`, and earlier entries see the original text while
later entries see the partially-rewritten output.  Order matters when one
pattern is a substring of another -- e.g. money (``$3.02``) must run
before plain decimals (``3.02``), otherwise ``$3.02`` would first be
rewritten to ``$3 point 02`` and then never recognised as currency.

Adding a new pattern is one entry in :data:`NORMALIZERS`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Normalizer:
    """A single text-normalization pattern.

    Attributes:
        name: Human-readable identifier (used in tests and debugging).
        pattern: Compiled regex matching the surface form to rewrite.
        handler: Callable receiving the regex ``Match`` and the language
            stem; returns the spoken-form replacement string.
    """

    name: str
    pattern: re.Pattern[str]
    handler: Callable[[re.Match[str], str], str]


# ---------------------------------------------------------------------------
# Decimals
# ---------------------------------------------------------------------------

# Spoken form of the decimal point for each supported language config stem.
# Languages not listed here fall back to ``"point"``.
DECIMAL_WORD: dict[str, str] = {
    "english": "point",
    "french": "virgule",
    "french_24l": "virgule",
    "german": "Komma",
    "german_24l": "Komma",
    "spanish": "coma",
    "spanish_24l": "coma",
    "portuguese": "vírgula",
    "portuguese_24l": "vírgula",
    "italian": "virgola",
    "italian_24l": "virgola",
}

_DECIMAL_RE = re.compile(r"(\d+)\.(\d+)")


def _decimal_handler(match: re.Match[str], language: str) -> str:
    word = DECIMAL_WORD.get(language, "point")
    return f"{match.group(1)} {word} {match.group(2)}"


# ---------------------------------------------------------------------------
# Currency / money
# ---------------------------------------------------------------------------

# Currency symbol -> (unit_singular, unit_plural, fraction_singular, fraction_plural).
# Wording is English: the symbols imply USD/EUR/GBP and inline use of these
# symbols in non-English text is rare enough that we don't translate.  Add
# language-specific overrides here if needed.
CURRENCY_WORDS: dict[str, tuple[str, str, str, str]] = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "€": ("euro", "euros", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
}

# Match a currency symbol followed by either:
#   - integer-with-optional-thousands and optional .cents:  $3 / $1,234 / $3.02 / $1,234.56
#   - cents-only form with no integer part:                  $.50
_MONEY_RE = re.compile(
    r"([$€£])(?:(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?|\.(\d{2}))"
)


def _money_handler(match: re.Match[str], language: str) -> str:
    del language  # currency wording is symbol-driven, not language-driven
    symbol = match.group(1)
    int_str = match.group(2)
    cents_str = match.group(3) if match.group(3) is not None else match.group(4)

    unit_singular, unit_plural, frac_singular, frac_plural = CURRENCY_WORDS[symbol]

    units = int(int_str.replace(",", "")) if int_str is not None else 0
    fractional = int(cents_str) if cents_str is not None else None

    parts: list[str] = []
    if units > 0:
        word = unit_singular if units == 1 else unit_plural
        parts.append(f"{units} {word}")
    if fractional is not None and fractional > 0:
        if parts:
            parts.append("and")
        word = frac_singular if fractional == 1 else frac_plural
        parts.append(f"{fractional} {word}")
    if not parts:
        # $0 / $0.00 -- preserve "0 dollars" rather than emitting nothing
        parts.append(f"0 {unit_plural}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

NORMALIZERS: tuple[Normalizer, ...] = (
    # Money runs first: ``$3.02`` contains a digit.digit pattern that the
    # decimal normalizer would otherwise rewrite to ``$3 point 02``.
    Normalizer(name="money", pattern=_MONEY_RE, handler=_money_handler),
    Normalizer(name="decimal", pattern=_DECIMAL_RE, handler=_decimal_handler),
)


# ---------------------------------------------------------------------------
# User dictionary
# ---------------------------------------------------------------------------

# Section name treated as "applies to every language".
COMMON_SECTION = "common"


def _is_word_char(char: str) -> bool:
    """Return True if ``char`` is a ``\\b``-significant word character."""
    return char.isalnum() or char == "_"


@dataclass(frozen=True)
class DictionaryEntry:
    """A single user-defined pronunciation override.

    Attributes:
        match: The text to find.  When ``regex`` is False (default), this is
            treated as a literal string anchored on word boundaries.  When
            ``regex`` is True, it is compiled directly as a Python regex.
        replace: The replacement text (supports backreferences like ``\\1``
            when ``regex`` is True).
        regex: If True, treat ``match`` as a regex; otherwise as a literal
            with word-boundary anchoring.  Defaults to False.
        case_insensitive: If True, match is case-insensitive.  Defaults to
            False.
    """

    match: str
    replace: str
    regex: bool = False
    case_insensitive: bool = False

    def compile(self) -> re.Pattern[str]:
        """Compile this entry to a :class:`re.Pattern`.

        For literal matches, ``\\b`` is added on each side only when the
        adjacent character of ``match`` is a word character (``[A-Za-z0-9_]``).
        ``\\b`` only fires at a word/non-word transition, so anchoring it
        next to a non-word character (e.g. the trailing ``.`` in ``"Mr."``)
        would prevent the pattern from ever matching when followed by
        another non-word character such as whitespace.

        Raises:
            re.error: If ``regex`` is True and ``match`` is malformed.
            ValueError: If ``match`` is empty.
        """
        if not self.match:
            raise ValueError("DictionaryEntry.match must be non-empty")
        flags = re.IGNORECASE if self.case_insensitive else 0
        if self.regex:
            return re.compile(self.match, flags)
        prefix = r"\b" if _is_word_char(self.match[0]) else ""
        suffix = r"\b" if _is_word_char(self.match[-1]) else ""
        return re.compile(prefix + re.escape(self.match) + suffix, flags)


@dataclass
class UserDictionary:
    """User-supplied pronunciation overrides, organised per language.

    Applied *after* the built-in normalizers (decimals, money, ...) so the
    user can layer overrides on top of structural rewrites.  For each call
    to :meth:`apply`, entries under the requested ``language`` section run
    first, followed by entries under the :data:`COMMON_SECTION` section.

    Construct via :meth:`from_dict`, :meth:`from_file`, or directly with a
    pre-built ``entries`` mapping.

    Example:
        >>> d = UserDictionary.from_dict({
        ...     "english": [
        ...         {"match": "API", "replace": "ay pee eye"},
        ...         {"match": "lol", "replace": "laugh out loud",
        ...          "case_insensitive": True},
        ...     ],
        ...     "common": [{"match": "&", "replace": " and "}],
        ... })
        >>> d.apply("The API is great, lol & I love it.", "english")
        'The ay pee eye is great, laugh out loud  and  I love it.'
    """

    entries: dict[str, list[DictionaryEntry]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate regex syntax eagerly so bad entries fail fast at load time
        # rather than on first use.
        for section, items in self.entries.items():
            for entry in items:
                try:
                    entry.compile()
                except re.error as exc:
                    raise ValueError(
                        f"Invalid regex in dictionary section {section!r}: "
                        f"{entry.match!r} ({exc})"
                    ) from exc

    @classmethod
    def from_dict(cls, raw: dict[str, Iterable[dict[str, Any] | DictionaryEntry]]) -> "UserDictionary":
        """Build a :class:`UserDictionary` from a nested mapping.

        ``raw`` must map language section names to iterables of either
        ``DictionaryEntry`` instances or plain ``dict`` rows with the same
        keys (``match`` required, ``replace`` required, ``regex`` /
        ``case_insensitive`` optional).
        """
        entries: dict[str, list[DictionaryEntry]] = {}
        for section, items in raw.items():
            section_entries: list[DictionaryEntry] = []
            for item in items:
                if isinstance(item, DictionaryEntry):
                    section_entries.append(item)
                    continue
                if "match" not in item or "replace" not in item:
                    raise ValueError(
                        f"Dictionary entry in section {section!r} must have "
                        f"'match' and 'replace' keys; got {item!r}"
                    )
                section_entries.append(
                    DictionaryEntry(
                        match=item["match"],
                        replace=item["replace"],
                        regex=bool(item.get("regex", False)),
                        case_insensitive=bool(item.get("case_insensitive", False)),
                    )
                )
            entries[section] = section_entries
        return cls(entries=entries)

    @classmethod
    def from_file(cls, path: str | Path) -> "UserDictionary":
        """Load a dictionary from a JSON or YAML file (auto-detected by extension).

        ``.json`` files use the stdlib ``json`` module.  ``.yaml`` / ``.yml``
        files require ``pyyaml`` to be installed; install it via the
        ``dictionary`` extra: ``pip install pocket-tts[dictionary]``.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        text = path.read_text(encoding="utf-8")
        if suffix == ".json":
            raw = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "Loading YAML dictionaries requires pyyaml.  Install with "
                    "`pip install pocket-tts[dictionary]` or use a .json file."
                ) from exc
            raw = yaml.safe_load(text)
        else:
            raise ValueError(
                f"Unsupported dictionary file extension: {suffix!r}.  "
                f"Use .json, .yaml, or .yml."
            )
        if not isinstance(raw, dict):
            raise ValueError(
                f"Dictionary file {path} must contain a top-level mapping of "
                f"language -> entries; got {type(raw).__name__}."
            )
        return cls.from_dict(raw)

    def merge(self, other: "UserDictionary") -> "UserDictionary":
        """Return a new dictionary containing ``self`` then ``other``.

        Within each section, entries from ``other`` run *after* entries from
        ``self``, so ``other`` can override a substitution produced by
        ``self`` (later substitutions act on the result of earlier ones).
        Section order in the result follows insertion: sections only in
        ``self`` keep their original position; new sections from ``other``
        are appended.
        """
        merged: dict[str, list[DictionaryEntry]] = {
            section: list(items) for section, items in self.entries.items()
        }
        for section, items in other.entries.items():
            merged.setdefault(section, []).extend(items)
        return UserDictionary(entries=merged)

    def apply(self, text: str, language: str = "english") -> str:
        """Run every entry whose section matches ``language`` (plus common).

        Section lookup is case-sensitive: ``"English"`` and ``"english"`` are
        distinct sections.  An empty dictionary returns ``text`` unchanged.
        """
        for section in (language, COMMON_SECTION):
            for entry in self.entries.get(section, ()):
                pattern = entry.compile()
                text = pattern.sub(entry.replace, text)
        return text


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def normalize_text(
    text: str,
    language: str = "english",
    dictionary: UserDictionary | None = None,
) -> str:
    """Apply every registered normalizer (and the optional dictionary) to ``text``.

    Args:
        text: Input text to normalise.
        language: Language config stem (e.g. ``"english"``, ``"german"``).
            Controls the spoken form chosen by language-aware normalizers
            such as :data:`DECIMAL_WORD`, and selects the section consulted
            in ``dictionary``.  Defaults to ``"english"``.
        dictionary: Optional :class:`UserDictionary` of pronunciation
            overrides to apply *after* the built-in normalizers.  ``None``
            (default) skips the user-dictionary pass entirely.

    Returns:
        Text with all matched patterns rewritten to spoken form.
    """
    for normalizer in NORMALIZERS:
        handler = normalizer.handler
        text = normalizer.pattern.sub(lambda m, _h=handler: _h(m, language), text)
    if dictionary is not None:
        text = dictionary.apply(text, language=language)
    return text
