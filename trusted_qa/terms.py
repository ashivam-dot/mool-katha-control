"""Deterministic spoken Hindi terms that both raw ASR citations must show."""

from __future__ import annotations

import unicodedata

from .common import require


GRAMMATICAL_WORDS = frozenset({
    "का", "की", "के", "को", "में", "से", "ने", "पर", "है", "हैं", "था", "थी", "थे", "लिए",
})


def _words(text: str, *, frozen_beat: bool) -> list[str]:
    normalized = unicodedata.normalize("NFC", text)
    words: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            words.append("".join(current))
            current.clear()

    for character in normalized:
        category = unicodedata.category(character)
        if frozen_beat:
            require(not character.isdecimal(),
                    "frozen spoken beat contains a digit that ASR cannot compare deterministically")
            require(not (category.startswith("L") and not "\u0900" <= character <= "\u097f"),
                    "frozen spoken beat contains a non-Devanagari letter")
        if "\u0900" <= character <= "\u097f" and category[0] in "LM":
            current.append(character)
        else:
            flush()
    flush()
    return words


def required_beat_terms(text: str) -> list[str]:
    """Return one term for each unique content word in a frozen spoken beat."""
    require(isinstance(text, str) and bool(text.strip()), "frozen spoken beat is empty")
    required = list(dict.fromkeys(word for word in _words(text, frozen_beat=True)
                                  if word not in GRAMMATICAL_WORDS))
    require(bool(required), "frozen spoken beat has no checkable Hindi content word")
    return required


def contains_adjacent_words(term: str, excerpt: str) -> bool:
    """Match a whole Devanagari word or contiguous phrase, not a substring."""
    if not isinstance(term, str) or not isinstance(excerpt, str):
        return False
    expected = _words(term, frozen_beat=False)
    heard = _words(excerpt, frozen_beat=False)
    return bool(expected) and any(heard[index:index + len(expected)] == expected
                                  for index in range(len(heard) - len(expected) + 1))


def require_exact_ordered_beat(expected: str, recognized: str) -> None:
    """Hold if either recognizer changes a word, its order, or its repetitions."""
    require(isinstance(expected, str) and isinstance(recognized, str),
            "ASR beat comparison needs two spoken Hindi strings")
    frozen_words = _words(expected, frozen_beat=True)
    recognized_words = _words(recognized, frozen_beat=True)
    require(bool(frozen_words) and frozen_words == recognized_words,
            "ASR beat differs in ordered spoken words or repetitions")
