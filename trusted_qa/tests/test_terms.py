"""Frozen Hindi coverage cannot be reduced to model-selected names."""

from __future__ import annotations

import unittest

from trusted_qa.common import QaHold
from trusted_qa.terms import contains_adjacent_words, required_beat_terms


class RequiredBeatTermsTests(unittest.TestCase):
    def test_derives_every_unique_non_grammar_word(self) -> None:
        self.assertEqual(required_beat_terms("राम, नल से कहते हैं। राम नल के लिए।"),
                         ["राम", "नल", "कहते"])
        self.assertTrue(contains_adjacent_words("नल", "राम नल से कहते हैं।"))
        self.assertFalse(contains_adjacent_words("नल", "अनल से कहते हैं।"))

    def test_uncheckable_spoken_text_holds(self) -> None:
        for beat in ("राम 15 पर हैं।", "राम says नल।", "का से हैं।"):
            with self.subTest(beat=beat), self.assertRaises(QaHold):
                required_beat_terms(beat)


if __name__ == "__main__":
    unittest.main()
