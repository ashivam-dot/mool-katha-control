import unittest
from pathlib import Path
from unittest.mock import patch

from lite import gate

SCRIPT = "हनुमान ने सुरसा के मुँह में प्रवेश किया और तुरंत बाहर निकल आए। सुरसा ने उन्हें आशीर्वाद दिया।"
LOUD_OK = {"integrated_lufs": -14.1, "true_peak_dbfs": -1.8}


class CoverageTest(unittest.TestCase):
    def test_exact_transcript_is_full_coverage(self):
        self.assertEqual(gate.coverage(SCRIPT, SCRIPT)["coverage"], 1.0)

    def test_spelling_variants_still_count(self):
        # Nukta, chandrabindu vs anusvara and long/short vowels are folded or fuzzy-matched.
        heard = "हनुमान ने सुरसा के मुंह में प्रवेश किया और तुरंत बाहर निकल आये सुरसा ने उन्हे आशिर्वाद दिया"
        self.assertGreaterEqual(gate.coverage(SCRIPT, heard)["coverage"], 0.95)

    def test_missing_half_the_words_fails(self):
        heard = "हनुमान ने सुरसा के मुँह में"
        result = gate.coverage(SCRIPT, heard)
        self.assertLess(result["coverage"], gate.MIN_COVERAGE)
        self.assertIn("आशीर्वाद", result["missed"])

    def test_empty_script(self):
        self.assertEqual(gate.coverage("", "कुछ")["coverage"], 0.0)


class EvaluateTest(unittest.TestCase):
    def test_all_checks_pass(self):
        result = gate.evaluate(script=SCRIPT, heard=SCRIPT, duration=51.2, loud=LOUD_OK, review_ok=True)
        self.assertTrue(result["passed"])

    def test_duration_outside_window_fails(self):
        for seconds in (41.9, 62.1):
            result = gate.evaluate(script=SCRIPT, heard=SCRIPT, duration=seconds, loud=LOUD_OK, review_ok=True)
            self.assertFalse(result["passed"])
            self.assertFalse(result["checks"]["duration"]["ok"])

    def test_loudness_and_peak(self):
        quiet = gate.evaluate(script=SCRIPT, heard=SCRIPT, duration=50, loud={"integrated_lufs": -20.0,
                              "true_peak_dbfs": -3.0}, review_ok=True)
        hot = gate.evaluate(script=SCRIPT, heard=SCRIPT, duration=50, loud={"integrated_lufs": -14.0,
                            "true_peak_dbfs": 0.2}, review_ok=True)
        self.assertFalse(quiet["checks"]["loudness"]["ok"])
        self.assertFalse(hot["checks"]["loudness"]["ok"])

    def test_failed_source_review_fails_gate(self):
        result = gate.evaluate(script=SCRIPT, heard=SCRIPT, duration=50, loud=LOUD_OK, review_ok=False,
                               review_issues=["beat 3 is not supported"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"]["source_support"]["issues"], ["beat 3 is not supported"])

    def test_second_pass_by_pauses_when_the_first_drops_a_window(self):
        passes = []

        def fake(media, by_pauses=False):
            passes.append(by_pauses)
            return SCRIPT if by_pauses else "हनुमान ने सुरसा के मुँह में"

        with patch.object(gate, "transcribe", side_effect=fake), \
                patch.object(gate, "probe_duration", return_value=50.0), \
                patch.object(gate, "loudness", return_value=LOUD_OK):
            result = gate.run(Path("x.mp4"), SCRIPT, True)
        self.assertEqual(passes, [False, True])
        self.assertTrue(result["passed"])
        self.assertEqual(result["heard"], SCRIPT)

    def test_no_second_pass_when_the_first_covers_the_script(self):
        with patch.object(gate, "transcribe", return_value=SCRIPT) as transcribe, \
                patch.object(gate, "probe_duration", return_value=50.0), \
                patch.object(gate, "loudness", return_value=LOUD_OK):
            gate.run(Path("x.mp4"), SCRIPT, True)
        self.assertEqual(transcribe.call_count, 1)

    def test_both_passes_short_still_fails(self):
        with patch.object(gate, "transcribe", return_value="हनुमान ने सुरसा के मुँह में"), \
                patch.object(gate, "probe_duration", return_value=50.0), \
                patch.object(gate, "loudness", return_value=LOUD_OK):
            result = gate.run(Path("x.mp4"), SCRIPT, True)
        self.assertFalse(result["checks"]["speech_coverage"]["ok"])

    def test_parse_ebur128_summary(self):
        log = ("[Parsed_ebur128_0] t: 1 M: -20 S: -20 I: -30.0 LUFS\n"
               "[Parsed_ebur128_0] Summary:\n\n  Integrated loudness:\n    I:         -14.3 LUFS\n"
               "    Threshold: -24.5 LUFS\n\n  True peak:\n    Peak:       -1.6 dBFS\n")
        self.assertEqual(gate.parse_ebur128(log), {"integrated_lufs": -14.3, "true_peak_dbfs": -1.6})


if __name__ == "__main__":
    unittest.main()
