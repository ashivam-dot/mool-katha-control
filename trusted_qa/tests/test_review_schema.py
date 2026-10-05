import unittest

from trusted_qa.assemble import AUDIO_CHECKS, CLAIM_CHECKS, CRITICAL_FLAGS, _asr_item, _listed, _semantic, _without
from trusted_qa.reviewer import _response_schema


class ReviewSchemaTests(unittest.TestCase):
    def test_schema_fixes_the_layout_the_assembler_reads(self):
        packet = {"full_final_audio_asr": [{"file": "agent-asr-gemini.json"}, {"file": "agent-asr-whisper.json"}],
                  "script_beats": [{}, {}, {}], "qc": {"speech_differences": ["a"], "warnings": ["w", "w"]}}
        schema = _response_schema(packet)
        audio = schema["properties"]["audio_review"]
        beats = audio["properties"]["beat_reconciliation"]
        beat = beats["items"]
        self.assertEqual(set(beat["required"]),
                         {"beat", "decision", "notes", "unresolved_items", "asr_evidence"} | CRITICAL_FLAGS)
        citation = beat["properties"]["asr_evidence"]["items"]
        self.assertEqual(set(citation["required"]), {"asr_file", "segment_indices", "asr_excerpt"})
        self.assertNotIn("checked_narration_transform", audio["required"])
        self.assertTrue(AUDIO_CHECKS <= set(audio["required"]))
        claim = schema["properties"]["claim_findings"]["items"]
        self.assertTrue(CLAIM_CHECKS <= set(claim["required"]))
        self.assertEqual(claim["properties"]["decision"]["enum"], ["approved", "hold"])
        warnings = schema["properties"]["qc_warning_dispositions"]
        self.assertEqual(warnings["required"], ["warning_01", "warning_02"])
        self.assertEqual(audio["properties"]["speech_difference_dispositions"]["required"], ["difference_01"])
        self.assertEqual(warnings["properties"]["warning_01"], {"$ref": "#/$defs/warning"})
        self.assertIn("reason", schema["$defs"]["warning"]["required"])

    def test_inapplicable_optional_fields_are_dropped(self):
        self.assertEqual(_without({"id": "x", "license_excerpt": "n/a"}, {"license_excerpt"}), {"id": "x"})
        self.assertEqual(_without("not an object", {"x"}), "not an object")

    def test_unconstrained_citation_layout_is_read_without_weakening_flags(self):
        flags = {name: False for name in CRITICAL_FLAGS}
        beat = {"beat_number": 1, "decision": "approved", "notes": "n", "unresolved_items": [],
                "asr_evidence_1": {"asr_file": "a.json", "segment_index": 3, "verbatim_asr_excerpt": "x", **flags},
                "asr_evidence_2": {"asr_file": "b.json", "segment_indices": [4], "verbatim_excerpt": "y", **flags}}
        item = _asr_item(beat)
        self.assertEqual(item["beat"], 1)
        self.assertEqual(item["asr_evidence"], [
            {"asr_file": "a.json", "segment_indices": [3], "asr_excerpt": "x"},
            {"asr_file": "b.json", "segment_indices": [4], "asr_excerpt": "y"}])
        self.assertTrue(all(item[name] is False for name in CRITICAL_FLAGS))
        beat["asr_evidence_2"][next(iter(CRITICAL_FLAGS))] = True
        self.assertTrue(any(_asr_item(beat)[name] for name in CRITICAL_FLAGS))

    def test_short_reason_label_is_joined_with_its_notes(self):
        item = _semantic({"difference": "d", "reason": "Phonetic variation.", "decision": "accepted",
                          "notes": "The ASR wrote Kashi as Kasi; the narration is unchanged.", "unresolved_items": []},
                         {"difference", "reason"}, "difference 1", decision="accepted")
        self.assertTrue(item["reason"].startswith("Phonetic variation: The ASR wrote"))
        self.assertGreaterEqual(len(item["reason"]), 30)

    def test_keyed_dispositions_cannot_be_skipped_and_read_back_in_order(self):
        keyed = {"difference_02": "b", "difference_01": "a", "difference_03": "c"}
        self.assertEqual(_listed(keyed, "difference"), ["a", "b", "c"])
        self.assertEqual(_listed({"difference_01": "a", "difference_03": "c"}, "difference"),
                         {"difference_01": "a", "difference_03": "c"})
        self.assertEqual(_listed(["a"], "difference"), ["a"])
