import unittest

from trusted_qa.assemble import AUDIO_CHECKS, CLAIM_CHECKS, CRITICAL_FLAGS, _without
from trusted_qa.reviewer import _response_schema


class ReviewSchemaTests(unittest.TestCase):
    def test_schema_fixes_the_layout_the_assembler_reads(self):
        packet = {"full_final_audio_asr": [{"file": "agent-asr-gemini.json"}, {"file": "agent-asr-whisper.json"}],
                  "script_beats": [{}, {}, {}], "qc": {"speech_differences": ["a"], "warnings": ["w", "w"]}}
        schema = _response_schema(packet)
        audio = schema["properties"]["audio_review"]
        beats = audio["properties"]["beat_reconciliation"]
        self.assertEqual((beats["minItems"], beats["maxItems"]), (3, 3))
        beat = beats["items"]
        self.assertEqual(set(beat["required"]),
                         {"beat", "decision", "notes", "unresolved_items", "asr_evidence"} | CRITICAL_FLAGS)
        citation = beat["properties"]["asr_evidence"]["items"]
        self.assertEqual(set(citation["required"]), {"asr_file", "segment_indices", "asr_excerpt"})
        self.assertEqual(schema["properties"]["qc_warning_dispositions"]["minItems"], 2)
        self.assertNotIn("checked_narration_transform", audio["required"])
        self.assertTrue(AUDIO_CHECKS <= set(audio["required"]))
        claim = schema["properties"]["claim_findings"]["items"]
        self.assertTrue(CLAIM_CHECKS <= set(claim["required"]))
        self.assertEqual(claim["properties"]["decision"]["enum"], ["approved", "hold"])

    def test_inapplicable_optional_fields_are_dropped(self):
        self.assertEqual(_without({"id": "x", "license_excerpt": "n/a"}, {"license_excerpt"}), {"id": "x"})
        self.assertEqual(_without("not an object", {"x"}), "not an object")
