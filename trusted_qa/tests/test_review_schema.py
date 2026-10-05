import unittest

from trusted_qa.assemble import _hindi_words, reviewed_warnings, AUDIO_CHECKS, CLAIM_CHECKS, CRITICAL_FLAGS, _asr_item, _listed, _semantic, _without
from trusted_qa.reviewer import _complete_dispositions, _ordered, _response_schema


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
        # Gemini rejects required keyed objects with HTTP 400; it serves these arrays.
        warnings = schema["properties"]["qc_warning_dispositions"]
        self.assertEqual(warnings["type"], "array")
        self.assertIn("reason", warnings["items"]["required"])
        self.assertNotIn("$defs", schema)

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

    def test_answers_are_put_in_expected_order_with_repeats(self):
        given = [{"warning": "b", "n": 1}, {"warning": "a", "n": 2}, {"warning": "a", "n": 3}]
        ordered, missing = _ordered(["a", "b", "a", "c"], given, "warning")
        self.assertEqual([item["n"] for item in ordered], [2, 1, 3])
        self.assertEqual(missing, ["c"])

    def test_left_out_items_are_requested_once_and_merged_in_order(self):
        import json
        import tempfile
        from pathlib import Path

        packet = {"qc": {"speech_differences": ["d1", "d2", "d3"], "warnings": ["w1"]}}
        decision = {"audio_review": {"speech_difference_dispositions": [{"difference": "d2", "k": 2}]},
                    "qc_warning_dispositions": [{"warning": "w1"}]}
        followup = {"speech_difference_dispositions": [{"difference": "d3", "k": 3},
                                                          {"difference": "d1", "k": 1}],
                    "qc_warning_dispositions": []}
        calls = []

        def post(generation, name):
            calls.append((name, generation))
            body = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(followup)}]}}],
                    "responseId": "resp-1", "modelVersion": "model-1"}
            return json.dumps(body).encode(), "m"

        with tempfile.TemporaryDirectory() as directory:
            parts = [{"text": "prompt"}]
            done = _complete_dispositions(decision, packet, parts, {"temperature": 0}, post, Path(directory))
            self.assertTrue((Path(directory) / "review-model-followup-response.json").is_file())
        self.assertEqual([name for name, _ in calls], ["review-model-followup-request.json"])
        self.assertIn('"d1"', parts[-1]["text"])
        self.assertNotIn('"d2"', parts[-1]["text"])
        self.assertEqual([item["k"] for item in done["audio_review"]["speech_difference_dispositions"]], [1, 2, 3])

    def test_complete_answer_needs_no_follow_up(self):
        packet = {"qc": {"speech_differences": ["d1"], "warnings": []}}
        decision = {"audio_review": {"speech_difference_dispositions": [{"difference": "d1"}]},
                    "qc_warning_dispositions": []}
        def post(*_args):
            raise AssertionError("no follow-up expected")
        self.assertEqual(_complete_dispositions(decision, packet, [], {}, post, None), decision)

    def test_speech_warning_repeating_a_difference_is_not_asked_twice(self):
        difference = "'चण्डिका' heard as 'चंडिका' after 'डराया'"
        warning = f"speech: {difference} (fix the narration, or a Hindi reviewer approves it in speech-approvals.json)"
        other = "speech: 'x' heard as 'y' after 'z' (fix the narration, or a Hindi reviewer approves it in speech-approvals.json)"
        check = {"speech_differences": [difference], "warnings": [warning, "loudness: -13 LUFS", other]}
        self.assertEqual(reviewed_warnings(check), ["loudness: -13 LUFS", other])

    def test_nasal_consonant_spelling_matches_anusvara(self):
        self.assertEqual(_hindi_words("चण्डिका"), _hindi_words("चंडिका"))
