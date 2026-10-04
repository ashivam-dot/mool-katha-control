"""Synthetic fixture checks the strict review contract without a release key."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from trusted_qa.assemble import assemble_approved_review, save_unsigned_review
from trusted_qa.asr import _result, save_asr_results
from trusted_qa.audio_quality import EPISODE_OBSERVATION_FILE, AudioQualityObservation
from trusted_qa.candidate import load_candidate
from trusted_qa.common import (QaHold, digest_bytes, digest_file, utc_now,
                               write_bytes_new, write_json_new)
from trusted_qa.fetch import FetchObservation
from trusted_qa.frame_audit import AUDIT_POLICY, HEIGHT, WIDTH, FrameBatchEvidence
from trusted_qa.media import VisualEvidence
from trusted_qa.observations import ObservationSet, collect_observations
from trusted_qa.reviewer import ReviewModelResult
from trusted_qa.sample_plan import SAMPLE_POLICY, canonical_sha256, plan_sampled_frames
from trusted_qa.tests.test_candidate import EPISODE, PRODUCER_ID, QA_ID, candidate_fixture


NOTE = "I checked the exact cited evidence and found the scoped Hindi claim supported."
FLAGS = {"sacred_name_disagreement": False, "source_reference_disagreement": False,
         "offensive_reading": False, "meaning_changing_disagreement": False}


def _observation(candidate, source: dict) -> FetchObservation:
    text = (f"Printed label {source['printed_verse_label']}. {source['excerpt']} "
            "The edition page contains this whole quoted passage with source context.")
    raw_response = text.encode("utf-8")
    response_digest = digest_bytes(raw_response)
    response_ref = f"agent-qa-responses/{response_digest}.bin"
    response_target = candidate.episode_dir / response_ref
    if not response_target.exists():
        write_bytes_new(response_target, raw_response)
    encoded = (text + "\n").encode("utf-8")
    digest = digest_bytes(encoded)
    ref = f"agent-qa-snapshots/{digest}.txt"
    target = candidate.episode_dir / ref
    if not target.exists():
        write_bytes_new(target, encoded)
    return FetchObservation(source["url"], utc_now(), 200, response_digest,
                            response_ref, ref, digest, text, source["url"], "text/plain")


def build_synthetic_approved_review(root: Path, *,
                                    tamper_source_snapshot: bool = False,
                                    tamper_source_response: bool = False,
                                    tamper_quality_observation: bool = False,
                                    tamper_frame_sheet: bool = False,
                                    duplicate_source: str | None = None) -> tuple[Path, dict, object]:
    """Build test-only provider/model responses; never use this for an actual episode."""
    repo, archive, commit = candidate_fixture(root)
    candidate = load_candidate(repo, archive, root / "snapshot", EPISODE, commit, QA_ID)
    claim = candidate.ledger["claims"][0]
    pages = {key: _observation(candidate, claim[key]) for key in ("primary", "corroboration")}
    if duplicate_source == "redirect":
        pages["corroboration"] = replace(pages["corroboration"],
                                         final_url=pages["primary"].final_url)
    elif duplicate_source == "visible_text":
        pages["corroboration"] = replace(pages["corroboration"],
                                         snapshot_sha256=pages["primary"].snapshot_sha256)
    rights = {identity: {"origin": asset["origin"], "checked_at": utc_now(),
                         "content_sha256": asset["sha256"],
                         "provenance_excerpt": asset["rights_basis"]}
              for identity, asset in candidate.assets.items()}
    observations = ObservationSet({claim["id"]: pages}, rights, {})
    utterance = "राम नल से कहते हैं।"
    raw_transcript = {"segments": [{"start_seconds": 0.2, "end_seconds": 42.0,
                                    "text": utterance}], "full_transcript": utterance}
    gemini_raw = {"responseId": "synthetic-gemini-run", "modelVersion": "gemini-test-version",
                  "candidates": [{"finishReason": "STOP", "content": {"parts": [
                      {"text": json.dumps(raw_transcript, ensure_ascii=False)}]}}]}
    segments = [{"index": 1, "start_seconds": 0.2, "end_seconds": 42.0, "text": utterance}]
    gemini = _result("Google Gemini API", "Google Gemini multimodal audio", "gemini-test-model",
                     "gemini-test-version", "synthetic-gemini-run", candidate.hashes["video"],
                     43.0, segments, utterance, gemini_raw, utc_now())
    whisper_raw = {"segments": [{"id": 0, "start": 0.2, "end": 42.0, "text": utterance}],
                   "info": {"language": "hi", "duration": 43.0},
                   "settings": {"vad_filter": False}}
    whisper = _result("local faster-whisper", "OpenAI Whisper", "synthetic-whisper-model",
                      "synthetic-whisper-version", "synthetic-whisper-run",
                      candidate.hashes["video"], 43.0, segments, utterance,
                      whisper_raw, utc_now())
    refs = save_asr_results(candidate.episode_dir, gemini, whisper)
    asr = {"agent-asr-gemini.json": gemini, "agent-asr-whisper.json": whisper}
    contact = candidate.episode_dir / "agent-video-contact.jpg"
    contact.write_bytes(b"synthetic contact sheet, not real decoded frames")
    visual = VisualEvidence("agent-video-contact.jpg", digest_file(contact),
                            [0.1, 8.0, 16.0, 20.0, 24.0, 32.0, 40.0, 42.9], {1: 20.0}, [],
                            {"decode_tool": "synthetic ffmpeg fixture", "decode_exit_code": 0,
                             "decoded_frame_count": 1290, "decoded_duration_seconds": 43.0,
                             "decoded_full_video": True})
    sheet_bytes = b"synthetic indexed frame sheet, not actual decoded frames"
    sheet_hash = digest_bytes(sheet_bytes)
    sheet = candidate.episode_dir / f"agent-video-frames/{sheet_hash}.jpg"
    write_bytes_new(sheet, sheet_bytes)
    frame_records = [{"index": index, "seconds": round((index - 1) / 30, 6),
                      "rgb_sha256": f"{index:064x}", "mean_luma": 50.0,
                      "stddev_luma": 20.0, "dark_fraction": 0.1,
                      "bright_fraction": 0.0,
                      "delta_previous": 0.5 if index > 1 else 0.0,
                      "rgb_delta_previous": 0.5 if index > 1 else 0.0,
                      "rgb_delta_two_back": 0.5 if index > 2 else 0.0}
                     for index in range(1, 1291)]
    plan = plan_sampled_frames(frame_records, candidate.manifest["beats"],
                               candidate.check["duration"])
    indices = plan["sampled_indices"]
    batch_identity = {"file": f"agent-video-frames/{sheet_hash}.jpg", "sha256": sheet_hash,
                      "indices": indices, "start_index": indices[0], "end_index": indices[-1],
                      "first_seconds": frame_records[indices[0] - 1]["seconds"],
                      "last_seconds": frame_records[indices[-1] - 1]["seconds"]}
    audit = candidate.episode_dir / "agent-video-frame-audit.json"
    write_json_new(audit, {"kind": "all_frame_pixel_temporal_audit_v2",
                           "input_video_sha256": candidate.hashes["video"],
                           "decoded_frame_count": 1290, "scaled_width": WIDTH,
                           "scaled_height": HEIGHT, "qc_duration_seconds": 43.0,
                           "audit_policy": AUDIT_POLICY,
                           "audit_policy_sha256": canonical_sha256(AUDIT_POLICY),
                           "frames": frame_records,
                           "frame_batches": [batch_identity],
                           "anomalies": plan["anomalies"],
                           "sample_plan_sha256": canonical_sha256(plan)})
    frame_call = {"provider": "Google Gemini API", "model": "gemini-test-model",
                  "model_version": "test-version", "request_id": "synthetic-frame-run"}
    batch_review = {**batch_identity, "decision": "clear", "uncertainty": "low",
                    "checked_indices": indices, "defect_indices": [], "notes": NOTE,
                    "model_call": frame_call, "request_sha256": "b" * 64,
                    "response_sha256": "c" * 64}
    coverage = {"mode": "sampled", "selected_frames": len(indices),
                "decoded_frames": 1290, "unsampled_frames": 1290 - len(indices),
                "fraction": round(len(indices) / 1290, 6), "sheets": 1,
                "selection_rule": SAMPLE_POLICY["kind"],
                "selection_rule_sha256": plan["selection_rule_sha256"],
                "selection_sha256": canonical_sha256(plan),
                "max_gap_seconds": plan["max_gap_seconds"],
                "reason": "Capacity-bounded model sample; unselected frames received deterministic pixel and temporal checks only.",
                "provider_errors": []}
    frames = FrameBatchEvidence(
        {"kind": "frame_sampled_visual_review_v1", "input_video_sha256": candidate.hashes["video"],
         "all_frame_audit_file": audit.name, "all_frame_audit_sha256": digest_file(audit),
         "decoded_frame_count": 1290, "sample_plan": plan,
         "sample_plan_sha256": canonical_sha256(plan),
         "model_visual_coverage": coverage, "batches": [batch_review]},
        [frame_call], audit, [sheet])
    citations = [{"asr_file": name, "segment_indices": [1], "asr_excerpt": utterance} for name in asr]
    decision = {
        "claim_findings": [{"id": claim["id"], "decision": "approved", "notes": NOTE,
                            "unresolved_items": [], "primary_excerpt": claim["primary"]["excerpt"],
                            "corroboration_excerpt": claim["corroboration"]["excerpt"],
                            "checked_primary_page": True, "checked_corroboration_page": True,
                            "checked_source_independence": True, "checked_hindi_entailment": True,
                            "checked_variant_scope": True}],
        "asset_findings": [{"id": identity, "decision": "approved", "notes": NOTE,
                            "unresolved_items": [], "checked_exact_asset_bytes": True,
                            "checked_origin_and_rights_evidence": True, "checked_license_terms": True,
                            "checked_commercial_use": True, "checked_derivatives": True,
                            "checked_credit": True} for identity in candidate.assets],
        "audio_review": {"decision": "approved", "notes": NOTE, "unresolved_items": [],
                         "checked_full_asr_coverage": True, "checked_entire_spoken_script": True,
                         "checked_names_and_source_refs": True,
                         "beat_reconciliation": [{"beat": 1, "decision": "approved", "notes": NOTE,
                                                  "unresolved_items": [],
                                                  "asr_evidence": citations, **FLAGS}],
                         "speech_difference_dispositions": []},
        "video_review": {"decision": "approved", "notes": NOTE, "unresolved_items": [],
                         "inspected_sampled_frames": True, "checked_hindi_captions": True,
                         "checked_first_frame_source": True, "checked_artwork_labels": True,
                         "checked_visual_integrity": True, "checked_timeline_alignment": True,
                         "critical_defects": []},
        "qc_warning_dispositions": [],
        "release_review": {"decision": "approved", "notes": NOTE, "unresolved_items": [],
                           "verified_all_claims": True, "verified_all_assets": True,
                           "verified_asr_and_decoded_frames": True,
                           "verified_no_unresolved_concerns": True},
    }
    model = ReviewModelResult(decision, {"provider": "Google Gemini API",
                                         "model": "gemini-test-model", "model_version": "test-version",
                                         "request_id": "synthetic-review-run"})
    quality_dir = root / "synthetic-quality"
    quality_request = quality_dir / "request.json"
    quality_response = quality_dir / "response.json"
    quality_record = quality_dir / "observation.json"
    write_bytes_new(quality_request, b"synthetic audio-quality request, not actual listening")
    write_bytes_new(quality_response, b"synthetic audio-quality response, not a provider response")
    quality_data = {"kind": "full_final_audio_quality_model_observation_v1",
                    "basis": "Synthetic test fixture, not human or actual model listening",
                    "input_video_sha256": candidate.hashes["video"],
                    "input_audio_sha256": digest_bytes(b"synthetic audio bytes"),
                    "audio_duration_seconds": 43.0,
                    "request_sha256": digest_file(quality_request),
                    "response_sha256": digest_file(quality_response),
                    "observed_at": utc_now(),
                    "model_call": {"provider": "Google Gemini API", "model": "gemini-test-model",
                                   "model_version": "test-version", "request_id": "synthetic-quality-run"},
                    "decision": {"decision": "clear", "uncertainty": "low",
                                 "summary": "Synthetic fixture voice-quality conclusion for schema testing only.",
                                 "uncertainty_notes": "Synthetic test fixture; no actual audio was heard by a model.",
                                 "observations": [
                                     {"aspect": aspect, "status": "clear", "start_seconds": 0.2,
                                      "end_seconds": 42.0,
                                      "detail": "Synthetic fixture observation, not actual audio evidence."}
                                     for aspect in ("natural_indian_hindi", "clarity", "cadence",
                                                    "sacred_names", "artifacts")]}}
    write_json_new(quality_record, quality_data)
    episode_quality_record = candidate.episode_dir / EPISODE_OBSERVATION_FILE
    write_json_new(episode_quality_record, quality_data)
    quality = AudioQualityObservation(quality_data, quality_record, episode_quality_record,
                                      digest_file(quality_record), quality_request, quality_response)
    run = {"system": "github_actions", "repository": "ashivam-dot/mool-katha-control",
           "workflow_ref": "ashivam-dot/mool-katha-control/.github/workflows/qa.yml@refs/heads/main",
           "workflow_sha": "f" * 40, "run_id": 123, "run_attempt": 1}
    if tamper_source_snapshot:
        (candidate.episode_dir / pages["primary"].snapshot_ref).write_text(
            "tampered before assembly", encoding="utf-8")
    if tamper_source_response:
        (candidate.episode_dir / pages["primary"].response_ref).write_bytes(b"tampered raw response")
    if tamper_quality_observation:
        episode_quality_record.write_bytes(b"tampered model observation")
    if tamper_frame_sheet:
        sheet.write_bytes(b"tampered indexed frame sheet")
    review = assemble_approved_review(candidate, observations, asr, refs, visual, frames, model, quality, run)
    path = save_unsigned_review(candidate, review)
    return path, review, candidate


class AssemblyTests(unittest.TestCase):
    def test_collection_holds_when_distinct_urls_resolve_to_one_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, archive, commit = candidate_fixture(root)
            candidate = load_candidate(repo, archive, root / "snapshot", EPISODE, commit, QA_ID)
            claim = candidate.ledger["claims"][0]
            primary = _observation(candidate, claim["primary"])
            corroboration = _observation(candidate, claim["corroboration"])
            redirected = replace(corroboration, final_url=primary.final_url)
            pages = {primary.url: primary, corroboration.url: redirected}
            with patch("trusted_qa.observations.fetch_observation",
                       side_effect=lambda url, *_args: pages[url]):
                with self.assertRaisesRegex(QaHold, "same final page"):
                    collect_observations(candidate, root / "private")

    def test_collection_holds_on_identical_visible_source_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, archive, commit = candidate_fixture(root)
            candidate = load_candidate(repo, archive, root / "snapshot", EPISODE, commit, QA_ID)
            claim = candidate.ledger["claims"][0]
            same_page = (f"15.8 {claim['primary']['excerpt']} "
                         f"{claim['corroboration']['excerpt']} "
                         "One copied passage is served under two unrelated source URLs.").encode()
            with patch("trusted_qa.fetch._request_once",
                       return_value=(200, {"content-type": "text/plain"}, same_page)):
                with self.assertRaisesRegex(QaHold, "identical visible text"):
                    collect_observations(candidate, root / "private")

    def test_assembly_rechecks_independent_source_pages(self) -> None:
        for mode, message in (("redirect", "same final page"),
                              ("visible_text", "identical visible text")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(QaHold, message):
                    build_synthetic_approved_review(Path(directory), duplicate_source=mode)

    def test_complete_synthetic_contract_and_snapshot_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, review, candidate = build_synthetic_approved_review(Path(directory))
            self.assertEqual(review["kind"], "agent_episode_qa_v1")
            self.assertEqual(review["qa_agent_id"], QA_ID)
            self.assertEqual(len(review["qa_run"]["review_model_calls"]), 3)
            self.assertEqual(review["qa_run"]["frame_batch_review"]["decoded_frame_count"], 1290)
            self.assertEqual(set(review["claim_findings"][0]), {
                "agent_id", "method", "reviewed_at", "decision", "unresolved_items", "notes", "id",
                "primary_url", "corroboration_url", "primary_printed_label",
                "corroboration_printed_label", "checked_primary_page", "checked_corroboration_page",
                "checked_source_independence", "checked_hindi_entailment", "checked_variant_scope",
                "primary_fetch", "corroboration_fetch"})
            self.assertEqual(json.loads(path.read_text())["video_sha256"], candidate.hashes["video"])
            self.assertEqual(review["audio_review"]["beat_reconciliation"][0]["critical_terms"],
                             ["राम", "नल", "कहते"])
            quality_ref = review["audio_review"]["quality_observation"]
            self.assertEqual(quality_ref["file"], EPISODE_OBSERVATION_FILE)
            self.assertEqual(digest_file(candidate.episode_dir / quality_ref["file"]),
                             quality_ref["sha256"])
            snapshot = candidate.episode_dir / review["claim_findings"][0]["primary_fetch"]["snapshot_ref"]
            snapshot.write_text("tampered", encoding="utf-8")
            self.assertNotEqual(digest_file(snapshot), review["claim_findings"][0]["primary_fetch"]["snapshot_sha256"])

    def test_snapshot_tamper_before_assembly_holds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(QaHold, "fetched source snapshot changed"):
                build_synthetic_approved_review(Path(directory), tamper_source_snapshot=True)

    def test_response_tamper_before_assembly_holds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(QaHold, "fetched source response changed"):
                build_synthetic_approved_review(Path(directory), tamper_source_response=True)

    def test_quality_observation_tamper_before_assembly_holds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(QaHold, "quality observation changed"):
                    build_synthetic_approved_review(Path(directory), tamper_quality_observation=True)

    def test_frame_sheet_tamper_before_assembly_holds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(QaHold, "indexed frame sheet changed"):
                build_synthetic_approved_review(Path(directory), tamper_frame_sheet=True)


if __name__ == "__main__":
    unittest.main()
