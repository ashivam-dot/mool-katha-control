"""Fail-closed construction of the exact unsigned agent_episode_qa_v1 review."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from .asr import _asr_segments
from .audio_quality import EPISODE_OBSERVATION_FILE, AudioQualityObservation
from .candidate import Candidate
from .common import (QA_REPOSITORY, QaHold, digest_file, gemini_text_response,
                     json_object, require, timestamp, utc_now, valid_qa_workflow_ref,
                     write_json_new)
from .fetch import FetchObservation
from .media import VisualEvidence
from .observations import ObservationSet
from .reviewer import ReviewModelResult
from .terms import contains_adjacent_words, required_beat_terms


BASE = {"decision", "notes", "unresolved_items"}
CRITICAL_FLAGS = {"sacred_name_disagreement", "source_reference_disagreement",
                  "offensive_reading", "meaning_changing_disagreement"}
CLAIM_CHECKS = {"checked_primary_page", "checked_corroboration_page",
                "checked_source_independence", "checked_hindi_entailment", "checked_variant_scope"}
ASSET_CHECKS = {"checked_exact_asset_bytes", "checked_origin_and_rights_evidence",
                "checked_license_terms", "checked_commercial_use", "checked_derivatives", "checked_credit"}
AUDIO_CHECKS = {"checked_full_asr_coverage", "checked_entire_spoken_script",
                "checked_names_and_source_refs"}
VIDEO_CHECKS = {"inspected_sampled_frames", "checked_hindi_captions", "checked_first_frame_source",
                "checked_artwork_labels", "checked_visual_integrity", "checked_timeline_alignment"}
RELEASE_CHECKS = {"verified_all_claims", "verified_all_assets",
                  "verified_asr_and_decoded_frames", "verified_no_unresolved_concerns"}
INCOMPATIBLE = re.compile(r"\b(?:NC|ND)\b|non.?commercial|no.?derivatives|editorial.only|all.rights.reserved|unknown|unspecified", re.I)


def _exact(value: Any, keys: set[str], label: str) -> dict:
    require(isinstance(value, dict) and set(value) == keys,
            f"{label}: independent reviewer returned missing or unknown fields")
    return value


def _semantic(value: Any, keys: set[str], label: str, *, decision: str = "approved") -> dict:
    item = _exact(value, BASE | keys, label)
    require(item["decision"] == decision and item["unresolved_items"] == [] and
            isinstance(item["notes"], str) and len(item["notes"].strip()) >= 30,
            f"{label}: unresolved or unreasoned review must hold")
    return item


def _checks(value: dict, names: set[str], label: str) -> None:
    for name in names:
        require(value.get(name) is True, f"{label}: {name} was not independently completed")


def _no_critical(value: dict, label: str) -> None:
    for name in CRITICAL_FLAGS:
        require(value.get(name) is False, f"{label}: {name} requires a new take or hold")


def _verdict(qa_id: str, method: str, reviewed_at: str, item: dict) -> dict:
    return {"agent_id": qa_id, "method": method, "reviewed_at": reviewed_at,
            "decision": item["decision"], "unresolved_items": [], "notes": item["notes"]}


_VARIANTS = str.maketrans({"\u093c": None, "\u0901": "\u0902", "\u094d": None,
                          "\u0940": "\u093f", "\u0942": "\u0941", "\u0908": "\u0907", "\u090a": "\u0909"})
_YE = re.compile("\u092f\u0947$")


def _hindi_words(text: str) -> list[str]:
    words = ["".join(character for character in unicodedata.normalize("NFC", part).lower()
                     if unicodedata.category(character)[0] in "LMN") for part in text.split()]
    return [_YE.sub("\u090f", word.translate(_VARIANTS)) for word in words if word]


def _phrase_in(phrase: str, excerpt: str) -> bool:
    expected = _hindi_words(phrase)
    heard = _hindi_words(excerpt)
    return bool(expected) and any(heard[index:index + len(expected)] == expected
                                  for index in range(len(heard) - len(expected) + 1))


def _citations(value: Any, asr: dict[str, dict], label: str,
               *, critical_terms: list[str] | None = None, expected_phrase: str | None = None,
               beat_span: tuple[float, float] | None = None) -> list[dict]:
    require(isinstance(value, list) and len(value) == len(asr),
            f"{label}: cite both actual full-final-audio recognizers")
    citations: list[dict] = []
    seen: set[str] = set()
    for raw in value:
        item = _exact(raw, {"asr_file", "segment_indices", "asr_excerpt"}, label)
        name = item["asr_file"]
        require(isinstance(name, str) and name in asr and name not in seen,
                f"{label}: unknown or duplicate ASR file")
        seen.add(name)
        indices = item["segment_indices"]
        segments = asr[name]["segments"]
        require(isinstance(indices, list) and bool(indices) and
                all(isinstance(index, int) and not isinstance(index, bool) and 1 <= index <= len(segments)
                    for index in indices), f"{label}: ASR segment indices are invalid")
        require(indices == list(range(indices[0], indices[-1] + 1)),
                f"{label}: ASR segment indices must be ascending and contiguous")
        if beat_span is not None:
            start, end = beat_span
            require(all(isinstance(number, (int, float)) and not isinstance(number, bool) and
                        math.isfinite(number) for number in beat_span) and 0 <= start < end,
                    f"{label}: frozen beat span is invalid")
            require(all(segments[index - 1]["start_seconds"] < end and
                        segments[index - 1]["end_seconds"] > start for index in indices),
                    f"{label}: cited ASR segments do not overlap the frozen beat")
        excerpt = item["asr_excerpt"]
        joined = " ".join(segments[index - 1]["text"] for index in indices)
        require(isinstance(excerpt, str) and len(excerpt.strip()) >= 2 and excerpt in joined,
                f"{label}: ASR excerpt was not returned by that recognizer")
        if critical_terms:
            require(all(contains_adjacent_words(term, excerpt) for term in critical_terms),
                    f"{label}: cited recognizer omitted a signed critical term")
        if expected_phrase:
            require(_phrase_in(expected_phrase, excerpt),
                    f"{label}: cited recognizer omitted exact QC expected phrase")
        citations.append({"asr_file": name, "segment_indices": indices,
                          "asr_excerpt": excerpt})
    require(seen == set(asr), f"{label}: both ASR files are required")
    return citations


def _quoted_expected(difference: str) -> str | None:
    match = re.search(r"'([^']+)' heard as ", difference)
    return match[1] if match and match[1] != "(nothing)" else None


def _indexed(value: Any, expected: set[str], label: str) -> dict[str, dict]:
    require(isinstance(value, list) and len(value) == len(expected),
            f"{label}: one finding is required for every exact ID")
    found: dict[str, dict] = {}
    for item in value:
        require(isinstance(item, dict) and isinstance(item.get("id"), str),
                f"{label}: finding lacks an ID")
        require(item["id"] in expected and item["id"] not in found,
                f"{label}: finding has an unexpected or duplicate ID")
        found[item["id"]] = item
    return found


def _verify_raw_asr(result: dict, label: str) -> None:
    require(isinstance(result, dict), f"{label}: ASR result is not an object")
    raw = result.get("raw_response")
    require(isinstance(raw, dict) and bool(raw), f"{label}: raw recognizer output is missing")
    duration = result.get("audio_duration_seconds")
    require(isinstance(duration, (int, float)) and not isinstance(duration, bool) and
            math.isfinite(duration) and duration > 0, f"{label}: audio duration is invalid")
    if label == "gemini":
        response_id, version, transcript_text = gemini_text_response(raw, "Gemini ASR raw response")
        require(response_id == result.get("run_id") and version == result.get("model_version"),
                "Gemini ASR wrapper differs from provider response")
        raw_transcript = json_object(transcript_text, "Gemini raw transcript")
        segments, joined = _asr_segments(raw_transcript.get("segments"), duration, "Gemini")
        require(raw_transcript.get("full_transcript") == joined,
                "Gemini raw full transcript differs from its segments")
    else:
        settings = raw.get("settings")
        info = raw.get("info")
        reported_duration = info.get("duration") if isinstance(info, dict) else None
        require(isinstance(settings, dict) and settings.get("vad_filter") is False and
                isinstance(info, dict) and info.get("language") == "hi" and
                isinstance(reported_duration, (int, float)) and
                not isinstance(reported_duration, bool) and math.isfinite(reported_duration) and
                abs(reported_duration - duration) <= 0.5,
                "Whisper did not process the whole Hindi audio")
        segments, joined = _asr_segments(raw.get("segments"), duration, "Whisper")
    require(result.get("segments") == segments and result.get("full_transcript") == joined,
            f"{label}: normalized ASR was edited after recognition")


def _qa_run(candidate: Candidate, run: dict[str, Any], model_calls: list[dict[str, str]],
            completed_at: str) -> dict:
    required = {"system", "repository", "workflow_ref", "workflow_sha", "run_id", "run_attempt"}
    require(set(run) == required and run["system"] == "github_actions" and
            run["repository"] == QA_REPOSITORY and
            valid_qa_workflow_ref(run["repository"], run["workflow_ref"]) and
            isinstance(run["workflow_sha"], str) and re.fullmatch(r"[0-9a-f]{40}", run["workflow_sha"]) and
            all(isinstance(run[key], int) and not isinstance(run[key], bool) and run[key] > 0
                for key in ("run_id", "run_attempt")), "trusted QA run provenance is invalid")
    expected_id = f"agent:github_actions/{run['repository']}/{run['run_id']}/{run['run_attempt']}"
    require(candidate.qa_agent_id == expected_id and candidate.producer_agent_id != expected_id,
            "QA identity differs from the cloud run or producer")
    require(isinstance(model_calls, list) and len(model_calls) == 2 and
            all(isinstance(call, dict) and
                set(call) == {"provider", "model", "model_version", "request_id"} and
                all(isinstance(value, str) and len(value.strip()) >= 4 for value in call.values())
                for call in model_calls) and
            len({call["request_id"] for call in model_calls}) == len(model_calls),
            "independent model call provenance is incomplete or repeated")
    timestamp(completed_at, "QA completion")
    return {**run, "completed_at": completed_at, "review_model_calls": model_calls}


def assemble_approved_review(candidate: Candidate, observations: ObservationSet,
                             asr_results: dict[str, dict], asr_refs: list[dict],
                             visual: VisualEvidence, model: ReviewModelResult,
                             quality: AudioQualityObservation,
                             run: dict[str, Any]) -> dict[str, Any]:
    """Reject any partial/model-authored provenance before writing a release review."""
    candidate.recheck()
    quality.recheck(candidate.hashes["video"])
    require(quality.episode_record_path == candidate.episode_dir / EPISODE_OBSERVATION_FILE,
            "full-audio quality observation is outside the exact episode")
    require(abs(quality.record["audio_duration_seconds"] - candidate.check["duration"]) <= 0.5,
            "full-audio quality observation differs from final MP4 duration")
    require(set(asr_results) == {"agent-asr-gemini.json", "agent-asr-whisper.json"},
            "two specific full-MP4 ASR results are required")
    for name, label in (("agent-asr-gemini.json", "gemini"),
                        ("agent-asr-whisper.json", "whisper")):
        result = asr_results[name]
        _verify_raw_asr(result, label)
        require(result["input_video_sha256"] == candidate.hashes["video"] and
                abs(result["audio_duration_seconds"] - candidate.check["duration"]) <= 0.5,
                f"{label}: ASR is not bound to the full final MP4")
    require(len(asr_refs) == 2 and {item["file"] for item in asr_refs} == set(asr_results),
            "ASR artifact references are incomplete")
    for reference in asr_refs:
        require(digest_file(candidate.episode_dir / reference["file"]) == reference["sha256"],
                "ASR result changed after review")
    require(digest_file(candidate.episode_dir / visual.contact_sheet_file) == visual.contact_sheet_sha256,
            "contact sheet changed after visual review")
    for crop in visual.readable_crops:
        require(digest_file(candidate.episode_dir / crop["file"]) == crop["sha256"],
                "readable crop changed after visual review")
    for pages in observations.claim_pages.values():
        for page in pages.values():
            require(digest_file(candidate.episode_dir / page.response_ref) == page.response_sha256,
                    "fetched source response changed after observation")
            require(digest_file(candidate.episode_dir / page.snapshot_ref) == page.snapshot_sha256,
                    "fetched source snapshot changed after observation")
    for rights in observations.asset_rights.values():
        if isinstance(rights, FetchObservation):
            require(digest_file(candidate.episode_dir / rights.response_ref) == rights.response_sha256,
                    "fetched rights response changed after observation")
            require(digest_file(candidate.episode_dir / rights.snapshot_ref) == rights.snapshot_sha256,
                    "fetched rights snapshot changed after observation")
    decision = model.decision
    reviewed_at = utc_now()
    qa_id = candidate.qa_agent_id

    model_claims = _indexed(decision["claim_findings"],
                            {claim["id"] for claim in candidate.ledger["claims"]}, "claim findings")
    claim_findings: list[dict] = []
    for claim in candidate.ledger["claims"]:
        identity = claim["id"]
        matched = "correspondence" in claim
        keys = {"id", "primary_excerpt", "corroboration_excerpt"} | CLAIM_CHECKS
        if matched:
            keys |= {"checked_cross_edition_alignment", "alignment_notes"}
        item = _semantic(model_claims[identity], keys, f"claim {identity}")
        _checks(item, CLAIM_CHECKS | ({"checked_cross_edition_alignment"} if matched else set()),
                f"claim {identity}")
        if matched:
            require(isinstance(item["alignment_notes"], str) and
                    len(item["alignment_notes"].strip()) >= 30,
                    f"claim {identity}: cross-edition alignment is unexplained")
        pages = observations.claim_pages[identity]
        primary = claim["primary"]
        other = claim["corroboration"]
        for key, source in (("primary", primary), ("corroboration", other)):
            page = pages[key]
            require(source["excerpt"] in page.text and source["printed_verse_label"] in page.text,
                    f"claim {identity} {key}: fetched page lost ledger passage or label")
            require(isinstance(item[f"{key}_excerpt"], str) and
                    len(item[f"{key}_excerpt"].strip()) >= 20,
                    f"claim {identity} {key}: source quotation is too short")
        finding = {**_verdict(qa_id, "agent_checked_independent_sources", reviewed_at, item),
                   "id": identity, "primary_url": primary["url"],
                   "corroboration_url": other["url"],
                   "primary_printed_label": primary["printed_verse_label"],
                   "corroboration_printed_label": other["printed_verse_label"],
                   **{name: item[name] for name in CLAIM_CHECKS},
                   "primary_fetch": pages["primary"].source_record(item["primary_excerpt"],
                                                                      primary["printed_verse_label"]),
                   "corroboration_fetch": pages["corroboration"].source_record(
                       item["corroboration_excerpt"], other["printed_verse_label"])}
        if matched:
            finding["checked_cross_edition_alignment"] = True
            finding["alignment_notes"] = item["alignment_notes"]
        claim_findings.append(finding)

    model_assets = _indexed(decision["asset_findings"], set(candidate.assets), "asset findings")
    asset_findings: list[dict] = []
    for identity, asset in candidate.assets.items():
        rights = observations.asset_rights[identity]
        keys = {"id"} | ASSET_CHECKS | ({"license_excerpt"} if isinstance(rights, FetchObservation) else set())
        item = _semantic(model_assets[identity], keys, f"asset {identity}")
        _checks(item, ASSET_CHECKS, f"asset {identity}")
        if isinstance(rights, FetchObservation):
            excerpt = item["license_excerpt"]
            require(isinstance(excerpt, str) and len(excerpt.strip()) >= 10 and
                    not INCOMPATIBLE.search(excerpt),
                    f"asset {identity}: rights excerpt is unclear or incompatible")
            rights_fetch = rights.rights_record(excerpt)
        else:
            rights_fetch = rights
            require(len(rights_fetch["provenance_excerpt"].strip()) >= 20 and
                    rights_fetch["content_sha256"] == asset["sha256"],
                    f"asset {identity}: internal provenance is incomplete")
        asset_findings.append({**_verdict(qa_id, "agent_checked_origin_and_rights", reviewed_at, item),
                               "id": identity, "sha256": asset["sha256"],
                               **{name: True for name in ASSET_CHECKS}, "rights_fetch": rights_fetch})

    audio_keys = AUDIO_CHECKS | {"beat_reconciliation", "speech_difference_dispositions"}
    transformed = "narration_transform" in candidate.manifest
    if transformed:
        audio_keys.add("checked_narration_transform")
    audio_item = _semantic(decision["audio_review"], audio_keys, "audio review")
    _checks(audio_item, AUDIO_CHECKS | ({"checked_narration_transform"} if transformed else set()),
            "audio review")
    raw_beats = audio_item["beat_reconciliation"]
    require(isinstance(raw_beats, list) and len(raw_beats) == len(candidate.script["beats"]),
            "audio review does not cover every frozen beat")
    beat_reconciliation: list[dict] = []
    for index, (expected, raw) in enumerate(zip(candidate.script["beats"], raw_beats), 1):
        keys = {"beat", "asr_evidence"} | CRITICAL_FLAGS
        item = _semantic(raw, keys, f"audio beat {index}")
        require(item["beat"] == index, f"audio beat {index} has a wrong beat number")
        _no_critical(item, f"audio beat {index}")
        terms = required_beat_terms(expected["text"])
        render_beat = candidate.manifest["beats"][index - 1]
        citations = _citations(item["asr_evidence"], asr_results, f"audio beat {index}",
                               critical_terms=terms,
                               beat_span=(render_beat["start"], render_beat["end"]))
        beat_reconciliation.append({**_verdict(qa_id, "agent_reconciled_asr_beat", reviewed_at, item),
                                    "beat": index, "expected_text": expected["text"],
                                    "critical_terms": terms, "asr_evidence": citations,
                                    **{name: False for name in CRITICAL_FLAGS}})

    raw_differences = audio_item["speech_difference_dispositions"]
    differences = candidate.check["speech_differences"]
    require(isinstance(raw_differences, list) and len(raw_differences) == len(differences),
            "every exact QC speech difference needs a disposition")
    speech_dispositions: list[dict] = []
    for index, (expected, raw) in enumerate(zip(differences, raw_differences), 1):
        item = _semantic(raw, {"difference", "reason", "asr_evidence"} | CRITICAL_FLAGS,
                         f"speech difference {index}", decision="accepted")
        require(item["difference"] == expected and isinstance(item["reason"], str) and
                len(item["reason"].strip()) >= 30,
                f"speech difference {index}: exact QC finding lacks a reason")
        _no_critical(item, f"speech difference {index}")
        citations = _citations(item["asr_evidence"], asr_results, f"speech difference {index}",
                               expected_phrase=_quoted_expected(expected))
        speech_dispositions.append({**_verdict(qa_id, "agent_reconciled_asr_difference", reviewed_at, item),
                                    "difference": expected, "reason": item["reason"],
                                    "asr_evidence": citations,
                                    **{name: False for name in CRITICAL_FLAGS}})
    voice_id = candidate.manifest["narration_asset_id"]
    audio_review = {**_verdict(qa_id, "agent_reconciled_two_full_audio_asr_results", reviewed_at, audio_item),
                    "voice_asset_id": voice_id, "voice_sha256": candidate.assets[voice_id]["sha256"],
                    "asr_results": asr_refs, **{name: True for name in AUDIO_CHECKS},
                    "quality_observation": {"file": EPISODE_OBSERVATION_FILE,
                                            "sha256": quality.record_sha256},
                    "beat_reconciliation": beat_reconciliation,
                    "speech_difference_dispositions": speech_dispositions}
    audio_review["notes"] += (
        " Gemini full-audio model observation (not human listening): " +
        quality.decision["summary"] + " Uncertainty: " + quality.decision["uncertainty_notes"])
    if transformed:
        audio_review["checked_narration_transform"] = True

    video_item = _semantic(decision["video_review"], VIDEO_CHECKS | {"critical_defects"}, "video review")
    _checks(video_item, VIDEO_CHECKS, "video review")
    require(video_item["critical_defects"] == [], "sampled visual review found critical defects")
    video_review = {**_verdict(qa_id, "agent_decoded_and_inspected_samples", reviewed_at, video_item),
                    "video_sha256": candidate.hashes["video"], **visual.decoder,
                    "contact_sheet_file": visual.contact_sheet_file,
                    "contact_sheet_sha256": visual.contact_sheet_sha256,
                    "sample_times_seconds": visual.sample_times_seconds,
                    "readable_crops": visual.readable_crops, "critical_defects": [],
                    **{name: True for name in VIDEO_CHECKS}}

    warnings = candidate.check["warnings"]
    raw_warnings = decision["qc_warning_dispositions"]
    require(isinstance(raw_warnings, list) and len(raw_warnings) == len(warnings),
            "every exact QC warning needs one disposition, including repeats")
    warning_dispositions: list[dict] = []
    for index, (expected, raw) in enumerate(zip(warnings, raw_warnings), 1):
        speech_warning = expected.startswith("speech:") or "recognizer heard" in expected
        fields = {"warning", "reason"} | ({"asr_evidence"} | CRITICAL_FLAGS if speech_warning else set())
        item = _semantic(raw, fields, f"QC warning {index}", decision="accepted")
        require(item["warning"] == expected and isinstance(item["reason"], str) and
                len(item["reason"].strip()) >= 30,
                f"QC warning {index}: exact warning lacks reasoned disposition")
        disposition = {**_verdict(qa_id, "agent_inspected_qc_warning", reviewed_at, item),
                       "warning": expected, "reason": item["reason"]}
        if speech_warning:
            _no_critical(item, f"QC warning {index}")
            disposition["asr_evidence"] = _citations(item["asr_evidence"], asr_results,
                                                      f"QC warning {index}",
                                                      expected_phrase=_quoted_expected(expected))
            disposition.update({name: False for name in CRITICAL_FLAGS})
        warning_dispositions.append(disposition)

    release_item = _semantic(decision["release_review"], RELEASE_CHECKS, "release review")
    _checks(release_item, RELEASE_CHECKS, "release review")
    release_review = {**_verdict(qa_id, "agent_independent_release_review", reviewed_at, release_item),
                      **{name: True for name in RELEASE_CHECKS}}
    completed_at = utc_now()
    review = {"kind": "agent_episode_qa_v1", "episode_id": candidate.episode_id,
              "production_agent_id": candidate.producer_agent_id,
              "qa_agent_id": qa_id,
              "qa_run": _qa_run(candidate, run, [quality.model_call, model.model_call], completed_at),
              **{f"{name}_sha256": candidate.hashes[name] for name in
                 ("script", "spec", "video", "qc", "evidence", "manifest")},
              "claim_findings": claim_findings, "asset_findings": asset_findings,
              "audio_review": audio_review, "video_review": video_review,
              "qc_warning_dispositions": warning_dispositions,
              "release_review": release_review}
    candidate.recheck()
    quality.recheck(candidate.hashes["video"])
    return review


def save_unsigned_review(candidate: Candidate, review: dict) -> Path:
    """Save only a fully assembled approval; signing belongs to another job."""
    require(review.get("kind") == "agent_episode_qa_v1" and review.get("episode_id") == candidate.episode_id,
            "cannot save a review of a different candidate")
    candidate.recheck()
    path = candidate.episode_dir / "agent-release-review.json"
    write_json_new(path, review)
    require(json_object(path.read_bytes(), "saved agent review") == review,
            "saved agent review changed during serialization")
    return path
