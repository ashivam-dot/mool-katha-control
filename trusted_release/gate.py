"""Trusted control gate: recheck signed QA evidence, then sign exact release inputs.

This module has no CLI, workflow, key lookup, or publisher. The gate signing key
belongs to a separate control job from the QA signer and release publisher.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trusted_qa.candidate import Candidate, load_candidate
from trusted_qa.common import QaHold, digest_file, json_object, path_under, timestamp, write_json_new
from trusted_qa.fetch import _readable_and_links
from trusted_qa.media import FRAME_BATCH_SIZE, MAX_REVIEW_BATCHES, extract_full_final_audio
from trusted_qa.observations import FONT_ORIGIN, OFL_GRANTS
from trusted_qa.provenance import font_source, verify_generated_asset
from trusted_qa.voice_exchange import (QA_BUNDLE_DIR, VoiceExchangePolicy,
                                       verify_candidate_voice)

from .executor import (GATE_CONTEXT, ReleaseHold, ReleasePlan, ReleasePolicy,
                       _check_signature, _read_archive_video, _verify_frozen_source,
                       require)


def _same_ids(rows: Any, expected: set[str], label: str) -> dict[str, dict]:
    require(type(rows) is list and len(rows) == len(expected) and
            all(type(row) is dict and type(row.get("id")) is str for row in rows),
            f"gate {label} are incomplete")
    indexed = {row["id"]: row for row in rows}
    require(len(indexed) == len(rows) and set(indexed) == expected,
            f"gate {label} differ from source evidence")
    return indexed


def _approved(row: Any, qa_id: str, label: str) -> None:
    require(type(row) is dict and row.get("agent_id") == qa_id and
            row.get("decision") == "approved" and row.get("unresolved_items") == [],
            f"gate {label} is not an approved independent QA verdict")


def _bound_file(root: Path, reference: Any, digest: Any, label: str) -> None:
    require(type(digest) is str and len(digest) == 64, f"gate {label} hash is invalid")
    try:
        path = path_under(root, reference, label)
        require(path.is_file() and not path.is_symlink() and digest_file(path) == digest,
                f"gate {label} changed after QA")
    except QaHold as exc:
        raise ReleaseHold(f"gate {label} is missing or unsafe") from exc


def _check_fetch(root: Path, record: Any, label: str) -> None:
    require(type(record) is dict, f"gate {label} evidence is missing")
    for kind in ("response", "snapshot"):
        _bound_file(root, record.get(f"{kind}_ref"), record.get(f"{kind}_sha256"),
                    f"{label} {kind}")


def _fetched_text(root: Path, record: dict, label: str) -> str:
    _check_fetch(root, record, label)
    raw = path_under(root, record["snapshot_ref"], label).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ReleaseHold(f"gate {label} snapshot is not UTF-8") from exc
    require(text.endswith("\n"), f"gate {label} snapshot is incomplete")
    return text[:-1]


def _check_http_asset_origin(root: Path, source: dict, row: dict, identity: str) -> None:
    proof = row.get("origin_proof")
    require(type(proof) is dict and set(proof) ==
            {"source_object_id", "rights_basis", "origin_fetch", "exact_file"} and
            proof["source_object_id"] == source.get("source_object_id") and
            proof["rights_basis"] == source.get("rights_basis"),
            f"gate asset {identity} object-specific origin is incomplete")
    origin = proof["origin_fetch"]
    rights = row["rights_fetch"]
    require(type(origin) is dict and set(origin) ==
            {"url", "fetched_at", "http_status", "final_url", "content_type",
             "response_sha256", "response_ref", "snapshot_ref", "snapshot_sha256",
             "visible_links"} and
            origin["url"] == source.get("origin") and
            origin["http_status"] == 200 and
            type(origin["final_url"]) is str and
            origin["final_url"].startswith("https://") and
            type(origin["content_type"]) is str and
            type(origin["visible_links"]) is list and
            len(origin["visible_links"]) <= 4096 and
            all(type(link) is str and len(link) <= 4096 for link in origin["visible_links"]) and
            rights.get("url") == source.get("rights_url") and
            rights.get("http_status") == 200,
            f"gate asset {identity} HTTP origin or rights URL differs from source")
    origin_text = _fetched_text(root, origin, f"asset {identity} origin")
    rights_text = _fetched_text(root, rights, f"asset {identity} rights")
    response = path_under(root, origin["response_ref"], f"asset {identity} origin response")
    observed_text, raw_links = _readable_and_links(response.read_bytes(), origin["content_type"])
    observed_links = [urljoin(origin["final_url"], link) for link in raw_links
                      if urlsplit(urljoin(origin["final_url"], link)).scheme == "https"]
    require(observed_text == origin_text and observed_links == origin["visible_links"],
            f"gate asset {identity} origin snapshot differs from raw HTTP response")
    object_id = proof["source_object_id"]
    require(type(object_id) is str and len(object_id) >= 8 and
            object_id.casefold() in unquote(urlsplit(origin["final_url"]).path + "?" +
                                           urlsplit(origin["final_url"]).query).casefold() and
            object_id.casefold() in origin_text.casefold() and
            object_id.casefold() in rights_text.casefold() and
            type(source.get("license")) is str and
            source["license"].casefold() in rights_text.casefold() and
            type(rights.get("license_excerpt")) is str and
            rights["license_excerpt"] in rights_text,
            f"gate asset {identity} origin or licence is not object-specific")
    official_url = source.get("official_asset_url")
    exact = proof["exact_file"]
    require(type(exact) is dict and exact.get("sha256") == source["sha256"],
            f"gate asset {identity} exact-file relation is incomplete")
    if official_url is None:
        require(set(exact) == {"method", "sha256"} and
                exact["method"] == "visible_sha256" and
                source["sha256"].casefold() in origin_text.casefold(),
                f"gate asset {identity} origin does not fingerprint the exact file")
    else:
        require(set(exact) == {"method", "sha256", "official_asset_url", "response_ref"} and
                exact["method"] == "official_download" and
                exact["official_asset_url"] == official_url and
                (official_url in origin["visible_links"] or official_url in origin_text),
                f"gate asset {identity} official download is not linked from origin")
        _bound_file(root, exact["response_ref"], source["sha256"],
                    f"asset {identity} official download")
    require(rights["url"] == origin["url"] or
            origin["url"].casefold() in rights_text.casefold() or
            source["sha256"].casefold() in rights_text.casefold() or
            type(official_url) is str and official_url.casefold() in rights_text.casefold(),
            f"gate asset {identity} rights do not refer to the exact object")


def _check_generated_asset_origin(candidate: Candidate, source: dict,
                                  row: dict, identity: str) -> None:
    proof = row.get("origin_proof")
    rights = row.get("rights_fetch")
    require(type(rights) is dict and set(rights) ==
            {"origin", "checked_at", "content_sha256", "provenance_excerpt"} and
            rights["origin"] == source.get("origin") and
            rights["content_sha256"] == source["sha256"] and
            type(rights["provenance_excerpt"]) is str and
            len(rights["provenance_excerpt"].strip()) >= 20 and
            rights["provenance_excerpt"] in source.get("rights_basis", ""),
            f"gate asset {identity} internal rights record is incomplete")
    try:
        timestamp(rights["checked_at"], f"gate asset {identity} generation check")
        checked = verify_generated_asset(candidate, source)
    except QaHold as exc:
        raise ReleaseHold(f"gate asset {identity} failed independent generation check: {exc}") from exc
    require(proof == checked,
            f"gate asset {identity} signed generation proof differs from control reproduction")


def _check_font_asset_origin(root: Path, candidate: Candidate, source: dict,
                             row: dict, identity: str) -> None:
    proof = row.get("origin_proof")
    rights = row.get("rights_fetch")
    require(type(proof) is dict and set(proof) ==
            {"kind", "source", "upstream_revision", "font_download", "license_sha256"} and
            proof["kind"] == "control_upstream_font_v1" and
            type(rights) is dict and set(rights) ==
            {"url", "fetched_at", "http_status", "response_sha256", "response_ref",
             "snapshot_ref", "snapshot_sha256", "license_excerpt"},
            f"gate asset {identity} exact upstream font proof is incomplete")
    try:
        exact_source = font_source(candidate, source)
    except QaHold as exc:
        raise ReleaseHold(f"gate asset {identity} font source failed Git check") from exc
    origin = source.get("origin")
    match = FONT_ORIGIN.fullmatch(origin) if type(origin) is str else None
    require(match is not None and proof["source"] == exact_source and
            proof["upstream_revision"] == match[1] and
            match[2] == match[3].split("-")[0].lower() and
            source["file"] == f"pipeline/assets/fonts/{match[3]}" and
            source.get("license") == "SIL Open Font License 1.1",
            f"gate asset {identity} font URL or source differs from frozen Google Fonts release")
    expected_rights_url = (f"https://raw.githubusercontent.com/google/fonts/{match[1]}/"
                           f"ofl/{match[2]}/OFL.txt")
    require(source.get("rights_url") == expected_rights_url and
            rights.get("url") == expected_rights_url and rights.get("http_status") == 200,
            f"gate asset {identity} upstream OFL URL differs from exact font release")
    download = proof["font_download"]
    require(type(download) is dict and set(download) ==
            {"url", "final_url", "response_ref", "sha256"} and
            download["url"] == origin and
            type(download["final_url"]) is str and
            download["final_url"].startswith("https://") and
            download["sha256"] == source["sha256"],
            f"gate asset {identity} exact font download is incomplete")
    _bound_file(root, download["response_ref"], source["sha256"],
                f"asset {identity} Google Fonts TTF")
    _check_fetch(root, rights, f"asset {identity} upstream OFL")
    raw_license = path_under(root, rights["response_ref"],
                             f"asset {identity} upstream OFL raw response").read_bytes()
    try:
        visible, links = _readable_and_links(raw_license, "text/plain")
    except QaHold as exc:
        raise ReleaseHold(f"gate asset {identity} upstream OFL text cannot be checked") from exc
    snapshot = _fetched_text(root, rights, f"asset {identity} upstream OFL")
    require(links == () and visible == snapshot and
            proof["license_sha256"] == exact_source["license"]["sha256"] and
            rights["response_sha256"] == proof["license_sha256"] and
            type(rights["license_excerpt"]) is str and
            rights["license_excerpt"] in snapshot and
            all(clause in snapshot for clause in OFL_GRANTS),
            f"gate asset {identity} upstream OFL bytes or grant differ from exact font")


def _replay_control_voice_origin(root: Path, candidate: Candidate, source: dict,
                                 row: dict, identity: str, *,
                                 voice_policy: VoiceExchangePolicy | None,
                                 final_audio_path: Path | None) -> dict:
    """Replay the QA-copied signed bundle against a fresh final-video audio decode."""
    require(isinstance(voice_policy, VoiceExchangePolicy) and
            isinstance(final_audio_path, Path),
            f"gate asset {identity} lacks an independently pinned voice policy or final audio")
    try:
        replayed = verify_candidate_voice(candidate, source, root / QA_BUNDLE_DIR,
                                          final_audio_path, voice_policy)
    except QaHold as exc:
        raise ReleaseHold(f"gate asset {identity} control voice replay failed: {exc}") from exc
    signed = row.get("origin_proof")
    stable = set(replayed) - {"final_audio_sha256", "audio_match"}
    require(type(signed) is dict and set(signed) == set(replayed) and
            all(signed[key] == replayed[key] for key in stable),
            f"gate asset {identity} signed control voice proof differs from independent replay")
    metrics = signed["audio_match"]
    require(type(signed["final_audio_sha256"]) is str and
            re.fullmatch(r"[0-9a-f]{64}", signed["final_audio_sha256"]) is not None and
            type(metrics) is dict and set(metrics) ==
            {"method", "offset_samples_16k", "global_correlation",
             "median_window_correlation", "checked_windows", "strong_windows"} and
            metrics["method"] == "waveform_correlation_v1" and
            type(metrics["offset_samples_16k"]) is int and
            abs(metrics["offset_samples_16k"]) <= 8_000 and
            all(type(metrics[key]) in (int, float) and math.isfinite(metrics[key])
                for key in ("global_correlation", "median_window_correlation")) and
            0.45 <= metrics["global_correlation"] <= 1 and
            0.50 <= metrics["median_window_correlation"] <= 1 and
            type(metrics["checked_windows"]) is int and
            3 <= metrics["checked_windows"] <= 30 and
            type(metrics["strong_windows"]) is int and
            math.ceil(0.7 * metrics["checked_windows"]) <=
            metrics["strong_windows"] <= metrics["checked_windows"],
            f"gate asset {identity} signed control voice audio observation is malformed")
    return replayed


def _check_asset_origin(root: Path, candidate: Candidate,
                        source: dict, row: dict, identity: str, *,
                        voice_policy: VoiceExchangePolicy | None = None,
                        final_audio_path: Path | None = None) -> None:
    if source.get("role") == "voice" and str(source.get("origin", "")).startswith(
            "control:gemini:"):
        _replay_control_voice_origin(root, candidate, source, row, identity,
                                     voice_policy=voice_policy,
                                     final_audio_path=final_audio_path)
        raise ReleaseHold(f"gate asset {identity} commercial-use rights remain unresolved; "
                          "an independently verified explicit commercial-use basis is required")
    voice_spec = candidate.spec.get("voice")
    if source.get("role") == "voice" and type(voice_spec) is dict and voice_spec.get(
            "engine") == "gemini":
        raise ReleaseHold(f"gate asset {identity} Gemini voice lacks a signed "
                          "control provider-origin exchange")
    if source.get("role") == "font" and (
        "font_sources" in candidate.manifest or
        str(source.get("origin", "")).startswith(
            "https://raw.githubusercontent.com/google/fonts/")):
        _check_font_asset_origin(root, candidate, source, row, identity)
    elif source.get("role") in {"visual", "music"} and str(source.get("origin", "")).startswith("internal:"):
        _check_generated_asset_origin(candidate, source, row, identity)
    else:
        _check_http_asset_origin(root, source, row, identity)


def _check_frame_review(root: Path, candidate: Any, review: dict) -> None:
    run = review["qa_run"]
    summary = run.get("frame_batch_review")
    require(type(summary) is dict and set(summary) ==
            {"kind", "input_video_sha256", "all_frame_audit_file",
             "all_frame_audit_sha256", "decoded_frame_count", "batches"} and
            summary["kind"] == "frame_indexed_visual_review_v1" and
            summary["input_video_sha256"] == candidate.hashes["video"] and
            summary["all_frame_audit_file"] == "agent-video-frame-audit.json" and
            type(summary["decoded_frame_count"]) is int and
            1 <= summary["decoded_frame_count"] <= FRAME_BATCH_SIZE * MAX_REVIEW_BATCHES and
            summary["decoded_frame_count"] == review["video_review"].get("decoded_frame_count"),
            "gate signed frame review is incomplete or describes another video")
    _bound_file(root, summary["all_frame_audit_file"],
                summary["all_frame_audit_sha256"], "all-frame pixel audit")
    audit = json_object((root / summary["all_frame_audit_file"]).read_bytes(),
                        "all-frame pixel audit")
    count = summary["decoded_frame_count"]
    frames = audit.get("frames")
    batches = summary["batches"]
    require(audit.get("kind") == "all_frame_pixel_temporal_audit_v1" and
            audit.get("input_video_sha256") == candidate.hashes["video"] and
            audit.get("decoded_frame_count") == count and
            audit.get("anomalies") == [] and
            type(frames) is list and len(frames) == count and
            [frame.get("index") for frame in frames if type(frame) is dict] ==
            list(range(1, count + 1)) and
            type(batches) is list and 1 <= len(batches) <= MAX_REVIEW_BATCHES and
            type(audit.get("frame_batches")) is list and
            len(audit["frame_batches"]) == len(batches) and
            len(run["review_model_calls"]) == len(batches) + 2,
            "gate all-frame audit has missing or extra frames or model calls")
    next_index = 1
    for index, (batch, audited) in enumerate(zip(batches, audit["frame_batches"])):
        require(type(batch) is dict and set(batch) ==
                {"file", "sha256", "start_index", "end_index", "first_seconds",
                 "last_seconds", "decision", "uncertainty", "checked_indices",
                 "defect_indices", "notes", "model_call", "request_sha256",
                 "response_sha256"} and
                type(audited) is dict and
                {key: batch[key] for key in ("file", "sha256", "start_index", "end_index",
                                              "first_seconds", "last_seconds")} == audited and
                batch["start_index"] == next_index and
                type(batch["end_index"]) is int and
                next_index <= batch["end_index"] < next_index + FRAME_BATCH_SIZE and
                batch["checked_indices"] == list(range(next_index, batch["end_index"] + 1)) and
                batch["decision"] == "clear" and batch["uncertainty"] == "low" and
                batch["defect_indices"] == [] and
                type(batch["notes"]) is str and len(batch["notes"].strip()) >= 30 and
                batch["model_call"] == run["review_model_calls"][index] and
                all(type(batch[key]) is str and len(batch[key]) == 64
                    for key in ("request_sha256", "response_sha256")),
                f"gate frame batch {index + 1} has missing coverage or an unresolved defect")
        _bound_file(root, batch["file"], batch["sha256"], f"frame batch {index + 1}")
        next_index = batch["end_index"] + 1
    require(next_index == count + 1, "gate frame review did not cover every decoded frame")


def _check_review_evidence(artifact_dir: Path, candidate: Any, review: dict, *,
                           voice_policy: VoiceExchangePolicy | None = None,
                           final_audio_path: Path | None = None) -> None:
    qa_id = candidate.qa_agent_id
    claims = _same_ids(review.get("claim_findings"),
                       {item["id"] for item in candidate.ledger["claims"]}, "claim findings")
    for source in candidate.ledger["claims"]:
        row = claims[source["id"]]
        _approved(row, qa_id, f"claim {source['id']}")
        for side in ("primary", "corroboration"):
            require(row.get(f"{side}_url") == source[side]["url"] and
                    row.get(f"{side}_printed_label") == source[side]["printed_verse_label"],
                    f"gate claim {source['id']} source differs from frozen ledger")
            _check_fetch(artifact_dir, row.get(f"{side}_fetch"), f"claim {source['id']} {side}")
    assets = _same_ids(review.get("asset_findings"), set(candidate.assets), "asset findings")
    for identity, source in candidate.assets.items():
        row = assets[identity]
        _approved(row, qa_id, f"asset {identity}")
        require(row.get("sha256") == source["sha256"],
                f"gate asset {identity} differs from exact media")
        if source.get("role") == "voice" and str(source.get("origin", "")).startswith(
                "control:gemini:"):
            _check_asset_origin(artifact_dir, candidate, source, row, identity,
                                voice_policy=voice_policy,
                                final_audio_path=final_audio_path)
        rights = row.get("rights_fetch")
        require(type(rights) is dict, f"gate asset {identity} rights are missing")
        _check_asset_origin(artifact_dir, candidate, source, row, identity)
    audio = review.get("audio_review")
    _approved(audio, qa_id, "audio review")
    require(type(audio.get("asr_results")) is list and len(audio["asr_results"]) == 2 and
            {ref.get("file") for ref in audio["asr_results"] if type(ref) is dict} ==
            {"agent-asr-gemini.json", "agent-asr-whisper.json"},
            "gate independent ASR references are incomplete")
    for ref in audio["asr_results"]:
        _bound_file(artifact_dir, ref["file"], ref.get("sha256"), "full audio ASR")
        result = json_object((artifact_dir / ref["file"]).read_bytes(), "full audio ASR")
        require(result.get("input_video_sha256") == candidate.hashes["video"],
                "gate ASR describes another video")
    beats = audio.get("beat_reconciliation")
    require(type(beats) is list and len(beats) == len(candidate.script["beats"]),
            "gate audio does not cover every frozen beat")
    for index, beat in enumerate(beats, 1):
        _approved(beat, qa_id, f"audio beat {index}")
        require(beat.get("beat") == index and
                beat.get("expected_text") == candidate.script["beats"][index - 1]["text"] and
                type(beat.get("asr_evidence")) is list and len(beat["asr_evidence"]) >= 2,
                f"gate audio beat {index} is incomplete")
    require(type(audio.get("speech_difference_dispositions")) is list and
            [item.get("difference") for item in audio["speech_difference_dispositions"]] ==
            candidate.check["speech_differences"],
            "gate audio speech differences are incomplete")
    for row in audio["speech_difference_dispositions"]:
        _approved(row, qa_id, "speech difference")
    video = review.get("video_review")
    _approved(video, qa_id, "video review")
    require(video.get("video_sha256") == candidate.hashes["video"] and
            video.get("critical_defects") == [], "gate decoded video verdict is incomplete")
    _bound_file(artifact_dir, video.get("contact_sheet_file"),
                video.get("contact_sheet_sha256"), "contact sheet")
    require(type(video.get("readable_crops")) is list, "gate video crops are incomplete")
    for crop in video["readable_crops"]:
        require(type(crop) is dict, "gate video crop is malformed")
        _bound_file(artifact_dir, crop.get("file"), crop.get("sha256"), "readable crop")
    _check_frame_review(artifact_dir, candidate, review)
    warnings = review.get("qc_warning_dispositions")
    require(type(warnings) is list and [item.get("warning") for item in warnings] ==
            candidate.check["warnings"], "gate QC warnings are incomplete")
    for row in warnings:
        _approved(row, qa_id, "QC warning")
    quality_ref = audio.get("quality_observation")
    require(type(quality_ref) is dict, "gate voice-quality reference is missing")
    _bound_file(artifact_dir, quality_ref.get("file"), quality_ref.get("sha256"),
                "voice-quality observation")
    quality = json_object((artifact_dir / quality_ref["file"]).read_bytes(), "voice quality")
    decision = quality.get("decision")
    require(type(decision) is dict and decision.get("decision") == "clear" and
            decision.get("uncertainty") == "low" and
            type(decision.get("observations")) is list and len(decision["observations"]) == 5 and
            all(type(item) is dict and item.get("status") == "clear"
                for item in decision["observations"]),
            "gate voice-quality observation has a concern or uncertainty")


def sign_gate_attestation(plan: ReleasePlan, policy: ReleasePolicy, artifact_dir: Path,
                          archive: Path, source_repo: Path, gate_run: dict,
                          key_id: str, private_key_seed: bytes) -> tuple[Path, Path]:
    """Sign only after separate, fixed-code checks over actual QA and source bytes."""
    require(type(key_id) is str and key_id in policy.gate_keys and
            type(private_key_seed) is bytes and len(private_key_seed) == 32,
            "trusted gate signing key is unavailable")
    key = Ed25519PrivateKey.from_private_bytes(private_key_seed)
    public = key.public_key().public_bytes(serialization.Encoding.Raw,
                                            serialization.PublicFormat.Raw)
    require(public == policy.gate_keys[key_id], "trusted gate private key differs from reviewed policy")
    require(type(gate_run) is dict and set(gate_run) ==
            {"system", "repository", "workflow_ref", "workflow_sha", "run_id", "run_attempt"} and
            gate_run["system"] == "github_actions" and
            gate_run["repository"] == policy.qa_repository and
            gate_run["workflow_ref"] == policy.gate_workflow_ref and
            gate_run["workflow_sha"] == policy.gate_workflow_sha and
            all(type(gate_run[name]) is int and gate_run[name] > 0
                for name in ("run_id", "run_attempt")),
            "trusted gate run is not pinned to the control workflow")
    try:
        actual_run = {"system": "github_actions", "repository": os.environ["GITHUB_REPOSITORY"],
                      "workflow_ref": os.environ["GITHUB_WORKFLOW_REF"],
                      "workflow_sha": os.environ["GITHUB_WORKFLOW_SHA"],
                      "run_id": int(os.environ["GITHUB_RUN_ID"]),
                      "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"])}
    except (KeyError, ValueError) as exc:
        raise ReleaseHold("trusted gate cloud run context is unavailable") from exc
    require(gate_run == actual_run, "trusted gate run differs from actual cloud context")
    review = _check_signature(artifact_dir, plan, policy)
    frozen = _verify_frozen_source(source_repo, artifact_dir, plan, review)
    with tempfile.TemporaryDirectory(prefix="mool-katha-gate-") as scratch:
        root = Path(scratch)
        _read_archive_video(archive, plan, root / "final.mp4")
        try:
            candidate = load_candidate(source_repo, archive, root / "snapshot",
                                       plan.episode_id, plan.source_commit, review["qa_agent_id"])
        except QaHold as exc:
            raise ReleaseHold("gate candidate snapshot failed independent validation") from exc
        require(all(candidate.hashes[name] == review[f"{name}_sha256"]
                    for name in ("script", "spec", "video", "qc", "evidence", "manifest")),
                "gate candidate differs from signed QA hashes")
        try:
            final_audio_path = None
            if any(asset.get("role") == "voice" and
                   str(asset.get("origin", "")).startswith("control:gemini:")
                   for asset in candidate.assets.values()):
                final_audio_path = root / "gate-full-final-audio.wav"
                extract_full_final_audio(candidate.video_path, final_audio_path,
                                         candidate.check["duration"])
            _check_review_evidence(artifact_dir, candidate, review,
                                   voice_policy=policy.voice_policy,
                                   final_audio_path=final_audio_path)
            candidate.recheck()
            require(digest_file(archive) == plan.archive_sha256,
                    "gate private archive changed during evidence validation")
        except QaHold as exc:
            raise ReleaseHold("gate QA evidence is missing or changed") from exc
    # Catch a changed review, source checkout, or quality file before signing.
    require(_check_signature(artifact_dir, plan, policy) == review,
            "gate signed QA review changed during validation")
    require(_verify_frozen_source(source_repo, artifact_dir, plan, review) == frozen,
            "gate frozen source changed during validation")
    signature_bytes = (artifact_dir / "agent-release-signature.json").read_bytes()
    attestation = {
        "kind": "control_release_gate_attestation_v1", "decision": "approved",
        "episode_id": plan.episode_id, "source_commit": plan.source_commit,
        "archive_sha256": plan.archive_sha256, "review_sha256": plan.review_sha256,
        "review_signature_sha256": hashlib.sha256(signature_bytes).hexdigest(),
        "video_sha256": plan.video_sha256, "frozen_sha256": frozen,
        "quality_observation_sha256": review["audio_review"]["quality_observation"]["sha256"],
        "gate_run": gate_run,
    }
    if policy.voice_policy is not None:
        attestation["voice_policy_sha256"] = policy.voice_policy_sha256()
    attestation_path = artifact_dir / "release-gate-attestation.json"
    signature_path = artifact_dir / "release-gate-signature.json"
    require(not attestation_path.exists() and not signature_path.exists(),
            "gate attestation or signature already exists")
    raw = (json.dumps(attestation, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    signature = key.sign(GATE_CONTEXT + raw)
    envelope = {"kind": "control_release_gate_signature_v1", "algorithm": "Ed25519",
                "key_id": key_id, "attestation_sha256": hashlib.sha256(raw).hexdigest(),
                "signature": base64.b64encode(signature).decode("ascii")}
    # Files are created exclusively. A partial write never grants release authority.
    write_json_new(attestation_path, attestation)
    write_json_new(signature_path, envelope)
    return attestation_path, signature_path
