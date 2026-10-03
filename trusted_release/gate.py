"""Trusted control gate: recheck signed QA evidence, then sign exact release inputs.

This module has no CLI, workflow, key lookup, or publisher. The gate signing key
belongs to a separate control job from the QA signer and release publisher.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trusted_qa.candidate import load_candidate
from trusted_qa.common import QaHold, digest_file, json_object, path_under, write_json_new

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


def _check_review_evidence(artifact_dir: Path, candidate: Any, review: dict) -> None:
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
        rights = row.get("rights_fetch")
        require(type(rights) is dict, f"gate asset {identity} rights are missing")
        if "response_ref" in rights or "snapshot_ref" in rights:
            _check_fetch(artifact_dir, rights, f"asset {identity} rights")
        else:
            require(rights.get("content_sha256") == source["sha256"] and
                    type(rights.get("provenance_excerpt")) is str and
                    len(rights["provenance_excerpt"].strip()) >= 20,
                    f"gate asset {identity} provenance is incomplete")
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
            _check_review_evidence(artifact_dir, candidate, review)
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
