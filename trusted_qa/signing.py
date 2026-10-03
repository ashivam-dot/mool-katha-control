"""Sign only the review just assembled by the trusted QA process."""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .audio_quality import AudioQualityObservation
from .candidate import Candidate
from .common import json_object, require, write_json_new

QA_CONTEXT = b"mool-katha-agent-release-v1\0"


def sign_assembled_review(candidate: Candidate, review: dict, review_path: Path,
                          quality: AudioQualityObservation, run: dict,
                          key_id: str, private_key_seed: bytes) -> Path:
    """The caller must invoke this in the same successful QA run, not on an upload."""
    require(type(key_id) is str and re.fullmatch(r"[A-Za-z0-9._-]{4,64}", key_id) is not None,
            "QA signing key ID is invalid")
    require(type(private_key_seed) is bytes and len(private_key_seed) == 32,
            "QA signing key is unavailable")
    require(review_path == candidate.episode_dir / "agent-release-review.json" and
            review_path.is_file() and not review_path.is_symlink(),
            "QA review is outside the assembled candidate")
    candidate.recheck()
    quality.recheck(candidate.hashes["video"])
    raw = review_path.read_bytes()
    require(json_object(raw, "assembled QA review") == review and
            review.get("episode_id") == candidate.episode_id and
            review.get("video_sha256") == candidate.hashes["video"] and
            run.get("workflow_ref", "").endswith("@refs/tags/qa-v1") and
            review.get("qa_run") == {**run,
                                     "completed_at": review.get("qa_run", {}).get("completed_at"),
                                     "review_model_calls": review.get("qa_run", {}).get("review_model_calls")},
            "QA review changed after strict assembly or lacks tagged provenance")
    require(review.get("release_review", {}).get("decision") == "approved" and
            review.get("audio_review", {}).get("quality_observation") ==
            {"file": "agent-audio-quality-observation.json", "sha256": quality.record_sha256},
            "QA approval or full audio observation is unavailable")
    signature = Ed25519PrivateKey.from_private_bytes(private_key_seed).sign(QA_CONTEXT + raw)
    envelope = {"kind": "agent_release_signature_v1", "algorithm": "Ed25519",
                "key_id": key_id, "review_sha256": hashlib.sha256(raw).hexdigest(),
                "signature": base64.b64encode(signature).decode("ascii")}
    output = candidate.episode_dir / "agent-release-signature.json"
    write_json_new(output, envelope)
    candidate.recheck()
    quality.recheck(candidate.hashes["video"])
    return output
