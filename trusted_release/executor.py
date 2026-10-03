"""Release one signed, exact-byte candidate through a narrowly injected publisher.

This module imports no producer code. The policy and plan must come from reviewed
control code, never from an episode, model output, or production repository.
There is deliberately no command-line entry point or automatic workflow.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
import subprocess
import tarfile
import tempfile
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

CONTEXT = b"mool-katha-agent-release-v1\0"
IST = ZoneInfo("Asia/Kolkata")
EPISODE = re.compile(r"ep[0-9]{3}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SHA1 = re.compile(r"[0-9a-f]{40}\Z")
KEY_ID = re.compile(r"[A-Za-z0-9._-]{4,64}\Z")
MAX_ARCHIVE_BYTES = 750 * 1024 * 1024
MAX_VIDEO_BYTES = 260 * 1024 * 1024
MAX_REVIEW_BYTES = 8 * 1024 * 1024
MAX_SIGNATURE_BYTES = 16 * 1024
MAX_QUALITY_BYTES = 128 * 1024
MAX_POSTS = 500
FROZEN_FILES = {"script.json": "script_sha256", "short.yaml": "spec_sha256",
                "qc.json": "qc_sha256", "evidence.json": "evidence_sha256",
                "work/manifest.json": "manifest_sha256"}
GATE_CONTEXT = b"mool-katha-control-release-gate-v1\0"
EXPECTED_IG_METADATA = {"instagram": {"type": "reel", "shouldShareToFeed": True,
                                       "isAiGenerated": True}}


class ReleaseHold(RuntimeError):
    """No further release mutation is safe without a human reconciliation."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ReleaseHold(message)


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"{label} has a duplicate field")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeError, ValueError) as exc:
        raise ReleaseHold(f"{label} is invalid JSON") from exc
    require(type(value) is dict, f"{label} must be a JSON object")
    return value


def _read_regular(path: Path, limit: int, label: str) -> bytes:
    require(path.is_file() and not path.is_symlink(), f"{label} is unavailable")
    require(0 < path.stat().st_size <= limit, f"{label} is empty or oversized")
    data = path.read_bytes()
    require(len(data) == path.stat().st_size, f"{label} changed during read")
    return data


@dataclass(frozen=True)
class ReleasePolicy:
    """Trusted control-owned identity pins; construct only from reviewed code."""

    qa_repository: str
    qa_workflow_ref: str
    qa_workflow_sha: str
    qa_keys: dict[str, bytes]
    gate_workflow_ref: str
    gate_workflow_sha: str
    gate_keys: dict[str, bytes]
    youtube_channel_id: str
    youtube_handle: str
    instagram_channel_id: str
    instagram_handle: str
    cloudinary_cloud: str

    def __post_init__(self) -> None:
        require(self.qa_repository == "ashivam-dot/mool-katha-control", "QA repository pin is invalid")
        require(self.qa_workflow_ref.startswith(self.qa_repository + "/.github/workflows/")
                and "@refs/tags/" in self.qa_workflow_ref, "QA workflow must be an immutable control tag")
        require(SHA1.fullmatch(self.qa_workflow_sha) is not None, "QA workflow SHA pin is invalid")
        require(bool(self.qa_keys) and all(type(k) is str and KEY_ID.fullmatch(k) and
                                           type(v) is bytes and len(v) == 32
                                           for k, v in self.qa_keys.items()), "QA public keys are invalid")
        object.__setattr__(self, "qa_keys", MappingProxyType(dict(self.qa_keys)))
        require(self.gate_workflow_ref.startswith(self.qa_repository + "/.github/workflows/")
                and "@refs/tags/" in self.gate_workflow_ref,
                "gate workflow must be an immutable control tag")
        require(SHA1.fullmatch(self.gate_workflow_sha) is not None, "gate workflow SHA pin is invalid")
        require(bool(self.gate_keys) and all(type(k) is str and KEY_ID.fullmatch(k) and
                                             type(v) is bytes and len(v) == 32
                                             for k, v in self.gate_keys.items()),
                "gate public keys are invalid")
        require(not set(self.qa_keys.values()) & set(self.gate_keys.values()) and
                self.qa_workflow_ref != self.gate_workflow_ref,
                "QA and signer gate must use separate keys and workflows")
        object.__setattr__(self, "gate_keys", MappingProxyType(dict(self.gate_keys)))
        require(self.youtube_channel_id != self.instagram_channel_id,
                "YouTube and Instagram destinations must differ")
        require(bool(self.youtube_channel_id and self.instagram_channel_id
                     and self.youtube_handle and self.instagram_handle), "destination pins are incomplete")
        require(re.fullmatch(r"[A-Za-z0-9_-]+", self.cloudinary_cloud) is not None,
                "Cloudinary cloud pin is invalid")


@dataclass(frozen=True)
class ReleasePlan:
    """Reviewed control-owned intent for one candidate; no field comes from producer code."""

    episode_id: str
    source_commit: str
    archive_sha256: str
    video_sha256: str
    review_sha256: str
    due_at: datetime
    youtube_text: str
    instagram_text: str
    youtube_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        require(EPISODE.fullmatch(self.episode_id) is not None, "episode ID is invalid")
        require(SHA1.fullmatch(self.source_commit) is not None, "source commit pin is invalid")
        for name in ("archive_sha256", "video_sha256", "review_sha256"):
            require(SHA256.fullmatch(getattr(self, name)) is not None, f"{name} pin is invalid")
        require(self.due_at.tzinfo is not None and self.due_at.utcoffset() is not None,
                "posting slot has no timezone")
        local = self.due_at.astimezone(IST)
        require((local.hour, local.minute, local.second, local.microsecond) in
                ((19, 0, 0, 0), (20, 30, 0, 0)), "posting slot is not an approved IST slot")
        require(0 < len(self.youtube_text.encode("utf-8")) <= 5000 and
                0 < len(self.instagram_text) <= 2200, "caption or description length is invalid")
        metadata = self.youtube_metadata
        require(type(metadata) is dict and set(metadata) == {"youtube"} and
                type(metadata["youtube"]) is dict and
                set(metadata["youtube"]) == {"title", "categoryId", "privacy", "madeForKids",
                                             "notifySubscribers", "isAiGenerated", "embeddable"} and
                type(metadata["youtube"]["isAiGenerated"]) is bool and
                metadata["youtube"]["privacy"] == "public" and
                metadata["youtube"]["madeForKids"] is False and
                type(metadata["youtube"]["title"]) is str and
                0 < len(metadata["youtube"]["title"]) <= 100,
                "YouTube metadata or AI disclosure is incomplete")
        object.__setattr__(self, "youtube_metadata", copy.deepcopy(metadata))

    def intent_sha256(self, policy: ReleasePolicy) -> str:
        intent = {"episode_id": self.episode_id, "source_commit": self.source_commit,
                  "archive_sha256": self.archive_sha256, "video_sha256": self.video_sha256,
                  "review_sha256": self.review_sha256, "due_at": self.due_at.isoformat(),
                  "youtube_text": self.youtube_text, "instagram_text": self.instagram_text,
                  "youtube_metadata": self.youtube_metadata, "instagram_metadata": EXPECTED_IG_METADATA,
                  "youtube_channel_id": policy.youtube_channel_id,
                  "instagram_channel_id": policy.instagram_channel_id,
                  "youtube_handle": policy.youtube_handle, "instagram_handle": policy.instagram_handle,
                  "cloudinary_cloud": policy.cloudinary_cloud}
        return hashlib.sha256(json.dumps(intent, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class BufferPublisher(Protocol):
    def channels(self) -> list[dict[str, Any]]: ...
    def posts(self, channel_id: str) -> list[dict[str, Any]]: ...
    def post_detail(self, post_id: str, service: str) -> dict[str, Any]: ...
    def create(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class MediaHost(Protocol):
    def ensure_video(self, video: Path, public_id: str) -> str: ...


class DurableReleaseStore(ABC):
    """Control-owned store whose methods work across independent runner machines.

    `exclusive` must hold a cross-run lock for the whole release. `save` must
    synchronously commit and fsync or obtain equivalent durable acknowledgement
    before returning. A local runner file or process lock does not qualify.
    This prototype intentionally provides no production implementation.
    """

    @abstractmethod
    def exclusive(self, episode_id: str) -> AbstractContextManager[None]: ...

    @abstractmethod
    def load(self, episode_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def save(self, episode_id: str, document: dict[str, Any]) -> None: ...


def _source_blob(repo: Path, commit: str, path: str) -> bytes:
    try:
        entry = subprocess.run(["git", "-C", str(repo), "ls-tree", "-z", commit, "--", path],
                               capture_output=True, timeout=20, check=True).stdout
        require(entry.endswith(b"\0") and entry.count(b"\0") == 1,
                f"committed {path} is missing or ambiguous")
        header, actual = entry[:-1].split(b"\t", 1)
        mode, kind, oid = header.decode("ascii").split()
        require(actual.decode("utf-8") == path and mode in ("100644", "100755") and
                kind == "blob" and SHA1.fullmatch(oid) is not None,
                f"committed {path} is not an exact regular blob")
        size = int(subprocess.run(["git", "-C", str(repo), "cat-file", "-s", oid],
                                  capture_output=True, timeout=20, check=True).stdout)
        require(0 < size <= 20 * 1024 * 1024, f"committed {path} is oversized")
        blob = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", oid],
                              capture_output=True, timeout=20, check=True).stdout
        require(len(blob) == size, f"committed {path} changed while read")
        return blob
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError) as exc:
        raise ReleaseHold(f"committed {path} cannot be read safely") from exc


def _verify_frozen_source(repo: Path, artifact_dir: Path, plan: ReleasePlan,
                          review: dict[str, Any]) -> dict[str, str]:
    require(repo.is_dir() and (repo / ".git").exists(), "private source checkout is unavailable")
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, timeout=20, check=True).stdout.decode("ascii").strip()
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise ReleaseHold("private source commit cannot be read") from exc
    require(head == plan.source_commit, "private checkout differs from control source commit pin")
    hashes: dict[str, str] = {}
    base = f"content/episodes/{plan.episode_id}/"
    for filename, review_key in FROZEN_FILES.items():
        committed = _source_blob(repo, plan.source_commit, base + filename)
        saved = _read_regular(artifact_dir / filename, 20 * 1024 * 1024, filename)
        require(saved == committed, f"QA snapshot {filename} differs from exact source commit")
        digest = hashlib.sha256(committed).hexdigest()
        require(review[review_key] == digest,
                f"signed QA {review_key} differs from exact source bytes")
        hashes[filename] = digest
    return hashes


def _verify_gate_attestation(artifact_dir: Path, plan: ReleasePlan, policy: ReleasePolicy,
                             review: dict[str, Any], frozen: dict[str, str]) -> None:
    raw = _read_regular(artifact_dir / "release-gate-attestation.json", MAX_SIGNATURE_BYTES,
                        "signer gate attestation")
    signed = _json_object(raw, "signer gate attestation")
    envelope = _json_object(_read_regular(artifact_dir / "release-gate-signature.json",
                                          MAX_SIGNATURE_BYTES, "signer gate signature"),
                            "signer gate signature")
    require(set(envelope) == {"kind", "algorithm", "key_id", "attestation_sha256", "signature"} and
            envelope["kind"] == "control_release_gate_signature_v1" and
            envelope["algorithm"] == "Ed25519" and
            envelope["attestation_sha256"] == hashlib.sha256(raw).hexdigest(),
            "signer gate signature envelope is invalid")
    key_id = envelope["key_id"]
    require(type(key_id) is str and key_id in policy.gate_keys,
            "signer gate key is untrusted")
    try:
        encoded = envelope["signature"]
        require(type(encoded) is str, "signer gate signature encoding is invalid")
        signature = base64.b64decode(encoded, validate=True)
        require(len(signature) == 64 and base64.b64encode(signature).decode("ascii") == encoded,
                "signer gate signature encoding is invalid")
        Ed25519PublicKey.from_public_bytes(policy.gate_keys[key_id]).verify(signature, GATE_CONTEXT + raw)
    except (InvalidSignature, ValueError) as exc:
        raise ReleaseHold("signer gate signature is invalid") from exc
    require(set(signed) == {"kind", "decision", "episode_id", "source_commit", "archive_sha256",
                            "review_sha256", "review_signature_sha256", "video_sha256",
                            "frozen_sha256", "quality_observation_sha256", "gate_run"} and
            signed["kind"] == "control_release_gate_attestation_v1" and
            signed["decision"] == "approved" and signed["episode_id"] == plan.episode_id and
            signed["source_commit"] == plan.source_commit and
            signed["archive_sha256"] == plan.archive_sha256 and
            signed["review_sha256"] == plan.review_sha256 and
            signed["video_sha256"] == plan.video_sha256 and
            signed["frozen_sha256"] == frozen and
            signed["review_signature_sha256"] == hashlib.sha256(
                _read_regular(artifact_dir / "agent-release-signature.json", MAX_SIGNATURE_BYTES,
                              "QA signature")).hexdigest() and
            signed["quality_observation_sha256"] ==
            review["audio_review"]["quality_observation"]["sha256"],
            "signer gate attestation does not bind the exact reviewed candidate")
    run = signed["gate_run"]
    require(type(run) is dict and set(run) == {"system", "repository", "workflow_ref",
                                              "workflow_sha", "run_id", "run_attempt"} and
            run["system"] == "github_actions" and run["repository"] == policy.qa_repository and
            run["workflow_ref"] == policy.gate_workflow_ref and
            run["workflow_sha"] == policy.gate_workflow_sha and
            all(type(run[name]) is int and run[name] > 0 for name in ("run_id", "run_attempt")),
            "signer gate attestation is not from the trusted control workflow")


def _check_signature(artifact_dir: Path, plan: ReleasePlan, policy: ReleasePolicy) -> dict[str, Any]:
    review_raw = _read_regular(artifact_dir / "agent-release-review.json", MAX_REVIEW_BYTES, "QA review")
    signature_raw = _read_regular(artifact_dir / "agent-release-signature.json", MAX_SIGNATURE_BYTES,
                                  "QA signature")
    require(hashlib.sha256(review_raw).hexdigest() == plan.review_sha256,
            "QA review differs from control release pin")
    review = _json_object(review_raw, "QA review")
    signature = _json_object(signature_raw, "QA signature")
    require(set(signature) == {"kind", "algorithm", "key_id", "review_sha256", "signature"},
            "QA signature has unexpected fields")
    require(signature["kind"] == "agent_release_signature_v1" and signature["algorithm"] == "Ed25519",
            "QA signature protocol is invalid")
    key_id = signature["key_id"]
    require(type(key_id) is str and KEY_ID.fullmatch(key_id) is not None and key_id in policy.qa_keys,
            "QA signing key is untrusted")
    require(signature["review_sha256"] == plan.review_sha256, "QA signature names another review")
    encoded = signature["signature"]
    try:
        require(type(encoded) is str, "QA signature encoding is invalid")
        decoded = base64.b64decode(encoded, validate=True)
        require(len(decoded) == 64 and base64.b64encode(decoded).decode("ascii") == encoded,
                "QA signature encoding is invalid")
        Ed25519PublicKey.from_public_bytes(policy.qa_keys[key_id]).verify(decoded, CONTEXT + review_raw)
    except (InvalidSignature, ValueError) as exc:
        raise ReleaseHold("QA signature is invalid") from exc
    require(review.get("kind") == "agent_episode_qa_v1" and review.get("episode_id") == plan.episode_id,
            "signed QA review names another episode or protocol")
    require(set(review) == {"kind", "episode_id", "production_agent_id", "qa_agent_id", "qa_run",
                            "script_sha256", "spec_sha256", "video_sha256", "qc_sha256",
                            "evidence_sha256", "manifest_sha256", "claim_findings", "asset_findings",
                            "audio_review", "video_review", "qc_warning_dispositions", "release_review"},
            "signed QA review schema is incomplete")
    for name in ("script_sha256", "spec_sha256", "qc_sha256", "evidence_sha256", "manifest_sha256"):
        require(type(review[name]) is str and SHA256.fullmatch(review[name]) is not None,
                f"signed QA {name} is invalid")
    require(review.get("video_sha256") == plan.video_sha256,
            "signed QA review names another final MP4")
    run = review.get("qa_run")
    require(type(run) is dict and set(run) == {"system", "repository", "workflow_ref", "workflow_sha",
                                               "run_id", "run_attempt", "completed_at", "review_model_calls"},
            "signed QA run provenance is incomplete")
    require(run["system"] == "github_actions" and run["repository"] == policy.qa_repository and
            run["workflow_ref"] == policy.qa_workflow_ref and
            run["workflow_sha"] == policy.qa_workflow_sha and
            all(type(run[k]) is int and run[k] > 0 for k in ("run_id", "run_attempt")) and
            type(run["review_model_calls"]) is list and len(run["review_model_calls"]) >= 2 and
            all(type(call) is dict and set(call) ==
                {"provider", "model", "model_version", "request_id"} and
                all(type(value) is str and len(value.strip()) >= 4 for value in call.values())
                for call in run["review_model_calls"]),
            "signed QA run is not the trusted independent workflow")
    expected_qa = (f"agent:github_actions/{policy.qa_repository}/"
                   f"{run['run_id']}/{run['run_attempt']}")
    require(review.get("qa_agent_id") == expected_qa and
            type(review.get("production_agent_id")) is str and
            review["production_agent_id"] != expected_qa and
            (re.fullmatch(r"agent:modal/ashivam-dot/mool-katha/[A-Za-z0-9._-]{4,}",
                          review["production_agent_id"]) is not None or
             re.fullmatch(r"agent:github_actions/ashivam-dot/mool-katha/[1-9][0-9]*/[1-9][0-9]*",
                          review["production_agent_id"]) is not None),
            "signed QA identity is not independent of production")
    verdict = review.get("release_review")
    require(type(verdict) is dict and verdict.get("agent_id") == expected_qa and
            verdict.get("method") == "agent_independent_release_review" and
            verdict.get("decision") == "approved" and verdict.get("unresolved_items") == [] and
            all(verdict.get(flag) is True for flag in
                ("verified_all_claims", "verified_all_assets", "verified_asr_and_decoded_frames",
                 "verified_no_unresolved_concerns")),
            "signed QA release verdict is not approved")
    audio = review.get("audio_review")
    reference = audio.get("quality_observation") if type(audio) is dict else None
    require(type(reference) is dict and set(reference) == {"file", "sha256"} and
            reference["file"] == "agent-audio-quality-observation.json" and
            type(reference["sha256"]) is str and SHA256.fullmatch(reference["sha256"]) is not None,
            "signed QA review has no exact voice-quality observation reference")
    quality_raw = _read_regular(artifact_dir / reference["file"], MAX_QUALITY_BYTES,
                                "voice-quality observation")
    require(hashlib.sha256(quality_raw).hexdigest() == reference["sha256"],
            "voice-quality observation changed after signed QA")
    quality = _json_object(quality_raw, "voice-quality observation")
    require(quality.get("kind") == "full_final_audio_quality_model_observation_v1" and
            quality.get("basis") == "Gemini model observation of actual full final audio; not human listening" and
            quality.get("input_video_sha256") == plan.video_sha256,
            "voice-quality observation does not describe the signed final MP4")
    return review


def _read_archive_video(archive: Path, plan: ReleasePlan, output: Path) -> None:
    require(archive.is_file() and not archive.is_symlink(), "private media archive is unavailable")
    require(0 < archive.stat().st_size <= MAX_ARCHIVE_BYTES, "private media archive is oversized or empty")
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    require(digest.hexdigest() == plan.archive_sha256,
            "private archive differs from control release pin")
    wanted = f"content/episodes/{plan.episode_id}/{plan.episode_id}.mp4"
    found = False
    try:
        with tarfile.open(archive, mode="r:") as bundle:
            for index, member in enumerate(bundle):
                require(index < 500, "private archive has too many members")
                name = member.name.removeprefix("./")
                require(name and not name.startswith("/") and ".." not in Path(name).parts,
                        "private archive has an unsafe member path")
                require(member.isfile() or member.isdir(), "private archive has a link or special member")
                if name != wanted:
                    continue
                require(not found and member.isfile() and not member.issparse() and
                        0 < member.size <= MAX_VIDEO_BYTES,
                        "private archive has an invalid or duplicate MP4")
                found = True
                source = bundle.extractfile(member)
                require(source is not None, "private MP4 cannot be read")
                with output.open("xb") as target:
                    remaining = member.size
                    video_hash = hashlib.sha256()
                    while remaining:
                        block = source.read(min(1024 * 1024, remaining))
                        require(bool(block), "private MP4 is truncated")
                        target.write(block)
                        video_hash.update(block)
                        remaining -= len(block)
                require(video_hash.hexdigest() == plan.video_sha256,
                        "private MP4 differs from signed QA and release plan")
    except (OSError, tarfile.TarError, EOFError) as exc:
        raise ReleaseHold("private archive cannot be read safely") from exc
    require(found, "private archive has no final MP4")
    with archive.open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    require(digest.hexdigest() == plan.archive_sha256, "private archive changed during release")


def _media_identity(url: str, policy: ReleasePolicy, plan: ReleasePlan) -> bool:
    if type(url) is not str:
        return False
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "res.cloudinary.com" or parsed.query or
            parsed.fragment or "%2f" in parsed.path.lower()):
        return False
    path = unquote(parsed.path)
    prefix = f"/{policy.cloudinary_cloud}/video/upload/"
    if not path.startswith(prefix):
        return False
    suffix = path[len(prefix):]
    suffix = re.sub(r"^v[0-9]+/", "", suffix)
    return suffix == f"mool-katha/{plan.episode_id}-{plan.video_sha256}.mp4"


def _channel_ready(found: list[dict[str, Any]], policy: ReleasePolicy, service: str) -> str:
    channel_id = getattr(policy, f"{service}_channel_id")
    matches = [channel for channel in found if channel.get("id") == channel_id]
    require(len(matches) == 1, f"{service} destination is missing or duplicated")
    channel = matches[0]
    require(channel.get("service") == service and
            all(channel.get(flag) is False for flag in
                ("isDisconnected", "isLocked", "isQueuePaused")),
            f"{service} destination is disconnected, locked, paused, or mislabeled")
    username = channel.get("name")
    require(type(username) is str and username and username.strip() == username and
            username.count("@") <= 1 and ("@" not in username or username.startswith("@")),
            f"{service} destination has no canonical username")
    handle = getattr(policy, f"{service}_handle").casefold().lstrip("@")
    require(username.casefold().lstrip("@") == handle,
            f"{service} destination handle differs from trusted policy")
    return channel_id


def _post_identity(post: dict[str, Any], channel_id: str, service: str, url: str,
                   text: str, due_at: datetime) -> bool:
    try:
        actual_due = datetime.fromisoformat(post["dueAt"])
    except (KeyError, TypeError, ValueError):
        return False
    videos = [a.get("source") for a in post.get("assets", []) if type(a) is dict]
    return (post.get("channelId") == channel_id and post.get("channelService") == service and
            post.get("status") == "scheduled" and
            actual_due.tzinfo is not None and actual_due.astimezone(timezone.utc) ==
            due_at.astimezone(timezone.utc) and post.get("text") == text and videos == [url])


def _reconcile(buffer: BufferPublisher, policy: ReleasePolicy, plan: ReleasePlan,
               media_url: str, receipts: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for service in ("youtube", "instagram"):
        channel_id = getattr(policy, f"{service}_channel_id")
        listed = buffer.posts(channel_id)
        require(type(listed) is list and len(listed) <= MAX_POSTS, f"{service} Buffer listing is incomplete")
        require(all(type(post) is dict and type(post.get("id")) is str and post["id"] for post in listed),
                f"{service} Buffer listing has malformed posts")
        ids = [post["id"] for post in listed]
        require(len(ids) == len(set(ids)),
                f"{service} Buffer listing has duplicate or malformed posts")
        expected_text = getattr(plan, f"{service}_text")
        candidates = []
        for post in listed:
            assets = post.get("assets")
            require(type(assets) is list, f"{service} Buffer post has ambiguous assets")
            sources = [a.get("source") for a in assets if type(a) is dict]
            related = any(_media_identity(source, policy, plan) for source in sources)
            due = post.get("dueAt")
            try:
                slot = datetime.fromisoformat(due).astimezone(timezone.utc) == plan.due_at.astimezone(timezone.utc)
            except (TypeError, ValueError):
                slot = False
            if related or slot or post.get("id") == receipts.get(service):
                candidates.append(post)
        require(len(candidates) <= 1, f"{service} Buffer has duplicate or conflicting posts")
        if not candidates:
            require(service not in receipts, f"{service} receipt has no matching Buffer post")
            continue
        post = candidates[0]
        require(_post_identity(post, channel_id, service, media_url, expected_text, plan.due_at),
                f"{service} Buffer post differs from signed release intent")
        require(receipts.get(service) == post["id"],
                f"{service} Buffer post has no trusted create receipt; reconcile manually")
        detail = buffer.post_detail(post["id"], service)
        require(type(detail) is dict and detail.get("metadata") ==
                (plan.youtube_metadata if service == "youtube" else EXPECTED_IG_METADATA),
                f"{service} Buffer AI and destination metadata differs from signed release intent")
        require(_post_identity(detail, channel_id, service, media_url, expected_text, plan.due_at),
                f"{service} Buffer detail differs from scheduled post")
        result[service] = post["id"]
    return result


def _load_journal(store: DurableReleaseStore, plan: ReleasePlan,
                  policy: ReleasePolicy) -> dict[str, Any]:
    try:
        saved = store.load(plan.episode_id)
    except Exception as exc:
        raise ReleaseHold("durable release journal could not be read") from exc
    if saved is None:
        return {"kind": "control_release_journal_v1", "episode_id": plan.episode_id,
                "intent_sha256": plan.intent_sha256(policy), "posts": {}, "create": None}
    require(type(saved) is dict and set(saved) ==
            {"kind", "episode_id", "intent_sha256", "posts", "create"} and
            saved["kind"] == "control_release_journal_v1" and
            saved["episode_id"] == plan.episode_id and
            saved["intent_sha256"] == plan.intent_sha256(policy),
            "durable release journal is bound to another candidate")
    posts = saved["posts"]
    require(type(posts) is dict and set(posts) <= {"youtube", "instagram"} and
            all(type(value) is str and value for value in posts.values()),
            "durable release journal has malformed post IDs")
    create = saved["create"]
    require(create is None or (type(create) is dict and set(create) ==
            {"service", "payload_sha256", "state"} and
            create["service"] in ("youtube", "instagram") and
            type(create["payload_sha256"]) is str and
            SHA256.fullmatch(create["payload_sha256"]) is not None and
            create["state"] in ("create_started", "unknown_outcome")),
            "durable release journal has malformed create state")
    require(create is None, "a Buffer create may have succeeded; reconcile its durable unknown outcome manually")
    return copy.deepcopy(saved)


def _save_journal(store: DurableReleaseStore, document: dict[str, Any]) -> None:
    episode_id = document["episode_id"]
    try:
        store.save(episode_id, copy.deepcopy(document))
        confirmed = store.load(episode_id)
    except Exception as exc:
        raise ReleaseHold("durable release journal did not confirm its write") from exc
    require(confirmed == document, "durable release journal did not read back its write")


def _release_pair_locked(plan: ReleasePlan, policy: ReleasePolicy, artifact_dir: Path, archive: Path,
                         source_repo: Path, store: DurableReleaseStore, buffer: BufferPublisher,
                         host: MediaHost, *, execute: bool, now: datetime | None) -> dict[str, str]:
    """Create or verify the exact YouTube/Instagram pair; hold every ambiguous state.

    The injected store must hold the cross-run lock and durably acknowledge
    every state transition before this method calls Buffer create.
    """
    require(execute is True, "release executor is dormant without explicit execution")
    require(plan.episode_id != "ep003", "ep003 pilot is outside this release prototype")
    clock = now or datetime.now(timezone.utc)
    require(clock.tzinfo is not None and plan.due_at - clock > timedelta(hours=2),
            "posting slot must leave more than two hours for safe reconciliation")
    review = _check_signature(artifact_dir, plan, policy)
    frozen = _verify_frozen_source(source_repo, artifact_dir, plan, review)
    _verify_gate_attestation(artifact_dir, plan, policy, review, frozen)
    journal = _load_journal(store, plan, policy)
    receipts = journal["posts"]
    channels = buffer.channels()
    require(type(channels) is list, "Buffer destinations are unavailable")
    for service in ("youtube", "instagram"):
        _channel_ready(channels, policy, service)
    with tempfile.TemporaryDirectory(prefix="mool-katha-release-") as scratch:
        video = Path(scratch) / f"{plan.episode_id}.mp4"
        _read_archive_video(archive, plan, video)
        public_id = f"mool-katha/{plan.episode_id}-{plan.video_sha256}"
        media_url = host.ensure_video(video, public_id)
        require(_media_identity(media_url, policy, plan), "host returned another media identity")
        state = _reconcile(buffer, policy, plan, media_url, receipts)
        for service in ("youtube", "instagram"):
            if service in state:
                continue
            payload = {"channelId": getattr(policy, f"{service}_channel_id"),
                       "text": getattr(plan, f"{service}_text"),
                       "schedulingType": "automatic", "mode": "customScheduled",
                       "dueAt": plan.due_at.isoformat(),
                       "assets": [{"video": {"url": media_url}}],
                       "metadata": (copy.deepcopy(plan.youtube_metadata) if service == "youtube"
                                    else copy.deepcopy(EXPECTED_IG_METADATA))}
            payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            journal["create"] = {"service": service, "payload_sha256": payload_hash,
                                 "state": "create_started"}
            _save_journal(store, journal)
            try:
                created = buffer.create(payload)
                require(type(created) is dict and type(created.get("id")) is str and created["id"],
                        f"{service} Buffer create response is uncertain")
            except Exception as exc:
                journal["create"]["state"] = "unknown_outcome"
                _save_journal(store, journal)
                raise ReleaseHold(f"{service} Buffer create outcome is unknown; reconcile manually") from exc
            receipts[service] = created["id"]
            journal["create"] = None
            _save_journal(store, journal)
            state = _reconcile(buffer, policy, plan, media_url, receipts)
        require(set(state) == {"youtube", "instagram"}, "Buffer did not confirm the complete pair")
        return state


def release_pair(plan: ReleasePlan, policy: ReleasePolicy, artifact_dir: Path, archive: Path,
                 source_repo: Path, store: DurableReleaseStore | None, buffer: BufferPublisher,
                 host: MediaHost, *, execute: bool = False,
                 now: datetime | None = None) -> dict[str, str]:
    """Require an injected cross-run lock and durable journal before any mutation."""
    require(execute is True, "release executor is dormant without explicit execution")
    require(plan.episode_id != "ep003", "ep003 pilot is outside this release prototype")
    require(isinstance(store, DurableReleaseStore),
            "cross-run durable release lock and journal are required")
    try:
        with store.exclusive(plan.episode_id):
            return _release_pair_locked(plan, policy, artifact_dir, archive, source_repo,
                                        store, buffer, host, execute=execute, now=now)
    except ReleaseHold:
        raise
    except Exception as exc:
        raise ReleaseHold("cross-run release lock failed or release state is uncertain") from exc
