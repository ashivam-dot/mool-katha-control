"""Release one signed, exact-byte candidate through a narrowly injected publisher.

This module imports no producer code. The policy and plan must come from reviewed
control code, never from an episode, model output, or production repository.
There is deliberately no command-line entry point or automatic workflow.
"""

from __future__ import annotations

import base64
import copy
import fcntl
import hashlib
import json
import os
import re
import tarfile
import tempfile
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
        require(bool(self.qa_keys) and all(KEY_ID.fullmatch(k) and len(v) == 32
                                           for k, v in self.qa_keys.items()), "QA public keys are invalid")
        object.__setattr__(self, "qa_keys", MappingProxyType(dict(self.qa_keys)))
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
            not any(channel.get(flag) for flag in ("isDisconnected", "isLocked", "isQueuePaused")),
            f"{service} destination is disconnected, locked, paused, or mislabeled")
    identity = " ".join(str(channel.get(k) or "") for k in ("name", "displayName")).lower()
    handle = getattr(policy, f"{service}_handle").lower().lstrip("@")
    require(re.search(rf"(?<![a-z0-9._])@?{re.escape(handle)}(?![a-z0-9._])", identity) is not None,
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


def _load_receipts(path: Path, plan: ReleasePlan, policy: ReleasePolicy) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = _read_regular(path, 16 * 1024, "release receipt")
    saved = _json_object(raw, "release receipt")
    require(saved.get("kind") == "control_release_receipt_v1" and
            saved.get("episode_id") == plan.episode_id and
            saved.get("intent_sha256") == plan.intent_sha256(policy) and
            set(saved) == {"kind", "episode_id", "intent_sha256", "posts"},
            "release receipt is bound to another candidate")
    posts = saved["posts"]
    require(type(posts) is dict and set(posts) <= {"youtube", "instagram"} and
            all(type(v) is str and bool(v) for v in posts.values()), "release receipt is malformed")
    return posts


def _save_receipts(path: Path, plan: ReleasePlan, policy: ReleasePolicy,
                   posts: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), "release receipt path is a symlink")
    payload = {"kind": "control_release_receipt_v1", "episode_id": plan.episode_id,
               "intent_sha256": plan.intent_sha256(policy), "posts": posts}
    descriptor, temporary = tempfile.mkstemp(prefix=".release-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(payload, target, sort_keys=True, separators=(",", ":"))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _release_pair_locked(plan: ReleasePlan, policy: ReleasePolicy, artifact_dir: Path, archive: Path,
                         receipt_path: Path, buffer: BufferPublisher, host: MediaHost, *,
                         execute: bool, now: datetime | None) -> dict[str, str]:
    """Create or verify the exact YouTube/Instagram pair; hold every ambiguous state.

    The caller must serialize calls for a plan and persist `receipt_path` in an
    independently controlled durable store. A local file alone is insufficient
    for crash recovery across ephemeral workers.
    """
    require(execute is True, "release executor is dormant without explicit execution")
    require(plan.episode_id != "ep003", "ep003 pilot is outside this release prototype")
    clock = now or datetime.now(timezone.utc)
    require(clock.tzinfo is not None and plan.due_at - clock > timedelta(hours=2),
            "posting slot must leave more than two hours for safe reconciliation")
    _check_signature(artifact_dir, plan, policy)
    receipts = _load_receipts(receipt_path, plan, policy)
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
            created = buffer.create(payload)
            require(type(created) is dict and type(created.get("id")) is str and created["id"],
                    f"{service} Buffer create response is uncertain")
            receipts[service] = created["id"]
            _save_receipts(receipt_path, plan, policy, receipts)
            state = _reconcile(buffer, policy, plan, media_url, receipts)
        require(set(state) == {"youtube", "instagram"}, "Buffer did not confirm the complete pair")
        return state


def release_pair(plan: ReleasePlan, policy: ReleasePolicy, artifact_dir: Path, archive: Path,
                 receipt_path: Path, buffer: BufferPublisher, host: MediaHost, *, execute: bool = False,
                 now: datetime | None = None) -> dict[str, str]:
    """Hold a local exclusive lock over verification and the whole Buffer pair.

    A future workflow also needs a control-owned cross-run lock and durable
    receipt storage. The local lock only protects processes sharing this path.
    """
    require(execute is True, "release executor is dormant without explicit execution")
    require(plan.episode_id != "ep003", "ep003 pilot is outside this release prototype")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = receipt_path.with_name(receipt_path.name + ".lock")
    require(not lock_path.is_symlink(), "release lock path is a symlink")
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReleaseHold("another release attempt holds this candidate lock") from exc
        return _release_pair_locked(plan, policy, artifact_dir, archive, receipt_path,
                                    buffer, host, execute=execute, now=now)
