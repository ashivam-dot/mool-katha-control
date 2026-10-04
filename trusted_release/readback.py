"""Read Buffer's exact pair after pinned code schedules a signed episode.

This module runs from the independent control checkout. It reads the producer
checkout as data and uses the pinned publisher only for read queries and its
existing signed-evidence and hosted-media checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit


EPISODE = re.compile(r"ep[0-9]{3}\Z")
RECEIPT = "control-platform-readback.json"


class ReadbackHold(RuntimeError):
    """The accepted pair cannot yet be proved by independent Buffer reads."""


def _unique(pairs: list[tuple[str, Any]]) -> dict:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReadbackHold("source JSON has a duplicate key")
        result[key] = value
    return result


def _object(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ReadbackHold(f"{path.name} is missing, linked, or oversized")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReadbackHold(f"{path.name} is invalid JSON") from None
    if not isinstance(value, dict):
        raise ReadbackHold(f"{path.name} is not an object")
    return value


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ReadbackHold("signed final media is missing or linked")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ReadbackHold("post due time is missing")
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ReadbackHold("post due time is invalid") from None
    if when.utcoffset() is None:
        raise ReadbackHold("post due time has no timezone")
    return when.astimezone(timezone.utc)


def _folder(root: Path, episode_id: str) -> Path:
    if not isinstance(episode_id, str) or EPISODE.fullmatch(episode_id) is None:
        raise ReadbackHold("release result has an invalid episode ID")
    content = root / "content"
    episodes = content / "episodes"
    folder = episodes / episode_id
    if (content.is_symlink() or episodes.is_symlink() or folder.is_symlink()
            or not folder.is_dir() or not folder.resolve().is_relative_to(root.resolve())):
        raise ReadbackHold("signed episode folder is missing or linked")
    return folder


def _platform_link(value: object, service: str) -> bool:
    if not isinstance(value, str):
        return False
    try:
        url = urlsplit(value)
        port = url.port
    except ValueError:
        return False
    if url.scheme != "https" or url.username or url.password or port or url.fragment:
        return False
    parts = url.path.strip("/").split("/")
    if service == "youtube":
        return ((url.hostname in {"youtube.com", "www.youtube.com"}
                 and not url.query and len(parts) == 2 and parts[0] == "shorts"
                 and re.fullmatch(r"[A-Za-z0-9_-]{11}", parts[1]) is not None)
                or (url.hostname in {"youtube.com", "www.youtube.com"}
                    and url.path == "/watch"
                    and re.fullmatch(r"v=[A-Za-z0-9_-]{11}", url.query) is not None)
                or (url.hostname == "youtu.be" and not url.query and len(parts) == 1
                    and re.fullmatch(r"[A-Za-z0-9_-]{11}", parts[0]) is not None))
    return (not url.query and url.hostname in {"instagram.com", "www.instagram.com"}
            and len(parts) == 2 and parts[0] in {"reel", "p"}
            and re.fullmatch(r"[A-Za-z0-9_-]+", parts[1]) is not None)


def _post(publisher: Any, rows: list[dict], *, episode_id: str, service: str,
          channel_id: str, post_id: str, due_at: str, media_url: str, text: str) -> dict:
    try:
        matches = publisher.episode_posts(rows, episode_id, channel_id)
    except Exception:
        raise ReadbackHold(f"{service} episode post scan is ambiguous") from None
    if len(matches) != 1:
        raise ReadbackHold(f"{service} has {len(matches)} posts for this episode; expected one")
    row = matches[0]
    assets = row.get("assets")
    if (row.get("id") != post_id or row.get("channelId") != channel_id
            or row.get("channelService") != service or row.get("text") != text
            or not isinstance(assets, list) or len(assets) != 1
            or not isinstance(assets[0], dict) or assets[0].get("source") != media_url
            or row.get("video") != media_url):
        raise ReadbackHold(f"{service} post differs from the signed receipt, destination, text, or video")
    if _time(row.get("dueAt")) != _time(due_at):
        raise ReadbackHold(f"{service} post has a different due time")
    status = row.get("status")
    if status not in ("scheduled", "sent"):
        raise ReadbackHold(f"{service} post is not scheduled or sent")
    if status == "sent":
        if not _platform_link(row.get("externalLink"), service):
            raise ReadbackHold(f"{service} sent post has no valid public link")
        if _time(row.get("sentAt")) > datetime.now(timezone.utc):
            raise ReadbackHold(f"{service} sent post has a future sent time")
    return {"post_id": post_id, "channel_id": channel_id, "status": status,
            "due_at": _time(row["dueAt"]).isoformat(),
            "public_link": row.get("externalLink") if status == "sent" else None}


def verify_pair(root: Path, episode_id: str, *, publisher: Any, policy: Any,
                evidence_gate: Callable[..., None], youtube_id: str,
                instagram_id: str, youtube_rows: list[dict], instagram_rows: list[dict]) -> dict:
    """Bind two live Buffer reads to one signed review and exact hosted MP4."""
    folder = _folder(root, episode_id)
    record = _object(folder / "publish.json")
    review_path = folder / "agent-release-review.json"
    review = _object(review_path)
    _object(folder / "agent-release-signature.json")
    video_hash = _sha(folder / f"{episode_id}.mp4")
    if review.get("video_sha256") != video_hash:
        raise ReadbackHold("signed review names different final media")
    try:
        evidence_gate(folder, policy=policy, studio_root=root)
    except Exception:
        raise ReadbackHold("signed episode evidence failed the pinned release gate") from None
    media_url = record.get("media_url")
    expected_public_id = f"mool-katha/{episode_id}-{video_hash}"
    try:
        parsed_id = publisher._episode_media_public_id(media_url, episode_id)
    except Exception:
        raise ReadbackHold("hosted media URL is invalid") from None
    if (record.get("id") != episode_id or not isinstance(media_url, str)
            or record.get("media_public_id") != expected_public_id
            or parsed_id != expected_public_id):
        raise ReadbackHold("published media is not the full-hash signed final MP4")
    crossposts = record.get("crossposts")
    companion = crossposts.get("instagram") if isinstance(crossposts, dict) else None
    if (not isinstance(companion, dict) or not isinstance(record.get("buffer_post_id"), str)
            or not record["buffer_post_id"] or not isinstance(companion.get("id"), str)
            or not companion["id"] or not isinstance(companion.get("due_at"), str)):
        raise ReadbackHold("both post IDs are not recorded")
    try:
        spec = publisher.ShortSpec.load(folder / "short.yaml")
        manifest = _object(folder / "work/manifest.json")
        youtube_text = publisher.description(spec, manifest)
        instagram_text = publisher.caption(spec)
        publisher.require_reviewed_hosted_video(folder, media_url, policy=policy)
    except Exception:
        raise ReadbackHold("hosted signed media or scheduled copy could not be verified") from None
    youtube = _post(publisher, youtube_rows, episode_id=episode_id, service="youtube",
                    channel_id=youtube_id, post_id=record["buffer_post_id"],
                    due_at=record.get("due_at"), media_url=media_url, text=youtube_text)
    instagram = _post(publisher, instagram_rows, episode_id=episode_id, service="instagram",
                      channel_id=instagram_id, post_id=companion["id"],
                      due_at=companion.get("due_at"), media_url=media_url, text=instagram_text)
    return {"schema": "mool_katha_control_platform_readback_v1", "episode_id": episode_id,
            "video_sha256": video_hash, "review_sha256": _sha(review_path),
            "signature_sha256": _sha(folder / "agent-release-signature.json"),
            "publish_sha256": _sha(folder / "publish.json"),
            "media_url": media_url, "verified_at": datetime.now(timezone.utc).isoformat(),
            "posts": {"youtube": youtube, "instagram": instagram}}


def _write_receipt(path: Path, receipt: dict) -> None:
    """Expose only a complete receipt, and never replace an existing one."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".control-platform-readback-", suffix=".tmp",
                                         delete=False) as output:
            temporary = Path(output.name)
            json.dump(receipt, output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    except FileExistsError:
        raise ReadbackHold("platform readback receipt already exists") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_after_release(root: Path, result: dict, *, publisher: Any, policy: Any,
                         evidence_gate: Callable[..., None], youtube_id: str,
                         instagram_id: str) -> list[dict]:
    """Verify new or previously unconfirmed pairs, then write bounded receipts."""
    root = Path(root).resolve()
    if not isinstance(result, dict) or result.get("status") not in (
            "scheduled", "instagram_repaired", "no_signed_candidate"):
        raise ReadbackHold("pinned release returned an unexpected status")
    if not youtube_id or not instagram_id or youtube_id == instagram_id:
        raise ReadbackHold("exact Buffer destinations are unavailable")
    current = result.get("episode_id") if result["status"] != "no_signed_candidate" else None
    if current is not None:
        _folder(root, current)
    targets = []
    for folder in sorted((root / "content/episodes").glob("ep[0-9][0-9][0-9]")):
        if ((folder / "publish.json").is_file() and
                (folder / "agent-release-signature.json").is_file() and
                not (folder / RECEIPT).exists()):
            targets.append(folder.name)
    if current is not None and current not in targets:
        raise ReadbackHold("newly released episode has a preexisting or missing readback state")
    if not targets:
        return []
    try:
        youtube_rows = publisher.posts(channel_id=youtube_id)
        instagram_rows = publisher.posts(channel_id=instagram_id)
    except Exception:
        raise ReadbackHold("Buffer post readback query failed") from None
    if not isinstance(youtube_rows, list) or not isinstance(instagram_rows, list):
        raise ReadbackHold("Buffer post readback is malformed")
    receipts = [verify_pair(root, episode_id, publisher=publisher, policy=policy,
                            evidence_gate=evidence_gate, youtube_id=youtube_id,
                            instagram_id=instagram_id, youtube_rows=youtube_rows,
                            instagram_rows=instagram_rows) for episode_id in targets]
    for receipt in receipts:
        path = _folder(root, receipt["episode_id"]) / RECEIPT
        _write_receipt(path, receipt)
    return [{"episode_id": item["episode_id"], "video_sha256": item["video_sha256"],
             "youtube_post_id": item["posts"]["youtube"]["post_id"],
             "instagram_post_id": item["posts"]["instagram"]["post_id"]} for item in receipts]
