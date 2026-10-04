"""Read producer receipts as data and watch both exact Buffer posts after due time.

The producer checkout supplies no executable code to the control monitor. A
successful scheduled readback is only the first receipt: after the due time,
both live Buffer rows must be sent and carry canonical public links.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

from trusted_release.readback import _platform_link


EPISODE = re.compile(r"ep[0-9]{3}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
MAX_EPISODES = 1000
UTC = timezone.utc


class SignedMonitorError(ValueError):
    """A producer data tree cannot be inspected safely."""


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SignedMonitorError("producer JSON has duplicate keys")
        result[key] = value
    return result


def _object(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise SignedMonitorError("producer receipt is missing, linked, or oversized")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    except (OSError, UnicodeError, ValueError):
        raise SignedMonitorError("producer receipt is malformed") from None
    if not isinstance(value, dict):
        raise SignedMonitorError("producer receipt is not an object")
    return value


def _time(value: object) -> datetime:
    if not isinstance(value, str):
        raise SignedMonitorError("timestamp is missing")
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise SignedMonitorError("timestamp is invalid") from None
    if when.utcoffset() is None:
        raise SignedMonitorError("timestamp has no time zone")
    return when.astimezone(UTC)


def _folders(root: Path) -> list[Path]:
    if not root or root.is_symlink() or not root.is_dir():
        raise SignedMonitorError("producer checkout is unavailable")
    content = root / "content"
    episodes = content / "episodes"
    if content.is_symlink() or episodes.is_symlink() or not episodes.is_dir():
        raise SignedMonitorError("producer episodes directory is unavailable")
    folders = []
    for folder in sorted(episodes.iterdir()):
        if EPISODE.fullmatch(folder.name) is None:
            continue
        if folder.is_symlink() or not folder.is_dir():
            raise SignedMonitorError("producer episode directory is linked or invalid")
        signature = folder / "agent-release-signature.json"
        if signature.is_symlink():
            raise SignedMonitorError("producer signature is linked")
        if signature.is_file():
            _object(signature)
            folders.append(folder)
    if len(folders) > MAX_EPISODES:
        raise SignedMonitorError("signed producer episodes exceed monitor limit")
    return folders


def _media(record: dict, episode_id: str, video_sha: str) -> bool:
    if SHA256.fullmatch(video_sha) is None:
        return False
    public_id = f"mool-katha/{episode_id}-{video_sha}"
    if record.get("media_public_id") != public_id:
        return False
    url = record.get("media_url")
    if not isinstance(url, str):
        return False
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return False
    path = unquote(parsed.path)
    return (parsed.scheme == "https" and parsed.hostname == "res.cloudinary.com"
            and not parsed.username and not parsed.password and not port
            and not parsed.query and not parsed.fragment
            and re.fullmatch(rf"/mw0oh0v8/video/upload/(?:v[0-9]+/)?{re.escape(public_id)}\.mp4", path)
            is not None)


def _receipt(folder: Path, record: dict, episode_id: str, youtube_due: datetime,
             instagram_due: datetime, youtube_id: str, instagram_id: str, now: datetime) -> bool:
    try:
        receipt = _object(folder / "control-platform-readback.json")
        posts = receipt["posts"]
        video_sha = receipt["video_sha256"]
        if (receipt.get("schema") != "mool_katha_control_platform_readback_v1"
                or receipt.get("episode_id") != episode_id
                or receipt.get("media_url") != record.get("media_url")
                or not isinstance(posts, dict) or not isinstance(video_sha, str)
                or not _media(record, episode_id, video_sha)
                or _time(receipt.get("verified_at")) > now + timedelta(minutes=5)):
            return False
        for service, post_id, channel_id, due in (
                ("youtube", record["buffer_post_id"], youtube_id, youtube_due),
                ("instagram", record["crossposts"]["instagram"]["id"], instagram_id, instagram_due)):
            post = posts[service]
            if (not isinstance(post, dict) or post.get("post_id") != post_id
                    or post.get("channel_id") != channel_id
                    or _time(post.get("due_at")) != due
                    or post.get("status") not in ("scheduled", "sent")):
                return False
        return True
    except (SignedMonitorError, KeyError, TypeError):
        return False


def _post(rows: list[dict], *, service: str, channel_id: str, post_id: str,
          media_url: str, due: datetime, now: datetime) -> tuple[dict, list[str]]:
    issues = []
    matches = [row for row in rows if row.get("id") == post_id]
    summary = {"post_id": post_id, "status": "missing", "public_link": None}
    if len(matches) != 1:
        issues.append(f"{service} exact Buffer post is missing or duplicated")
        return summary, issues
    row = matches[0]
    summary["status"] = row.get("status") if row.get("status") in ("scheduled", "sent", "error", "draft") else "unknown"
    if (row.get("channel_id") != channel_id or row.get("service") != service
            or row.get("media_sources") != [media_url]):
        issues.append(f"{service} Buffer destination or video differs from signed release")
    try:
        if _time(row.get("due_at")) != due:
            issues.append(f"{service} Buffer due time differs from signed release")
    except SignedMonitorError:
        issues.append(f"{service} Buffer due time is invalid")
    if row.get("status") != "sent":
        issues.append(f"{service} Buffer post is not sent after due time")
    else:
        try:
            if _time(row.get("sent_at")) > now + timedelta(minutes=5):
                raise SignedMonitorError("future sent time")
        except SignedMonitorError:
            issues.append(f"{service} Buffer sent time is invalid")
        link = row.get("external_link")
        if not _platform_link(link, service):
            issues.append(f"{service} Buffer sent post has no valid public link")
        else:
            summary["public_link"] = link
    return summary, issues


def inspect(now: datetime, source_root: Path, posts: dict[str, list[dict]],
            youtube_id: str, instagram_id: str) -> dict:
    """Return one bounded issue per missing receipt or post condition."""
    folders = _folders(source_root)
    issues: list[str] = []
    due_episodes = []
    for folder in folders:
        episode_id = folder.name
        record_path = folder / "publish.json"
        if not record_path.exists() and not record_path.is_symlink():
            continue
        try:
            record = _object(record_path)
            if record.get("id") != episode_id:
                raise SignedMonitorError("episode identity differs")
            youtube_due = _time(record.get("due_at"))
            crossposts = record.get("crossposts")
            companion = crossposts.get("instagram") if isinstance(crossposts, dict) else None
            if (not isinstance(companion, dict)
                    or not isinstance(record.get("buffer_post_id"), str) or not record["buffer_post_id"]
                    or not isinstance(companion.get("id"), str) or not companion["id"]):
                raise SignedMonitorError("exact pair IDs are missing")
            instagram_due = _time(companion.get("due_at"))
        except SignedMonitorError:
            issues.append(f"{episode_id}: signed publish record is malformed")
            continue
        if youtube_due > now:
            continue
        receipt_ok = _receipt(folder, record, episode_id, youtube_due, instagram_due,
                              youtube_id, instagram_id, now)
        if not receipt_ok:
            issues.append(f"{episode_id}: control platform readback receipt is missing or mismatched")
        episode = {"episode_id": episode_id, "due_at": youtube_due.isoformat(),
                   "readback_receipt": "valid" if receipt_ok else "missing_or_mismatched", "posts": {}}
        for service, channel_id, post_id, due in (
                ("youtube", youtube_id, record["buffer_post_id"], youtube_due),
                ("instagram", instagram_id, companion["id"], instagram_due)):
            summary, post_issues = _post(posts.get(service, []), service=service,
                                         channel_id=channel_id, post_id=post_id,
                                         media_url=record.get("media_url"), due=due, now=now)
            episode["posts"][service] = summary
            issues.extend(f"{episode_id}: {issue}" for issue in post_issues)
        due_episodes.append(episode)
    return {"schema": "mool_katha_signed_pair_monitor_v1",
            "status": "alert" if issues else "ok", "signed_episodes": len(folders),
            "due_episodes": len(due_episodes), "episodes": due_episodes, "issues": issues}
