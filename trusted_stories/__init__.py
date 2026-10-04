"""Share each sent, signed Mool Reel once as an Instagram Story the next morning.

The Story reuses the exact hosted MP4 of a Reel that Buffer already reports as sent, after re-hashing the
hosted bytes against the signed SHA-256 in the episode's receipt. It never creates a Reel, never touches the
YouTube post, and keeps its own receipts, so a Story failure cannot affect the signed release.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
STORY_TIME = time(8, 30)
MIN_GAP_AFTER_REEL = timedelta(hours=10)
MIN_LEAD = timedelta(minutes=30)
MAX_REEL_AGE = timedelta(days=4)
MAX_STORY_SECONDS = 59.5
MIN_STORY_SECONDS = 3.0
MAX_MEDIA_BYTES = 60 * 1024 * 1024
SCHEMA = "mool.instagram-story/v1"
EPISODE = re.compile(r"ep[0-9]{3}\Z")
PUBLIC_ID = re.compile(r"mool-katha/(ep[0-9]{3})-([0-9a-f]{64})\Z")
MEDIA_HOST = "res.cloudinary.com"

CREATE = """
mutation Create($input: CreatePostInput!) {
  createPost(input: $input) {
    __typename
    ... on PostActionSuccess { post { id status dueAt } }
    ... on MutationError { message }
  }
}"""


class StoryHold(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    episode: str
    reel_post_id: str
    media_url: str
    video_sha256: str
    reel_sent_at: datetime


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else None


def story_due(reel_sent_at: datetime, now: datetime) -> datetime:
    """The first 08:30 IST at least ten hours after the Reel went out and half an hour from now."""
    earliest = max(reel_sent_at + MIN_GAP_AFTER_REEL, now + MIN_LEAD).astimezone(IST)
    due = datetime.combine(earliest.date(), STORY_TIME, IST)
    return due if due >= earliest else due + timedelta(days=1)


def mp4_seconds(data: bytes) -> float:
    """Movie duration from the MP4 `mvhd` box, without ffprobe."""
    at = data.find(b"mvhd")
    if at < 4:
        raise StoryHold("hosted media has no MP4 movie header")
    version = data[at + 4]
    if version == 1:
        timescale, duration = struct.unpack(">IQ", data[at + 24:at + 36])
    else:
        timescale, duration = struct.unpack(">II", data[at + 16:at + 24])
    if not timescale:
        raise StoryHold("hosted media has no MP4 timescale")
    return duration / timescale


def _receipt_candidate(folder: Path) -> tuple[str, str, str, str] | None:
    """(Instagram Reel post id, media URL, signed SHA-256, public id) from a release receipt, or None."""
    path = folder / "publish.json"
    if not path.is_file() or path.is_symlink():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    instagram = ((record.get("crossposts") or {}).get("instagram") or {}) if isinstance(record, dict) else {}
    public_id = record.get("media_public_id") if isinstance(record, dict) else None
    match = PUBLIC_ID.fullmatch(public_id) if isinstance(public_id, str) else None
    media_url = record.get("media_url")
    if not (match and match.group(1) == folder.name and isinstance(instagram.get("id"), str)
            and isinstance(media_url, str) and media_url.startswith(f"https://{MEDIA_HOST}/")
            and media_url.endswith(f"/{public_id}.mp4")):
        return None
    signed = record.get("video_sha256", match.group(2))
    if signed != match.group(2):
        return None
    return instagram["id"], media_url, signed, public_id


def candidates(source: Path, sent_reels: dict[str, dict], now: datetime) -> list[Candidate]:
    found = []
    for folder in sorted((source / "content" / "episodes").iterdir()):
        if not EPISODE.fullmatch(folder.name):
            continue
        receipt = _receipt_candidate(folder)
        if receipt is None:
            continue
        reel_id, media_url, sha, _ = receipt
        row = sent_reels.get(reel_id)
        if not row or row.get("status") != "sent":
            continue
        sent_at = _parse_time(row.get("sentAt")) or _parse_time(row.get("dueAt"))
        if sent_at is None or now - sent_at > MAX_REEL_AGE:
            continue
        found.append(Candidate(folder.name, reel_id, media_url, sha, sent_at))
    return found


def verify_media(fetch: Callable[[str], bytes], candidate: Candidate) -> float:
    data = fetch(candidate.media_url)
    if len(data) > MAX_MEDIA_BYTES:
        raise StoryHold(f"{candidate.episode}: hosted media is too large for a Story")
    if hashlib.sha256(data).hexdigest() != candidate.video_sha256:
        raise StoryHold(f"{candidate.episode}: hosted media differs from the signed final MP4")
    return mp4_seconds(data)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reconcile(state: Path, publisher) -> list[dict]:
    """Record the final Buffer status of Stories still waiting to go out."""
    updates = []
    for path in sorted(state.glob("ep[0-9][0-9][0-9].json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("status") not in ("scheduled", "sending", "draft", "needs_approval"):
            continue
        post = publisher.post(receipt["story_post_id"])
        status = post.get("status") if post else "missing"
        if status != receipt["status"]:
            receipt["status"] = status
            if post and post.get("error"):
                receipt["error"] = (post["error"] or {}).get("message")
            _write(path, receipt)
            updates.append({"episode": receipt["episode"], "status": status})
    return updates


def run_once(source: Path, state: Path, publisher, instagram_id: str, *,
             fetch: Callable[[str], bytes], now: datetime | None = None) -> dict:
    """Schedule at most one Story for the newest sent Reel that has none yet."""
    now = now or datetime.now(timezone.utc)
    publisher._require_channel(instagram_id, "instagram", ready=True)
    updates = reconcile(state, publisher)
    rows = publisher.posts(since=now - MAX_REEL_AGE - timedelta(days=1), channel_id=instagram_id)
    sent_reels = {row["id"]: row for row in rows if row.get("channelId") == instagram_id}
    for candidate in reversed(candidates(source, sent_reels, now)):
        path = state / f"{candidate.episode}.json"
        if path.exists():
            continue
        try:
            seconds = verify_media(fetch, candidate)
        except StoryHold as exc:
            _write(path, {"schema": SCHEMA, "episode": candidate.episode, "status": "held",
                          "reason": str(exc), "checked_at": now.isoformat()})
            return {"decision": "held", "episode": candidate.episode, "reason": str(exc), "updates": updates}
        if not MIN_STORY_SECONDS <= seconds <= MAX_STORY_SECONDS:
            _write(path, {"schema": SCHEMA, "episode": candidate.episode, "status": "skipped",
                          "reason": f"video is {seconds:.1f} s; Stories take 3-60 s", "checked_at": now.isoformat()})
            continue
        due = story_due(candidate.reel_sent_at, now)
        payload = {
            "channelId": instagram_id,
            "schedulingType": "automatic",
            "mode": "customScheduled",
            "dueAt": due.isoformat(),
            "assets": [{"video": {"url": candidate.media_url}}],
            "metadata": {"instagram": {"type": "story", "shouldShareToFeed": False, "isAiGenerated": True}},
        }
        result = publisher._buffer(CREATE, {"input": payload})["createPost"]
        if result.get("__typename") != "PostActionSuccess":
            raise StoryHold(f"{candidate.episode}: Buffer refused the Story: {result.get('message')}")
        post = result["post"]
        _write(path, {"schema": SCHEMA, "episode": candidate.episode, "status": post["status"],
                      "story_post_id": post["id"], "due_at": post.get("dueAt") or due.isoformat(),
                      "reel_post_id": candidate.reel_post_id, "media_url": candidate.media_url,
                      "video_sha256": candidate.video_sha256, "video_seconds": round(seconds, 2),
                      "created_at": now.isoformat()})
        return {"decision": "scheduled", "episode": candidate.episode, "due_at": due.isoformat(),
                "updates": updates}
    return {"decision": "nothing_to_share", "updates": updates}
