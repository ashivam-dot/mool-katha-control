"""Confirm recent signed posts exist on their public channel pages after delivery."""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

from trusted_release.readback import _platform_link


YOUTUBE_CHANNEL_ID = "UCdqxVnoHDWXgA2ZVJWkSu8w"
INSTAGRAM_HANDLE = "moolkatha.hindi"
VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
SHORTCODE = re.compile(r"[A-Za-z0-9_-]{5,40}\Z")
EPISODE = re.compile(r"ep[0-9]{3}\Z")
POST_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
PROOF_SCHEMA = "mool_katha_public_delivery_proof_v1"
RECENT_WINDOW = timedelta(hours=72)
UTC = timezone.utc


class _OpenGraph(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        values = dict(attrs)
        if values.get("property") == "og:url" and isinstance(values.get("content"), str):
            self.urls.append(values["content"])


def _time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.utcoffset() is not None else None


def _youtube_id(link: object) -> str | None:
    if not isinstance(link, str):
        return None
    parsed = urlsplit(link)
    path = parsed.path.strip("/").split("/")
    if parsed.hostname in {"youtube.com", "www.youtube.com"} and len(path) == 2 and path[0] == "shorts":
        video_id = path[1]
    elif parsed.hostname in {"youtube.com", "www.youtube.com"} and parsed.path == "/watch":
        video_id = parsed.query.removeprefix("v=") if parsed.query.startswith("v=") else ""
    elif parsed.hostname == "youtu.be" and len(path) == 1:
        video_id = path[0]
    else:
        return None
    return video_id if VIDEO_ID.fullmatch(video_id) else None


def _instagram_path(link: object) -> tuple[str, str] | None:
    if not isinstance(link, str):
        return None
    parsed = urlsplit(link)
    path = parsed.path.strip("/").split("/")
    if (parsed.scheme != "https" or parsed.hostname not in {"instagram.com", "www.instagram.com"}
            or parsed.username or parsed.password or parsed.port or parsed.query or parsed.fragment
            or len(path) != 2 or path[0] not in {"reel", "p"}
            or not SHORTCODE.fullmatch(path[1])):
        return None
    return path[0], path[1]


def instagram_visible(kind: str, code: str) -> bool:
    """A 200 sign-in shell is insufficient: the public page must identify the owned post."""
    if kind not in {"reel", "p"} or not SHORTCODE.fullmatch(code):
        return False
    url = f"https://www.instagram.com/{kind}/{code}/"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; MoolKathaMonitor/1)"})
    with urllib.request.urlopen(request, timeout=25) as response:
        final = urlsplit(response.geturl())
        if (response.status != 200 or final.scheme != "https"
                or final.hostname not in {"instagram.com", "www.instagram.com"}
                or final.path.rstrip("/") not in {f"/{kind}/{code}",
                                                  f"/{INSTAGRAM_HANDLE}/{kind}/{code}"}
                or final.query or final.fragment):
            return False
        body = response.read(2_000_001)
    if len(body) > 2_000_000:
        return False
    metadata = _OpenGraph()
    metadata.feed(body.decode("utf-8", "replace"))
    expected = f"/{INSTAGRAM_HANDLE}/{kind}/{code}"
    if len(metadata.urls) != 1:
        return False
    found = urlsplit(metadata.urls[0])
    return (found.scheme == "https" and found.hostname in {"instagram.com", "www.instagram.com"}
            and found.path.rstrip("/") == expected and not found.query and not found.fragment
            and not found.username and not found.password and not found.port)


def _unique(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate public proof key")
        value[key] = item
    return value


def load_proofs(folder: Path) -> dict[str, dict]:
    """Read private, append-only proof receipts from the control data branch."""
    if folder.is_symlink():
        raise ValueError("public proof directory is linked")
    if not folder.exists():
        return {}
    if not folder.is_dir():
        raise ValueError("public proof directory is unavailable")
    paths = list(folder.iterdir())
    if len(paths) > 1000:
        raise ValueError("public proof directory exceeds its bound")
    proofs = {}
    for path in paths:
        if not EPISODE.fullmatch(path.stem) or path.suffix != ".json":
            raise ValueError("public proof filename is invalid")
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16_000:
            raise ValueError("public proof file is invalid")
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
        if not isinstance(value, dict) or value.get("episode_id") != path.stem:
            raise ValueError("public proof identity is invalid")
        proofs[path.stem] = value
    return proofs


def build_proof(episode: dict, result: dict, verified_at: str) -> dict:
    """Admit a proof only when both exact public IDs match the signed pair."""
    episode_id = episode.get("episode_id")
    due = _time(episode.get("due_at"))
    verified = _time(verified_at)
    posts = episode.get("posts")
    if (not isinstance(episode_id, str) or not EPISODE.fullmatch(episode_id)
            or due is None or verified is None or not due <= verified <= datetime.now(UTC) + timedelta(minutes=5)
            or not isinstance(posts, dict) or result.get("episode_id") != episode_id
            or result.get("youtube_visible") is not True or result.get("instagram_visible") is not True):
        raise ValueError("public proof has no verified signed pair")
    youtube, instagram = posts.get("youtube"), posts.get("instagram")
    if not isinstance(youtube, dict) or not isinstance(instagram, dict):
        raise ValueError("public proof post pair is unavailable")
    yt_link, ig_link = youtube.get("public_link"), instagram.get("public_link")
    yt_id, ig_path = _youtube_id(yt_link), _instagram_path(ig_link)
    yt_post, ig_post = youtube.get("post_id"), instagram.get("post_id")
    if (youtube.get("status") != "sent" or instagram.get("status") != "sent"
            or not _platform_link(yt_link, "youtube") or not _platform_link(ig_link, "instagram")
            or not yt_id or not ig_path or result.get("youtube_id") != yt_id
            or result.get("instagram_shortcode") != ig_path[1]
            or not isinstance(yt_post, str) or not POST_ID.fullmatch(yt_post)
            or not isinstance(ig_post, str) or not POST_ID.fullmatch(ig_post)):
        raise ValueError("public proof differs from exact Buffer post IDs or links")
    return {"schema": PROOF_SCHEMA, "episode_id": episode_id,
            "due_at": due.isoformat(), "verified_at": verified.isoformat(),
            "buffer_post_ids": {"youtube": yt_post, "instagram": ig_post},
            "public_links": {"youtube": yt_link, "instagram": ig_link},
            "youtube_id": yt_id, "instagram_shortcode": ig_path[1]}


def _proof_matches(proof: dict, episode: dict, now: datetime) -> bool:
    posts = episode.get("posts")
    if not isinstance(posts, dict):
        return False
    due, verified = _time(episode.get("due_at")), _time(proof.get("verified_at"))
    if (proof.get("schema") != PROOF_SCHEMA or proof.get("episode_id") != episode.get("episode_id")
            or due is None or verified is None or not due <= verified <= now + timedelta(minutes=5)
            or proof.get("due_at") != due.isoformat()):
        return False
    youtube, instagram = posts.get("youtube"), posts.get("instagram")
    if not isinstance(youtube, dict) or not isinstance(instagram, dict):
        return False
    yt_link, ig_link = youtube.get("public_link"), instagram.get("public_link")
    path = _instagram_path(ig_link)
    return (youtube.get("status") == "sent" and instagram.get("status") == "sent"
            and proof.get("buffer_post_ids") == {"youtube": youtube.get("post_id"),
                                                  "instagram": instagram.get("post_id")}
            and proof.get("public_links") == {"youtube": yt_link, "instagram": ig_link}
            and proof.get("youtube_id") == _youtube_id(yt_link)
            and path is not None and proof.get("instagram_shortcode") == path[1])


def inspect(now: datetime, signed_pairs: dict, youtube_feed: object,
            *, instagram_probe=instagram_visible, proofs: dict[str, dict] | None = None) -> dict:
    """Check exact recent public IDs; signed_pairs already binds their Buffer/media receipts."""
    issues: list[str] = []
    checked: list[dict] = []
    feed = youtube_feed if isinstance(youtube_feed, dict) else {}
    recent = feed.get("recent_videos") if feed.get("channel_id") == YOUTUBE_CHANNEL_ID else None
    for episode in signed_pairs.get("episodes", []):
        episode_id = episode.get("episode_id")
        due = _time(episode.get("due_at"))
        if not isinstance(episode_id, str) or due is None or due > now:
            continue
        posts = episode.get("posts") if isinstance(episode.get("posts"), dict) else {}
        youtube = posts.get("youtube") if isinstance(posts.get("youtube"), dict) else {}
        instagram = posts.get("instagram") if isinstance(posts.get("instagram"), dict) else {}
        result = {"episode_id": episode_id, "youtube_id": None, "youtube_visible": False,
                  "instagram_shortcode": None, "instagram_visible": False,
                  "proof_status": "unverified"}
        proof = (proofs or {}).get(episode_id)
        if proof is not None:
            if _proof_matches(proof, episode, now):
                result.update({"youtube_id": proof["youtube_id"], "youtube_visible": True,
                               "instagram_shortcode": proof["instagram_shortcode"],
                               "instagram_visible": True, "proof_status": "archived",
                               "verified_at": proof["verified_at"]})
            else:
                result["proof_status"] = "mismatched"
                issues.append(f"{episode_id}: archived public proof differs from signed release")
            checked.append(result)
            continue
        if now - due > RECENT_WINDOW:
            result["proof_status"] = "expired"
            issues.append(f"{episode_id}: public delivery was never verified within 72 hours")
            checked.append(result)
            continue
        if youtube.get("status") == "sent":
            video_id = _youtube_id(youtube.get("public_link"))
            result["youtube_id"] = video_id
            if video_id and isinstance(recent, list):
                result["youtube_visible"] = sum(
                    isinstance(item, dict) and item.get("id") == video_id for item in recent) == 1
            if not result["youtube_visible"]:
                issues.append(f"{episode_id}: exact YouTube video is absent from its public channel feed")
        if instagram.get("status") == "sent":
            path = _instagram_path(instagram.get("public_link"))
            if path:
                result["instagram_shortcode"] = path[1]
                try:
                    result["instagram_visible"] = bool(instagram_probe(*path))
                except Exception:
                    pass
            if not result["instagram_visible"]:
                issues.append(f"{episode_id}: exact Instagram Reel is not verifiable on its public page")
        if result["youtube_visible"] and result["instagram_visible"]:
            result["proof_status"] = "fresh"
            result["verified_at"] = now.isoformat()
        checked.append(result)
    return {"schema": "mool_katha_public_delivery_monitor_v1",
            "status": "alert" if issues else "ok", "checked_episodes": len(checked),
            "episodes": checked, "issues": issues}
