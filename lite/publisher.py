"""Host the finished MP4 on Cloudinary and schedule it on Mool Katha's own Buffer YouTube and Instagram channels.

Uses the producer's Buffer/Cloudinary helpers (`ytc.publish`): exact channel IDs with service, readiness and
handle checks, the same GraphQL calls, and the same Cloudinary signing.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import requests

from .slots import IST

log = logging.getLogger(__name__)

TITLE_CHARS, DESCRIPTION_BYTES, CAPTION_CHARS = 100, 5000, 2200
DISCLOSURE = "मूल कथा की शोध और प्रस्तुति। आवाज़ AI से बनाई गई है।"
SHARE_LINE = ("इस कथा को अपने परिवार और मित्रों को भेजें, ताकि मूल ग्रंथ की बात उन तक भी पहुँचे।\n"
              "मूल ग्रंथों की कथाओं के लिए फ़ॉलो करें @moolkatha.hindi")
YOUTUBE_ONLY_TAGS = {"shorts", "youtubeshorts", "ytshorts"}
CREATE = """
mutation Create($input: CreatePostInput!) {
  createPost(input: $input) {
    __typename
    ... on PostActionSuccess { post { id status dueAt } }
    ... on MutationError { message }
  }
}"""


def _safe(text: str) -> str:
    return text.replace("<", "‹").replace(">", "›")


def youtube_description(ep: dict) -> str:
    credits = []
    for pic in ep["pictures"]:
        c = pic["credit"]
        credits.append("- " + ", ".join(x for x in (f'"{c.get("title", "Untitled")}"', c.get("credit"),
                                                      c.get("license"), c.get("source"), c.get("url")) if x))
    blocks = [ep["description"],
              f"स्रोत: {ep['citation']}\n{ep['source_url']}\nअनुवाद: {ep['translator']}",
              "चित्र (सार्वजनिक डोमेन / CC0):\n" + "\n".join(credits) if credits else "",
              DISCLOSURE, " ".join(ep["hashtags"][:3])]
    text = _safe("\n\n".join(b for b in blocks if b))
    if len(text.encode()) > DESCRIPTION_BYTES:
        blocks[2] = "चित्र: " + ", ".join(dict.fromkeys(p["credit"].get("source", "") for p in ep["pictures"]))
        text = _safe("\n\n".join(b for b in blocks if b))
    return text.encode()[:DESCRIPTION_BYTES].decode(errors="ignore")


def instagram_caption(ep: dict) -> str:
    tags = [t for t in ep["hashtags"] if t.lower().lstrip("#") not in YOUTUBE_ONLY_TAGS][:5]
    if ep.get("shloka"):
        source_line = f"श्लोक {ep['shloka']['number']} ({ep['citation']}):\n{ep['shloka']['sanskrit']}"
    elif ep.get("key_quote"):
        source_line = f"मूल पाठ ({ep['citation']}):\n“{ep['key_quote']}”"
    else:
        source_line = f"स्रोत: {ep['citation']}"
    keywords = " · ".join(ep.get("keywords") or [])
    tail = " ".join(tags)
    fixed = [ep["title"], source_line, SHARE_LINE, keywords, DISCLOSURE, f"Source: {ep['source_url']}"]
    room = CAPTION_CHARS - len("\n\n".join(b for b in fixed if b)) - len(tail) - 8
    description = ep["description"][:max(room, 0)].rstrip()
    text = "\n\n".join(b for b in (ep["title"], description, source_line, SHARE_LINE, keywords, DISCLOSURE,
                                   f"Source: {ep['source_url']}") if b)
    return text[:CAPTION_CHARS - len(tail) - 2].rstrip() + ("\n\n" + tail if tail else "")


def youtube_payload(channel_id: str, ep: dict, due: datetime, media_url: str) -> dict:
    return {
        "channelId": channel_id,
        "text": youtube_description(ep),
        "schedulingType": "automatic",
        "mode": "customScheduled",
        "dueAt": due.isoformat(),
        "assets": [{"video": {"url": media_url}}],
        "metadata": {"youtube": {"title": _safe(ep["title"])[:TITLE_CHARS], "categoryId": "27", "privacy": "public",
                                 "madeForKids": False, "notifySubscribers": True, "isAiGenerated": False,
                                 "embeddable": True}},
    }


def instagram_payload(channel_id: str, ep: dict, due: datetime, media_url: str) -> dict:
    return {
        "channelId": channel_id,
        "text": instagram_caption(ep),
        "schedulingType": "automatic",
        "mode": "customScheduled",
        "dueAt": due.isoformat(),
        "assets": [{"video": {"url": media_url}}],
        "metadata": {"instagram": {"type": "reel", "shouldShareToFeed": True, "isAiGenerated": True}},
    }


def destinations() -> tuple[str, str]:
    from ytc import publish

    youtube = os.environ["BUFFER_YOUTUBE_CHANNEL_ID"].strip()
    instagram = os.environ["BUFFER_INSTAGRAM_CHANNEL_ID"].strip()
    publish._require_channel(youtube, "youtube", ready=True)
    publish._require_channel(instagram, "instagram", ready=True)
    return youtube, instagram


def recent_posts(channel_id: str, now: datetime) -> list[dict]:
    from ytc import publish

    return publish.posts(since=now.astimezone(IST) - timedelta(days=2), channel_id=channel_id)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def host(video: Path, episode_id: str) -> tuple[str, str]:
    """(public_id, url) on Cloudinary, verified byte for byte against the local file."""
    from ytc import publish

    digest = sha256(video)
    public_id = f"mool-katha/{episode_id}-{digest[:16]}"
    url = publish.host_video(video, public_id)
    hosted = hashlib.sha256()
    with requests.get(url, stream=True, timeout=(15, 120)) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(1 << 20):
            hosted.update(chunk)
    if hosted.hexdigest() != digest:
        publish.unhost_video(public_id)
        raise RuntimeError("hosted video differs from the rendered MP4")
    return public_id, url


def _twin(rows: list[dict], media_url: str) -> dict | None:
    for row in rows:
        sources = [a.get("source") for a in row.get("assets") or [] if isinstance(a, dict)]
        if media_url in sources and row.get("status") not in ("error", "draft"):
            return row
    return None


def create(payload: dict) -> dict:
    from ytc import publish

    result = publish._buffer(CREATE, {"input": payload})["createPost"]
    if result["__typename"] != "PostActionSuccess":
        raise RuntimeError(f"Buffer rejected the post: {result.get('message')}")
    return result["post"]


def schedule(ep: dict, video: Path, due: datetime, youtube_id: str, instagram_id: str, now: datetime) -> dict:
    """Host once, then create the YouTube Short and the Instagram Reel for the same slot, never twice."""
    public_id, media_url = host(video, ep["id"])
    record = {"media_public_id": public_id, "media_url": media_url, "due_at": due.isoformat()}
    for service, channel_id, build in (("youtube", youtube_id, youtube_payload),
                                       ("instagram", instagram_id, instagram_payload)):
        existing = _twin(recent_posts(channel_id, now), media_url)
        if existing:
            record[service] = {"id": existing["id"], "status": existing.get("status"), "adopted": True}
            continue
        try:
            post = create(build(channel_id, ep, due, media_url))
            record[service] = {"id": post["id"], "status": post.get("status"), "due_at": post.get("dueAt")}
        except Exception as err:
            log.error("%s post for %s failed: %s", service, ep["id"], err)
            record[service] = {"error": re.sub(r"\s+", " ", str(err))[:300]}
    return record


def readback(record: dict, youtube_id: str, instagram_id: str, now: datetime) -> dict:
    """Read both posts back from Buffer and confirm destination, video URL and due time."""
    out = {}
    for service, channel_id in (("youtube", youtube_id), ("instagram", instagram_id)):
        post_id = (record.get(service) or {}).get("id")
        row = next((r for r in recent_posts(channel_id, now) if r.get("id") == post_id), None) if post_id else None
        if not row:
            out[service] = {"ok": False, "reason": "post not found in Buffer"}
            continue
        sources = [a.get("source") for a in row.get("assets") or [] if isinstance(a, dict)]
        due_ok = row.get("dueAt") and datetime.fromisoformat(row["dueAt"]) == datetime.fromisoformat(record["due_at"])
        out[service] = {"ok": bool(row.get("channelId") == channel_id and sources == [record["media_url"]] and due_ok
                                   and row.get("status") not in ("error", "draft")),
                        "status": row.get("status"), "dueAt": row.get("dueAt"), "externalLink": row.get("externalLink")}
    return out
