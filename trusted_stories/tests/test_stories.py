import hashlib
import json
import struct
from datetime import datetime, timedelta, timezone

import pytest

import trusted_stories as stories

NOW = datetime(2026, 10, 5, 3, 0, tzinfo=timezone.utc)
IG = "ig-channel"


def mp4(seconds: float, pad: bytes = b"") -> bytes:
    body = b"\x00" * 12 + struct.pack(">II", 1000, int(seconds * 1000)) + b"\x00" * 80
    return struct.pack(">I", 8 + len(body)) + b"mvhd" + body + pad


class Publisher:
    def __init__(self, rows, posts=None):
        self.rows, self.created, self.posts_by_id = rows, [], posts or {}

    def _require_channel(self, channel_id, service, ready=False):
        assert (channel_id, service, ready) == (IG, "instagram", True)

    def posts(self, since=None, channel_id=None):
        return self.rows

    def post(self, post_id):
        return self.posts_by_id.get(post_id)

    def _buffer(self, query, variables):
        self.created.append(variables["input"])
        return {"createPost": {"__typename": "PostActionSuccess",
                               "post": {"id": f"story-{len(self.created)}", "status": "scheduled",
                                        "dueAt": variables["input"]["dueAt"]}}}


def episode(root, ep, data, reel_id="reel-1"):
    sha = hashlib.sha256(data).hexdigest()
    public_id = f"mool-katha/{ep}-{sha}"
    folder = root / "content" / "episodes" / ep
    folder.mkdir(parents=True)
    (folder / "publish.json").write_text(json.dumps({
        "media_public_id": public_id, "video_sha256": sha,
        "media_url": f"https://res.cloudinary.com/x/video/upload/v1/{public_id}.mp4",
        "crossposts": {"instagram": {"id": reel_id, "status": "scheduled"}}}), encoding="utf-8")
    return f"https://res.cloudinary.com/x/video/upload/v1/{public_id}.mp4"


def sent(reel_id="reel-1", when=NOW - timedelta(hours=14)):
    return {"id": reel_id, "status": "sent", "sentAt": when.isoformat(), "channelId": IG}


def test_story_is_due_next_morning_ist():
    reel = datetime(2026, 10, 4, 13, 30, tzinfo=timezone.utc)  # 19:00 IST
    due = stories.story_due(reel, reel + timedelta(minutes=5))
    assert due.isoformat() == "2026-10-05T08:30:00+05:30"
    late = datetime(2026, 10, 5, 3, 10, tzinfo=timezone.utc)  # 08:40 IST, already past today's slot
    assert stories.story_due(reel, late).isoformat() == "2026-10-06T08:30:00+05:30"


def test_schedules_one_story_with_the_exact_signed_media(tmp_path):
    data = mp4(44.0)
    url = episode(tmp_path / "src", "ep022", data)
    publisher = Publisher([sent()])
    result = stories.run_once(tmp_path / "src", tmp_path / "state", publisher, IG,
                              fetch=lambda u: data, now=NOW)
    assert result["decision"] == "scheduled"
    (payload,) = publisher.created
    assert payload["assets"] == [{"video": {"url": url}}]
    assert payload["metadata"]["instagram"]["type"] == "story"
    assert "text" not in payload
    receipt = json.loads((tmp_path / "state" / "ep022.json").read_text())
    assert receipt["status"] == "scheduled" and receipt["video_seconds"] == 44.0
    again = stories.run_once(tmp_path / "src", tmp_path / "state", publisher, IG, fetch=lambda u: data, now=NOW)
    assert again["decision"] == "nothing_to_share" and len(publisher.created) == 1


def test_unsent_reel_gets_no_story(tmp_path):
    data = mp4(40.0)
    episode(tmp_path / "src", "ep022", data)
    publisher = Publisher([{**sent(), "status": "scheduled"}])
    result = stories.run_once(tmp_path / "src", tmp_path / "state", publisher, IG, fetch=lambda u: data, now=NOW)
    assert result["decision"] == "nothing_to_share" and not publisher.created


def test_changed_hosted_media_is_held(tmp_path):
    data = mp4(40.0)
    episode(tmp_path / "src", "ep022", data)
    publisher = Publisher([sent()])
    result = stories.run_once(tmp_path / "src", tmp_path / "state", publisher, IG,
                              fetch=lambda u: mp4(40.0, b"tampered"), now=NOW)
    assert result["decision"] == "held" and not publisher.created


def test_video_over_a_minute_is_skipped(tmp_path):
    data = mp4(61.0)
    episode(tmp_path / "src", "ep022", data)
    publisher = Publisher([sent()])
    result = stories.run_once(tmp_path / "src", tmp_path / "state", publisher, IG, fetch=lambda u: data, now=NOW)
    assert result["decision"] == "nothing_to_share" and not publisher.created
    assert json.loads((tmp_path / "state" / "ep022.json").read_text())["status"] == "skipped"


def test_old_reels_are_not_backfilled(tmp_path):
    data = mp4(40.0)
    episode(tmp_path / "src", "ep003", data)
    publisher = Publisher([sent(when=NOW - timedelta(days=5))])
    result = stories.run_once(tmp_path / "src", tmp_path / "state", publisher, IG, fetch=lambda u: data, now=NOW)
    assert result["decision"] == "nothing_to_share"


def test_receipt_for_another_episode_or_host_is_ignored(tmp_path):
    data = mp4(40.0)
    episode(tmp_path / "src", "ep022", data)
    path = tmp_path / "src/content/episodes/ep022/publish.json"
    record = json.loads(path.read_text())
    record["media_url"] = record["media_url"].replace("res.cloudinary.com", "evil.example")
    path.write_text(json.dumps(record))
    publisher = Publisher([sent()])
    result = stories.run_once(tmp_path / "src", tmp_path / "state", publisher, IG, fetch=lambda u: data, now=NOW)
    assert result["decision"] == "nothing_to_share"


def test_reconcile_records_sent_and_error(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    for ep in ("ep021", "ep022"):
        (state / f"{ep}.json").write_text(json.dumps({"episode": ep, "status": "scheduled",
                                                      "story_post_id": f"s-{ep}"}))
    publisher = Publisher([], posts={"s-ep021": {"status": "sent"},
                                     "s-ep022": {"status": "error", "error": {"message": "media"}}})
    updates = stories.reconcile(state, publisher)
    assert {u["episode"]: u["status"] for u in updates} == {"ep021": "sent", "ep022": "error"}
    assert json.loads((state / "ep022.json").read_text())["error"] == "media"


def test_mp4_without_header_is_held():
    with pytest.raises(stories.StoryHold):
        stories.mp4_seconds(b"not a video")
