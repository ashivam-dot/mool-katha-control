"""Post-due checks use source receipts as data and live exact Buffer rows."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trusted_monitor.signed import SignedMonitorError, inspect


NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
YT_CHANNEL = "youtube-channel"
IG_CHANNEL = "instagram-channel"


class SignedPairTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / "content/episodes/ep012"
        self.folder.mkdir(parents=True)
        self.due = (NOW - timedelta(hours=1)).isoformat()
        self.hash = "a" * 64
        self.url = ("https://res.cloudinary.com/mw0oh0v8/video/upload/v123/"
                    f"mool-katha/ep012-{self.hash}.mp4")
        self.record = {"id": "ep012", "buffer_post_id": "yt-12", "due_at": self.due,
                       "media_public_id": f"mool-katha/ep012-{self.hash}", "media_url": self.url,
                       "crossposts": {"instagram": {"id": "ig-12", "due_at": self.due}}}
        self.receipt = {"schema": "mool_katha_control_platform_readback_v1",
                        "episode_id": "ep012", "video_sha256": self.hash,
                        "media_url": self.url, "verified_at": (NOW - timedelta(hours=2)).isoformat(),
                        "posts": {"youtube": {"post_id": "yt-12", "channel_id": YT_CHANNEL,
                                              "due_at": self.due, "status": "scheduled"},
                                  "instagram": {"post_id": "ig-12", "channel_id": IG_CHANNEL,
                                                "due_at": self.due, "status": "scheduled"}}}
        (self.folder / "agent-release-signature.json").write_text("{}\n", encoding="utf-8")
        self._save()
        self.posts = {"youtube": [self._post("youtube", YT_CHANNEL, "yt-12",
                                                 "https://www.youtube.com/shorts/ABCDEFGHIJK")],
                      "instagram": [self._post("instagram", IG_CHANNEL, "ig-12",
                                                   "https://www.instagram.com/reel/ABC123/")]}

    def _save(self) -> None:
        (self.folder / "publish.json").write_text(json.dumps(self.record) + "\n", encoding="utf-8")
        (self.folder / "control-platform-readback.json").write_text(
            json.dumps(self.receipt) + "\n", encoding="utf-8")

    def _post(self, service: str, channel: str, post_id: str, link: str) -> dict:
        return {"id": post_id, "channel_id": channel, "service": service,
                "status": "sent", "due_at": self.due,
                "sent_at": (NOW - timedelta(minutes=30)).isoformat(),
                "external_link": link, "media_sources": [self.url]}

    def _inspect(self) -> dict:
        return inspect(NOW, self.root, self.posts, YT_CHANNEL, IG_CHANNEL)

    def test_valid_sent_pair_and_control_receipt(self) -> None:
        result = self._inspect()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["signed_episodes"], 1)
        self.assertEqual(result["due_episodes"], 1)
        self.assertEqual(result["episodes"][0]["posts"]["youtube"]["public_link"],
                         "https://www.youtube.com/shorts/ABCDEFGHIJK")

    def test_due_pair_alerts_on_pending_instagram_even_with_sent_youtube(self) -> None:
        self.posts["instagram"][0]["status"] = "scheduled"
        self.posts["instagram"][0]["external_link"] = None
        result = self._inspect()
        self.assertEqual(result["status"], "alert")
        self.assertIn("ep012: instagram Buffer post is not sent after due time", result["issues"])
        self.assertNotIn("ep012: youtube Buffer post is not sent after due time", result["issues"])

    def test_missing_or_mismatched_receipt_and_live_links_alert(self) -> None:
        (self.folder / "control-platform-readback.json").unlink()
        self.posts["youtube"][0]["media_sources"] = ["https://other.example/video.mp4"]
        self.posts["instagram"][0]["external_link"] = "https://www.instagram.com/reel/"
        result = self._inspect()
        self.assertEqual(len(result["issues"]), 3)
        self.assertIn("ep012: control platform readback receipt is missing or mismatched", result["issues"])
        self.assertIn("ep012: youtube Buffer destination or video differs from signed release", result["issues"])
        self.assertIn("ep012: instagram Buffer sent post has no valid public link", result["issues"])

    def test_future_due_pair_does_not_alert_while_scheduling(self) -> None:
        self.record["due_at"] = (NOW + timedelta(hours=1)).isoformat()
        self._save()
        (self.folder / "control-platform-readback.json").unlink()
        result = inspect(NOW, self.root, {}, YT_CHANNEL, IG_CHANNEL)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["due_episodes"], 0)

    def test_signed_source_symlink_is_rejected(self) -> None:
        (self.folder / "publish.json").unlink()
        (self.folder / "publish.json").symlink_to("/etc/hosts")
        self.assertIn("ep012: signed publish record is malformed", self._inspect()["issues"])
        (self.folder / "agent-release-signature.json").unlink()
        (self.folder / "agent-release-signature.json").symlink_to("/etc/hosts")
        with self.assertRaisesRegex(SignedMonitorError, "linked"):
            self._inspect()


if __name__ == "__main__":
    unittest.main()
