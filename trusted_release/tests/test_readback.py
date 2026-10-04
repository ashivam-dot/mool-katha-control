"""Exact two-channel readback checks without a publisher credential."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trusted_release.readback import RECEIPT, ReadbackHold, _write_receipt, verify_after_release


class FakePublisher:
    ShortSpec = SimpleNamespace(load=lambda _path: SimpleNamespace(id="ep012"))

    def __init__(self, rows: dict[str, list[dict]], expected_public_id: str):
        self.rows = rows
        self.expected_public_id = expected_public_id
        self.queries: list[str] = []
        self.hosted_checks = 0

    def posts(self, *, channel_id: str) -> list[dict]:
        self.queries.append(channel_id)
        return self.rows[channel_id]

    def episode_posts(self, rows: list[dict], episode_id: str, channel_id: str) -> list[dict]:
        matches = []
        for row in rows:
            source = row["assets"][0]["source"]
            if row["channelId"] == channel_id and f"/{episode_id}-" in source:
                matches.append(row | {"video": source})
        return matches

    def _episode_media_public_id(self, media_url: str, _episode_id: str) -> str | None:
        return self.expected_public_id if media_url.endswith(self.expected_public_id + ".mp4") else None

    def description(self, _spec: object, _manifest: dict) -> str:
        return "exact YouTube text"

    def caption(self, _spec: object) -> str:
        return "exact Instagram caption"

    def require_reviewed_hosted_video(self, _folder: Path, _url: str, *, policy: object) -> None:
        self.hosted_checks += 1


class ReadbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "content/episodes/ep012"
        (self.folder / "work").mkdir(parents=True)
        self.video = b"exact signed MP4"
        self.hash = hashlib.sha256(self.video).hexdigest()
        self.public_id = f"mool-katha/ep012-{self.hash}"
        self.url = f"https://res.cloudinary.com/test/video/upload/v123/{self.public_id}.mp4"
        self.due = (datetime.now(timezone.utc) + timedelta(hours=8)).replace(microsecond=0).isoformat()
        (self.folder / "ep012.mp4").write_bytes(self.video)
        (self.folder / "short.yaml").write_text("id: ep012\n", encoding="utf-8")
        (self.folder / "work/manifest.json").write_text("{}\n", encoding="utf-8")
        (self.folder / "agent-release-review.json").write_text(
            json.dumps({"video_sha256": self.hash}) + "\n", encoding="utf-8")
        (self.folder / "agent-release-signature.json").write_text("{}\n", encoding="utf-8")
        self.record = {"id": "ep012", "media_public_id": self.public_id,
                       "media_url": self.url, "buffer_post_id": "yt-12", "due_at": self.due,
                       "crossposts": {"instagram": {"id": "ig-12", "due_at": self.due}}}
        self._save_record()
        self.rows = {"yt-channel": [self._post("youtube", "yt-channel", "yt-12", "exact YouTube text")],
                     "ig-channel": [self._post("instagram", "ig-channel", "ig-12", "exact Instagram caption")]}
        self.publisher = FakePublisher(self.rows, self.public_id)
        self.gates: list[Path] = []

    def _save_record(self) -> None:
        (self.folder / "publish.json").write_text(json.dumps(self.record) + "\n", encoding="utf-8")

    def _post(self, service: str, channel: str, post_id: str, text: str) -> dict:
        return {"id": post_id, "channelId": channel, "channelService": service,
                "status": "scheduled", "dueAt": self.due, "text": text,
                "assets": [{"source": self.url}]}

    def _gate(self, folder: Path, *, policy: object, studio_root: Path) -> None:
        self.assertEqual(folder, self.folder.resolve())
        self.assertEqual(studio_root, self.root.resolve())
        self.gates.append(folder)

    def _verify(self, status: str = "scheduled") -> list[dict]:
        return verify_after_release(self.root, {"status": status, "episode_id": "ep012"}
                                    if status != "no_signed_candidate" else {"status": status},
                                    publisher=self.publisher, policy=object(), evidence_gate=self._gate,
                                    youtube_id="yt-channel", instagram_id="ig-channel")

    def test_exact_pair_writes_bounded_receipt_and_retry_does_not_recreate(self) -> None:
        result = self._verify()
        self.assertEqual(len(result), 1)
        self.assertEqual(self.publisher.queries, ["yt-channel", "ig-channel"])
        self.assertEqual(self.publisher.hosted_checks, 1)
        saved = json.loads((self.folder / RECEIPT).read_text(encoding="utf-8"))
        self.assertEqual(saved["video_sha256"], self.hash)
        self.assertEqual(saved["posts"]["youtube"]["post_id"], "yt-12")
        self.assertEqual(saved["posts"]["instagram"]["post_id"], "ig-12")
        self.assertEqual(self._verify("no_signed_candidate"), [])
        self.assertEqual(self.publisher.queries, ["yt-channel", "ig-channel"])

    def test_wrong_signed_or_hosted_hash_holds_without_receipt(self) -> None:
        self.record["media_public_id"] = "mool-katha/ep012-" + "0" * 64
        self._save_record()
        with self.assertRaisesRegex(ReadbackHold, "full-hash signed final MP4"):
            self._verify()
        self.assertEqual(self.publisher.hosted_checks, 0)
        self.assertFalse((self.folder / RECEIPT).exists())

    def test_duplicate_or_wrong_destination_post_holds_without_receipt(self) -> None:
        self.rows["yt-channel"].append(self._post("youtube", "yt-channel", "yt-duplicate", "exact YouTube text"))
        with self.assertRaisesRegex(ReadbackHold, "2 posts"):
            self._verify()
        self.rows["yt-channel"].pop()
        self.rows["ig-channel"][0]["channelService"] = "youtube"
        with self.assertRaisesRegex(ReadbackHold, "destination"):
            self._verify()
        self.assertFalse((self.folder / RECEIPT).exists())

    def test_unconfirmed_pair_is_retried_after_an_interrupted_run(self) -> None:
        result = self._verify("no_signed_candidate")
        self.assertEqual(result[0]["episode_id"], "ep012")
        self.assertTrue((self.folder / RECEIPT).is_file())

    def test_malformed_companion_and_invalid_link_hold(self) -> None:
        self.record["crossposts"] = ["instagram"]
        self._save_record()
        with self.assertRaisesRegex(ReadbackHold, "both post IDs"):
            self._verify()
        self.record["crossposts"] = {"instagram": {"id": "ig-12", "due_at": self.due}}
        self._save_record()
        self.rows["yt-channel"][0]["status"] = "sent"
        self.rows["yt-channel"][0]["sentAt"] = datetime.now(timezone.utc).isoformat()
        self.rows["yt-channel"][0]["externalLink"] = "https://www.youtube.com/shorts/short"
        with self.assertRaisesRegex(ReadbackHold, "valid public link"):
            self._verify()
        self.assertFalse((self.folder / RECEIPT).exists())

    def test_receipt_write_is_atomic_and_cannot_replace_existing_file(self) -> None:
        path = self.folder / RECEIPT
        with patch("trusted_release.readback.os.link", side_effect=OSError("interrupted")):
            with self.assertRaisesRegex(OSError, "interrupted"):
                _write_receipt(path, {"episode_id": "ep012"})
        self.assertFalse(path.exists())
        self.assertEqual(list(self.folder.glob(".control-platform-readback-*.tmp")), [])
        path.write_text("existing", encoding="utf-8")
        with self.assertRaisesRegex(ReadbackHold, "already exists"):
            _write_receipt(path, {"episode_id": "ep012"})
        self.assertEqual(path.read_text(encoding="utf-8"), "existing")
        self.assertEqual(list(self.folder.glob(".control-platform-readback-*.tmp")), [])

    def test_sent_pair_requires_platform_links(self) -> None:
        for service, channel in (("youtube", "yt-channel"), ("instagram", "ig-channel")):
            self.rows[channel][0]["status"] = "sent"
            self.rows[channel][0]["sentAt"] = datetime.now(timezone.utc).isoformat()
            self.rows[channel][0]["externalLink"] = ("https://www.youtube.com/shorts/ABCDEFGHIJK"
                                                      if service == "youtube"
                                                      else "https://www.instagram.com/reel/ABC123/")
        self._verify()
        (self.folder / RECEIPT).unlink()
        self.rows["ig-channel"][0]["externalLink"] = "https://www.instagram.com/reel/"
        with self.assertRaisesRegex(ReadbackHold, "valid public link"):
            self._verify()


if __name__ == "__main__":
    unittest.main()
