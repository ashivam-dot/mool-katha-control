import hashlib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from lite import publisher

DUE = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 10, 6, 8, 0, tzinfo=timezone.utc)
EP = {
    "id": "lite-20261006-1340", "title": "हनुमान को सुरसा ने क्यों रोका? | वाल्मीकि रामायण",
    "description": "सुन्दरकाण्ड के पहले सर्ग में हनुमान की समुद्र-यात्रा का प्रसंग।",
    "citation": "वाल्मीकि रामायण · सुन्दरकाण्ड · सर्ग 1",
    "source_url": "https://www.valmikiramayan.net/utf8/sundara/sarga1/sundarasans1.htm",
    "translator": "K. M. K. Murthy, valmikiramayan.net (Sanskrit with English)",
    "hashtags": ["#ramayan", "#hanuman", "#shorts", "#sundarkand"], "keywords": ["हनुमान", "Surasa"],
    "key_quote": "Hanuma entered her mouth and came out", "shloka": {"number": "5-1-160", "sanskrit": "स सागरम् ||५-१-१६०"},
    "pictures": [{"url": "https://upload.wikimedia.org/x.jpg", "credit": {
        "source": "Wikimedia Commons", "title": "Hanuman and Surasa", "credit": "Unknown artist",
        "license": "Public domain", "url": "https://commons.wikimedia.org/wiki/File:Hanuman_and_Surasa.jpg"}}],
}
MEDIA = "https://res.cloudinary.com/demo/video/upload/v1/mool-katha/lite-20261006-1340-abc.mp4"


class PayloadTest(unittest.TestCase):
    def test_youtube_payload(self):
        payload = publisher.youtube_payload("yt1", EP, DUE, MEDIA)
        self.assertEqual(payload["channelId"], "yt1")
        self.assertEqual(payload["mode"], "customScheduled")
        self.assertEqual(payload["dueAt"], "2026-10-06T12:00:00+00:00")
        self.assertEqual(payload["assets"], [{"video": {"url": MEDIA}}])
        meta = payload["metadata"]["youtube"]
        self.assertEqual(meta["title"], EP["title"])
        self.assertEqual((meta["privacy"], meta["madeForKids"], meta["categoryId"]), ("public", False, "27"))
        text = payload["text"]
        self.assertIn(EP["citation"], text)
        self.assertIn(EP["source_url"], text)
        self.assertIn("Hanuman and Surasa", text)
        self.assertIn("आवाज़ AI से बनाई गई है", text)
        self.assertNotIn("<", text)
        self.assertLessEqual(len(text.encode()), publisher.DESCRIPTION_BYTES)

    def test_instagram_payload_is_a_reel_with_the_shloka(self):
        payload = publisher.instagram_payload("ig1", EP, DUE, MEDIA)
        self.assertEqual(payload["metadata"], {"instagram": {"type": "reel", "shouldShareToFeed": True,
                                                             "isAiGenerated": True}})
        caption = payload["text"]
        self.assertIn("श्लोक 5-1-160", caption)
        self.assertIn("स सागरम्", caption)
        self.assertIn("@moolkatha.hindi", caption)
        self.assertNotIn("#shorts", caption)
        self.assertTrue(caption.rstrip().endswith("#sundarkand"))
        self.assertLessEqual(len(caption), publisher.CAPTION_CHARS)

    def test_caption_falls_back_to_the_quote_without_sanskrit(self):
        caption = publisher.instagram_caption(EP | {"shloka": None})
        self.assertIn("“Hanuma entered her mouth and came out”", caption)

    def test_long_description_is_cut_to_fit(self):
        caption = publisher.instagram_caption(EP | {"description": "क" * 5000})
        self.assertLessEqual(len(caption), publisher.CAPTION_CHARS)
        self.assertIn("@moolkatha.hindi", caption)


class ScheduleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = Path(self.tmp.name) / "v.mp4"
        self.video.write_bytes(b"video-bytes")

    def tearDown(self):
        self.tmp.cleanup()

    def _patches(self, rows_by_channel, created):
        def buffer(query, variables):
            created.append(variables["input"])
            return {"createPost": {"__typename": "PostActionSuccess",
                                   "post": {"id": f"p{len(created)}", "status": "scheduled",
                                            "dueAt": variables["input"]["dueAt"]}}}

        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [b"video-bytes"]
        return [
            mock.patch("ytc.publish._buffer", side_effect=buffer),
            mock.patch("ytc.publish.host_video", return_value=MEDIA),
            mock.patch("ytc.publish.unhost_video"),
            mock.patch("ytc.publish.posts", side_effect=lambda since=None, channel_id=None: rows_by_channel.get(channel_id, [])),
            mock.patch("lite.publisher.requests.get", return_value=response),
        ]

    def test_creates_both_posts_once_for_the_slot(self):
        created = []
        patches = self._patches({}, created)
        for p in patches:
            p.start()
        try:
            record = publisher.schedule(EP, self.video, DUE, "yt1", "ig1", NOW)
        finally:
            for p in patches:
                p.stop()
        self.assertEqual([c["channelId"] for c in created], ["yt1", "ig1"])
        self.assertEqual(record["youtube"]["id"], "p1")
        self.assertEqual(record["instagram"]["id"], "p2")
        self.assertEqual(record["media_public_id"],
                         f"mool-katha/{EP['id']}-{hashlib.sha256(b'video-bytes').hexdigest()[:16]}")

    def test_existing_post_with_same_video_is_adopted_not_duplicated(self):
        created = []
        existing = {"id": "old-yt", "status": "scheduled", "assets": [{"source": MEDIA}]}
        patches = self._patches({"yt1": [existing]}, created)
        for p in patches:
            p.start()
        try:
            record = publisher.schedule(EP, self.video, DUE, "yt1", "ig1", NOW)
        finally:
            for p in patches:
                p.stop()
        self.assertEqual([c["channelId"] for c in created], ["ig1"])
        self.assertEqual(record["youtube"], {"id": "old-yt", "status": "scheduled", "adopted": True})

    def test_hosted_bytes_must_match(self):
        created = []
        patches = self._patches({}, created)
        for p in patches:
            p.start()
        try:
            with mock.patch("lite.publisher.sha256", return_value="different"):
                with self.assertRaises(RuntimeError):
                    publisher.schedule(EP, self.video, DUE, "yt1", "ig1", NOW)
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(created, [])

    def test_readback_checks_destination_video_and_due_time(self):
        record = {"media_url": MEDIA, "due_at": DUE.isoformat(), "youtube": {"id": "p1"}, "instagram": {"id": "p2"}}
        rows = {"yt1": [{"id": "p1", "channelId": "yt1", "status": "scheduled", "dueAt": "2026-10-06T17:30:00+05:30",
                         "assets": [{"source": MEDIA}]}],
                "ig1": [{"id": "p2", "channelId": "ig1", "status": "scheduled", "dueAt": "2026-10-06T13:00:00+00:00",
                         "assets": [{"source": MEDIA}]}]}
        with mock.patch("ytc.publish.posts", side_effect=lambda since=None, channel_id=None: rows[channel_id]):
            result = publisher.readback(record, "yt1", "ig1", NOW)
        self.assertTrue(result["youtube"]["ok"])
        self.assertFalse(result["instagram"]["ok"])  # due an hour late


if __name__ == "__main__":
    unittest.main()
