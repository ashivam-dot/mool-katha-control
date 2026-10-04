from __future__ import annotations

import io
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from trusted_monitor import public


NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
YOUTUBE_ID = "ABCDEFGHIJK"
INSTAGRAM_CODE = "DeCLARCCcgC"


def signed_pair(due: datetime) -> dict:
    return {"episodes": [{"episode_id": "ep022", "due_at": due.isoformat(),
                           "posts": {"youtube": {"post_id": "yt-post-22", "status": "sent", "public_link":
                                                  f"https://www.youtube.com/shorts/{YOUTUBE_ID}"},
                                     "instagram": {"post_id": "ig-post-22", "status": "sent", "public_link":
                                                   f"https://www.instagram.com/reel/{INSTAGRAM_CODE}/"}}}]}


class Response:
    status = 200

    def __init__(self, body: str, url: str) -> None:
        self.body = io.BytesIO(body.encode())
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def geturl(self) -> str:
        return self.url

    def read(self, size: int) -> bytes:
        return self.body.read(size)


class PublicDeliveryTests(unittest.TestCase):
    def test_recent_exact_pair_is_visible_on_owned_public_pages(self) -> None:
        feed = {"channel_id": public.YOUTUBE_CHANNEL_ID, "recent_videos": [{"id": YOUTUBE_ID}]}
        with patch.object(public, "instagram_visible", return_value=True) as probe:
            result = public.inspect(NOW, signed_pair(NOW - timedelta(hours=1)), feed,
                                    instagram_probe=probe)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["episodes"][0]["youtube_id"], YOUTUBE_ID)
        self.assertEqual(result["episodes"][0]["instagram_shortcode"], INSTAGRAM_CODE)
        probe.assert_called_once_with("reel", INSTAGRAM_CODE)

    def test_sent_links_are_not_enough_when_public_ownership_is_unproved(self) -> None:
        wrong_feed = {"channel_id": "other-channel", "recent_videos": [{"id": YOUTUBE_ID}]}
        result = public.inspect(NOW, signed_pair(NOW - timedelta(hours=1)), wrong_feed,
                                instagram_probe=lambda *_: False)
        self.assertEqual(result["status"], "alert")
        self.assertEqual(len(result["issues"]), 2)
        self.assertFalse(result["episodes"][0]["youtube_visible"])
        self.assertFalse(result["episodes"][0]["instagram_visible"])

    def test_archived_proof_keeps_old_pair_verified_without_repeated_page_reads(self) -> None:
        due = NOW - timedelta(days=5)
        pair = signed_pair(due)
        fresh = {"episode_id": "ep022", "youtube_id": YOUTUBE_ID, "youtube_visible": True,
                 "instagram_shortcode": INSTAGRAM_CODE, "instagram_visible": True}
        proof = public.build_proof(pair["episodes"][0], fresh, (due + timedelta(hours=1)).isoformat())
        result = public.inspect(NOW, pair, None, proofs={"ep022": proof},
                                instagram_probe=lambda *_: self.fail("archived post was probed"))
        self.assertEqual(result["checked_episodes"], 1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["episodes"][0]["proof_status"], "archived")

    def test_unverified_old_pair_keeps_alerting_after_public_feed_window(self) -> None:
        result = public.inspect(NOW, signed_pair(NOW - timedelta(days=5)), None,
                                instagram_probe=lambda *_: self.fail("old post was probed"))
        self.assertEqual(result["status"], "alert")
        self.assertEqual(result["episodes"][0]["proof_status"], "expired")

    def test_instagram_200_shell_must_name_exact_owned_reel(self) -> None:
        url = f"https://www.instagram.com/reel/{INSTAGRAM_CODE}/"
        good = (f'<meta property="og:url" content="https://www.instagram.com/'
                f'{public.INSTAGRAM_HANDLE}/reel/{INSTAGRAM_CODE}/">')
        with patch.object(public.urllib.request, "urlopen", return_value=Response(good, url)):
            self.assertTrue(public.instagram_visible("reel", INSTAGRAM_CODE))
        wrong = good.replace(public.INSTAGRAM_HANDLE, "other.account")
        with patch.object(public.urllib.request, "urlopen", return_value=Response(wrong, url)):
            self.assertFalse(public.instagram_visible("reel", INSTAGRAM_CODE))
        with patch.object(public.urllib.request, "urlopen", return_value=Response("sign in", url)):
            self.assertFalse(public.instagram_visible("reel", INSTAGRAM_CODE))


if __name__ == "__main__":
    unittest.main()
