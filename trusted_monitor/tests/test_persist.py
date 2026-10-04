"""Durable analytics contain only allowlisted metrics and public links."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trusted_monitor import persist


CAPTURED = datetime(2026, 10, 4, 8, 0, tzinfo=timezone.utc).isoformat()


def channel(service: str) -> dict:
    link = ("https://www.youtube.com/shorts/ABCDEFGHIJK" if service == "youtube"
            else "https://www.instagram.com/reel/ABC123/")
    return {"id": persist.BUFFER_CHANNELS[service], "service": service,
            "sent_posts_30d": 1, "counts": {"sent": 1, "scheduled": 0, "error": 0, "draft": 0},
            "metrics_30d": {kind: {"total": 10 if kind == "views" else None,
                                   "reported_posts": 1 if kind == "views" else 0}
                            for kind in persist.METRICS},
            "posts": [{"id": service + "-post-1", "sent_at": CAPTURED,
                       "external_link": link, "metrics_updated_at": CAPTURED,
                       "metrics": {"views": 10}, "raw_provider_text": "DO_NOT_SAVE"}]}


def report() -> dict:
    return {"schema": "mool_katha_control_monitor_v1", "captured_at": CAPTURED,
            "ist_date": "2026-10-04", "issues": ["DO_NOT_SAVE"],
            "sources": {"buffer": {"organization_id": persist.BUFFER_ORG_ID,
                                    "channels": {service: channel(service)
                                                 for service in ("youtube", "instagram")}},
                        "youtube_owned": {"channel": {"channel_id": persist.YOUTUBE_ID,
                                                      "title": "DO_NOT_SAVE", "subscribers": None,
                                                      "views": 1000, "videos": 3},
                                          "last_28_days": {"start": "2026-09-06", "end": "2026-10-03",
                                                           "daily": [{"day": "2026-10-03", "views": 12,
                                                                      "subscribersGained": 1,
                                                                      "subscribersLost": 0}]}},
                        "youtube_public": {"channel_id": persist.YOUTUBE_ID,
                                           "recent_videos": [{"id": "ABCDEFGHIJK",
                                                              "published": CAPTURED, "views": 20}]},
                        "cloudinary": {"api_secret": "DO_NOT_SAVE"}}}


class PersistenceTests(unittest.TestCase):
    def test_snapshot_is_append_only_and_drops_provider_text_and_secrets(self) -> None:
        snapshot = persist.extract(report(), "37190534024-1")
        self.assertIsNotNone(snapshot)
        self.assertEqual(set(snapshot["sources"]), {"buffer", "youtube_owned", "youtube_public"})
        self.assertNotIn("DO_NOT_SAVE", json.dumps(snapshot))
        self.assertEqual(snapshot["sources"]["buffer"]["instagram"]["metrics_30d"]["reach"],
                         {"total": None, "reported_posts": 0})
        with tempfile.TemporaryDirectory() as location:
            target = persist.write(snapshot, Path(location) / "analytics")
            self.assertEqual(target.name, "37190534024-1.json")
            self.assertEqual(json.loads(target.read_text()), snapshot)
            with self.assertRaisesRegex(persist.PersistError, "already exists"):
                persist.write(snapshot, Path(location) / "analytics")

    def test_noncanonical_link_or_false_metric_coverage_is_rejected(self) -> None:
        source = report()
        source["sources"]["buffer"]["channels"]["instagram"]["posts"][0]["external_link"] = "https://evil.example/secret"
        with self.assertRaisesRegex(persist.PersistError, "public link"):
            persist.extract(source, "37190534024-1")
        source = report()
        source["sources"]["buffer"]["channels"]["instagram"]["metrics_30d"]["views"]["reported_posts"] = 2
        with self.assertRaisesRegex(persist.PersistError, "coverage exceeds"):
            persist.extract(source, "37190534024-1")

    def test_partial_analytics_can_be_saved_without_unavailable_reason(self) -> None:
        source = report()
        source["sources"]["buffer"] = {"status": "unavailable", "reason": "secret provider text"}
        source["sources"]["youtube_owned"] = {"status": "unavailable", "reason": "secret provider text"}
        snapshot = persist.extract(source, "37190534024-1")
        self.assertEqual(set(snapshot["sources"]), {"youtube_public"})
        self.assertNotIn("secret provider text", json.dumps(snapshot))
        source["sources"]["youtube_public"] = {"status": "unavailable"}
        self.assertIsNone(persist.extract(source, "37190534024-1"))


if __name__ == "__main__":
    unittest.main()
