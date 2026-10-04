from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from trusted_monitor import __main__ as monitor


NOW = datetime(2026, 10, 4, 6, 0, tzinfo=timezone.utc)
YT_BUFFER_ID = monitor.BUFFER_YOUTUBE_ID
ENV = {"GITHUB_REPOSITORY": monitor.REPOSITORY, "GITHUB_REF": "refs/heads/main",
       "GH_TOKEN": "secret-gh", "BUFFER_API_KEY": "secret-buffer", "BUFFER_ORG_ID": monitor.BUFFER_ORG_ID,
       "BUFFER_YOUTUBE_CHANNEL_ID": YT_BUFFER_ID, "BUFFER_INSTAGRAM_CHANNEL_ID": monitor.BUFFER_INSTAGRAM_ID,
       "CLOUDINARY_URL": "cloudinary://private-key:private-secret@mw0oh0v8"}


def channels():
    return [{"id": YT_BUFFER_ID, "name": "Mool Katha", "displayName": "moolkatha", "service": "youtube",
             "isDisconnected": False, "isLocked": False, "isQueuePaused": False},
            {"id": monitor.BUFFER_INSTAGRAM_ID, "name": "@moolkatha.hindi", "displayName": "Mool Katha",
             "service": "instagram", "isDisconnected": False, "isLocked": False, "isQueuePaused": False}]


def post(channel_id, service, *, status="sent"):
    return {"id": f"{service}-post-1", "channelId": channel_id, "channelService": service, "status": status,
            "sentAt": NOW.isoformat(), "dueAt": NOW.isoformat(),
            "externalLink": "https://www.youtube.com/shorts/ABCDEFGHIJK" if service == "youtube"
                            else "https://www.instagram.com/reel/ABC123/",
            "metricsUpdatedAt": NOW.isoformat(), "metrics": [{"type": "views", "value": 10},
                                                              {"type": "shares", "value": 2}]}


class MonitorTests(unittest.TestCase):
    def test_exact_buffer_channels_and_metric_coverage(self):
        def fake_buffer(query, variables, key):
            self.assertEqual(key, ENV["BUFFER_API_KEY"])
            self.assertEqual(variables["input"]["organizationId"], monitor.BUFFER_ORG_ID)
            if "query Channels" in query:
                return {"channels": channels()}
            channel_id = variables["input"]["filter"]["channelIds"][0]
            service = "youtube" if channel_id == YT_BUFFER_ID else "instagram"
            return {"posts": {"edges": [{"node": post(channel_id, service)}],
                              "pageInfo": {"hasNextPage": False, "endCursor": None}}}

        with patch.object(monitor, "_buffer", side_effect=fake_buffer):
            result = monitor.buffer(NOW, ENV)
        self.assertEqual(result["channels"]["instagram"]["metrics_30d"]["views"],
                         {"total": 10, "reported_posts": 1})
        self.assertEqual(result["channels"]["youtube"]["sent_posts_30d"], 1)
        self.assertNotIn("secret-buffer", json.dumps(result))

    def test_channel_handle_cannot_match_longer_lookalike(self):
        for name in ("@moolkatha.hindi.extra", "@notmoolkatha.hindi"):
            row = channels()[1] | {"name": name, "displayName": "Other"}
            with self.assertRaisesRegex(monitor.MonitorError, "handle mismatch"):
                monitor._channel(row, expected_id=monitor.BUFFER_INSTAGRAM_ID,
                                 service="instagram", handle=monitor.INSTAGRAM_HANDLE)

    def test_post_from_other_destination_fails_closed(self):
        wrong = post("some-other-channel", "instagram")
        with patch.object(monitor, "_buffer", return_value={"posts": {
                "edges": [{"node": wrong}], "pageInfo": {"hasNextPage": False}}}):
            with self.assertRaisesRegex(monitor.MonitorError, "destination mismatch"):
                monitor._posts("secret", monitor.BUFFER_INSTAGRAM_ID, "instagram", NOW)

    def test_release_gate_off_does_not_alert_on_skipped_runs(self):
        def fake_json(url, **_):
            if url.endswith("/runs?branch=main&per_page=30"):
                return {"workflow_runs": [{"event": "schedule", "head_branch": "main",
                                            "created_at": (NOW - timedelta(days=2)).isoformat(),
                                            "status": "completed", "conclusion": "skipped"}]}
            return {"path": ".github/workflows/signed-release.yml", "state": "active"}

        with patch.object(monitor, "_json", side_effect=fake_json):
            self.assertEqual(monitor.release(NOW, ENV)["status"], "gate_off")
            enabled = monitor.release(NOW, ENV | {"YTC_ENABLE_CONTROL_RELEASE": "1"})
        self.assertEqual(enabled["status"], "alert")
        self.assertEqual(len(enabled["issues"]), 2)

    def test_report_never_contains_provider_exception_or_secrets(self):
        with (patch.object(monitor, "release", side_effect=RuntimeError("secret-gh")),
              patch.object(monitor, "buffer", side_effect=monitor.MonitorError("Buffer query failed")),
              patch.object(monitor, "cloudinary", return_value={"cloud_name": "exact-cloud"}),
              patch.object(monitor, "youtube_owned", return_value={"channel": {"channel_id": monitor.YOUTUBE_ID}}),
              patch.object(monitor, "youtube_public", return_value={"channel_id": monitor.YOUTUBE_ID})):
            unexpected = monitor.collect(NOW, ENV)
        self.assertEqual(unexpected["sources"]["release"]["reason"], "read failed")
        self.assertNotIn("secret-gh", json.dumps(unexpected))
        with (patch.object(monitor, "release", return_value={"issues": [], "status": "gate_off"}),
              patch.object(monitor, "buffer", side_effect=monitor.MonitorError("Buffer query failed")),
              patch.object(monitor, "cloudinary", return_value={"cloud_name": "exact-cloud"}),
              patch.object(monitor, "youtube_owned", return_value={"channel": {"channel_id": monitor.YOUTUBE_ID}}),
              patch.object(monitor, "youtube_public", return_value={"channel_id": monitor.YOUTUBE_ID})):
            report = monitor.collect(NOW, ENV)
        self.assertEqual(report["status"], "alert")
        self.assertNotIn("secret", json.dumps(report))

    def test_pinned_destination_ids_reject_configuration_drift(self):
        with self.assertRaisesRegex(monitor.MonitorError, "channel IDs"):
            monitor.buffer(NOW, ENV | {"BUFFER_YOUTUBE_CHANNEL_ID": "other-channel"})
        with self.assertRaisesRegex(monitor.MonitorError, "cloud name mismatch"):
            monitor.cloudinary(ENV | {"CLOUDINARY_URL": "cloudinary://key:secret@other-cloud"})


if __name__ == "__main__":
    unittest.main()
