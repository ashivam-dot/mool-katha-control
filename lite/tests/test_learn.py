import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from lite import learn, run

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


def episode(i, work, hours_ago=60):
    return {"id": f"lite-{i}", "topic_key": f"{work}-1-{i}",
            "publish": {"due_at": (NOW - timedelta(hours=hours_ago)).isoformat(),
                        "youtube": {"id": f"yt{i}"}, "instagram": {"id": f"ig{i}"}}}


def fake_buffer(views):
    def call(query, variables):
        assert variables["input"]["filter"]["channelIds"] == ["YT", "IG"]
        edges = [{"node": {"id": pid, "channelId": "x", "metrics": [{"type": "views", "value": v},
                                                                   {"type": "unknown", "value": 1}]}}
                 for pid, v in views.items()]
        return {"posts": {"edges": edges, "pageInfo": {"hasNextPage": False, "endCursor": None}}}
    return call


class LearnTest(unittest.TestCase):
    def test_views_at_interpolates_and_waits(self):
        self.assertAlmostEqual(learn.views_at([{"age_h": 24, "views": 100}, {"age_h": 72, "views": 300}], 48), 200)
        self.assertIsNone(learn.views_at([{"age_h": 10, "views": 5}], 48))

    def test_lifts_need_samples_and_stay_bounded(self):
        self.assertEqual(learn.lifts({"gita": [100, 200, 300]})["lifts"], {})
        out = learn.lifts({"gita": [9000] * 4, "ramayana": [5] * 4})["lifts"]
        self.assertEqual(out["gita"]["lift"], learn.LIFT_RANGE[1])
        self.assertEqual(out["ramayana"]["lift"], learn.LIFT_RANGE[0])

    @mock.patch.dict(os.environ, {"BUFFER_ORG_ID": "org"})
    def test_refresh_scores_both_platforms_and_throttles(self):
        ledger = {"episodes": [episode(i, "gita") for i in range(4)] + [episode(9, "ramayana", hours_ago=-3)]}
        views = {f"yt{i}": 100 for i in range(4)} | {f"ig{i}": 900 for i in range(4)}
        learned = learn.refresh(ledger, "YT", "IG", NOW, buffer=fake_buffer(views))
        self.assertEqual(learned["scored"], 4)
        self.assertIn("gita", learned["lifts"])
        snap = ledger["performance"]["lite-0"]["snapshots"][0]
        self.assertEqual((snap["views"], snap["youtube_views"], snap["instagram_views"]), (1000, 100, 900))
        self.assertNotIn("youtube_unknown", snap)
        self.assertEqual(ledger["performance"]["lite-9"]["snapshots"], [])
        self.assertIsNone(learn.refresh(ledger, "YT", "IG", NOW + timedelta(hours=1), buffer=fake_buffer(views)))
        self.assertIn("gita", learn.report(ledger))

    def test_next_topic_prefers_the_strong_work_only_within_the_window(self):
        topics = json.loads(run.TOPICS.read_text(encoding="utf-8"))
        ledger = {"episodes": [], "skipped": {}, "pending": None, "ready": None}
        first = run.next_topic(ledger)
        self.assertEqual(first["key"], topics[0]["key"])
        window = topics[:learn.WINDOW]
        other = next((t for t in window if t["work"] != first["work"]), None)
        if other is None:
            self.skipTest("the first curated topics share one work")
        ledger["learned"] = {"lifts": {other["work"]: {"lift": 1.3}}}
        self.assertEqual(run.next_topic(ledger)["work"], other["work"])
        outside = [t["work"] for t in topics[learn.WINDOW:]]
        ledger["learned"] = {"lifts": {w: {"lift": 1.35} for w in outside if w not in {t["work"] for t in window}}}
        self.assertEqual(run.next_topic(ledger)["key"], topics[0]["key"])


if __name__ == "__main__":
    unittest.main()
