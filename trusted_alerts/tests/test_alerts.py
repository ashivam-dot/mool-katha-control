"""Dedupe, retry, pinned-run identity, and first-slot schedule coverage."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from trusted_alerts import __main__ as alerts


NOW = datetime(2026, 10, 4, 14, 0, tzinfo=timezone.utc)
RUN_URL = "https://github.com/ashivam-dot/mool-katha-control/actions/runs/37190291404"


class FakeIssues:
    def __init__(self):
        self.rows = []
        self.calls = []

    def api(self, method, path, token, payload=None):
        self.calls.append((method, path))
        if method == "GET" and path.startswith("/issues?"):
            return [row.copy() for row in self.rows if row.get("state") == "open"]
        if method == "POST" and path == "/issues":
            row = {"number": len(self.rows) + 1, "title": payload["title"],
                   "body": payload["body"], "state": "open"}
            self.rows.append(row)
            return row.copy()
        if method == "PATCH" and path.startswith("/issues/"):
            number = int(path.split("/")[-1])
            row = next(row for row in self.rows if row["number"] == number)
            row.update(payload)
            return row.copy()
        raise AssertionError((method, path))


class AlertTests(unittest.TestCase):
    def test_one_phone_push_per_active_incident_and_issue_closes_on_recovery(self):
        fake = FakeIssues()
        pushed = []
        with (patch.object(alerts, "_api", side_effect=fake.api),
              patch.object(alerts, "_phone", side_effect=lambda *_args: pushed.append(True))):
            args = {"run_url": RUN_URL, "now": NOW, "token": "secret-gh", "topic": "private_topic_123"}
            first = alerts.sync("pinned-qa", ["Pinned QA failure: latest run did not succeed"], **args)
            repeat = alerts.sync("pinned-qa", ["Pinned QA failure: latest run did not succeed"], **args)
            changed = alerts.sync("pinned-qa", ["Pinned QA failure: new condition"], **args)
            recovered = alerts.sync("pinned-qa", [], **args)
        self.assertEqual((first, repeat, changed, recovered),
                         ("notified", "already_notified", "notified", "resolved"))
        self.assertEqual(len(pushed), 2)
        self.assertEqual(len(fake.rows), 1)
        self.assertEqual(fake.rows[0]["state"], "closed")
        self.assertNotIn("secret-gh", fake.rows[0]["body"])

    def test_failed_phone_stays_pending_and_retries_without_new_issue(self):
        fake = FakeIssues()
        calls = []

        def phone(*_args):
            calls.append(True)
            if len(calls) == 1:
                raise alerts.AlertError("phone notification failed; private issue remains open")

        with (patch.object(alerts, "_api", side_effect=fake.api),
              patch.object(alerts, "_phone", side_effect=phone)):
            args = {"run_url": RUN_URL, "now": NOW, "token": "secret-gh", "topic": "private_topic_123"}
            with self.assertRaisesRegex(alerts.AlertError, "private issue remains open"):
                alerts.sync("control-monitor", ["Control monitor failure: latest run did not succeed"], **args)
            self.assertIn("pending -->", fake.rows[0]["body"])
            self.assertEqual(alerts.sync("control-monitor", [
                "Control monitor failure: latest run did not succeed"], **args), "notified")
        self.assertEqual(len(fake.rows), 1)
        self.assertEqual(len(calls), 2)
        self.assertIn("sent -->", fake.rows[0]["body"])

    def test_unpinned_qa_run_does_not_open_issue(self):
        event = {"repository": {"full_name": alerts.REPOSITORY}, "workflow_run": {
            "name": "Signed independent QA", "head_branch": "qa-v5", "head_sha": "f" * 40,
            "head_repository": {"full_name": alerts.REPOSITORY},
            "status": "completed", "conclusion": "failure"}}
        with patch.object(alerts, "_api", side_effect=AssertionError("must not query")):
            self.assertEqual(alerts.from_workflow_run(event, token="secret", topic="private_topic_123",
                                                       now=NOW), "ignored")

    def test_pinned_qa_failure_notifies_but_newer_success_clears_stale_event(self):
        run = {"name": "Signed independent QA", "head_branch": "qa-v5",
               "head_sha": alerts.QA_SHA, "head_repository": {"full_name": alerts.REPOSITORY},
               "status": "completed", "conclusion": "failure", "id": 1,
               "created_at": "2026-10-04T12:00:00Z", "html_url": RUN_URL}
        event = {"repository": {"full_name": alerts.REPOSITORY}, "workflow_run": run}
        called = []

        def fake_sync(category, messages, **kwargs):
            called.append((category, messages))
            return "recorded"

        with (patch.object(alerts, "_runs", return_value=[run]),
              patch.object(alerts, "sync", side_effect=fake_sync)):
            alerts.from_workflow_run(event, token="secret", topic="private_topic_123", now=NOW)
        self.assertEqual(called, [("pinned-qa", ["Pinned QA failure: latest run did not succeed"])])
        called.clear()
        success = run | {"id": 2, "created_at": "2026-10-04T13:00:00Z", "conclusion": "success"}
        with (patch.object(alerts, "_runs", return_value=[run, success]),
              patch.object(alerts, "sync", side_effect=fake_sync)):
            alerts.from_workflow_run(event, token="secret", topic="private_topic_123", now=NOW)
        self.assertEqual(called, [("pinned-qa", [])])

    def test_feedback_failure_alerts_separately_and_newer_success_resolves(self):
        run = {"name": "Independent QA failure feedback", "head_branch": "main",
               "head_sha": "a" * 40, "head_repository": {"full_name": alerts.REPOSITORY},
               "status": "completed", "conclusion": "failure", "id": 1,
               "created_at": "2026-10-04T12:00:00Z", "html_url": RUN_URL}
        event = {"repository": {"full_name": alerts.REPOSITORY}, "workflow_run": run}
        called = []

        def fake_sync(category, messages, **kwargs):
            called.append((category, messages))
            return "recorded"

        with (patch.object(alerts, "_runs", return_value=[run]),
              patch.object(alerts, "sync", side_effect=fake_sync)):
            alerts.from_workflow_run(event, token="secret", topic="private_topic_123", now=NOW)
        self.assertEqual(called, [("qa-feedback", ["QA hold feedback failure: latest run did not succeed"])])
        called.clear()
        success = run | {"id": 2, "created_at": "2026-10-04T13:00:00Z", "conclusion": "success"}
        with (patch.object(alerts, "_runs", return_value=[run, success]),
              patch.object(alerts, "sync", side_effect=fake_sync)):
            alerts.from_workflow_run(event, token="secret", topic="private_topic_123", now=NOW)
        self.assertEqual(called, [("qa-feedback", [])])

    def test_first_dispatch_slot_warms_up_then_missing_schedule_alerts(self):
        seen = []

        def fake_api(method, path, token, payload=None):
            if path.startswith("/actions/workflows/") and "/runs?" not in path:
                filename = path.rsplit("/", 1)[-1]
                return {"path": f".github/workflows/{filename}", "state": "active",
                        "created_at": "2026-10-04T03:40:02Z"}
            if path.startswith("/commits?"):
                return [{"commit": {"committer": {"date": "2026-10-04T06:41:41Z"}}}]
            if "/runs?" in path:
                filename = path.split("/")[3]
                if filename == "dispatch-release-qa.yml":
                    return {"workflow_runs": []}
                branch = "qa-v5" if filename == "release-qa.yml" else "main"
                sha = alerts.QA_SHA if filename == "release-qa.yml" else "f" * 40
                event = "workflow_dispatch" if filename == "release-qa.yml" else "schedule"
                return {"workflow_runs": [{"id": 1, "event": event, "head_branch": branch,
                                           "head_sha": sha, "created_at": "2026-10-04T13:00:00Z"}]}
            raise AssertionError(path)

        def fake_sync(category, messages, **_kwargs):
            seen.append((category, messages))
            return "recorded"

        with (patch.object(alerts, "_api", side_effect=fake_api),
              patch.object(alerts, "sync", side_effect=fake_sync)):
            alerts.check_missed(token="secret", topic="private_topic_123", now=NOW, gate_enabled=False)
            self.assertEqual(next(messages for category, messages in seen
                                  if category == "qa-dispatch-missed"), [])
            seen.clear()
            alerts.check_missed(token="secret", topic="private_topic_123",
                                now=datetime(2026, 10, 4, 17, 0, tzinfo=timezone.utc), gate_enabled=False)
        self.assertEqual(len(next(messages for category, messages in seen
                                  if category == "qa-dispatch-missed")), 1)


if __name__ == "__main__":
    unittest.main()
