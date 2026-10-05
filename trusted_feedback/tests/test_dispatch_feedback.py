"""A dispatched feedback request must name a failed pinned QA run; every reviewed draft is inspected."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trusted_feedback.__main__ import FeedbackError, artifact_name, candidates, inspect_all, verify_run

REPO = "ashivam-dot/mool-katha-control"
TAG = "qa-v20"
SHA = "7" * 40
COMMIT = "b" * 40
ROWS = [{"episode_id": "ep020", "video_sha256": "a" * 64, "source_commit": COMMIT},
        {"episode_id": "ep022", "video_sha256": "c" * 64, "source_commit": COMMIT}]


def _run(**changes):
    return {"id": 37295717095, "path": ".github/workflows/release-qa.yml", "event": "workflow_dispatch",
            "head_branch": TAG, "head_sha": SHA, "repository": {"full_name": REPO},
            "head_repository": {"full_name": REPO}, "status": "completed", "conclusion": "failure",
            "run_attempt": 1, **changes}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class DispatchFeedbackTests(unittest.TestCase):
    def test_only_the_failed_pinned_qa_run_is_accepted(self) -> None:
        self.assertEqual(verify_run(_run(), "37295717095", REPO, TAG, SHA), "1")
        for change in ({"conclusion": "success"}, {"status": "in_progress"}, {"head_sha": "8" * 40},
                       {"head_branch": "qa-v17"}, {"path": ".github/workflows/other.yml"},
                       {"event": "push"}, {"head_repository": {"full_name": "someone/fork"}},
                       {"id": 1}, {"run_attempt": "1"}):
            with self.subTest(change=change), self.assertRaises(FeedbackError):
                verify_run(_run(**change), "37295717095", REPO, TAG, SHA)

    def test_each_reviewed_draft_is_inspected_and_only_content_holds_are_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            discovery = root / "result.json"
            _write(discovery, {"status": "pending", "stage": "discovery", "reason": None,
                               "source_commit": COMMIT, "candidates": ROWS})
            self.assertEqual(candidates(discovery), ROWS)
            held, operational = ROWS
            _write(root / "qa" / artifact_name(held) / "episode" / "private" / "agent-qa-hold.json",
                   {"status": "hold", "stage": "source_and_rights",
                    "reason": "claim c01 primary ledger passage is absent from the fetched page",
                    **held, "held_at": "2026-10-05T10:00:00+00:00"})
            _write(root / "qa" / artifact_name(operational) / "episode" / "private" / "agent-qa-hold.json",
                   {"status": "hold", "stage": "strict_review_validation",
                    "reason": "unexpected trusted QA failure (KeyError)",
                    **operational, "held_at": "2026-10-05T10:00:00+00:00"})
            self.assertEqual(inspect_all(discovery, root / "qa"), [
                {"episode": "ep020", "video_sha": "a" * 64, "source_commit": COMMIT,
                 "reason_code": "source_passage"}])

    def test_repeated_or_foreign_discovery_rows_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            discovery = Path(directory) / "result.json"
            for rows, commit in (([ROWS[0], ROWS[0]], COMMIT), (ROWS, "c" * 40), (ROWS * 3, COMMIT)):
                _write(discovery, {"status": "pending", "stage": "discovery", "reason": None,
                                   "source_commit": commit, "candidates": rows})
                with self.subTest(rows=len(rows), commit=commit[0]), self.assertRaises(FeedbackError):
                    candidates(discovery)


if __name__ == "__main__":
    unittest.main()
