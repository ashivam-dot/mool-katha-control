"""Pinned QA feedback must lock only the exact held producer draft."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from trusted_feedback.__main__ import FeedbackError, apply_lock, candidate, inspect_hold


EPISODE = "ep012"
VIDEO_SHA = "a" * 64


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


class ArtifactTests(unittest.TestCase):
    def test_discovery_and_content_hold_require_matching_exact_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commit = "b" * 40
            expected = {"episode_id": EPISODE, "video_sha256": VIDEO_SHA, "source_commit": commit}
            discovery = root / "result.json"
            _json(discovery, {"status": "pending", "stage": "discovery", "reason": None,
                              "source_commit": commit, "candidates": [expected]})
            self.assertEqual(candidate(discovery), expected)
            hold = root / "agent-qa-hold.json"
            _json(hold, {"status": "hold", "stage": "source_and_rights",
                         "reason": "claim c03: primary and corroboration resolve to the same final page",
                         **expected, "held_at": "2026-10-04T08:00:00+00:00"})
            self.assertEqual(inspect_hold(hold, expected), "duplicate_source")
            _json(hold, {"status": "hold", "stage": "source_and_rights",
                         "reason": "asset visual:123 has no inspectable rights evidence",
                         **expected, "held_at": "2026-10-04T08:00:00+00:00"})
            self.assertEqual(inspect_hold(hold, expected), "rights_evidence")
            _json(hold, {"status": "hold", "stage": "source_and_rights",
                         "reason": "source HTTPS fetch failed", **expected,
                         "held_at": "2026-10-04T08:00:00+00:00"})
            self.assertIsNone(inspect_hold(hold, expected))
            _json(hold, {"status": "hold", "stage": "strict_review_validation",
                         "reason": "unexpected trusted QA failure (KeyError)", **expected,
                         "held_at": "2026-10-04T08:00:00+00:00"})
            self.assertIsNone(inspect_hold(hold, expected))
            _json(hold, {"status": "hold", "stage": "frame_batch_review",
                         "reason": "frame batch model found a defect, uncertainty, or skipped frame",
                         **expected, "held_at": "2026-10-04T08:00:00+00:00"})
            self.assertEqual(inspect_hold(hold, expected), "frame_defect")

    def test_mismatched_or_multiple_discovery_keys_cannot_authorize_a_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"episode_id": EPISODE, "video_sha256": VIDEO_SHA, "source_commit": "b" * 40}
            discovery = root / "result.json"
            _json(discovery, {"status": "pending", "stage": "discovery", "reason": None,
                              "source_commit": "b" * 40, "candidates": [row, row]})
            with self.assertRaises(FeedbackError):
                candidate(discovery)
            _json(discovery, {"status": "pending", "stage": "discovery", "reason": None,
                              "source_commit": "c" * 40, "candidates": [row]})
            with self.assertRaises(FeedbackError):
                candidate(discovery)
            hold = root / "hold.json"
            _json(hold, {"status": "hold", "stage": "frame_batch_review",
                         "reason": "frame batch model found a defect, uncertainty, or skipped frame",
                         "episode_id": EPISODE, "video_sha256": "c" * 64,
                         "source_commit": "b" * 40, "held_at": "2026-10-04T08:00:00+00:00"})
            with self.assertRaises(FeedbackError):
                inspect_hold(hold, row)


class ProducerWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.remote = self.root / "remote.git"
        self.seed = self.root / "seed"
        self.source = self.root / "source"
        _git("init", "--bare", str(self.remote), cwd=self.root)
        _git("init", "-b", "main", str(self.seed), cwd=self.root)
        _git("config", "user.name", "Test", cwd=self.seed)
        _git("config", "user.email", "test@example.test", cwd=self.seed)
        folder = self.seed / "content" / "episodes" / EPISODE
        _json(folder / "evidence_pending.json", {"episode_id": EPISODE, "video_sha256": VIDEO_SHA})
        _json(folder / "work" / "manifest.json", {"id": EPISODE, "video_sha256": VIDEO_SHA})
        _git("add", ".", cwd=self.seed)
        _git("commit", "-m", "candidate", cwd=self.seed)
        self.source_commit = _git("rev-parse", "HEAD", cwd=self.seed)
        _git("remote", "add", "origin", str(self.remote), cwd=self.seed)
        _git("push", "origin", "main", cwd=self.seed)
        _git("symbolic-ref", "HEAD", "refs/heads/main", cwd=self.remote)
        _git("clone", str(self.remote), str(self.source), cwd=self.root)

    def test_exact_hold_pushes_once_and_uses_only_fixed_reason(self) -> None:
        # A producer status commit may arrive after QA discovered the draft.
        _json(self.seed / "status" / "status.json", {"result": "ok"})
        _git("add", ".", cwd=self.seed)
        _git("commit", "-m", "unrelated producer progress", cwd=self.seed)
        _git("push", "origin", "main", cwd=self.seed)
        status = apply_lock(self.source, EPISODE, VIDEO_SHA, self.source_commit,
                            "frame_defect", "123", "1")
        self.assertEqual(status, "held")
        lock = json.loads((self.source / "content" / "episodes" / EPISODE /
                           "editorial-lock.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["video_sha256"], VIDEO_SHA)
        self.assertEqual(lock["qa_run_id"], 123)
        self.assertEqual(lock["reason_code"], "frame_defect")
        self.assertNotIn("frame batch model", json.dumps(lock))
        self.assertEqual(apply_lock(self.source, EPISODE, VIDEO_SHA, self.source_commit,
                                    "frame_defect", "123", "1"), "stale_or_already_held")
        self.assertEqual(_git("rev-parse", "HEAD", cwd=self.source),
                         _git("rev-parse", "main", cwd=self.remote))

    def test_changed_candidate_or_release_state_is_never_overwritten(self) -> None:
        folder = self.seed / "content" / "episodes" / EPISODE
        _json(folder / "evidence_pending.json", {"episode_id": EPISODE, "video_sha256": VIDEO_SHA,
                                                   "editorial_update": True})
        _git("add", ".", cwd=self.seed)
        _git("commit", "-m", "change candidate", cwd=self.seed)
        _git("push", "origin", "main", cwd=self.seed)
        self.assertEqual(apply_lock(self.source, EPISODE, VIDEO_SHA, self.source_commit,
                                    "frame_defect", "123", "1"), "stale_or_already_held")
        self.assertFalse((self.source / "content" / "episodes" / EPISODE / "editorial-lock.json").exists())
        _json(folder / "publish.json", {"status": "scheduled"})
        _git("add", ".", cwd=self.seed)
        _git("commit", "-m", "release state", cwd=self.seed)
        _git("push", "origin", "main", cwd=self.seed)
        self.assertEqual(apply_lock(self.source, EPISODE, VIDEO_SHA, self.source_commit,
                                    "frame_defect", "123", "1"), "stale_or_already_held")


if __name__ == "__main__":
    unittest.main()
