"""Unexpected provider output leaves a private, inspectable hold artifact."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trusted_qa.common import QaHold
from trusted_qa.runner import github_run_context, run_one
from trusted_qa.tests.test_candidate import EPISODE, candidate_fixture


class _Response:
    status = 200

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, *_):
        return self.data


class RunnerHoldTests(unittest.TestCase):
    def test_qa_identity_is_bound_to_control_repository(self) -> None:
        values = {"GITHUB_REPOSITORY": "ashivam-dot/mool-katha-control",
                  "GITHUB_WORKFLOW_REF": "ashivam-dot/mool-katha-control/.github/workflows/qa.yml@refs/heads/main",
                  "GITHUB_WORKFLOW_SHA": "f" * 40,
                  "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}
        _, identity = github_run_context(values)
        self.assertEqual(identity, "agent:github_actions/ashivam-dot/mool-katha-control/123/1")
        values["GITHUB_WORKFLOW_REF"] = (
            "ashivam-dot/mool-katha-control/.github/workflows/qa.yml@refs/tags/qa-v14")
        self.assertEqual(github_run_context(values)[1], identity)
        values["GITHUB_REPOSITORY"] = "ashivam-dot/mool-katha"
        with self.assertRaisesRegex(QaHold, "mool-katha-control"):
            github_run_context(values)

    def test_missing_model_key_holds_before_archive_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "qa-output"
            env = {"GITHUB_REPOSITORY": "ashivam-dot/mool-katha-control",
                   "GITHUB_WORKFLOW_REF": "ashivam-dot/mool-katha-control/.github/workflows/qa.yml@refs/heads/main",
                   "GITHUB_WORKFLOW_SHA": "f" * 40,
                   "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}
            with patch("trusted_qa.runner.fetch_modal_archive") as fetch:
                with self.assertRaisesRegex(QaHold, "Gemini credential"):
                    run_one(root, EPISODE, "0" * 40, output, env=env)
            fetch.assert_not_called()
            hold = json.loads((output / "private" / "agent-qa-hold.json").read_text())
            self.assertEqual(hold["stage"], "configuration")

    def test_malformed_gemini_response_records_hold_and_raw_body(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, archive, commit = candidate_fixture(root)
            output = root / "qa-output"
            env = {"GITHUB_REPOSITORY": "ashivam-dot/mool-katha-control",
                   "GITHUB_WORKFLOW_REF": "ashivam-dot/mool-katha-control/.github/workflows/qa.yml@refs/heads/main",
                   "GITHUB_WORKFLOW_SHA": "f" * 40,
                   "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1",
                   "QA_GEMINI_API_KEY": "test-only",
                   "QA_GEMINI_ASR_MODEL": "gemini-test-model",
                   "QA_GEMINI_QUALITY_MODEL": "gemini-test-model",
                   "QA_GEMINI_AUDIO_MODEL": "gemini-test-model",
                   "QA_GEMINI_REVIEW_MODEL": "gemini-test-model",
                   "QA_WHISPER_MODEL_DIR": str(root / "model"),
                   "QA_WHISPER_MODEL_REPO": "owner/test-model",
                   "QA_WHISPER_MODEL_REVISION": "0" * 40,
                   "QA_WHISPER_MODEL_SHA256": "0" * 64}
            malformed = json.dumps({"responseId": "provider-1234", "modelVersion": "version-1234",
                                    "candidates": [42]}).encode("utf-8")

            def fake_extract(_video: Path, audio: Path, duration: float) -> float:
                audio.write_bytes(b"test-only full audio")
                return duration

            with patch("trusted_qa.runner.collect_observations", return_value=None), \
                 patch("trusted_qa.runner.extract_full_final_audio", side_effect=fake_extract), \
                 patch("trusted_qa.runner.make_visual_evidence",
                       return_value=SimpleNamespace(decoder={"decoded_frame_count": 2})), \
                 patch("trusted_qa.runner.audit_and_review_frames", return_value=None), \
                 patch("urllib.request.urlopen", return_value=_Response(malformed)):
                with self.assertRaisesRegex(QaHold, "partial or untracked"):
                    run_one(repo, EPISODE, commit, output, archive_path=archive, env=env)
            hold = json.loads((output / "private" / "agent-qa-hold.json").read_text())
            self.assertEqual(hold["stage"], "independent_asr")
            self.assertEqual(hold["status"], "hold")
            self.assertEqual((output / "private" / "gemini-asr-response.json").read_bytes(), malformed)
            self.assertFalse((output / "snapshot" / "content" / "episodes" / EPISODE /
                              "agent-release-review.json").exists())


if __name__ == "__main__":
    unittest.main()
