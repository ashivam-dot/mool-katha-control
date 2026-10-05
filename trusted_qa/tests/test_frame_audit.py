"""Real decode and mocked provider checks for indexed all-frame QA."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from trusted_qa.common import QaHold, digest_file
from trusted_qa.frame_audit import FRAME_BYTES, _model_batch, _reject_long_identical_span, audit_and_review_frames


class _Response:
    status = 200

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit: int = -1):
        return self.data if limit < 0 else self.data[:limit]


def _provider(decision: dict) -> bytes:
    return json.dumps({"responseId": "frame-request-1234", "modelVersion": "gemini-test-version",
                       "candidates": [{"finishReason": "STOP", "content": {"parts": [
                           {"text": json.dumps(decision)}]}}]}).encode()


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools are required")
class FrameAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.video = cls.root / "motion.mp4"
        result = subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=10:duration=1",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
            "-t", "1", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "40",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(cls.video),
        ], capture_output=True, timeout=90, check=False)
        if result.returncode:
            raise AssertionError(result.stderr.decode(errors="replace"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_real_all_frame_measurement_and_indexed_sheet_contract(self) -> None:
        episode = self.root / "ep011"
        private = self.root / "private"
        episode.mkdir()
        private.mkdir()
        video_hash = digest_file(self.video)

        def fake_model(_sheet, records, _private, *, key, model, batch_number):
            self.assertEqual((key, model, batch_number), ("test-key", "gemini-test-model", 1))
            indices = [row["index"] for row in records]
            return ({"decision": "clear", "uncertainty": "low", "checked_indices": indices,
                     "defect_indices": [],
                     "notes": "I inspected each indexed tile for continuity and visible corruption.",
                     "request_sha256": "a" * 64, "response_sha256": "b" * 64,
                     "model_call": {"provider": "Google Gemini API", "model": model,
                                    "model_version": "test-version", "request_id": "frame-call-1234"}},
                    {"provider": "Google Gemini API", "model": model,
                     "model_version": "test-version", "request_id": "frame-call-1234"})

        with patch("trusted_qa.frame_audit._model_batch", side_effect=fake_model):
            evidence = audit_and_review_frames(self.video, episode, private, video_hash,
                                               10, 1.0, key="test-key", model="gemini-test-model")
        audit = json.loads(evidence.audit_path.read_text())
        self.assertEqual(audit["kind"], "all_frame_pixel_temporal_audit_v1")
        self.assertEqual(audit["decoded_frame_count"], 10)
        self.assertEqual(audit["anomalies"], [])
        self.assertEqual([row["index"] for row in audit["frames"]], list(range(1, 11)))
        self.assertTrue(all(3 <= row["stddev_luma"] <= 128 for row in audit["frames"]))
        self.assertEqual(audit["frame_batches"], [{key: evidence.review["batches"][0][key]
                                                    for key in ("file", "sha256", "start_index", "end_index",
                                                                "first_seconds", "last_seconds")}])
        self.assertEqual(evidence.review["batches"][0]["checked_indices"], [1])
        self.assertEqual(digest_file(evidence.audit_path), evidence.review["all_frame_audit_sha256"])
        evidence.recheck(video_hash, 10)
        evidence.sheet_paths[0].write_bytes(b"tampered sheet")
        with self.assertRaisesRegex(QaHold, "indexed frame sheet changed"):
            evidence.recheck(video_hash, 10)

    def test_frame_count_mismatch_holds_before_model(self) -> None:
        episode = self.root / "ep012"
        private = self.root / "private-2"
        episode.mkdir()
        private.mkdir()
        with patch("trusted_qa.frame_audit._model_batch") as model:
            with self.assertRaisesRegex(QaHold, "times differ"):
                audit_and_review_frames(self.video, episode, private, digest_file(self.video),
                                        11, 1.0, key="test-key", model="gemini-test-model")
        model.assert_not_called()

    def test_flat_decoded_frames_hold_before_model(self) -> None:
        episode = self.root / "ep013"
        private = self.root / "private-3"
        episode.mkdir()
        private.mkdir()

        def flat_frames(_video, destination):
            destination.write_bytes(bytes(10 * FRAME_BYTES))

        with patch("trusted_qa.frame_audit._raw_scaled_frames", side_effect=flat_frames), \
             patch("trusted_qa.frame_audit._model_batch") as model:
            with self.assertRaisesRegex(QaHold, "flat"):
                audit_and_review_frames(self.video, episode, private, digest_file(self.video),
                                        10, 1.0, key="test-key", model="gemini-test-model")
        model.assert_not_called()


class FrameModelContractTests(unittest.TestCase):
    def test_long_identical_frame_span_holds(self) -> None:
        frames = [{"index": index + 1, "seconds": index * 0.1, "rgb_sha256": "a" * 64}
                  for index in range(23)]
        with self.assertRaisesRegex(QaHold, "identical for over two seconds"):
            _reject_long_identical_span(frames)

    def test_provider_clear_response_binds_raw_request_and_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            private = Path(directory)
            decision = {"decision": "clear", "uncertainty": "low", "checked_indices": [1, 2],
                        "defect_indices": [],
                        "notes": "Both indexed tiles are clear with consistent visual continuity."}
            records = [{"index": 1, "seconds": 0.0}, {"index": 2, "seconds": 0.04}]
            with patch("urllib.request.urlopen", return_value=_Response(_provider(decision))):
                verdict, call = _model_batch(b"test sheet", records, private,
                                             key="test-key", model="gemini-test-model", batch_number=1)
            self.assertEqual(verdict["checked_indices"], [1, 2])
            self.assertEqual(call["request_id"], "frame-request-1234")
            self.assertEqual(verdict["request_sha256"],
                             digest_file(private / "frame-batch-001-request.json"))
            self.assertEqual(verdict["response_sha256"],
                             digest_file(private / "frame-batch-001-response.json"))

    def test_rate_limited_batch_retries_then_holds_when_exhausted(self) -> None:
        decision = {"decision": "clear", "uncertainty": "low", "checked_indices": [1, 2],
                    "defect_indices": [],
                    "notes": "Both indexed tiles are clear with consistent visual continuity."}
        records = [{"index": 1, "seconds": 0.0}, {"index": 2, "seconds": 0.04}]
        body = json.dumps({"error": {"code": 429, "details": [{"retryDelay": "12s"}]}}).encode()

        def limited() -> urllib.error.HTTPError:
            return urllib.error.HTTPError("https://example.invalid", 429, "Too Many Requests", {}, io.BytesIO(body))

        with tempfile.TemporaryDirectory() as directory:
            with patch("urllib.request.urlopen", side_effect=[limited(), _Response(_provider(decision))]), \
                 patch("trusted_qa.common.time.sleep") as slept:
                verdict, _ = _model_batch(b"test sheet", records, Path(directory),
                                          key="test-key", model="gemini-test-model", batch_number=1)
            self.assertEqual(verdict["checked_indices"], [1, 2])
            slept.assert_called_once_with(12.0)
        with tempfile.TemporaryDirectory() as directory:
            with patch("urllib.request.urlopen", side_effect=[limited() for _ in range(5)]), \
                 patch("trusted_qa.common.time.sleep"):
                with self.assertRaisesRegex(QaHold, "frame batch model request failed"):
                    _model_batch(b"test sheet", records, Path(directory),
                                 key="test-key", model="gemini-test-model", batch_number=1)
        daily = json.dumps({"error": {"code": 429, "details": [
            {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
            {"retryDelay": "29849s"}]}}).encode()
        with tempfile.TemporaryDirectory() as directory:
            with patch("urllib.request.urlopen", side_effect=[urllib.error.HTTPError(
                    "https://example.invalid", 429, "Too Many Requests", {}, io.BytesIO(daily))]), \
                 patch("trusted_qa.common.time.sleep") as slept:
                with self.assertRaisesRegex(QaHold, "daily quota spent"):
                    _model_batch(b"test sheet", records, Path(directory),
                                 key="test-key", model="gemini-test-model", batch_number=1)
            slept.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            spent = urllib.error.HTTPError("https://example.invalid", 429, "Too Many Requests", {},
                                           io.BytesIO(daily))
            with patch("urllib.request.urlopen", side_effect=[spent, _Response(_provider(decision))]) as opened, \
                 patch("trusted_qa.common.time.sleep") as slept:
                verdict, call = _model_batch(b"test sheet", records, Path(directory), key="test-key",
                                             model="gemini-first,gemini-second", batch_number=1)
            self.assertEqual(call["model"], "gemini-second")
            self.assertEqual(verdict["model_call"]["model"], "gemini-second")
            self.assertIn("/gemini-second:generateContent", opened.call_args_list[1].args[0].full_url)
            slept.assert_not_called()

    def test_uncertain_or_skipped_model_batch_holds(self) -> None:
        for decision in (
            {"decision": "hold", "uncertainty": "high", "checked_indices": [1, 2],
             "defect_indices": [2], "notes": "The second indexed tile appears corrupted and needs review."},
            {"decision": "clear", "uncertainty": "low", "checked_indices": [1],
             "defect_indices": [], "notes": "Only the first indexed tile was inspected by the model."},
        ):
            with self.subTest(decision=decision), tempfile.TemporaryDirectory() as directory:
                records = [{"index": 1, "seconds": 0.0}, {"index": 2, "seconds": 0.04}]
                with patch("urllib.request.urlopen", return_value=_Response(_provider(decision))):
                    with self.assertRaisesRegex(QaHold, "defect, uncertainty, or skipped"):
                        _model_batch(b"test sheet", records, Path(directory),
                                     key="test-key", model="gemini-test-model", batch_number=1)


if __name__ == "__main__":
    unittest.main()
