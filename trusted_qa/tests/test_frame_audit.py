"""Real decode and mocked provider checks for indexed all-frame QA."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trusted_qa.common import QaHold, digest_file
from trusted_qa.frame_audit import (FRAME_BYTES, _model_batch, _reject_long_identical_span,
                                    audit_and_review_frames, validate_full_frame_metrics)


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
        beats = [{"start": 0.0, "end": 1.0}]

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
                                               10, 1.0, beats,
                                               key="test-key", model="gemini-test-model")
        audit = json.loads(evidence.audit_path.read_text())
        self.assertEqual(audit["kind"], "all_frame_pixel_temporal_audit_v2")
        self.assertEqual(audit["decoded_frame_count"], 10)
        self.assertEqual(audit["anomalies"], evidence.review["sample_plan"]["anomalies"])
        self.assertEqual([row["index"] for row in audit["frames"]], list(range(1, 11)))
        self.assertTrue(all(3 <= row["stddev_luma"] <= 128 for row in audit["frames"]))
        self.assertEqual(audit["frame_batches"], [{key: evidence.review["batches"][0][key]
                                                    for key in ("file", "sha256", "indices", "start_index", "end_index",
                                                                "first_seconds", "last_seconds")}])
        self.assertEqual(evidence.review["kind"], "frame_sampled_visual_review_v1")
        self.assertEqual(evidence.review["batches"][0]["checked_indices"],
                         evidence.review["sample_plan"]["sampled_indices"])
        self.assertEqual(digest_file(evidence.audit_path), evidence.review["all_frame_audit_sha256"])
        evidence.recheck(video_hash, 10, beats, 1.0)
        original_plan = evidence.review["sample_plan"]
        evidence.review["sample_plan"] = {**original_plan,
                                          "sampled_indices": original_plan["sampled_indices"][:-1]}
        with self.assertRaisesRegex(QaHold, "sampled visual review changed its plan"):
            evidence.recheck(video_hash, 10, beats, 1.0)
        evidence.review["sample_plan"] = original_plan
        evidence.sheet_paths[0].write_bytes(b"tampered sheet")
        with self.assertRaisesRegex(QaHold, "indexed frame sheet changed"):
            evidence.recheck(video_hash, 10, beats, 1.0)

    def test_frame_count_mismatch_holds_before_model(self) -> None:
        episode = self.root / "ep012"
        private = self.root / "private-2"
        episode.mkdir()
        private.mkdir()
        with patch("trusted_qa.frame_audit._model_batch") as model:
            with self.assertRaisesRegex(QaHold, "times differ"):
                audit_and_review_frames(self.video, episode, private, digest_file(self.video),
                                        11, 1.0, [{"start": 0.0, "end": 1.0}],
                                        key="test-key", model="gemini-test-model")
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
                                        10, 1.0, [{"start": 0.0, "end": 1.0}],
                                        key="test-key", model="gemini-test-model")
        model.assert_not_called()


class FrameModelContractTests(unittest.TestCase):
    @staticmethod
    def _metrics(count: int, *, interval: float = 0.1) -> list[dict]:
        return [{"index": index, "seconds": round((index - 1) * interval, 6),
                 "rgb_sha256": f"{index:064x}", "mean_luma": 100.0,
                 "stddev_luma": 20.0, "dark_fraction": 0.1,
                 "bright_fraction": 0.0, "delta_previous": 0.5 if index > 1 else 0.0,
                 "rgb_delta_previous": 0.5 if index > 1 else 0.0,
                 "rgb_delta_two_back": 0.5 if index > 2 else 0.0}
                for index in range(1, count + 1)]

    def test_near_freeze_timestamp_gap_and_single_frame_spike_hold(self) -> None:
        frozen = self._metrics(23)
        for frame in frozen[1:]:
            frame["rgb_delta_previous"] = 0.1
        with self.assertRaisesRegex(QaHold, "near-frozen"):
            validate_full_frame_metrics(frozen, 2.3)

        gap = self._metrics(10)
        gap[5]["seconds"] = 0.7
        with self.assertRaisesRegex(QaHold, "timestamp gap"):
            validate_full_frame_metrics(gap, 1.0)

        short_tail = self._metrics(10)
        with self.assertRaisesRegex(QaHold, "terminal timestamp gap"):
            validate_full_frame_metrics(short_tail, 2.0)

        spike = self._metrics(4)
        spike[1]["rgb_delta_previous"] = 10.0
        spike[2]["rgb_delta_previous"] = 10.0
        spike[2]["rgb_delta_two_back"] = 0.1
        with self.assertRaisesRegex(QaHold, "one-frame visual spike"):
            validate_full_frame_metrics(spike, 0.4)

    def test_black_or_white_frame_holds(self) -> None:
        for mean in (7.9, 247.1):
            with self.subTest(mean=mean):
                frames = self._metrics(3)
                frames[1]["mean_luma"] = mean
                with self.assertRaisesRegex(QaHold, "black, white"):
                    validate_full_frame_metrics(frames, 0.3)

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

    def test_provider_checked_indices_must_match_noncontiguous_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = [{"index": 1, "seconds": 0.0},
                       {"index": 4, "seconds": 0.3}]
            clear = {"decision": "clear", "uncertainty": "low",
                     "checked_indices": [1, 4], "defect_indices": [],
                     "notes": "Both displayed sampled tiles are clear and individually inspectable."}
            with patch("urllib.request.urlopen", return_value=_Response(_provider(clear))):
                verdict, _ = _model_batch(b"sampled sheet", records, Path(directory),
                                          key="test-key", model="gemini-test-model", batch_number=1)
            self.assertEqual(verdict["checked_indices"], [1, 4])

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
