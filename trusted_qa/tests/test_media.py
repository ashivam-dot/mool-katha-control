"""Real short MP4s exercise every-frame visual checks and brief anomaly holds."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trusted_qa.common import QaHold, digest_file
from trusted_qa.media import decode_complete_mp4, extract_full_final_audio, make_visual_evidence
from trusted_qa.reviewer import FrameBatchReview, gemini_review_frame_batches


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


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools are required")
class MediaEvidenceTests(unittest.TestCase):
    def test_full_decode_audio_and_hash_bound_visual_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "sample.mp4"
            result = subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=25:duration=2",
                "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
                "-t", "2", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "38",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video),
            ], capture_output=True, timeout=90, check=False)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            decoder = decode_complete_mp4(video, 2.0)
            self.assertEqual(decoder["decode_exit_code"], 0)
            self.assertGreaterEqual(decoder["decoded_frame_count"], 45)
            audio_duration = extract_full_final_audio(video, root / "full.wav", 2.0)
            self.assertAlmostEqual(audio_duration, 2.0, delta=0.5)
            episode = root / "ep004"
            episode.mkdir()
            visual = make_visual_evidence(video, episode,
                                          [{"start": 0.0, "end": 2.0}],
                                          2.0, digest_file(video), root / "audit")
            self.assertGreaterEqual(len(visual.sample_times_seconds), 3)
            self.assertEqual(digest_file(episode / visual.contact_sheet_file),
                             visual.contact_sheet_sha256)
            self.assertEqual(len(visual.readable_crops), 3)
            for crop in visual.readable_crops:
                self.assertEqual(digest_file(episode / crop["file"]), crop["sha256"])
            self.assertEqual(visual.frame_batches[0]["start_index"], 1)
            self.assertEqual(visual.frame_batches[-1]["end_index"], decoder["decoded_frame_count"])
            self.assertEqual(digest_file(visual.frame_audit_path), visual.frame_audit_sha256)
            for batch in visual.frame_batches:
                self.assertEqual(digest_file(episode / batch["file"]), batch["sha256"])
            candidate = SimpleNamespace(episode_dir=episode,
                                        hashes={"video": digest_file(video)})
            issued = iter(enumerate(visual.frame_batches, 1))

            def respond(_request, timeout: int):
                number, batch = next(issued)
                decision = {"decision": "clear", "uncertainty": "low",
                            "checked_indices": list(range(batch["start_index"],
                                                          batch["end_index"] + 1)),
                            "defect_indices": [],
                            "notes": "Synthetic provider fixture checked the numbered sheet only."}
                provider = {"responseId": f"frame-batch-test-{number}",
                            "modelVersion": "gemini-test-version",
                            "candidates": [{"finishReason": "STOP", "content": {"parts": [
                                {"text": json.dumps(decision)}]}}]}
                return _Response(json.dumps(provider).encode())

            with patch("urllib.request.urlopen", side_effect=respond):
                frame_review = gemini_review_frame_batches(candidate, visual, root / "frame-review",
                                                           key="test-only", model="gemini-test-model")
            self.assertEqual(len(frame_review.records), len(visual.frame_batches))
            self.assertEqual(frame_review.signed_summary(candidate, visual)["decoded_frame_count"],
                             decoder["decoded_frame_count"])
            with self.assertRaisesRegex(QaHold, "missing or excessive batches"):
                FrameBatchReview(frame_review.records[:-1]).recheck(candidate, visual)
            first_batch = visual.frame_batches[0]
            all_indices = list(range(first_batch["start_index"],
                                     first_batch["end_index"] + 1))
            for label, indices, uncertainty in (
                ("omitted-index", all_indices[:-1], "low"),
                ("material-uncertainty", all_indices, "material"),
            ):
                with self.subTest(label=label):
                    decision = {"decision": "clear", "uncertainty": uncertainty,
                                "checked_indices": indices, "defect_indices": [],
                                "notes": "A synthetic provider claimed visual coverage of the frame sheet."}
                    provider = {"responseId": f"frame-batch-{label}",
                                "modelVersion": "gemini-test-version",
                                "candidates": [{"finishReason": "STOP", "content": {"parts": [
                                    {"text": json.dumps(decision)}]}}]}
                    with patch("urllib.request.urlopen",
                               return_value=_Response(json.dumps(provider).encode())):
                        with self.assertRaisesRegex(QaHold,
                                                    "incomplete visual coverage, defect, or uncertainty"):
                            gemini_review_frame_batches(candidate, visual,
                                                        root / f"frame-review-{label}",
                                                        key="test-only", model="gemini-test-model")
            frame_review.records[0].response_path.write_bytes(b"tampered provider response")
            with self.assertRaisesRegex(QaHold, "model evidence changed"):
                frame_review.recheck(candidate, visual)

    def test_dark_card_with_visible_contrast_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "dark-card.mp4"
            result = subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=black:size=1080x1920:rate=25:duration=2",
                "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
                "-vf", "drawbox=x=64:y=928:w=952:h=18:color=white:t=fill",
                "-t", "2", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "25",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video),
            ], capture_output=True, timeout=90, check=False)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            episode = root / "ep004"
            episode.mkdir()
            visual = make_visual_evidence(video, episode, [{"start": 0.0, "end": 2.0}],
                                          2.0, digest_file(video), root / "audit")
            audit = json.loads(visual.frame_audit_path.read_text())
            self.assertTrue(all(frame["mean_luma"] < 5 and frame["stddev_luma"] >= 3
                                for frame in audit["frames"]))

    def test_single_unsampled_blank_frame_holds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "one-flash.mp4"
            result = subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=25:duration=2",
                "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
                "-vf", "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable=eq(n\\,10)",
                "-t", "2", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "25",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video),
            ], capture_output=True, timeout=90, check=False)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            episode = root / "ep004"
            episode.mkdir()
            with self.assertRaisesRegex(QaHold, "frame 11: blank or uniform"):
                make_visual_evidence(video, episode, [{"start": 0.0, "end": 2.0}],
                                     2.0, digest_file(video), root / "audit")


if __name__ == "__main__":
    unittest.main()
