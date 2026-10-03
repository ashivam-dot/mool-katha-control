"""A real, tiny MP4 proves FFmpeg decodes the whole stream and samples frames."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from trusted_qa.common import digest_file
from trusted_qa.media import decode_complete_mp4, extract_full_final_audio, make_visual_evidence


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools are required")
class MediaEvidenceTests(unittest.TestCase):
    def test_full_decode_audio_and_hash_bound_visual_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "sample.mp4"
            result = subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=1080x1920:r=25:d=2",
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
                                          2.0, digest_file(video))
            self.assertGreaterEqual(len(visual.sample_times_seconds), 3)
            self.assertEqual(digest_file(episode / visual.contact_sheet_file),
                             visual.contact_sheet_sha256)
            self.assertEqual(len(visual.readable_crops), 3)
            for crop in visual.readable_crops:
                self.assertEqual(digest_file(episode / crop["file"]), crop["sha256"])


if __name__ == "__main__":
    unittest.main()
