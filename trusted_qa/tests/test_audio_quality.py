"""Full-audio quality observations stay model-labeled and hold on uncertainty."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trusted_qa.audio_quality import ASPECTS, gemini_voice_quality
from trusted_qa.common import QaHold, digest_file


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


def _provider_response(*, concern: bool = False) -> bytes:
    decision = {"decision": "hold" if concern else "clear",
                "uncertainty": "material" if concern else "low",
                "summary": "Model-observed Hindi speech quality is uncertain around a spoken name."
                           if concern else "Model-observed Hindi speech is clear in this synthetic test response.",
                "uncertainty_notes": "This is a model observation and cannot replace native human listening.",
                "observations": [
                    {"aspect": aspect, "status": "concern" if concern and aspect == "sacred_names" else "clear",
                     "start_seconds": 0.1, "end_seconds": 1.9,
                     "detail": "Synthetic provider response for audio-quality validation only."}
                    for aspect in sorted(ASPECTS)]}
    provider = {"responseId": "quality-request-1234", "modelVersion": "gemini-test-version",
                "candidates": [{"finishReason": "STOP", "content": {"parts": [
                    {"text": json.dumps(decision, ensure_ascii=False)}]}}]}
    return json.dumps(provider, ensure_ascii=False).encode("utf-8")


class AudioQualityTests(unittest.TestCase):
    def test_clear_model_observation_is_privately_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "full.wav"
            audio.write_bytes(b"synthetic full audio bytes")
            raw = _provider_response()
            with patch("urllib.request.urlopen", return_value=_Response(raw)):
                quality = gemini_voice_quality(audio, 2.0, "a" * 64,
                                               root / "episode", root / "private",
                                               key="test-only", model="gemini-test-model")
            self.assertEqual(quality.record["basis"],
                             "Gemini model observation of actual full final audio; not human listening")
            self.assertEqual(quality.record["response_sha256"], digest_file(quality.response_path))
            self.assertEqual(quality.model_call["request_id"], "quality-request-1234")
            quality.recheck("a" * 64)
            self.assertEqual(digest_file(quality.episode_record_path), quality.record_sha256)
            quality.episode_record_path.write_bytes(b"tampered episode observation")
            with self.assertRaisesRegex(QaHold, "observation changed"):
                quality.recheck("a" * 64)
            quality.episode_record_path.write_bytes(quality.record_path.read_bytes())
            quality.response_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(QaHold, "provider evidence changed"):
                quality.recheck("a" * 64)

    def test_quality_concern_records_raw_response_then_holds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "full.wav"
            audio.write_bytes(b"synthetic full audio bytes")
            raw = _provider_response(concern=True)
            with patch("urllib.request.urlopen", return_value=_Response(raw)):
                with self.assertRaisesRegex(QaHold, "voice-quality concern"):
                    gemini_voice_quality(audio, 2.0, "a" * 64,
                                         root / "episode", root / "private",
                                         key="test-only", model="gemini-test-model")
            self.assertEqual((root / "private" / "audio-quality-response.json").read_bytes(), raw)
            observation = json.loads((root / "private" / "audio-quality-observation.json").read_text())
            self.assertEqual(observation["decision"]["uncertainty"], "material")


if __name__ == "__main__":
    unittest.main()
