"""Cautious Gemini observations of the complete mixed Hindi audio, never human listening."""

from __future__ import annotations

import base64
import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asr import GEMINI_ENDPOINT, MAX_GEMINI_AUDIO_BYTES
from .common import (QaHold, digest_bytes, digest_file, gemini_post, gemini_text_response,
                     json_object, require, utc_now, valid_gemini_models, write_bytes_new,
                     write_json_new)


ASPECTS = frozenset({"natural_indian_hindi", "clarity", "cadence", "sacred_names", "artifacts"})
EPISODE_OBSERVATION_FILE = "agent-audio-quality-observation.json"
MAX_QUALITY_RESPONSE_BYTES = 2 * 1024 * 1024
PROMPT = """Listen to this entire final mixed audio track as a model, not as a human or native-Hindi reviewer.
Assess natural Indian Hindi accent, intelligibility, cadence, sacred or proper-name pronunciation,
and audible synthesis/clipping/distortion artifacts. Ground each observation in an actual time span
from the supplied audio. Do not infer a name from a script: no script is provided. If speech is unclear,
an accent sounds unnatural, a name is uncertain, or the audio has material artifacts, mark a concern
or uncertainty. Never claim human listening or certainty you cannot support.
Return exactly one JSON object with keys decision ('clear' or 'hold'), uncertainty ('low' or 'material'),
summary, uncertainty_notes, and observations. observations must have exactly one object for each aspect:
natural_indian_hindi, clarity, cadence, sacred_names, artifacts. Each observation needs exactly aspect,
status ('clear', 'concern', or 'uncertain'), start_seconds, end_seconds, and detail. Cite a valid span in
seconds and describe what you heard, including the name if one was audible. A clear decision requires
all five aspects clear with low uncertainty. A low uncertainty still needs a specific uncertainty note
acknowledging that this is a model observation, not native-human approval.
"""


def _quality_decision(value: dict[str, Any], duration: float) -> None:
    require(set(value) == {"decision", "uncertainty", "summary", "uncertainty_notes", "observations"},
            "full-audio quality observation has an unexpected shape")
    require(value["decision"] in ("clear", "hold") and
            value["uncertainty"] in ("low", "material") and
            isinstance(value["summary"], str) and 30 <= len(value["summary"].strip()) <= 1500 and
            isinstance(value["uncertainty_notes"], str) and
            20 <= len(value["uncertainty_notes"].strip()) <= 1000,
            "full-audio quality observation lacks a reasoned decision or uncertainty")
    observations = value["observations"]
    require(isinstance(observations, list) and len(observations) == len(ASPECTS),
            "full-audio quality observation does not cover every aspect")
    seen: set[str] = set()
    for item in observations:
        require(isinstance(item, dict) and
                set(item) == {"aspect", "status", "start_seconds", "end_seconds", "detail"},
                "full-audio quality aspect has an unexpected shape")
        aspect = item["aspect"]
        start = item["start_seconds"]
        end = item["end_seconds"]
        require(isinstance(aspect, str) and aspect in ASPECTS and aspect not in seen and
                item["status"] in ("clear", "concern", "uncertain") and
                all(isinstance(number, (int, float)) and not isinstance(number, bool) and
                    math.isfinite(number) for number in (start, end)) and
                0 <= start < end <= duration + 0.5 and
                isinstance(item["detail"], str) and 20 <= len(item["detail"].strip()) <= 1000,
                "full-audio quality aspect is missing a grounded observation")
        seen.add(aspect)
    require(seen == ASPECTS, "full-audio quality observation omitted an aspect")


@dataclass(frozen=True)
class AudioQualityObservation:
    record: dict[str, Any]
    record_path: Path
    episode_record_path: Path
    record_sha256: str
    request_path: Path
    response_path: Path

    @property
    def model_call(self) -> dict[str, str]:
        return self.record["model_call"]

    @property
    def decision(self) -> dict[str, Any]:
        return self.record["decision"]

    def recheck(self, video_sha256: str) -> None:
        require(digest_file(self.record_path) == self.record_sha256 and
                digest_file(self.episode_record_path) == self.record_sha256 and
                json_object(self.record_path.read_bytes(), "saved audio quality observation") == self.record,
                "full-audio quality observation changed after review")
        require(digest_file(self.request_path) == self.record["request_sha256"] and
                digest_file(self.response_path) == self.record["response_sha256"],
                "full-audio quality provider evidence changed after review")
        require(self.record["input_video_sha256"] == video_sha256,
                "full-audio quality observation names another video")
        duration = self.record["audio_duration_seconds"]
        require(isinstance(duration, (int, float)) and not isinstance(duration, bool) and
                math.isfinite(duration) and duration > 0,
                "full-audio quality observation has invalid duration")
        _quality_decision(self.decision, duration)
        require(self.decision["decision"] == "clear" and self.decision["uncertainty"] == "low" and
                all(item["status"] == "clear" for item in self.decision["observations"]),
                "full-audio model found a voice-quality concern or material uncertainty")


def gemini_voice_quality(audio_path: Path, duration: float, video_sha256: str,
                         episode_dir: Path, private_audit_dir: Path, *, key: str,
                         model: str) -> AudioQualityObservation:
    """Save raw evidence and hold unless the full-audio model observation is clear."""
    require(isinstance(key, str) and bool(key.strip()), "Gemini audio-quality key is unavailable")
    require(valid_gemini_models(model), "Gemini audio-quality model must be explicitly named")
    require(isinstance(duration, (int, float)) and not isinstance(duration, bool) and
            math.isfinite(duration) and duration > 0,
            "audio-quality observation needs a measured full-audio duration")
    require(audio_path.is_file() and not audio_path.is_symlink() and
            0 < audio_path.stat().st_size <= MAX_GEMINI_AUDIO_BYTES,
            "whole final WAV is unavailable for audio-quality observation")
    wav = audio_path.read_bytes()
    audio_sha = digest_bytes(wav)
    body = {"contents": [{"role": "user", "parts": [
        {"text": PROMPT + f"\nThe complete audio duration is {duration:.3f} seconds."},
        {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(wav).decode("ascii")}},
    ]}], "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                              "maxOutputTokens": 4096}}
    request_bytes = json.dumps(body, ensure_ascii=False, separators=(",", ":"),
                               allow_nan=False).encode("utf-8")
    request_path = private_audit_dir / "audio-quality-request.json"
    response_path = private_audit_dir / "audio-quality-response.json"
    record_path = private_audit_dir / "audio-quality-observation.json"
    write_bytes_new(request_path, request_bytes)
    raw_response, model = gemini_post(model, request_bytes, key=key, timeout=240,
                                      max_bytes=MAX_QUALITY_RESPONSE_BYTES,
                                      failure="Gemini full-audio quality request failed",
                                      not_ok="Gemini full-audio quality request did not succeed")
    require(len(raw_response) <= MAX_QUALITY_RESPONSE_BYTES,
            "Gemini full-audio quality response is oversized")
    write_bytes_new(response_path, raw_response)
    provider = json_object(raw_response, "Gemini full-audio quality response")
    request_id, version, answer = gemini_text_response(provider, "Gemini full-audio quality")
    decision = json_object(answer, "Gemini full-audio quality decision")
    _quality_decision(decision, duration)
    require(digest_file(audio_path) == audio_sha,
            "whole final WAV changed during audio-quality observation")
    record = {"kind": "full_final_audio_quality_model_observation_v1",
              "basis": "Gemini model observation of actual full final audio; not human listening",
              "input_video_sha256": video_sha256, "input_audio_sha256": audio_sha,
              "audio_duration_seconds": duration, "request_sha256": digest_bytes(request_bytes),
              "response_sha256": digest_bytes(raw_response), "observed_at": utc_now(),
              "model_call": {"provider": "Google Gemini API", "model": model,
                             "model_version": version, "request_id": request_id},
              "decision": decision}
    write_json_new(record_path, record)
    episode_record_path = episode_dir / EPISODE_OBSERVATION_FILE
    write_json_new(episode_record_path, record)
    observation = AudioQualityObservation(record, record_path, episode_record_path,
                                          digest_file(record_path), request_path, response_path)
    observation.recheck(video_sha256)
    return observation
