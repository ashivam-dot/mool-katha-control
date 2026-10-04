"""Two independent Hindi recognizers over the same complete final MP4 audio.

Gemini receives the extracted whole WAV in one request. The local Whisper model
receives that same whole WAV with VAD off. Their raw outputs are saved unchanged
inside each result artifact; no script text is sent to either recognizer.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import urllib.error
import urllib.request
from importlib import metadata
from pathlib import Path
from typing import Any

from .common import (QaHold, digest_file, expect_sha, gemini_post, gemini_text_response, json_object,
                     require, utc_now, write_bytes_new, write_json_new)


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
MAX_GEMINI_AUDIO_BYTES = 10 * 1024 * 1024
MAX_GEMINI_RESPONSE_BYTES = 5 * 1024 * 1024
# A 30-second window omitted spoken ep022 beats that a 15-second window recovered.
# The recognizer still receives and processes the complete final WAV, with VAD off.
WHISPER_CHUNK_SECONDS = 15


def _asr_segments(raw: Any, duration: float, provider: str) -> tuple[list[dict], str]:
    require(isinstance(raw, list) and bool(raw), f"{provider} returned no numbered speech segments")
    normalized: list[dict] = []
    previous_end = 0.0
    for index, value in enumerate(raw, 1):
        require(isinstance(value, dict), f"{provider} segment {index} is not an object")
        start = value.get("start_seconds", value.get("start"))
        end = value.get("end_seconds", value.get("end"))
        text = value.get("text")
        require(isinstance(start, (int, float)) and isinstance(end, (int, float)) and
                not isinstance(start, bool) and not isinstance(end, bool) and
                math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + 0.5 and
                start + 0.02 >= previous_end, f"{provider} segment {index} has invalid timing")
        require(isinstance(text, str) and bool(text.strip()), f"{provider} segment {index} has no transcript")
        normalized.append({"index": index, "start_seconds": float(start),
                           "end_seconds": float(end), "text": text.strip()})
        previous_end = float(end)
    transcript = " ".join(segment["text"] for segment in normalized)
    require(len(transcript) >= 8, f"{provider} transcript is too short")
    return normalized, transcript


def _result(provider: str, family: str, model: str, version: str, run_id: str,
            video_sha256: str, audio_duration: float, segments: list[dict],
            full_transcript: str, raw_response: dict, generated_at: str) -> dict:
    require(all(isinstance(value, str) and bool(value.strip()) for value in
                (provider, family, model, version, run_id, video_sha256, generated_at)),
            "ASR provenance is incomplete")
    require(isinstance(raw_response, dict) and bool(raw_response), "ASR raw response is missing")
    return {"kind": "independent_full_audio_asr_v1", "provider": provider,
            "model_family": family, "model": model, "model_version": version,
            "run_id": run_id, "generated_at": generated_at,
            "input_video_sha256": video_sha256, "processed_full_final_audio": True,
            "audio_duration_seconds": audio_duration, "segments": segments,
            "full_transcript": full_transcript, "raw_response": raw_response}


def gemini_full_audio(audio_path: Path, duration: float, video_sha256: str,
                      *, key: str, model: str, audit_path: Path | None = None) -> dict:
    require(isinstance(key, str) and bool(key.strip()), "Gemini ASR key is unavailable")
    require(isinstance(model, str) and re.fullmatch(r"gemini-[A-Za-z0-9._-]+", model) is not None,
            "Gemini ASR model must be explicitly named")
    require(audio_path.is_file() and 0 < audio_path.stat().st_size <= MAX_GEMINI_AUDIO_BYTES,
            "whole final WAV is missing or too large for the single Gemini request")
    wav = audio_path.read_bytes()
    prompt = (
        "Transcribe ALL spoken Hindi from this complete final short-video audio track. "
        "Do not use a script, translate, summarize, autocorrect names, or omit uncertain words. "
        "Return one JSON object with exactly `segments` (array of objects with numeric "
        "`start_seconds`, `end_seconds`, and verbatim Hindi `text`) and `full_transcript` "
        "(the segment texts joined with single spaces). Time every spoken utterance in seconds "
        "from the audio start. Preserve offensive or unclear readings exactly; mark unclear words "
        "as [अस्पष्ट]. Leading/trailing silence and music are not speech. "
        f"The complete supplied audio lasts {duration:.3f} seconds."
    )
    body = {"contents": [{"role": "user", "parts": [
        {"text": prompt},
        {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(wav).decode("ascii")}},
    ]}], "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                              "maxOutputTokens": 8192}}
    endpoint = f"{GEMINI_ENDPOINT}/{model}:generateContent"
    request = urllib.request.Request(endpoint, data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "x-goog-api-key": key}, method="POST")
    raw = gemini_post(request, timeout=240, max_bytes=MAX_GEMINI_RESPONSE_BYTES,
                      failure="Gemini full-audio ASR request failed",
                      not_ok="Gemini ASR request did not succeed")
    require(len(raw) <= MAX_GEMINI_RESPONSE_BYTES, "Gemini ASR response is oversized")
    if audit_path is not None:
        write_bytes_new(audit_path, raw)
    provider_response = json_object(raw, "Gemini ASR response")
    response_id, version, transcript = gemini_text_response(provider_response, "Gemini ASR")
    transcript_json = json_object(transcript, "Gemini transcript")
    require(set(transcript_json) == {"segments", "full_transcript"},
            "Gemini transcript has an unexpected shape")
    segments, joined = _asr_segments(transcript_json["segments"], duration, "Gemini")
    require(transcript_json["full_transcript"] == joined,
            "Gemini full transcript differs from its raw segments")
    return _result("Google Gemini API", "Google Gemini multimodal audio", model, version,
                   response_id, video_sha256, duration, segments, joined,
                   provider_response, utc_now())


def whisper_full_audio(audio_path: Path, duration: float, video_sha256: str,
                       *, model_dir: Path, model_repo: str, revision: str,
                       model_sha256: str,
                       github_run_id: int, github_run_attempt: int) -> dict:
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
            "Whisper model revision must be a pinned Git commit")
    require(re.fullmatch(r"[A-Za-z0-9._/-]+", model_repo) is not None and "/" in model_repo,
            "Whisper model repository is invalid")
    require(model_dir.is_dir() and not model_dir.is_symlink(),
            "pinned local Whisper model directory is unavailable")
    model_file = model_dir / "model.bin"
    require(model_file.is_file() and not model_file.is_symlink(),
            "pinned local CTranslate2 Whisper model.bin is unavailable")
    model_hash = digest_file(model_file)
    require(model_hash == expect_sha(model_sha256, "Whisper model.bin SHA-256"),
            "local Whisper model.bin differs from the pinned revision artifact")
    for filename in ("config.json", "tokenizer.json", "vocabulary.json"):
        companion = model_dir / filename
        require(companion.is_file() and not companion.is_symlink() and
                0 < companion.stat().st_size <= 8_000_000,
                f"local Whisper {filename} is missing or unsafe")
    require(isinstance(github_run_id, int) and github_run_id > 0 and
            isinstance(github_run_attempt, int) and github_run_attempt > 0,
            "local Whisper needs the actual cloud run identity")
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise QaHold("faster-whisper is unavailable on the trusted QA runner") from exc
    try:
        model = WhisperModel(str(model_dir), device="cpu", compute_type="int8")
    except (OSError, ValueError, RuntimeError) as exc:
        raise QaHold("pinned local Whisper model could not be loaded") from exc
    try:
        iterator, info = model.transcribe(str(audio_path), language="hi", beam_size=5,
                                          temperature=0, condition_on_previous_text=False,
                                          vad_filter=False, chunk_length=WHISPER_CHUNK_SECONDS)
        raw_segments = [{"id": segment.id, "start": segment.start, "end": segment.end,
                         "text": segment.text, "avg_logprob": segment.avg_logprob,
                         "no_speech_prob": segment.no_speech_prob,
                         "compression_ratio": segment.compression_ratio}
                        for segment in iterator]
    except (OSError, ValueError, RuntimeError) as exc:
        raise QaHold("local Whisper full-audio recognition failed") from exc
    require(info.language == "hi" and abs(float(info.duration) - duration) <= 0.5,
            "local Whisper did not process the expected full Hindi audio")
    raw_response = {"segments": raw_segments, "info": {
        "language": info.language, "language_probability": info.language_probability,
        "duration": info.duration, "duration_after_vad": info.duration_after_vad},
        "settings": {"language": "hi", "beam_size": 5, "temperature": 0,
                     "condition_on_previous_text": False, "vad_filter": False,
                     "chunk_length_seconds": WHISPER_CHUNK_SECONDS}}
    segments, joined = _asr_segments(raw_segments, duration, "Whisper")
    library_version = metadata.version("faster-whisper")
    version = f"{model_repo}@{revision}; faster-whisper {library_version}; model.bin sha256:{model_hash}"
    run_id = f"local-whisper-gha-{github_run_id}-{github_run_attempt}-{video_sha256[:16]}"
    return _result("local faster-whisper", "OpenAI Whisper", model_repo, version,
                   run_id, video_sha256, duration, segments, joined, raw_response, utc_now())


def save_asr_results(episode_dir: Path, gemini: dict, whisper: dict) -> list[dict[str, str]]:
    require(gemini["provider"] != whisper["provider"] and
            gemini["model_family"] != whisper["model_family"] and
            gemini["run_id"] != whisper["run_id"],
            "ASR must use two distinct providers, model families, and runs")
    references: list[dict[str, str]] = []
    for filename, result in (("agent-asr-gemini.json", gemini), ("agent-asr-whisper.json", whisper)):
        path = episode_dir / filename
        write_json_new(path, result)
        references.append({"file": filename, "sha256": digest_file(path)})
    return references
