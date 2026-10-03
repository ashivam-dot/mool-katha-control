"""Control-owned Gemini TTS exchange and exact-byte studio handoff.

The exchange is issued before production by a separate, pinned control job.
Its private response and audio are signed with a key unavailable to the
producer. QA verifies that signature, the frozen script, the exact narration
WAV used by the studio, and its signal in the final mixed video audio.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
import os
import re
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .candidate import Candidate, _git, _git_blob, _yaml_object
from .common import (EPISODE, QA_REPOSITORY, QaHold, digest_bytes, digest_file,
                     expect_sha, json_object, path_under, require, timestamp,
                     utc_now, write_bytes_new, write_json_new)


EXCHANGE_KIND = "control_gemini_tts_exchange_v1"
SIGNATURE_KIND = "control_gemini_tts_exchange_signature_v1"
PROOF_KIND = "control_gemini_voice_v1"
CONTEXT = b"mool-katha-control-gemini-tts-v1\0"
TERMS_URL = "https://ai.google.dev/gemini-api/terms"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_DIRECTION = (
    "Read the transcript below aloud as a warm, natural storyteller. Speak only the transcript.")
REQUEST_FILE = "request.json"
RESPONSE_FILE = "response.json"
PROVIDER_AUDIO_FILE = "provider-audio.wav"
NARRATION_FILE = "narration.wav"
EXCHANGE_FILE = "exchange.json"
SIGNATURE_FILE = "signature.json"
QA_BUNDLE_DIR = "agent-control-voice"
FILES = (REQUEST_FILE, RESPONSE_FILE, PROVIDER_AUDIO_FILE, NARRATION_FILE,
         EXCHANGE_FILE, SIGNATURE_FILE)
MAX_RESPONSE_BYTES = 24 * 1024 * 1024
MAX_AUDIO_BYTES = 16 * 1024 * 1024
MAX_RECORD_BYTES = 128 * 1024
SAMPLE_RATE = 24_000
_OVERRIDE = re.compile(r"\[([^\]]+)\]\(/[^)]*/\)")
_MODEL = re.compile(r"gemini-3\.8-[A-Za-z0-9._-]+-tts\Z")
_VOICE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,63}\Z")
_KEY_ID = re.compile(r"[A-Za-z0-9._-]{4,64}\Z")


@dataclass(frozen=True)
class VoiceExchangePolicy:
    """Owner-reviewed pins; never load these from the candidate or producer."""

    workflow_ref: str
    workflow_sha: str
    keys: Mapping[str, bytes]

    def __post_init__(self) -> None:
        require(isinstance(self.workflow_ref, str) and
            re.fullmatch(re.escape(QA_REPOSITORY) +
                         r"/\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml@refs/tags/voice-v1",
                         self.workflow_ref) is not None and
            isinstance(self.workflow_sha, str) and
            re.fullmatch(r"[0-9a-f]{40}", self.workflow_sha) is not None,
            "control voice workflow pin is invalid")
        require(isinstance(self.keys, Mapping) and bool(self.keys) and
                all(isinstance(key_id, str) and
                _KEY_ID.fullmatch(key_id) is not None and
                isinstance(public, bytes) and len(public) == 32
                for key_id, public in self.keys.items()),
            "control voice public keys are invalid")
        object.__setattr__(self, "keys", MappingProxyType(dict(self.keys)))

    def digest(self) -> str:
        """Bind the exact reviewed workflow and public keys into release policy."""
        pins = {"workflow_ref": self.workflow_ref, "workflow_sha": self.workflow_sha,
                "keys": {key_id: base64.b64encode(public).decode("ascii")
                         for key_id, public in self.keys.items()}}
        return digest_bytes(b"mool-katha-control-voice-policy-v1\0" + _canonical(pins))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _run_context(policy: VoiceExchangePolicy, env: Mapping[str, str]) -> dict[str, Any]:
    require(env.get("GITHUB_REPOSITORY") == QA_REPOSITORY and
            env.get("GITHUB_WORKFLOW_REF") == policy.workflow_ref and
            env.get("GITHUB_WORKFLOW_SHA") == policy.workflow_sha,
            "control voice exchange is outside the pinned owner workflow")
    try:
        run_id = int(env.get("GITHUB_RUN_ID", ""))
        attempt = int(env.get("GITHUB_RUN_ATTEMPT", ""))
    except (TypeError, ValueError) as exc:
        raise QaHold("control voice run identity is invalid") from exc
    require(run_id > 0 and attempt > 0, "control voice run identity is invalid")
    return {"system": "github_actions", "repository": QA_REPOSITORY,
            "workflow_ref": policy.workflow_ref, "workflow_sha": policy.workflow_sha,
            "run_id": run_id, "run_attempt": attempt}


def _voice_request(spec: dict[str, Any], script: dict[str, Any],
                   episode_id: str) -> tuple[dict[str, Any], str, str, int]:
    require(spec.get("id") == episode_id and isinstance(spec.get("beats"), list) and
            isinstance(script.get("beats"), list) and
            bool(spec["beats"]) and len(spec["beats"]) == len(script["beats"]),
            "control voice needs one frozen script and matching spec")
    for beat, written in zip(spec["beats"], script["beats"]):
        require(isinstance(beat, dict) and isinstance(written, dict) and
                isinstance(beat.get("text"), str) and beat["text"].strip() and
                beat["text"] == written.get("text"),
                "control voice script differs from frozen spec")
    voice = spec.get("voice")
    require(isinstance(voice, dict) and voice.get("engine") == "gemini" and
            isinstance(voice.get("model"), str) and
            _MODEL.fullmatch(voice["model"]) is not None and
            isinstance(voice.get("voice"), str) and
            _VOICE.fullmatch(voice["voice"]) is not None and
            voice.get("lang_code") == "h" and
            type(voice.get("speed", 1.0)) in (int, float) and
            voice.get("speed", 1.0) == 1.0 and
            type(spec.get("narration_tempo", 1.0)) in (int, float) and
            spec.get("narration_tempo", 1.0) == 1.0,
            "control voice requires a pinned Gemini 3.8 Hindi voice without later tempo changes")
    direction = voice.get("direction") or DEFAULT_DIRECTION
    lead = voice.get("lead_in", 0.08)
    require(isinstance(direction, str) and 10 <= len(direction) <= 4_000 and
            isinstance(lead, (int, float)) and not isinstance(lead, bool) and
            math.isfinite(lead) and 0 <= lead <= 1,
            "control voice direction or lead-in is malformed")
    transcript = "\n".join(_OVERRIDE.sub(r"\1", beat["text"])
                           for beat in spec["beats"])
    require(0 < len(transcript.encode("utf-8")) <= 100_000,
            "control voice transcript is empty or oversized")
    body = {"model": voice["model"],
            "input": [{"type": "user_input", "content": [{"type": "text", "text": transcript,
                       "annotations": [{"type": "speech_metadata", "style": direction}]}]}],
            "response_format": {"type": "audio", "mime_type": "audio/wav"},
            "generation_config": {"speech_config": [{"voice": voice["voice"]}]},
            "store": False}
    return body, transcript, direction, int(lead * SAMPLE_RATE)


def _frozen_voice(source_repo: Path, source_commit: str,
                  episode_id: str) -> tuple[dict, dict, bytes, bytes]:
    require(isinstance(episode_id, str) and EPISODE.fullmatch(episode_id) is not None and
            isinstance(source_commit, str) and
            re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None,
            "control voice needs an exact episode and source commit")
    require(source_repo.is_dir() and (source_repo / ".git").exists() and
            _git(source_repo, "rev-parse", "HEAD").decode("ascii").strip() == source_commit,
            "control voice source checkout differs from the frozen commit")
    base = f"content/episodes/{episode_id}/"
    spec_raw = _git_blob(source_repo, source_commit, base + "short.yaml")
    script_raw = _git_blob(source_repo, source_commit, base + "script.json")
    assert spec_raw is not None and script_raw is not None
    return (_yaml_object(spec_raw), json_object(script_raw, "control voice script"),
            spec_raw, script_raw)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _post_provider(request_bytes: bytes, key: str) -> bytes:
    """The only paid call in this module; tests replace this function entirely."""
    require(isinstance(key, str) and bool(key.strip()),
            "control voice Gemini credential is unavailable")
    request = urllib.request.Request(
        ENDPOINT, data=request_bytes,
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "x-goog-api-key": key}, method="POST")
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=300) as response:
            require(response.status == 200 and response.geturl() == ENDPOINT,
                    "control voice Gemini response is not from the requested endpoint")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise QaHold("control voice Gemini request failed") from exc
    require(0 < len(raw) <= MAX_RESPONSE_BYTES,
            "control voice Gemini response is empty or oversized")
    return raw


def _provider_audio(raw_response: bytes, requested_model: str) -> tuple[bytes, dict]:
    provider = json_object(raw_response, "control voice provider response")
    require(provider.get("status") == "completed" and
            provider.get("model") == requested_model and
            isinstance(provider.get("usage"), dict) and bool(provider["usage"]),
            "control voice provider response is incomplete or served another model")
    response_id = provider.get("id")
    require(response_id is None or isinstance(response_id, str) and len(response_id) >= 4,
            "control voice provider response ID is malformed")
    steps = provider.get("steps")
    require(isinstance(steps, list) and bool(steps),
            "control voice provider response has no output steps")
    audio_parts: list[dict] = []
    for step in steps:
        require(isinstance(step, dict), "control voice provider step is malformed")
        if step.get("type") != "model_output":
            continue
        content = step.get("content")
        require(isinstance(content, list), "control voice provider output is malformed")
        for part in content:
            require(isinstance(part, dict), "control voice provider output part is malformed")
            if part.get("type") == "audio":
                audio_parts.append(part)
    require(len(audio_parts) == 1 and audio_parts[0].get("mime_type") == "audio/wav" and
            isinstance(audio_parts[0].get("data"), str),
            "control voice needs one complete provider WAV block")
    try:
        audio = base64.b64decode(audio_parts[0]["data"], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise QaHold("control voice provider audio is invalid base64") from exc
    require(0 < len(audio) <= MAX_AUDIO_BYTES,
            "control voice provider audio is empty or oversized")
    metadata = {"served_model": provider["model"], "response_id": response_id,
                "usage": provider["usage"], "status": provider["status"]}
    return audio, metadata


def _derive_narration(provider_audio: bytes, lead_in_frames: int) -> tuple[bytes, dict[str, int]]:
    """Trim only measured outer silence and prepend the frozen cover-frame lead-in."""
    try:
        import numpy as np
    except ImportError as exc:
        raise QaHold("control voice WAV derivation needs locked NumPy") from exc
    require(np.__version__ == "2.5.3" and isinstance(lead_in_frames, int) and
            0 <= lead_in_frames <= SAMPLE_RATE,
            "control voice WAV derivation uses unreviewed parameters")
    try:
        with wave.open(io.BytesIO(provider_audio), "rb") as stream:
            require(stream.getnchannels() == 1 and stream.getsampwidth() == 2 and
                    stream.getframerate() == SAMPLE_RATE and stream.getcomptype() == "NONE",
                    "control voice provider WAV must be mono 24 kHz PCM16")
            frames = stream.getnframes()
            require(5 * SAMPLE_RATE <= frames <= 120 * SAMPLE_RATE,
                    "control voice provider WAV duration is outside the reviewed bound")
            pcm = stream.readframes(frames)
    except (OSError, EOFError, wave.Error) as exc:
        raise QaHold("control voice provider WAV cannot be decoded") from exc
    require(len(pcm) == 2 * frames, "control voice provider WAV is truncated")
    values = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    peak = float(np.max(np.abs(values)))
    require(peak >= 100, "control voice provider WAV has no measurable speech")
    loud = np.flatnonzero(np.abs(values) > 0.01 * peak)
    require(bool(loud.size), "control voice provider WAV has no speech bounds")
    keep = int(0.05 * SAMPLE_RATE)
    start = max(int(loud[0]) - keep, 0)
    end = min(int(loud[-1]) + keep + 1, frames)
    require(end - start >= 4 * SAMPLE_RATE,
            "control voice trimmed narration is too short")
    output_pcm = b"\0\0" * lead_in_frames + pcm[2 * start:2 * end]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(output_pcm)
    return buffer.getvalue(), {"provider_frames": frames, "trim_start_frame": start,
                               "trim_end_frame": end, "lead_in_frames": lead_in_frames,
                               "narration_frames": lead_in_frames + end - start}


def issue_voice_exchange(source_repo: Path, source_commit: str, episode_id: str,
                         output_dir: Path, *, policy: VoiceExchangePolicy,
                         key_id: str, private_key_seed: bytes, api_key: str,
                         env: Mapping[str, str] | None = None) -> Path:
    """Issue one signed private handoff from a frozen script in the control job."""
    require(isinstance(policy, VoiceExchangePolicy),
            "control voice needs a reviewed owner policy")
    run = _run_context(policy, os.environ if env is None else env)
    require(isinstance(key_id, str) and key_id in policy.keys and
            isinstance(private_key_seed, bytes) and
            len(private_key_seed) == 32, "control voice signing key is unavailable")
    private_key = Ed25519PrivateKey.from_private_bytes(private_key_seed)
    actual_public = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    require(actual_public == policy.keys[key_id],
            "control voice private key differs from reviewed policy")
    spec, script, spec_raw, script_raw = _frozen_voice(source_repo, source_commit, episode_id)
    body, transcript, direction, lead_frames = _voice_request(spec, script, episode_id)
    request_bytes = _canonical(body)
    require(not output_dir.exists(), "control voice exchange output directory must be new")
    output_dir.mkdir(parents=True, mode=0o700)
    write_bytes_new(output_dir / REQUEST_FILE, request_bytes)
    requested_at = utc_now()
    response_bytes = _post_provider(request_bytes, api_key)
    write_bytes_new(output_dir / RESPONSE_FILE, response_bytes)
    provider_audio, metadata = _provider_audio(response_bytes, body["model"])
    narration, geometry = _derive_narration(provider_audio, lead_frames)
    write_bytes_new(output_dir / PROVIDER_AUDIO_FILE, provider_audio)
    write_bytes_new(output_dir / NARRATION_FILE, narration)
    record = {"kind": EXCHANGE_KIND, "episode_id": episode_id,
              "source_commit": source_commit,
              "spec_sha256": digest_bytes(spec_raw), "script_sha256": digest_bytes(script_raw),
              "transcript_sha256": digest_bytes(transcript.encode("utf-8")),
              "direction_sha256": digest_bytes(direction.encode("utf-8")),
              "model": body["model"], "voice": spec["voice"]["voice"],
              "endpoint": ENDPOINT, "terms_url": TERMS_URL,
              "request_sha256": digest_bytes(request_bytes),
              "response_sha256": digest_bytes(response_bytes),
              "provider_audio_sha256": digest_bytes(provider_audio),
              "narration_sha256": digest_bytes(narration),
              "response_metadata": metadata, "geometry": geometry,
              "requested_at": requested_at, "completed_at": utc_now(), "run": run}
    write_json_new(output_dir / EXCHANGE_FILE, record)
    raw_record = (output_dir / EXCHANGE_FILE).read_bytes()
    signature = private_key.sign(CONTEXT + raw_record)
    write_json_new(output_dir / SIGNATURE_FILE,
                   {"kind": SIGNATURE_KIND, "algorithm": "Ed25519",
                    "key_id": key_id, "exchange_sha256": digest_bytes(raw_record),
                    "signature": base64.b64encode(signature).decode("ascii")})
    verify_voice_exchange(output_dir, policy)
    return output_dir / EXCHANGE_FILE


def _read_file(root: Path, name: str, limit: int) -> bytes:
    path = path_under(root, name, f"control voice {name}")
    require(path.is_file() and not path.is_symlink() and
            0 < path.stat().st_size <= limit,
            f"control voice {name} is missing, unsafe, or oversized")
    raw = path.read_bytes()
    require(len(raw) == path.stat().st_size,
            f"control voice {name} changed during read")
    return raw


def verify_voice_exchange(bundle_dir: Path, policy: VoiceExchangePolicy) -> dict[str, Any]:
    """Replay signed raw request, complete response, audio, and WAV derivation."""
    require(isinstance(policy, VoiceExchangePolicy),
            "control voice needs a reviewed owner policy")
    require(bundle_dir.is_dir() and not bundle_dir.is_symlink(),
            "control voice bundle directory is missing or unsafe")
    raw_record = _read_file(bundle_dir, EXCHANGE_FILE, MAX_RECORD_BYTES)
    raw_signature = _read_file(bundle_dir, SIGNATURE_FILE, 16_384)
    record = json_object(raw_record, "control voice exchange")
    envelope = json_object(raw_signature, "control voice signature")
    require(set(envelope) == {"kind", "algorithm", "key_id", "exchange_sha256", "signature"} and
            envelope["kind"] == SIGNATURE_KIND and envelope["algorithm"] == "Ed25519" and
            isinstance(envelope["key_id"], str) and
            envelope["key_id"] in policy.keys and
            isinstance(envelope["signature"], str) and
            envelope["exchange_sha256"] == digest_bytes(raw_record),
            "control voice signature envelope is invalid")
    try:
        signature = base64.b64decode(envelope["signature"], validate=True)
        require(len(signature) == 64 and
                base64.b64encode(signature).decode("ascii") == envelope["signature"],
                "control voice signature encoding is invalid")
        Ed25519PublicKey.from_public_bytes(policy.keys[envelope["key_id"]]).verify(
            signature, CONTEXT + raw_record)
    except (TypeError, ValueError, binascii.Error, InvalidSignature) as exc:
        raise QaHold("control voice exchange signature is invalid") from exc
    require(set(record) == {"kind", "episode_id", "source_commit", "spec_sha256",
                            "script_sha256", "transcript_sha256", "direction_sha256",
                            "model", "voice", "endpoint", "terms_url", "request_sha256",
                            "response_sha256", "provider_audio_sha256", "narration_sha256",
                            "response_metadata", "geometry", "requested_at", "completed_at", "run"} and
            record["kind"] == EXCHANGE_KIND and record["endpoint"] == ENDPOINT and
            record["terms_url"] == TERMS_URL and
            isinstance(record["episode_id"], str) and
            EPISODE.fullmatch(record["episode_id"]) is not None and
            isinstance(record["source_commit"], str) and
            re.fullmatch(r"[0-9a-f]{40}", record["source_commit"]) is not None and
            isinstance(record["model"], str) and _MODEL.fullmatch(record["model"]) is not None and
            isinstance(record["voice"], str) and _VOICE.fullmatch(record["voice"]) is not None,
            "control voice exchange record is malformed")
    for name in ("spec_sha256", "script_sha256", "transcript_sha256",
                 "direction_sha256", "request_sha256", "response_sha256",
                 "provider_audio_sha256", "narration_sha256"):
        expect_sha(record[name], f"control voice {name}")
    timestamp(record["requested_at"], "control voice request time")
    timestamp(record["completed_at"], "control voice completion time")
    run = record["run"]
    require(isinstance(run, dict) and set(run) ==
            {"system", "repository", "workflow_ref", "workflow_sha", "run_id", "run_attempt"} and
            run["system"] == "github_actions" and run["repository"] == QA_REPOSITORY and
            run["workflow_ref"] == policy.workflow_ref and
            run["workflow_sha"] == policy.workflow_sha and
            all(type(run[key]) is int and run[key] > 0 for key in ("run_id", "run_attempt")),
            "control voice exchange is not from the pinned owner run")
    request_raw = _read_file(bundle_dir, REQUEST_FILE, MAX_RECORD_BYTES)
    response_raw = _read_file(bundle_dir, RESPONSE_FILE, MAX_RESPONSE_BYTES)
    provider_audio = _read_file(bundle_dir, PROVIDER_AUDIO_FILE, MAX_AUDIO_BYTES)
    narration = _read_file(bundle_dir, NARRATION_FILE, MAX_AUDIO_BYTES)
    require(digest_bytes(request_raw) == record["request_sha256"] and
            digest_bytes(response_raw) == record["response_sha256"] and
            digest_bytes(provider_audio) == record["provider_audio_sha256"] and
            digest_bytes(narration) == record["narration_sha256"],
            "control voice exchange bytes differ from signed hashes")
    request = json_object(request_raw, "control voice request")
    try:
        transcript = request["input"][0]["content"][0]["text"]
        direction = request["input"][0]["content"][0]["annotations"][0]["style"]
        voice = request["generation_config"]["speech_config"][0]["voice"]
    except (KeyError, IndexError, TypeError) as exc:
        raise QaHold("control voice request body is malformed") from exc
    require(isinstance(transcript, str) and isinstance(direction, str) and
            isinstance(voice, str) and record["voice"] == voice and
            record["transcript_sha256"] == digest_bytes(transcript.encode("utf-8")) and
            record["direction_sha256"] == digest_bytes(direction.encode("utf-8")) and
            request == {"model": record["model"],
                        "input": [{"type": "user_input", "content": [{"type": "text", "text": transcript,
                                   "annotations": [{"type": "speech_metadata", "style": direction}]}]}],
                        "response_format": {"type": "audio", "mime_type": "audio/wav"},
                        "generation_config": {"speech_config": [{"voice": voice}]},
                        "store": False} and request_raw == _canonical(request),
            "control voice request differs from the signed exact request")
    extracted_audio, metadata = _provider_audio(response_raw, record["model"])
    require(extracted_audio == provider_audio and record["response_metadata"] == metadata,
            "control voice saved audio or provider metadata differs from complete response")
    geometry = record["geometry"]
    require(isinstance(geometry, dict) and type(geometry.get("lead_in_frames")) is int,
            "control voice WAV geometry is malformed")
    derived, expected_geometry = _derive_narration(provider_audio, geometry["lead_in_frames"])
    require(geometry == expected_geometry and narration == derived,
            "control voice narration WAV differs from returned provider audio")
    return record


def copy_voice_exchange(source_dir: Path, episode_dir: Path,
                        policy: VoiceExchangePolicy) -> Path:
    """Copy a verified control artifact into QA evidence with exclusive writes."""
    verify_voice_exchange(source_dir, policy)
    target = episode_dir / QA_BUNDLE_DIR
    require(not target.exists(), "control voice QA evidence already exists")
    target.mkdir(parents=True, mode=0o700)
    for name in FILES:
        limit = MAX_RESPONSE_BYTES if name == RESPONSE_FILE else (
            MAX_AUDIO_BYTES if name in (PROVIDER_AUDIO_FILE, NARRATION_FILE) else MAX_RECORD_BYTES)
        write_bytes_new(target / name, _read_file(source_dir, name, limit))
    verify_voice_exchange(target, policy)
    return target


def _read_pcm16(path: Path, rate: int, label: str) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise QaHold("control voice audio comparison needs locked NumPy") from exc
    require(np.__version__ == "2.5.3", "control voice audio comparison needs reviewed NumPy")
    try:
        with wave.open(str(path), "rb") as stream:
            require(stream.getnchannels() == 1 and stream.getsampwidth() == 2 and
                    stream.getframerate() == rate and stream.getcomptype() == "NONE",
                    f"{label} must be mono PCM16 at {rate} Hz")
            frames = stream.getnframes()
            require(0 < frames <= 120 * rate, f"{label} duration is outside the reviewed bound")
            raw = stream.readframes(frames)
    except (OSError, EOFError, wave.Error) as exc:
        raise QaHold(f"{label} WAV cannot be decoded") from exc
    require(len(raw) == 2 * frames, f"{label} WAV is truncated")
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def compare_final_audio(narration_path: Path, final_audio_path: Path) -> dict[str, Any]:
    """Find the exact narration waveform inside the complete mixed final track."""
    import numpy as np

    source = _read_pcm16(narration_path, SAMPLE_RATE, "control narration")
    final = _read_pcm16(final_audio_path, 16_000, "complete final video audio")
    n = int(round(len(source) * 16_000 / SAMPLE_RATE))
    voice = np.interp(np.arange(n) * SAMPLE_RATE / 16_000,
                      np.arange(len(source)), source)
    require(len(final) >= len(voice) - 16_000 and
            float(np.sqrt(np.mean(voice * voice))) >= 0.01,
            "control narration or final audio duration/level is unsupported")
    voice = voice - voice.mean()
    final = final - final.mean()
    search = 8_000  # AAC encoder delay and the first render sample stay within half a second.
    padded = np.pad(final, (search, search + max(0, len(voice) - len(final))))
    size = 1 << (len(padded) + len(voice) - 2).bit_length()
    correlation = np.fft.irfft(np.fft.rfft(padded, size) *
                               np.conj(np.fft.rfft(voice, size)), size)
    choices = correlation[:2 * search + 1]
    start = int(np.argmax(choices))
    aligned = padded[start:start + len(voice)]
    global_score = float(np.dot(voice, aligned) /
                         (np.linalg.norm(voice) * np.linalg.norm(aligned) + 1e-12))
    window = 4 * 16_000
    scores: list[float] = []
    for offset in range(0, len(voice) - window + 1, window):
        a, b = voice[offset:offset + window], aligned[offset:offset + window]
        if float(np.sqrt(np.mean(a * a))) < 0.01:
            continue
        scores.append(float(np.dot(a, b) /
                            (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    require(len(scores) >= 3 and global_score >= 0.45 and
            float(np.median(scores)) >= 0.50 and
            sum(score >= 0.35 for score in scores) >= math.ceil(0.7 * len(scores)),
            "complete final video audio does not contain the exact control narration")
    return {"method": "waveform_correlation_v1", "offset_samples_16k": start - search,
            "global_correlation": round(global_score, 3),
            "median_window_correlation": round(float(np.median(scores)), 3),
            "checked_windows": len(scores),
            "strong_windows": sum(score >= 0.35 for score in scores)}


def verify_candidate_voice(candidate: Candidate, asset: dict, bundle_dir: Path,
                           final_audio_path: Path, policy: VoiceExchangePolicy) -> dict[str, Any]:
    """Recheck the handoff and the final mixed audio; final-video ASR remains mandatory."""
    candidate.recheck()
    record = verify_voice_exchange(bundle_dir, policy)
    exchange_sha = digest_file(bundle_dir / EXCHANGE_FILE)
    signature_sha = digest_file(bundle_dir / SIGNATURE_FILE)
    expected_link = {"schema": EXCHANGE_KIND, "exchange_sha256": exchange_sha,
                     "narration_sha256": record["narration_sha256"],
                     "narration_asset_id": asset["id"]}
    require(asset.get("role") == "voice" and
            record["episode_id"] == candidate.episode_id and
            candidate.manifest.get("narration_asset_id") == asset["id"] and
            candidate.manifest.get("control_voice_exchange") == expected_link and
            "voice_take" not in candidate.manifest and
            "narration_transform" not in candidate.manifest and
            asset.get("file") == f"content/episodes/{candidate.episode_id}/work/narration.wav" and
            asset.get("origin") == f"control:gemini:{exchange_sha}" and
            asset.get("rights_url") == TERMS_URL and
            asset.get("license") == "Gemini API Additional Terms" and
            asset["sha256"] == record["narration_sha256"] and
            digest_file(candidate.asset_paths[asset["id"]]) == record["narration_sha256"],
            f"asset {asset['id']}: final narration does not use the signed control WAV")
    base = f"content/episodes/{candidate.episode_id}/"
    require(candidate.committed_blob(base + "short.yaml", record["source_commit"]) ==
            (candidate.episode_dir / "short.yaml").read_bytes() and
            candidate.committed_blob(base + "script.json", record["source_commit"]) ==
            (candidate.episode_dir / "script.json").read_bytes() and
            candidate.hashes["spec"] == record["spec_sha256"] and
            candidate.hashes["script"] == record["script_sha256"],
            f"asset {asset['id']}: signed TTS request differs from frozen source")
    body, transcript, direction, lead_frames = _voice_request(candidate.spec, candidate.script,
                                                               candidate.episode_id)
    require(_read_file(bundle_dir, REQUEST_FILE, MAX_RECORD_BYTES) == _canonical(body) and
            record["transcript_sha256"] == digest_bytes(transcript.encode("utf-8")) and
            record["direction_sha256"] == digest_bytes(direction.encode("utf-8")) and
            record["geometry"]["lead_in_frames"] == lead_frames,
            f"asset {asset['id']}: signed TTS request differs from frozen voice and beats")
    metrics = compare_final_audio(candidate.asset_paths[asset["id"]], final_audio_path)
    return {"kind": PROOF_KIND, "exchange_file": f"{QA_BUNDLE_DIR}/{EXCHANGE_FILE}",
            "exchange_sha256": exchange_sha,
            "signature_file": f"{QA_BUNDLE_DIR}/{SIGNATURE_FILE}",
            "signature_sha256": signature_sha,
            "provider_response_sha256": record["response_sha256"],
            "provider_audio_sha256": record["provider_audio_sha256"],
            "narration_sha256": record["narration_sha256"],
            "input_video_sha256": candidate.hashes["video"],
            "final_audio_sha256": digest_file(final_audio_path),
            "audio_match": metrics}
