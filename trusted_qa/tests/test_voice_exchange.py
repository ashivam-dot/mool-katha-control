"""Offline tests for the signed control voice exchange and studio handoff."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trusted_qa import voice_exchange as voice
from trusted_qa.candidate import Candidate
from trusted_qa.common import QA_REPOSITORY, QaHold, digest_bytes, digest_file


EPISODE = "ep004"
MODEL = "gemini-3.8-flash-tts"
KEY_ID = "test-key-v1"
SEED = bytes(range(32))
WORKFLOW_REF = QA_REPOSITORY + "/.github/workflows/control-voice.yml@refs/tags/voice-v1"
WORKFLOW_SHA = "a" * 40
ENV = {"GITHUB_REPOSITORY": QA_REPOSITORY,
       "GITHUB_WORKFLOW_REF": WORKFLOW_REF,
       "GITHUB_WORKFLOW_SHA": WORKFLOW_SHA,
       "GITHUB_RUN_ID": "12345", "GITHUB_RUN_ATTEMPT": "2"}


def _json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _write(root: Path, name: str, raw: bytes) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def _source(root: Path) -> tuple[Path, str, dict, dict, bytes, bytes]:
    """Make a real Git commit so production code reads immutable source blobs."""
    repo = root / "source"
    repo.mkdir()
    beats = ["[राम](/say-ram/) कथा सुनाते हैं।", "सीता ध्यान से सुनती हैं।"]
    spec = {"id": EPISODE, "beats": [{"text": text} for text in beats],
            "voice": {"engine": "gemini", "model": MODEL, "voice": "Kore",
                      "lang_code": "h", "speed": 1.0, "lead_in": 0.08,
                      "direction": "Read each Hindi beat with a warm, natural cadence."},
            "narration_tempo": 1.0}
    script = {"beats": [{"text": text} for text in beats]}
    spec_raw, script_raw = _json(spec), _json(script)
    base = f"content/episodes/{EPISODE}"
    _write(repo, f"{base}/short.yaml", spec_raw)  # JSON is valid YAML.
    _write(repo, f"{base}/script.json", script_raw)
    for args in (("init", "-q"), ("config", "user.name", "QA Fixture"),
                 ("config", "user.email", "fixture@example.test"),
                 ("add", "."), ("commit", "-qm", "frozen voice source")):
        subprocess.run(["git", "-C", str(repo), *args], check=True,
                       capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"]).decode().strip()
    return repo, commit, spec, script, spec_raw, script_raw


def _policy(*, workflow_ref: str = WORKFLOW_REF,
            workflow_sha: str = WORKFLOW_SHA) -> voice.VoiceExchangePolicy:
    public = Ed25519PrivateKey.from_private_bytes(SEED).public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return voice.VoiceExchangePolicy(workflow_ref, workflow_sha, {KEY_ID: public})


def _wav(*, rate: int = voice.SAMPLE_RATE) -> bytes:
    """A deterministic, nonperiodic 13-second PCM16 signal with outer silence."""
    rng = np.random.default_rng(1729)
    samples = np.clip(rng.normal(0, 3_500, 13 * rate), -14_000, 14_000).astype("<i2")
    silence = np.zeros(rate // 4, dtype="<i2")
    pcm = np.concatenate((silence, samples, silence))
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm.tobytes())
    return output.getvalue()


def _response(provider_wav: bytes) -> bytes:
    return _json({"id": "synthetic-response-1234", "status": "completed",
                  "model": MODEL, "usage": {"input_tokens": 42},
                  "steps": [{"type": "model_output", "content": [
                      {"type": "audio", "mime_type": "audio/wav",
                       "data": base64.b64encode(provider_wav).decode("ascii")}]}]})


def _final_audio(path: Path, narration: bytes, *, related: bool) -> Path:
    with wave.open(io.BytesIO(narration), "rb") as stream:
        source = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(
            np.float64) / 32768.0
    count = int(round(len(source) * 16_000 / voice.SAMPLE_RATE))
    resampled = np.interp(np.arange(count) * voice.SAMPLE_RATE / 16_000,
                          np.arange(len(source)), source)
    if related:
        mixed = np.pad(0.8 * resampled, (3_200, 4_800))
        mixed += 0.008 * np.sin(2 * np.pi * 440 * np.arange(len(mixed)) / 16_000)
    else:
        mixed = np.random.default_rng(31415).normal(0, 0.12, count + 8_000)
    pcm = (np.clip(mixed, -0.99, 0.99) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(pcm.tobytes())
    return path


def _candidate(root: Path, source: tuple[Path, str, dict, dict, bytes, bytes],
               bundle_dir: Path) -> tuple[Candidate, dict]:
    repo, commit, spec, script, spec_raw, script_raw = source
    snapshot = root / "snapshot"
    base = f"content/episodes/{EPISODE}"
    spec_path = _write(snapshot, f"{base}/short.yaml", spec_raw)
    script_path = _write(snapshot, f"{base}/script.json", script_raw)
    narration_raw = (bundle_dir / voice.NARRATION_FILE).read_bytes()
    narration_name = f"{base}/work/narration.wav"
    narration_path = _write(snapshot, narration_name, narration_raw)
    exchange_sha = digest_file(bundle_dir / voice.EXCHANGE_FILE)
    asset = {"id": "voice:one", "role": "voice", "file": narration_name,
             "sha256": digest_bytes(narration_raw),
             "origin": f"control:gemini:{exchange_sha}",
             "rights_url": voice.TERMS_URL, "license": "Gemini API Additional Terms"}
    manifest = {"narration_asset_id": asset["id"],
                "control_voice_exchange": {
                    "schema": voice.EXCHANGE_KIND, "exchange_sha256": exchange_sha,
                    "narration_sha256": asset["sha256"], "narration_asset_id": asset["id"]}}
    manifest_path = _write(snapshot, f"{base}/work/manifest.json", _json(manifest))
    evidence_path = _write(snapshot, f"{base}/evidence.json", b"{}\n")
    qc_path = _write(snapshot, f"{base}/qc.json", b"{}\n")
    pending_path = _write(snapshot, f"{base}/evidence_pending.json", b"{}\n")
    video_path = _write(snapshot, f"{base}/{EPISODE}.mp4", b"synthetic final video")
    hashes = {"spec": digest_file(spec_path), "script": digest_file(script_path),
              "manifest": digest_file(manifest_path), "evidence": digest_file(evidence_path),
              "qc": digest_file(qc_path), "pending": digest_file(pending_path),
              "video": digest_file(video_path)}
    candidate = Candidate(root=snapshot, source_repo=repo, episode_id=EPISODE,
                          source_commit=commit, qa_agent_id="fixture-qa",
                          producer_agent_id="fixture-producer", script=script, spec=spec,
                          ledger={}, manifest=manifest, qc={}, pending={}, hashes=hashes,
                          assets={asset["id"]: asset},
                          asset_paths={asset["id"]: narration_path},
                          internal_references={}, archive_members={})
    return candidate, asset


class VoiceExchangeOfflineTests(unittest.TestCase):
    def _issue(self, root: Path, *, dirty_checkout: bool = False):
        source = _source(root)
        repo, commit, _, _, spec_raw, _ = source
        if dirty_checkout:
            _write(repo, f"content/episodes/{EPISODE}/short.yaml", b"id: ep999\n")
        provider_wav = _wav()
        response = _response(provider_wav)
        bundle = root / "issued"
        policy = _policy()
        with patch.object(voice, "_post_provider", return_value=response) as post:
            voice.issue_voice_exchange(repo, commit, EPISODE, bundle, policy=policy,
                                       key_id=KEY_ID, private_key_seed=SEED,
                                       api_key="test-only-api-key", env=ENV)
        post.assert_called_once_with((bundle / voice.REQUEST_FILE).read_bytes(),
                                     "test-only-api-key")
        record = voice.verify_voice_exchange(bundle, policy)
        self.assertEqual(record["spec_sha256"], digest_bytes(spec_raw))
        return source, bundle, policy, provider_wav, record

    def test_issues_from_frozen_git_bytes_and_replays_exact_pcm_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, bundle, policy, provider_wav, record = self._issue(
                root, dirty_checkout=True)
            request = json.loads((bundle / voice.REQUEST_FILE).read_bytes())
            self.assertEqual(request["input"][0]["content"][0]["text"],
                             "राम कथा सुनाते हैं।\nसीता ध्यान से सुनती हैं।")
            self.assertFalse(request["store"])
            self.assertEqual((bundle / voice.PROVIDER_AUDIO_FILE).read_bytes(), provider_wav)
            self.assertEqual(record["run"]["workflow_sha"], WORKFLOW_SHA)
            self.assertEqual(record["geometry"]["lead_in_frames"], 1_920)
            with wave.open(str(bundle / voice.NARRATION_FILE), "rb") as stream:
                self.assertEqual((stream.getnchannels(), stream.getsampwidth(),
                                  stream.getframerate()), (1, 2, 24_000))
                self.assertEqual(stream.getnframes(), record["geometry"]["narration_frames"])
            copied = voice.copy_voice_exchange(bundle, root / "qa-episode", policy)
            self.assertEqual(voice.verify_voice_exchange(copied, policy), record)
            self.assertEqual(source[1], record["source_commit"])

    def test_rejects_wrong_workflow_context_and_reviewed_policy_pin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, commit, *_ = _source(root)
            policy = _policy()
            for index, (name, value) in enumerate((("GITHUB_REPOSITORY", "other/repo"),
                                                    ("GITHUB_WORKFLOW_REF", "wrong/ref"),
                                                    ("GITHUB_WORKFLOW_SHA", "b" * 40))):
                with self.subTest(name=name):
                    bad_env = dict(ENV, **{name: value})
                    output = root / f"rejected-{index}"
                    with patch.object(voice, "_post_provider",
                                      side_effect=AssertionError("provider must not be called")) as post:
                        with self.assertRaisesRegex(QaHold, "outside the pinned owner workflow"):
                            voice.issue_voice_exchange(repo, commit, EPISODE, output,
                                                       policy=policy, key_id=KEY_ID,
                                                       private_key_seed=SEED,
                                                       api_key="test-only-api-key", env=bad_env)
                    post.assert_not_called()
                    self.assertFalse(output.exists())

            response = _response(_wav())
            bundle = root / "issued"
            with patch.object(voice, "_post_provider", return_value=response):
                voice.issue_voice_exchange(repo, commit, EPISODE, bundle, policy=policy,
                                           key_id=KEY_ID, private_key_seed=SEED,
                                           api_key="test-only-api-key", env=ENV)
            wrong_ref = QA_REPOSITORY + "/.github/workflows/other.yml@refs/tags/voice-v1"
            for wrong_policy in (_policy(workflow_ref=wrong_ref),
                                 _policy(workflow_sha="b" * 40)):
                with self.subTest(wrong_policy=wrong_policy.workflow_ref,
                                  sha=wrong_policy.workflow_sha):
                    with self.assertRaisesRegex(QaHold, "not from the pinned owner run"):
                        voice.verify_voice_exchange(bundle, wrong_policy)

    def test_rejects_signature_and_saved_byte_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, bundle, policy, _, _ = self._issue(Path(directory))
            for name in (voice.REQUEST_FILE, voice.RESPONSE_FILE,
                         voice.PROVIDER_AUDIO_FILE, voice.NARRATION_FILE):
                with self.subTest(name=name):
                    path = bundle / name
                    original = path.read_bytes()
                    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                    with self.assertRaisesRegex(QaHold, "bytes differ from signed hashes"):
                        voice.verify_voice_exchange(bundle, policy)
                    path.write_bytes(original)

            signature_path = bundle / voice.SIGNATURE_FILE
            original_signature = signature_path.read_bytes()
            envelope = json.loads(original_signature)
            signature = bytearray(base64.b64decode(envelope["signature"]))
            signature[0] ^= 1
            envelope["signature"] = base64.b64encode(signature).decode("ascii")
            signature_path.write_bytes(_json(envelope))
            with self.assertRaisesRegex(QaHold, "exchange signature is invalid"):
                voice.verify_voice_exchange(bundle, policy)

            exchange_path = bundle / voice.EXCHANGE_FILE
            record = json.loads(exchange_path.read_bytes())
            record["voice"] = "OtherVoice"
            changed_record = _json(record)
            exchange_path.write_bytes(changed_record)
            envelope = json.loads(original_signature)
            envelope["exchange_sha256"] = digest_bytes(changed_record)
            signature_path.write_bytes(_json(envelope))
            with self.assertRaisesRegex(QaHold, "exchange signature is invalid"):
                voice.verify_voice_exchange(bundle, policy)

    def test_malformed_signature_key_id_and_policy_keys_hold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, bundle, policy, _, _ = self._issue(root)
            signature_path = bundle / voice.SIGNATURE_FILE
            envelope = json.loads(signature_path.read_bytes())
            envelope["key_id"] = [KEY_ID]
            signature_path.write_bytes(_json(envelope))
            with self.assertRaisesRegex(QaHold, "signature envelope is invalid"):
                voice.verify_voice_exchange(bundle, policy)
            with patch.object(voice, "_post_provider") as post:
                with self.assertRaisesRegex(QaHold, "signing key is unavailable"):
                    voice.issue_voice_exchange(source[0], source[1], EPISODE,
                                               root / "bad-issuer", policy=policy,
                                               key_id=[KEY_ID], private_key_seed=SEED,
                                               api_key="test-only", env=ENV)
            post.assert_not_called()

        for bad_keys in ([KEY_ID], "test-key-v1", {KEY_ID: "not raw public bytes"}):
            with self.subTest(bad_keys=bad_keys):
                with self.assertRaisesRegex(QaHold, "public keys are invalid"):
                    voice.VoiceExchangePolicy(WORKFLOW_REF, WORKFLOW_SHA, bad_keys)

    def test_rejects_unreviewed_wav_format(self) -> None:
        with self.assertRaisesRegex(QaHold, "must be mono 24 kHz PCM16"):
            voice._derive_narration(_wav(rate=16_000), 1_920)

    def test_final_mix_contains_control_voice_and_unrelated_audio_holds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, bundle, policy, _, _ = self._issue(root)
            candidate, asset = _candidate(root, source, bundle)
            copied = voice.copy_voice_exchange(bundle, candidate.episode_dir, policy)
            narration = (bundle / voice.NARRATION_FILE).read_bytes()
            related = _final_audio(root / "mixed-final.wav", narration, related=True)
            proof = voice.verify_candidate_voice(candidate, asset, copied, related, policy)
            self.assertEqual(proof["kind"], voice.PROOF_KIND)
            self.assertEqual(proof["audio_match"]["method"], "waveform_correlation_v1")
            self.assertGreater(proof["audio_match"]["global_correlation"], 0.9)
            unrelated = _final_audio(root / "unrelated-final.wav", narration, related=False)
            with self.assertRaisesRegex(QaHold, "does not contain the exact control narration"):
                voice.verify_candidate_voice(candidate, asset, copied, unrelated, policy)


if __name__ == "__main__":
    unittest.main()
