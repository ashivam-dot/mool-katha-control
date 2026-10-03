"""Focused, synthetic attack tests for independent asset provenance checks.

These fixtures exercise the control logic. They are not production rights or
provider evidence. The producer card and tanpura oracles are also compared
manually against fresh studio output when a generator version is reviewed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw
import numpy as np
import soundfile as sf

from trusted_qa.candidate import Candidate
from trusted_qa.common import QaHold, digest_bytes, digest_file
from trusted_qa.fetch import AssetDownloadObservation, FetchObservation
from trusted_qa.observations import OFL_GRANTS, VerifiedFontAssetRights, collect_observations
from trusted_qa.procedural import (_tanpura_samples, check_card, check_tanpura)
from trusted_qa.provenance import (TANPURA_PARAMETERS, font_source, inspect_voice_take,
                                   verify_generated_asset)


EPISODE = "ep004"
PREFIX = f"content/episodes/{EPISODE}"


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _git_repo(root: Path, files: dict[str, bytes]) -> tuple[Path, str]:
    repo = root / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "QA Fixture"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.test"], cwd=repo, check=True)
    for name, raw in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
    return repo, commit


class _Candidate:
    """Use the real immutable Git read while keeping unrelated episode checks out."""

    committed_blob = Candidate.committed_blob

    def __init__(self, root: Path, repo: Path, commit: str, *, spec: dict,
                 manifest: dict, assets: dict[str, dict], paths: dict[str, Path],
                 archived: dict[str, str], hashes: dict[str, str] | None = None):
        self.root, self.source_repo, self.source_commit = root, repo, commit
        self.episode_id = EPISODE
        self.spec, self.manifest, self.assets = spec, manifest, assets
        self.asset_paths, self.archive_members = paths, archived
        self.hashes = hashes or {}
        self.ledger = {"claims": []}

    @property
    def episode_dir(self) -> Path:
        return self.root / PREFIX

    def recheck(self) -> None:
        for name, digest in self.archive_members.items():
            if digest_file(self.root / name) != digest:
                raise QaHold("synthetic archive changed")
        for identity, path in self.asset_paths.items():
            if digest_file(path) != self.assets[identity]["sha256"]:
                raise QaHold("synthetic asset changed")


def _source(name: str, raw: bytes, commit: str | None) -> dict:
    return {"file": name, "sha256": digest_bytes(raw), "commit": commit}


def _save(root: Path, name: str, raw: bytes) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


class ProceduralMediaTests(unittest.TestCase):
    def test_card_rejects_added_picture_and_missing_typography(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "card.png"
            expected = Image.new("RGB", (1210, 2150), (16, 24, 32))
            ImageDraw.Draw(expected).rectangle((380, 460, 820, 780), fill=(244, 239, 230))
            expected.save(path)
            fonts = {"RozhaOne-Regular.ttf": b"fixture",
                     "Mukta-ExtraBold.ttf": b"fixture"}
            with patch("trusted_qa.procedural._reference_card", return_value=expected):
                self.assertEqual(check_card(path, {"kind": "fact", "big": "TEST", "small": ""},
                                            "0:TEST", "#101820", fonts)["background_outliers"], 0)
                added = expected.copy()
                ImageDraw.Draw(added).rectangle((0, 0, 200, 200), fill=(120, 40, 40))
                added.save(path)
                with self.assertRaisesRegex(QaHold, "pixels differ"):
                    check_card(path, {"kind": "fact", "big": "TEST", "small": ""},
                               "0:TEST", "#101820", fonts)
                blank = Image.new("RGB", (1210, 2150), (16, 24, 32))
                blank.save(path)
                with self.assertRaisesRegex(QaHold, "pixels differ"):
                    check_card(path, {"kind": "fact", "big": "TEST", "small": ""},
                               "0:TEST", "#101820", fonts)
            with patch("trusted_qa.procedural._reference_card", side_effect=OSError("bad TTF")):
                with self.assertRaisesRegex(QaHold, "font bytes cannot be rendered"):
                    check_card(path, {"kind": "fact", "big": "TEST", "small": ""},
                               "0:TEST", "#101820", fonts)

    def test_tanpura_checks_every_pcm_sample_and_rejects_an_inserted_recording(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tanpura.wav"
            samples = _tanpura_samples()
            sf.write(path, samples, 48_000)
            with patch("trusted_qa.procedural._tanpura_samples", return_value=samples):
                self.assertLessEqual(check_tanpura(path)["max_abs_lsb"], 1.1)
                edited, rate = sf.read(path, dtype="float32")
                edited[100_000:110_000] = np.float32(0.5)
                sf.write(path, edited, rate)
                with self.assertRaisesRegex(QaHold, "samples differ"):
                    check_tanpura(path)


class ReceiptBoundaryTests(unittest.TestCase):
    def _tanpura_candidate(self, root: Path) -> tuple[_Candidate, dict]:
        source_name = "pipeline/src/ytc/drone.py"
        generator = b"synthetic checked source, never executed\n"
        repo, commit = _git_repo(root, {source_name: generator})
        snapshot = root / "snapshot"
        name = f"{PREFIX}/work/sfx/tanpura.wav"
        wav = _save(snapshot, name, b"synthetic WAV; PCM check is patched only for receipt tests")
        asset = {"id": "music:one", "role": "music", "file": name,
                 "sha256": digest_file(wav), "origin": "internal:pipeline/src/ytc/drone.py",
                 "rights_url": "internal:pipeline/src/ytc/drone.py", "license": "Original"}
        record = {"schema": "ytc.generated-asset/v1", "kind": "synthesized_tanpura",
                  "generator": _source(source_name, generator, commit),
                  "parameters": TANPURA_PARAMETERS, "format": "WAV PCM_16",
                  "numpy_version": "2.5.3", "soundfile_version": "0.14.0",
                  "output": {"file": "tanpura.wav", "sha256": asset["sha256"]}}
        receipt_name = f"{PREFIX}/work/sfx/tanpura.provenance.json"
        receipt = _save(snapshot, receipt_name, _json(record))
        manifest = {"music_asset_id": asset["id"], "generated_asset_receipts": [
            {"asset_id": asset["id"], "file": receipt_name, "sha256": digest_file(receipt)}]}
        candidate = _Candidate(snapshot, repo, commit,
                               spec={"music": {"path": "tanpura"}}, manifest=manifest,
                               assets={asset["id"]: asset}, paths={asset["id"]: wav},
                               archived={name: digest_file(wav), receipt_name: digest_file(receipt)})
        return candidate, asset

    def test_receipt_needs_reviewed_source_and_control_reproduction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate, asset = self._tanpura_candidate(Path(directory))
            source_sha = candidate.committed_blob("pipeline/src/ytc/drone.py")
            with patch("trusted_qa.provenance.DRONE_SHA256", digest_bytes(source_sha)), \
                 patch("trusted_qa.provenance.check_tanpura", return_value={"frames": 2_649_600}):
                proof = verify_generated_asset(candidate, asset)
                self.assertEqual(proof["method"], "control_full_pcm_reproduction")
            with self.assertRaisesRegex(QaHold, "no reviewed control verifier"):
                verify_generated_asset(candidate, asset)
            with patch("trusted_qa.provenance.DRONE_SHA256", digest_bytes(source_sha)), \
                 patch("trusted_qa.provenance.check_tanpura",
                       side_effect=QaHold("synthesized samples do not match")):
                with self.assertRaisesRegex(QaHold, "samples do not match"):
                    verify_generated_asset(candidate, asset)

    def test_missing_and_changed_sidecars_cannot_be_laundered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate, asset = self._tanpura_candidate(Path(directory))
            original = candidate.manifest["generated_asset_receipts"]
            candidate.manifest["generated_asset_receipts"] = []
            with self.assertRaisesRegex(QaHold, "exact new production receipt is absent"):
                verify_generated_asset(candidate, asset)
            candidate.manifest["generated_asset_receipts"] = original
            sidecar = candidate.root / original[0]["file"]
            record = json.loads(sidecar.read_text())
            record["generator"]["commit"] = "f" * 40
            sidecar.write_bytes(_json(record))
            original[0]["sha256"] = digest_file(sidecar)
            candidate.archive_members[original[0]["file"]] = digest_file(sidecar)
            with self.assertRaisesRegex(QaHold, "claimed commit"):
                verify_generated_asset(candidate, asset)

    def test_malformed_receipt_index_and_music_spec_hold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate, asset = self._tanpura_candidate(Path(directory))
            index = candidate.manifest["generated_asset_receipts"][0]
            identity = index["asset_id"]
            index["asset_id"] = [identity]
            with self.assertRaisesRegex(QaHold, "receipt index is malformed"):
                verify_generated_asset(candidate, asset)
            index["asset_id"] = identity
            candidate.spec["music"] = "tanpura"
            with self.assertRaisesRegex(QaHold, "tanpura source or licence"):
                verify_generated_asset(candidate, asset)


class FontAndVoiceTests(unittest.TestCase):
    def _font_candidate(self, root: Path) -> tuple[_Candidate, dict, VerifiedFontAssetRights]:
        name = "pipeline/assets/fonts/Test-Regular.ttf"
        license_name = "pipeline/assets/fonts/Test-OFL.txt"
        font = b"synthetic TTF bytes from mocked upstream"
        license_raw = ("Copyright (c) 2026, Test Foundry.\n" + "\n".join(OFL_GRANTS) +
                       "\nThe OFL allows commercial video embedding and derivative fonts.\n").encode()
        repo, commit = _git_repo(root, {name: font, license_name: license_raw})
        snapshot = root / "snapshot"
        font_path = _save(snapshot, name, font)
        source = {"font": _source(name, font, commit),
                  "license": _source(license_name, license_raw, commit),
                  "license_name": "SIL Open Font License 1.1",
                  "copyright": "Copyright (c) 2026, Test Foundry."}
        revision = "a" * 40
        origin = f"https://raw.githubusercontent.com/google/fonts/{revision}/ofl/test/Test-Regular.ttf"
        rights_url = f"https://raw.githubusercontent.com/google/fonts/{revision}/ofl/test/OFL.txt"
        asset = {"id": "font:one", "role": "font", "file": name,
                 "sha256": digest_file(font_path), "origin": origin, "rights_url": rights_url,
                 "license": "SIL Open Font License 1.1"}
        episode_dir = snapshot / PREFIX
        response_ref = f"agent-qa-responses/{digest_bytes(font)}.bin"
        _save(episode_dir, response_ref, font)
        rights_ref = f"agent-qa-responses/{digest_bytes(license_raw)}.bin"
        _save(episode_dir, rights_ref, license_raw)
        visible = " ".join(license_raw.decode().split())
        snapshot_raw = (visible + "\n").encode()
        snapshot_ref = f"agent-qa-snapshots/{digest_bytes(snapshot_raw)}.txt"
        _save(episode_dir, snapshot_ref, snapshot_raw)
        evidence = VerifiedFontAssetRights(
            AssetDownloadObservation(origin, origin, digest_bytes(font), response_ref),
            FetchObservation(rights_url, "2026-10-03T12:00:00+00:00", 200,
                             digest_bytes(license_raw), rights_ref, snapshot_ref,
                             digest_bytes(snapshot_raw), visible, rights_url, "text/plain"),
            source)
        candidate = _Candidate(snapshot, repo, commit, spec={},
                               manifest={"font_asset_ids": [asset["id"]], "font_sources": [source]},
                               assets={asset["id"]: asset}, paths={asset["id"]: font_path},
                               archived={})
        return candidate, asset, evidence

    def test_font_requires_exact_upstream_ttf_and_same_revision_ofl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate, asset, evidence = self._font_candidate(Path(directory))
            with patch("PIL.ImageFont.truetype", return_value=object()):
                evidence.recheck(candidate, asset)
                self.assertEqual(evidence.signed_origin_proof(candidate, asset)["kind"],
                                 "control_upstream_font_v1")
                asset["rights_url"] = asset["rights_url"].replace("a" * 40, "b" * 40)
                with self.assertRaisesRegex(QaHold, "upstream font bytes|font does not use|upstream OFL"):
                    evidence.recheck(candidate, asset)

    def test_legacy_google_font_without_exact_source_record_holds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate, asset, _ = self._font_candidate(Path(directory))
            del candidate.manifest["font_sources"]
            with self.assertRaisesRegex(QaHold, "font and licence sources are missing"):
                font_source(candidate, asset)

    def test_malformed_rendered_font_ids_hold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate, asset, _ = self._font_candidate(Path(directory))
            candidate.manifest["font_asset_ids"].append({"unhashable": "id"})
            with self.assertRaisesRegex(QaHold, "rendered font IDs are malformed"):
                font_source(candidate, asset)

    def _voice_candidate(self, root: Path) -> tuple[_Candidate, dict, Path]:
        tts_name = "pipeline/src/ytc/tts.py"
        tts_raw = b"synthetic Gemini request builder source, never executed\n"
        repo, commit = _git_repo(root, {tts_name: tts_raw})
        snapshot = root / "snapshot"
        narration_name = f"{PREFIX}/work/narration.wav"
        take_name = f"{PREFIX}/work/take.wav"
        record_name = f"{PREFIX}/work/take.json"
        narration = _save(snapshot, narration_name, b"synthetic final narration")
        take = _save(snapshot, take_name, b"synthetic selected take")
        transcript = "राम नल से कहते हैं।"
        direction = "Read the exact Hindi script warmly."
        model = "gemini-3.8-flash-tts"
        request_body = {"model": model,
                        "input": [{"type": "user_input", "content": [{"type": "text", "text": transcript,
                                  "annotations": [{"type": "speech_metadata", "style": direction}]}]}],
                        "response_format": {"type": "audio", "mime_type": "audio/wav"},
                        "generation_config": {"speech_config": [{"voice": "Sulafat"}]},
                        "store": False}
        request_hash = digest_bytes(json.dumps(request_body, sort_keys=True, ensure_ascii=False,
                                               separators=(",", ":")).encode())
        provenance = {"api": "interactions",
                      "endpoint": "https://generativelanguage.googleapis.com/v1beta/interactions",
                      "requested_model": model, "served_model": model,
                      "response_id": None, "response_id_source": "not_returned",
                      "response_sha256": "a" * 64, "usage": {"totalTokens": 123},
                      "response_status": "completed", "response_created": "",
                      "voice": "Sulafat", "requested_at": "2026-10-03T12:00:00+00:00",
                      "request_body_sha256": request_hash, "request_body": request_body,
                      "response_metadata": {"status": "completed", "model": model,
                                            "usage": {"totalTokens": 123}},
                      "response_capture": "metadata_only_audio_omitted",
                      "transcript_sha256": digest_bytes(transcript.encode()),
                      "direction_sha256": digest_bytes(direction.encode()),
                      "returned_audio_sha256": "b" * 64,
                      "mime_type": "audio/wav",
                      "terms": {"url": "https://ai.google.dev/gemini-api/terms",
                                "last_updated": "2026-04-28",
                                "service": "Gemini Developer API (AI Studio key)"},
                      "generator": _source(tts_name, tts_raw, commit)}
        saved = {"request": {"script": transcript, "engine": "gemini",
                             "voice": "Sulafat", "speed": 1.0, "lang_code": "h",
                             "model": model, "direction": direction},
                 "provenance": provenance, "take_sha256": digest_file(take),
                 "narration_sha256": digest_file(narration)}
        record = _save(snapshot, record_name, _json(saved))
        asset = {"id": "voice:one", "role": "voice", "file": narration_name,
                 "sha256": digest_file(narration), "origin": f"internal:{record_name}",
                 "rights_url": "https://ai.google.dev/gemini-api/terms"}
        link = {"schema": "ytc.voice-take/v1", "record_file": record_name,
                "record_sha256": digest_file(record), "take_file": take_name,
                "take_sha256": digest_file(take),
                "source_narration_sha256": digest_file(narration),
                "final_narration_asset_id": asset["id"],
                "final_narration_sha256": asset["sha256"]}
        candidate = _Candidate(snapshot, repo, commit,
                               spec={"voice": {"engine": "gemini", "voice": "Sulafat",
                                               "speed": 1.0, "lang_code": "h", "model": model,
                                               "direction": direction},
                                     "beats": [{"text": transcript}]},
                               manifest={"narration_asset_id": asset["id"], "voice_take": link},
                               assets={asset["id"]: asset}, paths={asset["id"]: narration},
                               archived={take_name: digest_file(take),
                                         record_name: digest_file(record)})
        return candidate, asset, record

    def test_consistent_gemini_metadata_still_holds_for_unproved_provider_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, asset, _ = self._voice_candidate(root)
            generator = candidate.committed_blob("pipeline/src/ytc/tts.py")
            with patch("trusted_qa.provenance.TTS_SHA256", digest_bytes(generator)):
                checked = inspect_voice_take(candidate, asset)
                self.assertEqual(checked["kind"],
                                 "producer_gemini_receipt_checked_but_unproved_v1")
                with self.assertRaisesRegex(QaHold, "provider audio was omitted"):
                    collect_observations(candidate, root / "audit")

    def test_forged_gemini_request_hash_cannot_inherit_the_take(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate, asset, record_path = self._voice_candidate(Path(directory))
            saved = json.loads(record_path.read_text())
            saved["provenance"]["request_body"]["input"][0]["content"][0]["text"] = "unrelated narration"
            record_path.write_bytes(_json(saved))
            link = candidate.manifest["voice_take"]
            link["record_sha256"] = digest_file(record_path)
            candidate.archive_members[link["record_file"]] = digest_file(record_path)
            generator = candidate.committed_blob("pipeline/src/ytc/tts.py")
            with patch("trusted_qa.provenance.TTS_SHA256", digest_bytes(generator)):
                with self.assertRaisesRegex(QaHold, "request body"):
                    inspect_voice_take(candidate, asset)


if __name__ == "__main__":
    unittest.main()
