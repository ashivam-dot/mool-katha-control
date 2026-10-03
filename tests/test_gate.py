"""Gate checks over a synthetic QA fixture; never production evidence."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trusted_qa.audio_quality import AudioQualityObservation
from trusted_qa.common import QaHold, digest_bytes, digest_file
from trusted_qa.fetch import _readable_and_links
from trusted_qa.observations import OFL_GRANTS
from trusted_qa.provenance import TANPURA_PARAMETERS, verify_generated_asset
from trusted_qa.signing import sign_assembled_review
from trusted_qa.tests.test_assemble import build_synthetic_approved_review
from trusted_release.executor import ReleaseHold, ReleasePlan, ReleasePolicy
from trusted_release.gate import (_check_asset_origin, _check_review_evidence,
                                  sign_gate_attestation)


def _save(root: Path, name: str, raw: bytes) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def _asset_candidate(root: Path, *, blobs: dict[str, bytes], spec: dict,
                     manifest: dict, asset: dict, path: Path,
                     archived: dict[str, str] | None = None) -> SimpleNamespace:
    """Small gate fixture; Git ancestry itself is covered by QA provenance tests."""
    commit = "a" * 40
    archive_members = archived or {}

    def committed_blob(name: str, claimed_commit: str | None = None) -> bytes:
        if name not in blobs or claimed_commit not in (None, commit):
            raise QaHold("synthetic source commit or blob changed")
        return blobs[name]

    def recheck() -> None:
        if digest_file(path) != asset["sha256"] or any(
                digest_file(root / name) != expected
                for name, expected in archive_members.items()):
            raise QaHold("synthetic archive or asset changed")

    return SimpleNamespace(root=root, episode_dir=root / "content/episodes/ep004",
                           episode_id="ep004", spec=spec, manifest=manifest,
                           assets={asset["id"]: asset}, asset_paths={asset["id"]: path},
                           archive_members=archive_members, committed_blob=committed_blob,
                           recheck=recheck)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path, self.review, self.candidate = build_synthetic_approved_review(self.root)
        self.qa_private = Ed25519PrivateKey.generate()
        self.gate_private = Ed25519PrivateKey.generate()
        public = lambda key: key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.qa_ref = "ashivam-dot/mool-katha-control/.github/workflows/qa.yml@refs/tags/qa-v1"
        self.gate_ref = "ashivam-dot/mool-katha-control/.github/workflows/gate.yml@refs/tags/gate-v1"
        self.review["qa_run"]["workflow_ref"] = self.qa_ref
        self.review["qa_run"]["workflow_sha"] = "f" * 40
        # The fixture explicitly marks model evidence as synthetic. The gate's
        # success test changes only that label so it exercises protocol wiring.
        self.quality_dir = self.root / "synthetic-quality"
        quality_path = self.quality_dir / "observation.json"
        quality = json.loads(quality_path.read_text())
        quality["basis"] = "Gemini model observation of actual full final audio; not human listening"
        quality_raw = (json.dumps(quality, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")) + "\n").encode()
        quality_path.write_bytes(quality_raw)
        (self.candidate.episode_dir / "agent-audio-quality-observation.json").write_bytes(quality_raw)
        self.review["audio_review"]["quality_observation"]["sha256"] = hashlib.sha256(
            quality_raw).hexdigest()
        self.path.write_text(json.dumps(self.review, ensure_ascii=False, sort_keys=True) + "\n")
        self.quality = AudioQualityObservation(
            quality, quality_path, self.candidate.episode_dir / "agent-audio-quality-observation.json",
            digest_file(quality_path), self.quality_dir / "request.json",
            self.quality_dir / "response.json")
        self.run = {name: self.review["qa_run"][name] for name in
                    ("system", "repository", "workflow_ref", "workflow_sha", "run_id", "run_attempt")}
        sign_assembled_review(self.candidate, self.review, self.path, self.quality, self.run,
                              "qa-key-1", self.qa_private.private_bytes(
                                  serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                  serialization.NoEncryption()))
        self.policy = ReleasePolicy(
            qa_repository="ashivam-dot/mool-katha-control", qa_workflow_ref=self.qa_ref,
            qa_workflow_sha="f" * 40, qa_keys={"qa-key-1": public(self.qa_private)},
            gate_workflow_ref=self.gate_ref, gate_workflow_sha="e" * 40,
            gate_keys={"gate-key-1": public(self.gate_private)},
            youtube_channel_id="youtube-id", youtube_handle="moolkatha",
            instagram_channel_id="instagram-id", instagram_handle="moolkatha.hindi",
            cloudinary_cloud="mooltest")
        self.plan = ReleasePlan(
            episode_id=self.candidate.episode_id, source_commit=self.candidate.source_commit,
            archive_sha256=digest_file(self.root / "media.tar"),
            video_sha256=self.candidate.hashes["video"], review_sha256=digest_file(self.path),
            due_at=datetime(2030, 1, 2, 13, 30, tzinfo=timezone.utc),
            youtube_text="Synthetic source-checked test", instagram_text="Synthetic test caption",
            youtube_metadata={"youtube": {"title": "Synthetic test", "categoryId": "27",
                                          "privacy": "public", "madeForKids": False,
                                          "notifySubscribers": True, "isAiGenerated": False,
                                          "embeddable": True}})
        self.gate_run = {"system": "github_actions", "repository": self.policy.qa_repository,
                         "workflow_ref": self.gate_ref, "workflow_sha": "e" * 40,
                         "run_id": 124, "run_attempt": 1}
        self.gate_seed = self.gate_private.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
            serialization.NoEncryption())

    def run_gate(self):
        env = {"GITHUB_REPOSITORY": self.gate_run["repository"],
               "GITHUB_WORKFLOW_REF": self.gate_run["workflow_ref"],
               "GITHUB_WORKFLOW_SHA": self.gate_run["workflow_sha"],
               "GITHUB_RUN_ID": str(self.gate_run["run_id"]),
               "GITHUB_RUN_ATTEMPT": str(self.gate_run["run_attempt"])}
        with patch.dict("os.environ", env):
            return sign_gate_attestation(self.plan, self.policy, self.candidate.episode_dir,
                                         self.root / "media.tar", self.root / "source",
                                         self.gate_run, "gate-key-1", self.gate_seed)

    def test_complete_signed_gate_binds_exact_inputs(self):
        attestation, signature = self.run_gate()
        self.assertTrue(attestation.is_file() and signature.is_file())
        self.assertEqual(json.loads(attestation.read_text())["review_sha256"], self.plan.review_sha256)
        self.assertEqual(json.loads(signature.read_text())["key_id"], "gate-key-1")

    def test_changed_source_fetch_blocks_gate_before_signing(self):
        ref = self.review["claim_findings"][0]["primary_fetch"]["response_ref"]
        (self.candidate.episode_dir / ref).write_bytes(b"tampered")
        with self.assertRaises(ReleaseHold):
            self.run_gate()
        self.assertFalse((self.candidate.episode_dir / "release-gate-signature.json").exists())

    def test_changed_frame_sheet_blocks_gate_before_signing(self):
        ref = self.review["qa_run"]["frame_batch_review"]["batches"][0]["file"]
        (self.candidate.episode_dir / ref).write_bytes(b"changed after frame review")
        with self.assertRaisesRegex(ReleaseHold, "frame batch"):
            self.run_gate()
        self.assertFalse((self.candidate.episode_dir / "release-gate-signature.json").exists())

    def test_changed_origin_response_blocks_gate_before_signing(self):
        ref = self.review["asset_findings"][0]["origin_proof"]["origin_fetch"]["response_ref"]
        (self.candidate.episode_dir / ref).write_bytes(b"changed after origin review")
        with self.assertRaisesRegex(ReleaseHold, "origin"):
            self.run_gate()
        self.assertFalse((self.candidate.episode_dir / "release-gate-signature.json").exists())

    def test_missing_origin_or_frame_coverage_holds(self):
        review = json.loads(json.dumps(self.review))
        del review["asset_findings"][0]["origin_proof"]
        with self.assertRaisesRegex(ReleaseHold, "origin"):
            _check_review_evidence(self.candidate.episode_dir, self.candidate, review)
        review = json.loads(json.dumps(self.review))
        review["qa_run"]["frame_batch_review"]["batches"][0]["checked_indices"] = []
        with self.assertRaisesRegex(ReleaseHold, "frame batch"):
            _check_review_evidence(self.candidate.episode_dir, self.candidate, review)

    def test_wrong_gate_workflow_or_key_blocks_signing(self):
        self.gate_run["workflow_sha"] = "d" * 40
        with self.assertRaisesRegex(ReleaseHold, "not pinned"):
            self.run_gate()
        self.assertFalse((self.candidate.episode_dir / "release-gate-signature.json").exists())

    def test_generated_proof_replays_receipt_and_control_result(self):
        root = self.root / "generated-proof"
        source_name = "pipeline/src/ytc/drone.py"
        generator = b"synthetic checked source, never executed\n"
        name = "content/episodes/ep004/work/sfx/tanpura.wav"
        wav = _save(root, name, b"synthetic WAV; PCM reproduction is patched in this gate test")
        asset = {"id": "music:one", "role": "music", "file": name,
                 "sha256": digest_file(wav), "origin": "internal:pipeline/src/ytc/drone.py",
                 "rights_url": "internal:pipeline/src/ytc/drone.py", "license": "Original",
                 "rights_basis": "Original tanpura synthesized by the owned generator for this exact episode."}
        record = {"schema": "ytc.generated-asset/v1", "kind": "synthesized_tanpura",
                  "generator": {"file": source_name, "sha256": digest_bytes(generator),
                                "commit": "a" * 40},
                  "parameters": TANPURA_PARAMETERS, "format": "WAV PCM_16",
                  "numpy_version": "2.5.3", "soundfile_version": "0.14.0",
                  "output": {"file": "tanpura.wav", "sha256": asset["sha256"]}}
        receipt_name = "content/episodes/ep004/work/sfx/tanpura.provenance.json"
        receipt = _save(root, receipt_name, (json.dumps(record, sort_keys=True) + "\n").encode())
        manifest = {"music_asset_id": asset["id"], "generated_asset_receipts": [
            {"asset_id": asset["id"], "file": receipt_name, "sha256": digest_file(receipt)}]}
        candidate = _asset_candidate(root, blobs={source_name: generator},
                                     spec={"music": {"path": "tanpura"}}, manifest=manifest,
                                     asset=asset, path=wav,
                                     archived={name: asset["sha256"],
                                               receipt_name: digest_file(receipt)})
        metrics = {"frames": 2_649_600, "max_abs_lsb": 0.5}
        with patch("trusted_qa.provenance.DRONE_SHA256", digest_bytes(generator)), \
             patch("trusted_qa.provenance.check_tanpura", return_value=metrics):
            proof = verify_generated_asset(candidate, asset)
            row = {"origin_proof": proof, "rights_fetch": {
                "origin": asset["origin"], "checked_at": "2026-10-03T12:00:00+00:00",
                "content_sha256": asset["sha256"],
                "provenance_excerpt": "Original tanpura synthesized by the owned generator"}}
            _check_asset_origin(candidate.episode_dir, candidate, asset, row, asset["id"])

            changed = deepcopy(row)
            changed["origin_proof"]["metrics"]["frames"] -= 1
            with self.assertRaisesRegex(ReleaseHold, "signed generation proof"):
                _check_asset_origin(candidate.episode_dir, candidate, asset, changed, asset["id"])

            changed = deepcopy(row)
            changed["rights_fetch"]["content_sha256"] = "f" * 64
            with self.assertRaisesRegex(ReleaseHold, "internal rights"):
                _check_asset_origin(candidate.episode_dir, candidate, asset, changed, asset["id"])

            receipt = candidate.root / proof["receipt_file"]
            receipt.write_bytes(b"changed after the signed generation proof")
            with self.assertRaisesRegex(ReleaseHold, "generation check"):
                _check_asset_origin(candidate.episode_dir, candidate, asset, row, asset["id"])

    def test_font_proof_rechecks_pinned_ttf_and_ofl_bytes(self):
        root = self.root / "font-proof"
        name = "pipeline/assets/fonts/Test-Regular.ttf"
        license_name = "pipeline/assets/fonts/Test-OFL.txt"
        font = b"synthetic TTF bytes from the pinned upstream response"
        license_raw = ("Copyright (c) 2026, Test Foundry.\n" + "\n".join(OFL_GRANTS) + "\n").encode()
        font_path = _save(root, name, font)
        source = {"font": {"file": name, "sha256": digest_bytes(font), "commit": "a" * 40},
                  "license": {"file": license_name, "sha256": digest_bytes(license_raw),
                              "commit": "a" * 40},
                  "license_name": "SIL Open Font License 1.1",
                  "copyright": "Copyright (c) 2026, Test Foundry."}
        revision = "a" * 40
        origin = f"https://raw.githubusercontent.com/google/fonts/{revision}/ofl/test/Test-Regular.ttf"
        rights_url = f"https://raw.githubusercontent.com/google/fonts/{revision}/ofl/test/OFL.txt"
        asset = {"id": "font:one", "role": "font", "file": name,
                 "sha256": digest_file(font_path), "origin": origin, "rights_url": rights_url,
                 "license": "SIL Open Font License 1.1"}
        candidate = _asset_candidate(root, blobs={name: font, license_name: license_raw},
                                     spec={}, manifest={"font_asset_ids": [asset["id"]],
                                                        "font_sources": [source]},
                                     asset=asset, path=font_path)
        response_ref = f"agent-qa-responses/{asset['sha256']}.bin"
        _save(candidate.episode_dir, response_ref, font)
        license_ref = f"agent-qa-responses/{digest_bytes(license_raw)}.bin"
        _save(candidate.episode_dir, license_ref, license_raw)
        visible, links = _readable_and_links(license_raw, "text/plain")
        self.assertEqual(links, ())
        snapshot_raw = (visible + "\n").encode()
        snapshot_ref = f"agent-qa-snapshots/{digest_bytes(snapshot_raw)}.txt"
        _save(candidate.episode_dir, snapshot_ref, snapshot_raw)
        row = {"origin_proof": {"kind": "control_upstream_font_v1", "source": source,
                                "upstream_revision": revision,
                                "font_download": {"url": origin, "final_url": origin,
                                                  "response_ref": response_ref,
                                                  "sha256": asset["sha256"]},
                                "license_sha256": digest_bytes(license_raw)},
               "rights_fetch": {"url": rights_url, "fetched_at": "2026-10-03T12:00:00+00:00",
                                "http_status": 200, "response_sha256": digest_bytes(license_raw),
                                "response_ref": license_ref, "snapshot_ref": snapshot_ref,
                                "snapshot_sha256": digest_bytes(snapshot_raw),
                                "license_excerpt": "SIL OPEN FONT LICENSE Version 1.1"}}
        with patch("PIL.ImageFont.truetype", return_value=object()):
            _check_asset_origin(candidate.episode_dir, candidate, asset, row, asset["id"])

            changed = deepcopy(row)
            changed["origin_proof"]["upstream_revision"] = "b" * 40
            with self.assertRaisesRegex(ReleaseHold, "font URL or source"):
                _check_asset_origin(candidate.episode_dir, candidate, asset, changed, asset["id"])

            changed = deepcopy(row)
            changed["origin_proof"]["license_sha256"] = "f" * 64
            with self.assertRaisesRegex(ReleaseHold, "upstream OFL bytes"):
                _check_asset_origin(candidate.episode_dir, candidate, asset, changed, asset["id"])

            ttf = candidate.episode_dir / row["origin_proof"]["font_download"]["response_ref"]
            ttf.write_bytes(b"changed after the pinned download")
            with self.assertRaisesRegex(ReleaseHold, "Google Fonts TTF"):
                _check_asset_origin(candidate.episode_dir, candidate, asset, row, asset["id"])


if __name__ == "__main__":
    unittest.main()
