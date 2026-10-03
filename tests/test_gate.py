"""Gate checks over a synthetic QA fixture; never production evidence."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trusted_qa.audio_quality import AudioQualityObservation
from trusted_qa.common import digest_file
from trusted_qa.signing import sign_assembled_review
from trusted_qa.tests.test_assemble import build_synthetic_approved_review
from trusted_release.executor import ReleaseHold, ReleasePlan, ReleasePolicy
from trusted_release.gate import _check_review_evidence, sign_gate_attestation


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


if __name__ == "__main__":
    unittest.main()
