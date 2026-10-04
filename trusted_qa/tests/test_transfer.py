"""The signer binds one approved QA result to exact bytes without publishing."""

from __future__ import annotations

import base64
import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trusted_qa.common import QaHold
from trusted_qa.frame_audit import AUDIT_POLICY, HEIGHT, WIDTH
from trusted_qa.sample_plan import SAMPLE_POLICY, canonical_sha256, plan_sampled_frames
from trusted_qa.transfer import SIGNING_CONTEXT, sign_transfer


COMMIT = "a" * 40
WORKFLOW_SHA = "b" * 40
WORKFLOW_REF = "ashivam-dot/mool-katha-control/.github/workflows/release-qa.yml@refs/tags/qa-v3"


def _save(path: Path, value: dict | bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class TransferSignerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.output = self.base / "qa"
        self.episode = self.output / "snapshot/content/episodes/ep014"
        self.private = self.output / "private"
        script = {"beats": [{"text": "नमस्ते"}]}
        raw_script_hash = _save(self.episode / "script.json", script)
        normalized = hashlib.sha256(json.dumps(script, ensure_ascii=False, sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
        qc = {"check": {"duration": 1.0}}
        render = {"beats": [{"start": 0.0, "end": 1.0}]}
        hashes = {"script_sha256": raw_script_hash,
                  "spec_sha256": _save(self.episode / "short.yaml", b"id: ep014\n"),
                  "video_sha256": _save(self.episode / "ep014.mp4", b"test-video-bytes"),
                  "qc_sha256": _save(self.episode / "qc.json", qc),
                  "evidence_sha256": _save(self.episode / "evidence.json", {}),
                  "manifest_sha256": _save(self.episode / "work/manifest.json", render)}
        _save(self.episode / "evidence_pending.json", {
            "episode_id": "ep014", "script_sha256": normalized,
            "video_sha256": hashes["video_sha256"]})
        frame_records = [{"index": index, "seconds": round((index - 1) / 10, 6),
                          "rgb_sha256": f"{index:064x}", "mean_luma": 100.0,
                          "stddev_luma": 20.0, "dark_fraction": 0.1,
                          "bright_fraction": 0.0,
                          "delta_previous": 0.5 if index > 1 else 0.0,
                          "rgb_delta_previous": 0.5 if index > 1 else 0.0,
                          "rgb_delta_two_back": 0.5 if index > 2 else 0.0}
                         for index in range(1, 11)]
        plan = plan_sampled_frames(frame_records, render["beats"], 1.0)
        indices = plan["sampled_indices"]
        sheet_hash = _save(self.episode / "agent-video-frames" / ("d" * 64 + ".jpg"),
                           b"synthetic sampled sheet")
        sheet_path = self.episode / "agent-video-frames" / ("d" * 64 + ".jpg")
        sheet_path.rename(sheet_path.with_name(sheet_hash + ".jpg"))
        sheet_name = f"agent-video-frames/{sheet_hash}.jpg"
        identity = {"file": sheet_name, "sha256": sheet_hash,
                    "indices": indices, "start_index": indices[0], "end_index": indices[-1],
                    "first_seconds": frame_records[indices[0] - 1]["seconds"],
                    "last_seconds": frame_records[indices[-1] - 1]["seconds"]}
        audit_hash = _save(self.episode / "agent-video-frame-audit.json", {
            "kind": "all_frame_pixel_temporal_audit_v2",
            "input_video_sha256": hashes["video_sha256"],
            "decoded_frame_count": 10, "scaled_width": WIDTH, "scaled_height": HEIGHT,
            "qc_duration_seconds": 1.0, "audit_policy": AUDIT_POLICY,
            "audit_policy_sha256": canonical_sha256(AUDIT_POLICY),
            "frames": frame_records, "frame_batches": [identity],
            "anomalies": plan["anomalies"],
            "sample_plan_sha256": canonical_sha256(plan)})
        call = {"provider": "Google Gemini API", "model": "gemini-test-model",
                "model_version": "test-version", "request_id": "synthetic-frame-request"}
        batch = {**identity, "decision": "clear", "uncertainty": "low",
                 "checked_indices": indices, "defect_indices": [],
                 "notes": "Synthetic sampled sheet contract for transfer testing only.",
                 "model_call": call, "request_sha256": "a" * 64,
                 "response_sha256": "b" * 64}
        coverage = {"mode": "sampled", "selected_frames": len(indices),
                    "decoded_frames": 10, "unsampled_frames": 10 - len(indices),
                    "fraction": round(len(indices) / 10, 6), "sheets": 1,
                    "selection_rule": SAMPLE_POLICY["kind"],
                    "selection_rule_sha256": plan["selection_rule_sha256"],
                    "selection_sha256": canonical_sha256(plan),
                    "max_gap_seconds": plan["max_gap_seconds"],
                    "reason": "Capacity-bounded model sample; unselected frames received deterministic pixel and temporal checks only.",
                    "provider_errors": []}
        frame_review = {"kind": "frame_sampled_visual_review_v1",
                        "input_video_sha256": hashes["video_sha256"],
                        "all_frame_audit_file": "agent-video-frame-audit.json",
                        "all_frame_audit_sha256": audit_hash,
                        "decoded_frame_count": 10,
                        "sample_plan": plan,
                        "sample_plan_sha256": canonical_sha256(plan),
                        "model_visual_coverage": coverage,
                        "batches": [batch]}
        _save(self.episode / "agent-qa-snapshots" / ("c" * 64 + ".txt"), b"source snapshot")
        approved = {"decision": "approved", "unresolved_items": [], "notes": "Checked exact input."}
        review = {
            "kind": "agent_episode_qa_v1", "episode_id": "ep014", **hashes,
            "qa_run": {"repository": "ashivam-dot/mool-katha-control",
                       "workflow_ref": WORKFLOW_REF, "workflow_sha": WORKFLOW_SHA,
                       "run_id": 123, "run_attempt": 1,
                       "frame_batch_review": frame_review},
            "release_review": approved, "audio_review": approved,
            "video_review": {**approved, "model_visual_coverage": coverage},
            "claim_findings": [approved],
            "asset_findings": [approved],
        }
        review_hash = _save(self.episode / "agent-release-review.json", review)
        _save(self.private / "run-result.json", {
            "status": "reviewed_unsigned", "episode_id": "ep014",
            "video_sha256": hashes["video_sha256"], "review_sha256": review_hash})
        _save(self.private / "candidate-source.json", {
            "episode_id": "ep014", "source_commit": COMMIT})
        self.seed = bytes(range(32))
        self.options = {"key_id": "qa-key-1", "private_key_b64": base64.b64encode(self.seed).decode(),
                        "workflow_sha": WORKFLOW_SHA, "run_id": 123, "attempt": 1}

    def test_signed_transfer_binds_review_and_nested_support(self) -> None:
        output = self.base / "transfer.tar"
        sign_transfer(self.output, "ep014", COMMIT, output, **self.options)
        with tarfile.open(output) as bundle:
            names = bundle.getnames()
            self.assertEqual(names[0], "qa-transfer.json")
            self.assertIn("agent-qa-snapshots/" + "c" * 64 + ".txt", names)
            manifest = json.load(bundle.extractfile("qa-transfer.json"))
            signature = json.load(bundle.extractfile("agent-release-signature.json"))
            review = bundle.extractfile("agent-release-review.json").read()
            self.assertEqual(manifest["kind"], "mool_katha_qa_transfer_v1")
            self.assertNotEqual(manifest["script_sha256"],
                                hashlib.sha256((self.episode / "script.json").read_bytes()).hexdigest())
            self.assertEqual({item["path"] for item in manifest["files"]}, set(names[1:]))
            for item in manifest["files"]:
                raw = bundle.extractfile(item["path"]).read()
                self.assertEqual(item["sha256"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(item["size"], len(raw))
            Ed25519PrivateKey.from_private_bytes(self.seed).public_key().verify(
                base64.b64decode(signature["signature"]), SIGNING_CONTEXT + review)

    def test_changed_review_is_held_before_signing(self) -> None:
        with (self.episode / "agent-release-review.json").open("ab") as output:
            output.write(b"\n")
        with self.assertRaisesRegex(QaHold, "different review bytes"):
            sign_transfer(self.output, "ep014", COMMIT, self.base / "transfer.tar", **self.options)

    def test_untrusted_workflow_is_held(self) -> None:
        with self.assertRaisesRegex(QaHold, "pinned release workflow"):
            sign_transfer(self.output, "ep014", COMMIT, self.base / "transfer.tar",
                          **(self.options | {"workflow_sha": "d" * 40}))

    def test_missing_all_frame_audit_is_held(self) -> None:
        (self.episode / "agent-video-frame-audit.json").unlink()
        with self.assertRaisesRegex(QaHold, "all-frame audit"):
            sign_transfer(self.output, "ep014", COMMIT, self.base / "transfer.tar", **self.options)

    def test_selected_indices_are_recomputed_before_signing(self) -> None:
        review_path = self.episode / "agent-release-review.json"
        review = json.loads(review_path.read_text())
        selected = review["qa_run"]["frame_batch_review"]["batches"][0]["checked_indices"]
        review["qa_run"]["frame_batch_review"]["batches"][0]["checked_indices"] = selected[:-1]
        new_hash = _save(review_path, review)
        result_path = self.private / "run-result.json"
        result = json.loads(result_path.read_text())
        result["review_sha256"] = new_hash
        _save(result_path, result)
        with self.assertRaisesRegex(QaHold, "did not inspect every selected tile"):
            sign_transfer(self.output, "ep014", COMMIT, self.base / "transfer.tar", **self.options)


if __name__ == "__main__":
    unittest.main()
