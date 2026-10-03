from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trusted_release.executor import CONTEXT, ReleaseHold, ReleasePlan, ReleasePolicy, release_pair


class FakeHost:
    def __init__(self):
        self.calls = 0

    def ensure_video(self, video: Path, public_id: str) -> str:
        self.calls += 1
        assert hashlib.sha256(video.read_bytes()).hexdigest() == public_id.rsplit("-", 1)[1]
        return f"https://res.cloudinary.com/mooltest/video/upload/{public_id}.mp4"


class FakeBuffer:
    def __init__(self):
        self.rows = {"youtube-id": [], "instagram-id": []}
        self.creates = []
        self.fail_instagram = False
        self.accept_then_lose_response = False
        self.channel_override = None

    def channels(self):
        return self.channel_override or [
            {"id": "youtube-id", "service": "youtube", "name": "@moolkatha", "displayName": "Mool Katha",
             "isDisconnected": False, "isLocked": False, "isQueuePaused": False},
            {"id": "instagram-id", "service": "instagram", "name": "@moolkatha.hindi",
             "displayName": "Mool Katha", "isDisconnected": False, "isLocked": False,
             "isQueuePaused": False},
        ]

    def posts(self, channel_id):
        return list(self.rows[channel_id])

    def post_detail(self, post_id, service):
        channel_id = "youtube-id" if service == "youtube" else "instagram-id"
        rows = [row for row in self.rows[channel_id] if row["id"] == post_id]
        return dict(rows[0]) if len(rows) == 1 else {}

    def create(self, payload):
        if payload["channelId"] == "instagram-id" and self.fail_instagram:
            raise ReleaseHold("injected Buffer failure")
        self.creates.append(payload)
        channel_id = payload["channelId"]
        service = "youtube" if channel_id == "youtube-id" else "instagram"
        post_id = f"post-{len(self.creates)}"
        post = {"id": post_id, "status": "scheduled", "dueAt": payload["dueAt"],
                "text": payload["text"], "channelId": channel_id, "channelService": service,
                "assets": [{"source": payload["assets"][0]["video"]["url"]}],
                "metadata": payload["metadata"]}
        self.rows[channel_id].append(post)
        if self.accept_then_lose_response:
            self.accept_then_lose_response = False
            raise ReleaseHold("response lost after Buffer accepted post")
        return {"id": post_id}


class ReleaseExecutorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifact = self.root / "artifact"
        self.artifact.mkdir()
        self.video = b"synthetic MP4 fixture; exact-byte verification only"
        self.video_hash = hashlib.sha256(self.video).hexdigest()
        self.archive = self.root / "draft.tar"
        self._write_archive(self.video)
        self.private_key = Ed25519PrivateKey.generate()
        public_key = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
        self.policy = ReleasePolicy(
            qa_repository="ashivam-dot/mool-katha-control",
            qa_workflow_ref=("ashivam-dot/mool-katha-control/.github/workflows/"
                             "trusted-qa.yml@refs/tags/qa-v1"),
            qa_workflow_sha="a" * 40, qa_keys={"qa-key-1": public_key},
            youtube_channel_id="youtube-id", youtube_handle="moolkatha",
            instagram_channel_id="instagram-id", instagram_handle="moolkatha.hindi",
            cloudinary_cloud="mooltest",
        )
        quality = {"kind": "full_final_audio_quality_model_observation_v1",
                   "basis": "Gemini model observation of actual full final audio; not human listening",
                   "input_video_sha256": self.video_hash}
        quality_raw = json.dumps(quality, sort_keys=True).encode()
        (self.artifact / "agent-audio-quality-observation.json").write_bytes(quality_raw)
        self.review = {
            "kind": "agent_episode_qa_v1", "episode_id": "ep004",
            "video_sha256": self.video_hash,
            "script_sha256": "1" * 64, "spec_sha256": "2" * 64,
            "qc_sha256": "3" * 64, "evidence_sha256": "4" * 64,
            "manifest_sha256": "5" * 64,
            "claim_findings": [], "asset_findings": [],
            "audio_review": {"quality_observation": {
                "file": "agent-audio-quality-observation.json",
                "sha256": hashlib.sha256(quality_raw).hexdigest()}},
            "video_review": {}, "qc_warning_dispositions": [],
            "production_agent_id": "agent:modal/ashivam-dot/mool-katha/call-12345",
            "qa_agent_id": "agent:github_actions/ashivam-dot/mool-katha-control/42/1",
            "qa_run": {"system": "github_actions", "repository": "ashivam-dot/mool-katha-control",
                       "workflow_ref": self.policy.qa_workflow_ref, "workflow_sha": "a" * 40,
                       "run_id": 42, "run_attempt": 1, "completed_at": "2030-01-01T01:00:00+00:00",
                       "review_model_calls": [
                           {"provider": "provider-one", "model": "model-one", "model_version": "v-one",
                            "request_id": "request-one"},
                           {"provider": "provider-two", "model": "model-two", "model_version": "v-two",
                            "request_id": "request-two"}]},
            "release_review": {"agent_id": "agent:github_actions/ashivam-dot/mool-katha-control/42/1",
                               "method": "agent_independent_release_review", "decision": "approved",
                               "unresolved_items": [], "verified_all_claims": True,
                               "verified_all_assets": True, "verified_asr_and_decoded_frames": True,
                               "verified_no_unresolved_concerns": True},
        }
        self._sign_review()
        self.plan = ReleasePlan(
            episode_id="ep004", source_commit="b" * 40,
            archive_sha256=hashlib.sha256(self.archive.read_bytes()).hexdigest(),
            video_sha256=self.video_hash,
            review_sha256=hashlib.sha256((self.artifact / "agent-release-review.json").read_bytes()).hexdigest(),
            due_at=datetime(2030, 1, 2, 13, 30, tzinfo=timezone.utc),
            youtube_text="Hindi source-checked Short\n\nAI voice disclosed",
            instagram_text="Hindi source-checked Reel\n\nAI voice disclosed",
            youtube_metadata={"youtube": {"title": "Hindi source-checked Short",
                                          "categoryId": "27", "privacy": "public",
                                          "madeForKids": False, "notifySubscribers": True,
                                          "isAiGenerated": False, "embeddable": True}},
        )
        self.buffer = FakeBuffer()
        self.host = FakeHost()
        self.receipt = self.root / "control-receipt.json"
        self.now = datetime(2030, 1, 1, tzinfo=timezone.utc)

    def _write_archive(self, video: bytes):
        with tarfile.open(self.archive, "w") as bundle:
            info = tarfile.TarInfo("content/episodes/ep004/ep004.mp4")
            info.size = len(video)
            bundle.addfile(info, io.BytesIO(video))

    def _sign_review(self):
        raw = json.dumps(self.review, sort_keys=True).encode()
        (self.artifact / "agent-release-review.json").write_bytes(raw)
        signature = {"kind": "agent_release_signature_v1", "algorithm": "Ed25519",
                     "key_id": "qa-key-1", "review_sha256": hashlib.sha256(raw).hexdigest(),
                     "signature": base64.b64encode(self.private_key.sign(CONTEXT + raw)).decode()}
        (self.artifact / "agent-release-signature.json").write_text(json.dumps(signature))

    def run_release(self, plan=None):
        return release_pair(plan or self.plan, self.policy, self.artifact, self.archive,
                            self.receipt, self.buffer, self.host, execute=True, now=self.now)

    def test_creates_pair_once_with_bound_metadata(self):
        result = self.run_release()
        self.assertEqual(result, {"youtube": "post-1", "instagram": "post-2"})
        self.assertEqual(self.run_release(), result)
        self.assertEqual(len(self.buffer.creates), 2)
        self.assertEqual(self.buffer.creates[0]["metadata"], self.plan.youtube_metadata)
        self.assertEqual(self.buffer.creates[1]["metadata"]["instagram"]["isAiGenerated"], True)
        self.assertEqual(self.buffer.creates[0]["dueAt"], self.buffer.creates[1]["dueAt"])

    def test_partial_success_resumes_only_missing_companion(self):
        self.buffer.fail_instagram = True
        with self.assertRaises(ReleaseHold):
            self.run_release()
        self.assertEqual(json.loads(self.receipt.read_text())["posts"], {"youtube": "post-1"})
        self.buffer.fail_instagram = False
        self.assertEqual(self.run_release(), {"youtube": "post-1", "instagram": "post-2"})
        self.assertEqual([item["channelId"] for item in self.buffer.creates],
                         ["youtube-id", "instagram-id"])

    def test_duplicate_buffer_post_holds(self):
        self.run_release()
        self.buffer.rows["youtube-id"].append({**self.buffer.rows["youtube-id"][0], "id": "duplicate"})
        with self.assertRaisesRegex(ReleaseHold, "duplicate or conflicting"):
            self.run_release()
        self.assertEqual(len(self.buffer.creates), 2)

    def test_changed_archive_media_holds_before_host_or_buffer(self):
        self._write_archive(b"different MP4")
        with self.assertRaisesRegex(ReleaseHold, "archive differs"):
            self.run_release()
        self.assertEqual(self.host.calls, 0)
        self.assertEqual(self.buffer.creates, [])

    def test_changed_remote_media_holds_without_creating_again(self):
        self.run_release()
        self.buffer.rows["instagram-id"][0]["assets"] = [
            {"source": "https://res.cloudinary.com/mooltest/video/upload/other.mp4"}]
        with self.assertRaisesRegex(ReleaseHold, "differs from signed release intent"):
            self.run_release()
        self.assertEqual(len(self.buffer.creates), 2)

    def test_changed_remote_ai_metadata_holds(self):
        self.run_release()
        self.buffer.rows["youtube-id"][0]["metadata"]["youtube"]["isAiGenerated"] = True
        with self.assertRaisesRegex(ReleaseHold, "metadata differs"):
            self.run_release()
        self.assertEqual(len(self.buffer.creates), 2)

    def test_unexpected_early_sent_post_holds(self):
        self.run_release()
        self.buffer.rows["youtube-id"][0]["status"] = "sent"
        with self.assertRaisesRegex(ReleaseHold, "differs from signed release intent"):
            self.run_release()
        self.assertEqual(len(self.buffer.creates), 2)

    def test_bad_signature_holds_before_host_or_buffer(self):
        signature = json.loads((self.artifact / "agent-release-signature.json").read_text())
        signature["signature"] = base64.b64encode(b"x" * 64).decode()
        (self.artifact / "agent-release-signature.json").write_text(json.dumps(signature))
        with self.assertRaisesRegex(ReleaseHold, "signature is invalid"):
            self.run_release()
        self.assertEqual(self.host.calls, 0)
        self.assertEqual(self.buffer.creates, [])

    def test_changed_quality_artifact_holds_before_host_or_buffer(self):
        (self.artifact / "agent-audio-quality-observation.json").write_text("{}")
        with self.assertRaisesRegex(ReleaseHold, "observation changed"):
            self.run_release()
        self.assertEqual(self.host.calls, 0)
        self.assertEqual(self.buffer.creates, [])

    def test_wrong_destination_holds_before_host_or_buffer(self):
        channels = self.buffer.channels()
        channels[1] = {**channels[1], "name": "@another.account"}
        self.buffer.channel_override = channels
        with self.assertRaisesRegex(ReleaseHold, "destination handle"):
            self.run_release()
        self.assertEqual(self.host.calls, 0)
        self.assertEqual(self.buffer.creates, [])

    def test_signed_review_with_wrong_qa_workflow_holds(self):
        self.review["qa_run"]["workflow_sha"] = "c" * 40
        self._sign_review()
        changed = replace(self.plan, review_sha256=hashlib.sha256(
            (self.artifact / "agent-release-review.json").read_bytes()).hexdigest())
        with self.assertRaisesRegex(ReleaseHold, "trusted independent workflow"):
            self.run_release(changed)
        self.assertEqual(self.host.calls, 0)

    def test_accepted_post_with_lost_response_holds_on_retry(self):
        self.buffer.accept_then_lose_response = True
        with self.assertRaises(ReleaseHold):
            self.run_release()
        self.assertFalse(self.receipt.exists())
        with self.assertRaisesRegex(ReleaseHold, "no trusted create receipt"):
            self.run_release()
        self.assertEqual(len(self.buffer.creates), 1)

    def test_changed_intent_cannot_reuse_receipt(self):
        self.run_release()
        changed = replace(self.plan, instagram_text="Changed caption")
        with self.assertRaisesRegex(ReleaseHold, "another candidate"):
            self.run_release(changed)
        self.assertEqual(len(self.buffer.creates), 2)

    def test_explicit_execution_and_lead_time(self):
        with self.assertRaisesRegex(ReleaseHold, "dormant"):
            release_pair(self.plan, self.policy, self.artifact, self.archive,
                         self.receipt, self.buffer, self.host, now=self.now)
        with self.assertRaisesRegex(ReleaseHold, "two hours"):
            release_pair(self.plan, self.policy, self.artifact, self.archive,
                         self.receipt, self.buffer, self.host, execute=True,
                         now=self.plan.due_at - timedelta(hours=1))


if __name__ == "__main__":
    unittest.main()
