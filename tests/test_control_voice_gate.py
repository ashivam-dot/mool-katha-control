"""Gate voice replay uses QA-copied bytes and a fresh final-audio observation."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trusted_qa import voice_exchange as voice
from trusted_qa.tests.test_voice_exchange import (
    ENV, EPISODE, KEY_ID, SEED, _candidate, _final_audio, _policy,
    _response, _source, _wav,
)
from trusted_release.executor import ReleaseHold
from trusted_release.gate import _check_asset_origin, _replay_control_voice_origin


class ControlVoiceGateTests(unittest.TestCase):
    def test_replay_rejects_tampering_and_keeps_commercial_rights_hold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _source(root)
            bundle = root / "owner-artifact"
            policy = _policy()
            with patch.object(voice, "_post_provider", return_value=_response(_wav())):
                voice.issue_voice_exchange(source[0], source[1], EPISODE, bundle,
                                           policy=policy, key_id=KEY_ID,
                                           private_key_seed=SEED, api_key="test-only",
                                           env=ENV)
            candidate, asset = _candidate(root, source, bundle)
            qa_bundle = voice.copy_voice_exchange(bundle, candidate.episode_dir, policy)
            narration = (bundle / voice.NARRATION_FILE).read_bytes()
            final_audio = _final_audio(root / "final.wav", narration, related=True)
            proof = voice.verify_candidate_voice(candidate, asset, qa_bundle,
                                                 final_audio, policy)
            row = {"origin_proof": proof}
            replay = lambda checked=row, audio=final_audio, pinned=policy: (
                _replay_control_voice_origin(candidate.episode_dir, candidate, asset,
                                             checked, asset["id"], voice_policy=pinned,
                                             final_audio_path=audio))
            self.assertEqual(replay(), proof)
            with self.assertRaisesRegex(ReleaseHold, "commercial-use rights remain unresolved"):
                _check_asset_origin(candidate.episode_dir, candidate, asset, row,
                                    asset["id"], voice_policy=policy,
                                    final_audio_path=final_audio)

            changed = copy.deepcopy(row)
            changed["origin_proof"]["provider_audio_sha256"] = "f" * 64
            with self.assertRaisesRegex(ReleaseHold, "signed control voice proof differs"):
                replay(changed)
            different_decoder = copy.deepcopy(row)
            different_decoder["origin_proof"]["final_audio_sha256"] = "a" * 64
            self.assertEqual(replay(different_decoder), proof)
            changed = copy.deepcopy(row)
            changed["origin_proof"]["audio_match"]["global_correlation"] = 0.1
            with self.assertRaisesRegex(ReleaseHold, "audio observation is malformed"):
                replay(changed)
            unrelated = _final_audio(root / "unrelated.wav", narration, related=False)
            with self.assertRaisesRegex(ReleaseHold, "control voice replay failed"):
                replay(audio=unrelated)
            with self.assertRaisesRegex(ReleaseHold, "pinned voice policy"):
                replay(pinned=None)
            spoofed = dict(asset, origin="https://example.org/claimed-voice")
            with self.assertRaisesRegex(ReleaseHold, "lacks a signed control provider-origin"):
                _check_asset_origin(candidate.episode_dir, candidate, spoofed, row,
                                    asset["id"])

            response_path = qa_bundle / voice.RESPONSE_FILE
            raw = response_path.read_bytes()
            response_path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            with self.assertRaisesRegex(ReleaseHold, "control voice replay failed"):
                replay()


if __name__ == "__main__":
    unittest.main()
