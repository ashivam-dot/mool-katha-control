"""The signed voice origin can be checked offline while rights still hold."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trusted_qa import voice_exchange as voice
from trusted_qa.common import QaHold
from trusted_qa.observations import collect_observations
from trusted_qa.tests.test_voice_exchange import (
    ENV, EPISODE, KEY_ID, SEED, _candidate, _final_audio, _policy,
    _response, _source, _wav,
)


class ControlVoiceObservationTests(unittest.TestCase):
    def test_requires_an_external_signed_bundle_and_owner_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _source(root)
            bundle = root / "control-artifact"
            policy = _policy()
            with patch.object(voice, "_post_provider", return_value=_response(_wav())):
                voice.issue_voice_exchange(source[0], source[1], EPISODE, bundle,
                                           policy=policy, key_id=KEY_ID,
                                           private_key_seed=SEED, api_key="test-only",
                                           env=ENV)
            candidate, _ = _candidate(root, source, bundle)
            candidate.ledger["claims"] = []
            audio = _final_audio(root / "final.wav",
                                 (bundle / voice.NARRATION_FILE).read_bytes(), related=True)
            audit = root / "audit"
            audit.mkdir()
            with patch("trusted_qa.observations.fetch_observation") as fetch:
                with self.assertRaisesRegex(QaHold, "pinned control voice policy"):
                    collect_observations(candidate, audit,
                                         voice_bundle_path=bundle,
                                         final_audio_path=audio)
            fetch.assert_not_called()
            self.assertFalse((candidate.episode_dir / voice.QA_BUNDLE_DIR).exists())
            candidate.assets["voice:one"]["origin"] = "https://example.org/claimed-voice"
            with patch("trusted_qa.observations.fetch_observation") as fetch:
                with self.assertRaisesRegex(QaHold, "requires a signed control exchange"):
                    collect_observations(candidate, audit)
            fetch.assert_not_called()

    def test_verified_origin_is_copied_before_explicit_rights_hold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _source(root)
            bundle = root / "control-artifact"
            policy = _policy()
            with patch.object(voice, "_post_provider", return_value=_response(_wav())):
                voice.issue_voice_exchange(source[0], source[1], EPISODE, bundle,
                                           policy=policy, key_id=KEY_ID,
                                           private_key_seed=SEED, api_key="test-only",
                                           env=ENV)
            candidate, _ = _candidate(root, source, bundle)
            candidate.ledger["claims"] = []
            audio = _final_audio(root / "final.wav",
                                 (bundle / voice.NARRATION_FILE).read_bytes(), related=True)
            audit = root / "audit"
            audit.mkdir()
            with patch("trusted_qa.observations.fetch_observation") as fetch:
                with self.assertRaisesRegex(QaHold, "commercial-use rights remain unresolved"):
                    collect_observations(candidate, audit,
                                         voice_bundle_path=bundle, voice_policy=policy,
                                         final_audio_path=audio)
            fetch.assert_not_called()
            copied = candidate.episode_dir / voice.QA_BUNDLE_DIR
            self.assertEqual(voice.verify_voice_exchange(copied, policy)["episode_id"], EPISODE)
            observation = json.loads((audit / "control-voice-origin.json").read_text())
            self.assertEqual(observation["rights_status"], "hold")
            self.assertEqual(observation["origin_proof"]["kind"], voice.PROOF_KIND)


if __name__ == "__main__":
    unittest.main()
