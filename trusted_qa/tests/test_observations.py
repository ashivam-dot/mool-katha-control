"""Adversarial origin, rights, and exact-file linkage checks."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from trusted_qa.candidate import load_candidate
from trusted_qa.common import QaHold
from trusted_qa.fetch import fetch_exact_asset, fetch_observation
from trusted_qa.observations import VerifiedHttpAssetRights, collect_observations
from trusted_qa.tests.test_candidate import EPISODE, QA_ID, candidate_fixture


def _responses(candidate) -> dict[str, tuple[int, dict[str, str], bytes]]:
    pages: dict[str, tuple[int, dict[str, str], bytes]] = {}
    claim = candidate.ledger["claims"][0]
    for key in ("primary", "corroboration"):
        source = claim[key]
        text = (f"Printed {source['printed_verse_label']}. {source['excerpt']} "
                "Independent edition context provides the complete quoted passage.")
        pages[source["url"]] = (200, {"content-type": "text/plain"}, text.encode())
    for asset in candidate.assets.values():
        object_id = asset["source_object_id"]
        origin = (f"Official asset object {object_id}. Exact SHA-256 {asset['sha256']}. "
                  "This source page identifies the precise media object and its supplied file.")
        rights = (f"Rights for object {object_id} at {asset['origin']}. "
                  "CC0 1.0 permits commercial reuse, derivative work, and appropriate credit "
                  "for this exact media object.")
        pages[asset["origin"]] = (200, {"content-type": "text/plain"}, origin.encode())
        pages[asset["rights_url"]] = (200, {"content-type": "text/plain"}, rights.encode())
    return pages


class RightsObservationTests(unittest.TestCase):
    def _candidate(self, root: Path, *, external: bool = True):
        repo, archive, commit = candidate_fixture(root, external_rights=external)
        return load_candidate(repo, archive, root / "snapshot", EPISODE, commit, QA_ID)

    def test_fetches_both_pages_and_rechecks_object_and_file_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._candidate(root)
            responses = _responses(candidate)
            with patch("trusted_qa.fetch._request_once", side_effect=lambda url: responses[url]) as fetched:
                observations = collect_observations(candidate, root / "audit")
            called = {call.args[0] for call in fetched.call_args_list}
            for asset in candidate.assets.values():
                self.assertIn(asset["origin"], called)
                self.assertIn(asset["rights_url"], called)
                observations.asset_rights[asset["id"]].recheck(candidate, asset)
            first = next(iter(candidate.assets.values()))
            proof = observations.asset_rights[first["id"]]
            (candidate.episode_dir / proof.origin_page.response_ref).write_bytes(b"tampered origin page")
            with self.assertRaisesRegex(QaHold, "origin or rights page changed"):
                proof.recheck(candidate, first)

    def test_generic_cc0_page_for_unrelated_file_holds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._candidate(root)
            responses = _responses(candidate)
            first = next(iter(candidate.assets.values()))
            responses[first["rights_url"]] = (
                200, {"content-type": "text/plain"},
                b"CC0 1.0 is a generic licence. It says nothing about a particular image or file.")
            with patch("trusted_qa.fetch._request_once", side_effect=lambda url: responses[url]):
                with self.assertRaisesRegex(QaHold, "same object"):
                    collect_observations(candidate, root / "audit")

    def test_rights_page_cannot_claim_object_id_without_linking_the_exact_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._candidate(root)
            responses = _responses(candidate)
            first = next(iter(candidate.assets.values()))
            responses[first["rights_url"]] = (
                200, {"content-type": "text/plain"},
                (f"Rights label {first['source_object_id']}. CC0 1.0 is available for some media "
                 "but this page gives no exact origin URL or file fingerprint.").encode())
            with patch("trusted_qa.fetch._request_once", side_effect=lambda url: responses[url]):
                with self.assertRaisesRegex(QaHold, "generic licence page unrelated"):
                    collect_observations(candidate, root / "audit")

    def test_unrelated_origin_file_hash_holds_even_with_specific_rights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._candidate(root)
            responses = _responses(candidate)
            first = next(iter(candidate.assets.values()))
            object_id = first["source_object_id"]
            responses[first["origin"]] = (
                200, {"content-type": "text/plain"},
                (f"Official asset object {object_id}. Exact SHA-256 {'0' * 64}. "
                 "This is another media file in the same collection, with unrelated bytes.").encode())
            with patch("trusted_qa.fetch._request_once", side_effect=lambda url: responses[url]):
                with self.assertRaisesRegex(QaHold, "fingerprint the exact used file"):
                    collect_observations(candidate, root / "audit")

    def test_internal_generic_notes_hold_without_independent_role_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._candidate(root, external=False)
            claim = candidate.ledger["claims"][0]
            responses = {}
            for key in ("primary", "corroboration"):
                source = claim[key]
                text = (f"Printed {source['printed_verse_label']}. {source['excerpt']} "
                        "Independent edition context provides the complete quoted passage.")
                responses[source["url"]] = (200, {"content-type": "text/plain"}, text.encode())
            with patch("trusted_qa.fetch._request_once", side_effect=lambda url: responses[url]):
                with self.assertRaisesRegex(QaHold, "control-verifiable visual creation"):
                    collect_observations(candidate, root / "audit")

    def test_visible_official_download_must_match_exact_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._candidate(root)
            asset = next(iter(candidate.assets.values())).copy()
            official = "https://cdn.example.org/download/synthetic-visual-one-1234.png"
            asset["official_asset_url"] = official
            object_id = asset["source_object_id"]
            origin_html = (f"<p>Official source object {object_id} with a direct file link.</p>"
                           f"<a href='{official}'>Download exact visual media asset</a>").encode()
            rights_text = (f"Rights for {object_id} at {asset['origin']}. CC0 1.0 permits commercial "
                           "reuse and derivative work for the listed object.").encode()
            responses = {
                asset["origin"]: (200, {"content-type": "text/html"}, origin_html),
                asset["rights_url"]: (200, {"content-type": "text/plain"}, rights_text),
                official: (200, {"content-type": "image/png"},
                           candidate.asset_paths[asset["id"]].read_bytes()),
            }
            with patch("trusted_qa.fetch._request_once", side_effect=lambda url: responses[url]):
                origin = fetch_observation(asset["origin"], candidate.episode_dir, root / "audit")
                rights = fetch_observation(asset["rights_url"], candidate.episode_dir, root / "audit")
                download = fetch_exact_asset(official, asset["sha256"], candidate.episode_dir,
                                             root / "audit")
            VerifiedHttpAssetRights(origin, rights, object_id, download).recheck(candidate, asset)
            signed = VerifiedHttpAssetRights(origin, rights, object_id, download).signed_origin_proof(
                candidate, asset)
            self.assertEqual(signed["origin_fetch"]["visible_links"], [official])
            self.assertEqual(signed["exact_file"], {"method": "official_download",
                                                    "sha256": asset["sha256"],
                                                    "official_asset_url": official,
                                                    "response_ref": download.response_ref})
            (candidate.episode_dir / download.response_ref).write_bytes(b"unrelated image")
            with self.assertRaisesRegex(QaHold, "official URL bytes differ"):
                VerifiedHttpAssetRights(origin, rights, object_id, download).recheck(candidate, asset)

    def test_hidden_official_link_cannot_prove_exact_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._candidate(root)
            asset = next(iter(candidate.assets.values())).copy()
            official = "https://cdn.example.org/download/synthetic-visual-one-1234.png"
            asset["official_asset_url"] = official
            object_id = asset["source_object_id"]
            rights_text = (f"Rights for {object_id} at {asset['origin']}. CC0 1.0 permits "
                           "commercial reuse and derivative work for the listed object.").encode()
            for number, hidden_link in enumerate((
                f"<a hidden href='{official}'>Download exact media</a>",
                f"<style>.concealed{{display:none}}</style>"
                f"<a class='concealed' href='{official}'>Download exact media</a>",
            )):
                with self.subTest(hidden_link=hidden_link):
                    origin_html = (f"<p>Official source object {object_id} with an exact media file."
                                   f"</p>{hidden_link}").encode()
                    responses = {
                        asset["origin"]: (200, {"content-type": "text/html"}, origin_html),
                        asset["rights_url"]: (200, {"content-type": "text/plain"}, rights_text),
                        official: (200, {"content-type": "image/png"},
                                   candidate.asset_paths[asset["id"]].read_bytes()),
                    }
                    with patch("trusted_qa.fetch._request_once",
                               side_effect=lambda url: responses[url]):
                        origin = fetch_observation(asset["origin"], candidate.episode_dir,
                                                   root / f"audit-{number}")
                        rights = fetch_observation(asset["rights_url"], candidate.episode_dir,
                                                   root / f"audit-{number}")
                        download = fetch_exact_asset(official, asset["sha256"],
                                                     candidate.episode_dir, root / f"audit-{number}")
                    self.assertNotIn(official, origin.visible_links)
                    with self.assertRaisesRegex(QaHold, "absent from the visible origin page"):
                        VerifiedHttpAssetRights(origin, rights, object_id, download).recheck(
                            candidate, asset)
                    forged = replace(origin, visible_links=(official,))
                    with self.assertRaisesRegex(QaHold, "page extraction changed"):
                        VerifiedHttpAssetRights(forged, rights, object_id, download).recheck(
                            candidate, asset)


if __name__ == "__main__":
    unittest.main()
