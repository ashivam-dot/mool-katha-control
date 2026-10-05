"""A public museum asset is released only with its object page and an official copy of the exact bytes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trusted_qa.common import QaHold, digest_bytes, utc_now, write_bytes_new
from trusted_qa.fetch import FetchObservation
from trusted_qa.observations import origin_proof

OBJECT_PAGE = "https://openaccess-api.clevelandart.org/api/artworks/136475"
OFFICIAL = "https://openaccess-cdn.clevelandart.org/1960.51/1960.51_print.jpg"
IMAGE = b"\xff\xd8exact museum print bytes\xff\xd9"


def _page(episode_dir: Path, body: dict, url: str = OBJECT_PAGE) -> FetchObservation:
    raw = json.dumps(body).encode("utf-8")
    response_sha = digest_bytes(raw)
    response_ref = f"agent-qa-responses/{response_sha}.bin"
    write_bytes_new(episode_dir / response_ref, raw)
    text = raw.decode("utf-8")
    snapshot = (text + "\n").encode("utf-8")
    snapshot_sha = digest_bytes(snapshot)
    snapshot_ref = f"agent-qa-snapshots/{snapshot_sha}.txt"
    write_bytes_new(episode_dir / snapshot_ref, snapshot)
    return FetchObservation(url, utc_now(), 200, response_sha, response_ref, snapshot_ref, snapshot_sha,
                            text, url, "application/json")


def _asset(**changes) -> dict:
    asset = {"origin": OBJECT_PAGE, "rights_url": OBJECT_PAGE, "official_asset_url": OFFICIAL,
             "source_object_id": "136475", "license": "CC0", "sha256": digest_bytes(IMAGE),
             "rights_basis": "The final visual SHA-256 matches the frozen CMA print for object 136475."}
    asset.update(changes)
    return asset


def _record(**changes) -> dict:
    return {"data": {"id": 136475, "accession_number": "1960.51", "share_license_status": "CC0",
                     "images": {"print": {"url": OFFICIAL}}, **changes}}


class OriginProofTests(unittest.TestCase):
    def _proof(self, asset: dict, body: dict, served: bytes = IMAGE):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        episode_dir = Path(directory.name) / "episode"
        private = Path(directory.name) / "private"
        page = _page(episode_dir, body)
        with patch("trusted_qa.fetch._request_once", return_value=(200, {"content-type": "image/jpeg"}, served)):
            return origin_proof("visual:test", asset, page, page.text, episode_dir, private), episode_dir

    def test_binds_object_page_and_official_exact_bytes(self) -> None:
        proof, episode_dir = self._proof(_asset(), _record())
        record = proof.record()
        self.assertEqual(set(record), {"source_object_id", "rights_basis", "origin_fetch", "exact_file"})
        self.assertEqual(record["source_object_id"], "136475")
        fetch = record["origin_fetch"]
        self.assertEqual((fetch["url"], fetch["final_url"], fetch["http_status"]), (OBJECT_PAGE, OBJECT_PAGE, 200))
        self.assertIn(OFFICIAL, fetch["visible_links"])
        self.assertEqual(record["exact_file"], {"method": "official_download", "sha256": digest_bytes(IMAGE),
                                                "official_asset_url": OFFICIAL,
                                                "response_ref": f"agent-qa-responses/{digest_bytes(IMAGE)}.bin"})
        self.assertEqual((episode_dir / record["exact_file"]["response_ref"]).read_bytes(), IMAGE)

    def test_official_copy_with_other_bytes_holds(self) -> None:
        with self.assertRaisesRegex(QaHold, "official file bytes differ"):
            self._proof(_asset(), _record(), served=b"\xff\xd8a different crop\xff\xd9")

    def test_page_that_does_not_name_the_object_holds(self) -> None:
        with self.assertRaisesRegex(QaHold, "does not identify object"):
            self._proof(_asset(source_object_id="999999"), _record())

    def test_official_file_not_linked_by_the_page_holds(self) -> None:
        with self.assertRaisesRegex(QaHold, "does not link the official file"):
            self._proof(_asset(), _record(images={"print": {"url": "https://example.org/other.jpg"}}))

    def test_rights_page_without_the_licence_holds(self) -> None:
        with self.assertRaisesRegex(QaHold, "rights page does not name the object and its licence"):
            self._proof(_asset(license="CC BY 4.0"), _record())


if __name__ == "__main__":
    unittest.main()
