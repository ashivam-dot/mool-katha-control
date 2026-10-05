import unittest

from trusted_qa.common import QaHold
from trusted_qa.fetch import FetchObservation, text_window

PAGE = '{"data": {"id": 164838, "share_license_status": "CC0", "tombstone": "Durga"}}'


def observation(text=PAGE):
    return FetchObservation(url="https://openaccess-api.clevelandart.org/api/artworks/164838",
                            fetched_at="2026-10-05T10:00:00+00:00", http_status=200, response_sha256="a" * 64,
                            response_ref="r", snapshot_ref="s", snapshot_sha256="b" * 64, text=text,
                            final_url="https://openaccess-api.clevelandart.org/api/artworks/164838",
                            content_type="application/json")


class RightsExcerptTests(unittest.TestCase):
    def test_key_value_rendering_records_the_exact_json_pair(self):
        record = observation().rights_record("share_license_status: CC0")
        self.assertEqual(record["license_excerpt"], '"share_license_status": "CC0"')

    def test_verbatim_excerpt_is_kept(self):
        self.assertEqual(observation().rights_record('"share_license_status": "CC0"')["license_excerpt"],
                         '"share_license_status": "CC0"')

    def test_a_value_the_page_does_not_state_still_holds(self):
        for excerpt in ("share_license_status: Copyrighted", "license: CC0 public domain"):
            with self.assertRaises(QaHold):
                observation().rights_record(excerpt)

    def test_overlapping_rights_windows_merge_instead_of_dropping_the_later_grant(self):
        page = "Gemini API Additional Terms " + "x" * 3000 + " Google won't claim ownership over that content. " + "y" * 9000
        window = text_window(page, ["Gemini API Additional Terms", "ownership"])
        self.assertIn("Google won't claim ownership over that content.", window)
        self.assertIn("Gemini API Additional Terms", window)


if __name__ == "__main__":
    unittest.main()
