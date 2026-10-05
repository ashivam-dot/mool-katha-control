import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from lite import art, run

ORIGINAL = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Hanuman_Surasa.jpg"


def _jpeg(width=1200, height=1600) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (120, 80, 40)).save(buf, "JPEG")
    return buf.getvalue()


class Response:
    def __init__(self, status, content=b"", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise art.requests.HTTPError(str(self.status_code))


def pic(n):
    return {"url": f"https://example.org/{n}.jpg", "credit": {"source": "Wikimedia Commons", "url": f"https://c/{n}"}}


class ThumbUrl(unittest.TestCase):
    def test_largest_standard_width_below_the_original(self):
        self.assertEqual(art.thumb_url(ORIGINAL, 2500),
                         "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Hanuman_Surasa.jpg/1920px-Hanuman_Surasa.jpg")
        self.assertTrue(art.thumb_url(ORIGINAL, 1920).endswith("/1280px-Hanuman_Surasa.jpg"))

    def test_small_or_foreign_originals_are_kept(self):
        self.assertEqual(art.thumb_url(ORIGINAL, 900), ORIGINAL)
        self.assertEqual(art.thumb_url("https://example.org/x.jpg", 4000), "https://example.org/x.jpg")


class Download(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        sleep = mock.patch.object(art.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

    def test_saves_pictures_and_honours_retry_after(self):
        replies = iter([Response(429, headers={"Retry-After": "2"}), Response(200, _jpeg()),
                        Response(200, _jpeg()), Response(200, _jpeg())])
        with mock.patch.object(art.requests, "get", side_effect=lambda *a, **k: next(replies)):
            got = art.download([pic(1), pic(2), pic(3), 1], self.tmp / "pics")
        self.assertEqual([g["path"] for g in got[:3]], ["pics/pic00.jpg", "pics/pic01.jpg", "pics/pic02.jpg"])
        self.assertEqual(got[3], 1)
        art.time.sleep.assert_any_call(2.0)
        self.assertTrue((self.tmp / "pics" / "pic02.jpg").is_file())

    def test_failed_picture_repeats_the_previous_one(self):
        replies = iter([Response(200, _jpeg()), Response(404), Response(200, _jpeg()), Response(200, _jpeg())])
        with mock.patch.object(art.requests, "get", side_effect=lambda *a, **k: next(replies)):
            got = art.download([pic(1), pic(2), pic(3), pic(4)], self.tmp / "pics")
        self.assertEqual(got[1], 1)
        self.assertEqual(sum(isinstance(g, dict) for g in got), 3)

    def test_too_few_pictures_is_an_error(self):
        with mock.patch.object(art.requests, "get", return_value=Response(404)), self.assertRaises(LookupError):
            art.download([pic(1), pic(2), pic(3)], self.tmp / "pics")

    def test_spec_uses_downloaded_files(self):
        script = {"beats": [{"text": "क"}, {"text": "ख"}], "title": "t", "description": "d", "hook_text": "h",
                  "hashtags": [], "keywords": [], "names": []}
        passage = mock.Mock(citation_hi="वाल्मीकि रामायण · सुंदरकांड · सर्ग 1", url="https://v")
        spec = run.build_spec("lite-x", script, passage, [{**pic(1), "path": "pics/pic00.jpg"}, 1])
        self.assertEqual(spec["beats"][0]["visual"]["source"], "file")
        self.assertEqual(spec["beats"][0]["visual"]["path"], "pics/pic00.jpg")
        self.assertEqual(spec["beats"][1]["visual"]["reuse"], 1)


if __name__ == "__main__":
    unittest.main()
