import io
import unittest
import urllib.error
from unittest import mock

from trusted_qa import common


class _Ok:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        return b"{}"


class GeminiFallbackTests(unittest.TestCase):
    def test_an_overloaded_model_hands_the_request_to_the_next(self):
        seen = []

        def urlopen(request, timeout):
            seen.append(request.full_url)
            if "first-model" in request.full_url:
                raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, io.BytesIO(b"{}"))
            return _Ok()

        with mock.patch.object(common.urllib.request, "urlopen", urlopen), \
                mock.patch.object(common.time, "sleep", lambda seconds: None):
            body, model = common.gemini_post("gemini-first-model,gemini-second-model", b"{}", key="k",
                                             timeout=5, max_bytes=100, failure="test request failed",
                                             not_ok="not ok")
        self.assertEqual((body, model), (b"{}", "gemini-second-model"))
        self.assertEqual(sum("first-model" in url for url in seen), common.GEMINI_ATTEMPTS)

    def test_the_last_overloaded_model_holds(self):
        def urlopen(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, io.BytesIO(b"{}"))

        with mock.patch.object(common.urllib.request, "urlopen", urlopen), \
                mock.patch.object(common.time, "sleep", lambda seconds: None):
            with self.assertRaisesRegex(common.QaHold, "HTTP 503"):
                common.gemini_post("gemini-only-model", b"{}", key="k", timeout=5, max_bytes=100,
                                   failure="test request failed", not_ok="not ok")
