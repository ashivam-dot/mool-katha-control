"""Provider raw-output binding and real-page snapshot receipt contracts."""

from __future__ import annotations

import http.client
import json
import socket
import tempfile
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import unquote

from trusted_qa.asr import gemini_full_audio, whisper_full_audio
from trusted_qa.assemble import _citations, _verify_raw_asr
from trusted_qa.common import QaHold, digest_bytes, digest_file
from trusted_qa.fetch import FetchObservation, _request_once, _safe_url, fetch_observation
from trusted_qa.observations import require_distinct_claim_pages


class _Response:
    status = 200

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, *_):
        return self.data


class ProviderAndFetchTests(unittest.TestCase):
    def test_unicode_source_path_and_query_are_ascii_on_the_http_wire(self) -> None:
        url = ("https://sa.wikisource.org/wiki/रामायणम्/सुन्दरकाण्डम्/सर्गः_१७"
               "?first=%E0%A4%95&next=नल#viewer")
        public_address = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]

        class RecordedConnection:
            target = ""

            def request(self, method: str, target: str, **_kwargs) -> None:
                # This is the stdlib call that failed in the cloud run, without a socket.
                http.client.HTTPConnection("sa.wikisource.org").putrequest(method, target)
                self.target = target

            def getresponse(self):
                return SimpleNamespace(status=200, getheaders=lambda: [], read=lambda _size: b"ok")

            def close(self) -> None:
                pass

        connection = RecordedConnection()
        with patch("trusted_qa.fetch.socket.getaddrinfo", return_value=public_address), \
             patch("trusted_qa.fetch._PinnedHTTPS", return_value=connection):
            self.assertEqual(_request_once(url), (200, {}, b"ok"))
        self.assertEqual(unquote(connection.target),
                         "/wiki/रामायणम्/सुन्दरकाण्डम्/सर्गः_१७?first=क&next=नल")
        self.assertIn("?first=%E0%A4%95&next=%E0%A4%A8%E0%A4%B2", connection.target)
        self.assertTrue(connection.target.isascii())

    def test_https_transport_failure_tries_another_validated_public_address(self) -> None:
        addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))
                     for ip in ("9.9.9.9", "8.8.8.8")]

        class Connection:
            def __init__(self, *, fail: bool) -> None:
                self.fail = fail
                self.closed = False

            def request(self, *_args, **_kwargs) -> None:
                if self.fail:
                    raise ConnectionResetError("test transport failure")

            def getresponse(self):
                return SimpleNamespace(status=200, getheaders=lambda: [], read=lambda _size: b"fresh page")

            def close(self) -> None:
                self.closed = True

        first, second = Connection(fail=True), Connection(fail=False)
        with patch("trusted_qa.fetch.socket.getaddrinfo", return_value=addresses) as resolved, \
             patch("trusted_qa.fetch._PinnedHTTPS", side_effect=[first, second]) as connected, \
             patch("trusted_qa.fetch.time.sleep") as paused:
            self.assertEqual(_request_once("https://example.org/rights"), (200, {}, b"fresh page"))
        self.assertEqual([call.args for call in connected.call_args_list],
                         [("example.org", "9.9.9.9"), ("example.org", "8.8.8.8")])
        self.assertEqual(resolved.call_count, 3)  # initial URL check plus each connection
        paused.assert_called_once()
        self.assertTrue(first.closed and second.closed)

    def test_https_retry_refuses_dns_rebinding_to_a_private_address(self) -> None:
        public = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]
        private = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

        class FailingConnection:
            def request(self, *_args, **_kwargs) -> None:
                raise ConnectionResetError("test transport failure")

            def close(self) -> None:
                pass

        with patch("trusted_qa.fetch.socket.getaddrinfo", side_effect=[public, public, private]), \
             patch("trusted_qa.fetch._PinnedHTTPS", return_value=FailingConnection()) as connected, \
             patch("trusted_qa.fetch.time.sleep"):
            with self.assertRaisesRegex(QaHold, "private address"):
                _request_once("https://example.org/rights")
        connected.assert_called_once_with("example.org", "8.8.8.8")

    def test_https_transport_retries_are_bounded_without_a_cached_page(self) -> None:
        public = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]

        class FailingConnection:
            def request(self, *_args, **_kwargs) -> None:
                raise ConnectionResetError("test transport failure")

            def close(self) -> None:
                pass

        with patch("trusted_qa.fetch.socket.getaddrinfo", return_value=public) as resolved, \
             patch("trusted_qa.fetch._PinnedHTTPS", side_effect=lambda *_: FailingConnection()) as connected, \
             patch("trusted_qa.fetch.time.sleep") as paused:
            with self.assertRaisesRegex(QaHold, "source HTTPS fetch failed"):
                _request_once("https://example.org/rights")
        self.assertEqual(connected.call_count, 4)
        self.assertEqual(resolved.call_count, 5)  # every retry rechecks public DNS
        self.assertEqual(paused.call_count, 3)

    def test_percent_encoded_and_unicode_final_urls_compare_as_one_source(self) -> None:
        primary = FetchObservation(
            "https://sa.wikisource.org/wiki/राम?next=नल", "2026-10-04T00:00:00Z", 200,
            "a" * 64, "primary.bin", "primary.txt", "b" * 64, "primary text",
            "https://sa.wikisource.org/wiki/राम?next=नल", "text/plain")
        corroboration = replace(
            primary, url="https://other.example/independent", snapshot_sha256="c" * 64,
            final_url=("https://sa.wikisource.org:443/wiki/%e0%a4%b0%e0%a4%be%e0%a4%ae"
                       "?next=%E0%A4%A8%E0%A4%B2"))
        with self.assertRaisesRegex(QaHold, "same final page"):
            require_distinct_claim_pages("claim-1", {"primary": primary,
                                                      "corroboration": corroboration})

    def test_redirect_to_private_host_holds_before_second_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("trusted_qa.fetch._request_once", return_value=(
                302, {"location": "https://127.0.0.1/private"}, b"")) as requested:
                with self.assertRaisesRegex(QaHold, "private or reserved"):
                    fetch_observation("https://example.org/verse", root / "episode", root / "audit")
            requested.assert_called_once()
            self.assertFalse((root / "episode").exists())

    def test_fetch_receipt_uses_actual_bytes_and_blocks_private_url(self) -> None:
        with self.assertRaises(QaHold):
            _safe_url("https://127.0.0.1/private")
        with self.assertRaisesRegex(QaHold, "malformed percent escape"):
            _safe_url("https://example.org/verse%not-an-escape")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = "1.1 राम ने नल से कहा। The printed passage independently supports the claim."
            raw = text.encode("utf-8")
            with patch("trusted_qa.fetch._request_once", return_value=(200, {"content-type": "text/plain"}, raw)):
                page = fetch_observation("https://example.org/verse", root / "episode", root / "audit")
            self.assertEqual(page.response_sha256, digest_bytes(raw))
            self.assertEqual(digest_file(root / "episode" / page.response_ref), page.response_sha256)
            self.assertEqual(digest_file(root / "episode" / page.snapshot_ref), page.snapshot_sha256)
            receipt = page.source_record("राम ने नल से कहा। The printed passage", "1.1")
            self.assertEqual(receipt["url"], "https://example.org/verse")
            self.assertEqual(receipt["response_ref"], page.response_ref)
            with self.assertRaises(QaHold):
                page.source_record("invented passage", "1.1")

    def test_gemini_full_wav_keeps_provider_response_and_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "whole.wav"
            with wave.open(str(wav), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\x00\x00" * 32000)
            transcription = {"segments": [{"start_seconds": 0.2, "end_seconds": 1.9,
                                            "text": "राम नल से कहते हैं।"}],
                             "full_transcript": "राम नल से कहते हैं।"}
            raw = {"responseId": "provider-request-123", "modelVersion": "gemini-2.5-flash",
                   "candidates": [{"finishReason": "STOP", "content": {"parts": [
                       {"text": json.dumps(transcription, ensure_ascii=False)}]}}]}
            with patch("urllib.request.urlopen", return_value=_Response(json.dumps(raw).encode())) as opened:
                result = gemini_full_audio(wav, 2.0, "a" * 64, key="test-only", model="gemini-2.5-flash")
            self.assertEqual(result["run_id"], "provider-request-123")
            self.assertEqual(result["raw_response"], raw)
            _verify_raw_asr(result, "gemini")
            request = json.loads(opened.call_args.args[0].data)
            self.assertEqual(request["contents"][0]["parts"][1]["inlineData"]["mimeType"], "audio/wav")
            result["segments"][0]["text"] = "changed by QA"
            with self.assertRaisesRegex(QaHold, "edited after recognition"):
                _verify_raw_asr(result, "gemini")

    def test_local_whisper_provenance_and_citations_bind_raw_segments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "model.bin").write_bytes(b"pinned CTranslate2 model fixture")
            for filename in ("config.json", "tokenizer.json", "vocabulary.json"):
                (model_dir / filename).write_bytes(b"{}")
            audio = root / "whole.wav"
            audio.write_bytes(b"whole audio fixture")
            segment = SimpleNamespace(id=0, start=0.2, end=1.9, text="राम नल से कहते हैं।",
                                      avg_logprob=-0.1, no_speech_prob=0.01, compression_ratio=1.0)
            info = SimpleNamespace(language="hi", language_probability=1.0,
                                   duration=2.0, duration_after_vad=2.0)

            class FakeModel:
                def __init__(self, *_args, **_kwargs):
                    pass

                def transcribe(self, *_args, **kwargs):
                    self_test.assertIs(kwargs["vad_filter"], False)
                    return iter([segment]), info

            self_test = self
            with patch("faster_whisper.WhisperModel", FakeModel):
                result = whisper_full_audio(audio, 2.0, "a" * 64, model_dir=model_dir,
                                            model_repo="owner/pinned-model", revision="0" * 40,
                                            model_sha256=digest_file(model_dir / "model.bin"),
                                            github_run_id=123, github_run_attempt=1)
            self.assertIn(digest_file(model_dir / "model.bin"), result["model_version"])
            with self.assertRaisesRegex(QaHold, "pinned revision artifact"):
                whisper_full_audio(audio, 2.0, "a" * 64, model_dir=model_dir,
                                   model_repo="owner/pinned-model", revision="0" * 40,
                                   model_sha256="0" * 64,
                                   github_run_id=123, github_run_attempt=1)
            self.assertEqual(result["run_id"], "local-whisper-gha-123-1-aaaaaaaaaaaaaaaa")
            _verify_raw_asr(result, "whisper")
            asr = {"agent-asr-gemini.json": result, "agent-asr-whisper.json": result}
            cited = [{"asr_file": name, "segment_indices": [1],
                      "asr_excerpt": "राम नल से कहते हैं।"} for name in asr]
            self.assertEqual(len(_citations(cited, asr, "beat", critical_terms=["नल"])), 2)
            with self.assertRaisesRegex(QaHold, "critical term"):
                _citations(cited, asr, "beat", critical_terms=["विश्वकर्मा"])

    def test_beat_citations_are_contiguous_and_each_segment_overlaps(self) -> None:
        segments = [
            {"index": 1, "start_seconds": 0.0, "end_seconds": 1.0, "text": "राम"},
            {"index": 2, "start_seconds": 1.0, "end_seconds": 2.0, "text": "नल"},
            {"index": 3, "start_seconds": 3.0, "end_seconds": 4.0, "text": "कहते"},
        ]
        asr = {name: {"segments": segments} for name in
               ("agent-asr-gemini.json", "agent-asr-whisper.json")}

        def citations(indices: list[int], excerpt: str) -> list[dict]:
            return [{"asr_file": name, "segment_indices": indices, "asr_excerpt": excerpt}
                    for name in asr]

        self.assertEqual(len(_citations(citations([1, 2], "राम नल"), asr, "beat",
                                        critical_terms=["राम", "नल"], beat_span=(0.0, 2.0))), 2)
        with self.assertRaisesRegex(QaHold, "ascending and contiguous"):
            _citations(citations([1, 3], "राम कहते"), asr, "beat", beat_span=(0.0, 2.0))
        with self.assertRaisesRegex(QaHold, "do not overlap"):
            _citations(citations([2, 3], "नल कहते"), asr, "beat", beat_span=(0.0, 2.0))


if __name__ == "__main__":
    unittest.main()
