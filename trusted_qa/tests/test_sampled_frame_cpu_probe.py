"""Fail-closed contracts for the private exact ep022 CPU capacity experiment."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/qa_sampled_frame_cpu_probe.py"
SPEC = importlib.util.spec_from_file_location("ep022_cpu_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class FakeResponse:
    def __init__(self, body: bytes, declared: int | None = None) -> None:
        self.body = io.BytesIO(body)
        self.headers = {"Content-Length": str(declared)} if declared is not None else {}

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        self.body.close()

    def read(self, size: int) -> bytes:
        return self.body.read(size)


def answer(codes: str) -> dict:
    return {"decision": "clear", "uncertainty": "low", "scene_codes": codes,
            "issue_positions": [], "uncertain_positions": [],
            "notes": "The sheet changes between distinct large artwork and dark title cards."}


def cli_output(value: dict) -> bytes:
    return ("main: loading model: /tmp/pinned.gguf\nencoding image slice...\n\n" +
            json.dumps(value) + "\n\n").encode()


class PinnedDownloadTests(unittest.TestCase):
    def test_exact_size_and_sha256_are_required(self) -> None:
        body = b"public-model-file" * 100
        digest = hashlib.sha256(body).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "weights.bin"
            with patch.object(probe.urllib.request, "urlopen",
                              return_value=FakeResponse(body, len(body))):
                probe._download("https://example.test/weights", target, len(body), digest,
                                time.monotonic() + 10)
            self.assertEqual(target.read_bytes(), body)
            with patch.object(probe.urllib.request, "urlopen",
                              return_value=FakeResponse(body, len(body))):
                with self.assertRaisesRegex(probe.ProbeHold, "sha256_pin_mismatch"):
                    probe._download("https://example.test/weights", Path(temporary) / "bad.bin",
                                    len(body), "0" * 64, time.monotonic() + 10)

    def test_length_oversize_and_deadline_hold(self) -> None:
        body = b"bigger-than-pin"
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(probe.urllib.request, "urlopen",
                              return_value=FakeResponse(body, len(body))):
                with self.assertRaisesRegex(probe.ProbeHold, "length_pin_mismatch"):
                    probe._download("https://example.test/weights", Path(temporary) / "length",
                                    len(body) - 1, "0" * 64, time.monotonic() + 10)
            with patch.object(probe.urllib.request, "urlopen",
                              return_value=FakeResponse(body)):
                with self.assertRaisesRegex(probe.ProbeHold, "oversized"):
                    probe._download("https://example.test/weights", Path(temporary) / "oversize",
                                    len(body) - 1, "0" * 64, time.monotonic() + 10)
            with patch.object(probe.urllib.request, "urlopen") as opened:
                with self.assertRaisesRegex(probe.ProbeHold, "download_deadline"):
                    probe._download("https://example.test/weights", Path(temporary) / "deadline",
                                    len(body), "0" * 64, time.monotonic() - 1)
                opened.assert_not_called()


class ModelResponseTests(unittest.TestCase):
    def test_one_exact_response_with_a_matching_visual_sequence(self) -> None:
        reference = probe.SCENE_REFERENCES[0]
        parsed = probe._parse_output(cli_output(answer(reference)), 35)
        mismatch, clear = probe._assess_answer(parsed, reference)
        self.assertEqual(mismatch, [])
        self.assertTrue(clear)

    def test_copied_coverage_without_correct_image_codes_holds(self) -> None:
        reference = probe.SCENE_REFERENCES[0]
        wrong = "A" * len(reference)
        parsed = probe._parse_output(cli_output(answer(wrong)), 35)
        mismatch, clear = probe._assess_answer(parsed, reference)
        self.assertEqual(len(mismatch), 27)
        self.assertFalse(clear)

    def test_uncertainty_and_defects_hold_even_when_codes_match(self) -> None:
        reference = probe.SCENE_REFERENCES[1]
        for field, value in (("uncertainty", "high"), ("decision", "hold"),
                             ("issue_positions", [1]), ("uncertain_positions", [28])):
            with self.subTest(field=field):
                response = answer(reference)
                response[field] = value
                parsed = probe._parse_output(cli_output(response), 28)
                self.assertFalse(probe._assess_answer(parsed, reference)[1])

    def test_malformed_duplicate_extra_and_skipped_tiles_hold(self) -> None:
        response = answer(probe.SCENE_REFERENCES[0])
        valid = cli_output(response)
        malformed = [valid[:-3], valid + b'{"second":true}',
                     cli_output({**response, "unrequested": 1}),
                     cli_output({**response, "scene_codes": response["scene_codes"][:-1]}),
                     cli_output({**response, "issue_positions": [True]})]
        duplicate = valid.replace(b'"decision": "clear"',
                                  b'"decision": "clear", "decision": "hold"', 1)
        malformed.append(duplicate)
        for raw in malformed:
            with self.subTest(raw=raw[-70:]):
                with self.assertRaises(probe.ProbeHold):
                    probe._parse_output(raw, 35)


class ProcessBoundaryTests(unittest.TestCase):
    def test_nonzero_model_process_cannot_become_a_matching_sheet(self) -> None:
        reference = probe.SCENE_REFERENCES[0]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            binary = folder / "fake-cli"
            binary.write_text("#!/usr/bin/env python3\nimport sys\nprint(" +
                              repr(json.dumps(answer(reference))) + ")\nsys.exit(3)\n")
            binary.chmod(0o755)
            with patch.object(probe, "_rss_bytes", return_value=1024), patch.object(
                    probe.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=1024)):
                result = probe._run_sheet(binary, folder / "model", folder / "mmproj",
                                          folder / "sheet.jpg", 1, 35, reference, folder)
        self.assertEqual(result["status"], "held")
        self.assertEqual(result["reason"], "model_process_nonzero_exit")
        self.assertEqual(result["exit_code"], 3)


class ExactSourceAndReceiptTests(unittest.TestCase):
    def test_sheet_selection_cannot_skip_or_reorder_an_index(self) -> None:
        indices = list(range(1, 64))
        batches = [(b"sheet-one", [{"index": index} for index in indices[:35]]),
                   (b"sheet-two", [{"index": index} for index in indices[35:]])]
        probe._validate_sheets(batches, indices)
        batches[1][1][0] = {"index": 999}
        with self.assertRaisesRegex(probe.ProbeHold, "indices_changed"):
            probe._validate_sheets(batches, indices)

    def test_failed_model_calls_keep_the_final_capacity_receipt_held(self) -> None:
        indices = list(range(1, 64))
        batches = [(b"sheet-one", [{"index": index} for index in indices[:35]]),
                   (b"sheet-two", [{"index": index} for index in indices[35:]])]
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = Path(temporary) / "receipt.json"
            with patch.object(probe, "RECEIPT", receipt_path), patch.object(
                    probe, "_source_sheets", return_value=(batches, indices)), patch.object(
                    probe, "_download"), patch.object(probe, "_pinned_binary",
                    return_value=Path("/bin/true")), patch.object(
                    probe, "_run_sheet", return_value={"status": "held", "reason": "test_model_hold"}), patch.object(
                    probe.shutil, "disk_usage", return_value=SimpleNamespace(free=6_000_000_000)):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        probe.main()
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt["capacity_result"], "held")
            self.assertEqual(receipt["release_decision"], "held")
            self.assertFalse(receipt["editorial_approval"])
            self.assertEqual(len(receipt["calls"]), 2)
            self.assertEqual(receipt["reason"], "one_or_more_sheets_held")

    def test_source_failure_leaves_persisted_held_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = Path(temporary) / "receipt.json"
            with patch.object(probe, "RECEIPT", receipt_path), patch.object(
                    probe, "_source_sheets", side_effect=probe.ProbeHold("test_source_hold")):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        probe.main()
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt["capacity_result"], "held")
            self.assertEqual(receipt["release_decision"], "held")
            self.assertFalse(receipt["editorial_approval"])
            self.assertEqual(receipt["reason"], "test_source_hold")


if __name__ == "__main__":
    unittest.main()
