"""Private one-off diagnostic for the exact ep022 indexed frame request."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


SOURCE = Path("qa-output/episode/private/frame-batch-001-request.json")
EXPECTED_SHA256 = "fcdc4698af9d027ed94d4874eaf356ec00fd77dca8caea5c368b46d6b887e299"
MAX_RESPONSE_BYTES = 1024 * 1024
EXPECTED_KEYS = {"decision", "uncertainty", "checked_indices", "defect_indices", "notes"}


def post(model: str, body: bytes, key: str) -> tuple[str, bytes | None]:
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return f"http_{response.status}", response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        return f"http_{exc.code}", None
    except (OSError, urllib.error.URLError) as exc:
        return f"transport_{type(exc).__name__}", None


def shape_valid(raw: bytes | None, first: int, last: int) -> tuple[bool, str, int]:
    if raw is None or len(raw) > MAX_RESPONSE_BYTES:
        return False, "invalid", 0
    try:
        provider = json.loads(raw)
        candidates = provider["candidates"]
        if (not isinstance(provider.get("responseId"), str) or
                len(provider["responseId"]) < 4 or
                not isinstance(provider.get("modelVersion"), str) or
                len(provider["modelVersion"]) < 4 or
                len(candidates) != 1 or candidates[0]["finishReason"] != "STOP"):
            return False, "invalid", 0
        parts = candidates[0]["content"]["parts"]
        if len(parts) != 1 or not isinstance(parts[0]["text"], str):
            return False, "invalid", 0
        decision = json.loads(parts[0]["text"])
        checked = decision.get("checked_indices")
        defects = decision.get("defect_indices")
        label = decision.get("decision")
        valid = (set(decision) == EXPECTED_KEYS and
                 checked == list(range(first, last + 1)) and
                 isinstance(defects, list) and
                 label in ("clear", "hold") and
                 decision.get("uncertainty") in ("low", "high") and
                 isinstance(decision.get("notes"), str) and
                 len(decision["notes"].strip()) >= 30)
        return valid, label if valid else "invalid", len(defects) if valid else 0
    except (KeyError, IndexError, TypeError, ValueError):
        return False, "invalid", 0


def small_batch(exact: bytes) -> bytes:
    body = copy.deepcopy(json.loads(exact))
    parts = body["contents"][0]["parts"]
    source = base64.b64decode(parts[1]["inlineData"]["data"], validate=True)
    with Image.open(io.BytesIO(source)) as sheet:
        if sheet.size != (840, 1205):
            raise SystemExit("exact indexed sheet dimensions changed")
        with io.BytesIO() as output:
            sheet.crop((0, 0, 840, 241)).save(output, format="JPEG", quality=85)
            cropped = output.getvalue()
    parts[0]["text"] = (
        "Inspect all seven labeled frame tiles, indexed 1 through 7. "
        "Return exactly one JSON object with decision, uncertainty, checked_indices, "
        "defect_indices and notes. Include every index from 1 through 7 in checked_indices. "
        "Use clear only if every tile is inspectable and has no defect."
    )
    parts[1]["inlineData"]["data"] = base64.b64encode(cropped).decode("ascii")
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def main() -> None:
    exact = SOURCE.read_bytes()
    if hashlib.sha256(exact).hexdigest() != EXPECTED_SHA256:
        raise SystemExit("saved frame request digest mismatch")
    key = os.environ.get("QA_GEMINI_API_KEY", "")
    if not key:
        raise SystemExit("QA model credential is unavailable")

    accepted = False
    for attempt in range(1, 4):
        status, response = post("gemini-3.8-flash", exact, key)
        valid, decision, defect_count = shape_valid(response, 1, 35)
        print(f"exact_38_attempt_{attempt}={status} schema_valid={valid} "
              f"decision={decision} defect_count={defect_count}")
        if status == "http_200" and valid:
            accepted = True
            break
        if status not in ("http_429", "http_500", "http_502", "http_503", "http_504"):
            break
        if attempt < 3:
            time.sleep((3, 9)[attempt - 1])

    smaller = small_batch(exact)
    status, response = post("gemini-3.8-flash", smaller, key)
    valid, decision, defect_count = shape_valid(response, 1, 7)
    print(f"small_38={status} schema_valid={valid} decision={decision} "
          f"defect_count={defect_count} request_bytes={len(smaller)}")

    if not accepted:
        status, response = post("gemini-3.7-flash", exact, key)
        valid, decision, defect_count = shape_valid(response, 1, 35)
        print(f"exact_37={status} schema_valid={valid} decision={decision} "
              f"defect_count={defect_count}")


if __name__ == "__main__":
    main()
