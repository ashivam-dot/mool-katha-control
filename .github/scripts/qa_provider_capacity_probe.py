"""Report allowlisted Gemini quota/status fields without provider body or key."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


SOURCE = Path("qa-output/episode/private/frame-batch-001-request.json")
EXPECTED_SHA256 = "fcdc4698af9d027ed94d4874eaf356ec00fd77dca8caea5c368b46d6b887e299"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
KNOWN_ERROR_STATUS = {
    "RESOURCE_EXHAUSTED", "UNAVAILABLE", "NOT_FOUND", "INVALID_ARGUMENT",
    "PERMISSION_DENIED", "FAILED_PRECONDITION", "INTERNAL", "DEADLINE_EXCEEDED",
}


def body(parts: list[dict]) -> bytes:
    return json.dumps({
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 32},
    }, separators=(",", ":")).encode("utf-8")


def image_part() -> dict:
    exact = SOURCE.read_bytes()
    if hashlib.sha256(exact).hexdigest() != EXPECTED_SHA256:
        raise SystemExit("saved frame request digest mismatch")
    original = json.loads(exact)
    sheet_bytes = base64.b64decode(
        original["contents"][0]["parts"][1]["inlineData"]["data"], validate=True)
    with Image.open(io.BytesIO(sheet_bytes)) as sheet:
        if sheet.size != (840, 1205):
            raise SystemExit("exact indexed sheet dimensions changed")
        with io.BytesIO() as output:
            sheet.crop((0, 27, 120, 241)).save(output, format="JPEG", quality=85)
            tile = output.getvalue()
    return {"inlineData": {"mimeType": "image/jpeg",
                           "data": base64.b64encode(tile).decode("ascii")}}


def sanitized_error(exc: urllib.error.HTTPError) -> str:
    raw = exc.read(65537)
    status = "unknown"
    quota = False
    retry = False
    if len(raw) <= 65536:
        try:
            error = json.loads(raw).get("error", {})
            value = error.get("status")
            if value in KNOWN_ERROR_STATUS:
                status = value
            details = error.get("details", [])
            if isinstance(details, list):
                kinds = {part["@type"] for part in details if isinstance(part, dict)
                         and isinstance(part.get("@type"), str)}
                quota = any(kind.endswith("google.rpc.QuotaFailure") for kind in kinds)
                retry = any(kind.endswith("google.rpc.RetryInfo") for kind in kinds)
        except (AttributeError, TypeError, ValueError):
            pass
    retry_header = bool(exc.headers.get("Retry-After"))
    return (f"http_{exc.code} provider_status={status} "
            f"quota_failure={quota} retry_info={retry} retry_header={retry_header}")


def probe(label: str, content: bytes, key: str) -> None:
    request = urllib.request.Request(
        ENDPOINT, data=content,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read(65537)
            status = response.status
    except urllib.error.HTTPError as exc:
        print(f"{label}={sanitized_error(exc)}")
        return
    except (OSError, urllib.error.URLError) as exc:
        print(f"{label}=transport_{type(exc).__name__}")
        return
    candidate = False
    if status == 200 and len(raw) <= 65536:
        try:
            candidate = bool(json.loads(raw).get("candidates"))
        except (AttributeError, TypeError, ValueError):
            pass
    print(f"{label}=http_{status} has_candidate={candidate}")


def main() -> None:
    key = os.environ.get("QA_GEMINI_API_KEY", "")
    if not key:
        raise SystemExit("QA model credential is unavailable")
    text = body([{"text": "Reply with only OK."}])
    image = body([{"text": "Describe this image in one word."}, image_part()])
    probe("text_38", text, key)
    probe("one_tile_38", image, key)


if __name__ == "__main__":
    main()
