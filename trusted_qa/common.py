"""Small, strict parsing and byte-binding helpers shared by trusted QA code."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
EPISODE = re.compile(r"ep[0-9]{3}\Z")
AGENT = re.compile(r"agent:[A-Za-z0-9._/-]{4,}\Z")
PRODUCTION_REPOSITORY = "ashivam-dot/mool-katha"
QA_REPOSITORY = "ashivam-dot/mool-katha-control"


def valid_qa_workflow_ref(repository: str, workflow_ref: str) -> bool:
    """Allow a shadow branch run or the pinned release tag in the control repo."""
    if repository != QA_REPOSITORY or not isinstance(workflow_ref, str):
        return False
    pattern = (re.escape(QA_REPOSITORY) +
               r"/\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml@refs/(?:heads/[A-Za-z0-9._/-]+|tags/qa-v11)\Z")
    return re.fullmatch(pattern, workflow_ref) is not None


class QaHold(ValueError):
    """The exact candidate or independent review cannot be cleared."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QaHold(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label}: missing timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise QaHold(f"{label}: invalid ISO timestamp") from exc
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
            f"{label}: timestamp needs a timezone")
    return value


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode), f"{path.name}: expected a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expect_sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256.fullmatch(value) is not None,
            f"{label}: expected lowercase SHA-256")
    return value


def relative_path(name: Any, label: str) -> PurePosixPath:
    require(isinstance(name, str) and bool(name) and "\\" not in name and "\x00" not in name,
            f"{label}: unsafe path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and all(part not in ("", ".", "..") for part in name.split("/"))
            and path.as_posix() == name, f"{label}: unsafe path")
    return path


def path_under(root: Path, name: Any, label: str) -> Path:
    relative = relative_path(name, label)
    path = root.joinpath(*relative.parts)
    require(path.resolve().is_relative_to(root.resolve()), f"{label}: escapes snapshot")
    return path


def unique_json(raw: bytes | str, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise QaHold(f"{label}: duplicate JSON key {key}")
            value[key] = item
        return value

    def bad_constant(value: str) -> None:
        raise QaHold(f"{label}: nonfinite JSON number {value}")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QaHold(f"{label}: invalid UTF-8 JSON") from exc


def json_object(raw: bytes | str, label: str) -> dict[str, Any]:
    value = unique_json(raw, label)
    require(isinstance(value, dict), f"{label}: expected JSON object")
    return value


def gemini_text_response(provider: dict[str, Any], label: str) -> tuple[str, str, str]:
    """Extract one complete provider text without trusting nested response shapes."""
    response_id = provider.get("responseId")
    model_version = provider.get("modelVersion")
    candidates = provider.get("candidates")
    require(isinstance(response_id, str) and len(response_id) >= 4 and
            isinstance(model_version, str) and len(model_version) >= 4 and
            isinstance(candidates, list) and len(candidates) == 1 and
            isinstance(candidates[0], dict) and candidates[0].get("finishReason") == "STOP",
            f"{label}: partial or untracked provider output")
    content = candidates[0].get("content")
    require(isinstance(content, dict), f"{label}: provider content is missing")
    parts = content.get("parts")
    require(isinstance(parts, list) and len(parts) == 1 and isinstance(parts[0], dict) and
            isinstance(parts[0].get("text"), str) and bool(parts[0]["text"].strip()),
            f"{label}: expected one text response")
    return response_id, model_version, parts[0]["text"]


# Free-tier Gemini answers a burst of QA calls with 429 and a suggested retry delay.
GEMINI_ATTEMPTS = 5
GEMINI_MAX_RETRY_SECONDS = 70.0
GEMINI_RETRY_CODES = frozenset({429, 500, 502, 503, 504})


def _retry_delay(exc: urllib.error.HTTPError) -> float | None:
    """Seconds to wait before retrying, or None when a daily quota is spent and waiting can't help."""
    try:
        details = json.loads(exc.read(64 * 1024).decode("utf-8", "replace"))["error"].get("details", [])
        quotas = [v.get("quotaId", "") for d in details if isinstance(d, dict)
                  for v in d.get("violations", []) if isinstance(v, dict)]
        if any("PerDay" in str(quota) for quota in quotas):
            return None
        delay = next(d["retryDelay"] for d in details if isinstance(d, dict) and "retryDelay" in d)
        return min(GEMINI_MAX_RETRY_SECONDS, max(1.0, float(str(delay).rstrip("s"))))
    except (ValueError, KeyError, TypeError, StopIteration, AttributeError, OSError):
        return 30.0


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODELS = re.compile(r"gemini-[A-Za-z0-9._-]+(?:,gemini-[A-Za-z0-9._-]+)*\Z")


def valid_gemini_models(value: Any) -> bool:
    """One explicitly named model, or a comma-separated preference list of them."""
    return isinstance(value, str) and GEMINI_MODELS.fullmatch(value) is not None


class _NextModel(Exception):
    pass


def _post_once(request: urllib.request.Request, *, timeout: float, max_bytes: int,
               failure: str, not_ok: str) -> bytes:
    for attempt in range(1, GEMINI_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                require(response.status == 200, not_ok)
                return response.read(max_bytes + 1)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise _NextModel(f"{failure} (HTTP 404, model unavailable)") from exc
            if exc.code not in GEMINI_RETRY_CODES or attempt == GEMINI_ATTEMPTS:
                raise QaHold(f"{failure} (HTTP {exc.code})") from exc
            delay = _retry_delay(exc)
            if delay is None:
                raise _NextModel(f"{failure} (HTTP {exc.code}, daily quota spent)") from exc
            time.sleep(delay)
        except (OSError, urllib.error.URLError) as exc:
            raise QaHold(failure) from exc
    raise QaHold(failure)


def gemini_post(models: str, body: bytes, *, key: str, timeout: float, max_bytes: int,
                failure: str, not_ok: str) -> tuple[bytes, str]:
    """POST to the first listed Gemini model that can answer; return the response and that model.

    Rate limits and transient errors are waited out on the same model. A model whose daily quota is
    spent, or that no longer exists, passes the request to the next one; the last one's failure holds."""
    require(valid_gemini_models(models), "Gemini model list is not explicitly named")
    names = models.split(",")
    for index, model in enumerate(names):
        request = urllib.request.Request(f"{GEMINI_ENDPOINT}/{model}:generateContent", data=body,
                                         headers={"Content-Type": "application/json", "x-goog-api-key": key},
                                         method="POST")
        try:
            return _post_once(request, timeout=timeout, max_bytes=max_bytes, failure=failure, not_ok=not_ok), model
        except _NextModel as exc:
            if index == len(names) - 1:
                raise QaHold(str(exc)) from exc.__cause__
    raise QaHold(failure)


def write_bytes_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_json_new(path: Path, data: Any) -> None:
    write_bytes_new(path, (json.dumps(data, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
