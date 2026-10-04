"""Small, strict parsing and byte-binding helpers shared by trusted QA code."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
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
               r"/\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml@refs/(?:heads/[A-Za-z0-9._/-]+|tags/qa-v3)\Z")
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
