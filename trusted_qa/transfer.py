"""Sign an independently approved QA result and package exact release evidence.

This command belongs in a separate Actions job with only the signing secret. It
never imports producer code, hosts media, or calls a publisher.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import os
import re
import stat
import tarfile
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .common import EPISODE, QA_REPOSITORY, QaHold, digest_file, json_object, require


KIND = "mool_katha_qa_transfer_v1"
SIGNING_CONTEXT = b"mool-katha-agent-release-v1\0"
QA_FILE = re.compile(
    r"(?:agent-release-(?:review|signature)\.json|"
    r"agent-video-frame-audit\.json|agent-audio-quality-observation\.json|"
    r"agent-video-contact\.(?:jpg|jpeg|png|webp)|agent-asr-[a-z0-9_-]+\.json|"
    r"agent-(?:qa-snapshots/[0-9a-f]{64}\.txt|"
    r"qa-responses/[0-9a-f]{64}\.bin|"
    r"video-frames/[0-9a-f]{64}\.jpg|"
    r"video-crops/[0-9a-f]{64}\.png))\Z"
)
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_FILES = 256


def _regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise QaHold(f"{label} is missing") from exc
    require(stat.S_ISREG(mode), f"{label} is not a regular file")


def _json(path: Path, label: str) -> dict:
    _regular(path, label)
    return json_object(path.read_bytes(), label)


def _normalized_script_hash(script: dict) -> str:
    raw = json.dumps(script, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _approved(value: object, label: str) -> None:
    require(isinstance(value, dict) and value.get("decision") == "approved"
            and value.get("unresolved_items") == [], f"{label} is not an unqualified approval")


def _check_review(qa_output: Path, episode_id: str, source_commit: str,
                  workflow_sha: str, run_id: int, attempt: int) -> tuple[Path, dict, str]:
    root = qa_output / "snapshot" / "content" / "episodes" / episode_id
    review_path = root / "agent-release-review.json"
    review = _json(review_path, "independent QA review")
    result = _json(qa_output / "private" / "run-result.json", "QA run result")
    candidate = _json(qa_output / "private" / "candidate-source.json", "QA candidate source")
    require(result.get("status") == "reviewed_unsigned" and result.get("episode_id") == episode_id,
            "QA run did not finish with an approved unsigned result")
    require(candidate.get("episode_id") == episode_id and candidate.get("source_commit") == source_commit,
            "QA candidate came from another source commit")
    require(result.get("review_sha256") == digest_file(review_path),
            "QA run result names different review bytes")
    require(review.get("kind") == "agent_episode_qa_v1" and review.get("episode_id") == episode_id,
            "QA review names another episode or schema")
    run = review.get("qa_run")
    require(isinstance(run, dict) and run.get("repository") == QA_REPOSITORY
            and run.get("workflow_ref") == f"{QA_REPOSITORY}/.github/workflows/release-qa.yml@refs/tags/qa-v20"
            and run.get("workflow_sha") == workflow_sha and run.get("run_id") == run_id
            and run.get("run_attempt") == attempt,
            "QA review did not come from this pinned release workflow run")
    require(isinstance(run.get("frame_batch_review"), dict)
            and run["frame_batch_review"].get("kind") == "frame_indexed_visual_review_v1",
            "QA review lacks indexed all-frame inspection")
    for field, name in (("script_sha256", "script.json"), ("spec_sha256", "short.yaml"),
                        ("video_sha256", f"{episode_id}.mp4"), ("qc_sha256", "qc.json"),
                        ("evidence_sha256", "evidence.json"),
                        ("manifest_sha256", "work/manifest.json")):
        path = root / name
        require(review.get(field) == digest_file(path), f"QA review {field} differs from exact candidate")
    require(result.get("video_sha256") == review["video_sha256"],
            "QA result and review name different final video bytes")
    _approved(review.get("release_review"), "release verdict")
    _approved(review.get("audio_review"), "audio verdict")
    _approved(review.get("video_review"), "video verdict")
    for field in ("claim_findings", "asset_findings"):
        findings = review.get(field)
        require(isinstance(findings, list) and bool(findings), f"QA review has no {field}")
        for index, finding in enumerate(findings):
            _approved(finding, f"{field}[{index}]")
    pending = _json(root / "evidence_pending.json", "candidate pending marker")
    script = _json(root / "script.json", "candidate script")
    normalized = _normalized_script_hash(script)
    require(pending.get("episode_id") == episode_id
            and pending.get("script_sha256") == normalized
            and pending.get("video_sha256") == review["video_sha256"],
            "candidate pending marker differs from signed episode")
    return root, review, normalized


def _support_files(root: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        name = path.relative_to(root).as_posix()
        if not name.startswith("agent-"):
            continue
        require(QA_FILE.fullmatch(name) is not None, f"unexpected QA file {name}")
        _regular(path, name)
        require(0 < path.stat().st_size <= MAX_FILE_BYTES, f"QA file {name} is oversized or empty")
        files[name] = path.read_bytes()
    require("agent-release-review.json" in files
            and "agent-release-signature.json" not in files,
            "QA snapshot has no unsigned review or already contains a signature")
    require("agent-video-frame-audit.json" in files,
            "QA snapshot has no all-frame audit")
    return files


def _tar_member(name: str, data: bytes) -> tuple[tarfile.TarInfo, io.BytesIO]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o600
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info, io.BytesIO(data)


def sign_transfer(qa_output: Path, episode_id: str, source_commit: str,
                  output: Path, *, key_id: str, private_key_b64: str,
                  workflow_sha: str, run_id: int, attempt: int) -> Path:
    """Create a deterministic, hash-listed tar only from one cleared QA run."""
    require(EPISODE.fullmatch(episode_id) is not None, "invalid episode ID")
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None,
            "source commit is not immutable")
    require(re.fullmatch(r"[0-9a-f]{40}", workflow_sha) is not None,
            "release workflow SHA is invalid")
    require(re.fullmatch(r"[A-Za-z0-9._-]{4,64}", key_id) is not None,
            "signing key ID is invalid")
    require(type(run_id) is int and run_id > 0 and type(attempt) is int and attempt > 0,
            "QA run identity is invalid")
    require(not output.exists(), "QA transfer output already exists")
    root, review, normalized_script = _check_review(
        qa_output, episode_id, source_commit, workflow_sha, run_id, attempt)
    files = _support_files(root)
    review_bytes = files["agent-release-review.json"]
    try:
        seed = base64.b64decode(private_key_b64, validate=True)
        require(len(seed) == 32, "signing key must be a 32-byte Ed25519 seed")
        key = Ed25519PrivateKey.from_private_bytes(seed)
    except (ValueError, TypeError, binascii.Error) as exc:
        raise QaHold("signing key is unavailable or malformed") from exc
    signature = base64.b64encode(key.sign(SIGNING_CONTEXT + review_bytes)).decode("ascii")
    record = {"kind": "agent_release_signature_v1", "algorithm": "Ed25519",
              "key_id": key_id, "review_sha256": hashlib.sha256(review_bytes).hexdigest(),
              "signature": signature}
    files["agent-release-signature.json"] = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    require(len(files) <= MAX_FILES and sum(map(len, files.values())) <= MAX_TOTAL_BYTES,
            "QA transfer exceeds the bounded file count or size")
    rows = [{"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            for name, data in sorted(files.items())]
    manifest = {"kind": KIND, "episode_id": episode_id, "source_commit": source_commit,
                "script_sha256": normalized_script, "video_sha256": review["video_sha256"],
                "files": rows}
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    require(len(manifest_bytes) <= 65536, "QA transfer manifest is oversized")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    require(not temporary.exists(), "temporary QA transfer already exists")
    try:
        with tarfile.open(temporary, "x") as bundle:
            for name, data in [("qa-transfer.json", manifest_bytes), *sorted(files.items())]:
                info, stream = _tar_member(name, data)
                bundle.addfile(info, stream)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sign one independently approved QA transfer")
    parser.add_argument("--qa-output", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        path = sign_transfer(
            args.qa_output, args.episode, args.source_commit, args.output,
            key_id=os.environ.get("QA_SIGNING_KEY_ID", ""),
            private_key_b64=os.environ.get("QA_SIGNING_PRIVATE_KEY_B64", ""),
            workflow_sha=os.environ.get("GITHUB_WORKFLOW_SHA", ""),
            run_id=int(os.environ.get("GITHUB_RUN_ID", "0")),
            attempt=int(os.environ.get("GITHUB_RUN_ATTEMPT", "0")),
        )
        print(f"Signed transfer: {path}")
        return 0
    except (QaHold, ValueError) as exc:
        print(f"QA transfer hold: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
