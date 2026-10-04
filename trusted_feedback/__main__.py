"""Park a draft only when pinned QA recorded a specific, reproducible content hold.

The workflow_run inspector reads untrusted artifact bytes without a source write
credential. The writer receives only validated identifiers and a fixed reason
code, then rechecks the current producer tree before committing a lock.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


EPISODE = re.compile(r"ep[0-9]{3}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
POSITIVE_INT = re.compile(r"[1-9][0-9]*\Z")
IDENTITY = r"[A-Za-z0-9_:-]+"
REASON_MESSAGES = {
    "duplicate_source": "Independent QA found duplicate primary and corroborating source pages",
    "source_passage": "Independent QA could not find the cited passage on its source page",
    "rights_evidence": "Independent QA found missing or unclear asset rights evidence",
    "frame_defect": "Independent QA found a visual defect or uncertain frame review",
    "voice_quality": "Independent QA found a voice-quality concern or material uncertainty",
    "review_verdict": "Independent QA could not approve the exact content review",
}
RELEASE_STATE = ("agent-release-review.json", "agent-release-signature.json",
                 "final-review.json", "hold.json", "publish.json", "remote.json")


class FeedbackError(RuntimeError):
    """A feedback artifact or destination failed a trust check."""


def _object(path: Path, *, max_bytes: int = 16_384) -> dict:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > max_bytes:
        raise FeedbackError("expected bounded feedback record is unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FeedbackError("feedback record is unreadable") from exc
    if type(value) is not dict:
        raise FeedbackError("feedback record is not an object")
    return value


def candidate(path: Path) -> dict[str, str] | None:
    """Trust one exact discovery key, never a filename or model-generated text."""
    value = _object(path)
    if set(value) != {"status", "stage", "reason", "source_commit", "candidates"} or \
            value["stage"] != "discovery":
        raise FeedbackError("discovery artifact has an unexpected schema")
    if value["status"] == "hold" and value["candidates"] == []:
        return None
    rows = value["candidates"]
    if (value["status"] != "pending" or value["reason"] is not None or
            type(rows) is not list or len(rows) != 1 or type(rows[0]) is not dict or
            set(rows[0]) != {"episode_id", "video_sha256", "source_commit"}):
        raise FeedbackError("discovery artifact does not name one pending candidate")
    row = rows[0]
    if (type(row["episode_id"]) is not str or not EPISODE.fullmatch(row["episode_id"]) or
            type(row["video_sha256"]) is not str or not SHA256.fullmatch(row["video_sha256"]) or
            type(row["source_commit"]) is not str or not COMMIT.fullmatch(row["source_commit"]) or
            value["source_commit"] != row["source_commit"]):
        raise FeedbackError("discovery candidate identity is malformed")
    return row


def _hold_reason(stage: str, reason: str) -> str | None:
    """Leave infrastructure, provider, decoder, and unknown failures retryable."""
    if stage == "source_and_rights":
        if re.fullmatch(
            rf"claim {IDENTITY}: primary and corroboration "
            r"(?:resolve to the same final page|have identical visible text)", reason,
        ):
            return "duplicate_source"
        if re.fullmatch(
            rf"claim {IDENTITY} (?:primary|corroboration) "
            r"(?:ledger passage|printed label) is absent from the fetched page", reason,
        ):
            return "source_passage"
        if re.fullmatch(
            rf"asset {IDENTITY} (?:has no inspectable rights evidence|"
            r"internal provenance explanation is too short)", reason,
        ):
            return "rights_evidence"
    if stage == "frame_batch_review":
        if (reason == "frame batch model found a defect, uncertainty, or skipped frame" or
                re.fullmatch(r"frame [1-9][0-9]* is flat, corrupt, or outside pixel audit bounds", reason) or
                re.fullmatch(r"frames [1-9][0-9]*-[1-9][0-9]* are identical for over two seconds", reason)):
            return "frame_defect"
    if stage in ("full_audio_quality", "strict_review_validation") and \
            reason == "full-audio model found a voice-quality concern or material uncertainty":
        return "voice_quality"
    if stage == "strict_review_validation":
        if reason == "sampled visual review found critical defects":
            return "frame_defect"
        if re.fullmatch(rf"asset {IDENTITY}: rights excerpt is unclear or incompatible", reason):
            return "rights_evidence"
        if re.fullmatch(
            rf"(?:claim {IDENTITY}|asset {IDENTITY}|audio review|audio beat [1-9][0-9]*|"
            r"speech difference [1-9][0-9]*|video review|QC warning [1-9][0-9]*|release review): "
            r"(?:unresolved or unreasoned review must hold|[a-z_]+ was not independently completed|"
            r"[a-z_]+ requires a new take or hold)", reason,
        ):
            return "review_verdict"
    return None


def inspect_hold(path: Path, expected: dict[str, str]) -> str | None:
    """An absent or operational hold never writes to the producer repository."""
    if not path.is_file():
        return None
    value = _object(path)
    if set(value) != {"status", "stage", "reason", "episode_id", "source_commit",
                      "video_sha256", "held_at"}:
        raise FeedbackError("QA hold artifact has an unexpected schema")
    if (value["status"] != "hold" or value["episode_id"] != expected["episode_id"] or
            value["source_commit"] != expected["source_commit"] or
            value["video_sha256"] != expected["video_sha256"] or
            type(value["stage"]) is not str or type(value["reason"]) is not str or
            len(value["reason"]) > 500 or type(value["held_at"]) is not str):
        raise FeedbackError("QA hold artifact differs from the discovered candidate")
    return _hold_reason(value["stage"], value["reason"])


def _output(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as target:
        for key, value in values.items():
            if not re.fullmatch(r"[a-z_]+", key) or "\n" in value or "\r" in value:
                raise FeedbackError("feedback output is unsafe")
            target.write(f"{key}={value}\n")


def _git(repo: Path, *args: str, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False,
                            timeout=45)
    if result.returncode and not allow_failure:
        raise FeedbackError("producer Git operation failed")
    return result


def _current_candidate(repo: Path, episode: str, video_sha: str, source_commit: str) -> tuple[str, Path] | None:
    """Require byte-identical tracked episode data at current main before parking it."""
    folder = repo / "content" / "episodes" / episode
    if any((folder / name).exists() for name in RELEASE_STATE):
        return None
    if (folder / "editorial-lock.json").exists():
        return None
    if _git(repo, "merge-base", "--is-ancestor", source_commit, "HEAD", allow_failure=True).returncode:
        return None
    tree_path = f"content/episodes/{episode}"
    old = _git(repo, "rev-parse", f"{source_commit}:{tree_path}", allow_failure=True)
    current = _git(repo, "rev-parse", f"HEAD:{tree_path}", allow_failure=True)
    if old.returncode or current.returncode or old.stdout.strip() != current.stdout.strip():
        return None
    pending = _object(folder / "evidence_pending.json", max_bytes=100_000)
    manifest = _object(folder / "work" / "manifest.json", max_bytes=100_000)
    if (pending.get("episode_id") != episode or pending.get("video_sha256") != video_sha or
            manifest.get("id") != episode or manifest.get("video_sha256") != video_sha):
        return None
    return old.stdout.strip(), folder


def apply_lock(repo: Path, episode: str, video_sha: str, source_commit: str,
               reason_code: str, run_id: str, run_attempt: str) -> str:
    """Commit one reversible editorial hold, retrying only concurrent main advances."""
    if (not EPISODE.fullmatch(episode) or not SHA256.fullmatch(video_sha) or
            not COMMIT.fullmatch(source_commit) or reason_code not in REASON_MESSAGES or
            not POSITIVE_INT.fullmatch(run_id) or not POSITIVE_INT.fullmatch(run_attempt)):
        raise FeedbackError("feedback arguments are invalid")
    if _git(repo, "branch", "--show-current").stdout.strip() != "main":
        raise FeedbackError("producer checkout is not on main")
    if _git(repo, "status", "--porcelain").stdout.strip():
        raise FeedbackError("producer checkout is not clean")
    for attempt in range(3):
        _git(repo, "fetch", "origin", "main")
        _git(repo, "reset", "--hard", "origin/main")
        current = _current_candidate(repo, episode, video_sha, source_commit)
        if current is None:
            return "stale_or_already_held"
        _, folder = current
        lock = folder / "editorial-lock.json"
        record = {"schema": "mool_katha_qa_editorial_hold_v1", "episode_id": episode,
                  "video_sha256": video_sha, "source_commit": source_commit,
                  "qa_run_id": int(run_id), "qa_run_attempt": int(run_attempt),
                  "reason_code": reason_code, "reason": REASON_MESSAGES[reason_code]}
        with lock.open("x", encoding="utf-8") as target:
            json.dump(record, target, sort_keys=True, indent=2)
            target.write("\n")
        _git(repo, "add", "--", str(lock.relative_to(repo)))
        _git(repo, "-c", "user.name=Mool Katha independent QA feedback",
             "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
             "commit", "-m", f"qa: hold {episode} after independent review {run_id}")
        pushed = _git(repo, "push", "origin", "HEAD:main", allow_failure=True)
        if pushed.returncode == 0:
            return "held"
        if attempt < 2:
            time.sleep(2 * (attempt + 1))
    raise FeedbackError("producer editorial hold could not be pushed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pinned QA failure feedback")
    commands = parser.add_subparsers(dest="command", required=True)
    discovery = commands.add_parser("candidate")
    discovery.add_argument("--discovery", type=Path, required=True)
    discovery.add_argument("--github-output", type=Path, required=True)
    inspection = commands.add_parser("inspect-hold")
    inspection.add_argument("--hold", type=Path, required=True)
    inspection.add_argument("--episode", required=True)
    inspection.add_argument("--video-sha", required=True)
    inspection.add_argument("--source-commit", required=True)
    inspection.add_argument("--github-output", type=Path, required=True)
    writer = commands.add_parser("apply")
    writer.add_argument("--source-repo", type=Path, required=True)
    writer.add_argument("--episode", required=True)
    writer.add_argument("--video-sha", required=True)
    writer.add_argument("--source-commit", required=True)
    writer.add_argument("--reason-code", required=True)
    writer.add_argument("--run-id", required=True)
    writer.add_argument("--run-attempt", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "candidate":
            row = candidate(args.discovery)
            _output(args.github_output, {"candidate": "true", "episode": row["episode_id"],
                                         "video_sha": row["video_sha256"],
                                         "source_commit": row["source_commit"]} if row else
                    {"candidate": "false"})
        elif args.command == "inspect-hold":
            row = {"episode_id": args.episode, "video_sha256": args.video_sha,
                   "source_commit": args.source_commit}
            if not EPISODE.fullmatch(args.episode) or not SHA256.fullmatch(args.video_sha) or \
                    not COMMIT.fullmatch(args.source_commit):
                raise FeedbackError("expected QA candidate identity is malformed")
            code = inspect_hold(args.hold, row)
            _output(args.github_output, {"hold": "true", "reason_code": code} if code else
                    {"hold": "false"})
        else:
            status = apply_lock(args.source_repo, args.episode, args.video_sha, args.source_commit,
                                args.reason_code, args.run_id, args.run_attempt)
            print(status)
        return 0
    except (FeedbackError, OSError, subprocess.TimeoutExpired) as exc:
        # Never repeat artifact data, Git output, credential material, or URLs.
        print(f"QA feedback held: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
