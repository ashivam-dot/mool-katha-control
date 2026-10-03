"""Unattended, unsigned independent QA orchestration with private failure artifacts."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

from .archive import fetch_modal_archive
from .asr import gemini_full_audio, save_asr_results, whisper_full_audio
from .audio_quality import gemini_voice_quality
from .assemble import assemble_approved_review, save_unsigned_review
from .candidate import FORBIDDEN_STATE, _git, _git_blob, _production_identity, load_candidate
from .common import (EPISODE, QA_REPOSITORY, QaHold, digest_file, json_object,
                     expect_sha, require, utc_now, valid_qa_workflow_ref,
                     write_json_new)
from .media import extract_full_final_audio, make_visual_evidence
from .observations import collect_observations
from .reviewer import gemini_independent_review, gemini_review_frame_batches
from .signing import sign_assembled_review
from .voice_exchange import VoiceExchangePolicy


def github_run_context(env: dict[str, str] | None = None) -> tuple[dict[str, Any], str]:
    """Use actual Actions environment, never a model-proposed QA identity."""
    values = os.environ if env is None else env
    repository = values.get("GITHUB_REPOSITORY", "")
    workflow_ref = values.get("GITHUB_WORKFLOW_REF", "")
    workflow_sha = values.get("GITHUB_WORKFLOW_SHA", "")
    require(repository == QA_REPOSITORY,
            "QA must run in the owner-controlled mool-katha-control repository")
    require(valid_qa_workflow_ref(repository, workflow_ref),
            "QA workflow ref is unavailable or malformed")
    require(re.fullmatch(r"[0-9a-f]{40}", workflow_sha) is not None,
            "QA workflow commit SHA is unavailable")
    try:
        run_id = int(values.get("GITHUB_RUN_ID", ""))
        attempt = int(values.get("GITHUB_RUN_ATTEMPT", ""))
    except ValueError as exc:
        raise QaHold("QA cloud run ID or attempt is unavailable") from exc
    require(run_id > 0 and attempt > 0, "QA cloud run identity is invalid")
    identity = f"agent:github_actions/{repository}/{run_id}/{attempt}"
    return ({"system": "github_actions", "repository": repository,
             "workflow_ref": workflow_ref, "workflow_sha": workflow_sha,
             "run_id": run_id, "run_attempt": attempt}, identity)


def discover_pending(repo: Path, source_commit: str, *, limit: int = 8) -> list[dict[str, str]]:
    """List independent pending keys from one immutable private source commit."""
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None,
            "pending discovery needs an immutable source commit")
    require(_git(repo, "rev-parse", "HEAD").decode("ascii").strip() == source_commit,
            "pending discovery checkout differs from source commit")
    require(isinstance(limit, int) and 1 <= limit <= 32, "pending discovery limit is invalid")
    names = _git(repo, "ls-tree", "-rz", "--name-only", source_commit,
                 "--", "content/episodes").split(b"\x00")
    pending: list[dict[str, str]] = []
    for name in names:
        match = re.fullmatch(rb"content/episodes/(ep[0-9]{3})/evidence_pending\.json", name)
        if not match:
            continue
        episode = match[1].decode("ascii")
        name = name.decode("ascii")
        base = f"content/episodes/{episode}/"
        try:
            if any(_git_blob(repo, source_commit, base + state, missing_ok=True) is not None
                   for state in FORBIDDEN_STATE):
                continue
            marker_bytes = _git_blob(repo, source_commit, name)
            manifest_bytes = _git_blob(repo, source_commit, base + "work/manifest.json", missing_ok=True)
            if marker_bytes is None or manifest_bytes is None:
                continue
            marker = json_object(marker_bytes, name)
            manifest = json_object(manifest_bytes, base + "work/manifest.json")
        except QaHold:
            # One malformed episode must not prevent independent work on the others.
            continue
        digest = marker.get("video_sha256")
        if (marker.get("episode_id") == manifest.get("id") == episode and
                isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) and
                manifest.get("video_sha256") == digest and
                _production_identity(manifest.get("production_agent_id"))):
            pending.append({"episode_id": episode, "video_sha256": digest,
                            "source_commit": source_commit})
            if len(pending) == limit:
                break
    return pending[:limit]


def run_one(repo: Path, episode_id: str, source_commit: str, output_dir: Path,
            archive_path: Path | None = None, env: dict[str, str] | None = None,
            qa_signing_key: bytes | None = None, qa_signing_key_id: str | None = None,
            *, voice_bundle_path: Path | None = None,
            voice_policy: VoiceExchangePolicy | None = None) -> Path:
    """Produce an unsigned or explicitly signed review after all checks pass."""
    values = os.environ if env is None else env
    require(EPISODE.fullmatch(episode_id) is not None, "QA needs an epNNN candidate")
    require(not output_dir.exists(), "QA output directory must be new")
    output_dir.mkdir(parents=True, mode=0o700)
    private = output_dir / "private"
    private.mkdir(mode=0o700)
    stage = "qa_run_provenance"
    video_hash: str | None = None
    try:
        run, qa_id = github_run_context(values)
        stage = "configuration"
        key = values.get("QA_GEMINI_API_KEY", "")
        require(isinstance(key, str) and bool(key.strip()), "QA Gemini credential is unavailable")
        for name in ("QA_GEMINI_ASR_MODEL", "QA_GEMINI_QUALITY_MODEL", "QA_GEMINI_REVIEW_MODEL"):
            value = values.get(name, "")
            require(isinstance(value, str) and re.fullmatch(r"gemini-[A-Za-z0-9._-]+", value) is not None,
                    f"{name} must be explicitly named")
        require(bool(values.get("QA_WHISPER_MODEL_DIR", "")),
                "QA_WHISPER_MODEL_DIR is unavailable")
        whisper_repo = values.get("QA_WHISPER_MODEL_REPO", "")
        require(isinstance(whisper_repo, str) and
                re.fullmatch(r"[A-Za-z0-9._/-]+", whisper_repo) is not None and
                "/" in whisper_repo,
                "QA_WHISPER_MODEL_REPO is invalid")
        whisper_revision = values.get("QA_WHISPER_MODEL_REVISION", "")
        require(isinstance(whisper_revision, str) and
                re.fullmatch(r"[0-9a-f]{40}", whisper_revision) is not None,
                "QA_WHISPER_MODEL_REVISION is not pinned")
        expect_sha(values.get("QA_WHISPER_MODEL_SHA256"), "QA_WHISPER_MODEL_SHA256")
        stage = "candidate"
        if archive_path is None:
            archive_path = fetch_modal_archive(episode_id, private / "draft-media.tar")
        require(archive_path.is_file() and not archive_path.is_symlink(),
                "private draft archive is unavailable")
        archive_sha = digest_file(archive_path)
        write_json_new(private / "candidate-source.json", {
            "episode_id": episode_id, "source_commit": source_commit,
            "archive_sha256": archive_sha, "qa_agent_id": qa_id})
        candidate = load_candidate(repo, archive_path, output_dir / "snapshot", episode_id,
                                   source_commit, qa_id)
        require(digest_file(archive_path) == archive_sha,
                "private draft archive changed during candidate collection")
        video_hash = candidate.hashes["video"]
        if any(asset.get("role") == "voice" and
               isinstance(asset.get("origin"), str) and
               asset["origin"].startswith("control:gemini:")
               for asset in candidate.assets.values()):
            stage = "control_voice_input"
            require(isinstance(voice_policy, VoiceExchangePolicy) and
                    isinstance(voice_bundle_path, Path),
                    "control voice needs a separately pinned owner policy and signed bundle")
        stage = "media_decode"
        audio_path = private / "full-final-audio.wav"
        duration = extract_full_final_audio(candidate.video_path, audio_path, candidate.check["duration"])
        stage = "source_and_rights"
        observations = collect_observations(candidate, private,
                                            voice_bundle_path=voice_bundle_path,
                                            voice_policy=voice_policy,
                                            final_audio_path=audio_path)
        stage = "media_decode"
        visual = make_visual_evidence(candidate.video_path, candidate.episode_dir,
                                      candidate.manifest["beats"], candidate.check["duration"],
                                      candidate.hashes["video"], private)
        stage = "independent_asr"
        gemini = gemini_full_audio(audio_path, duration, video_hash, key=key,
                                   model=values.get("QA_GEMINI_ASR_MODEL", ""),
                                   audit_path=private / "gemini-asr-response.json")
        whisper = whisper_full_audio(
            audio_path, duration, video_hash,
            model_dir=Path(values.get("QA_WHISPER_MODEL_DIR", "")),
            model_repo=values.get("QA_WHISPER_MODEL_REPO", ""),
            revision=values.get("QA_WHISPER_MODEL_REVISION", ""),
            model_sha256=values.get("QA_WHISPER_MODEL_SHA256", ""),
            github_run_id=run["run_id"], github_run_attempt=run["run_attempt"],
        )
        references = save_asr_results(candidate.episode_dir, gemini, whisper)
        results = {"agent-asr-gemini.json": gemini, "agent-asr-whisper.json": whisper}
        stage = "full_audio_quality"
        quality = gemini_voice_quality(
            audio_path, duration, video_hash, candidate.episode_dir, private, key=key,
            model=values.get("QA_GEMINI_QUALITY_MODEL", ""))
        stage = "frame_indexed_visual_review"
        frame_review = gemini_review_frame_batches(
            candidate, visual, private, key=key,
            model=values.get("QA_GEMINI_REVIEW_MODEL", ""))
        stage = "independent_review"
        model = gemini_independent_review(candidate, observations, [gemini, whisper], visual,
                                          quality, frame_review, private, key=key,
                                          model=values.get("QA_GEMINI_REVIEW_MODEL", ""))
        stage = "strict_review_validation"
        review = assemble_approved_review(candidate, observations, results, references,
                                          visual, frame_review, model, quality, run)
        path = save_unsigned_review(candidate, review)
        if qa_signing_key is not None or qa_signing_key_id is not None:
            stage = "qa_signing"
            require(qa_signing_key is not None and qa_signing_key_id is not None,
                    "QA signing configuration is incomplete")
            sign_assembled_review(candidate, review, path, quality, run,
                                  qa_signing_key_id, qa_signing_key)
        write_json_new(private / "run-result.json", {
            "status": "reviewed_signed" if qa_signing_key is not None else "reviewed_unsigned",
            "signature_sha256": (digest_file(candidate.episode_dir / "agent-release-signature.json")
                                 if qa_signing_key is not None else None),
            "video_sha256": video_hash, "review_sha256": digest_file(path),
            "audio_quality_observation_sha256": quality.record_sha256,
            "audio_quality_response_sha256": quality.record["response_sha256"],
            "review_file": path.relative_to(output_dir).as_posix(),
            "completed_at": utc_now()})
        return path
    except QaHold as exc:
        write_json_new(private / "agent-qa-hold.json", {
            "status": "hold", "stage": stage, "reason": str(exc), "episode_id": episode_id,
            "source_commit": source_commit, "video_sha256": video_hash,
            "held_at": utc_now()})
        raise
    except Exception as exc:
        reason = f"unexpected trusted QA failure ({type(exc).__name__})"
        write_json_new(private / "agent-qa-hold.json", {
            "status": "hold", "stage": stage, "reason": reason, "episode_id": episode_id,
            "source_commit": source_commit, "video_sha256": video_hash,
            "held_at": utc_now()})
        raise QaHold(reason) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unsigned independent Mool Katha QA runner")
    command = parser.add_subparsers(dest="command", required=True)
    pending = command.add_parser("discover", help="list pending (episode, video hash) keys at one Git commit")
    pending.add_argument("--source-repo", type=Path, required=True)
    pending.add_argument("--source-commit", required=True)
    pending.add_argument("--limit", type=int, default=8)
    fetch = command.add_parser("fetch-archive", help="read one private Modal draft tar without QA secrets")
    fetch.add_argument("--episode", required=True)
    fetch.add_argument("--output", type=Path, required=True)
    run = command.add_parser("run", help="produce private, unsigned independent QA artifacts")
    run.add_argument("--source-repo", type=Path, required=True)
    run.add_argument("--source-commit", required=True)
    run.add_argument("--episode", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--archive", type=Path, help="already fetched private Modal draft tar")
    run.add_argument("--qa-key-file", type=Path, help="private 32-byte Ed25519 seed; signed run only")
    run.add_argument("--qa-key-id", help="reviewed QA signing key ID; signed run only")
    args = parser.parse_args(argv)
    try:
        if args.command == "discover":
            import json
            print(json.dumps(discover_pending(args.source_repo, args.source_commit, limit=args.limit),
                             separators=(",", ":")))
        elif args.command == "fetch-archive":
            path = fetch_modal_archive(args.episode, args.output)
            print(f"Private draft archive: {path}")
        else:
            require((args.qa_key_file is None) == (args.qa_key_id is None),
                    "QA signing key file and ID must be supplied together")
            key = None
            if args.qa_key_file is not None:
                require(args.qa_key_file.is_file() and not args.qa_key_file.is_symlink() and
                        args.qa_key_file.stat().st_size == 32,
                        "QA signing key file must be an exact private seed")
                key = args.qa_key_file.read_bytes()
            path = run_one(args.source_repo, args.episode, args.source_commit,
                           args.output_dir, args.archive,
                           qa_signing_key=key, qa_signing_key_id=args.qa_key_id)
            print(f"Independent QA review: {path}")
        return 0
    except QaHold as exc:
        print(f"QA hold: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
