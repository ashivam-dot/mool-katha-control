"""Read one immutable producer commit and one untrusted draft-media tar.

This code is intended to move to an owner-controlled QA repository. It reads
Git objects rather than executing or importing anything in the source checkout.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import (AGENT, EPISODE, PRODUCTION_REPOSITORY, QaHold, digest_bytes,
                     digest_file, expect_sha, json_object, path_under, relative_path,
                     require, write_bytes_new)
from .terms import required_beat_terms


MAX_ARCHIVE_BYTES = 750 * 1024 * 1024
MAX_MEMBER_BYTES = 260 * 1024 * 1024
MAX_MEMBERS = 500
MAX_GIT_BLOB_BYTES = 110 * 1024 * 1024
SOURCE_FILES = ("script.json", "short.yaml", "evidence.json", "work/manifest.json",
                "qc.json", "evidence_pending.json")
FORBIDDEN_STATE = ("final-review.json", "hold.json", "publish.json", "remote.json",
                   "agent-release-review.json", "agent-release-signature.json")
HARD_WARNING = re.compile(r"^(?:final file:|duration |loudness |true peak |speech check failed:|beat \d+: the recognizer heard only)")
BAD_RIGHTS = re.compile(r"\b(?:NC|ND)\b|non.?commercial|no.?derivatives|editorial.only|all.rights.reserved|unknown|unspecified", re.I)
MODAL_PRODUCER = re.compile(r"agent:modal/" + re.escape(PRODUCTION_REPOSITORY) +
                            r"/[A-Za-z0-9._-]{4,}\Z")
ACTIONS_PRODUCER = re.compile(r"agent:github_actions/" + re.escape(PRODUCTION_REPOSITORY) +
                              r"/([0-9]+)/([0-9]+)\Z")


def _production_identity(value: Any) -> bool:
    if not isinstance(value, str) or AGENT.fullmatch(value) is None:
        return False
    if MODAL_PRODUCER.fullmatch(value) is not None:
        return True
    actions = ACTIONS_PRODUCER.fullmatch(value)
    return actions is not None and bool(actions[1].strip("0")) and bool(actions[2].strip("0"))


def _git(repo: Path, *args: str, timeout: int = 30) -> bytes:
    try:
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QaHold("source Git object could not be read") from exc
    require(result.returncode == 0, "source Git object could not be read")
    return result.stdout


def _git_blob(repo: Path, commit: str, name: str, *, missing_ok: bool = False) -> bytes | None:
    relative_path(name, "Git blob path")
    try:
        result = subprocess.run(["git", "-C", str(repo), "ls-tree", "-z", commit, "--", name],
                                capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QaHold("source Git tree could not be read") from exc
    require(result.returncode == 0, "source Git tree could not be read")
    if not result.stdout:
        if missing_ok:
            return None
        raise QaHold(f"committed {name} is missing")
    entries = result.stdout.rstrip(b"\x00").split(b"\x00")
    require(len(entries) == 1, f"committed {name} is ambiguous")
    try:
        header, actual_name = entries[0].split(b"\t", 1)
        mode, kind, oid = header.decode("ascii").split()
        actual = actual_name.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise QaHold(f"committed {name} has an invalid Git entry") from exc
    require(actual == name and kind == "blob" and mode in ("100644", "100755"),
            f"committed {name} is not a regular file")
    size_raw = _git(repo, "cat-file", "-s", oid)
    try:
        size = int(size_raw)
    except ValueError as exc:
        raise QaHold(f"committed {name} has an invalid size") from exc
    require(0 <= size <= MAX_GIT_BLOB_BYTES, f"committed {name} is oversized")
    data = _git(repo, "cat-file", "blob", oid)
    require(len(data) == size, f"committed {name} changed during read")
    return data


def _yaml_object(raw: bytes) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise QaHold("trusted QA needs PyYAML to parse short.yaml") from exc

    class UniqueLoader(yaml.SafeLoader):
        pass

    def unique_mapping(loader: UniqueLoader, node: Any) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key_node, item_node in node.value:
            key = loader.construct_object(key_node)
            require(isinstance(key, str) and key not in value, "short.yaml has a duplicate or nontext key")
            value[key] = loader.construct_object(item_node)
        return value

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
    try:
        value = yaml.load(raw.decode("utf-8"), Loader=UniqueLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise QaHold("short.yaml is invalid") from exc
    require(isinstance(value, dict), "short.yaml must be an object")
    return value


def _archive_members(archive: Path, root: Path, episode: str) -> dict[str, str]:
    require(archive.is_file() and not archive.is_symlink(), "private draft archive is unavailable")
    require(0 < archive.stat().st_size <= MAX_ARCHIVE_BYTES, "private draft archive is oversized or empty")
    allowed_episode = f"content/episodes/{episode}/"
    hashes: dict[str, str] = {}
    total = 0
    try:
        with tarfile.open(archive, mode="r:") as bundle:
            for count, member in enumerate(bundle, 1):
                require(count <= MAX_MEMBERS, "draft archive has too many members")
                name = member.name
                relative_path(name, "draft archive member")
                require(name.startswith(allowed_episode) or name.startswith("pipeline/.cache/"),
                        "draft archive contains another episode or an unexpected root")
                require(member.isreg() and not member.linkname and
                        not any(key.startswith("GNU.sparse") for key in member.pax_headers),
                        "draft archive has a link, sparse file, or nonregular member")
                require(name not in hashes, "draft archive has a duplicate member")
                require(0 <= member.size <= MAX_MEMBER_BYTES, "draft archive has an oversized member")
                total += member.size
                require(total <= MAX_ARCHIVE_BYTES, "draft archive expands beyond its size limit")
                if name.startswith(allowed_episode):
                    relative_episode = name[len(allowed_episode):]
                    require(relative_episode not in FORBIDDEN_STATE,
                            "draft archive contains release state")
                source = bundle.extractfile(member)
                require(source is not None, "draft archive member cannot be read")
                data = source.read(member.size + 1)
                require(len(data) == member.size, "draft archive member has a truncated body")
                target = path_under(root, name, "draft archive member")
                if target.exists():
                    require(target.is_file() and not target.is_symlink() and target.read_bytes() == data,
                            f"draft archive conflicts with committed {name}")
                else:
                    write_bytes_new(target, data)
                hashes[name] = digest_file(target)
    except (OSError, tarfile.TarError, EOFError) as exc:
        raise QaHold("private draft archive is unreadable") from exc
    require(bool(hashes), "private draft archive has no regular files")
    return hashes


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _records(value: Any, label: str) -> dict[str, dict[str, Any]]:
    require(isinstance(value, list) and bool(value), f"{label}: expected nonempty list")
    found: dict[str, dict[str, Any]] = {}
    for item in value:
        require(isinstance(item, dict), f"{label}: expected object")
        identity = item.get("id")
        require(isinstance(identity, str) and re.fullmatch(r"[A-Za-z0-9._:-]+", identity) is not None,
                f"{label}: invalid ID")
        require(identity not in found, f"{label}: duplicate ID {identity}")
        found[identity] = item
    return found


@dataclass(frozen=True)
class Candidate:
    root: Path
    source_repo: Path
    episode_id: str
    source_commit: str
    qa_agent_id: str
    producer_agent_id: str
    script: dict[str, Any]
    spec: dict[str, Any]
    ledger: dict[str, Any]
    manifest: dict[str, Any]
    qc: dict[str, Any]
    pending: dict[str, Any]
    hashes: dict[str, str]
    assets: dict[str, dict[str, Any]]
    asset_paths: dict[str, Path]
    internal_references: dict[str, str]
    archive_members: dict[str, str]

    @property
    def episode_dir(self) -> Path:
        return self.root / "content" / "episodes" / self.episode_id

    @property
    def video_path(self) -> Path:
        return self.episode_dir / f"{self.episode_id}.mp4"

    @property
    def check(self) -> dict[str, Any]:
        return self.qc["check"]

    def recheck(self) -> None:
        files = {"script": self.episode_dir / "script.json", "spec": self.episode_dir / "short.yaml",
                 "evidence": self.episode_dir / "evidence.json", "manifest": self.episode_dir / "work/manifest.json",
                 "qc": self.episode_dir / "qc.json", "video": self.video_path}
        for name, path in files.items():
            require(digest_file(path) == self.hashes[name], f"candidate {name} changed after QA started")
        require(digest_file(self.episode_dir / "evidence_pending.json") == self.hashes["pending"],
                "candidate pending marker changed after QA started")
        for identity, path in self.asset_paths.items():
            require(digest_file(path) == self.assets[identity]["sha256"],
                    f"candidate asset {identity} changed after QA started")
        for name, digest in self.internal_references.items():
            require(digest_file(self.root / name) == digest,
                    f"candidate internal provenance {name} changed after QA started")
        for name, digest in self.archive_members.items():
            require(digest_file(self.root / name) == digest,
                    f"candidate archived file {name} changed after QA started")

    def committed_blob(self, name: str, claimed_commit: str | None = None) -> bytes:
        """Read exact source bytes from the frozen commit, checking any older claimed commit."""
        pinned = _git_blob(self.source_repo, self.source_commit, name)
        assert pinned is not None
        if claimed_commit is not None:
            require(isinstance(claimed_commit, str) and
                    re.fullmatch(r"[0-9a-f]{40}", claimed_commit) is not None,
                    f"source {name}: claimed Git commit is invalid")
            # A shallow checkout cannot verify the claim and must hold. The QA
            # workflow fetches complete source history for this check.
            try:
                relation = subprocess.run(
                    ["git", "-C", str(self.source_repo), "merge-base", "--is-ancestor",
                     claimed_commit, self.source_commit], capture_output=True,
                    timeout=20, check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise QaHold(f"source {name}: claimed Git history could not be checked") from exc
            require(relation.returncode == 0 and
                    _git_blob(self.source_repo, claimed_commit, name) == pinned,
                    f"source {name}: claimed commit does not contain the frozen bytes")
        return pinned


def _asset_file(root: Path, episode: str, name: Any) -> Path:
    relative = relative_path(name, "manifest asset file")
    if relative.parts[0] == "episode":
        require(len(relative.parts) > 1, "legacy manifest asset file is incomplete")
        return path_under(root, f"content/episodes/{episode}/" + "/".join(relative.parts[1:]),
                          "manifest asset file")
    return path_under(root, name, "manifest asset file")


def _pending_review(value: Any, label: str, asset_hash: str | None = None) -> None:
    require(isinstance(value, dict) and value.get("method") == "human" and
            value.get("decision") == "pending" and value.get("reviewer") == "" and
            value.get("reviewed_at") == "", f"{label}: human review must remain pending")
    if asset_hash is not None:
        require(value.get("asset_sha256") == asset_hash,
                f"{label}: pending review must name exact asset hash")


def _check_transform(candidate_root: Path, episode: str, spec: dict, manifest: dict,
                     uses: dict[str, dict]) -> None:
    tempo = spec.get("narration_tempo", 1.0)
    require(_number(tempo) and 0.8 <= tempo <= 1.0, "invalid narration tempo")
    transform = manifest.get("narration_transform")
    if tempo == 1.0:
        require("narration_transform" not in manifest,
                "unexpected narration transform for unpaced narration")
        return
    require(isinstance(transform, dict) and transform.get("method") == "ffmpeg_atempo" and
            transform.get("tempo") == tempo, "narration transform does not match spec")
    names = (("source_file", "source_narration_sha256", "narration-source.wav"),
             ("take_file", "take_sha256", "take.wav"),
             ("take_record_file", "take_record_sha256", "take.json"),
             ("output_file", "output_narration_sha256", "narration.wav"))
    for file_key, hash_key, basename in names:
        path = _asset_file(candidate_root, episode, transform.get(file_key))
        require(path == candidate_root / "content" / "episodes" / episode / "work" / basename,
                f"narration transform {file_key} names another file")
        require(path.is_file() and digest_file(path) == expect_sha(transform.get(hash_key), hash_key),
                f"narration transform {hash_key} changed")
    require(transform["source_narration_sha256"] != transform["output_narration_sha256"],
            "narration transform did not produce a distinct file")
    record_path = _asset_file(candidate_root, episode, transform["take_record_file"])
    take = json_object(record_path.read_bytes(), "take record")
    require(take.get("narration_sha256") == transform["source_narration_sha256"] and
            take.get("take_sha256") == transform["take_sha256"], "take record differs from transform")
    voice = uses.get(manifest.get("narration_asset_id"))
    require(voice is not None and voice.get("file") == transform["output_file"] and
            voice.get("sha256") == transform["output_narration_sha256"],
            "final voice differs from narration transform")


def load_candidate(repo: Path, archive: Path, destination: Path, episode_id: str,
                   source_commit: str, qa_agent_id: str) -> Candidate:
    """Copy and validate one candidate before any paid or semantic QA call."""
    require(isinstance(episode_id, str) and EPISODE.fullmatch(episode_id) is not None,
            "candidate needs an epNNN ID")
    require(isinstance(source_commit, str) and re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None,
            "candidate needs an exact source commit")
    require(isinstance(qa_agent_id, str) and AGENT.fullmatch(qa_agent_id) is not None,
            "QA agent identity is invalid")
    require(repo.is_dir() and (repo / ".git").exists(), "read-only source checkout is unavailable")
    require(_git(repo, "rev-parse", "HEAD").decode("ascii").strip() == source_commit,
            "source checkout HEAD differs from the requested immutable commit")
    require(not destination.exists(), "candidate snapshot destination must be new")
    destination.mkdir(parents=True, mode=0o700)
    base = f"content/episodes/{episode_id}/"
    for filename in SOURCE_FILES:
        name = base + filename
        data = _git_blob(repo, source_commit, name)
        assert data is not None
        write_bytes_new(path_under(destination, name, "committed source"), data)
    for filename in FORBIDDEN_STATE:
        require(_git_blob(repo, source_commit, base + filename, missing_ok=True) is None,
                f"candidate already has {filename}")
    folder = destination / base
    script = json_object((folder / "script.json").read_bytes(), "script.json")
    spec = _yaml_object((folder / "short.yaml").read_bytes())
    ledger = json_object((folder / "evidence.json").read_bytes(), "evidence.json")
    manifest = json_object((folder / "work/manifest.json").read_bytes(), "work/manifest.json")
    qc = json_object((folder / "qc.json").read_bytes(), "qc.json")
    pending = json_object((folder / "evidence_pending.json").read_bytes(), "evidence_pending.json")
    producer = manifest.get("production_agent_id")
    require(_production_identity(producer) and producer != qa_agent_id,
            "manifest needs a distinct cloud production_agent_id before independent QA")

    archive_members = _archive_members(archive, destination, episode_id)
    video = folder / f"{episode_id}.mp4"
    require(video.is_file() and not video.is_symlink(), "private archive is missing the final MP4")
    hashes = {"script": digest_file(folder / "script.json"),
              "spec": digest_file(folder / "short.yaml"),
              "evidence": digest_file(folder / "evidence.json"),
              "manifest": digest_file(folder / "work/manifest.json"),
              "qc": digest_file(folder / "qc.json"), "video": digest_file(video),
              "pending": digest_file(folder / "evidence_pending.json")}
    require(spec.get("id") == manifest.get("id") == ledger.get("episode_id") ==
            pending.get("episode_id") == episode_id, "candidate records identify different episodes")
    require(manifest.get("language") == "hi", "candidate narration is not Hindi")
    require(manifest.get("video_sha256") == ledger.get("video_sha256") ==
            qc.get("video_sha256") == pending.get("video_sha256") == hashes["video"],
            "candidate video hash differs across manifest, QC, ledger, or pending marker")
    require(qc.get("spec_sha256") == hashes["spec"] and qc.get("kind") == "automated",
            "QC is not bound to the exact final spec")
    normalized_script = json.dumps(script, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False).encode("utf-8")
    require(ledger.get("script_sha256") == pending.get("script_sha256") == digest_bytes(normalized_script),
            "ledger or pending marker differs from the frozen script")

    beats = script.get("beats")
    spec_beats = spec.get("beats")
    render_beats = manifest.get("beats")
    require(all(isinstance(value, list) and value for value in (beats, spec_beats, render_beats)) and
            len(beats) == len(spec_beats) == len(render_beats),
            "script, spec, and final render beats differ")
    for index, (script_beat, spec_beat, render_beat) in enumerate(zip(beats, spec_beats, render_beats), 1):
        require(all(isinstance(value, dict) for value in (script_beat, spec_beat, render_beat)) and
                isinstance(script_beat.get("text"), str) and script_beat["text"].strip() and
                script_beat["text"] == spec_beat.get("text") == render_beat.get("text"),
                f"beat {index} narration differs across frozen records")
        required_beat_terms(script_beat["text"])
    require(spec.get("hook_text") == manifest.get("hook_text") and
            spec.get("citation") == manifest.get("citation"),
            "first-frame headline or source differs from the final render")

    claim_records = _records(ledger.get("claims"), "ledger claims")
    spoken: set[str] = set()
    for index, beat in enumerate(beats, 1):
        ids = beat.get("claim_ids", beat.get("claims"))
        require(isinstance(ids, list), f"beat {index} has no claim ID list")
        for identity in ids:
            require(isinstance(identity, str), f"beat {index} has an invalid claim ID")
            spoken.add(identity)
    require(bool(spoken) and spoken == set(claim_records), "script and ledger claim IDs differ")
    pending_claims = pending.get("pending_claims")
    require(isinstance(pending_claims, list) and len(pending_claims) == len(set(pending_claims)) and
            set(pending_claims) == set(claim_records),
            "pending marker does not name every exact claim")
    for identity, claim in claim_records.items():
        require(claim.get("type") == "scriptural" and claim.get("risk") == "low",
                f"claim {identity} is not low-risk scriptural material")
        _pending_review(claim.get("review"), f"claim {identity} review")
        for name in ("primary", "corroboration"):
            source = claim.get(name)
            require(isinstance(source, dict) and isinstance(source.get("url"), str) and
                    source["url"].startswith("https://") and
                    isinstance(source.get("printed_verse_label"), str) and
                    bool(source["printed_verse_label"].strip()) and
                    isinstance(source.get("excerpt"), str) and bool(source["excerpt"].strip()),
                    f"claim {identity} lacks inspectable {name} source evidence")
        require(claim["primary"]["url"] != claim["corroboration"]["url"],
                f"claim {identity} repeats one source as corroboration")
        if "correspondence" in claim:
            correspondence = claim["correspondence"]
            require(isinstance(correspondence, dict) and
                    correspondence.get("primary_printed_label") == claim["primary"]["printed_verse_label"] and
                    correspondence.get("corroboration_printed_label") == claim["corroboration"]["printed_verse_label"] and
                    isinstance(correspondence.get("alignment_note"), str) and
                    len(correspondence["alignment_note"].strip()) >= 30,
                    f"claim {identity} has unresolved cross-edition correspondence")

    uses = _records(manifest.get("asset_uses"), "manifest asset uses")
    assets = _records(ledger.get("assets"), "ledger assets")
    require(set(uses) == set(assets), "manifest and ledger used asset IDs differ")
    paths: dict[str, Path] = {}
    internal_references: dict[str, str] = {}
    for identity, use in uses.items():
        asset = assets[identity]
        require(asset.get("role") != "animation",
                f"asset {identity}: generated animation needs dedicated validation before agent QA")
        for field in ("role", "file", "sha256"):
            require(use.get(field) == asset.get(field),
                    f"asset {identity} differs between ledger and final manifest")
        expected = expect_sha(use.get("sha256"), f"asset {identity} SHA-256")
        path = _asset_file(destination, episode_id, use.get("file"))
        if not path.exists():
            git_name = path.relative_to(destination).as_posix()
            blob = _git_blob(repo, source_commit, git_name, missing_ok=True)
            require(blob is not None, f"manifest asset {identity} is missing from Git and archive")
            write_bytes_new(path, blob)
        else:
            git_name = path.relative_to(destination).as_posix()
            blob = _git_blob(repo, source_commit, git_name, missing_ok=True)
            if blob is not None:
                require(path.read_bytes() == blob, f"manifest asset {identity} conflicts with Git")
        require(digest_file(path) == expected, f"manifest asset {identity} SHA-256 differs from exact bytes")
        paths[identity] = path
        _pending_review(asset.get("rights_review"), f"asset {identity} rights review", expected)
        require(asset.get("commercial_use") is True and asset.get("derivatives_allowed") is True and
                isinstance(asset.get("license"), str) and not BAD_RIGHTS.search(asset["license"]) and
                isinstance(asset.get("credit"), str) and bool(asset["credit"].strip()) and
                isinstance(asset.get("rights_basis"), str) and len(asset["rights_basis"].strip()) >= 12,
                f"asset {identity} has incompatible or undocumented rights")
        rights_url = asset.get("rights_url")
        require(isinstance(rights_url, str) and
                (rights_url.startswith("https://") or rights_url.startswith("internal:")),
                f"asset {identity} lacks an inspectable rights source")
        for reference_key in ("origin", "rights_url"):
            reference = asset.get(reference_key)
            if not isinstance(reference, str) or not reference.startswith("internal:"):
                continue
            internal_name = reference[len("internal:"):]
            internal = _asset_file(destination, episode_id, internal_name)
            git_name = internal.relative_to(destination).as_posix()
            blob = _git_blob(repo, source_commit, git_name, missing_ok=True)
            if not internal.exists():
                require(blob is not None, f"asset {identity} internal {reference_key} is missing")
                write_bytes_new(internal, blob)
            elif blob is not None:
                require(internal.read_bytes() == blob,
                        f"asset {identity} internal {reference_key} conflicts with Git")
            require(internal.is_file() and not internal.is_symlink() and internal.stat().st_size <= 5_000_000,
                    f"asset {identity} internal {reference_key} cannot be inspected")
            internal_hash = digest_file(internal)
            if blob is not None:
                require(internal_hash == digest_bytes(blob),
                        f"asset {identity} internal {reference_key} differs from committed Git")
            require(git_name not in internal_references or internal_references[git_name] == internal_hash,
                    f"asset {identity} internal {reference_key} changed during candidate collection")
            internal_references[git_name] = internal_hash

    voice_id = manifest.get("narration_asset_id")
    require(voice_id in uses and uses[voice_id].get("role") == "voice",
            "final narration asset is missing from manifest")
    for font_id in manifest.get("font_asset_ids", []):
        require(font_id in uses and uses[font_id].get("role") == "font",
                "rendered font asset is missing from manifest")
    for role, key in (("music", "music_asset_id"),):
        value = manifest.get(key)
        if value is not None:
            require(value in uses and uses[value].get("role") == role,
                    f"rendered {role} asset is missing from manifest")
    for sfx_id in manifest.get("sfx_asset_ids", []):
        require(sfx_id in uses and uses[sfx_id].get("role") == "sfx",
                "rendered sound effect is missing from manifest")
    for index, beat in enumerate(render_beats, 1):
        visual = beat.get("asset")
        require(isinstance(visual, dict) and visual.get("asset_id") in uses and
                uses[visual["asset_id"]].get("role") == "visual",
                f"beat {index} visual asset is missing from manifest")
        for key in ("then_card_asset_id",):
            if key in beat:
                require(beat[key] in uses and uses[beat[key]].get("role") == "visual",
                        f"beat {index} second visual asset is missing from manifest")
        for derived_id in beat.get("derived_asset_ids", []):
            require(derived_id in uses and uses[derived_id].get("role") == "animation",
                    f"beat {index} animation asset is missing from manifest")
    _check_transform(destination, episode_id, spec, manifest, uses)

    check = qc.get("check")
    require(isinstance(check, dict), "QC check is missing")
    duration = check.get("duration")
    require(_number(duration) and 42 <= duration <= 62, "hard QC Hindi duration failed")
    require(_number(check.get("integrated_lufs")) and abs(check["integrated_lufs"] + 14) <= 1,
            "hard QC loudness failed")
    require(_number(check.get("true_peak_dbfs")) and check["true_peak_dbfs"] <= -1,
            "hard QC true peak failed")
    require(isinstance(check.get("words"), int) and not isinstance(check["words"], bool) and
            check["words"] > 0, "hard QC word count failed")
    require(check.get("stream_problems") == [], "hard QC stream probe failed")
    require(check.get("speech_error") in (None, "") and "speech_error" in check,
            "hard QC speech comparison failed")
    for field in ("warnings", "speech_differences"):
        require(isinstance(check.get(field), list) and
                all(isinstance(item, str) and bool(item.strip()) for item in check[field]),
                f"QC {field} is invalid")
    require(isinstance(check.get("sources"), list) and len(check["sources"]) == len(beats),
            "QC rendered sources differ from frozen beats")
    for warning in check["warnings"]:
        require(not HARD_WARNING.match(warning) and "is a plain gradient" not in warning,
                f"hard QC warning requires repair: {warning}")
    return Candidate(destination, repo, episode_id, source_commit, qa_agent_id, producer, script, spec,
                     ledger, manifest, qc, pending, hashes, assets, paths,
                     internal_references, archive_members)
