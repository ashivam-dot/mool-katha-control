"""Attack-oriented tests for the immutable candidate and private tar boundary."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from trusted_qa.candidate import load_candidate
from trusted_qa.common import QaHold, digest_bytes, unique_json
from trusted_qa.runner import discover_pending


EPISODE = "ep004"
QA_ID = "agent:github_actions/ashivam-dot/mool-katha-control/123/1"
PRODUCER_ID = "agent:modal/ashivam-dot/mool-katha/fc-synthetic-1234"


def _json(data: dict) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _run(*args: str, cwd: Path) -> str:
    return subprocess.check_output(args, cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()


def candidate_fixture(root: Path, *, producer: str | None = PRODUCER_ID,
                      warning: str | None = None,
                      media_override: dict[str, bytes] | None = None,
                      extra_members: list[tarfile.TarInfo] | None = None,
                      extra_internal_provenance: bool = False) -> tuple[Path, Path, str]:
    repo = root / "source"
    repo.mkdir()
    _run("git", "init", "-q", cwd=repo)
    _run("git", "config", "user.name", "QA Test", cwd=repo)
    _run("git", "config", "user.email", "qa@example.test", cwd=repo)
    base = repo / "content" / "episodes" / EPISODE
    (base / "work").mkdir(parents=True)
    font = repo / "pipeline" / "assets" / "fonts" / "test.ttf"
    font.parent.mkdir(parents=True)
    font.write_bytes(b"font bytes")
    spoken = "राम नल से कहते हैं।"
    script = {"hook_text": "राम नल से कहते हैं", "beats": [{"text": spoken, "claim_ids": ["c1"]}]}
    citation = "वाल्मीकि रामायण · युद्धकाण्ड · वैद्य 1971 आलोचनात्मक पाठ · सर्ग 15"
    spec = {"id": EPISODE, "hook_text": script["hook_text"], "citation": citation,
            "beats": [{"text": spoken, "visual": {"source": "card"}}]}
    video = b"synthetic MP4 bytes for candidate boundary only"
    media = {
        f"content/episodes/{EPISODE}/{EPISODE}.mp4": video,
        f"content/episodes/{EPISODE}/work/narration.wav": b"synthetic voice bytes",
        f"content/episodes/{EPISODE}/work/assets/beat01.png": b"synthetic visual bytes",
    }
    if media_override:
        media.update(media_override)

    def asset(identity: str, role: str, path: str, data: bytes) -> dict:
        digest = digest_bytes(data)
        return {"id": identity, "role": role, "file": path, "sha256": digest,
                "origin": f"internal:content/episodes/{EPISODE}/short.yaml",
                "creator": "Mool Katha", "license": "Original",
                "rights_url": f"internal:content/episodes/{EPISODE}/short.yaml",
                "rights_basis": "Original generated asset from a frozen episode card specification.",
                "credit": "Mool Katha original work", "commercial_use": True,
                "derivatives_allowed": True, "retrieved_at": "2026-10-03",
                "rights_review": {"method": "human", "reviewer": "", "decision": "pending",
                                  "reviewed_at": "", "asset_sha256": digest}}

    visual_name = f"content/episodes/{EPISODE}/work/assets/beat01.png"
    voice_name = f"content/episodes/{EPISODE}/work/narration.wav"
    assets = [asset("visual:one", "visual", visual_name, media[visual_name]),
              asset("voice:one", "voice", voice_name, media[voice_name]),
              asset("font:one", "font", "pipeline/assets/fonts/test.ttf", font.read_bytes())]
    if extra_internal_provenance:
        provenance = repo / "pipeline" / "src" / "ytc" / "test_provenance.txt"
        provenance.parent.mkdir(parents=True, exist_ok=True)
        provenance.write_text("Original visual generated from an owned card design.\n", encoding="utf-8")
        assets[0]["rights_url"] = "internal:pipeline/src/ytc/test_provenance.txt"
    uses = [{key: record[key] for key in ("id", "role", "file", "sha256")} for record in assets]
    manifest = {"id": EPISODE, "hook_text": spec["hook_text"], "citation": spec["citation"],
                "language": "hi", "duration": 43.0, "video_sha256": digest_bytes(video),
                "asset_uses": uses, "narration_asset_id": "voice:one",
                "font_asset_ids": ["font:one"], "music_asset_id": None, "sfx_asset_ids": [],
                "beats": [{"start": 0.0, "end": 43.0, "text": spoken,
                           "asset": {"asset_id": "visual:one"}, "derived_asset_ids": []}]}
    if producer is not None:
        manifest["production_agent_id"] = producer
    claim = {"id": "c1", "type": "scriptural", "risk": "low", "hindi": spoken,
             "claim": "राम नल से कहते हैं", "tradition": "वाल्मीकि रामायण",
             "variant_caveat": "", "screen_source": citation,
             "primary": {"kind": "passage", "work": "Valmiki Ramayana",
                         "edition": "P. L. Vaidya (ed.), Critical Edition, vol. VI, Yuddhakanda, Oriental Institute Baroda, 1971",
                         "creator": "P. L. Vaidya, editor; Oriental Institute, Baroda",
                         "division": "Yuddhakanda", "chapter": "15", "verse": "8",
                         "url": "https://example.org/one", "printed_verse_label": "15.8",
                         "excerpt": "राम ने नल को कहा कि आगे चलो"},
             "corroboration": {"kind": "translation", "work": "Valmiki Ramayana",
                               "edition": "K. M. K. Murthy, independent English translation, 2004",
                               "creator": "K. M. K. Murthy", "division": "Yuddhakanda",
                               "chapter": "15", "verse": "8",
                               "url": "https://example.net/two", "printed_verse_label": "15.8",
                               "excerpt": "राम ने नल से बात की और आगे बढ़े",
                               "independence_note": "This separately produced translation is independent of the primary critical edition."},
             "review": {"method": "human", "reviewer": "", "decision": "pending", "reviewed_at": ""}}
    script_digest = hashlib.sha256(json.dumps(script, ensure_ascii=False, sort_keys=True,
                                              separators=(",", ":")).encode("utf-8")).hexdigest()
    ledger = {"episode_id": EPISODE, "script_sha256": script_digest,
              "video_sha256": digest_bytes(video), "claims": [claim], "assets": assets}
    spec_bytes = _json(spec)
    qc = {"kind": "automated", "spec_sha256": digest_bytes(spec_bytes),
          "video_sha256": digest_bytes(video),
          "check": {"duration": 43.0, "words": 5, "integrated_lufs": -14.0,
                    "true_peak_dbfs": -2.0, "stream_problems": [], "warnings": [warning] if warning else [],
                    "speech_differences": [], "speech_error": None,
                    "sources": ["designed card"]}}
    pending = {"episode_id": EPISODE, "video_sha256": digest_bytes(video),
               "script_sha256": script_digest, "pending_claims": ["c1"]}
    for name, value in (("script.json", script), ("short.yaml", spec),
                        ("evidence.json", ledger), ("work/manifest.json", manifest),
                        ("qc.json", qc), ("evidence_pending.json", pending)):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_json(value))
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-q", "-m", "candidate", cwd=repo)
    commit = _run("git", "rev-parse", "HEAD", cwd=repo)
    archive = root / "media.tar"
    with tarfile.open(archive, "w") as bundle:
        for name, data in media.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
        for member in extra_members or []:
            bundle.addfile(member, io.BytesIO(b"x" * member.size) if member.isreg() else None)
    return repo, archive, commit


class CandidateBoundaryTests(unittest.TestCase):
    def test_valid_immutable_snapshot_and_asset_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, archive, commit = candidate_fixture(root)
            candidate = load_candidate(repo, archive, root / "snapshot", EPISODE, commit, QA_ID)
            self.assertEqual(candidate.hashes["video"], candidate.manifest["video_sha256"])
            self.assertEqual(len(candidate.asset_paths), 3)
            candidate.recheck()
            (candidate.video_path).write_bytes(b"changed")
            with self.assertRaisesRegex(QaHold, "changed after QA started"):
                candidate.recheck()

    def test_missing_producer_identity_holds_before_media(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, archive, commit = candidate_fixture(root, producer=None)
            with self.assertRaisesRegex(QaHold, "production_agent_id"):
                load_candidate(repo, archive, root / "snapshot", EPISODE, commit, QA_ID)

    def test_unrelated_producer_identity_holds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, archive, commit = candidate_fixture(root, producer="agent:producer/studio-123")
            with self.assertRaisesRegex(QaHold, "production_agent_id"):
                load_candidate(repo, archive, root / "snapshot", EPISODE, commit, QA_ID)

    def test_internal_git_provenance_is_rechecked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, archive, commit = candidate_fixture(root, extra_internal_provenance=True)
            candidate = load_candidate(repo, archive, root / "snapshot", EPISODE, commit, QA_ID)
            name = "pipeline/src/ytc/test_provenance.txt"
            self.assertIn(name, candidate.internal_references)
            (candidate.root / name).write_text("tampered after observation", encoding="utf-8")
            with self.assertRaisesRegex(QaHold, "internal provenance"):
                candidate.recheck()

    def test_discovery_skips_malformed_and_legacy_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, _, _ = candidate_fixture(root)
            malformed = repo / "content" / "episodes" / "ep001"
            (malformed / "work").mkdir(parents=True)
            (malformed / "evidence_pending.json").write_bytes(b'{"episode_id":')
            (malformed / "work" / "manifest.json").write_bytes(b'{}')
            legacy = repo / "content" / "episodes" / "ep002"
            (legacy / "work").mkdir(parents=True)
            video_hash = digest_bytes(b"legacy video")
            (legacy / "evidence_pending.json").write_bytes(_json(
                {"episode_id": "ep002", "video_sha256": video_hash}))
            (legacy / "work" / "manifest.json").write_bytes(_json(
                {"id": "ep002", "video_sha256": video_hash}))
            _run("git", "add", ".", cwd=repo)
            _run("git", "commit", "-q", "-m", "malformed candidate", cwd=repo)
            commit = _run("git", "rev-parse", "HEAD", cwd=repo)
            self.assertEqual(discover_pending(repo, commit), [
                {"episode_id": EPISODE, "video_sha256": digest_bytes(
                    b"synthetic MP4 bytes for candidate boundary only"), "source_commit": commit}])

    def test_release_discovery_skips_committed_editorial_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, _, _ = candidate_fixture(root)
            (repo / "content" / "episodes" / EPISODE / "editorial-lock.json").write_bytes(
                _json({"episode_id": EPISODE, "reason": "curated draft"}))
            _run("git", "add", ".", cwd=repo)
            _run("git", "commit", "-q", "-m", "lock candidate", cwd=repo)
            commit = _run("git", "rev-parse", "HEAD", cwd=repo)
            self.assertEqual(len(discover_pending(repo, commit)), 1)
            self.assertEqual(discover_pending(repo, commit, unlocked_only=True), [])

    def test_duplicate_tar_member_holds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = tarfile.TarInfo(f"content/episodes/{EPISODE}/{EPISODE}.mp4")
            duplicate.size = 1
            repo, archive, commit = candidate_fixture(root, extra_members=[duplicate])
            with self.assertRaisesRegex(QaHold, "duplicate member"):
                load_candidate(repo, archive, root / "snapshot", EPISODE, commit, QA_ID)

    def test_tar_symlink_and_traversal_hold(self) -> None:
        for member in (tarfile.TarInfo(f"content/episodes/{EPISODE}/link"),
                       tarfile.TarInfo("../escape")):
            with self.subTest(member=member.name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                if member.name.endswith("link"):
                    member.type = tarfile.SYMTYPE
                    member.linkname = "../../outside"
                else:
                    member.size = 1
                repo, archive, commit = candidate_fixture(root, extra_members=[member])
                with self.assertRaises(QaHold):
                    load_candidate(repo, archive, root / "snapshot", EPISODE, commit, QA_ID)
                self.assertFalse((root / "escape").exists())

    def test_asset_digest_mismatch_and_hard_qc_hold(self) -> None:
        for altered in ("asset", "qc"):
            with self.subTest(altered=altered), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                override = ({f"content/episodes/{EPISODE}/work/narration.wav": b"changed voice"}
                            if altered == "asset" else None)
                warning = "duration 40 s is outside 42-62 s" if altered == "qc" else None
                repo, archive, commit = candidate_fixture(root, warning=warning,
                                                          media_override=override)
                if altered == "asset":
                    # Change the archived body after the manifest committed its original hash.
                    with tarfile.open(archive, "w") as bundle:
                        for name, data in ((f"content/episodes/{EPISODE}/{EPISODE}.mp4",
                                            b"synthetic MP4 bytes for candidate boundary only"),
                                           (f"content/episodes/{EPISODE}/work/narration.wav", b"different"),
                                           (f"content/episodes/{EPISODE}/work/assets/beat01.png",
                                            b"synthetic visual bytes")):
                            info = tarfile.TarInfo(name); info.size = len(data)
                            bundle.addfile(info, io.BytesIO(data))
                with self.assertRaises(QaHold):
                    load_candidate(repo, archive, root / "snapshot", EPISODE, commit, QA_ID)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(QaHold, "duplicate JSON key"):
            unique_json(b'{"claim": 1, "claim": 2}', "fixture")


if __name__ == "__main__":
    unittest.main()
