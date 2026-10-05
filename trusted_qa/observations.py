"""Build source and rights evidence from real current pages and exact assets."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .candidate import Candidate
from .common import QaHold, digest_file, require, utc_now, write_json_new
from .fetch import (FetchObservation, FileDownload, _request_identity, download_file, fetch_observation,
                    text_window, visible_links)


@dataclass(frozen=True)
class ObservationSet:
    claim_pages: dict[str, dict[str, FetchObservation]]
    asset_rights: dict[str, FetchObservation | dict[str, Any]]
    review_packet: dict[str, Any]
    asset_origins: dict[str, OriginProof] = field(default_factory=dict)


@dataclass(frozen=True)
class OriginProof:
    """A public object page and the official file whose bytes equal the used asset."""
    page: FetchObservation
    links: list[str]
    download: FileDownload
    source_object_id: str
    rights_basis: str

    def record(self) -> dict[str, Any]:
        page = self.page
        return {"source_object_id": self.source_object_id, "rights_basis": self.rights_basis,
                "origin_fetch": {"url": page.url, "fetched_at": page.fetched_at,
                                 "http_status": page.http_status, "final_url": page.final_url,
                                 "content_type": page.content_type,
                                 "response_sha256": page.response_sha256, "response_ref": page.response_ref,
                                 "snapshot_ref": page.snapshot_ref, "snapshot_sha256": page.snapshot_sha256,
                                 "visible_links": list(self.links)},
                "exact_file": {"method": "official_download", "sha256": self.download.response_sha256,
                               "official_asset_url": self.download.url,
                               "response_ref": self.download.response_ref}}


def origin_proof(identity: str, asset: dict[str, Any], page: FetchObservation, rights_text: str,
                 episode_dir: Path, private_audit_dir: Path) -> OriginProof:
    """Bind an HTTPS asset to its object page and an official download of the exact used bytes."""
    object_id = asset.get("source_object_id")
    require(isinstance(object_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{5,127}", object_id),
            f"asset {identity} has no source object ID")
    final = urlsplit(page.final_url)
    require(page.final_url.startswith("https://") and
            object_id.casefold() in unquote(final.path + "?" + final.query).casefold() and
            object_id.casefold() in page.text.casefold(),
            f"asset {identity}: object page does not identify object {object_id}")
    license_name = asset.get("license")
    require(object_id.casefold() in rights_text.casefold() and isinstance(license_name, str) and
            license_name.casefold() in rights_text.casefold(),
            f"asset {identity}: rights page does not name the object and its licence")
    official = asset.get("official_asset_url")
    links = visible_links(page, episode_dir)
    require(isinstance(official, str) and official.startswith("https://") and
            (official in links or official in page.text),
            f"asset {identity}: object page does not link the official file")
    require(asset.get("rights_url") == asset.get("origin") or
            any(isinstance(value, str) and value.casefold() in rights_text.casefold()
                for value in (asset.get("origin"), asset.get("sha256"), official)),
            f"asset {identity}: rights page is not linked to this exact object")
    download = download_file(official, episode_dir, private_audit_dir)
    require(download.response_sha256 == asset["sha256"],
            f"asset {identity}: official file bytes differ from the used asset")
    return OriginProof(page, links, download, object_id, asset["rights_basis"])


def require_distinct_claim_pages(identity: str, pages: dict[str, FetchObservation]) -> None:
    """Different ledger URLs must resolve to different visible source pages."""
    require(isinstance(pages, dict) and set(pages) == {"primary", "corroboration"} and
            all(isinstance(page, FetchObservation) for page in pages.values()),
            f"claim {identity}: primary and corroboration pages are incomplete")
    primary, corroboration = pages["primary"], pages["corroboration"]
    require(_request_identity(primary.final_url) != _request_identity(corroboration.final_url),
            f"claim {identity}: primary and corroboration resolve to the same final page")
    require(primary.snapshot_sha256 != corroboration.snapshot_sha256,
            f"claim {identity}: primary and corroboration have identical visible text")


def _internal_text(candidate: Candidate, reference: str) -> str:
    from .candidate import _asset_file
    require(reference.startswith("internal:"), "internal provenance reference is malformed")
    path = _asset_file(candidate.root, candidate.episode_id, reference[len("internal:"):])
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 5_000_000,
            "internal provenance file is missing or oversized")
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8", errors="strict")[:12000]
    except UnicodeDecodeError:
        return f"Binary provenance file, SHA-256 {digest_file(path)}, {len(raw)} bytes."


def collect_observations(candidate: Candidate, private_audit_dir: Path) -> ObservationSet:
    """Fetch each distinct actual page once and build per-claim/asset context."""
    candidate.recheck()
    cache: dict[str, FetchObservation] = {}

    def observe(url: str) -> FetchObservation:
        if url not in cache:
            cache[url] = fetch_observation(url, candidate.episode_dir, private_audit_dir)
        return cache[url]

    claim_pages: dict[str, dict[str, FetchObservation]] = {}
    claim_packet: list[dict[str, Any]] = []
    for claim in candidate.ledger["claims"]:
        identity = claim["id"]
        pages: dict[str, FetchObservation] = {}
        visible: dict[str, Any] = {}
        for key in ("primary", "corroboration"):
            source = claim[key]
            page = observe(source["url"])
            require(source["excerpt"] in page.text,
                    f"claim {identity} {key} ledger passage is absent from the fetched page")
            require(source["printed_verse_label"] in page.text,
                    f"claim {identity} {key} printed label is absent from the fetched page")
            pages[key] = page
            visible[key] = {"ledger": source,
                            "fetch": {"url": page.url, "fetched_at": page.fetched_at,
                                      "final_url": page.final_url,
                                      "http_status": page.http_status,
                                      "response_sha256": page.response_sha256,
                                      "response_ref": page.response_ref,
                                      "snapshot_ref": page.snapshot_ref,
                                      "snapshot_sha256": page.snapshot_sha256},
                            "fetched_context": text_window(page.text, [source["excerpt"],
                                                              source["printed_verse_label"]])}
        require_distinct_claim_pages(identity, pages)
        claim_pages[identity] = pages
        claim_packet.append({"id": identity, "claim": claim.get("claim"), "hindi": claim.get("hindi"),
                             "tradition": claim.get("tradition"), "variant_caveat": claim.get("variant_caveat"),
                             "correspondence": claim.get("correspondence"), **visible})

    asset_rights: dict[str, FetchObservation | dict[str, Any]] = {}
    asset_origins: dict[str, OriginProof] = {}
    asset_packet: list[dict[str, Any]] = []
    for identity, asset in candidate.assets.items():
        exact_path = candidate.asset_paths[identity]
        require(digest_file(exact_path) == asset["sha256"], f"used asset {identity} changed during rights check")
        rights_url = asset["rights_url"]
        if rights_url.startswith("https://"):
            page = observe(rights_url)
            asset_rights[identity] = page
            evidence = {"kind": "http", "url": rights_url,
                        "fetched_at": page.fetched_at, "response_sha256": page.response_sha256,
                        "response_ref": page.response_ref,
                        "snapshot_ref": page.snapshot_ref, "snapshot_sha256": page.snapshot_sha256,
                        "fetched_context": text_window(page.text, [asset.get("license", ""), "ownership",
                                                                   "commercial", "derivative", "credit",
                                                                   "CC0", "public domain", "Creative Commons"])}
        elif rights_url.startswith("internal:"):
            require(len(asset["rights_basis"].strip()) >= 20,
                    f"asset {identity} internal provenance explanation is too short")
            provenance = {"origin": asset["origin"], "checked_at": utc_now(),
                          "content_sha256": asset["sha256"],
                          "provenance_excerpt": asset["rights_basis"]}
            asset_rights[identity] = provenance
            evidence = {"kind": "internal", "rights_reference": rights_url,
                        "rights_reference_text": _internal_text(candidate, rights_url)[:8000],
                        "origin_reference_text": (_internal_text(candidate, asset["origin"])[:8000]
                                                  if str(asset.get("origin", "")).startswith("internal:")
                                                  else None),
                        "provenance": provenance}
        else:
            raise QaHold(f"asset {identity} has no inspectable rights evidence")
        if str(asset.get("origin", "")).startswith("https://"):
            rights_page = asset_rights[identity]
            require(isinstance(rights_page, FetchObservation),
                    f"asset {identity}: a public object needs a fetched rights page")
            asset_origins[identity] = origin_proof(identity, asset, observe(asset["origin"]), rights_page.text,
                                                   candidate.episode_dir, private_audit_dir)
        asset_packet.append({"id": identity, "role": asset["role"], "file": asset["file"],
                             "sha256": asset["sha256"], "origin": asset.get("origin"),
                             "creator": asset.get("creator"), "license": asset.get("license"),
                             "rights_basis": asset["rights_basis"], "credit": asset.get("credit"),
                             "commercial_use_claimed": asset.get("commercial_use"),
                             "derivatives_allowed_claimed": asset.get("derivatives_allowed"),
                             "rights_evidence": evidence})
    candidate.recheck()
    packet = {"episode_id": candidate.episode_id, "video_sha256": candidate.hashes["video"],
              "claims": claim_packet, "assets": asset_packet}
    write_json_new(private_audit_dir / "observations.json", packet)
    return ObservationSet(claim_pages, asset_rights, packet, asset_origins)
