"""Build source and rights evidence from current pages and exact used assets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

from .candidate import Candidate
from .common import QaHold, digest_file, require, write_json_new
from .fetch import (AssetDownloadObservation, FetchObservation, _readable_and_links,
                    fetch_exact_asset, fetch_observation, text_window)


OBJECT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}\Z")
INTERNAL_RECORDS_REQUIRED = {
    "visual": "a control-verifiable visual creation or licence record",
    "voice": "a control-verifiable voice take and provider licence record",
    "font": "an exact-font licence and file provenance record",
    "music": "a control-verifiable composition or music licence record",
    "sfx": "a control-verifiable sound creation or licence record",
    "animation": "dedicated generated-animation provenance and frame validation",
}


def _page_unchanged(candidate: Candidate, page: FetchObservation) -> None:
    response = candidate.episode_dir / page.response_ref
    snapshot = candidate.episode_dir / page.snapshot_ref
    require(digest_file(response) == page.response_sha256 and
            digest_file(snapshot) == page.snapshot_sha256 and
            snapshot.read_bytes() == (page.text + "\n").encode("utf-8"),
            "asset origin or rights page changed after observation")
    observed_text, raw_links = _readable_and_links(response.read_bytes(), page.content_type)
    observed_links = tuple(urljoin(page.final_url, link) for link in raw_links
                           if urlsplit(urljoin(page.final_url, link)).scheme == "https")
    require(observed_text == page.text and observed_links == page.visible_links,
            "asset origin or rights page extraction changed after observation")


@dataclass(frozen=True)
class VerifiedHttpAssetRights:
    origin_page: FetchObservation
    rights_page: FetchObservation
    object_id: str
    exact_download: AssetDownloadObservation | None

    def signed_origin_proof(self, candidate: Candidate, asset: dict[str, Any]) -> dict[str, Any]:
        """Bind the fetched origin and exact-file relation into the review."""
        self.recheck(candidate, asset)
        origin = self.origin_page
        exact_file: dict[str, Any] = {"method": "visible_sha256", "sha256": asset["sha256"]}
        if self.exact_download is not None:
            exact_file = {"method": "official_download", "sha256": asset["sha256"],
                          "official_asset_url": self.exact_download.url,
                          "response_ref": self.exact_download.response_ref}
        return {"source_object_id": self.object_id, "rights_basis": asset["rights_basis"],
                "origin_fetch": {"url": origin.url, "fetched_at": origin.fetched_at,
                                 "http_status": origin.http_status,
                                 "final_url": origin.final_url,
                                 "content_type": origin.content_type,
                                 "response_sha256": origin.response_sha256,
                                 "response_ref": origin.response_ref,
                                 "snapshot_ref": origin.snapshot_ref,
                                 "snapshot_sha256": origin.snapshot_sha256,
                                 "visible_links": list(origin.visible_links)},
                "exact_file": exact_file}

    def recheck(self, candidate: Candidate, asset: dict[str, Any]) -> None:
        """Revalidate a specific licence and origin proof before assembly."""
        identity = asset["id"]
        expected_sha = asset["sha256"]
        origin_url = asset.get("origin")
        rights_url = asset.get("rights_url")
        require(self.origin_page.url == origin_url and self.rights_page.url == rights_url and
                isinstance(origin_url, str) and origin_url.startswith("https://") and
                isinstance(rights_url, str) and rights_url.startswith("https://"),
                f"asset {identity}: exact HTTP origin and rights pages are required")
        require(isinstance(self.object_id, str) and OBJECT_ID.fullmatch(self.object_id) is not None and
                asset.get("source_object_id") == self.object_id,
                f"asset {identity}: object-specific source identifier is missing")
        object_id = self.object_id.casefold()
        require(isinstance(self.origin_page.visible_links, tuple) and
                len(self.origin_page.visible_links) <= 4096 and
                all(isinstance(link, str) and len(link) <= 4096 and
                    not any(ord(character) < 32 for character in link)
                    for link in self.origin_page.visible_links),
                f"asset {identity}: origin page has unbounded visible links")
        parts = urlsplit(self.origin_page.final_url)
        origin_location = unquote(parts.path + "?" + parts.query).casefold()
        require(object_id in origin_location and
                object_id in self.origin_page.text.casefold() and
                object_id in self.rights_page.text.casefold(),
                f"asset {identity}: origin and rights pages do not identify the same object")
        license_name = asset.get("license")
        require(isinstance(license_name, str) and
                license_name.casefold() in self.rights_page.text.casefold(),
                f"asset {identity}: exact licence is absent from the object-specific rights page")
        official_url = asset.get("official_asset_url")
        if official_url is None:
            require(self.exact_download is None and
                    expected_sha.casefold() in self.origin_page.text.casefold(),
                    f"asset {identity}: origin page does not fingerprint the exact used file")
        else:
            linked = (isinstance(official_url, str) and official_url.startswith("https://") and
                      (official_url in self.origin_page.visible_links or
                       official_url in self.origin_page.text))
            require(linked,
                    f"asset {identity}: official asset URL is absent from the visible origin page")
            require(self.exact_download is not None and self.exact_download.url == official_url and
                    self.exact_download.response_sha256 == expected_sha and
                    digest_file(candidate.episode_dir / self.exact_download.response_ref) == expected_sha,
                    f"asset {identity}: official URL bytes differ from the exact used file")
        specific_link = (self.rights_page.url == self.origin_page.url or
                         origin_url.casefold() in self.rights_page.text.casefold() or
                         expected_sha.casefold() in self.rights_page.text.casefold() or
                         isinstance(official_url, str) and
                         official_url.casefold() in self.rights_page.text.casefold())
        require(specific_link,
                f"asset {identity}: rights page is a generic licence page unrelated to the used file")
        require(digest_file(candidate.asset_paths[identity]) == expected_sha,
                f"asset {identity}: used file changed during rights review")
        _page_unchanged(candidate, self.origin_page)
        _page_unchanged(candidate, self.rights_page)


@dataclass(frozen=True)
class ObservationSet:
    claim_pages: dict[str, dict[str, FetchObservation]]
    asset_rights: dict[str, VerifiedHttpAssetRights]
    review_packet: dict[str, Any]


def collect_observations(candidate: Candidate, private_audit_dir: Path) -> ObservationSet:
    """Fetch each distinct page once and hold rights lacking exact object proof."""
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
                                      "http_status": page.http_status,
                                      "response_sha256": page.response_sha256,
                                      "response_ref": page.response_ref,
                                      "snapshot_ref": page.snapshot_ref,
                                      "snapshot_sha256": page.snapshot_sha256},
                            "fetched_context": text_window(page.text, [source["excerpt"],
                                                              source["printed_verse_label"]])}
        claim_pages[identity] = pages
        claim_packet.append({"id": identity, "claim": claim.get("claim"), "hindi": claim.get("hindi"),
                             "tradition": claim.get("tradition"), "variant_caveat": claim.get("variant_caveat"),
                             "correspondence": claim.get("correspondence"), **visible})

    asset_rights: dict[str, VerifiedHttpAssetRights] = {}
    asset_packet: list[dict[str, Any]] = []
    for identity, asset in candidate.assets.items():
        exact_path = candidate.asset_paths[identity]
        require(digest_file(exact_path) == asset["sha256"],
                f"used asset {identity} changed during rights check")
        origin_url = asset.get("origin")
        rights_url = asset.get("rights_url")
        if (not isinstance(origin_url, str) or not origin_url.startswith("https://") or
                not isinstance(rights_url, str) or not rights_url.startswith("https://")):
            requirement = INTERNAL_RECORDS_REQUIRED.get(asset["role"], "an independently verifiable record")
            raise QaHold(f"asset {identity}: internal rights need {requirement}; producer notes cannot prove rights")
        origin_page = observe(origin_url)
        rights_page = observe(rights_url)
        official_url = asset.get("official_asset_url")
        require(official_url is None or
                isinstance(official_url, str) and official_url.startswith("https://"),
                f"asset {identity}: official asset URL is malformed")
        download = (fetch_exact_asset(official_url, asset["sha256"], candidate.episode_dir,
                                      private_audit_dir) if official_url is not None else None)
        evidence = VerifiedHttpAssetRights(origin_page, rights_page,
                                           asset.get("source_object_id"), download)
        evidence.recheck(candidate, asset)
        asset_rights[identity] = evidence
        asset_packet.append({"id": identity, "role": asset["role"], "file": asset["file"],
                             "sha256": asset["sha256"], "origin": origin_url,
                             "creator": asset.get("creator"), "license": asset.get("license"),
                             "rights_basis": asset["rights_basis"], "credit": asset.get("credit"),
                             "commercial_use_claimed": asset.get("commercial_use"),
                             "derivatives_allowed_claimed": asset.get("derivatives_allowed"),
                             "rights_evidence": {
                                 "kind": "object_specific_http",
                                 "source_object_id": evidence.object_id,
                                 "origin_page": {"url": origin_page.url,
                                                 "final_url": origin_page.final_url,
                                                 "response_sha256": origin_page.response_sha256,
                                                 "snapshot_sha256": origin_page.snapshot_sha256,
                                                 "fetched_context": text_window(
                                                     origin_page.text, [evidence.object_id,
                                                                        asset["sha256"]])},
                                 "rights_page": {"url": rights_page.url,
                                                 "final_url": rights_page.final_url,
                                                 "response_sha256": rights_page.response_sha256,
                                                 "snapshot_sha256": rights_page.snapshot_sha256,
                                                 "fetched_context": text_window(
                                                     rights_page.text, [evidence.object_id,
                                                                        asset["license"]])},
                                 "exact_file": {"method": "official_download" if download else "visible_sha256",
                                                "sha256": asset["sha256"],
                                                "official_asset_url": official_url}}})
    candidate.recheck()
    packet = {"episode_id": candidate.episode_id, "video_sha256": candidate.hashes["video"],
              "claims": claim_packet, "assets": asset_packet}
    write_json_new(private_audit_dir / "observations.json", packet)
    return ObservationSet(claim_pages, asset_rights, packet)
