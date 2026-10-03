"""Fetch and retain real source and rights observations with bounded HTTPS.

The signed receipt binds a readable snapshot. Raw HTTP response bytes are kept
in the private audit directory under their SHA-256, outside release inputs.
"""

from __future__ import annotations

import html
import http.client
import ipaddress
import re
import socket
import ssl
import subprocess
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from .common import QaHold, digest_bytes, require, utc_now, write_bytes_new, write_json_new


MAX_RESPONSE_BYTES = 12 * 1024 * 1024
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 4


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript", "svg"):
            self.hidden += 1
        if tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3"):
            self.chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript", "svg") and self.hidden:
            self.hidden -= 1
        if tag in ("p", "div", "li", "tr", "h1", "h2", "h3"):
            self.chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.chunks.append(data)


def _safe_url(url: str) -> tuple[str, str]:
    require(isinstance(url, str) and len(url) < 4096 and
            not any(ord(character) < 32 for character in url), "source URL is malformed")
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError as exc:
        raise QaHold("source URL has an invalid port") from exc
    require(parts.scheme == "https" and parts.hostname is not None and
            parts.username is None and parts.password is None and port in (None, 443),
            "source URL must be public HTTPS on port 443")
    host = parts.hostname
    require(host not in ("localhost", "localhost.localdomain") and
            not host.endswith((".local", ".internal")), "source URL names a local host")
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise QaHold("source host cannot be resolved") from exc
    ips = {result[4][0] for result in addresses}
    require(bool(ips) and all(ipaddress.ip_address(ip).is_global for ip in ips),
            "source URL resolves to a private or reserved address")
    target = parts.path or "/"
    if parts.query:
        target += "?" + parts.query
    return host, target


class _PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host: str, ip: str) -> None:
        super().__init__(host, timeout=20, context=ssl.create_default_context())
        self._ip = ip

    def connect(self) -> None:
        sock = socket.create_connection((self._ip, 443), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _request_once(url: str) -> tuple[int, dict[str, str], bytes]:
    host, target = _safe_url(url)
    addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    ips = {result[4][0] for result in addresses}
    require(bool(ips) and all(ipaddress.ip_address(ip).is_global for ip in ips),
            "source host changed to a private address")
    connection = _PinnedHTTPS(host, sorted(ips)[0])
    try:
        connection.request("GET", target, headers={"Host": host, "User-Agent": "MoolKathaTrustedQA/1.0",
                                                   "Accept": "text/html,text/plain,application/pdf,image/*",
                                                   "Accept-Encoding": "identity"})
        response = connection.getresponse()
        headers = {key.lower(): value for key, value in response.getheaders()}
        require(headers.get("content-encoding", "identity").lower() == "identity",
                "source response used unsupported content encoding")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        require(len(raw) <= MAX_RESPONSE_BYTES, "source response is oversized")
        return response.status, headers, raw
    except (OSError, TimeoutError, http.client.HTTPException) as exc:
        raise QaHold("source HTTPS fetch failed") from exc
    finally:
        connection.close()


def _readable(raw: bytes, content_type: str) -> str:
    mime = content_type.split(";", 1)[0].strip().lower()
    if mime in ("text/html", "application/xhtml+xml"):
        charset = re.search(r"charset=([A-Za-z0-9_-]+)", content_type, re.I)
        encoding = charset[1] if charset else "utf-8"
        try:
            document = raw.decode(encoding, errors="strict")
        except (UnicodeError, LookupError) as exc:
            raise QaHold("source HTML could not be decoded") from exc
        parser = _VisibleText()
        parser.feed(document)
        text = " ".join(parser.chunks)
    elif mime in ("text/plain", "text/markdown", "application/json"):
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise QaHold("source text is not UTF-8") from exc
    elif mime == "application/pdf":
        try:
            result = subprocess.run(["pdftotext", "-layout", "-", "-"], input=raw,
                                    capture_output=True, timeout=40, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise QaHold("source PDF text extraction is unavailable") from exc
        require(result.returncode == 0, "source PDF text extraction failed")
        text = result.stdout.decode("utf-8", errors="replace")
    elif mime in ("image/png", "image/jpeg", "image/webp", "image/tiff"):
        try:
            result = subprocess.run(["tesseract", "stdin", "stdout", "-l", "hin+eng"],
                                    input=raw, capture_output=True, timeout=40, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise QaHold("source image OCR is unavailable") from exc
        require(result.returncode == 0, "source image OCR failed")
        text = result.stdout.decode("utf-8", errors="replace")
    else:
        raise QaHold(f"source content type {mime or '(missing)'} cannot be inspected")
    normalized = " ".join(html.unescape(text).split())
    encoded = normalized.encode("utf-8")
    require(40 <= len(encoded) <= MAX_SNAPSHOT_BYTES, "source snapshot is empty or oversized")
    return normalized


@dataclass(frozen=True)
class FetchObservation:
    url: str
    fetched_at: str
    http_status: int
    response_sha256: str
    response_ref: str
    snapshot_ref: str
    snapshot_sha256: str
    text: str
    final_url: str
    content_type: str

    def source_record(self, excerpt: str, printed_label: str) -> dict:
        require(isinstance(excerpt, str) and len(excerpt.strip()) >= 3 and excerpt in self.text,
                "source excerpt is absent from fetched snapshot")
        require(isinstance(printed_label, str) and printed_label in self.text,
                "printed source label is absent from fetched snapshot")
        return {"url": self.url, "fetched_at": self.fetched_at, "http_status": self.http_status,
                "response_sha256": self.response_sha256, "response_ref": self.response_ref,
                "snapshot_ref": self.snapshot_ref,
                "snapshot_sha256": self.snapshot_sha256, "excerpt": excerpt,
                "printed_verse_label": printed_label}

    def rights_record(self, license_excerpt: str) -> dict:
        require(isinstance(license_excerpt, str) and len(license_excerpt.strip()) >= 10 and
                license_excerpt in self.text, "license excerpt is absent from fetched snapshot")
        return {"url": self.url, "fetched_at": self.fetched_at, "http_status": self.http_status,
                "response_sha256": self.response_sha256, "response_ref": self.response_ref,
                "snapshot_ref": self.snapshot_ref,
                "snapshot_sha256": self.snapshot_sha256, "license_excerpt": license_excerpt}


def fetch_observation(url: str, episode_dir: Path, private_audit_dir: Path) -> FetchObservation:
    """Fetch a page now; never accept a model-authored URL, body, or status."""
    original = url
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        status, headers, raw = _request_once(current)
        if status in (301, 302, 303, 307, 308):
            location = headers.get("location")
            require(bool(location), "source redirect has no location")
            current = urljoin(current, location)
            _safe_url(current)
            continue
        require(status == 200, f"source fetch returned HTTP {status}")
        fetched_at = utc_now()
        response_sha = digest_bytes(raw)
        response_ref = f"agent-qa-responses/{response_sha}.bin"
        response_path = episode_dir / response_ref
        if response_path.exists():
            require(response_path.read_bytes() == raw, "source response digest collision")
        else:
            write_bytes_new(response_path, raw)
        text = _readable(raw, headers.get("content-type", ""))
        snapshot = (text + "\n").encode("utf-8")
        snapshot_sha = digest_bytes(snapshot)
        ref = f"agent-qa-snapshots/{snapshot_sha}.txt"
        snapshot_path = episode_dir / ref
        if snapshot_path.exists():
            require(snapshot_path.read_bytes() == snapshot, "source snapshot digest collision")
        else:
            write_bytes_new(snapshot_path, snapshot)
        raw_path = private_audit_dir / "http" / f"{response_sha}.bin"
        if raw_path.exists():
            require(raw_path.read_bytes() == raw, "raw HTTP response digest collision")
        else:
            write_bytes_new(raw_path, raw)
        receipt_path = private_audit_dir / "http" / f"{response_sha}.receipt.json"
        if not receipt_path.exists():
            write_json_new(receipt_path, {"url": original, "final_url": current, "fetched_at": fetched_at,
                                          "http_status": status, "content_type": headers.get("content-type", ""),
                                          "response_sha256": response_sha, "response_ref": response_ref,
                                          "snapshot_sha256": snapshot_sha})
        return FetchObservation(original, fetched_at, status, response_sha, response_ref, ref,
                                snapshot_sha, text, current, headers.get("content-type", ""))
    raise QaHold("source exceeded HTTPS redirect limit")


def text_window(snapshot: str, needles: list[str], max_chars: int = 12000) -> str:
    """Give the reviewer bounded page context around a claimed passage or grant."""
    positions = [snapshot.casefold().find(needle.casefold()) for needle in needles
                 if isinstance(needle, str) and len(needle.strip()) >= 3]
    positions = [position for position in positions if position >= 0]
    if not positions:
        return snapshot[:max_chars]
    chunks: list[str] = []
    used: list[tuple[int, int]] = []
    for position in positions[:8]:
        start = max(0, position - 1500)
        end = min(len(snapshot), position + 4500)
        if any(start < past_end and end > past_start for past_start, past_end in used):
            continue
        used.append((start, end))
        chunks.append(snapshot[start:end])
    return "\n[…]\n".join(chunks)[:max_chars]
