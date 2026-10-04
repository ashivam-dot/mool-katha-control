"""Append a sanitized analytics snapshot to the private control data branch."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from trusted_release.readback import _platform_link


BUFFER_CHANNELS = {"youtube": "6ac06721ea19ca0bde5dbe63",
                   "instagram": "6ac0687aea19ca0bde5dc879"}
BUFFER_ORG_ID = "6ac066851cde9b9edca25c7b"
YOUTUBE_ID = "UCdqxVnoHDWXgA2ZVJWkSu8w"
METRICS = ("views", "reach", "reactions", "comments", "shares", "saves", "follows")
IST = ZoneInfo("Asia/Kolkata")
UTC = timezone.utc


class PersistError(ValueError):
    """The report cannot safely become a durable learning snapshot."""


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise PersistError("monitor report has duplicate keys")
        result[key] = value
    return result


def _object(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
        raise PersistError("monitor report is missing, linked, or oversized")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    except (OSError, UnicodeError, ValueError):
        raise PersistError("monitor report is malformed") from None
    if not isinstance(value, dict):
        raise PersistError("monitor report is not an object")
    return value


def _time(value: object) -> str:
    if not isinstance(value, str) or len(value) > 64:
        raise PersistError("analytics timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise PersistError("analytics timestamp is invalid") from None
    if parsed.utcoffset() is None:
        raise PersistError("analytics timestamp has no time zone")
    return parsed.astimezone(UTC).isoformat()


def _date(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise PersistError("analytics date is invalid")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        raise PersistError("analytics date is invalid") from None


def _count(value: object, *, optional: bool = False) -> int | float | None:
    if value is None and optional:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise PersistError("analytics metric is invalid")
    return value


def _integer(value: object) -> int:
    if type(value) is not int or value < 0:
        raise PersistError("analytics count is invalid")
    return value


def _post(row: object, service: str) -> dict:
    if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,128}", row["id"]):
        raise PersistError("analytics post ID is invalid")
    link = row.get("external_link")
    if link is not None and not _platform_link(link, service):
        raise PersistError("analytics public link is invalid")
    values = row.get("metrics")
    if not isinstance(values, dict) or any(key not in METRICS for key in values):
        raise PersistError("analytics post metrics are invalid")
    updated = row.get("metrics_updated_at")
    return {"id": row["id"], "sent_at": _time(row.get("sent_at")),
            "public_link": link, "metrics_updated_at": _time(updated) if updated else None,
            "metrics": {kind: _count(value) for kind, value in values.items()}}


def _buffer(source: object) -> dict:
    if not isinstance(source, dict) or source.get("organization_id") != BUFFER_ORG_ID:
        raise PersistError("analytics Buffer organization is invalid")
    channels = source.get("channels")
    if not isinstance(channels, dict):
        raise PersistError("analytics Buffer channels are invalid")
    result = {}
    for service, channel_id in BUFFER_CHANNELS.items():
        state = channels.get(service)
        if (not isinstance(state, dict) or state.get("id") != channel_id
                or state.get("service") != service):
            raise PersistError("analytics Buffer destination is invalid")
        metrics = state.get("metrics_30d")
        posts = state.get("posts")
        counts = state.get("counts")
        if (not isinstance(metrics, dict) or not isinstance(posts, list) or len(posts) > 50
                or not isinstance(counts, dict)):
            raise PersistError("analytics Buffer metrics are malformed")
        sent_posts = _integer(state.get("sent_posts_30d"))
        coverage = {}
        for kind in METRICS:
            item = metrics.get(kind)
            if not isinstance(item, dict):
                raise PersistError("analytics metric coverage is missing")
            reported = _integer(item.get("reported_posts"))
            if reported > sent_posts:
                raise PersistError("analytics metric coverage exceeds sent posts")
            coverage[kind] = {"total": _count(item.get("total"), optional=True),
                              "reported_posts": reported}
        result[service] = {"channel_id": channel_id, "sent_posts_30d": sent_posts,
                           "counts": {name: _integer(counts.get(name))
                                      for name in ("sent", "scheduled", "error", "draft")},
                           "metrics_30d": coverage,
                           "posts": [_post(row, service) for row in posts]}
    return result


def _youtube_owned(source: object) -> dict:
    if not isinstance(source, dict) or not isinstance(source.get("channel"), dict):
        raise PersistError("owned YouTube analytics are malformed")
    channel = source["channel"]
    if channel.get("channel_id") != YOUTUBE_ID:
        raise PersistError("owned YouTube channel is invalid")
    period = source.get("last_28_days")
    if not isinstance(period, dict) or not isinstance(period.get("daily"), list) or len(period["daily"]) > 28:
        raise PersistError("owned YouTube daily analytics are malformed")
    start, end = _date(period.get("start")), _date(period.get("end"))
    daily = []
    seen = set()
    for row in period["daily"]:
        if not isinstance(row, dict):
            raise PersistError("owned YouTube daily row is malformed")
        day = _date(row.get("day"))
        if day in seen or day < start or day > end:
            raise PersistError("owned YouTube daily date is invalid")
        seen.add(day)
        daily.append({"day": day, **{kind: _count(row.get(kind)) for kind in (
            "views", "subscribersGained", "subscribersLost")}})
    return {"channel_id": YOUTUBE_ID,
            "channel": {kind: _count(channel.get(kind), optional=True)
                        for kind in ("subscribers", "views", "videos")},
            "last_28_days": {"start": start, "end": end, "daily": daily}}


def _youtube_public(source: object) -> dict:
    if (not isinstance(source, dict) or source.get("channel_id") != YOUTUBE_ID
            or not isinstance(source.get("recent_videos"), list)
            or len(source["recent_videos"]) > 50):
        raise PersistError("public YouTube analytics are malformed")
    videos = []
    for row in source["recent_videos"]:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{11}", row["id"])):
            raise PersistError("public YouTube video ID is malformed")
        published = row.get("published")
        videos.append({"id": row["id"], "published": _time(published) if published else None,
                       "views": _count(row.get("views"), optional=True)})
    return {"channel_id": YOUTUBE_ID, "recent_videos": videos}


def extract(report: dict, run_key: str) -> dict | None:
    if report.get("schema") != "mool_katha_control_monitor_v1" or not re.fullmatch(
            r"[0-9]{1,20}-[0-9]{1,4}", run_key):
        raise PersistError("monitor report identity is invalid")
    captured = _time(report.get("captured_at"))
    ist_date = _date(report.get("ist_date"))
    if datetime.fromisoformat(captured).astimezone(IST).date().isoformat() != ist_date:
        raise PersistError("monitor IST date differs from capture time")
    sources = report.get("sources")
    if not isinstance(sources, dict):
        raise PersistError("monitor sources are malformed")
    snapshot = {"schema": "mool_katha_control_analytics_snapshot_v1",
                "run_key": run_key, "captured_at": captured, "ist_date": ist_date,
                "sources": {}}
    for name, reader in (("buffer", _buffer), ("youtube_owned", _youtube_owned),
                         ("youtube_public", _youtube_public)):
        source = sources.get(name)
        if isinstance(source, dict) and source.get("status") == "unavailable":
            continue
        if source is not None:
            snapshot["sources"][name] = reader(source)
    return snapshot if snapshot["sources"] else None


def write(snapshot: dict, out_dir: Path) -> Path:
    if out_dir.is_symlink():
        raise PersistError("analytics output directory is linked")
    out_dir.mkdir(parents=True, exist_ok=True)
    day_dir = out_dir / snapshot["ist_date"]
    if day_dir.is_symlink():
        raise PersistError("analytics day directory is linked")
    day_dir.mkdir(exist_ok=True)
    target = day_dir / f"{snapshot['run_key']}.json"
    if target.is_symlink() or target.exists():
        raise PersistError("analytics run snapshot already exists")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=day_dir,
                                         prefix=".snapshot-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump(snapshot, output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, target)
    except FileExistsError:
        raise PersistError("analytics run snapshot already exists") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-key", required=True)
    args = parser.parse_args()
    try:
        snapshot = extract(_object(args.report), args.run_key)
        if snapshot is None:
            print("No usable analytics source in control report; snapshot skipped")
            return 0
        path = write(snapshot, args.out_dir)
        print(f"Sanitized analytics snapshot saved: {path.name}")
        return 0
    except PersistError as exc:
        print(f"Analytics snapshot needs retry: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print("Analytics snapshot needs retry: unexpected failure", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
