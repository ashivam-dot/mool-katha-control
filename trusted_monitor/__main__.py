"""Collect a bounded, secret-free report from exact Mool Katha destinations."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .signed import SignedMonitorError, inspect as inspect_signed
from .public import inspect as inspect_public, load_proofs


REPOSITORY = "ashivam-dot/mool-katha-control"
BUFFER_ORG_ID = "6ac066851cde9b9edca25c7b"
BUFFER_YOUTUBE_ID = "6ac06721ea19ca0bde5dbe63"
BUFFER_INSTAGRAM_ID = "6ac0687aea19ca0bde5dc879"
CLOUDINARY_CLOUD = "mw0oh0v8"
YOUTUBE_ID = "UCdqxVnoHDWXgA2ZVJWkSu8w"
YOUTUBE_HANDLE = "moolkatha"
INSTAGRAM_HANDLE = "moolkatha.hindi"
RELEASE_WORKFLOW = "signed-release.yml"
QA_DISPATCH_WORKFLOW = "dispatch-release-qa.yml"
QA_WORKFLOW = "release-qa.yml"
QA_WORKFLOW_SHA = "73c24433306cf16a74d5dd4efa23910ee4d268d2"
QA_QUIET_HOURS = 10
QA_DISPATCH_HOURS = (0, 6, 12, 18)
QA_DISPATCH_MINUTE = 10
METRICS = ("views", "reach", "reactions", "comments", "shares", "saves", "follows")
UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")


class MonitorError(ValueError):
    pass


def _request(url: str, *, headers: dict[str, str] | None = None,
             payload: dict | None = None, limit: int = 2_000_000) -> bytes:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers or {},
                                     method="POST" if data is not None else "GET")
    with urllib.request.urlopen(request, timeout=25) as response:
        body = response.read(limit + 1)
    if len(body) > limit:
        raise MonitorError("remote response exceeds size limit")
    return body


def _json(url: str, *, headers: dict[str, str] | None = None,
          payload: dict | None = None) -> dict:
    result = json.loads(_request(url, headers=headers, payload=payload))
    if not isinstance(result, dict):
        raise MonitorError("remote JSON is not an object")
    return result


def _time(value: object) -> datetime:
    if not isinstance(value, str):
        raise MonitorError("missing timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MonitorError("invalid timestamp") from exc
    if parsed.utcoffset() is None:
        raise MonitorError("timestamp has no time zone")
    return parsed.astimezone(UTC)


def _count(value: object) -> int | float:
    if type(value) not in (int, float) or value < 0 or value != value or value == float("inf"):
        raise MonitorError("invalid metric")
    return value


def _buffer(query: str, variables: dict, key: str) -> dict:
    data = _json("https://api.buffer.com", headers={"Authorization": f"Bearer {key}",
                                               "Content-Type": "application/json"},
                 payload={"query": query, "variables": variables})
    if data.get("errors") or not isinstance(data.get("data"), dict):
        # Provider error bodies may echo input. Never put them in reports or logs.
        raise MonitorError("Buffer query failed")
    return data["data"]


def _channel(row: object, *, expected_id: str, service: str, handle: str) -> dict:
    if not isinstance(row, dict) or row.get("id") != expected_id or row.get("service") != service:
        raise MonitorError(f"Buffer {service} destination mismatch")
    if any(type(row.get(flag)) is not bool for flag in ("isDisconnected", "isLocked", "isQueuePaused")):
        raise MonitorError(f"Buffer {service} readiness unknown")
    names = [str(row.get(field) or "").strip().lower() for field in ("name", "displayName")]
    if service == "instagram":
        exact = any(re.search(rf"(?<![a-z0-9._])@?{re.escape(handle)}(?![a-z0-9._])", name)
                    for name in names)
    else:
        exact = any(re.sub(r"[^a-z0-9]", "", name) == handle for name in names)
    if not exact:
        raise MonitorError(f"Buffer {service} handle mismatch")
    return {"id": expected_id, "service": service, "handle": handle,
            "disconnected": row["isDisconnected"], "locked": row["isLocked"],
            "queue_paused": row["isQueuePaused"]}


def _posts(key: str, channel_id: str, service: str, now: datetime) -> dict:
    query = """query Posts($input: PostsInput!, $after: String) {
      posts(input: $input, first: 50, after: $after) {
        edges { node { id status dueAt sentAt externalLink channelId channelService
                       error { message } metrics { type value } metricsUpdatedAt
                       assets { ... on VideoAsset { source } } } }
        pageInfo { hasNextPage endCursor } } }"""
    variables = {"input": {"organizationId": BUFFER_ORG_ID, "filter": {
        "channelIds": [channel_id]}}, "after": None}
    rows: list[dict] = []
    seen: set[str] = set()
    for _ in range(100):
        page = _buffer(query, variables, key).get("posts")
        if not isinstance(page, dict) or not isinstance(page.get("edges"), list):
            raise MonitorError("Buffer posts response is malformed")
        for edge in page["edges"]:
            row = edge.get("node") if isinstance(edge, dict) else None
            if not isinstance(row, dict) or row.get("channelId") != channel_id or row.get("channelService") != service:
                raise MonitorError(f"Buffer {service} post destination mismatch")
            post_id = row.get("id")
            if not isinstance(post_id, str) or not post_id or post_id in seen:
                raise MonitorError(f"Buffer {service} post identity is missing or duplicated")
            seen.add(post_id)
            rows.append(row)
        info = page.get("pageInfo")
        if not isinstance(info, dict) or type(info.get("hasNextPage")) is not bool:
            raise MonitorError("Buffer pagination is malformed")
        if not info["hasNextPage"]:
            break
        cursor = info.get("endCursor")
        if not isinstance(cursor, str) or not cursor or cursor == variables["after"]:
            raise MonitorError("Buffer pagination cursor is invalid")
        variables["after"] = cursor
    else:
        raise MonitorError("Buffer posts exceed page limit")

    counts = {"sent": 0, "scheduled": 0, "error": 0, "draft": 0}
    recent_errors: list[dict] = []
    sent: list[dict] = []
    release_rows: list[dict] = []
    for row in rows:
        assets = row.get("assets")
        sources = ([item.get("source") if isinstance(item, dict) else None for item in assets]
                   if isinstance(assets, list) else None)
        release_rows.append({"id": row["id"], "channel_id": row["channelId"],
                             "service": row["channelService"], "status": row.get("status"),
                             "due_at": row.get("dueAt"), "sent_at": row.get("sentAt"),
                             "external_link": row.get("externalLink"), "media_sources": sources})
        status = row.get("status")
        if status not in counts:
            # Buffer uses additional queued statuses. They are scheduled for monitoring.
            status = "scheduled"
        counts[status] += 1
        if status == "error":
            try:
                due = _time(row.get("dueAt"))
            except MonitorError:
                pass
            else:
                if now - timedelta(days=30) <= due <= now + timedelta(days=365):
                    recent_errors.append({"id": row["id"], "due_at": due.isoformat()})
        if status != "sent":
            continue
        sent_at = _time(row.get("sentAt"))
        if sent_at < now - timedelta(days=30) or sent_at > now + timedelta(minutes=5):
            continue
        link = row.get("externalLink")
        if link:
            parsed = urllib.parse.urlsplit(link)
            hosts = ("youtube.com", "www.youtube.com", "youtu.be") if service == "youtube" else ("instagram.com", "www.instagram.com")
            if parsed.scheme != "https" or parsed.hostname not in hosts:
                raise MonitorError(f"Buffer {service} public link destination mismatch")
        metrics: dict[str, int | float] = {}
        for item in row.get("metrics") or []:
            if not isinstance(item, dict):
                raise MonitorError("Buffer metrics are malformed")
            kind = item.get("type")
            if kind in METRICS:
                if kind in metrics:
                    raise MonitorError("Buffer metric is duplicated")
                metrics[kind] = _count(item.get("value"))
        sent.append({"id": row["id"], "sent_at": sent_at.isoformat(), "external_link": link,
                     "metrics_updated_at": row.get("metricsUpdatedAt"), "metrics": metrics})
    totals = {kind: {"total": sum(p["metrics"][kind] for p in sent) if sent and all(kind in p["metrics"] for p in sent) else None,
                     "reported_posts": sum(kind in p["metrics"] for p in sent)} for kind in METRICS}
    return {"counts": counts, "recent_errors": recent_errors[:20], "sent_posts_30d": len(sent),
            "metrics_30d": totals, "posts": sorted(sent, key=lambda p: (p["sent_at"], p["id"]))[-50:],
            "_release_rows": release_rows}


def buffer(now: datetime, env: dict[str, str]) -> dict:
    key = env.get("BUFFER_API_KEY", "")
    youtube_id = env.get("BUFFER_YOUTUBE_CHANNEL_ID", "")
    instagram_id = env.get("BUFFER_INSTAGRAM_CHANNEL_ID", "")
    if not key or youtube_id != BUFFER_YOUTUBE_ID or instagram_id != BUFFER_INSTAGRAM_ID:
        raise MonitorError("Buffer credentials or exact channel IDs are missing")
    if env.get("BUFFER_ORG_ID") != BUFFER_ORG_ID:
        raise MonitorError("Buffer organization ID mismatch")
    query = """query Channels($input: ChannelsInput!) {
      channels(input: $input) { id name displayName service isDisconnected isLocked isQueuePaused } }"""
    channels = _buffer(query, {"input": {"organizationId": BUFFER_ORG_ID}}, key).get("channels")
    if not isinstance(channels, list):
        raise MonitorError("Buffer channels response is malformed")
    result = {"organization_id": BUFFER_ORG_ID, "channels": {}}
    for service, channel_id, handle in (("youtube", youtube_id, YOUTUBE_HANDLE),
                                        ("instagram", instagram_id, INSTAGRAM_HANDLE)):
        matches = [item for item in channels if isinstance(item, dict) and item.get("id") == channel_id]
        if len(matches) != 1:
            raise MonitorError(f"Buffer {service} channel missing or duplicated")
        state = _channel(matches[0], expected_id=channel_id, service=service, handle=handle)
        state.update(_posts(key, channel_id, service, now))
        result["channels"][service] = state
    return result


def cloudinary(env: dict[str, str]) -> dict:
    value = env.get("CLOUDINARY_URL", "")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "cloudinary" or not parsed.hostname or not parsed.username or not parsed.password or parsed.path:
        raise MonitorError("Cloudinary credential is missing or malformed")
    cloud = parsed.hostname
    if cloud != CLOUDINARY_CLOUD:
        raise MonitorError("Cloudinary cloud name mismatch")
    auth = base64.b64encode(f"{urllib.parse.unquote(parsed.username)}:{urllib.parse.unquote(parsed.password)}".encode()).decode()
    usage = _json(f"https://api.cloudinary.com/v1_1/{urllib.parse.quote(cloud, safe='')}/usage",
                  headers={"Authorization": f"Basic {auth}"})
    credits = usage.get("credits")
    if not isinstance(credits, dict):
        raise MonitorError("Cloudinary usage response is malformed")
    return {"cloud_name": cloud, "credits_used": credits.get("usage"),
            "credits_limit": credits.get("limit"), "credits_used_percent": credits.get("used_percent")}


def youtube_public() -> dict:
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={YOUTUBE_ID}"
    root = ET.fromstring(_request(url, headers={"User-Agent": "MoolKathaControlMonitor/1"}))
    ns = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015",
          "media": "http://search.yahoo.com/mrss/"}
    owner = root.findtext("atom:author/atom:uri", namespaces=ns)
    if owner != f"https://www.youtube.com/channel/{YOUTUBE_ID}":
        raise MonitorError("public YouTube feed channel identity mismatch")
    videos = []
    for entry in root.findall("atom:entry", ns):
        video_id = entry.findtext("yt:videoId", namespaces=ns)
        channel_id = entry.findtext("yt:channelId", namespaces=ns)
        if channel_id != YOUTUBE_ID or not isinstance(video_id, str) or not re.fullmatch(r"[\w-]{11}", video_id):
            raise MonitorError("public YouTube feed video identity mismatch")
        statistics = entry.find("media:group/media:community/media:statistics", ns)
        views = statistics.get("views") if statistics is not None else None
        videos.append({"id": video_id, "published": entry.findtext("atom:published", namespaces=ns),
                       "views": int(views) if views and views.isdigit() else None})
    return {"channel_id": YOUTUBE_ID, "source": "public Atom feed", "recent_videos": videos}


def youtube_owned(now: datetime, env: dict[str, str]) -> dict:
    """Use the control-owned read token in memory; never persist a refreshed token."""
    try:
        client = json.loads(env.get("YTC_GOOGLE_CLIENT", ""))
        saved = json.loads(env.get("YTC_GOOGLE_TOKEN", ""))
        app = client.get("installed") or client.get("web")
        if (not isinstance(app, dict) or not isinstance(saved, dict)
                or not all(isinstance(app.get(k), str) and app[k] for k in ("client_id", "client_secret"))
                or saved.get("client_id") != app["client_id"]
                or saved.get("client_secret") != app["client_secret"]
                or not isinstance(saved.get("refresh_token"), str) or not saved["refresh_token"]):
            raise ValueError
    except (ValueError, AttributeError, TypeError) as exc:
        raise MonitorError("Google read credentials unavailable") from exc
    params = urllib.parse.urlencode({"client_id": app["client_id"], "client_secret": app["client_secret"],
                                     "refresh_token": saved["refresh_token"], "grant_type": "refresh_token"}).encode()
    request = urllib.request.Request("https://oauth2.googleapis.com/token", data=params,
                                     headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
    with urllib.request.urlopen(request, timeout=25) as response:
        refreshed = json.loads(response.read(100_000))
    token = refreshed.get("access_token") if isinstance(refreshed, dict) else None
    if not isinstance(token, str) or not token:
        raise MonitorError("Google token refresh failed")
    headers = {"Authorization": f"Bearer {token}"}
    channel = _json("https://www.googleapis.com/youtube/v3/channels?part=id,snippet,statistics&mine=true",
                    headers=headers)
    items = channel.get("items")
    if not isinstance(items, list) or len(items) != 1 or items[0].get("id") != YOUTUBE_ID:
        raise MonitorError("Google OAuth channel identity mismatch")
    stats = items[0].get("statistics") or {}
    snapshot = {"channel_id": YOUTUBE_ID, "title": (items[0].get("snippet") or {}).get("title"),
                "subscribers": int(stats["subscriberCount"]) if str(stats.get("subscriberCount", "")).isdigit() else None,
                "views": int(stats["viewCount"]) if str(stats.get("viewCount", "")).isdigit() else None,
                "videos": int(stats["videoCount"]) if str(stats.get("videoCount", "")).isdigit() else None}
    end = now.astimezone(IST).date() - timedelta(days=1)
    start = end - timedelta(days=27)
    query = urllib.parse.urlencode({"ids": "channel==MINE", "startDate": start.isoformat(),
                                    "endDate": end.isoformat(), "metrics": "views,subscribersGained,subscribersLost",
                                    "dimensions": "day", "sort": "day"})
    analytics = _json("https://youtubeanalytics.googleapis.com/v2/reports?" + query, headers=headers)
    columns = analytics.get("columnHeaders")
    rows = analytics.get("rows") or []
    expected = ["day", "views", "subscribersGained", "subscribersLost"]
    if (not isinstance(columns, list) or [c.get("name") for c in columns] != expected
            or not isinstance(rows, list) or len(rows) > 28):
        raise MonitorError("YouTube Analytics response is malformed")
    daily = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 4 or not isinstance(row[0], str):
            raise MonitorError("YouTube Analytics daily row is malformed")
        daily.append({"day": row[0], **{field: _count(value) for field, value in zip(expected[1:], row[1:])}})
    return {"channel": snapshot, "last_28_days": {"start": start.isoformat(), "end": end.isoformat(),
                                                      "daily": daily}}


def release(now: datetime, env: dict[str, str]) -> dict:
    token = env.get("GH_TOKEN", "")
    if not token:
        raise MonitorError("GitHub read token missing")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    url = f"https://api.github.com/repos/{REPOSITORY}/actions/workflows/{RELEASE_WORKFLOW}"
    workflow = _json(url, headers=headers)
    if workflow.get("path") != f".github/workflows/{RELEASE_WORKFLOW}":
        raise MonitorError("control release workflow identity mismatch")
    body = _json(url + "/runs?branch=main&per_page=30", headers=headers)
    runs = body.get("workflow_runs")
    if not isinstance(runs, list):
        raise MonitorError("control release runs response is malformed")
    scheduled = sorted((r for r in runs if isinstance(r, dict) and r.get("event") == "schedule" and
                        r.get("head_branch") == "main"), key=lambda r: r.get("created_at") or "", reverse=True)
    result = {"workflow": RELEASE_WORKFLOW, "state": workflow.get("state"),
              "gate_enabled": env.get("YTC_ENABLE_CONTROL_RELEASE") == "1", "latest_scheduled": None,
              "issues": []}
    if not result["gate_enabled"]:
        result["status"] = "gate_off"
        return result
    if workflow.get("state") != "active":
        result["issues"].append("control release workflow is not active")
    if not scheduled:
        result["issues"].append("no scheduled control release run found")
    else:
        latest = scheduled[0]
        created = _time(latest.get("created_at"))
        if now - created > timedelta(hours=5):
            result["issues"].append("no scheduled control release run within five hours")
        completed = next((run for run in scheduled if run.get("status") == "completed"), None)
        if completed and completed.get("conclusion") != "success":
            result["issues"].append("latest completed scheduled control release run did not succeed")
        result["latest_scheduled"] = {"id": latest.get("id"), "created_at": created.isoformat(),
                                      "status": latest.get("status"), "conclusion": latest.get("conclusion"),
                                      "url": latest.get("html_url")}
        if completed:
            result["latest_completed_scheduled"] = {
                "id": completed.get("id"), "conclusion": completed.get("conclusion"),
                "url": completed.get("html_url")}
    result["status"] = "alert" if result["issues"] else "ok"
    return result


def qa(now: datetime, env: dict[str, str]) -> dict:
    """Watch the scheduled dispatch and the immutable QA run it launches."""
    token = env.get("GH_TOKEN", "")
    if not token:
        raise MonitorError("GitHub read token missing")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    result = {"dispatch": None, "pinned_qa": None, "issues": []}
    for workflow_name, label, predicate in (
        (QA_DISPATCH_WORKFLOW, "scheduled independent QA dispatch",
         lambda run: run.get("event") == "schedule" and run.get("head_branch") == "main"),
        (QA_WORKFLOW, "pinned independent QA",
         lambda run: run.get("event") == "workflow_dispatch" and
         run.get("head_branch") == "qa-v19" and run.get("head_sha") == QA_WORKFLOW_SHA),
    ):
        url = f"https://api.github.com/repos/{REPOSITORY}/actions/workflows/{workflow_name}"
        workflow = _json(url, headers=headers)
        if workflow.get("path") != f".github/workflows/{workflow_name}":
            raise MonitorError(f"{label} workflow identity mismatch")
        if workflow.get("state") != "active":
            result["issues"].append(f"{label} workflow is not active")
        body = _json(url + "/runs?per_page=50", headers=headers)
        runs = body.get("workflow_runs")
        if not isinstance(runs, list):
            raise MonitorError(f"{label} runs response is malformed")
        relevant = sorted((run for run in runs if isinstance(run, dict) and predicate(run)),
                          key=lambda run: run.get("created_at") or "", reverse=True)
        key = "dispatch" if workflow_name == QA_DISPATCH_WORKFLOW else "pinned_qa"
        if not relevant:
            warming_up = False
            if workflow_name == QA_DISPATCH_WORKFLOW and workflow.get("created_at"):
                # A newly added or changed scheduled workflow cannot have a
                # scheduled run before its first following cron slot. GitHub
                # may delay that slot; retain the usual four-hour slack.
                changed_url = (f"https://api.github.com/repos/{REPOSITORY}/commits?"
                               f"path=.github%2Fworkflows%2F{QA_DISPATCH_WORKFLOW}&sha=main&per_page=1")
                commits = json.loads(_request(changed_url, headers=headers))
                if (not isinstance(commits, list) or len(commits) != 1
                        or not isinstance(commits[0], dict)):
                    raise MonitorError("QA dispatcher activation is unavailable")
                changed = _time(((commits[0].get("commit") or {}).get("committer") or {}).get("date"))
                activated = max(_time(workflow["created_at"]), changed)
                start = activated.replace(minute=0, second=0, microsecond=0)
                first_due = next((slot.replace(minute=QA_DISPATCH_MINUTE)
                                  for hour in range(48)
                                  if (slot := start + timedelta(hours=hour)).hour in QA_DISPATCH_HOURS
                                  and slot.replace(minute=QA_DISPATCH_MINUTE) > activated), None)
                if first_due is None:
                    raise MonitorError("QA dispatcher first due time is unavailable")
                warming_up = now <= first_due + timedelta(hours=QA_QUIET_HOURS - 6)
                if warming_up:
                    result[key] = {"workflow": workflow_name, "state": workflow.get("state"),
                                   "status": "warming_up", "first_due": first_due.isoformat()}
            if not warming_up:
                result["issues"].append(f"no {label} run found")
            continue
        latest = relevant[0]
        created = _time(latest.get("created_at"))
        if now - created > timedelta(hours=QA_QUIET_HOURS):
            result["issues"].append(f"no {label} run within {QA_QUIET_HOURS} hours")
        completed = next((run for run in relevant if run.get("status") == "completed"), None)
        if completed and completed.get("conclusion") != "success":
            result["issues"].append(f"latest completed {label} run did not succeed")
        result[key] = {"workflow": workflow_name, "state": workflow.get("state"),
                       "latest": {"id": latest.get("id"), "created_at": created.isoformat(),
                                  "status": latest.get("status"), "conclusion": latest.get("conclusion"),
                                  "url": latest.get("html_url")},
                       "latest_completed": ({"id": completed.get("id"),
                                             "conclusion": completed.get("conclusion"),
                                             "url": completed.get("html_url")}
                                            if completed else None)}
    result["status"] = "alert" if result["issues"] else "ok"
    return result


def collect(now: datetime, env: dict[str, str]) -> dict:
    if env.get("GITHUB_REPOSITORY") != REPOSITORY or env.get("GITHUB_REF") != "refs/heads/main":
        raise MonitorError("monitor must run from control main")
    report = {"schema": "mool_katha_control_monitor_v1", "captured_at": now.isoformat(),
              "ist_date": now.astimezone(IST).date().isoformat(), "repository": REPOSITORY,
              "sources": {}, "issues": [], "warnings": []}
    release_rows: dict[str, list[dict]] = {}
    for name, reader in (("release", lambda: release(now, env)), ("qa", lambda: qa(now, env)),
                         ("buffer", lambda: buffer(now, env)),
                         ("cloudinary", lambda: cloudinary(env)),
                         ("youtube_owned", lambda: youtube_owned(now, env)),
                         ("youtube_public", youtube_public)):
        try:
            value = reader()
            report["sources"][name] = value
            if name in ("release", "qa"):
                report["issues"].extend(value["issues"])
            if name == "buffer":
                for service, channel in value["channels"].items():
                    release_rows[service] = channel.pop("_release_rows", [])
                    if channel["disconnected"] or channel["locked"] or channel["queue_paused"]:
                        report["issues"].append(f"Buffer {service} channel is unavailable")
                    if channel["recent_errors"]:
                        report["issues"].append(f"Buffer {service} has {len(channel['recent_errors'])} failed posts")
        except Exception as exc:
            # Preserve the artifact even for an unexpected provider shape. No provider
            # body, request, credential, or raw exception is written to it.
            label = str(exc) if isinstance(exc, MonitorError) else "read failed"
            report["sources"][name] = {"status": "unavailable", "reason": label}
            (report["warnings"] if name in ("youtube_owned", "youtube_public") else report["issues"]).append(
                f"{name}: {label}")
    try:
        if not env.get("SOURCE_CHECKOUT"):
            raise SignedMonitorError("read-only producer checkout is unavailable")
        value = inspect_signed(now, Path(env["SOURCE_CHECKOUT"]), release_rows,
                               BUFFER_YOUTUBE_ID, BUFFER_INSTAGRAM_ID)
        report["sources"]["signed_pairs"] = value
        report["issues"].extend(value["issues"])
    except Exception as exc:
        label = str(exc) if isinstance(exc, SignedMonitorError) else "read failed"
        report["sources"]["signed_pairs"] = {"status": "unavailable", "reason": label}
        report["issues"].append(f"signed_pairs: {label}")
    else:
        try:
            proof_dir = env.get("PUBLIC_PROOF_DIR")
            proofs = load_proofs(Path(proof_dir)) if proof_dir else {}
            public = inspect_public(now, value, report["sources"].get("youtube_public"),
                                    proofs=proofs)
            report["sources"]["public_posts"] = public
            report["issues"].extend(public["issues"])
        except Exception:
            report["sources"]["public_posts"] = {"status": "unavailable", "reason": "read failed"}
            report["issues"].append("public_posts: read failed")
    if all(report["sources"][source].get("status") == "unavailable"
           for source in ("youtube_owned", "youtube_public")):
        report["issues"].append("both owned and public YouTube analytics are unavailable")
    report["status"] = "alert" if report["issues"] else "ok"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = collect(datetime.now(UTC), dict(os.environ))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    summary = Path(os.environ.get("GITHUB_STEP_SUMMARY", os.devnull))
    with summary.open("a", encoding="utf-8") as output:
        output.write(f"## Mool Katha control monitor: {report['status']}\n\n")
        output.write(f"Captured {report['captured_at']} (IST date {report['ist_date']}).\n\n")
        for name, value in report["sources"].items():
            output.write(f"- {name}: {value.get('status', 'ok')}\n")
        for issue in report["issues"]:
            output.write(f"- Alert: {issue}\n")
        for warning in report["warnings"]:
            output.write(f"- Warning: {warning}\n")
    print(f"Control monitor report saved: {report['status']}; {len(report['issues'])} issue(s)")
    return 1 if report["issues"] else 0


if __name__ == "__main__":
    sys.exit(main())
