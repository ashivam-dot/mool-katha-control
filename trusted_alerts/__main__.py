"""Notify once per active incident, with a private issue as durable fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit


REPOSITORY = "ashivam-dot/mool-katha-control"
QA_SHA = "07a920b8ff6499a8ea7ba12d2da9d1a6b43b5d39"
WATCHES = {
    "Control analytics and watchdog": ("control-monitor", "control-monitor.yml", "main", None),
    "Dispatch pinned independent QA": ("qa-dispatch", "dispatch-release-qa.yml", "main", None),
    "Signed independent QA": ("pinned-qa", "release-qa.yml", "qa-v20", QA_SHA),
    "Independent QA failure feedback": ("qa-feedback", "qa-failure-feedback.yml", "main", None),
    "Signed release from independent control": ("signed-release", "signed-release.yml", "main", None),
}
LABELS = {
    "control-monitor": "Control monitor failure",
    "qa-dispatch": "QA dispatcher failure",
    "pinned-qa": "Pinned QA failure",
    "qa-feedback": "QA hold feedback failure",
    "signed-release": "Signed release failure",
    "control-monitor-missed": "Control monitor schedule missed",
    "qa-dispatch-missed": "QA dispatcher schedule missed",
    "pinned-qa-missed": "Pinned QA run missed",
    "signed-release-missed": "Signed release schedule missed",
}
SCHEDULES = (
    # name, event, quiet hours, cron hours, cron minute, cron interval
    ("Control analytics and watchdog", "schedule", 6, tuple(range(0, 24, 3)), 25, 3),
    ("Dispatch pinned independent QA", "schedule", 10, (0, 6, 12, 18), 10, 6),
    ("Signed independent QA", "workflow_dispatch", 10, None, None, None),
    ("Signed release from independent control", "schedule", 5, tuple(range(0, 24, 2)), 17, 2),
)
UTC = timezone.utc
MARKER = re.compile(r"<!-- mool-control-alert ([a-z-]+) ([0-9a-f]{16}) (sent|pending) -->")


class AlertError(RuntimeError):
    """One or both alert destinations could not be updated safely."""


def _api(method: str, path: str, token: str, payload: dict | None = None) -> object:
    if not token:
        raise AlertError("GitHub alert token is unavailable")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPOSITORY}{path}",
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json", "X-GitHub-Api-Version": "2022-11-28"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            body = response.read(2_000_001)
        if len(body) > 2_000_000:
            raise AlertError("GitHub alert response is oversized")
        return json.loads(body) if body else {}
    except AlertError:
        raise
    except Exception:
        # Provider responses and request headers may contain private data.
        raise AlertError("GitHub alert state is unavailable") from None


def _time(value: object) -> datetime:
    if not isinstance(value, str):
        raise AlertError("workflow run time is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise AlertError("workflow run time is malformed") from None
    if parsed.utcoffset() is None:
        raise AlertError("workflow run time has no zone")
    return parsed.astimezone(UTC)


def _url(value: object) -> str:
    if not isinstance(value, str):
        return f"https://github.com/{REPOSITORY}/actions"
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return f"https://github.com/{REPOSITORY}/actions"
    if (parsed.scheme == "https" and parsed.hostname == "github.com" and not parsed.username
            and not parsed.password and not port
            and parsed.path.startswith(f"/{REPOSITORY}/actions/runs/")
            and re.fullmatch(rf"/{re.escape(REPOSITORY)}/actions/runs/[0-9]+", parsed.path)
            and not parsed.query and not parsed.fragment):
        return value
    return f"https://github.com/{REPOSITORY}/actions"


def _issues(token: str) -> list[dict]:
    found = []
    for page in range(1, 6):
        result = _api("GET", f"/issues?state=open&per_page=100&page={page}", token)
        if not isinstance(result, list):
            raise AlertError("private alert issue list is malformed")
        found.extend(item for item in result if isinstance(item, dict) and "pull_request" not in item)
        if len(result) < 100:
            return found
    raise AlertError("private alert issue list exceeds page limit")


def _body(category: str, messages: list[str], run_url: str, fingerprint: str,
          phone_state: str, now: datetime) -> str:
    lines = [f"## {LABELS[category]}", "", f"Last observed: {now.isoformat()}",
             f"Run: {run_url}", ""]
    lines.extend(f"- {message}" for message in messages)
    lines.extend(["", f"<!-- mool-control-alert {category} {fingerprint} {phone_state} -->", ""])
    return "\n".join(lines)


def _phone(topic: str, category: str, messages: list[str], run_url: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{10,128}", topic):
        raise AlertError("phone topic is unavailable; private issue remains open")
    message = "\n".join(messages)[:700] + "\n" + run_url
    payload = {"topic": topic, "title": "Mool Katha: " + LABELS[category],
               "message": message, "priority": 4, "tags": ["warning"], "click": run_url}
    request = urllib.request.Request(
        "https://ntfy.sh/", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read(1000)
    except Exception:
        raise AlertError("phone notification failed; private issue remains open") from None


def sync(category: str, messages: list[str], *, run_url: str, now: datetime,
         token: str, topic: str) -> str:
    """Keep one open issue per incident and phone only on a new incident."""
    if category not in LABELS or any(not isinstance(item, str) or len(item) > 200 for item in messages):
        raise AlertError("alert category or message is invalid")
    messages = sorted(set(messages))[:12]
    title = "Mool Katha control: " + LABELS[category]
    matching = [item for item in _issues(token) if item.get("title") == title]
    if len(matching) > 1:
        raise AlertError("duplicate private alert issues need review")
    current = matching[0] if matching else None
    if not messages:
        if current:
            _api("PATCH", f"/issues/{current['number']}", token, {"state": "closed"})
            return "resolved"
        return "clear"
    run_url = _url(run_url)
    fingerprint = hashlib.sha256("\n".join(messages).encode("utf-8")).hexdigest()[:16]
    existing = MARKER.search(current.get("body") or "") if current else None
    already_sent = bool(existing and existing.group(1) == category
                        and existing.group(2) == fingerprint and existing.group(3) == "sent")
    if already_sent:
        return "already_notified"
    pending = _body(category, messages, run_url, fingerprint, "pending", now)
    if current:
        _api("PATCH", f"/issues/{current['number']}", token, {"body": pending})
        number = current["number"]
    else:
        created = _api("POST", "/issues", token, {"title": title, "body": pending})
        if not isinstance(created, dict) or not isinstance(created.get("number"), int):
            raise AlertError("private alert issue creation was not confirmed")
        number = created["number"]
    _phone(topic, category, messages, run_url)
    sent_body = _body(category, messages, run_url, fingerprint, "sent", now)
    _api("PATCH", f"/issues/{number}", token, {"body": sent_body})
    return "notified"


def _runs(workflow_file: str, token: str) -> list[dict]:
    value = _api("GET", f"/actions/workflows/{workflow_file}/runs?per_page=100", token)
    if not isinstance(value, dict) or not isinstance(value.get("workflow_runs"), list):
        raise AlertError("workflow run list is malformed")
    return [row for row in value["workflow_runs"] if isinstance(row, dict)]


def _relevant(row: dict, branch: str, sha: str | None, event: str | None = None) -> bool:
    return (row.get("head_branch") == branch and (sha is None or row.get("head_sha") == sha)
            and (event is None or row.get("event") == event))


def _workflow_activation(workflow_file: str, workflow: dict, token: str) -> datetime:
    changed = _api("GET", f"/commits?path=.github%2Fworkflows%2F{quote(workflow_file)}&sha=main&per_page=1", token)
    if not isinstance(changed, list) or len(changed) != 1 or not isinstance(changed[0], dict):
        raise AlertError("watched workflow activation is unavailable")
    committed_at = ((changed[0].get("commit") or {}).get("committer") or {}).get("date")
    return max(_time(workflow.get("created_at")), _time(committed_at))


def _first_due(activated: datetime, hours: tuple[int, ...], minute: int) -> datetime:
    start = activated.replace(minute=0, second=0, microsecond=0)
    for offset in range(48):
        slot = (start + timedelta(hours=offset)).replace(minute=minute)
        if slot.hour in hours and slot > activated:
            return slot
    raise AlertError("watched workflow first due time is unavailable")


def from_workflow_run(event: dict, *, token: str, topic: str, now: datetime) -> str:
    run = event.get("workflow_run")
    if (not isinstance(run, dict) or (event.get("repository") or {}).get("full_name") != REPOSITORY
            or not isinstance(run.get("name"), str) or run["name"] not in WATCHES):
        return "ignored"
    category, workflow_file, branch, sha = WATCHES[run["name"]]
    head_repository = run.get("head_repository")
    if (isinstance(head_repository, dict) and head_repository.get("full_name") != REPOSITORY
            or not _relevant(run, branch, sha) or run.get("status") != "completed"):
        return "ignored"
    completed = [item for item in _runs(workflow_file, token)
                 if _relevant(item, branch, sha) and item.get("status") == "completed"]
    if completed:
        latest = max(completed, key=lambda item: (item.get("created_at") or "", item.get("id") or 0))
        if (latest.get("created_at") or "") >= (run.get("created_at") or ""):
            run = latest
    conclusion = run.get("conclusion")
    messages = [] if conclusion in ("success", "skipped") else [f"{LABELS[category]}: latest run did not succeed"]
    return sync(category, messages, run_url=run.get("html_url"), now=now, token=token, topic=topic)


def check_missed(*, token: str, topic: str, now: datetime, gate_enabled: bool) -> dict[str, str]:
    results = {}
    for name, event, quiet_hours, cron_hours, cron_minute, interval in SCHEDULES:
        category, workflow_file, branch, sha = WATCHES[name]
        missed_category = category + "-missed"
        if category == "signed-release" and not gate_enabled:
            results[missed_category] = sync(missed_category, [], run_url="", now=now,
                                            token=token, topic=topic)
            continue
        workflow = _api("GET", f"/actions/workflows/{workflow_file}", token)
        if (not isinstance(workflow, dict)
                or workflow.get("path") != f".github/workflows/{workflow_file}"):
            raise AlertError("watched workflow identity is unavailable")
        relevant = [row for row in _runs(workflow_file, token) if _relevant(row, branch, sha, event)]
        latest = max(relevant, key=lambda row: (row.get("created_at") or "", row.get("id") or 0)) if relevant else None
        messages = []
        if workflow.get("state") != "active":
            messages.append(f"{LABELS[missed_category]}: workflow is inactive")
        if latest is None:
            activated = _workflow_activation(workflow_file, workflow, token)
            if cron_hours is None:
                deadline = activated + timedelta(hours=quiet_hours)
            else:
                deadline = _first_due(activated, cron_hours, cron_minute) + timedelta(
                    hours=quiet_hours - interval)
            stale = now > deadline
        else:
            stale = now - _time(latest.get("created_at")) > timedelta(hours=quiet_hours)
        if stale:
            messages.append(f"{LABELS[missed_category]}: no exact run within {quiet_hours} hours")
        results[missed_category] = sync(missed_category, messages,
                                        run_url=latest.get("html_url") if latest else "",
                                        now=now, token=token, topic=topic)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("workflow-run", "missed"))
    parser.add_argument("--event", type=Path)
    args = parser.parse_args()
    token = os.environ.get("GH_TOKEN", "")
    topic = os.environ.get("YTC_NTFY_TOPIC", "")
    now = datetime.now(UTC)
    try:
        if args.mode == "workflow-run":
            if not args.event or not args.event.is_file() or args.event.stat().st_size > 2_000_000:
                raise AlertError("workflow event is unavailable")
            event = json.loads(args.event.read_text(encoding="utf-8"))
            if not isinstance(event, dict):
                raise AlertError("workflow event is malformed")
            result = from_workflow_run(event, token=token, topic=topic, now=now)
        else:
            result = check_missed(token=token, topic=topic, now=now,
                                  gate_enabled=os.environ.get("YTC_ENABLE_CONTROL_RELEASE") == "1")
        print(json.dumps({"alert_result": result}, sort_keys=True))
        return 0
    except AlertError as exc:
        print(f"Owner alert delivery needs retry: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print("Owner alert delivery needs retry: unexpected failure", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
