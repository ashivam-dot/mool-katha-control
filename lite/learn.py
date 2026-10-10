"""What the channel's own numbers say, fed back into the next topic.

Each publish run (at most every REFRESH) reads the Buffer metrics of every lite Short's YouTube post and Instagram
Reel, keeps a snapshot per episode in the ledger, scores each episode by its combined views SCORE_AT_H hours after
its slot, and learns a lift per source work (Ramayana, Gita, Mahabharata, ...). A lift needs MIN_SAMPLES scored
Shorts, is shrunk toward 1 by n / (n + PRIOR) and stays within LIFT_RANGE. next_topic only uses it to reorder the
next WINDOW unused curated topics, so the curated order and the variety of works still hold.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

REFRESH = timedelta(hours=6)
SCORE_AT_H = 48.0
MIN_SAMPLES = 4
PRIOR = 4.0
LIFT_RANGE = (0.75, 1.35)
WINDOW = 3
KEEP_SNAPSHOTS = 16
MAX_PAGES = 20
METRICS = ("views", "reach", "reactions", "comments", "shares", "saves")
QUERY = """query Posts($input: PostsInput!, $after: String) {
  posts(input: $input, first: 50, after: $after) {
    edges { node { id channelId metrics { type value } } }
    pageInfo { hasNextPage endCursor } } }"""


def work_of(topic_key: str) -> str:
    return topic_key.rsplit("-", 2)[0]


def views_at(snapshots: list[dict], hours: float) -> float | None:
    """Views `hours` after the slot: interpolated between snapshots, or scaled back from the first later one (views
    grow roughly with the square root of age after the first push). None while too young."""
    snaps = sorted(snapshots, key=lambda s: s["age_h"])
    if not snaps or snaps[-1]["age_h"] < hours:
        return None
    later = next(s for s in snaps if s["age_h"] >= hours)
    earlier = [s for s in snaps if s["age_h"] < hours]
    if not earlier:
        return later["views"] * math.sqrt(hours / later["age_h"]) if later["age_h"] > 0 else float(later["views"])
    a = earlier[-1]
    span = later["age_h"] - a["age_h"]
    return a["views"] + (later["views"] - a["views"]) * ((hours - a["age_h"]) / span if span else 1.0)


def lifts(scores: dict[str, list[float]]) -> dict:
    """scores: work -> views at SCORE_AT_H of each scored Short."""
    flat = [v for vs in scores.values() for v in vs]
    out = {"scored": len(flat), "min_samples": MIN_SAMPLES, "lifts": {}}
    if len(flat) < MIN_SAMPLES:
        return out
    base = sum(math.log1p(v) for v in flat) / len(flat)
    out["baseline_views"] = round(math.expm1(base))
    for work, vs in scores.items():
        n = len(vs)
        if n < MIN_SAMPLES:
            continue
        mean = sum(math.log1p(v) for v in vs) / n
        lift = math.exp((mean - base) * n / (n + PRIOR))
        out["lifts"][work] = {"n": n, "lift": round(min(max(lift, LIFT_RANGE[0]), LIFT_RANGE[1]), 3),
                              "typical_views": round(math.expm1(mean))}
    return out


def metrics(post_ids: set[str], channel_ids: list[str], buffer) -> dict[str, dict]:
    """Metrics of the wanted Buffer posts, paging until all are found."""
    variables = {"input": {"organizationId": os.environ["BUFFER_ORG_ID"], "filter": {"channelIds": channel_ids}},
                 "after": None}
    found: dict[str, dict] = {}
    for _ in range(MAX_PAGES):
        page = buffer(QUERY, variables)["posts"]
        for edge in page["edges"]:
            row = edge["node"]
            if row["id"] in post_ids:
                found[row["id"]] = {m["type"]: m["value"] for m in row.get("metrics") or []
                                    if m.get("type") in METRICS and isinstance(m.get("value"), (int, float))}
        info = page["pageInfo"]
        if found.keys() >= post_ids or not info["hasNextPage"]:
            break
        variables["after"] = info["endCursor"]
    return found


def update(ledger: dict, found: dict[str, dict], now: datetime) -> dict:
    """Snapshot each episode whose slot has passed, rescore, relearn. Changes `ledger` in place."""
    perf = ledger.setdefault("performance", {})
    scores: dict[str, list[float]] = {}
    for ep in ledger["episodes"]:
        pub = ep.get("publish") or {}
        if not pub.get("due_at"):
            continue
        age = (now - datetime.fromisoformat(pub["due_at"])).total_seconds() / 3600
        rows = {s: found.get((pub.get(s) or {}).get("id") or "") for s in ("youtube", "instagram")}
        rec = perf.setdefault(ep["id"], {"topic_key": ep["topic_key"], "due_at": pub["due_at"], "snapshots": []})
        if age >= 0 and any(rows.values()):
            snap = {"age_h": round(age, 2), "views": sum((r or {}).get("views", 0) for r in rows.values())}
            for s, r in rows.items():
                for k, v in (r or {}).items():
                    snap[f"{s}_{k}"] = v
            rec["snapshots"] = (rec["snapshots"] + [snap])[-KEEP_SNAPSHOTS:]
        rec["views_at_48h"] = views_at(rec["snapshots"], SCORE_AT_H)
        if rec["views_at_48h"] is not None:
            scores.setdefault(work_of(ep["topic_key"]), []).append(rec["views_at_48h"])
    ledger["learned"] = {"at": now.isoformat(timespec="seconds"), **lifts(scores)}
    return ledger


def report(ledger: dict) -> str:
    learned = ledger.get("learned") or {}
    lines = ["# What Mool Katha's own numbers say", "",
             f"Updated {learned.get('at')}. Scored Shorts (YouTube + Instagram views {SCORE_AT_H:.0f} h after the "
             f"slot): {learned.get('scored', 0)}. A work needs {MIN_SAMPLES} scored Shorts for a lift; it is shrunk "
             f"toward 1, kept within {LIFT_RANGE[0]}-{LIFT_RANGE[1]}, and only reorders the next {WINDOW} topics.", ""]
    if learned.get("baseline_views") is not None:
        lines.append(f"Baseline: a typical Short has {learned['baseline_views']} views at 48 h.")
    for work, v in sorted(learned.get("lifts", {}).items(), key=lambda kv: -kv[1]["lift"]):
        lines.append(f"- {work}: x{v['lift']} ({v['n']} Shorts, typical {v['typical_views']} views)")
    if not learned.get("lifts"):
        lines.append("- No lift yet: not enough scored Shorts per work. Topics go in curated order.")
    lines += ["", "| Short | topic | slot | YouTube views | Instagram views | views at 48 h |", "|---|---|---|---|---|---|"]
    for ep_id, rec in sorted((ledger.get("performance") or {}).items(), key=lambda kv: kv[1]["due_at"], reverse=True):
        last = rec["snapshots"][-1] if rec["snapshots"] else {}
        at48 = "" if rec.get("views_at_48h") is None else round(rec["views_at_48h"])
        lines.append(f"| {ep_id} | {rec['topic_key']} | {rec['due_at'][:16]} | {last.get('youtube_views', '')} "
                     f"| {last.get('instagram_views', '')} | {at48} |")
    return "\n".join(lines) + "\n"


def refresh(ledger: dict, youtube_id: str, instagram_id: str, now: datetime | None = None, buffer=None,
            report_path: Path | None = None) -> dict | None:
    """Read Buffer's numbers and relearn, at most once per REFRESH. Returns what was learned, or None if fresh."""
    now = now or datetime.now(timezone.utc)
    at = (ledger.get("learned") or {}).get("at")
    if at and now - datetime.fromisoformat(at) < REFRESH:
        return None
    if buffer is None:
        from ytc import publish
        buffer = publish._buffer
    wanted = {(ep.get("publish") or {}).get(s, {}).get("id") for ep in ledger["episodes"] for s in ("youtube", "instagram")}
    wanted.discard(None)
    update(ledger, metrics(wanted, [youtube_id, instagram_id], buffer) if wanted else {}, now)
    if report_path is not None:
        report_path.write_text(report(ledger), encoding="utf-8")
    return ledger["learned"]


def prefer(topics: list[dict], ledger: dict) -> list[dict]:
    """Lead with the best-performing work among the next WINDOW topics (ties keep the curated order)."""
    learned = (ledger.get("learned") or {}).get("lifts", {})
    if not learned:
        return topics
    head = sorted(topics[:WINDOW], key=lambda t: -learned.get(t.get("work") or work_of(t["key"]), {}).get("lift", 1.0))
    return head + topics[WINDOW:]
