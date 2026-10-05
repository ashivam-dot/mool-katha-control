"""When the lite lane may post: its own two daily slots, never more than two Shorts a day across both lanes."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
# The signed lane posts at 19:00 and 20:30 IST (ytc.publish.SLOTS[:2]); Instagram Stories go at 08:30 IST.
LITE_SLOTS = (time(17, 30), time(21, 45))
MAX_PER_DAY = 2
# No two posts on one destination closer than this.
MIN_GAP = timedelta(minutes=60)
# Time to host the file and let Buffer take the post before it is due.
LEAD = timedelta(minutes=40)
INACTIVE = ("error", "draft")


def _due(row: dict) -> datetime | None:
    try:
        due = datetime.fromisoformat(str(row.get("dueAt") or ""))
    except ValueError:
        return None
    return due if due.utcoffset() is not None else None


def active_times(rows: list[dict]) -> list[datetime]:
    """Due times of posts that will go out or already went out (failed and draft posts don't count)."""
    times = []
    for row in rows:
        if row.get("status") in INACTIVE:
            continue
        if (due := _due(row)) is not None:
            times.append(due)
    return times


def shorts_on(day: date, youtube_rows: list[dict]) -> int:
    """Shorts on one IST day, from the YouTube channel (every Short has exactly one YouTube post)."""
    return sum(t.astimezone(IST).date() == day for t in active_times(youtube_rows))


def free_slot(now: datetime, youtube_rows: list[dict], instagram_rows: list[dict],
              days_ahead: int = 1) -> datetime | None:
    """The earliest lite slot from today through `days_ahead` days later that keeps both daily rules."""
    now = now.astimezone(IST)
    taken = active_times(youtube_rows) + active_times(instagram_rows)
    for offset in range(days_ahead + 1):
        day = now.date() + timedelta(days=offset)
        if shorts_on(day, youtube_rows) >= MAX_PER_DAY:
            continue
        for slot in LITE_SLOTS:
            when = datetime.combine(day, slot, IST)
            if when < now + LEAD:
                continue
            if any(abs(when - t) < MIN_GAP for t in taken):
                continue
            return when
    return None


def utc(when: datetime) -> str:
    return when.astimezone(timezone.utc).isoformat(timespec="seconds")
