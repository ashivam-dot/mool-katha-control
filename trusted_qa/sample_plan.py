"""Conservative model sampling over a complete, deterministic frame audit.

This plan never substitutes samples for the complete video decode or the
per-frame pixel/temporal audit. It holds if required model samples exceed the
explicit two-sheet capacity, rather than silently dropping any anomaly.
"""

from __future__ import annotations

import bisect
import math
from typing import Any

from .common import QaHold, require


MAX_MODEL_SHEETS = 2
MODEL_SHEET_SIZE = 35
MAX_MODEL_FRAMES = MAX_MODEL_SHEETS * MODEL_SHEET_SIZE
MAX_SAMPLE_GAP_SECONDS = 2.0
RGB_JUMP_THRESHOLD = 5.0
EXTREME_MEAN_LOW = 8.0
EXTREME_MEAN_HIGH = 247.0


def _nearest_index(times: list[float], target: float, *, left: int = 0,
                   right: int | None = None) -> int:
    """Return a zero-based decoded-frame index, breaking ties toward earlier."""
    if right is None:
        right = len(times) - 1
    require(0 <= left <= right < len(times), "sample interval contains no decoded frame")
    position = bisect.bisect_left(times, target, left, right + 1)
    candidates = [value for value in (position - 1, position)
                  if left <= value <= right]
    require(bool(candidates), "sample target contains no decoded frame")
    return min(candidates, key=lambda value: (abs(times[value] - target), value))


def plan_sampled_frames(frames: list[dict[str, Any]], beats: list[dict[str, Any]],
                        duration: float) -> dict[str, Any]:
    """Require beat, transition, anomaly, and at-most-two-second model samples.

    Every frame remains in `frames` for the separate deterministic audit. The
    returned indices identify the subset that must pass indexed model review.
    """
    require(isinstance(frames, list) and bool(frames), "sample plan has no decoded frames")
    require(isinstance(duration, (int, float)) and not isinstance(duration, bool)
            and math.isfinite(duration) and duration > 0,
            "sample plan has an invalid duration")
    times: list[float] = []
    for index, frame in enumerate(frames, 1):
        require(isinstance(frame, dict) and isinstance(frame.get("index"), int) and
                not isinstance(frame["index"], bool) and frame["index"] == index and
                all(isinstance(frame.get(name), (int, float)) and
                    not isinstance(frame[name], bool) and math.isfinite(frame[name])
                    for name in ("seconds", "rgb_delta_previous", "mean_luma")),
                f"sample plan frame {index} is invalid")
        seconds = float(frame["seconds"])
        require(seconds >= 0 and seconds <= duration + 0.5 and
                (not times or seconds > times[-1]) and
                0 <= frame["rgb_delta_previous"] <= 255 and
                0 <= frame["mean_luma"] <= 255,
                f"sample plan frame {index} has invalid timing or pixels")
        times.append(seconds)
    require(isinstance(beats, list) and bool(beats), "sample plan has no render beats")
    reasons: dict[int, set[str]] = {}

    def add(index: int, reason: str) -> None:
        reasons.setdefault(index, set()).add(reason)
        require(len(reasons) <= MAX_MODEL_FRAMES,
                "required sampled frames exceed two-sheet model capacity")

    add(1, "opening")
    add(len(frames), "ending")
    beat_midpoints: list[dict[str, int]] = []
    boundaries: set[float] = set()
    for number, beat in enumerate(beats, 1):
        require(isinstance(beat, dict), f"sample plan beat {number} is invalid")
        start, end = beat.get("start"), beat.get("end")
        require(isinstance(start, (int, float)) and isinstance(end, (int, float)) and
                not isinstance(start, bool) and not isinstance(end, bool) and
                math.isfinite(start) and math.isfinite(end) and
                0 <= start < end <= duration + 0.5,
                f"sample plan beat {number} has an invalid span")
        midpoint = (start + end) / 2
        index = _nearest_index(times, midpoint) + 1
        add(index, f"beat_{number}_midpoint")
        beat_midpoints.append({"beat": number, "index": index})
        for boundary in (start, end):
            if times[0] < boundary < times[-1]:
                boundaries.add(float(boundary))

    boundary_pairs: list[dict[str, int | float]] = []
    for boundary in sorted(boundaries):
        after = bisect.bisect_left(times, boundary)
        require(0 < after < len(times), "beat boundary is outside decoded frames")
        before_index, after_index = after, after + 1
        add(before_index, "beat_boundary_before")
        add(after_index, "beat_boundary_after")
        boundary_pairs.append({"seconds": round(boundary, 6),
                               "before_index": before_index, "after_index": after_index})

    anomalies: list[dict[str, Any]] = []
    for frame in frames:
        index = frame["index"]
        delta = float(frame["rgb_delta_previous"])
        if index > 1 and delta >= RGB_JUMP_THRESHOLD:
            add(index - 1, "rgb_jump_before")
            add(index, "rgb_jump_after")
            anomalies.append({"index": index, "type": "rgb_jump",
                              "metric": round(delta, 3)})
        mean = float(frame["mean_luma"])
        if mean < EXTREME_MEAN_LOW or mean > EXTREME_MEAN_HIGH:
            add(index, "extreme_mean_luma")
            anomalies.append({"index": index, "type": "extreme_mean_luma",
                              "metric": round(mean, 3)})

    # Fill the longest uncovered time gaps. The final bound is checked against
    # actual decoded-frame timestamps, not nominal frame rate or target times.
    while True:
        ordered = sorted(reasons)
        gaps = [(times[right - 1] - times[left - 1], left, right)
                for left, right in zip(ordered, ordered[1:])]
        if not gaps or max(gap for gap, _, _ in gaps) <= MAX_SAMPLE_GAP_SECONDS:
            break
        _, left, right = max(gaps, key=lambda item: (item[0], -item[1]))
        chosen = _nearest_index(times, (times[left - 1] + times[right - 1]) / 2,
                                left=left, right=right - 2) + 1
        require(chosen not in reasons, "sample cadence cannot cover decoded frames")
        add(chosen, "cadence")

    ordered = sorted(reasons)
    max_gap = max((times[right - 1] - times[left - 1]
                   for left, right in zip(ordered, ordered[1:])), default=0.0)
    require(max_gap <= MAX_SAMPLE_GAP_SECONDS,
            "model sample gaps exceed the temporal coverage bound")
    return {"kind": "deterministic_sample_plan_v1",
            "decoded_frame_count": len(frames),
            "sampled_indices": ordered,
            "sample_reasons": [{"index": index, "reasons": sorted(reasons[index])}
                               for index in ordered],
            "beat_midpoints": beat_midpoints,
            "beat_boundary_pairs": boundary_pairs,
            "anomalies": anomalies,
            "max_gap_seconds": round(max_gap, 6),
            "max_allowed_gap_seconds": MAX_SAMPLE_GAP_SECONDS,
            "rgb_jump_threshold": RGB_JUMP_THRESHOLD,
            "model_frame_capacity": MAX_MODEL_FRAMES}
