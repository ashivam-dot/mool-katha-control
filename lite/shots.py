"""Two paintings per beat: a beat long enough is told over its wide scene and then a close-up, so no painting
holds the screen much past three or four seconds.

Owner, 2026-10-27, on the Shorts another agent scheduled ("really basic, you know what kind of standard is
really required"): the presentation standard wants a new picture every 3-4 seconds, and the lane's own Shorts held
one painting for a whole 6-9 second beat. The narration is one Gemini take of the whole script, so splitting a
beat into two lines costs no voice quota; the gates, evidence quotes and source review still see whole beats.
"""

from __future__ import annotations

# Ten paint jobs draw two pictures each in about 40 minutes, well inside their 75-minute timeout.
MAX_SHOTS = 20
# At about 140 words a minute a beat shorter than this lasts under five seconds and keeps one painting; longer
# beats split into halves of at least about two and a half seconds.
MIN_SPLIT_WORDS = 12
# Neither half may be shorter than this.
MIN_HALF_WORDS = 3
# A half may begin on one of these words, the natural seams of spoken Hindi.
JOINERS = ("और", "लेकिन", "पर", "परंतु", "किंतु", "तो", "तब", "जब", "फिर", "मगर", "क्योंकि", "तभी", "बल्कि")
BREAKS = (",", "।", ";", "—", "!", "?")


def split_text(text: str) -> tuple[str, str] | None:
    """The beat's words as two spoken halves, cut at the seam nearest the middle, or None for a short beat."""
    words = text.split()
    if len(words) < MIN_SPLIT_WORDS:
        return None
    middle = len(words) / 2
    cuts = range(MIN_HALF_WORDS, len(words) - MIN_HALF_WORDS + 1)
    seams = [k for k in cuts if words[k - 1].endswith(BREAKS) or words[k] in JOINERS]
    near = [k for k in seams if abs(k - middle) <= len(words) / 4]
    cut = min(near, key=lambda k: (abs(k - middle), k)) if near else round(middle)
    return " ".join(words[:cut]), " ".join(words[cut:])


def plan(beats: list[dict], limit: int = MAX_SHOTS) -> list[dict]:
    """The Short's shots in order: {"beat", "text", "scene", "emphasis", "close", "pause_after"}.

    Beats with a `close` scene are split, longest first, while the total stays within `limit`; the hook (beat 1)
    is already a close-up and stays whole."""
    spare = max(0, limit - len(beats))
    halves: dict[int, tuple[str, str]] = {}
    for i in sorted(range(1, len(beats)), key=lambda i: -len(beats[i].get("text", "").split())):
        if len(halves) >= spare:
            break
        if (beats[i].get("close") or "").strip() and (pair := split_text(beats[i].get("text", ""))):
            halves[i] = pair
    shots = []
    for i, beat in enumerate(beats):
        emphasis = beat.get("emphasis") or ""
        if i not in halves:
            shots.append({"beat": i, "text": beat["text"], "scene": beat.get("scene") or "", "emphasis": emphasis,
                          "close": False, "pause_after": 0.12})
            continue
        first, second = halves[i]
        for text, scene, close, pause in ((first, beat.get("scene") or "", False, 0.0),
                                          (second, beat["close"], True, 0.12)):
            shots.append({"beat": i, "text": text, "scene": scene,
                          "emphasis": emphasis if emphasis in text.split() else "", "close": close,
                          "pause_after": pause})
    return shots
