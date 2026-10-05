"""The lite lane's one lightweight QA on the finished MP4: speech coverage, duration and loudness.

(Source support is judged before rendering by review.py; the gate records that verdict alongside these checks.)
"""

from __future__ import annotations

import difflib
import json
import re
import subprocess
from pathlib import Path

MIN_COVERAGE = 0.85
DURATION = (42.0, 62.0)
LOUDNESS_LUFS = (-17.0, -11.0)
MAX_TRUE_PEAK = -0.5
# A word the recognizer spelled a little differently still counts (vowel length, nukta, half letters).
FUZZY_RATIO = 0.75


def _keys(text: str) -> list[str]:
    from ytc.hindi import letter_words

    return letter_words(text)[1]


def coverage(script: str, heard: str) -> dict:
    """Share of the script's words found in the transcript, in order, after Hindi spelling folding."""
    want, got = _keys(script), _keys(heard)
    if not want:
        return {"coverage": 0.0, "words": 0, "matched": 0, "missed": []}
    matched = [False] * len(want)
    matcher = difflib.SequenceMatcher(None, want, got, autojunk=False)
    for a, _b, size in matcher.get_matching_blocks():
        for k in range(size):
            matched[a + k] = True
    # Unmatched script words: accept a close spelling among nearby transcript words.
    opcodes = matcher.get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag not in ("replace", "delete"):
            continue
        window = got[max(j1 - 2, 0):j2 + 2]
        for i in range(i1, i2):
            if any(difflib.SequenceMatcher(None, want[i], w).ratio() >= FUZZY_RATIO for w in window):
                matched[i] = True
            elif len(want[i]) > 3 and any(want[i] in w or w in want[i] for w in window if len(w) > 2):
                matched[i] = True
    words, _ = letter_words_visible(script)
    missed = [words[i] for i, ok in enumerate(matched) if not ok][:30]
    share = sum(matched) / len(want)
    return {"coverage": round(share, 3), "words": len(want), "matched": sum(matched), "missed": missed}


def letter_words_visible(text: str) -> tuple[list[str], list[str]]:
    from ytc.hindi import letter_words

    return letter_words(text)


def transcribe(media: Path) -> str:
    """faster-whisper's Hindi transcript of the final file (the same model the narration aligner uses)."""
    from ytc import tts

    return " ".join(word for word, _start, _end in tts._heard(media, "hi"))


def probe_duration(media: Path) -> float:
    from ytc import ff

    return float(ff.duration(media))


def loudness(media: Path) -> dict:
    from ytc import ff

    proc = subprocess.run([ff.exe(), "-hide_banner", "-nostats", "-nostdin", "-i", str(media), "-af",
                           "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"loudness measurement failed: {proc.stderr[-500:]}")
    return parse_ebur128(proc.stderr)


def parse_ebur128(log: str) -> dict:
    summary = log[log.rfind("Summary:"):] if "Summary:" in log else log
    integrated = re.search(r"I:\s*(-?[\d.]+)\s*LUFS", summary)
    peak = re.search(r"Peak:\s*(-?[\d.]+|-inf)\s*dBFS", summary)
    if not integrated or not peak:
        raise RuntimeError("could not read the loudness summary")
    tp = float("-inf") if peak.group(1) == "-inf" else float(peak.group(1))
    return {"integrated_lufs": float(integrated.group(1)), "true_peak_dbfs": tp}


def evaluate(*, script: str, heard: str, duration: float, loud: dict, review_ok: bool,
             review_issues: list[str] | None = None) -> dict:
    """The gate decision from measured values (pure, so it is unit-tested without media)."""
    speech = coverage(script, heard)
    checks = {
        "source_support": {"ok": bool(review_ok), "issues": list(review_issues or [])},
        "speech_coverage": {"ok": speech["coverage"] >= MIN_COVERAGE, **speech, "minimum": MIN_COVERAGE},
        "duration": {"ok": DURATION[0] <= duration <= DURATION[1], "seconds": round(duration, 2),
                     "range": list(DURATION)},
        "loudness": {"ok": LOUDNESS_LUFS[0] <= loud["integrated_lufs"] <= LOUDNESS_LUFS[1]
                     and loud["true_peak_dbfs"] <= MAX_TRUE_PEAK, **loud,
                     "range_lufs": list(LOUDNESS_LUFS), "max_true_peak": MAX_TRUE_PEAK},
    }
    return {"passed": all(c["ok"] for c in checks.values()), "checks": checks}


def run(media: Path, script: str, review_ok: bool, review_issues: list[str] | None = None) -> dict:
    result = evaluate(script=script, heard=transcribe(media), duration=probe_duration(media),
                      loud=loudness(media), review_ok=review_ok, review_issues=review_issues)
    return result


def save(result: dict, path: Path) -> None:
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
