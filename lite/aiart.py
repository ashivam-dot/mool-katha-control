"""The AI paintings the paint jobs made for a planned Short: collected, screened by the vision model, and turned
into the pictures list build_spec takes. When too few pass, the Short falls back to museum paintings."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from . import vision

log = logging.getLogger(__name__)

# A Short goes out with AI paintings only when at least this share of its beats has its own passing picture;
# the other beats repeat the nearest earlier one.
MIN_SHARE = 0.6
# The paint jobs render 576x1024; the plate is cropped from a 1080x1920 frame with room for camera moves.
UPSCALE = (1152, 2048)
CREDIT = {"source": "AI (Z-Image-Turbo, Apache-2.0)", "title": "AI painting made for this Short",
          "credit": "Mool Katha", "license": "AI-generated", "url": "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo"}


def collect(art_dir: Path | None, episode: str, beats: int) -> dict[int, Path]:
    """Beat number -> painted file, for pictures painted for this episode (a stale artifact never leaks in)."""
    found: dict[int, Path] = {}
    if not art_dir or not art_dir.is_dir():
        return found
    for meta in sorted(art_dir.rglob("beat*.json")):
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        image = meta.with_suffix(".png")
        if info.get("episode") == episode and 0 <= int(info.get("beat", -1)) < beats and image.exists():
            found[int(info["beat"])] = image
    return found


def _prepare(source: Path, target: Path) -> None:
    from PIL import Image, ImageFilter

    image = Image.open(source).convert("RGB")
    if image.size != UPSCALE:
        image = image.resize(UPSCALE, Image.Resampling.LANCZOS).filter(
            ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))
    image.save(target, quality=94)


def pictures(found: dict[int, Path], beats: list[dict], folder: Path, story: str, ask,
             report: list[dict] | None = None) -> list[dict | int]:
    """The pictures list for build_spec: a dict per beat with its own passing picture, else the 1-based number of
    the nearest earlier beat to repeat. Raises LookupError when too few pass (the caller uses museum paintings)."""
    folder.mkdir(parents=True, exist_ok=True)
    items, order = [], []
    for i in sorted(found):
        target = folder / f"ai{i:02d}.jpg"
        _prepare(found[i], target)
        items.append((target, "AI painting", beats[i]["text"]))
        order.append(i)
    verdicts = vision.judge(items, story, ask, generated=True) if items else []
    accepted: dict[int, dict] = {}
    for i, (path, _, _), verdict in zip(order, items, verdicts):
        why = vision.rejected(verdict, generated=True)
        if report is not None:
            report.append({"beat": i + 1, "key": None, "title": "AI painting", "url": "", "shows": verdict.get("shows"),
                           "unsafe": any(verdict.get(f) for f in ("nudity", "sexual", "gore")), "rejected": why})
        if why is None:
            accepted[i] = {"path": f"{folder.name}/{path.name}", "url": "", "credit": dict(CREDIT), "ai": True}
        else:
            log.warning("AI picture for beat %d refused: %s", i + 1, why)
    need = max(3, round(MIN_SHARE * len(beats)))
    if len(accepted) < need or 0 not in accepted:
        raise LookupError(f"only {len(accepted)} of {len(beats)} AI pictures passed (need {need} and the hook's)")
    result: list[dict | int] = []
    for i in range(len(beats)):
        if i in accepted:
            result.append(accepted[i])
        else:
            result.append(next(n for n in range(i, 0, -1) if isinstance(result[n - 1], dict)))
    return result


def copy_report(found: dict[int, Path], out: Path) -> None:
    """Keep the raw paintings with the run's artifact, for review."""
    target = out / "ai-art"
    target.mkdir(parents=True, exist_ok=True)
    for i, path in found.items():
        shutil.copyfile(path, target / path.name)
        meta = path.with_suffix(".json")
        if meta.exists():
            shutil.copyfile(meta, target / meta.name)
