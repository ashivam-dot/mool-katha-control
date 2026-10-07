"""Looks at every picture before it can reach a Short: refuses nudity, sexual content and gore outright, and
refuses a picture that shows a different story, person or scene from the line it illustrates.

Titles and keywords can't see what a painting shows: on 2026-10-07 "Arjuna and river Nymph" (a bare-chested
nymph) and Parashurama fighting King Kartavirya Arjuna (a different Arjuna) both passed the title filter and
went out in a Gita Short, which the owner deleted. A vision model now judges the pixels, and when no model can
look, nothing is published.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Flash Lite (1,000 requests a day between its models) and Gemma read images on quota the scripts and narration
# don't use; Flash is the last resort. Five pictures a prompt suits every model in the ladder.
PER_PROMPT = 5
SIDE = 768

SCHEMA = {
    "type": "object",
    "properties": {
        "pictures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "shows": {"type": "string"},
                    "nudity": {"type": "boolean"},
                    "sexual": {"type": "boolean"},
                    "gore": {"type": "boolean"},
                    "fits": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["n", "shows", "nudity", "sexual", "gore", "fits", "reason"],
            },
        },
    },
    "required": ["pictures"],
}

PROMPT = """You are the picture editor of a family-friendly Hindi YouTube Shorts and Instagram Reels channel that
tells Hindu scripture stories to all ages. Each picture below will fill the screen while one narrated line plays.
Look at the pixels of each picture carefully, not only its title.

Story: {story}

For each picture answer:
- shows: one short sentence describing what is actually visible.
- nudity: true if any person or figure shows a bare female breast or nipple, bare buttocks or genitals, or is
  nude or nearly nude, including in classical paintings, sculpture-like figures, nymphs, apsaras or bathers.
  A bare-chested man in a dhoti, as warriors and sages are traditionally painted, is not nudity.
- sexual: true for embrace, kissing, lovemaking, seductive poses or a sexualised focus on the body.
- gore: true for severed heads or limbs, flowing wounds, mutilation or corpses shown in detail.
- fits: true only if the picture plausibly shows this line's scene or characters from THIS story. False if it
  shows a different story, a different person who shares a name, an unrelated scene, a page of text or a
  sheet of several unrelated pictures, or a modern object. A generic painting of the right characters is fine.
- reason: why, in a few words.

Pictures:
{lines}"""


def _small(path: Path) -> bytes:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image.thumbnail((SIDE, SIDE))
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def rejected(verdict: dict) -> str | None:
    """Why a verdict keeps its picture out of a Short, or None when the picture may be used."""
    for flag in ("nudity", "sexual", "gore"):
        if verdict.get(flag):
            return f"{flag}: {verdict.get('reason') or verdict.get('shows')}"
    if not verdict.get("fits"):
        return f"does not fit the line: {verdict.get('reason') or verdict.get('shows')}"
    return None


def judge(items: list[tuple[Path, str, str]], story: str, ask) -> list[dict]:
    """One verdict per (picture file, picture title, narrated line), in order. ask(prompt_parts, schema) answers
    like llm.generate; a missing or malformed verdict counts as a rejection."""
    verdicts: list[dict] = []
    for start in range(0, len(items), PER_PROMPT):
        chunk = items[start:start + PER_PROMPT]
        lines = "\n".join(f"{n}. title: {title!r}; narrated line: {line}" for n, (_, title, line) in enumerate(chunk, 1))
        parts: list = [PROMPT.format(story=story, lines=lines)]
        for n, (path, _, _) in enumerate(chunk, 1):
            parts += [f"Picture {n}:", _small(path)]
        answer = ask(parts, SCHEMA)
        got = {v.get("n"): v for v in (answer or {}).get("pictures", []) if isinstance(v, dict)}
        for n in range(1, len(chunk) + 1):
            verdict = got.get(n) or {"n": n, "shows": "", "nudity": False, "sexual": False, "gore": False,
                                     "fits": False, "reason": "no verdict"}
            verdicts.append(verdict)
    return verdicts
