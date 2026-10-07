"""One Hindi script from one fetched passage, checked locally against the lite lane's editorial rules."""

from __future__ import annotations

import json
import re

from .sources import Passage, quote_on_page

WORDS_MIN, WORDS_MAX = 100, 138
TARGET = "110-130"
BEATS_MIN, BEATS_MAX = 6, 10
HOOK_MAX_WORDS = 12
TITLE_CHARS = 95
# Openers the channel never uses: greetings, its own name, and worn-out phrasing.
BANNED_HOOK = ("क्या आप जानते", "क्या आपको पता", "कल्पना कीजिए", "कल्पना करें", "नमस्ते", "नमस्कार", "दोस्तों",
               "स्वागत", "मूल कथा", "आज हम", "आज की", "आइए")
TEXT_NAMES = ("रामायण", "महाभारत", "गीता", "पुराण")
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LATIN = re.compile(r"[A-Za-z]")

SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "title": {"type": "string"},
        "hook_text": {"type": "string"},
        "beats": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"},
            "evidence": {"type": "string"},
            "visual_query": {"type": "string"},
            "emphasis": {"type": "string"},
        }, "required": ["text", "evidence", "visual_query"]}},
        "description": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "shloka_verse": {"type": "string"},
        "key_quote": {"type": "string"},
        "names": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["usable", "title", "hook_text", "beats", "description", "hashtags", "keywords", "shloka_verse",
                 "key_quote", "names"],
}


def prompt(passage: Passage, angle: str, problems: list[str] | None = None, previous: dict | None = None,
           length_hint: str = "") -> str:
    has_sanskrit = any(v.sanskrit for v in passage.verses)
    lines = [
        "You write one Hindi YouTube Short / Instagram Reel script (45-60 seconds spoken) for the channel"
        " 'Mool Katha' (मूल कथा), which tells stories straight from the original Hindu texts.",
        f"Text: {passage.text_name_hi}. On-screen source line (fixed, do not change): {passage.citation_hi}.",
        f"Translation the passage comes from: {passage.translator}.",
        f"Suggested angle (use it only if the passage supports it; otherwise pick the most surprising true detail"
        f" in the passage): {angle}",
        "",
        "STRICT RULES",
        "1. Every factual statement must be directly supported by the PASSAGE below. Add nothing from memory,"
        " other versions (Tulsidas, TV serials, folk tellings) or later commentary. If a popular detail is not in the"
        " passage, leave it out. Never invent numbers, names, verse numbers or dialogue. Do not dramatise or"
        " intensify: no 'never', 'not even for a moment', 'all', 'first ever', superlatives, motives or feelings"
        " unless the passage itself says so. Each beat must say no more than its evidence quote says.",
        "2. Tell ONE story through a lesser-known true detail from this passage. Use the concrete details the"
        " passage gives (numbers, sizes, how many people, sounds, objects) wherever they carry the story.",
        "3. Beat 1 is the hook, spoken in the first 2 seconds: a sharp question or a surprising statement, at most"
        f" {HOOK_MAX_WORDS} words. Never start with a greeting, the channel name, 'क्या आप जानते', 'कल्पना कीजिए',"
        " 'आज हम' or 'आइए'. Build it on the most surprising concrete fact in the passage, as a 'why' or 'only'"
        " question the rest of the Short answers. The channel's best Reel (9,000 views, 2026-10-07) opened"
        " 'शिव धनुष टूटने पर केवल चार लोग ही होश में क्यों बचे?'. Hooks that only retell a famous story"
        " ('X ने Y कैसे पार की?') did far worse.",
        "4. The last beat closes the loop: it repeats the hook's question in the same or nearly the same words"
        " (for example 'यही कारण है कि लोग पूछते हैं, <the hook question>'), so the video replays into beat 1. Do not"
        " ask viewers to like, subscribe or follow in the narration.",
        f"5. Length: {TARGET} Hindi words in total across {BEATS_MIN}-{BEATS_MAX} beats; one or two short spoken"
        f" sentences per beat. Simple spoken Hindi (Devanagari only, no English words, no digits; write numbers as"
        f" Hindi words). {length_hint}",
        "6. Respectful toward every deity and sage. No gore, no sectarian comparison, no politics, no caste"
        " commentary.",
        "7. evidence: for every beat that states a fact, copy a short exact quote (8-30 words) from the PASSAGE"
        " (English translation or Sanskrit, character for character) that supports it. The hook may use \"\""
        " only if it is a question; the loop line may use \"\" only if it repeats an earlier fact.",
        "8. visual_query: 2-5 English words to find a public-domain Indian painting for the beat on Wikimedia"
        " Commons, naming the characters and scene (e.g. 'Hanuman Surasa', 'Rama breaking bow', 'Bhishma vow')."
        " Never ask for photos, idols or modern art.",
        "9. emphasis: one word copied from that beat's text to highlight, or \"\".",
        "10. title: a Hindi curiosity question that names the text (रामायण / महाभारत / गीता / पुराण), at most 90"
        " characters, ending with '?', that does NOT give away the answer.",
        "11. hook_text: a 2-6 word Hindi on-screen headline (a question) for the first frame, different from the"
        " title.",
        "12. description: 2-3 Hindi sentences for YouTube that say what the passage tells, without hype.",
        "13. hashtags: 3-5 relevant hashtags (lowercase Latin or Hindi, each starting with #, e.g. #ramayan).",
        "14. keywords: 5-8 search keywords (Hindi and English) for the Instagram caption.",
        ("15. shloka_verse: the printed verse number (exactly as shown in [brackets], e.g. 5-1-145) of the one"
         " Sanskrit verse that best carries the detail, or \"\"." if has_sanskrit else
         "15. shloka_verse: \"\" (this passage has no Sanskrit)."),
        "16. key_quote: one exact English sentence or clause (10-35 words) copied from the PASSAGE that best"
        " carries the detail.",
        "17. names: the Devanagari proper nouns in the script that the narrator must pronounce clearly.",
        "If the passage holds no clear, interesting story for a Short, return usable=false with empty fields.",
    ]
    if problems and previous:
        lines += ["", "YOUR PREVIOUS DRAFT FAILED THESE CHECKS; fix every one and return the full corrected JSON:",
                  *[f"- {p}" for p in problems], "PREVIOUS DRAFT:", json.dumps(previous, ensure_ascii=False)]
    lines += ["", "PASSAGE (source page: " + passage.url + ")", passage.writer_text()]
    return "\n".join(lines)


def word_count(script: dict) -> int:
    return sum(len(b.get("text", "").split()) for b in script.get("beats", []))


def problems(script: dict, passage: Passage) -> list[str]:
    """Everything the lite lane refuses in a script, in plain words the writer can fix."""
    found: list[str] = []
    if not script.get("usable"):
        return ["writer marked the passage unusable"]
    beats = script.get("beats") or []
    if not BEATS_MIN <= len(beats) <= BEATS_MAX:
        found.append(f"needs {BEATS_MIN}-{BEATS_MAX} beats, has {len(beats)}")
    words = word_count(script)
    if not WORDS_MIN <= words <= WORDS_MAX:
        found.append(f"needs {TARGET} Hindi words in total, has {words}")
    for i, beat in enumerate(beats, 1):
        text = beat.get("text", "")
        if not text.strip():
            found.append(f"beat {i} is empty")
            continue
        if _LATIN.search(text) or re.search(r"\d", text):
            found.append(f"beat {i} has Latin letters or digits; use Devanagari words only")
        if any(c in text for c in "{}\\<>"):
            found.append(f"beat {i} has a forbidden character")
        evidence = (beat.get("evidence") or "").strip()
        is_hook, is_loop = i == 1, i == len(beats)
        if evidence:
            if not quote_on_page(evidence, passage):
                found.append(f"beat {i}: its evidence is not an exact quote from the passage: {evidence[:80]!r}")
        elif not (is_hook and "?" in text) and not is_loop:
            found.append(f"beat {i} states something without an exact evidence quote")
        if not (beat.get("visual_query") or "").strip():
            found.append(f"beat {i} has no visual_query")
    if beats:
        hook = beats[0].get("text", "")
        if len(hook.split()) > HOOK_MAX_WORDS:
            found.append(f"beat 1 (the hook) has {len(hook.split())} words; at most {HOOK_MAX_WORDS}")
        if any(hook.strip().startswith(b) or b in hook for b in BANNED_HOOK):
            found.append("beat 1 uses a banned opener (greeting, channel name, 'क्या आप जानते', 'कल्पना कीजिए')")
    title = script.get("title", "")
    if "?" not in title:
        found.append("title must be a question ending with '?'")
    if not any(name in title for name in TEXT_NAMES):
        found.append("title must name the text (रामायण / महाभारत / गीता / पुराण)")
    if len(title) > TITLE_CHARS or any(c in title for c in "<>"):
        found.append(f"title must be at most {TITLE_CHARS} characters without < or >")
    hook_text = script.get("hook_text", "")
    if not 2 <= len(hook_text.split()) <= 7 or any(c in hook_text for c in "{}\\"):
        found.append("hook_text must be 2-6 Hindi words")
    tags = [t for t in script.get("hashtags") or [] if isinstance(t, str) and t.startswith("#") and " " not in t]
    if not 3 <= len(tags) <= 5:
        found.append("needs 3-5 hashtags, each starting with # and without spaces")
    verse = (script.get("shloka_verse") or "").strip()
    if verse and not any(v.number == verse for v in passage.verses):
        found.append(f"shloka_verse {verse!r} is not a verse number printed in the passage")
    quote = (script.get("key_quote") or "").strip()
    if quote and not quote_on_page(quote, passage):
        found.append("key_quote is not an exact quote from the passage")
    if not _DEVANAGARI.search(script.get("description", "")):
        found.append("description must be in Hindi")
    return found


def tidy(script: dict) -> dict:
    """Normalise small formatting differences the checks above allow."""
    script = dict(script)
    script["hashtags"] = [t.strip() for t in script.get("hashtags") or []
                          if isinstance(t, str) and t.strip().startswith("#") and " " not in t.strip()][:5]
    script["keywords"] = [k.strip() for k in script.get("keywords") or [] if isinstance(k, str) and k.strip()][:8]
    script["title"] = " ".join(script.get("title", "").split())
    script["hook_text"] = " ".join(script.get("hook_text", "").split())
    beats = []
    for beat in script.get("beats") or []:
        text = " ".join(str(beat.get("text", "")).split())
        emphasis = str(beat.get("emphasis") or "").strip()
        beats.append({"text": text, "evidence": str(beat.get("evidence") or "").strip(),
                      "visual_query": " ".join(str(beat.get("visual_query") or "").split()),
                      "emphasis": emphasis if emphasis and emphasis in text.split() else ""})
    script["beats"] = beats
    return script


def write(passage: Passage, angle: str, generate, *, length_hint: str = "", tries: int = 2) -> tuple[dict, list[str]]:
    """(script, remaining problems). `generate(prompt, schema)` is the LLM call; at most `tries` calls."""
    script: dict = {}
    found: list[str] = []
    for attempt in range(tries):
        text = prompt(passage, angle, found if attempt else None, script if attempt else None, length_hint)
        script = tidy(generate(text, SCHEMA))
        found = problems(script, passage)
        if not found or found == ["writer marked the passage unusable"]:
            break
    return script, found
