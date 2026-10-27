"""One Hindi script from one fetched passage, checked locally against the lite lane's editorial rules."""

from __future__ import annotations

import json
import re

from .sources import Passage, quote_on_page

# 2026-10-13 research: Hindi mythology Shorts that reach millions run 60-136 seconds and tell one emotional story
# with a payoff; the lane's 48-second retellings got 1-50 YouTube views. About 140 spoken words a minute.
WORDS_MIN, WORDS_MAX = 130, 185
TARGET = "140-175"
BEATS_MIN, BEATS_MAX = 8, 12
HOOK_MAX_WORDS = 12
TITLE_CHARS = 80
SCENE_WORDS = (8, 70)
# Openers the channel never uses: greetings, its own name, and worn-out phrasing.
BANNED_HOOK = ("क्या आप जानते", "क्या आपको पता", "कल्पना कीजिए", "कल्पना करें", "नमस्ते", "नमस्कार", "दोस्तों",
               "स्वागत", "मूल कथा", "आज हम", "आज की", "आइए")
# The closing formula every Short used until 2026-10-13; the same sentence in every video reads as a template.
BANNED_ANYWHERE = ("यही कारण है कि लोग पूछते", "यही कारण है कि पूछा जाता", "लाइक", "सब्सक्राइब", "फॉलो")
TEXT_NAMES = ("रामायण", "महाभारत", "गीता", "पुराण")
# Scene words that make the image model paint wounds (the first AI dry run asked for "severing Ravana's ten arms").
GORY_SCENE = re.compile(r"\b(sever\w*|behead\w*|decapitat\w*|blood(?![- ]?red)\w*|bleed\w*|wounded|wounds|corpses?"
                        r"|gore|gory|dismember\w*|mutilat\w*|cut(?:ting)? off|chop\w*|torn off|tear(?:ing)? off"
                        r"|stab(?:s|bed|bing)?|slit|impal\w*)\b", re.IGNORECASE)
# How a Short ends, rotated so consecutive Shorts don't close the same way.
ENDINGS = {
    "loop": "The last beat brings back the hook's question in fresh words (not a formula), so the video replays"
            " naturally into beat 1.",
    "question": "The last beat asks the viewer one short, sincere question about the main character's choice at the"
                " story's climax (not about a side character), to answer in the comments (never ask for likes,"
                " subscriptions or follows, and never promise a next part).",
    "lesson": "The last beat says in one plain sentence what this moment shows, staying within what the passage"
              " itself shows (no sermon, no modern moral the passage doesn't carry).",
}
ENDING_ORDER = ("loop", "question", "lesson")
# CHANNEL.md red lines (caste and varna, diet and meat): words that start with a stem, and exact words where the
# stem would also catch ordinary words (वर्णन, जातिका).
RED_LINE_STEMS = ("शूद्र", "शुद्र", "वैश्य", "चांडाल", "चाण्डाल", "अछूत", "दलित", "मांस", "गोमांस")
RED_LINE_WORDS = ("वर्ण", "वर्णों", "जाति", "जातियों", "जातियाँ", "जात-पात")
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
            "scene": {"type": "string"},
            "close": {"type": "string"},
            "visual_query": {"type": "string"},
            "emphasis": {"type": "string"},
        }, "required": ["text", "evidence", "scene", "visual_query"]}},
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


def ending_for(published: int) -> str:
    return ENDING_ORDER[published % len(ENDING_ORDER)]


def prompt(passage: Passage, angle: str, problems: list[str] | None = None, previous: dict | None = None,
           length_hint: str = "", ending: str = "loop") -> str:
    has_sanskrit = any(v.sanskrit for v in passage.verses)
    lines = [
        "You write one Hindi YouTube Short / Instagram Reel script (60-80 seconds spoken) for the channel"
        " 'Mool Katha' (मूल कथा), which tells gripping stories straight from the original Hindu texts.",
        f"Text: {passage.text_name_hi}. Source line for the description (fixed, do not change): {passage.citation_hi}.",
        f"Translation the passage comes from: {passage.translator}.",
        f"Suggested angle (use it only if the passage supports it; otherwise pick the most moving or mysterious true"
        f" moment in the passage): {angle}",
        "",
        "STRICT RULES",
        "1. Every factual statement must be directly supported by the PASSAGE below. Add nothing from memory,"
        " other versions (Tulsidas, TV serials, folk tellings) or later commentary. If a popular detail is not in the"
        " passage, leave it out. Never invent numbers, names, verse numbers or dialogue. No 'never', 'all', 'first"
        " ever' or superlatives unless the passage says so. Feelings and motives only where the passage states them;"
        " Valmiki often does (wept, trembled, rejoiced, was furious), and those feelings carry the story, so use them."
        " Each beat must say no more than its evidence quote says.",
        "2. Tell ONE story with an emotional arc: hook, one line of who and where, rising tension, the turn, then"
        " the payoff that answers the hook, then the ending. Use the concrete details the passage gives (numbers,"
        " sizes, sounds, objects, exact words spoken).",
        "3. Beat 1 is the hook, spoken in the first 2 seconds, at most"
        f" {HOOK_MAX_WORDS} words: a 'क्यों' question about the most striking concrete fact in the passage, or a"
        " stakes statement that makes the viewer need the answer. Never start with a greeting, the channel name,"
        " 'क्या आप जानते', 'कल्पना कीजिए', 'आज हम' or 'आइए'. Strong: 'शिव धनुष टूटने पर केवल चार लोग ही होश में"
        " क्यों बचे?' (the channel's best Reel). Weak: hooks that only retell a famous story ('X ने Y कैसे पार की?').",
        "4. The payoff beat answers the hook explicitly with the passage's own detail. " + ENDINGS[ending]
        + " Do not use the sentence 'यही कारण है कि लोग पूछते हैं' or any like/subscribe/follow request.",
        f"5. Length: {TARGET} Hindi words in total across {BEATS_MIN}-{BEATS_MAX} beats; one or two short spoken"
        " sentences per beat. Write the way a gripping kathavachak talks to a friend: everyday spoken Hindi, short"
        " sentences, active voice; present tense is fine for drama ('हनुमान छलाँग लगाते हैं'). Avoid textbook and"
        " translation phrasing (तत्पश्चात, उपरोक्त, अत्यंत, chains of 'द्वारा') and news or word-for-word phrasing"
        " ('समाचार सुना है', 'की ओर मुख किया'). Dialogue sounds like a person speaking aloud: 'सुना है तुमने शिव का"
        " धनुष तोड़ा है!', not 'मैंने शिव धनुष टूटने का समाचार सुना है।'. Devanagari only, no English words, no"
        f" digits; write numbers as Hindi words. {length_hint}",
        "6. Respectful toward every deity and sage. No gore, no sectarian comparison, no politics, no caste or"
        " varna detail (not even when the passage names someone's varna), no diet or meat.",
        "7. evidence: for every beat that states a fact, copy a short exact quote (8-30 words) from the PASSAGE"
        " (English translation or Sanskrit, character for character) that supports it. The hook may use \"\""
        " only if it is a question; the last beat may use \"\" only if it repeats an earlier fact or asks the viewer.",
        "8. scene: in English, 20-50 words describing ONE painting for this beat: who is in it (use these English"
        " names: Rama, Sita, Lakshmana, Hanuman, Ravana, Dasharatha, Bharata, Vishvamitra, Vasishtha, Jatayu,"
        " Sugriva, Vali, Shiva, Ganga, Krishna, Arjuna, Bhishma, Draupadi, and so on), what they are doing at this"
        " exact moment, the expression of EVERY person in it as fits the moment (a captive is frightened or"
        " grieving, never smiling), the setting, the light and time of day, and the framing (close-up, medium or"
        " wide). Beat 1's scene is a close-up of the face at the heart of the hook. Vary framing between"
        " beats. Show only what the line and passage say. For violence show the moment before or after, or a"
        " symbol (an arrow in flight, the glow of fire), never wounds. Deities and sages always dignified, never"
        " comic. No text or letters in the picture.",
        "8b. close: for every beat after the hook, in English, 15-40 words describing a SECOND painting shown while"
        " the second half of the line is spoken: a close-up of the one face, pair of hands or object that half"
        " turns on (Sita's tear-filled eyes, Hanuman's hand opening on the ring, the bow snapping), with the exact"
        " expression. Same characters as the scene, the same rules, and never the same framing as the scene.",
        "9. visual_query: 2-5 English words to find a public-domain Indian painting for the beat on Wikimedia"
        " Commons, naming the characters and scene (e.g. 'Hanuman Surasa', 'Rama breaking bow', 'Bhishma vow').",
        "10. emphasis: one word copied from that beat's text to highlight, or \"\".",
        f"11. title: Hindi, at most {TITLE_CHARS - 10} characters, ending with '?' or '!' (optionally followed by one"
        " fitting emoji such as 🙏 🔱 🚩 😢 😱). Front-load the emotional or mysterious detail and name the text at"
        " the end after ' | ', e.g. 'हनुमान ने खुद को ब्रह्मास्त्र से क्यों बँधने दिया? | रामायण'. Make the"
        " viewer curious; never give away the answer.",
        "12. hook_text: a 2-6 word Hindi on-screen headline for the first frame, different from the title, that"
        " teases the mystery or the emotion.",
        "13. description: 2-3 Hindi sentences for YouTube that say what the passage tells, without hype.",
        "14. hashtags: 3-5 relevant hashtags (lowercase Latin or Hindi, each starting with #, e.g. #ramayan).",
        "15. keywords: 5-8 search keywords (Hindi and English) for the Instagram caption.",
        ("16. shloka_verse: the printed verse number (exactly as shown in [brackets], e.g. 5-1-145) of the one"
         " Sanskrit verse that best carries the detail, or \"\"." if has_sanskrit else
         "16. shloka_verse: \"\" (this passage has no Sanskrit)."),
        "17. key_quote: one exact English sentence or clause (10-35 words) copied from the PASSAGE that best"
        " carries the detail.",
        "18. names: the Devanagari proper nouns in the script that the narrator must pronounce clearly.",
        "If the passage holds no clear, moving story for a Short, return usable=false with empty fields.",
    ]
    if problems and previous:
        lines += ["", "YOUR PREVIOUS DRAFT FAILED THESE CHECKS; fix every one and return the full corrected JSON:",
                  *[f"- {p}" for p in problems], "PREVIOUS DRAFT:", json.dumps(previous, ensure_ascii=False)]
    lines += ["", "PASSAGE (source page: " + passage.url + ")", passage.writer_text()]
    return "\n".join(lines)


def word_count(script: dict) -> int:
    return sum(len(b.get("text", "").split()) for b in script.get("beats", []))


def red_lines(script: dict) -> list[str]:
    """Red-line words in anything the viewer sees or hears (narration, title, headline, description)."""
    parts = [(f"beat {i}", b.get("text", "")) for i, b in enumerate(script.get("beats") or [], 1)]
    parts += [(field, script.get(field) or "") for field in ("title", "hook_text", "description")]
    found = []
    for where, text in parts:
        words = [w.strip("।॥,.?!:;\"'()‘’“”-") for w in str(text).split()]
        hits = [w for w in words if w and (w.startswith(RED_LINE_STEMS) or w in RED_LINE_WORDS)]
        if hits:
            found.append(f"{where} touches a red-line topic (caste or diet: {hits[0]!r}); leave that detail out")
    return found


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
        if any(b in text for b in BANNED_ANYWHERE):
            found.append(f"beat {i} uses a worn-out formula or a like/subscribe request; say it in fresh words")
        evidence = (beat.get("evidence") or "").strip()
        is_hook, is_last = i == 1, i == len(beats)
        if evidence:
            if not quote_on_page(evidence, passage):
                found.append(f"beat {i}: its evidence is not an exact quote from the passage: {evidence[:80]!r}")
        elif not (is_hook and "?" in text) and not is_last:
            found.append(f"beat {i} states something without an exact evidence quote")
        scene = beat.get("scene") or ""
        if not SCENE_WORDS[0] <= len(scene.split()) <= SCENE_WORDS[1] or _DEVANAGARI.search(scene):
            found.append(f"beat {i} needs an English scene of {SCENE_WORDS[0]}-{SCENE_WORDS[1]} words")
        elif gory := GORY_SCENE.search(scene):
            found.append(f"beat {i} scene shows violence directly ({gory.group(0).lower()!r}); paint the moment"
                         " before or after, or"
                         " a symbol such as an arrow in flight or the glow of fire, with no injury visible")
        close = beat.get("close") or ""
        if close and (not SCENE_WORDS[0] <= len(close.split()) <= SCENE_WORDS[1] or _DEVANAGARI.search(close)):
            found.append(f"beat {i} close needs an English close-up of {SCENE_WORDS[0]}-{SCENE_WORDS[1]} words")
        elif close and (gory := GORY_SCENE.search(close)):
            found.append(f"beat {i} close shows violence directly ({gory.group(0).lower()!r}); show a face or a"
                         " symbol instead, with no injury visible")
        if not (beat.get("visual_query") or "").strip():
            found.append(f"beat {i} has no visual_query")
    if beats:
        hook = beats[0].get("text", "")
        if len(hook.split()) > HOOK_MAX_WORDS:
            found.append(f"beat 1 (the hook) has {len(hook.split())} words; at most {HOOK_MAX_WORDS}")
        if any(hook.strip().startswith(b) or b in hook for b in BANNED_HOOK):
            found.append("beat 1 uses a banned opener (greeting, channel name, 'क्या आप जानते', 'कल्पना कीजिए')")
    title = script.get("title", "")
    if "?" not in title and "!" not in title:
        found.append("title must be a question ending with '?' (or an exclamation ending with '!')")
    if not any(name in title for name in TEXT_NAMES):
        found.append("title must name the text (रामायण / महाभारत / गीता / पुराण)")
    if len(title) > TITLE_CHARS or any(c in title for c in "<>"):
        found.append(f"title must be at most {TITLE_CHARS - 10} characters without < or >")
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
    found += red_lines(script)
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
                      "scene": " ".join(str(beat.get("scene") or "").split()),
                      "close": " ".join(str(beat.get("close") or "").split()),
                      "visual_query": " ".join(str(beat.get("visual_query") or "").split()),
                      "emphasis": emphasis if emphasis and emphasis in text.split() else ""})
    script["beats"] = beats
    return script


def write(passage: Passage, angle: str, generate, *, length_hint: str = "", tries: int = 2,
          ending: str = "loop") -> tuple[dict, list[str]]:
    """(script, remaining problems). `generate(prompt, schema)` is the LLM call; at most `tries` calls."""
    script: dict = {}
    found: list[str] = []
    for attempt in range(tries):
        text = prompt(passage, angle, found if attempt else None, script if attempt else None, length_hint, ending)
        script = tidy(generate(text, SCHEMA))
        found = problems(script, passage)
        if not found or found == ["writer marked the passage unusable"]:
            break
    return script, found
