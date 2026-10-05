"""The one source-support review: does the fetched passage support every factual sentence?"""

from __future__ import annotations

import json

from .sources import Passage

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "beat": {"type": "integer"},
            "factual": {"type": "boolean"},
            "supported": {"type": "boolean"},
            "note": {"type": "string"},
        }, "required": ["beat", "factual", "supported", "note"]}},
        "respectful": {"type": "boolean"},
        "title_reveals_answer": {"type": "boolean"},
        "summary": {"type": "string"},
    },
    "required": ["items", "respectful", "title_reveals_answer", "summary"],
}


def prompt(script: dict, passage: Passage) -> str:
    beats = [{"beat": i, "hindi": b["text"], "quoted_evidence": b.get("evidence", "")}
             for i, b in enumerate(script["beats"], 1)]
    return "\n".join([
        "You are a strict fact checker for a Hindi Short about a Hindu scripture. Judge ONLY against the PASSAGE.",
        "For each beat: factual = true if it asserts anything about the story (events, people, numbers, words"
        " spoken, causes). A pure question or a restatement of an earlier beat is not factual.",
        "supported = true only if the PASSAGE itself states it (a faithful Hindi paraphrase is fine; minor"
        " simplification is fine). Mark false if it adds a name, number, motive, sequence or detail the passage"
        " does not give, or if it comes from another version of the story. A beat that is not factual is supported.",
        "note: one short English reason; for a false item say exactly what is unsupported.",
        "respectful: false if anything is disrespectful to a deity or sage, gory, sectarian or political.",
        "title_reveals_answer: true if the title gives away the answer instead of raising curiosity.",
        "",
        f"TITLE: {script['title']}",
        f"ON-SCREEN HEADLINE: {script['hook_text']}",
        "BEATS:", json.dumps(beats, ensure_ascii=False, indent=1),
        "", f"PASSAGE ({passage.citation_hi}; {passage.translator}; {passage.url})", passage.writer_text(),
    ])


def verdict(result: dict, script: dict) -> tuple[bool, list[str]]:
    """Pass only when every beat was judged and every factual one is supported."""
    issues = []
    items = {int(i.get("beat", 0)): i for i in result.get("items") or [] if isinstance(i, dict)}
    for n, beat in enumerate(script["beats"], 1):
        item = items.get(n)
        if item is None:
            issues.append(f"beat {n} was not judged")
        elif item.get("factual") and not item.get("supported"):
            issues.append(f"beat {n} is not supported by the passage: {item.get('note', '')}")
    if result.get("respectful") is not True:
        issues.append("reviewer found disrespectful, gory, sectarian or political content")
    if result.get("title_reveals_answer") is True:
        issues.append("the title gives away the answer")
    return not issues, issues


def review(script: dict, passage: Passage, generate) -> tuple[bool, list[str], dict]:
    result = generate(prompt(script, passage), SCHEMA)
    ok, issues = verdict(result, script)
    return ok, issues, result
