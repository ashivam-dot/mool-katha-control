"""One lite run: at most one new Short, scheduled into the next free lite slot (or rendered only, in dry-run)."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import art, gate, review, slots, sources, writer

log = logging.getLogger("lite")

CONTROL_ROOT = Path(__file__).resolve().parents[1]
TOPICS = Path(__file__).with_name("topics.json")
LEDGER = Path(os.environ.get("LITE_LEDGER", CONTROL_ROOT / "lite-state" / "ledger.json"))
MAX_GEMINI_CALLS = 15
MAX_GATE_ATTEMPTS = 2
MAX_REPAIRS = 2
# Gemini TTS takes a Short usually needs; script repairs never spend these.
TTS_RESERVE = 4
# Runs a topic may fail at scripting or source review before it is set aside for good.
MAX_TOPIC_STRIKES = 2
# With Flash spent, OVH's shared Qwen stayed rate-limited for over an hour (2026-10-06) and the job timed out at
# 80 minutes; the lane gives up early and a later run (or the 07:00 UTC quota reset) picks the topic up again.
TEXT_WAIT_PER_CALL = 480
TEXT_WAIT_PER_RUN = 2400
# A picture the models won't answer about (an empty, safety-blocked reply) looks like an overload to llm.generate;
# the check gives up after this long and nothing is published.
VISION_WAIT = 300
VOICE = "Sulafat"
# Sargas per kanda on valmikiramayan.net, for picking fresh passages once the curated topics are used.
VALMIKI_SARGAS = {1: 77, 2: 119, 3: 75, 4: 67, 5: 68, 6: 128}
MOTIONS = ("zoom_in", "pan_left", "zoom_out", "pan_right", "pan_up", "zoom_in", "pan_down", "pan_left", "zoom_out")
CAPTIONS = {"font": "Mukta ExtraBold", "font_file": "Mukta-ExtraBold.ttf", "size": 128, "max_width": 0.74,
            "primary": "#FFFFFF", "highlight": "#FFD400", "accent": "#00E5FF", "outline": 9, "words_per_line": 3,
            "uppercase": True, "y": 0.69, "hook_font": "Rozha One", "hook_font_file": "RozhaOne-Regular.ttf",
            "hook_size": 72, "hook_y": 0.08, "citation_size": 36, "citation_y": 0.128}


class WaitForQuota(RuntimeError):
    """Gemini (or every fallback) is out of today's quota; try again on a later run without losing work."""


class Budget(RuntimeError):
    pass


# --- ledger -----------------------------------------------------------------------------------------------------

def load_ledger(path: Path = LEDGER) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"episodes": [], "skipped": {}, "pending": None, "ready": None}


def save_ledger(ledger: dict, path: Path = LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def used_keys(ledger: dict) -> set[str]:
    keys = {e["topic_key"] for e in ledger["episodes"]} | set(ledger.get("skipped") or {})
    for slot in ("pending", "ready"):
        if ledger.get(slot):
            keys.add(ledger[slot]["topic_key"])
    return keys


def next_topic(ledger: dict, rng: random.Random | None = None, exclude: set[str] = frozenset()) -> dict:
    used = used_keys(ledger) | set(exclude)
    for topic in json.loads(TOPICS.read_text(encoding="utf-8")):
        if topic["key"] not in used:
            return topic
    rng = rng or random.Random()
    for _ in range(200):
        kanda = rng.choice(list(VALMIKI_SARGAS))
        sarga = rng.randint(1, VALMIKI_SARGAS[kanda])
        key = f"ramayana-{kanda}-{sarga}"
        if key not in used:
            return {"key": key, "work": "ramayana", "book": kanda, "chapter": sarga,
                    "angle": "the most surprising lesser-known true detail in this sarga"}
    raise RuntimeError("no unused topic left")


# --- Gemini accounting ------------------------------------------------------------------------------------------

class Calls:
    """Counts billed Gemini requests (text and TTS) and the free fallback models that answered."""

    def __init__(self) -> None:
        self.gemini_text = 0
        self.gemini_tts = 0
        self.fallback = 0
        self.vision_calls = 0
        self._wrapped = False
        self._text_deadline = time.monotonic() + TEXT_WAIT_PER_RUN

    @property
    def gemini(self) -> int:
        return self.gemini_text + self.gemini_tts

    def check(self, need: int) -> None:
        if self.gemini + need > MAX_GEMINI_CALLS:
            raise Budget(f"Gemini budget: {self.gemini} used, {need} more would pass {MAX_GEMINI_CALLS}")

    def generate(self, purpose: str):
        from ytc import llm

        # Flash Lite and the light backups wrote Hindi scripts far under length with misquoted evidence
        # (2026-10-06), so with Flash spent the lane waits for quota rather than burn a topic on them.
        models = (*llm.FLASH, *(llm.JUDGES if "review" in purpose else llm.BACKUP_STRONG))

        def call(prompt: str, schema: dict):
            if time.monotonic() > self._text_deadline:
                raise WaitForQuota(f"text models stayed busy for {TEXT_WAIT_PER_RUN // 60} minutes this run")
            llm.OVERLOAD_WAIT = min(llm.OVERLOAD_WAIT, TEXT_WAIT_PER_CALL)
            before = len(llm.calls)
            try:
                return llm.generate(prompt, schema=schema, models=models, temperature=0.4, purpose=purpose)
            except (llm.OutOfQuota, llm.Overloaded) as err:
                raise WaitForQuota(str(err)) from err
            finally:
                for entry in llm.calls[before:]:
                    if ":" in entry["model"] or entry["model"].startswith("cursor/"):
                        self.fallback += 1
                    else:
                        self.gemini_text += 1
        return call

    def vision(self):
        """The picture check's model call: Flash Lite and Gemma first, on quota the script and narration don't
        use, then the free vision backups. Not counted against MAX_GEMINI_CALLS."""
        from ytc import llm

        models = (*llm.FLASH_LITE, *llm.GEMMA, *(m for m, spec in llm.BACKUPS.items() if spec.get("vision")))

        def call(prompt: list, schema: dict):
            llm.OVERLOAD_WAIT = min(llm.OVERLOAD_WAIT, VISION_WAIT)
            before = len(llm.calls)
            try:
                return llm.generate(prompt, schema=schema, models=models, temperature=0.0, purpose="lite picture check")
            except (llm.OutOfQuota, llm.Overloaded) as err:
                raise WaitForQuota(f"picture check: {err}") from err
            finally:
                self.vision_calls += len(llm.calls) - before
        return call

    def wrap_tts(self) -> None:
        from ytc import tts

        if self._wrapped:
            return
        original = tts._gemini_take

        def counted(*args, **kwargs):
            result = original(*args, **kwargs)
            self.gemini_tts += 1
            return result

        tts._gemini_take = counted
        self._wrapped = True

    def to_json(self) -> dict:
        return {"gemini_text": self.gemini_text, "gemini_tts": self.gemini_tts, "gemini_total": self.gemini,
                "free_fallback": self.fallback, "picture_check": self.vision_calls}


# --- episode ----------------------------------------------------------------------------------------------------

def direction(names: list[str]) -> str:
    named = ", ".join(n for n in names if n.strip())[:200]
    return ("Natural fluent Indian Hindi. A warm, respectful kathavachak telling a sacred story to family. Clear,"
            " engaged delivery at about 140 Hindi words per minute, with a brief natural pause after each sentence."
            + (f" Pronounce {named} clearly." if named else "")
            + " Speak only the exact Devanagari transcript; add no introduction or explanation.")


def build_spec(ep_id: str, script: dict, passage: sources.Passage, pictures: list) -> dict:
    beats = []
    for i, (beat, pic) in enumerate(zip(script["beats"], pictures)):
        motion = "zoom_out" if i == len(script["beats"]) - 1 else MOTIONS[i % len(MOTIONS)]
        if isinstance(pic, int):
            visual = {"reuse": pic, "motion": motion}
        else:
            visual = ({"source": "file", "path": pic["path"]} if pic.get("path") else {"source": "url", "url": pic["url"]})
            visual.update({"credit": pic["credit"], "motion": motion})
        beats.append({"text": beat["text"], "emphasis": [beat["emphasis"]] if beat.get("emphasis") else [],
                      "visual": visual, "pause_after": 0.12})
    return {
        "id": ep_id, "title": script["title"], "series": "मूल कथा लाइट", "description": script["description"],
        "hook_text": script["hook_text"], "citation": passage.citation_hi, "sources": [passage.url],
        "hashtags": script["hashtags"], "tags": script["keywords"], "category_id": "27", "synthetic_media": False,
        "depth_motion": False,
        "voice": ({"engine": "kokoro", "voice": "hf_alpha", "lang_code": "h", "lead_in": 0.35}
                  if os.environ.get("LITE_LOCAL_KOKORO") == "1" else
                  {"engine": "gemini", "voice": VOICE, "lang_code": "h", "lead_in": 0.35,
                   "model": "gemini-3.8-flash-tts", "direction": direction(script.get("names") or [])}),
        "music": {"path": "tanpura", "gain_db": -29.0},
        "captions": CAPTIONS,
        "beats": beats,
    }


def episode_record(ep_id: str, topic: dict, script: dict, passage: sources.Passage, pictures: list) -> dict:
    verse = next((v for v in passage.verses if script.get("shloka_verse") and v.number == script["shloka_verse"]), None)
    return {
        "id": ep_id, "topic_key": topic["key"], "title": script["title"], "description": script["description"],
        "citation": passage.citation_hi, "source_url": passage.url, "translator": passage.translator,
        "hashtags": script["hashtags"], "keywords": script["keywords"], "key_quote": script.get("key_quote", ""),
        "shloka": {"number": verse.number, "sanskrit": verse.sanskrit} if verse and verse.sanskrit else None,
        "pictures": [{"url": p["url"], "credit": p["credit"]} for p in pictures if isinstance(p, dict)],
    }


def story(script: dict, passage: sources.Passage) -> str:
    """What the picture check needs to know: which work and passage, the title and the people in it."""
    names = ", ".join(n for n in script.get("names") or [] if n.strip())
    return (f"{passage.work.replace('_', ' ').title()}, {passage.page_title}. Short title: {script['title']}."
            + (f" People in it: {names}." if names else ""))


def frames(video: Path, out: Path) -> list[str]:
    from ytc import ff

    duration = float(ff.duration(video))
    saved = []
    for name, at in (("frame-hook.jpg", 0.6), ("frame-middle.jpg", duration / 2), ("frame-end.jpg", duration - 2)):
        subprocess.run([ff.exe(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{max(at, 0):.2f}", "-i",
                        str(video), "-frames:v", "1", "-q:v", "3", str(out / name)], check=True)
        saved.append(name)
    return saved


def prepare(topic: dict, calls: Calls, length_hint: str = "") -> tuple[dict, sources.Passage, dict]:
    """Script and source-support review for one topic: (script, passage, review record). Raises ValueError
    when the topic can't make a supported script."""
    passage = sources.fetch(topic)
    calls.check(2)
    script, problems = writer.write(passage, topic.get("angle", ""), calls.generate("lite script"),
                                    length_hint=length_hint)
    if problems:
        raise ValueError("script: " + "; ".join(problems[:6]))
    calls.check(1)
    ok, issues, result = review.review(script, passage, calls.generate("lite source review"))
    for _ in range(MAX_REPAIRS):
        if ok or calls.gemini + 2 + TTS_RESERVE > MAX_GEMINI_CALLS:
            break
        log.info("source review rejected the draft; repairing: %s", "; ".join(issues[:4]))
        script, problems = writer.write(passage, topic.get("angle", ""), _repair(calls, issues, script), tries=1,
                                        length_hint=length_hint)
        if problems:
            ok, issues = False, [*issues, *problems]
            continue
        ok, issues, result = review.review(script, passage, calls.generate("lite source review"))
    if not ok:
        raise ValueError("source review: " + "; ".join(issues[:6]))
    return script, passage, {"ok": True, "issues": [], "result": result}


def _repair(calls: Calls, issues: list[str], previous: dict):
    """A writer call that sees the reviewer's objections (one call)."""
    base = calls.generate("lite script repair")

    def call(prompt: str, schema: dict):
        note = ("\n\nTHE FACT CHECKER REJECTED YOUR PREVIOUS DRAFT. Remove or correct every unsupported statement"
                " and return the full corrected JSON.\nObjections:\n" + "\n".join(f"- {i}" for i in issues)
                + "\nPREVIOUS DRAFT:\n" + json.dumps(previous, ensure_ascii=False))
        return base(prompt + note, schema)
    return call


def produce(ep_id: str, topic: dict, ledger: dict, calls: Calls, producer_root: Path, out: Path) -> dict:
    """Render and gate one Short; returns the episode record with its MP4 path. Raises WaitForQuota, ValueError."""
    from ytc import render, tts

    pending = ledger.get("pending")
    resumed = bool(pending and pending["topic_key"] == topic["key"])
    if resumed:
        script, review_record, ep_id = pending["script"], pending["review"], pending["id"]
        passage = sources.fetch(topic)
    else:
        script, passage, review_record = prepare(topic, calls)
    blocked = set(ledger.get("blocked_pictures") or [])
    pictures = art.choose(script["beats"], passage.work, blocked)
    ledger["pending"] = {"id": ep_id, "topic_key": topic["key"], "script": script, "review": review_record,
                         "gate_attempts": pending.get("gate_attempts", 0) if resumed else 0,
                         **({"replaces": pending["replaces"]} if resumed and pending.get("replaces") else {})}
    for attempt in range(2):
        work = producer_root / "lite-work" / ep_id
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        pictures = art.download(pictures, work / "pics")
        checked: list[dict] = []
        try:
            pictures = art.screen(pictures, script["beats"], work / "pics", story(script, passage),
                                  calls.vision(), blocked, report=checked)
        finally:
            out.mkdir(parents=True, exist_ok=True)
            (out / "pictures.json").write_text(json.dumps(checked, ensure_ascii=False, indent=1), encoding="utf-8")
            if unsafe := {r["key"] for r in checked if r["unsafe"] and r["key"]}:
                blocked |= unsafe
                ledger["blocked_pictures"] = sorted(blocked)
        spec = build_spec(ep_id, script, passage, pictures)
        (work / "short.yaml").write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
        calls.wrap_tts()
        calls.check(1)
        try:
            video = render.make_short(work / "short.yaml")
        except tts.TTSRejected as err:
            raise ValueError(f"narration rejected: {err}") from err
        except RuntimeError as err:
            if "429" in str(err) or "quota" in str(err).lower():
                raise WaitForQuota(f"Gemini TTS: {err}") from err
            raise
        spoken = "\n".join(b["text"] for b in script["beats"])
        result = gate.run(video, spoken, review_record["ok"], review_record["issues"])
        duration = result["checks"]["duration"]["seconds"]
        if result["passed"] or result["checks"]["duration"]["ok"] or attempt or calls.gemini + 5 > MAX_GEMINI_CALLS:
            break
        # Wrong length: one shorter or longer rewrite (writer + review), then a fresh narration.
        words = writer.word_count(script)
        target = max(writer.WORDS_MIN, min(writer.WORDS_MAX, int(words * 52.0 / max(duration, 1.0))))
        hint = f"The previous draft ran {duration:.0f} seconds with {words} words; write about {target} words."
        log.info("rewriting for length: %s", hint)
        script, passage, review_record = prepare(topic, calls, length_hint=hint)
        pictures = art.choose(script["beats"], passage.work, blocked)
        ledger["pending"].update({"script": script, "review": review_record})
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"{ep_id}.mp4"
    shutil.copyfile(video, final)
    shutil.copyfile(work / "short.yaml", out / "short.yaml")
    for name in ("manifest.json", "captions.srt", "take.json"):
        if (work / "work" / name).exists():
            shutil.copyfile(work / "work" / name, out / name)
    gate.save(result, out / "gate.json")
    (out / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "passage.json").write_text(json.dumps(passage.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "review.json").write_text(json.dumps(review_record, ensure_ascii=False, indent=2), encoding="utf-8")
    record = episode_record(ep_id, topic, script, passage, pictures)
    if ledger["pending"].get("replaces"):
        record["replaces"] = ledger["pending"]["replaces"]
    record.update({"gate": {"passed": result["passed"],
                            **{k: v["ok"] for k, v in result["checks"].items()},
                            "speech_coverage": result["checks"]["speech_coverage"]["coverage"],
                            "seconds": duration,
                            "lufs": result["checks"]["loudness"]["integrated_lufs"]},
                   "frames": frames(final, out), "video": str(final)})
    if not result["passed"]:
        ledger["pending"]["gate_attempts"] += 1
        if ledger["pending"]["gate_attempts"] >= MAX_GATE_ATTEMPTS:
            ledger.setdefault("skipped", {})[topic["key"]] = "gate failed twice"
            ledger["pending"] = None
        failed = [k for k, v in result["checks"].items() if not v["ok"]]
        raise ValueError(f"gate failed: {', '.join(failed)} ({json.dumps(record['gate'], ensure_ascii=False)})")
    return record


def run(mode: str, out: Path, producer_root: Path, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    ledger = load_ledger()
    calls = Calls()
    summary: dict = {"mode": mode, "started": now.isoformat(timespec="seconds")}
    youtube_id = instagram_id = None
    slot = None
    if mode == "publish":
        from . import publisher

        youtube_id, instagram_id = publisher.destinations()
        yt_rows = publisher.recent_posts(youtube_id, now)
        ig_rows = publisher.recent_posts(instagram_id, now)
        slot = (slots.replacement_slot(now, yt_rows, ig_rows) if replacing(ledger) else
                slots.free_slot(now, yt_rows, ig_rows, days_ahead=slots.BOOK_AHEAD_DAYS))
        summary["free_slot"] = slot.isoformat() if slot else None
        if slot is None:
            summary["status"] = "no_free_slot"
            return summary
        ready = ledger.get("ready")
        if ready:
            return _schedule(ready, Path(ready["video"]) if Path(ready.get("video", "")).exists() else None,
                             slot, ledger, youtube_id, instagram_id, now, summary, calls)
    ep_id = f"lite-{now.astimezone(slots.IST):%Y%m%d-%H%M}"
    tried: set[str] = set()
    while len(tried) < 2:
        topic = current_topic(ledger, exclude=tried)
        summary["topic"] = topic["key"]
        try:
            record = produce(ep_id, topic, ledger, calls, producer_root, out)
            break
        except WaitForQuota as err:
            summary.update(status="waiting_for_quota", reason=str(err)[:300], calls=calls.to_json())
            save_ledger(ledger)
            return summary
        except Budget as err:
            summary.update(status="budget_stop", reason=str(err), calls=calls.to_json())
            save_ledger(ledger)
            return summary
        except (ValueError, LookupError, sources.SourceError) as err:
            log.warning("topic %s failed: %s", topic["key"], err)
            summary.setdefault("failed_topics", []).append({"topic": topic["key"], "reason": str(err)[:400]})
            if "gate failed" not in str(err):
                strike(ledger, topic["key"], str(err))
                ledger["pending"] = None
            save_ledger(ledger)
            tried.add(topic["key"])
            if calls.gemini + 6 > MAX_GEMINI_CALLS or "gate failed" in str(err):
                summary.update(status="failed", calls=calls.to_json())
                return summary
    else:
        summary.update(status="failed", calls=calls.to_json())
        return summary
    record["calls"] = calls.to_json()
    summary["episode"] = {k: record[k] for k in ("id", "title", "citation", "source_url", "gate", "calls")}
    if mode != "publish":
        ledger["pending"] = None
        summary["status"] = "rendered"
        summary["video"] = record["video"]
        return summary
    # Re-read Buffer: the signed lane may have booked a slot while this Short rendered.
    from . import publisher

    rows = publisher.recent_posts(youtube_id, now), publisher.recent_posts(instagram_id, now)
    slot = (slots.replacement_slot(datetime.now(timezone.utc), *rows) if record.get("replaces") else
            slots.free_slot(datetime.now(timezone.utc), *rows, days_ahead=slots.BOOK_AHEAD_DAYS))
    ledger["pending"] = None
    return _schedule(record, Path(record["video"]), slot, ledger, youtube_id, instagram_id, now, summary, calls)


def strike(ledger: dict, key: str, reason: str) -> None:
    """Count a failed scripting/review run; a source or picture error, or a second strike, sets the topic aside."""
    strikes = ledger.setdefault("strikes", {})
    strikes[key] = strikes.get(key, 0) + 1
    retryable = reason.startswith(("source review:", "script:"))
    if not retryable or strikes[key] >= MAX_TOPIC_STRIKES:
        ledger.setdefault("skipped", {})[key] = reason[:200]


def replacing(ledger: dict) -> bool:
    """True when the pending Short replaces one the owner deleted (set by hand in the ledger)."""
    return bool((ledger.get("pending") or {}).get("replaces"))


def current_topic(ledger: dict, exclude: set[str] = frozenset()) -> dict:
    """The topic a pending (scripted but not yet published) Short belongs to, else the next unused one."""
    pending = ledger.get("pending")
    if not pending or pending["topic_key"] in exclude:
        return next_topic(ledger, exclude=exclude)
    key = pending["topic_key"]
    for topic in json.loads(TOPICS.read_text(encoding="utf-8")):
        if topic["key"] == key:
            return topic
    work, book, chapter = key.rsplit("-", 2)
    return {"key": key, "work": work, "book": int(book), "chapter": int(chapter),
            "angle": "the most surprising lesser-known true detail in this sarga"}


def _schedule(record: dict, video: Path | None, slot, ledger: dict, youtube_id: str, instagram_id: str,
              now: datetime, summary: dict, calls: Calls) -> dict:
    from . import publisher

    if slot is None:
        if video is not None:
            public_id, media_url = publisher.host(video, record["id"])
            record.update(media_public_id=public_id, media_url=media_url)
        ledger["ready"] = record
        save_ledger(ledger)
        summary["status"] = "ready_waiting_for_slot"
        return summary
    if video is None and record.get("media_url"):
        posted = _schedule_hosted(record, slot, youtube_id, instagram_id, now)
    else:
        posted = publisher.schedule(record, video, slot, youtube_id, instagram_id, now)
    posted["readback"] = publisher.readback(posted, youtube_id, instagram_id, now)
    entry = {k: v for k, v in record.items() if k not in ("video", "frames")} | {"publish": posted}
    ledger["episodes"].append(entry)
    ledger["ready"] = None
    save_ledger(ledger)
    ok = all((posted.get(s) or {}).get("id") for s in ("youtube", "instagram"))
    summary.update(status="scheduled" if ok else "partly_scheduled", due_utc=slots.utc(slot),
                   due_ist=slot.astimezone(slots.IST).isoformat(timespec="minutes"), publish=posted)
    return summary


def _schedule_hosted(record: dict, slot, youtube_id: str, instagram_id: str, now: datetime) -> dict:
    from . import publisher

    posted = {"media_public_id": record["media_public_id"], "media_url": record["media_url"], "due_at": slot.isoformat()}
    for service, channel_id, build in (("youtube", youtube_id, publisher.youtube_payload),
                                       ("instagram", instagram_id, publisher.instagram_payload)):
        existing = publisher._twin(publisher.recent_posts(channel_id, now), record["media_url"])
        if existing:
            posted[service] = {"id": existing["id"], "status": existing.get("status"), "adopted": True}
            continue
        post = publisher.create(build(channel_id, record, slot, record["media_url"]))
        posted[service] = {"id": post["id"], "status": post.get("status"), "due_at": post.get("dueAt")}
    return posted
