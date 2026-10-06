"""Fetch one chapter of a source text and keep only what a Short may cite from it.

Every number shown on screen (kanda/parva/book, sarga/section/chapter, verse) is parsed from the fetched page
itself, never from a URL or a catalog guess.
"""

from __future__ import annotations

import html
import re
import time
from dataclasses import asdict, dataclass, field

import requests

USER_AGENT = "MoolKathaLite/1.0 (+https://www.youtube.com/@moolkatha; educational retelling)"
VALMIKI = "https://www.valmikiramayan.net/utf8"
SACRED = "https://archive.sacred-texts.com/hin"

RAMAYANA_KANDA = {1: "बालकाण्ड", 2: "अयोध्याकाण्ड", 3: "अरण्यकाण्ड", 4: "किष्किन्धाकाण्ड", 5: "सुन्दरकाण्ड",
                  6: "युद्धकाण्ड", 7: "उत्तरकाण्ड"}
# valmikiramayan.net folder and file prefix for each kanda.
VALMIKI_PATHS = {1: ("baala", "bala"), 2: ("ayodhya", "ayodhya"), 3: ("aranya", "aranya"),
                 4: ("kish", "kishkindha"), 5: ("sundara", "sundara"), 6: ("yuddha", "yuddha")}
_KANDA_NAMES = {"bala": 1, "baala": 1, "ayodhya": 2, "aranya": 3, "kishkindha": 4, "sundara": 5, "yuddha": 6}
PARVA = {1: "आदि पर्व", 2: "सभा पर्व", 3: "वन पर्व", 4: "विराट पर्व", 5: "उद्योग पर्व", 6: "भीष्म पर्व",
         7: "द्रोण पर्व", 8: "कर्ण पर्व", 9: "शल्य पर्व", 10: "सौप्तिक पर्व", 11: "स्त्री पर्व", 12: "शान्ति पर्व",
         13: "अनुशासन पर्व", 14: "आश्वमेधिक पर्व", 15: "आश्रमवासिक पर्व", 16: "मौसल पर्व",
         17: "महाप्रस्थानिक पर्व", 18: "स्वर्गारोहण पर्व"}
WORK_HI = {"ramayana": "वाल्मीकि रामायण", "mahabharata": "महाभारत", "gita": "भगवद्गीता",
           "vishnu_purana": "विष्णु पुराण"}
TRANSLATOR = {"ramayana": "K. M. K. Murthy, valmikiramayan.net (Sanskrit with English)",
              "mahabharata": "Kisari Mohan Ganguli (1883-1896), sacred-texts.com",
              "gita": "Kisari Mohan Ganguli (1883-1896), sacred-texts.com",
              "vishnu_purana": "H. H. Wilson (1840), sacred-texts.com"}
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


class SourceError(RuntimeError):
    pass


@dataclass
class Verse:
    number: str  # as printed on the page, e.g. "5-1-1"; "" for prose paragraphs
    sanskrit: str
    english: str


@dataclass
class Passage:
    work: str
    url: str
    page_title: str
    book: int
    chapter: int
    translator: str
    verses: list[Verse] = field(default_factory=list)

    @property
    def citation_hi(self) -> str:
        if self.work == "ramayana":
            return f"{WORK_HI['ramayana']} · {RAMAYANA_KANDA[self.book]} · सर्ग {self.chapter}"
        if self.work == "mahabharata":
            return f"{WORK_HI['mahabharata']} · {PARVA[self.book]} · खंड {self.chapter} (गांगुली अनुवाद)"
        if self.work == "gita":
            return f"{WORK_HI['gita']} · {PARVA[self.book]} खंड {self.chapter} (गांगुली अनुवाद)"
        if self.work == "vishnu_purana":
            return f"{WORK_HI['vishnu_purana']} · अंश {self.book} · अध्याय {self.chapter} (विल्सन अनुवाद)"
        raise SourceError(f"unknown work {self.work}")

    @property
    def text_name_hi(self) -> str:
        return WORK_HI[self.work]

    def writer_text(self, limit: int = 60000) -> str:
        """The passage as the writer and reviewer see it: printed verse numbers, Sanskrit, then English."""
        parts = []
        for verse in self.verses:
            head = f"[{verse.number}] " if verse.number else ""
            body = "\n".join(x for x in (verse.sanskrit, verse.english) if x)
            parts.append(head + body)
        text = "\n\n".join(parts)
        return text[:limit]

    def to_json(self) -> dict:
        return asdict(self) | {"citation_hi": self.citation_hi}


def roman(value: str) -> int:
    value = value.upper().strip(". ")
    if value.isdigit():
        return int(value)
    if not value or any(c not in _ROMAN for c in value):
        raise SourceError(f"not a roman numeral: {value!r}")
    total = 0
    for i, c in enumerate(value):
        n = _ROMAN[c]
        total += -n if i + 1 < len(value) and _ROMAN[value[i + 1]] > n else n
    return total


def _get(url: str) -> str:
    for attempt in range(4):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=45)
        except requests.RequestException as err:
            if attempt == 3:
                raise SourceError(f"could not fetch {url}: {err}") from err
        else:
            if resp.status_code == 200:
                resp.encoding = resp.encoding if resp.encoding and resp.encoding.lower() != "iso-8859-1" else "utf-8"
                return resp.text
            if resp.status_code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise SourceError(f"{url} returned HTTP {resp.status_code}")
        time.sleep(3 * (attempt + 1))
    raise SourceError(f"could not fetch {url}")


def _plain(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", fragment))
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def clean(text: str) -> str:
    return " ".join(html.unescape(text).split())


# --- valmikiramayan.net ---------------------------------------------------------------------------------------

def valmiki_url(kanda: int, sarga: int) -> str:
    folder, prefix = VALMIKI_PATHS[kanda]
    return f"{VALMIKI}/{folder}/sarga{sarga}/{prefix}_{sarga}_frame.htm"


def parse_valmiki(page: str, url: str) -> Passage:
    title = clean((re.search(r"<title>(.*?)</title>", page, re.S | re.I) or [None, ""])[1])
    match = re.search(r"Valmiki Ramayana\s*-\s*(\w+)\s+Kanda\s*-\s*Sarga\s+(\d+)", title, re.I)
    if not match or match.group(1).lower() not in _KANDA_NAMES:
        raise SourceError(f"{url}: page title names no kanda and sarga ({title!r})")
    kanda, sarga = _KANDA_NAMES[match.group(1).lower()], int(match.group(2))
    verses = []
    for block in re.split(r'<p class="verloc">', page)[1:]:
        sanskrit = " ".join(_plain(m) for m in re.findall(r'<p class="SanSloka">(.*?)</p>', block, re.S | re.I))
        sanskrit = re.sub(r"\s+", " ", sanskrit).strip()
        english = " ".join(_plain(m) for m in re.findall(r'<p class="tat">(.*?)</p>', block, re.S | re.I))
        english = " ".join(english.split())
        if not sanskrit and not english:
            continue
        number = ""
        if found := re.search(r"([०-९\d]+)\s*-\s*([०-९\d]+)\s*-\s*([०-९\d]+)", sanskrit):
            number = "-".join(found.group(i).translate(_DEVANAGARI_DIGITS) for i in (1, 2, 3))
        verses.append(Verse(number, sanskrit, english))
    if len(verses) < 3:
        raise SourceError(f"{url}: fewer than three verses parsed")
    return Passage("ramayana", url, title, kanda, sarga, TRANSLATOR["ramayana"], verses)


def fetch_valmiki(kanda: int, sarga: int) -> Passage:
    frame_url = valmiki_url(kanda, sarga)
    frame = _get(frame_url)
    main = re.search(r'<frame[^>]+name="main"[^>]+src="([^"]+)"', frame, re.I)
    page_url = frame_url.rsplit("/", 1)[0] + "/" + main.group(1) if main else frame_url
    return parse_valmiki(_get(page_url), page_url)


# --- sacred-texts.com (Ganguli Mahabharata, Wilson Vishnu Purana) ----------------------------------------------

def parse_sacred(page: str, url: str, work: str) -> Passage:
    title = clean((re.search(r"<title>(.*?)</title>", page, re.S | re.I) or [None, ""])[1])
    title = title.split("|")[0].strip()
    book = re.search(r"Book\s+([IVXLC]+|\d+)\b", title)
    body = re.sub(r"<script.*?</script>|<style.*?</style>", "", page, flags=re.S | re.I)
    # Long titles are cut with "..." before the chapter, so the printed heading is the authority.
    chapter = None
    for heading in re.findall(r"<h[1-4][^>]*>(.*?)</h[1-4]>", body, re.S | re.I):
        if found := re.match(r"\s*(?:SECTION|CHAPTER|CHAP\.?|CANTO)\s+([IVXLC]+|\d+)\b", _plain(heading), re.I):
            chapter = found
            break
    chapter = chapter or re.search(r"(?:Section|Chapter|Canto)\s+([IVXLC]+|\d+)\b", title)
    if not book or not chapter:
        raise SourceError(f"{url}: page names no book and chapter ({title!r})")
    body = re.split(r"<h3[^>]*>\s*Footnotes\s*</h3>", body, flags=re.I)[0]
    body = re.sub(r"<a[^>]+href=\"#fn_\d+\"[^>]*>.*?</a>", "", body, flags=re.S | re.I)
    paragraphs = [" ".join(_plain(p).split()) for p in re.findall(r"<p[^>]*>(.*?)</p>", body, re.S | re.I)]
    paragraphs = [p for p in paragraphs if len(p) > 60 and "Next:" not in p and "Sacred Texts" not in p]
    if len(paragraphs) < 2:
        raise SourceError(f"{url}: no chapter text parsed")
    return Passage(work, url, title, roman(book.group(1)), roman(chapter.group(1)), TRANSLATOR[work],
                   [Verse("", "", p) for p in paragraphs])


def sacred_url(work: str, book: int, page: int) -> str:
    if work in ("mahabharata", "gita"):
        return f"{SACRED}/m{book:02d}/m{book:02d}{page:03d}.htm"
    if work == "vishnu_purana":
        return f"{SACRED}/vp/vp{page:03d}.htm"
    raise SourceError(f"no sacred-texts path for {work}")


def fetch(ref: dict) -> Passage:
    """ref: {"work", "book", "chapter"} for the Ramayana, or {"work", "book", "page"} for sacred-texts pages."""
    if ref["work"] == "ramayana":
        passage = fetch_valmiki(int(ref["book"]), int(ref["chapter"]))
        if (passage.book, passage.chapter) != (int(ref["book"]), int(ref["chapter"])):
            raise SourceError(f"{passage.url} is kanda {passage.book} sarga {passage.chapter}, not the requested one")
        return passage
    url = sacred_url(ref["work"], int(ref["book"]), int(ref["page"]))
    return parse_sacred(_get_sacred(url), url, ref["work"])


def sacred_mirrors(url: str) -> list[str]:
    """The same sacred-texts page from its other host, then the Wayback Machine's unmodified copy: sacred-texts
    refuses some cloud runners' addresses with HTTP 403."""
    path = url.removeprefix(SACRED.rsplit("/", 1)[0])
    return [url, f"https://sacred-texts.com{path}", f"https://web.archive.org/web/2025id_/https://sacred-texts.com{path}"]


def _get_sacred(url: str) -> str:
    errors = []
    for mirror in sacred_mirrors(url):
        try:
            return _get(mirror)
        except SourceError as err:
            errors.append(str(err))
    raise SourceError("; ".join(errors))


def normalize(text: str) -> str:
    """Lowercased letters and digits only, for a forgiving 'is this quote on the page' check."""
    text = html.unescape(text).lower().translate(_DEVANAGARI_DIGITS)
    return re.sub(r"[^\w]+", " ", text).strip()


def quote_on_page(quote: str, passage: Passage) -> bool:
    q = normalize(quote)
    if len(q) < 12:
        return False
    page = normalize(passage.writer_text(limit=10**7))
    if q in page:
        return True
    # Allow a dropped word or two: most of the quote's 6-word windows must appear verbatim.
    words = q.split()
    if len(words) < 8:
        return False
    windows = [" ".join(words[i:i + 6]) for i in range(len(words) - 5)]
    return sum(w in page for w in windows) / len(windows) >= 0.7
