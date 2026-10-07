import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from PIL import Image

from lite import art, slots, vision

OK = {"shows": "Krishna and Arjuna on a chariot", "nudity": False, "sexual": False, "gore": False, "fits": True,
      "reason": "right scene"}


def picture(folder: Path, n: int, title: str | None = None) -> dict:
    name = f"pic{n:02d}.jpg"
    Image.new("RGB", (1200, 1600), (20 * n % 255, 80, 40)).save(folder / name)
    return {"url": f"https://example.org/{n}.jpg", "key": f"commons:{n}", "path": f"{folder.name}/{name}",
            "credit": {"source": "Wikimedia Commons", "title": title or f"Painting {n}", "url": f"https://c/{n}"}}


def beats(n: int) -> list[dict]:
    return [{"text": f"पंक्ति {i}", "visual_query": f"Arjuna scene {i}"} for i in range(n)]


class Asker:
    """Answers like llm.generate; `bad` maps a picture title to the verdict fields that refuse it."""

    def __init__(self, bad: dict[str, dict] | None = None):
        self.bad = bad or {}
        self.prompts: list[list] = []

    def __call__(self, parts: list, schema: dict) -> dict:
        self.prompts.append(parts)
        titles = [line.split("title: ", 1)[1].split("; narrated", 1)[0].strip("'\"")
                  for line in parts[0].splitlines() if "title: " in line]
        return {"pictures": [{"n": n, **OK, **self.bad.get(t, {})} for n, t in enumerate(titles, 1)]}


class Rejected(unittest.TestCase):
    def test_unsafe_and_unfitting_pictures_are_refused(self):
        self.assertIsNone(vision.rejected(OK))
        self.assertTrue(vision.rejected({**OK, "nudity": True}).startswith("nudity"))
        self.assertTrue(vision.rejected({**OK, "sexual": True}).startswith("sexual"))
        self.assertTrue(vision.rejected({**OK, "gore": True}).startswith("gore"))
        self.assertTrue(vision.rejected({**OK, "fits": False}).startswith("does not fit"))

    def test_a_missing_verdict_is_a_refusal(self):
        tmp = Path(tempfile.mkdtemp())
        p = picture(tmp, 1)
        got = vision.judge([(tmp / "pic01.jpg", p["credit"]["title"], "line")], "Gita", lambda parts, schema: {})
        self.assertIsNotNone(vision.rejected(got[0]))


class Judge(unittest.TestCase):
    def test_pictures_go_five_to_a_prompt_and_small(self):
        tmp = Path(tempfile.mkdtemp())
        items = [(tmp / Path(picture(tmp, n)["path"]).name, f"Painting {n}", "line") for n in range(7)]
        ask = Asker()
        got = vision.judge(items, "Gita", ask)
        self.assertEqual(len(got), 7)
        self.assertEqual([sum(isinstance(p, bytes) for p in parts) for parts in ask.prompts], [5, 2])
        import io
        image = Image.open(io.BytesIO(next(p for p in ask.prompts[0] if isinstance(p, bytes))))
        self.assertLessEqual(max(image.size), vision.SIDE)


class Screen(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "pics"
        self.tmp.mkdir()
        sleep = mock.patch.object(art.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

    def replacement(self, n: int, title: str):
        def save(cand, folder, name):
            Image.new("RGB", (1200, 1600)).save(folder / name)
            return {**cand, "path": f"{folder.name}/{name}"}
        cand = {"url": f"https://example.org/r{n}.jpg", "key": f"commons:r{n}",
                "credit": {"source": "Wikimedia Commons", "title": title}}
        return cand, save

    def test_nude_picture_is_replaced_and_blocked_key_reported(self):
        pics = [picture(self.tmp, n, t) for n, t in enumerate(["Chariot", "Krishna", "Arjuna and river Nymph", "Bow"])]
        cand, save = self.replacement(1, "Arjuna grieving")
        report: list = []
        ask = Asker({"Arjuna and river Nymph": {"nudity": True, "reason": "bare breasts"}})
        with mock.patch.object(art, "pick", side_effect=[cand]), mock.patch.object(art, "_save", side_effect=save):
            got = art.screen([*pics, 1], beats(5), self.tmp, "Gita", ask, report=report)
        self.assertEqual(got[2]["credit"]["title"], "Arjuna grieving")
        self.assertEqual(got[4], 1)
        unsafe = [r for r in report if r["unsafe"]]
        self.assertEqual([r["key"] for r in unsafe], ["commons:2"])
        self.assertEqual(len(ask.prompts), 2)

    def test_wrong_story_with_no_replacement_repeats_the_previous_picture(self):
        pics = [picture(self.tmp, n, t) for n, t in enumerate(["Chariot", "Krishna", "Bow", "Kartavirya Arjuna"])]
        ask = Asker({"Kartavirya Arjuna": {"fits": False, "reason": "a different Arjuna"}})
        with mock.patch.object(art, "pick", return_value=None):
            got = art.screen(pics, beats(4), self.tmp, "Gita", ask)
        self.assertEqual(got[3], 3)

    def test_too_few_safe_pictures_is_an_error(self):
        pics = [picture(self.tmp, n) for n in range(4)]
        ask = Asker({f"Painting {n}": {"nudity": True} for n in range(1, 4)})
        with mock.patch.object(art, "pick", return_value=None), self.assertRaises(LookupError):
            art.screen(pics, beats(4), self.tmp, "Gita", ask)

    def test_no_model_answering_stops_the_short(self):
        pics = [picture(self.tmp, n) for n in range(3)]

        def busy(parts, schema):
            raise RuntimeError("quota")
        with self.assertRaises(RuntimeError):
            art.screen(pics, beats(3), self.tmp, "Gita", busy)

    def test_first_beat_refused_shows_the_first_accepted_picture(self):
        pics = [picture(self.tmp, n) for n in range(4)]
        ask = Asker({"Painting 0": {"gore": True}})
        with mock.patch.object(art, "pick", return_value=None):
            got = art.screen([*pics, 1], beats(5), self.tmp, "Gita", ask)
        self.assertEqual(got[0]["key"], "commons:1")
        self.assertEqual(got[4]["key"], "commons:1")


class TitleFilter(unittest.TestCase):
    def test_risky_titles_are_kept_out(self):
        for title in ("Arjuna and river Nymph", "Apsaras bathing", "Krishna stealing the clothes of the gopis",
                      "Kamadeva and Rati", "Jatayu Hinders Ravana's Chariot (recto/verso)"):
            self.assertTrue(art.ADULT.search(title) or art.REJECT.search(title), title)

    def test_ordinary_titles_pass(self):
        for title in ("Krishna and Arjuna at Kurukshetra", "Bharati painting of Rama", "Arjuna's penance"):
            self.assertFalse(art.ADULT.search(title) or art.REJECT.search(title), title)


class ReplacementSlot(unittest.TestCase):
    def test_books_tonight_an_hour_clear_of_the_deleted_post(self):
        deleted = {"status": "sent", "dueAt": "2026-10-07T16:15:00+00:00"}
        now = datetime(2026, 10, 7, 16, 35, tzinfo=timezone.utc)
        got = slots.replacement_slot(now, [deleted], [deleted])
        self.assertEqual(got.astimezone(slots.IST).strftime("%H:%M"), "22:45")

    def test_none_when_the_evening_is_over(self):
        now = datetime(2026, 10, 7, 17, 45, tzinfo=timezone.utc)
        self.assertIsNone(slots.replacement_slot(now, [], []))


if __name__ == "__main__":
    unittest.main()
