import json
import unittest

from lite import review, run, sources, writer

VALMIKI_PAGE = """<html><head><title>Valmiki Ramayana - Sundara Kanda - Sarga 1 &#65279;</title></head><body>
<p class="verloc"><a name="Verse1"></a>Verse Locator</p>
<p class="SanSloka">ततो रावणनीतायाः सीतायाः शत्रुकर्शनः |<br>इयेष पदमन्वेष्टुं चारणाचरिते पथि ||५-१-१</p>
<p class="pratipada"><em>1. tataH</em> = after that;</p>
<p class="tat">After that, Hanuma, the destroyer of foes, desired to travel in the sky to search for Seetha.</p>
<p class="verloc">Verse Locator</p>
<p class="SanSloka">दुष्करं निष्प्रतिद्वन्द्वं चिकीर्षन् कर्म वानरः |<br>समुदग्रशिरोग्रीवो गवांपतिरिवाबभौ || ५-१-२</p>
<p class="tat">Hanuma who desired to perform a deed done by no one else, shone like a bull.</p>
<p class="verloc">Verse Locator</p>
<p class="SanSloka">अथ वैडूर्यवर्णेषु शाद्वलेषु महाबलः |<br>धीरः सलिलकल्पेषु विच्चार यथासुखम् ||५-१-३</p>
<p class="tat">After that, the mighty and courageous Hanuma roamed at ease on the lawns with the hue of an emerald.</p>
</body></html>"""

SACRED_PAGE = """<html><head><title>The Mahabharata, Book 17: Mahaprasthanika Parva: Section... | Internet Sacred Text Archive</title></head>
<body><h1 align="center">SECTION III</h1><p>Vaishampayana said: "Then Shakra came to the son of Pritha on a car and asked him to ascend it.<A NAME="fr_1"></A><A HREF="#fn_1"><FONT SIZE="1">1</FONT></A></p>
<p>Yudhishthira said, 'This dog, O lord of the past and the future, is exceedingly devoted to me. Let him go with me.'</p>
<h3 align="center">Footnotes</h3><p>1. A footnote that must not be quoted as the text itself, long enough.</p></body></html>"""


class SourcesTest(unittest.TestCase):
    def test_valmiki_numbers_come_from_the_page(self):
        p = sources.parse_valmiki(VALMIKI_PAGE, "https://x/sundarasans1.htm")
        self.assertEqual((p.book, p.chapter, len(p.verses)), (5, 1, 3))
        self.assertEqual([v.number for v in p.verses], ["5-1-1", "5-1-2", "5-1-3"])
        self.assertEqual(p.citation_hi, "वाल्मीकि रामायण · सुन्दरकाण्ड · सर्ग 1")
        self.assertIn("[5-1-2]", p.writer_text())
        self.assertNotIn("tataH", p.writer_text())

    def test_sacred_texts_chapter_from_heading_when_title_is_cut(self):
        p = sources.parse_sacred(SACRED_PAGE, "https://x/m17003.htm", "mahabharata")
        self.assertEqual((p.book, p.chapter), (17, 3))
        self.assertEqual(p.citation_hi, "महाभारत · महाप्रस्थानिक पर्व · खंड 3 (गांगुली अनुवाद)")
        self.assertNotIn("footnote", p.writer_text().lower())

    def test_page_without_numbers_is_refused(self):
        with self.assertRaises(sources.SourceError):
            sources.parse_sacred("<title>Something</title><p>" + "text " * 30 + "</p>", "https://x", "mahabharata")

    def test_quote_on_page(self):
        p = sources.parse_valmiki(VALMIKI_PAGE, "https://x")
        self.assertTrue(sources.quote_on_page("Hanuma who desired to perform a deed done by no one else", p))
        self.assertTrue(sources.quote_on_page("hanuma, who desired to perform a deed done by no-one else,", p))
        self.assertFalse(sources.quote_on_page("Hanuma fought a giant serpent in the middle of the sea", p))

    def test_roman(self):
        self.assertEqual([sources.roman(x) for x in ("III", "XIX", "CXLVI", "12")], [3, 19, 146, 12])

    def test_topics_catalog_is_well_formed(self):
        topics = json.loads(run.TOPICS.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(topics), 30)
        self.assertEqual(len({t["key"] for t in topics}), len(topics))
        for t in topics:
            self.assertIn(t["work"], sources.WORK_HI)
            self.assertTrue(t["angle"])


def good_script():
    return {
        "usable": True, "title": "हनुमान को समुद्र के ऊपर किसने रोका? | वाल्मीकि रामायण",
        "hook_text": "हनुमान को किसने रोका?",
        "beats": [{"text": "समुद्र लांघते हनुमान को बीच रास्ते में किसने रोका?", "evidence": "", "visual_query": "Hanuman ocean", "emphasis": ""}]
        + [{"text": "हनुमान सीता को खोजने के लिए आकाश मार्ग से चलना चाहते थे और उनका मन दृढ़ था।",
            "evidence": "desired to travel in the sky to search for Seetha", "visual_query": "Hanuman flying", "emphasis": "सीता"}] * 6
        + [{"text": "तो फिर उन्हें बीच समुद्र में किसने रोका?", "evidence": "", "visual_query": "Hanuman", "emphasis": ""}],
        "description": "सुन्दरकाण्ड के पहले सर्ग का प्रसंग।", "hashtags": ["#ramayan", "#hanuman", "#sundarkand"],
        "keywords": ["हनुमान"], "shloka_verse": "5-1-1", "key_quote": "desired to travel in the sky to search for Seetha",
        "names": ["हनुमान"],
    }


class WriterRulesTest(unittest.TestCase):
    def setUp(self):
        self.passage = sources.parse_valmiki(VALMIKI_PAGE, "https://x")

    def test_good_script_passes(self):
        script = writer.tidy(good_script())
        self.assertTrue(writer.WORDS_MIN <= writer.word_count(script) <= writer.WORDS_MAX, writer.word_count(script))
        self.assertEqual(writer.problems(script, self.passage), [])

    def test_banned_hook_and_invented_quote_and_bad_title(self):
        script = good_script()
        script["beats"] = [dict(b) for b in script["beats"]]
        script["beats"][0]["text"] = "क्या आप जानते हैं हनुमान को किसने रोका?"
        script["beats"][2]["evidence"] = "Hanuma fought a giant serpent in the middle of the sea for days"
        script["title"] = "हनुमान और सुरसा की कथा"
        script["shloka_verse"] = "5-1-99"
        found = " | ".join(writer.problems(writer.tidy(script), self.passage))
        self.assertIn("banned opener", found)
        self.assertIn("beat 3: its evidence is not an exact quote", found)
        self.assertIn("title must be a question", found)
        self.assertIn("title must name the text", found)
        self.assertIn("shloka_verse", found)

    def test_unsupported_middle_beat_needs_evidence(self):
        script = good_script()
        script["beats"] = [dict(b) for b in script["beats"]]
        script["beats"][3]["evidence"] = ""
        self.assertIn("beat 4 states something without an exact evidence quote",
                      writer.problems(writer.tidy(script), self.passage))

    def test_write_retries_once_with_the_problems(self):
        prompts = []
        bad = good_script() | {"title": "कोई प्रश्न नहीं"}

        def fake(prompt, schema):
            prompts.append(prompt)
            return bad if len(prompts) == 1 else good_script()

        script, problems = writer.write(self.passage, "angle", fake)
        self.assertEqual(problems, [])
        self.assertEqual(len(prompts), 2)
        self.assertIn("title must be a question", prompts[1])


class ReviewTest(unittest.TestCase):
    def test_verdict(self):
        script = writer.tidy(good_script())
        items = [{"beat": n, "factual": n not in (1, 8), "supported": True, "note": ""} for n in range(1, 9)]
        ok, issues = review.verdict({"items": items, "respectful": True, "title_reveals_answer": False}, script)
        self.assertTrue(ok, issues)
        items[2]["supported"] = False
        items[2]["note"] = "adds a serpent"
        ok, issues = review.verdict({"items": items, "respectful": True, "title_reveals_answer": False}, script)
        self.assertFalse(ok)
        self.assertIn("beat 3 is not supported by the passage: adds a serpent", issues)
        ok, issues = review.verdict({"items": items[:5], "respectful": True, "title_reveals_answer": True}, script)
        self.assertIn("beat 8 was not judged", issues)
        self.assertIn("the title gives away the answer", issues)


class SpecTest(unittest.TestCase):
    def test_spec_validates_with_the_producer_schema(self):
        from ytc.spec import ShortSpec

        passage = sources.parse_valmiki(VALMIKI_PAGE, "https://x")
        script = writer.tidy(good_script())
        pic = {"url": "https://upload.wikimedia.org/a.jpg", "credit": {"source": "Wikimedia Commons", "title": "A",
               "credit": "Unknown", "license": "Public domain", "url": "https://commons.wikimedia.org/wiki/File:A.jpg"}}
        pictures = [pic, dict(pic, url="https://upload.wikimedia.org/b.jpg"), 2, dict(pic, url="https://upload.wikimedia.org/c.jpg"), 4, 4, 4, 1]
        spec = ShortSpec.model_validate(run.build_spec("lite-test", script, passage, pictures))
        self.assertEqual(spec.voice.voice, "Sulafat")
        self.assertEqual(spec.citation, "वाल्मीकि रामायण · सुन्दरकाण्ड · सर्ग 1")
        self.assertEqual(spec.beats[-1].visual.reuse, 1)
        self.assertEqual(spec.captions.font_file, "Mukta-ExtraBold.ttf")


class TopicTest(unittest.TestCase):
    def test_next_topic_skips_used_and_falls_back_to_fresh_sargas(self):
        topics = json.loads(run.TOPICS.read_text(encoding="utf-8"))
        ledger = {"episodes": [{"topic_key": topics[0]["key"]}], "skipped": {topics[1]["key"]: "x"},
                  "pending": None, "ready": None}
        self.assertEqual(run.next_topic(ledger)["key"], topics[2]["key"])
        ledger["skipped"] = {t["key"]: "x" for t in topics}
        fresh = run.next_topic(ledger)
        self.assertTrue(fresh["key"].startswith("ramayana-"))
        self.assertNotIn(fresh["key"], ledger["skipped"])


if __name__ == "__main__":
    unittest.main()
