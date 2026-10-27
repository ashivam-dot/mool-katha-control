import unittest

from lite import look, run, shots, sources, writer
from lite.tests.test_sources_writer import VALMIKI_PAGE, good_script

CLOSE = "Close-up of Hanuman's determined face against the bright sky, eyes fixed on the far horizon, wind in his fur."


def with_close(script: dict) -> dict:
    script["beats"] = [dict(b, close=CLOSE) if i else dict(b) for i, b in enumerate(script["beats"])]
    return script


class SplitTest(unittest.TestCase):
    def test_short_beat_keeps_one_painting(self):
        self.assertIsNone(shots.split_text("तो फिर उन्हें बीच समुद्र में किसने रोका?"))

    def test_eleven_words_still_keep_one_painting(self):
        self.assertIsNone(shots.split_text("एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह"))

    def test_cut_falls_on_a_spoken_seam_near_the_middle(self):
        first, second = shots.split_text("हनुमान सीता को खोजने के लिए आकाश मार्ग से चलना चाहते थे और उनका मन बहुत दृढ़ था।")
        self.assertTrue(second.startswith("और "), second)
        self.assertEqual(f"{first} {second}".split(), "हनुमान सीता को खोजने के लिए आकाश मार्ग से चलना चाहते थे और उनका मन बहुत दृढ़ था।".split())

    def test_comma_is_a_seam(self):
        first, _ = shots.split_text("जब लंका का राजा रावण सीता को देखा, उसका मन डोल गया और वह आगे बढ़ा")
        self.assertTrue(first.endswith(","), first)

    def test_without_a_seam_the_cut_is_halfway(self):
        first, second = shots.split_text("एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह")
        self.assertEqual((len(first.split()), len(second.split())), (6, 6))


class PlanTest(unittest.TestCase):
    def test_long_beats_get_a_close_up_and_the_hook_stays_whole(self):
        script = with_close(writer.tidy(good_script()))
        planned = shots.plan(script["beats"])
        self.assertEqual(len(planned), 2 * len(script["beats"]) - 2)
        self.assertEqual([s["beat"] for s in planned[:3]], [0, 1, 1])
        self.assertFalse(planned[0]["close"])
        self.assertEqual((planned[1]["pause_after"], planned[2]["pause_after"]), (0.0, 0.12))
        self.assertEqual(planned[2]["scene"], CLOSE)
        spoken = " ".join(s["text"] for s in planned).split()
        self.assertEqual(spoken, " ".join(b["text"] for b in script["beats"]).split())

    def test_emphasis_stays_with_its_half(self):
        planned = shots.plan(with_close(writer.tidy(good_script()))["beats"])
        for shot in planned:
            if shot["emphasis"]:
                self.assertIn(shot["emphasis"], shot["text"].split())
        self.assertEqual(sum(bool(s["emphasis"]) for s in planned if s["beat"] == 1), 1)

    def test_shot_count_is_capped_longest_beats_first(self):
        beats = with_close(writer.tidy(good_script()))["beats"]
        beats[5] = dict(beats[5], text=beats[5]["text"] + " वे आगे बढ़ते रहे")
        planned = shots.plan(beats, limit=11)
        self.assertEqual(len(planned), 11)
        self.assertEqual([s["beat"] for s in planned].count(5), 2)

    def test_old_scripts_without_close_scenes_paint_one_picture_a_beat(self):
        script = writer.tidy(good_script())
        self.assertEqual(len(shots.plan(script["beats"])), len(script["beats"]))

    def test_paint_plan_frames_close_shots_as_close_ups(self):
        planned = shots.plan(with_close(writer.tidy(good_script()))["beats"])
        prompts = look.plan("lite-test", planned)
        self.assertEqual([p["beat"] for p in prompts], list(range(len(planned))))
        self.assertIn(look.CLOSE, prompts[0]["prompt"])
        self.assertNotIn(look.CLOSE, prompts[1]["prompt"])
        self.assertNotIn(look.DETAIL, prompts[1]["prompt"])
        self.assertIn(look.DETAIL, prompts[2]["prompt"])
        self.assertNotIn(look.CLOSE, prompts[2]["prompt"])
        self.assertEqual(len(run.chunks(prompts)), run.PAINT_JOBS)


class WriterCloseTest(unittest.TestCase):
    def setUp(self):
        self.passage = sources.parse_valmiki(VALMIKI_PAGE, "https://x")

    def test_close_scene_is_kept_and_checked(self):
        script = with_close(writer.tidy(good_script()))
        self.assertEqual(writer.tidy(script)["beats"][1]["close"], CLOSE)
        self.assertEqual(writer.problems(script, self.passage), [])
        script["beats"][2]["close"] = "हनुमान का चेहरा"
        self.assertIn("beat 3 close needs an English close-up", " ".join(writer.problems(script, self.passage)))
        script["beats"][2]["close"] = CLOSE.replace("determined face", "wounded arm being severed")
        self.assertIn("beat 3 close shows violence", " ".join(writer.problems(script, self.passage)))


class SpecShotsTest(unittest.TestCase):
    def test_every_shot_is_a_rendered_beat_with_its_own_painting(self):
        from ytc.spec import ShortSpec

        passage = sources.parse_valmiki(VALMIKI_PAGE, "https://x")
        script = with_close(writer.tidy(good_script()))
        planned = shots.plan(script["beats"])
        pic = {"path": "pics/ai00.jpg", "url": "", "credit": {"source": "AI", "title": "AI painting", "credit": "Mool Katha",
               "license": "AI-generated", "url": "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo"}}
        pictures = [dict(pic, path=f"pics/ai{i:02d}.jpg") for i in range(len(planned) - 1)] + [3]
        spec = ShortSpec.model_validate(run.build_spec("lite-test", script, passage, pictures, ai_art=True,
                                                       shot_list=planned))
        self.assertEqual(len(spec.beats), len(planned))
        self.assertEqual([b.pause_after for b in spec.beats[1:3]], [0.0, 0.12])
        self.assertEqual(spec.beats[-1].visual.reuse, 3)
        self.assertTrue(spec.synthetic_media)


if __name__ == "__main__":
    unittest.main()
