import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from lite import aiart, look, run, sources

from lite.tests.test_sources_writer import VALMIKI_PAGE, good_script

OK = {"shows": "Hanuman flying over the sea", "nudity": False, "sexual": False, "gore": False, "fits": True,
      "malformed": False, "disrespectful": False, "reason": "right scene"}


class LookTest(unittest.TestCase):
    def test_prompt_carries_style_characters_and_no_text(self):
        text = look.prompt("Close-up of Hanuman looking at Rama across the sea at dawn", hook=True)
        self.assertIn(look.CLOSE, text)
        self.assertIn("golden-orange fur", text)
        self.assertIn("blue skin", text)
        self.assertLess(text.index("Hanuman:"), text.index("Lord Rama:"))
        self.assertIn("No text", text)
        self.assertIn("Ravi Varma", text)
        self.assertNotIn(look.CLOSE, look.prompt("Wide view of Lanka at night"))

    def test_sita_with_ravana_is_painted_as_his_captive(self):
        self.assertIn("captive", look.prompt("Ravana flies off in his chariot holding Sita"))
        self.assertNotIn("captive", look.prompt("Sita waits for Rama in the forest"))

    def test_names_inside_other_words_are_not_characters(self):
        self.assertEqual(look.characters("Ramayana manuscript on a table, Brahmin priests"), [])

    def test_plan_has_a_stable_seed_per_beat(self):
        beats = [{"scene": "Hanuman leaps"}, {"scene": ""}, {"scene": "Sita waits in a grove"}]
        planned = look.plan("lite-x", beats)
        self.assertEqual([b["beat"] for b in planned], [0, 1, 2])
        self.assertEqual(planned[1]["prompt"], "")
        self.assertEqual(planned, look.plan("lite-x", beats))
        self.assertNotEqual(planned[0]["seed"], look.plan("lite-y", beats)[0]["seed"])

    def test_chunks_spread_beats_over_jobs_hook_first(self):
        beats = [{"beat": i, "prompt": "p"} for i in range(11)]
        self.assertEqual(run.chunks(beats, jobs=6), ["0,6", "1,7", "2,8", "3,9", "4,10", "5"])
        self.assertEqual(run.chunks(beats[:3]), ["0", "1", "2"])
        self.assertEqual(len(run.chunks(beats)), run.PAINT_JOBS)


def painted(folder: Path, episode: str, beats: range) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in beats:
        Image.new("RGB", (576, 1024), (30 * i % 255, 90, 60)).save(folder / f"beat{i:02d}.png")
        (folder / f"beat{i:02d}.json").write_text(json.dumps({"episode": episode, "beat": i}), encoding="utf-8")


class AiArtTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.beats = [{"text": f"पंक्ति {i}"} for i in range(10)]

    def ask(self, refuse=()):
        def call(parts, schema):
            lines = [line for line in parts[0].splitlines() if "narrated line: " in line]
            return {"pictures": [{"n": n, **OK, **({"malformed": True} if line.endswith(tuple(refuse)) else {})}
                                 for n, line in enumerate(lines, 1)]}
        return call

    def test_collect_ignores_another_episodes_paintings(self):
        painted(self.tmp / "art" / "a", "lite-1", range(0, 5))
        painted(self.tmp / "art" / "b", "lite-old", range(5, 10))
        self.assertEqual(sorted(aiart.collect(self.tmp / "art", "lite-1", 10)), [0, 1, 2, 3, 4])
        self.assertEqual(aiart.collect(None, "lite-1", 10), {})

    def test_refused_and_missing_beats_repeat_the_nearest_earlier_picture(self):
        painted(self.tmp / "art", "lite-1", [0, 1, 2, 3, 5, 6, 7, 8])
        found = aiart.collect(self.tmp / "art", "lite-1", 10)
        result = aiart.pictures(found, self.beats, self.tmp / "pics", "story", self.ask(refuse=("पंक्ति 2",)))
        self.assertEqual([p if isinstance(p, int) else "ai" for p in result],
                         ["ai", "ai", 2, "ai", 4, "ai", "ai", "ai", "ai", 9])
        self.assertTrue(result[0]["ai"])
        self.assertEqual(Image.open(self.tmp / result[0]["path"]).size, aiart.UPSCALE)

    def test_too_few_or_no_hook_falls_back(self):
        painted(self.tmp / "art", "lite-1", range(10))
        found = aiart.collect(self.tmp / "art", "lite-1", 10)
        with self.assertRaisesRegex(LookupError, "AI pictures passed"):
            aiart.pictures(found, self.beats, self.tmp / "p1", "story", self.ask(refuse=("पंक्ति 0",)))
        with self.assertRaises(LookupError):
            aiart.pictures({i: found[i] for i in range(4)}, self.beats, self.tmp / "p2", "story", self.ask())


class AiSpecTest(unittest.TestCase):
    def test_ai_spec_turns_on_depth_and_synthetic_media(self):
        from ytc.spec import ShortSpec

        passage = sources.parse_valmiki(VALMIKI_PAGE, "https://x")
        script = good_script()
        pic = {"path": "pics/ai00.jpg", "url": "", "credit": dict(aiart.CREDIT), "ai": True}
        pictures = [pic] + [dict(pic, path=f"pics/ai{i:02d}.jpg") for i in range(1, len(script["beats"]) - 1)] + [1]
        spec = ShortSpec.model_validate(run.build_spec("lite-test", script, passage, pictures, ai_art=True))
        self.assertTrue(spec.synthetic_media)
        self.assertTrue(spec.depth_motion)
        record = run.episode_record("lite-test", {"key": "k"}, script, passage, pictures, ai_art=True)
        self.assertTrue(record["ai_art"])


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.ledger = {"episodes": [], "skipped": {}, "pending": None, "ready": None}

    def test_plan_scripts_the_topic_and_writes_prompts(self):
        script = good_script()
        with mock.patch.object(run, "load_ledger", return_value=self.ledger), \
                mock.patch.object(run, "prepare", return_value=(script, None, {"ok": True, "issues": []})):
            result = run.plan("dry-run", self.out)
        self.assertTrue(result["paint"])
        self.assertEqual(result["pending"]["script"], script)
        art = json.loads((self.out / "art-plan.json").read_text(encoding="utf-8"))
        self.assertEqual(art["episode"], result["episode"])
        self.assertEqual(len(art["beats"]), len(script["beats"]))
        self.assertEqual(run.load_plan(self.out / "plan.json")["chunks"], result["chunks"])

    def test_a_failed_script_is_recorded_for_the_run_job_and_nothing_is_painted(self):
        with mock.patch.object(run, "load_ledger", return_value=self.ledger), \
                mock.patch.object(run, "prepare", side_effect=ValueError("script: needs 8-12 beats")):
            result = run.plan("dry-run", self.out)
        self.assertFalse(result["paint"])
        self.assertTrue(result["failed"]["reason"].startswith("script:"))

    def test_replay_paints_a_reviewed_script_after_the_writer_checks(self):
        replay = self.out / "replay.json"
        replay.write_text(json.dumps({"topic_key": "ramayana-5-1", "script": good_script(),
                                      "review": {"ok": True, "issues": []}}), encoding="utf-8")
        passage = sources.parse_valmiki(VALMIKI_PAGE, "https://x")
        with mock.patch.object(run, "load_ledger", return_value=self.ledger), \
                mock.patch.object(run.sources, "fetch", return_value=passage), \
                mock.patch.object(run, "prepare", side_effect=AssertionError("no new script")):
            result = run.plan("dry-run", self.out, replay=replay)
            self.assertTrue(result["paint"])
            self.assertEqual(result["pending"]["topic_key"], "ramayana-5-1")
            with self.assertRaises(ValueError):
                run.plan("publish", self.out, replay=replay)

    def test_a_pending_script_from_before_ai_art_is_not_painted(self):
        old = good_script()
        old["beats"] = [{k: v for k, v in b.items() if k != "scene"} for b in old["beats"]]
        self.ledger["pending"] = {"id": "lite-old", "topic_key": "t", "script": old, "review": {}}
        with mock.patch.object(run, "load_ledger", return_value=self.ledger):
            result = run.plan("dry-run", self.out)
        self.assertFalse(result["paint"])
        self.assertFalse((self.out / "art-plan.json").exists())


class RunWithPlanTest(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.ledger = {"episodes": [], "skipped": {}, "pending": None, "ready": None}
        self.pending = {"id": "lite-1", "topic_key": "ramayana-5-1", "script": good_script(),
                        "review": {"ok": True, "issues": []}, "gate_attempts": 0}

    def run_with(self, planned: dict, error: Exception):
        topic = {"key": "ramayana-5-1"}
        with mock.patch.object(run, "load_ledger", return_value=self.ledger), \
                mock.patch.object(run, "load_plan", return_value=planned), \
                mock.patch.object(run, "current_topic", return_value=topic), \
                mock.patch.object(run, "produce", side_effect=error), \
                mock.patch.object(run, "save_ledger") as save:
            summary = run.run("dry-run", self.out, self.out)
        return summary, save

    def test_a_replay_never_reaches_the_saved_ledger(self):
        self.ledger["pending"] = {"id": "lite-0", "topic_key": "ramayana-1-1", "script": {}, "review": {}}
        summary, save = self.run_with({"paint": True, "replay": True, "pending": self.pending},
                                      ValueError("script: too long"))
        save.assert_not_called()
        self.assertEqual(summary["replay"], "ramayana-5-1")
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(len(summary["failed_topics"]), 1)

    def test_a_planned_script_becomes_the_pending_short(self):
        summary, save = self.run_with({"paint": True, "pending": self.pending}, run.WaitForQuota("TTS"))
        self.assertEqual(self.ledger["pending"]["id"], "lite-1")
        save.assert_called_once()
        self.assertEqual(summary["status"], "waiting_for_quota")


class ProduceFallbackTest(unittest.TestCase):
    def test_any_ai_painting_error_falls_back_to_museum_paintings(self):
        tmp = Path(tempfile.mkdtemp())
        script = good_script()
        ledger = {"episodes": [], "pending": {"id": "lite-1", "topic_key": "ramayana-5-1", "script": script,
                                              "review": {"ok": True, "issues": []}}}
        passage = sources.parse_valmiki(VALMIKI_PAGE, "https://x")

        class Museum(Exception):
            pass

        with mock.patch.object(run.sources, "fetch", return_value=passage), \
                mock.patch.object(run.aiart, "collect", return_value={0: tmp / "beat00.png"}), \
                mock.patch.object(run.aiart, "copy_report"), \
                mock.patch.object(run.aiart, "pictures", side_effect=OSError("cannot identify image file")), \
                mock.patch.object(run.art, "choose", return_value=[]), \
                mock.patch.object(run.art, "download", side_effect=Museum):
            with self.assertRaises(Museum):
                run.produce("lite-1", {"key": "ramayana-5-1"}, ledger, run.Calls(), tmp, tmp / "out")
        self.assertIn("OSError", ledger["pending"]["ai_art_note"])


if __name__ == "__main__":
    unittest.main()
