"""Coverage and capacity contracts for experimental sampled frame review."""

from __future__ import annotations

import unittest

from trusted_qa.common import QaHold
from trusted_qa.sample_plan import (MAX_MODEL_FRAMES, MAX_SAMPLE_GAP_SECONDS,
                                    canonical_sha256, plan_sampled_frames)


class SamplePlanTests(unittest.TestCase):
    def test_exact_episode_shape_covers_every_beat_cut_and_time_gap(self) -> None:
        cuts = {191, 344, 469, 533, 713, 891, 1013, 1136}
        frames = [{"index": index, "seconds": round((index - 1) / 30, 6),
                   "rgb_delta_previous": 50.0 if index in cuts else 0.5,
                   "mean_luma": 100.0}
                  for index in range(1, 1312)]
        spans = [(0.35, 6.34), (6.34, 11.42), (11.42, 17.74), (17.74, 23.74),
                 (23.74, 29.68), (29.68, 33.74), (33.74, 37.82), (37.82, 43.35)]
        plan = plan_sampled_frames(frames, [{"start": start, "end": end}
                                            for start, end in spans], 43.7)
        selected = set(plan["sampled_indices"])
        self.assertEqual(plan["kind"], "deterministic_sample_plan_v2")
        self.assertEqual(plan["selection_rule_sha256"], canonical_sha256(plan["selection_rule"]))
        self.assertEqual(plan["sampled_frame_count"], len(selected))
        self.assertEqual(plan["unsampled_frame_count"], 1311 - len(selected))
        self.assertLessEqual(len(selected), MAX_MODEL_FRAMES)
        self.assertEqual((plan["sampled_indices"][0], plan["sampled_indices"][-1]),
                         (1, 1311))
        self.assertLessEqual(plan["max_gap_seconds"], MAX_SAMPLE_GAP_SECONDS)
        self.assertEqual({item["index"] for item in plan["anomalies"]}, cuts)
        for cut in cuts:
            self.assertTrue({cut - 1, cut, cut + 1}.issubset(selected))
        self.assertEqual(len(plan["beat_midpoints"]), len(spans))
        self.assertTrue(all(row["index"] in selected for row in plan["beat_midpoints"]))
        self.assertTrue(all({row["before_index"], row["after_index"], row["next_index"]}.issubset(selected)
                            for row in plan["beat_boundary_pairs"]))

    def test_color_jump_and_extreme_luma_require_model_samples(self) -> None:
        frames = [{"index": index, "seconds": (index - 1) * 0.5,
                   "rgb_delta_previous": 6.0 if index == 4 else 0.2,
                   "mean_luma": 250.0 if index == 8 else 100.0}
                  for index in range(1, 11)]
        plan = plan_sampled_frames(frames, [{"start": 0.0, "end": 4.5}], 4.5)
        self.assertEqual({row["type"] for row in plan["anomalies"]},
                         {"rgb_jump", "extreme_mean_luma"})
        self.assertTrue({3, 4, 8}.issubset(plan["sampled_indices"]))

    def test_required_anomalies_over_capacity_hold_instead_of_sampling_away(self) -> None:
        frames = [{"index": index, "seconds": (index - 1) * 0.1,
                   "rgb_delta_previous": 6.0 if index % 2 == 0 else 0.2,
                   "mean_luma": 100.0}
                  for index in range(1, 151)]
        with self.assertRaisesRegex(QaHold, "exceed two-sheet model capacity"):
            plan_sampled_frames(frames, [{"start": 0.0, "end": 14.9}], 14.9)

    def test_sparse_decode_and_bad_timing_hold(self) -> None:
        sparse = [{"index": 1, "seconds": 0.0, "rgb_delta_previous": 0.0,
                   "mean_luma": 100.0},
                  {"index": 2, "seconds": 4.0, "rgb_delta_previous": 0.5,
                   "mean_luma": 100.0}]
        with self.assertRaisesRegex(QaHold, "no decoded frame"):
            plan_sampled_frames(sparse, [{"start": 0.0, "end": 4.0}], 4.0)
        sparse[1]["seconds"] = 0.0
        with self.assertRaisesRegex(QaHold, "timing or pixels"):
            plan_sampled_frames(sparse, [{"start": 0.0, "end": 4.0}], 4.0)


if __name__ == "__main__":
    unittest.main()
