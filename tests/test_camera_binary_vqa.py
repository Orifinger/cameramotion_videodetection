from __future__ import annotations

import unittest

from scripts.camera_binary_vqa.build_data import (
    balanced_pairs,
    no_video_controls,
    opposite_video_controls,
)
from scripts.camera_binary_vqa.evaluate import binary_metrics, evaluate_condition


class CameraBinaryVqaDataTests(unittest.TestCase):
    def test_balanced_pairs_and_controls(self) -> None:
        rows = [
            {
                "case_id": f"case_{index}",
                "real_video": f"/videos/{index}.mp4",
                "camera_labels": ["no-motion"] if index < 3 else [],
                "source_family": "test",
                "motion_bucket": "no-motion" if index < 3 else "complex-motion",
            }
            for index in range(8)
        ]
        records, stats = balanced_pairs(
            rows, split="dev", seed=7, max_per_class=0, minimum_per_class=1
        )
        selected = [row for row in records if row["camera_primitive"] == "no-motion"]
        self.assertEqual(len(selected), 6)
        self.assertEqual({row["answer"] for row in selected}, {"Yes", "No"})
        self.assertEqual(stats["no-motion"]["selected_per_answer"], 3)

        opposite = opposite_video_controls(selected)
        original_by_id = {row["sample_id"]: row for row in selected}
        for row in opposite:
            original = original_by_id[row["sample_id"]]
            self.assertNotEqual(row["videos"], original["videos"])
            self.assertEqual(row["answer"], original["answer"])

        no_video = no_video_controls(selected)
        self.assertTrue(all(not row["videos"] for row in no_video))
        self.assertTrue(all("<video>" not in row["messages"][-1]["content"] for row in no_video))


class CameraBinaryVqaMetricTests(unittest.TestCase):
    def test_perfect_binary_metrics(self) -> None:
        metrics = binary_metrics(
            [
                {"answer_id": 1, "yes_minus_no_score": 2.0},
                {"answer_id": 0, "yes_minus_no_score": -2.0},
            ]
        )
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["average_precision"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_paired_question_accuracy_requires_both_answers(self) -> None:
        gold = [
            {
                "sample_id": "yes",
                "pair_id": "p0",
                "answer_id": 1,
                "camera_primitive": "pan-left",
            },
            {
                "sample_id": "no",
                "pair_id": "p0",
                "answer_id": 0,
                "camera_primitive": "pan-left",
            },
        ]
        one_wrong = [
            {
                "sample_id": "yes",
                "answer_id": 1,
                "camera_primitive": "pan-left",
                "yes_minus_no_score": 1.0,
            },
            {
                "sample_id": "no",
                "answer_id": 0,
                "camera_primitive": "pan-left",
                "yes_minus_no_score": 1.0,
            },
        ]
        result = evaluate_condition(gold, one_wrong)
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["paired_question_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
