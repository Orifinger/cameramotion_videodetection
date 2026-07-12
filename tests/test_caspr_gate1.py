from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from scripts.caspr_gate1.metrics import aggregate_pairs, binary_auc

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_caspr_gate1_data", ROOT / "tools" / "build_caspr_gate1_data.py")
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class CasprDataBuilderTests(unittest.TestCase):
    def test_scoring_sample_keeps_image_tokens_aligned(self) -> None:
        record = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "a<image>b<image>c<image>d<image>e"},
                {"role": "assistant", "content": "<answer>Fake</answer>"},
            ],
            "images": ["/x/0.png", "/x/1.png", "/x/2.png", "/x/3.png"],
        }
        sample = BUILDER.scoring_sample(record, 2)
        self.assertEqual(sample["images"], ["/x/0.png", "/x/3.png"])
        self.assertEqual(sum(message["content"].count("<image>") for message in sample["messages"]), 2)
        self.assertEqual(sample["label"], "Fake")

    def test_round_robin_selection_covers_strata(self) -> None:
        rows = [
            {"case_id": f"a{i}", "source_family": "a", "motion_bucket": "complex-motion"}
            for i in range(5)
        ] + [
            {"case_id": f"b{i}", "source_family": "b", "motion_bucket": "no-motion"}
            for i in range(5)
        ]
        selected = BUILDER.round_robin_stratified(rows, 4, 7)
        self.assertEqual(len(selected), 4)
        self.assertEqual({row["source_family"] for row in selected}, {"a", "b"})


class CasprMetricTests(unittest.TestCase):
    def test_binary_auc(self) -> None:
        self.assertEqual(binary_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_pair_metrics(self) -> None:
        metrics = aggregate_pairs(
            [
                {"real_score": -1.0, "fake_score": 2.0},
                {"real_score": -0.5, "fake_score": 0.25},
            ]
        )
        self.assertEqual(metrics["auc"], 1.0)
        self.assertEqual(metrics["balanced_accuracy_at_zero"], 1.0)
        self.assertEqual(metrics["pair_accuracy_fake_gt_real"], 1.0)


if __name__ == "__main__":
    unittest.main()
