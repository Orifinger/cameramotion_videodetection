from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load_tool("build_datab_deepfakejudge_gate", "tools/build_datab_deepfakejudge_gate.py")
EVALUATOR = load_tool("eval_datab_deepfakejudge_gate", "tools/eval_datab_deepfakejudge_gate.py")


class DataBDeepfakeJudgeBuilderTests(unittest.TestCase):
    def test_ground_truth_comes_from_path(self) -> None:
        self.assertEqual(BUILDER.derive_ground_truth(["/root/train/fake/model/case/1.png"]), "fake")
        self.assertEqual(BUILDER.derive_ground_truth(["/root/train/real/model/case/1.png"]), "real")

    def test_bbox_shift_is_valid_and_changes_region(self) -> None:
        original = "at <bbox>[100, 200, 300, 500]</bbox>"
        shifted, changed = BUILDER.shift_bboxes(original)
        self.assertTrue(changed)
        values = tuple(float(value) for value in BUILDER.BBOX_RE.search(shifted).groups())
        self.assertTrue(0 <= values[0] < values[2] <= 1000)
        self.assertTrue(0 <= values[1] < values[3] <= 1000)
        self.assertNotEqual(values, (100.0, 200.0, 300.0, 500.0))

    def test_time_shift_stays_inside_frame_range(self) -> None:
        original = "in <t>[0.00, 1.00]</t>"
        shifted, changed = BUILDER.shift_time_intervals(original, [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertTrue(changed)
        start, end = (float(value) for value in BUILDER.TIME_RE.search(shifted).groups())
        self.assertTrue(0 <= start <= end <= 4)
        self.assertNotEqual((start, end), (0.0, 1.0))

    def test_type_swap_keeps_valid_taxonomy(self) -> None:
        original = "<type>Hand Anatomy Error</type>"
        swapped, changed = BUILDER.swap_artifact_types(original)
        self.assertTrue(changed)
        value = BUILDER.TYPE_RE.search(swapped).group(1)
        self.assertIn(value, BUILDER.ARTIFACT_CATEGORY_SET)
        self.assertNotEqual(value, "Hand Anatomy Error")

    def test_static_audit_uses_independent_gt(self) -> None:
        response = (
            "<think><type>Object Deformation</type> in <t>[0, 1]</t> "
            "at <bbox>[0, 0, 500, 500]</bbox></think><answer>Real</answer>"
        )
        audit = BUILDER.static_audit(response, "fake", [0.0, 1.0])
        self.assertFalse(audit["answer_matches_gt"])
        self.assertIn("answer_gt_mismatch", audit["hard_fail_reasons"])


class DataBDeepfakeJudgeEvaluatorTests(unittest.TestCase):
    @staticmethod
    def prediction(sample: str, variant: str, score: int) -> dict:
        return {
            "judge_id": f"{sample}::{variant}",
            "sample_id": sample,
            "variant": variant,
            "prediction": f"<reasoning>test</reasoning><score>{score}</score>",
            "error": None,
            "metadata": {
                "gt_label": "fake",
                "source_bucket": "model",
                "primary_artifact_type": "Object Deformation",
                "source_row_index": 0,
                "static_audit": {"hard_fail_reasons": []},
            },
        }

    def test_control_pair_metrics(self) -> None:
        rows = []
        for index in range(30):
            sample = f"s{index}"
            rows.append(self.prediction(sample, "original", 5))
            rows.append(self.prediction(sample, "shuffled_frames", 2))
            rows.append(self.prediction(sample, "shifted_bbox", 3))
        summary, _items = EVALUATOR.evaluate(rows, expected=len(rows))
        self.assertEqual(summary["paired_controls"]["shuffled_frames"]["original_gt_control_rate"], 1.0)
        self.assertEqual(summary["paired_controls"]["shifted_bbox"]["original_gt_control_rate"], 1.0)
        self.assertEqual(summary["status"], "passed")


if __name__ == "__main__":
    unittest.main()
