from __future__ import annotations

import unittest

from scripts.camera_pprl.summarize import build_summary
from tools.audit_camera_pprl_smoke import build_audit
from tools.build_camera_pprl_binary import convert_record, select_balanced_pairs


def camera_record(pair_id: str, primitive: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Answer Yes or No."},
            {"role": "user", "content": "<image><image>Is this pan-left?"},
            {"role": "assistant", "content": answer},
        ],
        "images": ["frame_1.png", "frame_2.png"],
        "answer": answer,
        "camera_primitive": primitive,
        "case_id": f"case-{pair_id}",
        "pair_id": pair_id,
        "sample_id": f"{pair_id}-{answer}",
        "source_family": "dataA_v1",
        "motion_bucket": "complex-motion",
    }


def camera_eval(balanced: float, macro_ap: float, opposite: float = 0.25) -> dict:
    return {
        "conditions": {
            "matched_frames": {
                "coverage": 1.0,
                "num_supported_labels": 32,
                "overall": {"balanced_accuracy": balanced},
                "macro": {"average_precision": macro_ap, "roc_auc": macro_ap},
                "paired_question_accuracy": 0.5,
            },
            "opposite_frames": {"overall": {"balanced_accuracy": opposite}},
            "no_frames": {"overall": {"balanced_accuracy": 0.5}},
        }
    }


def vif_eval(
    balanced: float,
    fake_f1: float,
    *,
    expected_predictions: int = 200,
    generator: str = "generator",
) -> dict:
    return {
        "num_expected_predictions": expected_predictions,
        "num_matched_predictions": expected_predictions,
        "coverage": 1.0,
        "format_valid_rate": 1.0,
        "average_across_fake_models": {
            "num_models": 1,
            "balanced_accuracy": balanced,
            "fake_recall": 0.6,
            "fake_f1": fake_f1,
        },
        "per_fake_model": {
            generator: {
                "num_pairs": 100,
                "real_recall": 0.6,
                "fake_recall": 0.6,
                "fake_precision": 0.6,
                "fake_f1": fake_f1,
                "balanced_accuracy": balanced,
                "confusion": {
                    "real_as_fake": 40,
                    "fake_as_fake": 60,
                },
            }
        },
    }


class CameraPprlDataTests(unittest.TestCase):
    def test_selection_keeps_complete_balanced_pairs(self) -> None:
        rows = []
        for index, primitive in enumerate(("pan-left", "pan-right", "tilt-up")):
            pair_id = f"pair-{index}"
            rows.extend(
                [
                    camera_record(pair_id, primitive, "Yes"),
                    camera_record(pair_id, primitive, "No"),
                ]
            )
        selected = select_balanced_pairs(rows, max_records=4, seed=7)
        by_pair: dict[str, set[str]] = {}
        for row in selected:
            by_pair.setdefault(row["pair_id"], set()).add(row["answer"])
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(answers == {"Yes", "No"} for answers in by_pair.values()))

    def test_conversion_removes_supervised_answer(self) -> None:
        converted = convert_record(camera_record("pair-1", "pan-left", "Yes"))
        self.assertEqual(converted["solution"], "Yes")
        self.assertTrue(all(message["role"] != "assistant" for message in converted["messages"]))
        self.assertEqual(len(converted["images"]), 2)

    def test_smoke_audit_requires_nonzero_group_advantages(self) -> None:
        passed = build_audit(
            [
                {"frac_reward_zero_std": 0.5, "reward_std": 0.2},
                {"train/frac_reward_zero_std": 0.75, "reward_std": 0.1},
            ],
            ["logging.jsonl"],
            max_zero_std_rate=0.8,
            min_log_points=2,
        )
        failed = build_audit(
            [
                {"frac_reward_zero_std": 1.0},
                {"frac_reward_zero_std": 0.9},
            ],
            ["logging.jsonl"],
            max_zero_std_rate=0.8,
            min_log_points=2,
        )
        self.assertEqual(passed["status"], "passed")
        self.assertEqual(failed["status"], "failed")


class CameraPprlSummaryTests(unittest.TestCase):
    def test_direct_pprl_candidate_is_separate_from_recovery(self) -> None:
        summary = build_summary(
            camera_eval(0.74, 0.86),
            camera_eval(0.76, 0.87),
            camera_eval(0.75, 0.85),
            vif_eval(0.60, 0.60),
            vif_eval(0.62, 0.62),
            vif_eval(0.62, 0.62),
            vif_eval(0.63, 0.63),
            vif_eval(0.60, 0.60),
        )
        self.assertEqual(summary["status"], "direct_pprl_candidate")
        self.assertAlmostEqual(
            summary["vif_deltas"]["pprl_minus_joint_sft"]["balanced_accuracy"],
            0.02,
        )

    def test_no_transfer_is_not_promoted_by_camera_retention(self) -> None:
        summary = build_summary(
            camera_eval(0.74, 0.86),
            camera_eval(0.75, 0.87),
            camera_eval(0.75, 0.86),
            vif_eval(0.60, 0.60),
            vif_eval(0.60, 0.60),
            vif_eval(0.60, 0.60),
            vif_eval(0.60, 0.60),
            vif_eval(0.60, 0.60),
        )
        self.assertEqual(summary["status"], "no_detection_transfer")

    def test_positive_core_gate_waits_for_detection_only_reference(self) -> None:
        summary = build_summary(
            camera_eval(0.74, 0.86),
            camera_eval(0.76, 0.87),
            camera_eval(0.75, 0.85),
            vif_eval(0.60, 0.60),
            vif_eval(0.62, 0.62),
            vif_eval(0.62, 0.62),
            vif_eval(0.63, 0.63),
        )
        self.assertEqual(summary["status"], "pending_detection_only_reference")

    def test_pprl_rebound_below_detection_control_is_not_a_candidate(self) -> None:
        summary = build_summary(
            camera_eval(0.74, 0.86),
            camera_eval(0.76, 0.87),
            camera_eval(0.75, 0.85),
            vif_eval(0.60, 0.60),
            vif_eval(0.62, 0.62),
            vif_eval(0.62, 0.62),
            vif_eval(0.63, 0.63),
            vif_eval(0.65, 0.65),
        )
        self.assertEqual(summary["status"], "no_detection_transfer")

    def test_incompatible_detection_reference_is_not_a_candidate(self) -> None:
        summary = build_summary(
            camera_eval(0.74, 0.86),
            camera_eval(0.76, 0.87),
            camera_eval(0.75, 0.85),
            vif_eval(0.60, 0.60),
            vif_eval(0.62, 0.62),
            vif_eval(0.62, 0.62),
            vif_eval(0.63, 0.63),
            vif_eval(0.60, 0.60, expected_predictions=198, generator="other"),
        )
        self.assertEqual(summary["status"], "no_detection_transfer")
        checks = summary["checks"]["direct_pprl_vs_detection_only"]
        self.assertFalse(checks["same_expected_predictions"])
        self.assertFalse(checks["same_fake_model_set"])

    def test_recovery_must_keep_visual_camera_dependence(self) -> None:
        summary = build_summary(
            camera_eval(0.74, 0.86),
            camera_eval(0.75, 0.87),
            camera_eval(0.55, 0.86, opposite=0.50),
            vif_eval(0.60, 0.60),
            vif_eval(0.60, 0.60),
            vif_eval(0.60, 0.60),
            vif_eval(0.62, 0.62),
            vif_eval(0.60, 0.60),
        )
        self.assertEqual(summary["status"], "no_detection_transfer")
        self.assertFalse(
            summary["checks"]["camera"]["recovery_depends_on_matched_frames"]
        )


if __name__ == "__main__":
    unittest.main()
