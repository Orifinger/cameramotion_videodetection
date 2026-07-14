from __future__ import annotations

import unittest
from collections import Counter, defaultdict

from scripts.camera_detection_joint_grpo.summarize import (
    blocks_pipeline,
    build_dataa_summary,
    build_vif_summary,
)
from tools.build_camera_detection_joint_grpo import (
    JOINT_SYSTEM_PROMPT,
    assistant_text,
    make_record,
    shuffled_reward_records,
    without_assistant,
)


def source_record(kind: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "old"},
            {
                "role": "user",
                "content": "Frames:\n[T=0] <image>\n[T=1] <image>\n\nOld question.",
            },
            {"role": "assistant", "content": f"<answer>{kind}</answer>"},
        ],
        "images": [f"/{kind.lower()}/1.png", f"/{kind.lower()}/2.png"],
    }


def joint_row(source: str, answer: str, case_id: str, labels: list[str]) -> dict:
    return {
        "messages": [{"role": "user", "content": "<image>"}],
        "images": ["frame.png"],
        "camera_labels": labels,
        "camera_labels_gold": labels,
        "camera_labels_reward": labels,
        "detection_label": answer,
        "sample_id": f"{source}:{answer}:{case_id}",
        "case_id": case_id,
        "source_dataset": source,
    }


def dataa_eval(balanced: float, f1: float, pair: float) -> dict:
    return {
        "num_gt_records": 200,
        "num_matched_records": 200,
        "basic": {
            "format_valid_rate": 1.0,
            "accuracy": balanced,
            "balanced_accuracy": balanced,
            "fake_recall": f1,
            "real_recall": balanced,
            "fake_f1": f1,
        },
        "pair": {"pair_accuracy": pair, "num_pairs": 100},
    }


def vif_eval(balanced: float, f1: float) -> dict:
    return {
        "num_expected_predictions": 200,
        "num_matched_predictions": 200,
        "coverage": 1.0,
        "format_valid_rate": 1.0,
        "average_across_fake_models": {
            "num_models": 1,
            "balanced_accuracy": balanced,
            "fake_recall": 0.6,
            "fake_f1": f1,
        },
        "per_fake_model": {
            "generator": {
                "num_pairs": 100,
                "balanced_accuracy": balanced,
                "real_recall": 0.6,
                "fake_recall": 0.6,
                "fake_precision": 0.6,
                "fake_f1": f1,
                "confusion": {"real_as_fake": 40, "fake_as_fake": 60},
            }
        },
    }


class JointDataTests(unittest.TestCase):
    def test_joint_target_orders_camera_before_detection(self) -> None:
        text = assistant_text(["no-motion", "regular-speed"], "Fake")
        self.assertLess(text.index("<camera_motion>"), text.index("<answer>"))
        row = make_record(
            source_record("Fake"),
            camera_labels=["no-motion", "regular-speed"],
            source_dataset="dataa_local_edit",
            sample_id="sample",
            case_id="case",
            include_assistant=True,
        )
        self.assertEqual(row["messages"][0]["content"], JOINT_SYSTEM_PROMPT)
        self.assertEqual(row["detection_label"], "Fake")
        self.assertEqual(sum(x["content"].count("<image>") for x in row["messages"]), 2)

    def test_grpo_conversion_removes_assistant(self) -> None:
        row = make_record(
            source_record("Real"),
            camera_labels=["pan-left"],
            source_dataset="datab_full_generation_replay",
            sample_id="sample",
            case_id="case",
            include_assistant=True,
        )
        converted = without_assistant([row])[0]
        self.assertTrue(all(message["role"] != "assistant" for message in converted["messages"]))

    def test_shuffled_control_preserves_class_conditioned_marginals(self) -> None:
        rows = []
        labels = [["pan-left"], ["pan-right"], ["tilt-up"], ["tilt-down"]]
        for index, camera in enumerate(labels):
            for answer in ("Real", "Fake"):
                rows.append(joint_row("dataa_local_edit", answer, f"a{index}", camera))
        for answer in ("Real", "Fake"):
            for index, camera in enumerate(labels):
                rows.append(
                    joint_row(
                        "datab_full_generation_replay",
                        answer,
                        f"b-{answer}-{index}",
                        camera,
                    )
                )
        shuffled = shuffled_reward_records(rows)
        changed = sum(
            row["camera_labels_reward"] != row["camera_labels_gold"] for row in shuffled
        )
        self.assertEqual(changed, len(rows))
        for source in {row["source_dataset"] for row in rows}:
            for answer in ("Real", "Fake"):
                gold = Counter(
                    tuple(row["camera_labels_gold"])
                    for row in rows
                    if row["source_dataset"] == source and row["detection_label"] == answer
                )
                control = Counter(
                    tuple(row["camera_labels_reward"])
                    for row in shuffled
                    if row["source_dataset"] == source and row["detection_label"] == answer
                )
                self.assertEqual(gold, control)
        per_case: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        for row in shuffled:
            if row["source_dataset"] == "dataa_local_edit":
                per_case[row["case_id"]].add(tuple(row["camera_labels_reward"]))
        self.assertTrue(all(len(values) == 1 for values in per_case.values()))


class JointSummaryTests(unittest.TestCase):
    def test_only_vif_failure_blocks_the_pipeline(self) -> None:
        self.assertFalse(blocks_pipeline("dataa", "failed"))
        self.assertFalse(blocks_pipeline("dataa", "passed"))
        self.assertFalse(blocks_pipeline("vif", "camera_candidate"))
        self.assertTrue(blocks_pipeline("vif", "no_camera_gain"))

    def test_dataa_gate_uses_detection_endpoints(self) -> None:
        summary = build_dataa_summary(
            dataa_eval(0.60, 0.60, 0.25),
            dataa_eval(0.63, 0.63, 0.28),
            dataa_eval(0.61, 0.61, 0.26),
            dataa_eval(0.60, 0.60, 0.25),
            min_coverage=0.99,
            min_format=0.95,
            min_gain=0.02,
            max_drop=0.01,
        )
        self.assertEqual(summary["status"], "passed")

    def test_camera_gain_cannot_hide_detection_failure(self) -> None:
        summary = build_dataa_summary(
            dataa_eval(0.60, 0.60, 0.25),
            dataa_eval(0.59, 0.59, 0.24),
            dataa_eval(0.61, 0.61, 0.26),
            dataa_eval(0.60, 0.60, 0.25),
            min_coverage=0.99,
            min_format=0.95,
            min_gain=0.02,
            max_drop=0.01,
        )
        self.assertEqual(summary["status"], "failed")

    def test_vif_gate_requires_both_camera_controls(self) -> None:
        summary = build_vif_summary(
            vif_eval(0.60, 0.60),
            vif_eval(0.63, 0.63),
            vif_eval(0.61, 0.61),
            vif_eval(0.60, 0.60),
            min_coverage=0.99,
            min_format=0.99,
            min_gain=0.01,
            max_drop=0.005,
        )
        self.assertEqual(summary["status"], "camera_candidate")


if __name__ == "__main__":
    unittest.main()
