from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

from scripts.camera_joint_sft_gate.evaluate_readiness import parse_binary_response
from scripts.camera_joint_sft_gate.summarize_pair import build_summary
from tools import build_camera_joint_sft_gate as builder


def detection_record(image_dir: str, answer: str) -> dict:
    images = [f"{image_dir}/0001.png", f"{image_dir}/0002.png"]
    return {
        "messages": [
            {"role": "system", "content": "Detect generated video artifacts."},
            {"role": "user", "content": "Frame 1: <image>\nFrame 2: <image>"},
            {"role": "assistant", "content": f"<think>test</think>\n<answer>{answer}</answer>"},
        ],
        "images": images,
    }


class CameraJointDataTests(unittest.TestCase):
    def test_stratified_split_is_deterministic_and_pair_level(self) -> None:
        case_ids = [f"dataA_v1_{index:05d}" for index in range(12)]
        labels = ("complex-motion", "minor-motion", "no-motion")
        camera = {
            case_id: {"labels": (labels[index % 3],), "caption": f"caption {index}"}
            for index, case_id in enumerate(case_ids)
        }
        first = builder.stratified_case_split(case_ids, camera, 0.25, 7)
        second = builder.stratified_case_split(case_ids, camera, 0.25, 7)
        self.assertEqual(first, second)
        train, test, _ = first
        self.assertFalse(train & test)
        self.assertEqual(train | test, set(case_ids))
        self.assertEqual(len(test), 3)

    def test_binary_controls_preserve_inputs_and_flip_targets(self) -> None:
        visual_yes = detection_record("/frames/dataA_v1_00001/real", "Real")
        visual_no = detection_record("/frames/dataA_v1_00002/real", "Real")
        yes = builder.make_binary_camera_record(
            "dataA_v1_00001", visual_yes, "pan-left", "Yes", "pan-left:0000",
            "test", "correct", False,
        )
        no = builder.make_binary_camera_record(
            "dataA_v1_00002", visual_no, "pan-left", "No", "pan-left:0000",
            "test", "correct", False,
        )
        opposite = builder.opposite_frame_controls([yes, no])
        self.assertEqual(opposite[0]["images"], no["images"])
        self.assertEqual(opposite[0]["answer"], yes["answer"])
        no_frames = builder.no_frame_controls([yes, no])
        self.assertFalse(no_frames[0]["images"])
        self.assertNotIn("<image>", no_frames[0]["messages"][-1]["content"])

    def test_full_builder_makes_equal_causal_branches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataa: list[dict] = []
            camera_rows: list[dict] = []
            motion = ("complex-motion", "minor-motion", "no-motion")
            for index in range(12):
                case_id = f"dataA_v1_{index + 1:05d}"
                dataa.append(detection_record(f"/frames/{case_id}/real", "Real"))
                dataa.append(detection_record(f"/frames/{case_id}/fake", "Fake"))
                camera_rows.append(
                    {
                        "path": f"/frames/{case_id}/real",
                        "labels": ["no-shaking", motion[index % len(motion)], "regular-speed"],
                        "caption": f"camera caption {index}",
                    }
                )
            datab: list[dict] = []
            datab_camera: list[dict] = []
            for index in range(20):
                for kind, answer in (("real", "Real"), ("fake", "Fake")):
                    frame_dir = f"/datab/{kind}/sample_{index:03d}"
                    datab.append(detection_record(frame_dir, answer))
                    datab_camera.append(
                        {
                            "path": frame_dir,
                            "labels": ["no-shaking", motion[index % len(motion)], "regular-speed"],
                            "caption": "unused",
                        }
                    )
            dataa_path = root / "dataa.json"
            camera_path = root / "dataa_camera.jsonl"
            datab_path = root / "datab.json"
            datab_camera_path = root / "datab_camera.jsonl"
            output = root / "output"
            dataa_path.write_text(json.dumps(dataa), encoding="utf-8")
            datab_path.write_text(json.dumps(datab), encoding="utf-8")
            camera_path.write_text(
                "".join(json.dumps(row) + "\n" for row in camera_rows), encoding="utf-8"
            )
            datab_camera_path.write_text(
                "".join(json.dumps(row) + "\n" for row in datab_camera), encoding="utf-8"
            )
            arguments = [
                "build_camera_joint_sft_gate.py",
                "--dataa-detection-json", str(dataa_path),
                "--dataa-camera-jsonl", str(camera_path),
                "--datab-detection-json", str(datab_path),
                "--datab-camera-jsonl", str(datab_camera_path),
                "--out-dir", str(output),
                "--expected-dataa-cases", "12",
                "--test-ratio", "0.25",
                "--min-per-answer", "1",
                "--seed", "7",
            ]
            with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                builder.main()

            summary = json.loads((output / "camera_joint_sft_data_summary.json").read_text())
            self.assertEqual(summary["split"]["train_cases"], 9)
            self.assertEqual(summary["split"]["test_cases"], 3)
            self.assertTrue(summary["integrity"]["branch_record_counts_equal"])
            self.assertTrue(summary["integrity"]["camera_text_absent_from_detection_prompts"])
            self.assertTrue(summary["shuffled_target_control"]["answer_marginal_preserved"])
            self.assertTrue(summary["shuffled_target_control"]["every_target_is_wrong"])
            self.assertFalse(summary["camera_supervision"]["caption_used_as_training_target"])

            branch_sizes = {value["records"] for value in summary["branch_counts"].values()}
            self.assertEqual(len(branch_sizes), 1)
            correct = json.loads((output / "camera_train_correct.json").read_text())
            flipped = json.loads((output / "camera_train_shuffled.json").read_text())
            self.assertEqual(Counter(row["answer"] for row in correct), Counter(row["answer"] for row in flipped))
            for original, control in zip(correct, flipped):
                self.assertEqual(original["images"], control["images"])
                self.assertEqual(original["messages"][:-1], control["messages"][:-1])
                self.assertNotEqual(original["answer"], control["answer"])

            dev = [
                json.loads(line)
                for line in (output / "camera_dev_matched_frames.jsonl").read_text().splitlines()
            ]
            self.assertTrue(dev)
            self.assertEqual(dev[0]["messages"][-1]["role"], "user")
            per_primitive: dict[str, Counter[str]] = defaultdict(Counter)
            for row in dev:
                per_primitive[row["camera_primitive"]][row["answer"]] += 1
            self.assertTrue(all(counts["Yes"] == counts["No"] for counts in per_primitive.values()))


class CameraJointRuntimeTests(unittest.TestCase):
    def test_binary_response_parser_requires_canonical_answer(self) -> None:
        self.assertEqual(parse_binary_response("Yes"), "Yes")
        self.assertEqual(parse_binary_response(" no \n"), "No")
        self.assertIsNone(parse_binary_response("Yes."))
        self.assertIsNone(parse_binary_response("The answer is No"))

    def test_pair_gate_requires_correct_supervision_and_visual_dependency(self) -> None:
        def metrics(balanced: float, macro_ap: float) -> dict:
            return {
                "coverage": 1.0,
                "num_supported_labels": 32,
                "overall": {
                    "balanced_accuracy": balanced,
                    "average_precision": macro_ap,
                    "roc_auc": macro_ap,
                },
                "macro": {
                    "balanced_accuracy": balanced,
                    "average_precision": macro_ap,
                    "roc_auc": macro_ap,
                },
                "paired_question_accuracy": balanced,
            }

        correct = {
            "conditions": {
                "matched_frames": metrics(0.80, 0.84),
                "opposite_frames": metrics(0.30, 0.35),
                "no_frames": metrics(0.50, 0.50),
            }
        }
        flipped = {"conditions": {"matched_frames": metrics(0.40, 0.45)}}
        summary = build_summary(correct, flipped, 20, 0.03, 0.05, 0.10, 0.08)
        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["checks"]["correct_supervision_beats_flipped_targets"])
        self.assertTrue(summary["checks"]["correct_model_depends_on_visual_frames"])


if __name__ == "__main__":
    unittest.main()
