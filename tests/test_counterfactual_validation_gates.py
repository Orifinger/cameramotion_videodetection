from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_script(relative: str, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(ROOT / relative), *map(str, args)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise AssertionError(
            f"{relative} failed with {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def detection_record(frame_dir: Path, label: str) -> dict:
    images = [str(frame_dir / f"{index:04d}.png") for index in range(1, 5)]
    timestamps = "\n".join(f"[T={(index - 1) * 0.25:.2f}s] <image>" for index in range(1, 5))
    artifact = "Object Identity Drift" if label == "Fake" else "Material Inconsistency"
    return {
        "messages": [
            {"role": "system", "content": "Detect local edits."},
            {"role": "user", "content": f"Frames:\n{timestamps}"},
            {
                "role": "assistant",
                "content": (
                    f"<think><type>{artifact}</type> in <t>[0.00, 0.75]</t> "
                    f"at <bbox>[250, 250, 750, 750]</bbox></think><answer>{label}</answer>"
                ),
            },
        ],
        "images": images,
    }


class CounterfactualGateIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.detection = []
        self.camera_rows = []
        self.index_rows = []
        for case_index in range(1, 5):
            case_id = f"dataA_v1_{case_index:05d}"
            mask = np.zeros((4, 32, 32), dtype=np.uint8)
            mask[:, 8:24, 8:24] = 1
            mask_path = self.root / case_id / "target_mask_gen.npz"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                mask_path,
                frame_indices=np.arange(4, dtype=np.int32),
                masks=mask,
            )
            for split, label in (("real", "Real"), ("fake", "Fake")):
                frame_dir = self.root / case_id / split
                frame_dir.mkdir(parents=True, exist_ok=True)
                for frame_index in range(1, 5):
                    image = np.full((32, 32, 3), 40 + case_index, dtype=np.uint8)
                    if split == "fake":
                        image[8:24, 8:24] = 180
                    Image.fromarray(image).save(frame_dir / f"{frame_index:04d}.png")
                self.detection.append(detection_record(frame_dir, label))
                self.camera_rows.append(
                    {
                        "path": str(frame_dir),
                        "labels": ["no-shaking", "minor-motion", "regular-speed"],
                        "caption": "Minor camera motion.",
                    }
                )
            self.index_rows.append(
                {
                    "case_id": case_id,
                    "mask_npz": str(mask_path),
                    "edit_bbox_xyxy": [8, 8, 24, 24],
                    "evidence_mask": {"mask_shape": {"frame_count": 4, "height": 32, "width": 32}},
                    "operation": "object_attribute_edit",
                    "generator_route": "vace",
                    "vace_model": "vace14b",
                }
            )
        self.detection_json = self.root / "dataa.json"
        self.camera_jsonl = self.root / "camera.jsonl"
        self.index_jsonl = self.root / "grounded.jsonl"
        write_json(self.detection_json, self.detection)
        self.camera_jsonl.write_text(
            "".join(json.dumps(row) + "\n" for row in self.camera_rows), encoding="utf-8"
        )
        self.index_jsonl.write_text(
            "".join(json.dumps(row) + "\n" for row in self.index_rows), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_gate_set_builder_signal_and_pair_evaluator(self) -> None:
        out_dir = self.root / "gate_sets"
        run_script(
            "tools/build_dataa_counterfactual_gate_sets.py",
            "--detection-json", self.detection_json,
            "--camera-jsonl", self.camera_jsonl,
            "--grounded-index-jsonl", self.index_jsonl,
            "--out-dir", out_dir,
            "--frames-per-video", "4",
            "--require-true-mask",
            "--check-mask-files",
        )
        summary = json.loads((out_dir / "dataa_counterfactual_gate_sets_summary.json").read_text())
        self.assertTrue(summary["formal_gate_eligible"])
        self.assertTrue(summary["leakage_audit"]["user_prompts_contain_no_gt_bbox"])
        self.assertEqual(summary["counts"]["selected_train_cases"], 3)
        self.assertEqual(summary["counts"]["selected_test_cases"], 1)
        self.assertEqual(summary["counts"]["dpo_records_per_variant"], 12)
        self.assertEqual(summary["counts"]["eval_records_per_variant"], 2)

        signal_dir = self.root / "signal"
        run_script(
            "tools/dataa_counterfactual_signal_gate.py",
            "--pair-manifest-jsonl", out_dir / "dataa_counterfactual_pair_manifest.jsonl",
            "--out-dir", signal_dir,
            "--workers", "1",
            "--fail-on-gate",
        )
        signal = json.loads((signal_dir / "dataa_counterfactual_signal_gate_summary.json").read_text())
        self.assertEqual(signal["status"], "passed")
        self.assertGreater(signal["overall"]["median_inside_outside_ratio"], 2.0)

        gt_path = out_dir / "dataa_counterfactual_eval_local_only.json"
        gt = json.loads(gt_path.read_text())
        predictions = [
            {
                "data_index": index,
                "case_id": record["case_id"],
                "images": record["images"],
                "response": record["messages"][-1]["content"],
            }
            for index, record in enumerate(gt)
        ]
        pred_path = self.root / "predictions.json"
        write_json(pred_path, predictions)
        pair_dir = self.root / "pair_eval"
        run_script(
            "eval/eval_dataa_counterfactual_pair_gate.py",
            "--gt-json", gt_path,
            "--pred-json", pred_path,
            "--out-dir", pair_dir,
            "--fail-on-gate",
        )
        pair_summary = json.loads((pair_dir / "dataa_counterfactual_pair_gate_summary.json").read_text())
        self.assertEqual(pair_summary["status"], "passed")
        self.assertEqual(pair_summary["overall"]["pair_choice_accuracy"], 1.0)
        self.assertEqual(pair_summary["swap_control"]["swap_consistency_rate"], 1.0)
        self.assertEqual(pair_summary["overall"]["mean_bbox_iou"], 1.0)

        biased = []
        for index, record in enumerate(gt):
            response = re.sub(
                r"<edited_video>[AB]</edited_video>",
                "<edited_video>A</edited_video>",
                record["messages"][-1]["content"],
            )
            biased.append(
                {
                    "data_index": index,
                    "case_id": record["case_id"],
                    "images": record["images"],
                    "response": response,
                }
            )
        biased_path = self.root / "biased_predictions.json"
        biased_dir = self.root / "biased_eval"
        write_json(biased_path, biased)
        run_script(
            "eval/eval_dataa_counterfactual_pair_gate.py",
            "--gt-json", gt_path,
            "--pred-json", biased_path,
            "--out-dir", biased_dir,
        )
        biased_summary = json.loads(
            (biased_dir / "dataa_counterfactual_pair_gate_summary.json").read_text()
        )
        self.assertEqual(biased_summary["status"], "failed")
        self.assertEqual(biased_summary["overall"]["pred_A_rate"], 1.0)

    def test_motion_matched_replay_and_transfer_gate(self) -> None:
        datab, camera_rows = [], []
        for label, split in (("Real", "real"), ("Fake", "fake")):
            for index in range(4):
                frame_dir = self.root / "datab" / split / f"sample_{index}"
                frame_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(frame_dir / "0001.png")
                datab.append(
                    {
                        "messages": [
                            {"role": "system", "content": "Detect."},
                            {"role": "user", "content": "<image>"},
                            {"role": "assistant", "content": f"<think>x</think><answer>{label}</answer>"},
                        ],
                        "images": [str(frame_dir / "0001.png")],
                    }
                )
                camera_rows.append(
                    {"path": str(frame_dir), "labels": ["minor-motion", "regular-speed", "no-shaking"]}
                )
        datab_json = self.root / "datab.json"
        datab_camera = self.root / "datab_camera.jsonl"
        write_json(datab_json, datab)
        datab_camera.write_text("".join(json.dumps(row) + "\n" for row in camera_rows), encoding="utf-8")
        replay_dir = self.root / "replay"
        run_script(
            "tools/build_local_global_detection_replay.py",
            "--dataa-detection-json", self.detection_json,
            "--datab-detection-json", datab_json,
            "--datab-camera-jsonl", datab_camera,
            "--datab-target-samples", "6",
            "--out-dir", replay_dir,
            "--require-target",
        )
        replay = json.loads((replay_dir / "local_global_detection_replay_summary.json").read_text())
        self.assertEqual(replay["datab_replay_labels"], {"Real": 3, "Fake": 3})
        self.assertTrue(replay["leakage_audit"]["dataa_all_train_cases_complete_pairs"])

        def detection_summary(bacc: float, f1: float) -> dict:
            return {"basic": {"accuracy": bacc, "balanced_accuracy": bacc, "fake_f1": f1, "fake_recall": f1}}

        def motion_summary(value: float) -> dict:
            return {
                "by_motion_bucket": {
                    "minor-motion": {"num_samples": 10, "balanced_accuracy": value, "fake_f1": value},
                    "complex-motion": {"num_samples": 10, "balanced_accuracy": value, "fake_f1": value},
                }
            }

        paths = {}
        for name, bacc, f1, motion in (
            ("control", 0.50, 0.50, 0.50),
            ("pair", 0.54, 0.54, 0.52),
            ("camera", 0.56, 0.56, 0.54),
        ):
            paths[f"{name}_dataa"] = self.root / f"{name}_dataa.json"
            paths[f"{name}_motion"] = self.root / f"{name}_motion.json"
            write_json(paths[f"{name}_dataa"], detection_summary(bacc, f1))
            write_json(paths[f"{name}_motion"], motion_summary(motion))
        transfer_out = self.root / "transfer.json"
        run_script(
            "eval/eval_counterfactual_transfer_gate.py",
            "--control-dataa-summary", paths["control_dataa"],
            "--control-motion-summary", paths["control_motion"],
            "--control-vif-acc", "0.84", "--control-vif-f1", "0.85",
            "--pair-dataa-summary", paths["pair_dataa"],
            "--pair-motion-summary", paths["pair_motion"],
            "--pair-vif-acc", "0.835", "--pair-vif-f1", "0.845",
            "--camera-dataa-summary", paths["camera_dataa"],
            "--camera-motion-summary", paths["camera_motion"],
            "--camera-vif-acc", "0.833", "--camera-vif-f1", "0.843",
            "--out", transfer_out,
            "--fail-on-gate",
        )
        transfer = json.loads(transfer_out.read_text())
        self.assertEqual(transfer["status"], "passed")
        self.assertEqual(transfer["camera_contribution_gate"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
