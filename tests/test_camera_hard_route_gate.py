import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.camera_hard_route_gate.route_manifest import (
    aggregate_routes,
    audit_binary_routes,
    build_vif_inputs,
    compose_predictions,
    summarize_gate,
)
from tools.build_camera_hard_route_gate import balanced_router_training_records, route_bucket


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class CameraRouteBucketTest(unittest.TestCase):
    def test_builder_module_entrypoint_resolves_project_tools_package(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tools.build_camera_hard_route_gate", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dataa-detection-json", result.stdout)

    def test_priority_and_static_alias(self) -> None:
        self.assertEqual(route_bucket(["static"]), "no-motion")
        self.assertEqual(route_bucket(["no_motion"]), "no-motion")
        self.assertEqual(route_bucket(["minor-motion"]), "minor-motion")
        self.assertEqual(route_bucket(["minor-motion", "complex-motion"]), "complex-motion")
        self.assertEqual(route_bucket([]), "unknown")

    def test_router_questions_have_equal_frequency_and_balanced_answers(self) -> None:
        assignments = {
            "dataA_v1_00001": "no-motion",
            "dataA_v1_00002": "no-motion",
            "dataA_v1_00003": "minor-motion",
            "dataA_v1_00004": "minor-motion",
            "dataA_v1_00005": "complex-motion",
            "dataA_v1_00006": "complex-motion",
            "dataA_v1_00007": "complex-motion",
        }
        pairs = {
            case_id: {
                "real": {
                    "images": [f"/tmp/{case_id}/real/1.png"],
                    "messages": [{"role": "assistant", "content": "<answer>Real</answer>"}],
                }
            }
            for case_id in assignments
        }
        camera = {
            case_id: {"route_bucket": bucket} for case_id, bucket in assignments.items()
        }
        rows, audit = balanced_router_training_records(
            sorted(assignments), pairs, camera, seed=7
        )
        counts = {}
        for bucket in ("no-motion", "minor-motion", "complex-motion"):
            selected = [row for row in rows if row["camera_primitive"] == bucket]
            counts[bucket] = len(selected)
            self.assertEqual(
                {answer: sum(row["answer"] == answer for row in selected) for answer in ("Yes", "No")},
                {"Yes": 2, "No": 2},
            )
        self.assertEqual(set(counts.values()), {4})
        self.assertEqual(audit["equal_frequency_contract"]["records_per_question"], 4)


class CameraRouteManifestTest(unittest.TestCase):
    def test_binary_audit_merges_minor_and_complex_without_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "three_class.jsonl"
            output_path = root / "binary.jsonl"
            summary_path = root / "summary.json"
            rows = []
            definitions = [
                ("case-1", "no-motion", "no-motion"),
                ("case-2", "no-motion", "no-motion"),
                ("case-3", "minor-motion", "complex-motion"),
                ("case-4", "complex-motion", "minor-motion"),
            ]
            for case_id, gold, predicted in definitions:
                for visual_kind in ("real", "fake"):
                    rows.append(
                        {
                            "video_id": f"{case_id}:{visual_kind}",
                            "case_id": case_id,
                            "visual_kind": visual_kind,
                            "source_family": "test-family",
                            "route_gold_bucket": gold,
                            "predicted_bucket": predicted,
                        }
                    )
            write_jsonl(input_path, rows)
            audit_binary_routes(
                argparse.Namespace(
                    input_manifest=str(input_path),
                    output_manifest=str(output_path),
                    output_summary=str(summary_path),
                    min_coverage=1.0,
                    min_accuracy=0.75,
                    min_balanced_accuracy=0.75,
                    min_per_route_recall=0.70,
                    min_pair_consistency=0.90,
                )
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            binary_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["metrics"]["accuracy"], 1.0)
            self.assertEqual(summary["metrics"]["real_fake_pair_route_consistency"], 1.0)
            self.assertTrue(all(row["binary_predicted_bucket"] in {"no-motion", "motion"} for row in binary_rows))
            self.assertTrue(
                all(
                    row["binary_wrong_route_bucket"] != row["binary_predicted_bucket"]
                    for row in binary_rows
                )
            )

    def test_build_vif_inputs_uses_exact_index_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_dir = root / "indices"
            frame_dir = root / "test_normalized" / "real" / "case-1"
            frame_dir.mkdir(parents=True)
            for index in range(1, 17):
                (frame_dir / f"{index}.png").write_bytes(b"x")
            (frame_dir / "timestamps.txt").write_text(
                "\n".join(str(index / 10) for index in range(16)), encoding="utf-8"
            )
            write_json(index_dir / "test_index.rank0.json", {"real": [str(frame_dir)]})
            output = root / "route_questions.jsonl"
            summary = root / "summary.json"
            build_vif_inputs(
                argparse.Namespace(
                    index_dir=str(index_dir),
                    output_jsonl=str(output),
                    summary_json=str(summary),
                    expected_ranks=1,
                    expected_frames=16,
                    check_frame_dirs=True,
                    require_timestamps=True,
                    allow_frame_count_mismatch=False,
                )
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["camera_primitive"] for row in rows}, {
                "no-motion", "minor-motion", "complex-motion"
            })
            self.assertTrue(all(len(row["images"]) == 16 for row in rows))

    def test_aggregate_builds_three_class_route_and_cyclic_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "inputs.jsonl"
            score_dir = root / "scores"
            rows = []
            scores = []
            definitions = {
                "case-1:real": ("no-motion", {"no-motion": 2.0, "minor-motion": 0.0, "complex-motion": -1.0}),
                "case-2:fake": ("complex-motion", {"no-motion": -1.0, "minor-motion": 0.0, "complex-motion": 2.0}),
            }
            for video_id, (gold, values) in definitions.items():
                for bucket, score in values.items():
                    sample_id = f"route:{video_id}:{bucket}"
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "video_id": video_id,
                            "case_id": video_id.split(":")[0],
                            "visual_kind": video_id.split(":")[1],
                            "camera_primitive": bucket,
                            "route_gold_bucket": gold,
                        }
                    )
                    scores.append(
                        {
                            "sample_id": sample_id,
                            "camera_primitive": bucket,
                            "yes_minus_no_score": score,
                        }
                    )
            write_jsonl(input_path, rows)
            write_jsonl(score_dir / "rank_00.jsonl", scores)
            manifest_path = root / "manifest.jsonl"
            summary_path = root / "summary.json"
            aggregate_routes(
                argparse.Namespace(
                    input_jsonl=str(input_path),
                    prediction_dir=str(score_dir),
                    output_manifest=str(manifest_path),
                    summary_json=str(summary_path),
                    min_top_probability=0.0,
                    min_margin=0.0,
                )
            )
            manifest = [json.loads(line) for line in manifest_path.read_text().splitlines()]
            by_id = {row["video_id"]: row for row in manifest}
            self.assertEqual(by_id["case-1:real"]["route_bucket"], "no-motion")
            self.assertEqual(by_id["case-1:real"]["cyclic_route_bucket"], "minor-motion")
            self.assertEqual(by_id["case-2:fake"]["route_bucket"], "complex-motion")
            self.assertEqual(json.loads(summary_path.read_text())["heldout_route_metrics"]["accuracy"], 1.0)

    def test_compose_selects_predictions_without_changing_video_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_dir = root / "indices"
            real_dir = root / "test_normalized" / "real" / "case-1"
            fake_dir = root / "test_normalized" / "model-a" / "case-1"
            real_dir.mkdir(parents=True)
            fake_dir.mkdir(parents=True)
            write_json(
                index_dir / "test_index.rank0.json",
                {"real": [str(real_dir)], "model-a": [str(fake_dir)]},
            )
            manifest_path = root / "manifest.jsonl"
            write_jsonl(
                manifest_path,
                [
                    {
                        "video_id": "real/case-1",
                        "predicted_bucket": "no-motion",
                        "route_bucket": "no-motion",
                        "cyclic_route_bucket": "minor-motion",
                        "fallback_to_shared": False,
                        "top_relative_probability": 0.8,
                        "relative_probability_margin": 0.6,
                    },
                    {
                        "video_id": "model-a/case-1",
                        "predicted_bucket": "complex-motion",
                        "route_bucket": "complex-motion",
                        "cyclic_route_bucket": "no-motion",
                        "fallback_to_shared": False,
                        "top_relative_probability": 0.7,
                        "relative_probability_margin": 0.4,
                    },
                ],
            )
            paths = {}
            for expert in ("no-motion", "minor-motion", "complex-motion", "shared"):
                path = root / expert
                paths[expert] = path
                write_json(
                    path / "rank_0" / "predictions.json",
                    [
                        {
                            "video_id": "real/case-1",
                            "aigc_model_name": "real",
                            "answer": "Real",
                            "expert_marker": expert,
                        },
                        {
                            "video_id": "model-a/case-1",
                            "aigc_model_name": "model-a",
                            "answer": "Fake",
                            "expert_marker": expert,
                        },
                    ],
                )
            output = root / "composed.json"
            summary = root / "composed_summary.json"
            compose_predictions(
                argparse.Namespace(
                    index_dir=str(index_dir),
                    route_manifest=str(manifest_path),
                    expert=[
                        ("no-motion", str(paths["no-motion"])),
                        ("minor-motion", str(paths["minor-motion"])),
                        ("complex-motion", str(paths["complex-motion"])),
                    ],
                    shared_prediction_dir=str(paths["shared"]),
                    route_mode="predicted",
                    output_predictions=str(output),
                    output_summary=str(summary),
                    expected_ranks=1,
                )
            )
            selected = {row["video_id"]: row for row in json.loads(output.read_text())}
            self.assertEqual(set(selected), {"real/case-1", "model-a/case-1"})
            self.assertEqual(selected["real/case-1"]["expert_marker"], "no-motion")
            self.assertEqual(selected["model-a/case-1"]["expert_marker"], "complex-motion")
            self.assertEqual(json.loads(summary.read_text())["evaluation"]["coverage"], 1.0)

    def test_summary_requires_route_to_beat_original_and_controls(self) -> None:
        def evaluation(balanced_accuracy: float, fake_f1: float) -> dict:
            metrics = {
                "balanced_accuracy": balanced_accuracy,
                "fake_recall": fake_f1,
                "fake_f1": fake_f1,
            }
            return {
                "coverage": 1.0,
                "format_valid_rate": 1.0,
                "average_across_fake_models": metrics,
                "per_fake_model": {"model-a": metrics},
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.json"
            shared = root / "shared.json"
            predicted = root / "predicted.json"
            cyclic = root / "cyclic.json"
            output = root / "gate.json"
            write_json(base, evaluation(0.70, 0.70))
            write_json(shared, {"evaluation": evaluation(0.71, 0.71)})
            write_json(predicted, {"evaluation": evaluation(0.73, 0.73)})
            write_json(cyclic, {"evaluation": evaluation(0.69, 0.69)})

            summarize_gate(
                argparse.Namespace(
                    base_eval=str(base),
                    shared_summary=str(shared),
                    predicted_summary=str(predicted),
                    cyclic_summary=str(cyclic),
                    output_json=str(output),
                    min_base_gain=0.005,
                    min_shared_gain=0.005,
                    min_cyclic_gain=0.01,
                    max_other_drop=0.005,
                )
            )
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "passed")
            self.assertTrue(summary["checks"]["predicted_route_beats_original_base"])
            self.assertAlmostEqual(
                summary["deltas"]["predicted_minus_base"]["balanced_accuracy"], 0.03
            )
            self.assertEqual(
                summary["thresholds"]["min_predicted_minus_original_base_primary_gain"],
                0.005,
            )


if __name__ == "__main__":
    unittest.main()
