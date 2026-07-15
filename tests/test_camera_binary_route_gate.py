import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.camera_binary_route_gate.route import compose_predictions, map_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def detection_record(identifier: str, bucket: str, label: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "<image>\nDetermine whether the video is real or fake."},
            {"role": "assistant", "content": f"<answer>{label}</answer>"},
        ],
        "images": [f"/tmp/{identifier}.png"],
        "route_record_id": identifier,
        "sample_id": identifier,
        "gate_task": "detection",
        "route_bucket": bucket,
        "route_domain": "unit-test",
        "gate_source": "unit-test",
        "detection_label": label,
    }


class CameraBinaryRouteDataTest(unittest.TestCase):
    def test_build_and_install_preserve_exact_binary_union(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hard"
            output = root / "binary"
            lf_data = root / "llamafactory"
            no_motion = [
                detection_record("no-real", "no-motion", "Real"),
                detection_record("no-fake", "no-motion", "Fake"),
            ]
            minor = [
                detection_record("minor-real", "minor-motion", "Real"),
                detection_record("minor-fake", "minor-motion", "Fake"),
            ]
            complex_motion = [
                detection_record("complex-real", "complex-motion", "Real"),
                detection_record("complex-fake", "complex-motion", "Fake"),
            ]
            write_json(source / "hard_route_no_motion.json", no_motion)
            write_json(source / "hard_route_minor_motion.json", minor)
            write_json(source / "hard_route_complex_motion.json", complex_motion)
            write_json(source / "hard_route_shared.json", no_motion + minor + complex_motion)
            audit = root / "binary_audit.json"
            write_json(audit, {"status": "passed", "checks": {"all": True}})

            build = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.build_camera_binary_route_gate",
                    "--hard-route-data-dir",
                    str(source),
                    "--binary-audit-summary",
                    str(audit),
                    "--out-dir",
                    str(output),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            summary = json.loads((output / "camera_binary_route_data_summary.json").read_text())
            self.assertTrue(summary["shared_is_exact_disjoint_binary_expert_union"])
            self.assertEqual(summary["outputs"]["shared"]["records"], 6)
            self.assertEqual(summary["outputs"]["no-motion"]["records"], 2)
            self.assertEqual(summary["outputs"]["motion"]["records"], 4)

            lf_data.mkdir()
            write_json(lf_data / "dataset_info.json", {})
            install = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.install_camera_binary_route_gate",
                    "--source-dir",
                    str(output),
                    "--llamafactory-data-dir",
                    str(lf_data),
                    "--smoke-samples",
                    "6",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(install.returncode, 0, install.stderr)
            dataset_info = json.loads((lf_data / "dataset_info.json").read_text())
            self.assertIn("camera_binary_route_shared", dataset_info)
            self.assertIn("camera_binary_route_no_motion", dataset_info)
            self.assertIn("camera_binary_route_motion", dataset_info)


class CameraBinaryRouteVifTest(unittest.TestCase):
    def test_map_and_wrong_route_swap_experts(self) -> None:
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
            three_class = root / "three_class.jsonl"
            write_jsonl(
                three_class,
                [
                    {
                        "video_id": "real/case-1",
                        "aigc_model_name": "real",
                        "predicted_bucket": "no-motion",
                    },
                    {
                        "video_id": "model-a/case-1",
                        "aigc_model_name": "model-a",
                        "predicted_bucket": "complex-motion",
                    },
                ],
            )
            binary_manifest = root / "binary.jsonl"
            route_summary = root / "route_summary.json"
            map_manifest(
                argparse.Namespace(
                    input_manifest=str(three_class),
                    output_manifest=str(binary_manifest),
                    output_summary=str(route_summary),
                )
            )

            paths = {}
            for expert in ("no-motion", "motion", "shared"):
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
            output = root / "wrong_predictions.json"
            summary = root / "wrong_summary.json"
            compose_predictions(
                argparse.Namespace(
                    index_dir=str(index_dir),
                    route_manifest=str(binary_manifest),
                    expert=[
                        ("no-motion", str(paths["no-motion"])),
                        ("motion", str(paths["motion"])),
                    ],
                    shared_prediction_dir=str(paths["shared"]),
                    route_mode="wrong",
                    output_predictions=str(output),
                    output_summary=str(summary),
                    expected_ranks=1,
                )
            )
            selected = {row["video_id"]: row for row in json.loads(output.read_text())}
            self.assertEqual(selected["real/case-1"]["expert_marker"], "motion")
            self.assertEqual(selected["model-a/case-1"]["expert_marker"], "no-motion")
            self.assertEqual(json.loads(summary.read_text())["evaluation"]["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
