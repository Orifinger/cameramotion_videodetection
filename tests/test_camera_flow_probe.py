from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from scripts.camera_flow_probe.contracts import RunSpec, build_probe_manifest
from scripts.camera_flow_probe.metrics import roc_auc
from scripts.camera_flow_probe.select_manifest import select_rows

try:
    import cv2  # noqa: F401

    from scripts.camera_flow_probe.geometry import (
        canvas_geometry,
        dense_transform_flow,
        fit_global_camera_transform,
        project_points,
    )
    from scripts.camera_flow_probe.masks import load_mask_tube
except ModuleNotFoundError:
    cv2 = None

try:
    import torch

    from scripts.camera_flow_probe.features import (
        GLOBAL_FEATURE_DIM,
        LOCAL_TRAJECTORY_DIM,
        align_patch_sequence,
        global_restrav_features,
        local_trajectory_features,
    )
    from scripts.camera_flow_probe.train_probe import main as train_probe_main
except ModuleNotFoundError:
    torch = None


class CameraFlowContractTests(unittest.TestCase):
    def test_build_manifest_enforces_sources_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_a, run_b = root / "run_a", root / "run_b"
            run_a.mkdir()
            run_b.mkdir()
            records = root / "records.jsonl"
            camera = root / "camera.jsonl"
            split = root / "test.json"
            rows = []
            camera_rows = []
            for index, run in enumerate((run_a, run_a, run_b), 1):
                case_id = f"dataA_v1_{index:05d}"
                attempt = run / case_id
                attempt.mkdir()
                paths = {}
                for name in ("real.mp4", "fake.mp4", "mask.npz", "case_manifest.json"):
                    path = attempt / name
                    path.write_bytes(b"x")
                    paths[name] = path
                rows.append(
                    {
                        "case_id": case_id,
                        "vace_model": "vace14b" if run == run_a else "vace13b",
                        "real_video": str(paths["real.mp4"]),
                        "fake_video": str(paths["fake.mp4"]),
                        "mask_npz": str(paths["mask.npz"]),
                        "case_manifest_path": str(paths["case_manifest.json"]),
                    }
                )
                for role in ("real", "fake"):
                    camera_rows.append(
                        {
                            "path": f"/frames/{case_id}/{role}",
                            "labels": ["complex-motion"],
                            "caption": "camera moves",
                        }
                    )
            records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            camera.write_text("".join(json.dumps(row) + "\n" for row in camera_rows), encoding="utf-8")
            split.write_text(json.dumps([{"images": ["/frames/dataA_v1_00003/real/0001.png"]}]), encoding="utf-8")
            out = root / "manifest.jsonl"
            summary = build_probe_manifest(
                records_jsonl=records,
                camera_jsonl=camera,
                test_split=split,
                out_jsonl=out,
                out_summary=root / "summary.json",
                run_specs=(
                    RunSpec("a", run_a, 2, "vace14b"),
                    RunSpec("b", run_b, 1, "vace13b"),
                ),
                expected_cases=3,
                expected_test_cases=1,
                check_files=True,
                strict_final_contract=True,
            )
            self.assertEqual(summary["source_counts"], {"a": 2, "b": 1})
            self.assertEqual(summary["split_counts"], {"train": 2, "test": 1})


@unittest.skipIf(cv2 is None, "OpenCV is not installed in the local test runtime")
class CameraGeometryTests(unittest.TestCase):
    def test_dense_flow_recovers_camera_homography_with_outliers(self) -> None:
        transform = np.array(
            [[1.01, 0.01, 4.0], [-0.01, 1.01, -3.0], [0.00002, -0.00001, 1.0]],
            dtype=np.float64,
        )
        flow = dense_transform_flow(transform, 160, 240)
        rng = np.random.default_rng(7)
        flow[40:90, 80:150] += rng.normal(0.0, 20.0, size=(50, 70, 2)).astype(np.float32)
        estimated, stats = fit_global_camera_transform(flow, grid_step=6, max_fb_error=100.0)
        points = np.array([[20, 20], [200, 30], [30, 130], [210, 140], [120, 80]], dtype=np.float64)
        error = np.linalg.norm(project_points(points, estimated) - project_points(points, transform), axis=1)
        self.assertLess(float(np.median(error)), 0.5)
        self.assertGreater(stats["inlier_rate"], 0.6)

    def test_canvas_geometry_preserves_aspect_ratio(self) -> None:
        geometry = canvas_geometry(480, 832, long_side=512, multiple=8)
        self.assertEqual(geometry.canvas_width % 8, 0)
        self.assertEqual(geometry.canvas_height % 8, 0)
        self.assertAlmostEqual(geometry.scale, 512 / 832)


@unittest.skipIf(torch is None, "PyTorch is not installed in the local test runtime")
class CameraFeatureTests(unittest.TestCase):
    def test_global_and_local_feature_shapes(self) -> None:
        rng = np.random.default_rng(5)
        cls = rng.normal(size=(16, 32)).astype(np.float32)
        self.assertEqual(global_restrav_features(cls).shape, (GLOBAL_FEATURE_DIM,))
        patches = rng.normal(size=(4, 12, 3, 5)).astype(np.float32)
        valid = np.ones((4, 3, 5), dtype=bool)
        local = local_trajectory_features(patches, valid)
        self.assertEqual(local.shape, (3, 5, LOCAL_TRAJECTORY_DIM))
        geometry = canvas_geometry(42, 70, long_side=70, multiple=14)
        aligned, aligned_valid = align_patch_sequence(
            patches,
            [np.eye(3) for _ in range(4)],
            geometry=geometry,
            patch_size=14,
            device=torch.device("cpu"),
        )
        np.testing.assert_allclose(aligned, patches, atol=1e-5)
        self.assertTrue(aligned_valid.all())

@unittest.skipIf(cv2 is None, "OpenCV is not installed in the local test runtime")
class CameraMaskTests(unittest.TestCase):
    def test_mask_tube_uses_canonical_source_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = np.zeros((3, 4, 6), dtype=np.uint8)
            masks[1, 1:3, 2:5] = 1
            np.savez(root / "mask.npz", masks=masks, frame_indices=np.array([0, 1, 2]))
            manifest = {
                "source_clip": {
                    "native": {"source_fps": 10.0, "start_time_sec": 2.0},
                    "canonical": {
                        "generation_fps": 2.0,
                        "frame_mapping": [
                            {"canonical_frame": 0, "source_frame_float": 20.0},
                            {"canonical_frame": 1, "source_frame_float": 25.0},
                            {"canonical_frame": 2, "source_frame_float": 30.0},
                        ],
                    },
                }
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            tube = load_mask_tube(root / "mask.npz", root / "manifest.json")
            self.assertEqual(int(tube.sample(2.5, height=4, width=6).sum()), 6)
            self.assertEqual(int(tube.sample(5.0, height=4, width=6).sum()), 0)


class CameraMetricTests(unittest.TestCase):
    def test_auc_with_ties(self) -> None:
        self.assertAlmostEqual(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertAlmostEqual(roc_auc([0, 1], [0.5, 0.5]), 0.5)

    def test_smoke_selection_covers_source_motion_cells(self) -> None:
        rows = [
            {
                "case_id": f"case_{source}_{motion}_{index}",
                "dataset_split": "train",
                "source_name": source,
                "motion_bucket": motion,
            }
            for source in ("source_a", "source_b")
            for motion in ("no-motion", "complex-motion")
            for index in range(3)
        ]
        selected, summary = select_rows(rows, split="train", per_source_motion=1)
        self.assertEqual(len(selected), 4)
        self.assertEqual(summary["by_source"], {"source_a": 2, "source_b": 2})
        self.assertEqual(summary["by_motion_bucket"], {"complex-motion": 2, "no-motion": 2})


@unittest.skipIf(torch is None, "PyTorch is not installed in the local test runtime")
class CameraProbeTrainingTests(unittest.TestCase):
    def test_synthetic_feature_gate_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_dir = root / "features"
            feature_dir.mkdir()
            manifest_rows = []
            rng = np.random.default_rng(17)
            for index in range(20):
                case_id = f"dataA_v1_{index + 1:05d}"
                split = "train" if index < 16 else "test"
                bucket = "complex-motion" if index % 2 else "no-motion"
                manifest_rows.append(
                    {
                        "case_id": case_id,
                        "dataset_split": split,
                        "motion_bucket": bucket,
                        "source_name": "synthetic",
                        "vace_model": "vace13b",
                    }
                )
                global_real = rng.normal(-0.5, 0.1, size=(2, 21)).astype(np.float32)
                global_fake = rng.normal(0.5, 0.1, size=(2, 21)).astype(np.float32)
                arrays = {
                    "real_global": global_real,
                    "fake_global": global_fake,
                }
                for variant, positive_shift in (("local_unaligned", 0.4), ("local_aligned", 2.0)):
                    real = rng.normal(-0.5, 0.2, size=(2, 12, 13)).astype(np.float32)
                    fake = rng.normal(-0.5, 0.2, size=(2, 12, 13)).astype(np.float32)
                    fake[:, :3, 0] += positive_shift
                    suffix = "aligned" if variant.endswith("aligned") and not variant.endswith("unaligned") else "unaligned"
                    labels = np.zeros((2, 12), dtype=np.uint8)
                    labels[:, :3] = 1
                    arrays[f"real_{variant}"] = real
                    arrays[f"fake_{variant}"] = fake
                    arrays[f"real_valid_{suffix}"] = np.ones((2, 12), dtype=bool)
                    arrays[f"fake_valid_{suffix}"] = np.ones((2, 12), dtype=bool)
                    arrays[f"real_label_{suffix}"] = np.zeros((2, 12), dtype=np.uint8)
                    arrays[f"fake_label_{suffix}"] = labels
                np.savez_compressed(feature_dir / f"{case_id}.npz", **arrays)
            manifest = root / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                result = train_probe_main(
                    [
                        "--manifest-jsonl",
                        str(manifest),
                        "--feature-dir",
                        str(feature_dir),
                        "--output-dir",
                        str(root / "out"),
                        "--device",
                        "cpu",
                        "--epochs",
                        "2",
                        "--batch-size",
                        "128",
                        "--bootstrap-iterations",
                        "10",
                        "--min-feature-coverage",
                        "1.0",
                    ]
                )
            self.assertEqual(result, 0)
            summary = json.loads((root / "out" / "camera_aligned_local_probe_summary.json").read_text())
            self.assertEqual(summary["num_test_cases"], 4)
            self.assertIn(summary["status"], {"passed", "failed", "inconclusive"})


if __name__ == "__main__":
    unittest.main()
