from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.camera_ctne_gate1.audit_features import continuous_motion_bucket
from scripts.camera_ctne_gate1.audit_overlap import audit as audit_overlap
from scripts.camera_ctne_gate1.build_manifest import main as build_manifest_main
from scripts.camera_ctne_gate1.controls import best_balanced_threshold, shuffled_donor_indices
from scripts.camera_ctne_gate1.sampling import frame_chunks, uniform_frame_indices
from scripts.camera_ctne_gate1.preprocessing import per_video_transition_weights, resample_sequence

try:
    import cv2  # noqa: F401
    from scripts.camera_ctne_gate1.transition_features import CAMERA_FEATURE_NAMES, build_transition_features, evidence_feature_names
except ImportError:
    cv2 = None


class CTNEVariableLengthContractTests(unittest.TestCase):
    def test_uniform_sampling_is_disabled_by_default(self) -> None:
        self.assertEqual(uniform_frame_indices(17, 0), list(range(17)))
        self.assertEqual(uniform_frame_indices(11, 0), list(range(11)))
        self.assertEqual(uniform_frame_indices(17, 16)[0], 0)
        self.assertEqual(uniform_frame_indices(17, 16)[-1], 16)
        self.assertEqual(len(set(uniform_frame_indices(17, 16))), 16)

    def test_chunking_covers_every_transition_once(self) -> None:
        for frames in (3, 11, 16, 17, 73):
            chunks = frame_chunks(frames, 8)
            transitions = [(index, index + 1) for start, end in chunks for index in range(start, end - 1)]
            self.assertEqual(transitions, [(index, index + 1) for index in range(frames - 1)])

    def test_datab_manifest_preserves_4_and_7_frame_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detection_rows = []
            camera_rows = []
            for sample_index, (count, label) in enumerate(((4, "Real"), (7, "Fake")), 1):
                frame_dir = root / ("real" if label == "Real" else "fake") / f"source_{sample_index}" / f"sample_{sample_index}"
                frame_dir.mkdir(parents=True)
                images = []
                for frame_index in range(count):
                    path = frame_dir / f"{frame_index + 1}.png"
                    path.write_bytes(b"not-decoded-in-this-test")
                    images.append(str(path))
                detection_rows.append(
                    {
                        "images": images,
                        "messages": [{"role": "assistant", "content": f"<answer>{label}</answer>"}],
                    }
                )
                camera_rows.append({"path": str(frame_dir), "labels": ["minor-motion"], "caption": "moves"})
            detection_path = root / "detection.json"
            camera_path = root / "camera.jsonl"
            manifest_path = root / "manifest.jsonl"
            summary_path = root / "summary.json"
            detection_path.write_text(json.dumps(detection_rows), encoding="utf-8")
            camera_path.write_text("".join(json.dumps(row) + "\n" for row in camera_rows), encoding="utf-8")
            code = build_manifest_main(
                [
                    "datab",
                    "--detection-json",
                    str(detection_path),
                    "--camera-jsonl",
                    str(camera_path),
                    "--val-ratio",
                    "0",
                    "--output-jsonl",
                    str(manifest_path),
                    "--summary-json",
                    str(summary_path),
                ]
            )
            self.assertEqual(code, 0)
            rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sorted(row["frame_count"] for row in rows), [4, 7])
            self.assertEqual(sorted(len(row["frame_paths"]) for row in rows), [4, 7])
            self.assertTrue(all(row["ctne_available"] for row in rows))

    def test_each_video_has_equal_total_training_weight(self) -> None:
        weights = per_video_transition_weights([2, 6, 16])
        self.assertAlmostEqual(float(weights[:2].sum()), 1.0)
        self.assertAlmostEqual(float(weights[2:8].sum()), 1.0)
        self.assertAlmostEqual(float(weights[8:].sum()), 1.0)

    def test_variable_length_camera_resampling_keeps_endpoints(self) -> None:
        source = np.asarray([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]], dtype=np.float32)
        output = resample_sequence(source, 7)
        self.assertEqual(output.shape, (7, 2))
        np.testing.assert_allclose(output[0], source[0])
        np.testing.assert_allclose(output[-1], source[-1])


@unittest.skipIf(cv2 is None, "OpenCV is unavailable")
class CTNETransitionFeatureTests(unittest.TestCase):
    def test_synthetic_features_have_fixed_finite_contract(self) -> None:
        rng = np.random.default_rng(9)
        cls = rng.normal(size=(4, 6)).astype(np.float32)
        patches = rng.normal(size=(4, 8, 2, 3)).astype(np.float32)
        forward = np.zeros((3, 48, 64, 2), dtype=np.float32)
        forward[..., 0] = 2.0
        forward[..., 1] = -1.0
        backward = -forward
        camera, evidence, last_delta, quality = build_transition_features(
            cls,
            patches,
            forward,
            backward,
            grid_step=8,
            max_fb_error=10.0,
        )
        self.assertEqual(camera.shape, (3, len(CAMERA_FEATURE_NAMES)))
        self.assertEqual(evidence.shape, (3, len(evidence_feature_names(6))))
        self.assertEqual(last_delta.shape, (6,))
        self.assertEqual(len(quality), 3)
        self.assertTrue(np.isfinite(camera).all())
        self.assertTrue(np.isfinite(evidence).all())

    def test_numeric_camera_bucket_is_not_sidecar_dependent(self) -> None:
        static = np.zeros((5, len(CAMERA_FEATURE_NAMES)), dtype=np.float32)
        complex_motion = static.copy()
        complex_motion[:, 0] = 0.03
        self.assertEqual(continuous_motion_bucket(static), "static/no-motion")
        self.assertEqual(continuous_motion_bucket(complex_motion), "complex-motion")


class CTNEControlTests(unittest.TestCase):
    def test_shuffled_camera_is_deterministic_and_never_self(self) -> None:
        rows = [
            {
                "sample_id": f"sample_{index}",
                "dataset_name": "benchmark",
                "source_name": "source",
                "motion_bucket": "minor-motion",
                "frame_count_bin": "8-15",
            }
            for index in range(6)
        ]
        first, first_counts = shuffled_donor_indices(rows, 17)
        second, second_counts = shuffled_donor_indices(rows, 17)
        self.assertEqual(first, second)
        self.assertEqual(first_counts, second_counts)
        self.assertTrue(all(index != donor for index, donor in enumerate(first)))

    def test_threshold_selection_uses_balanced_accuracy(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        scores = np.asarray([0.0, 0.1, 0.2, 0.7, 0.8, 0.9])
        threshold, metrics = best_balanced_threshold(labels, scores)
        self.assertGreater(threshold, 0.2)
        self.assertLessEqual(threshold, 0.7)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)

    def test_threshold_selection_stays_finite_for_tied_scores(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.zeros(4)
        threshold, metrics = best_balanced_threshold(labels, scores)
        self.assertTrue(np.isfinite(threshold))
        self.assertEqual(metrics["num_samples"], 4)

    def test_overlap_audit_detects_same_identity_tail(self) -> None:
        train = [
            {
                "sample_id": "train",
                "frame_dir_path": "/train/fake/model/abc",
                "label_name": "Fake",
                "generator_name": "model",
            }
        ]
        test = [
            {
                "sample_id": "test",
                "frame_dir_path": "/benchmark/fake/model/abc",
                "label_name": "Fake",
                "generator_name": "model",
            }
        ]
        self.assertEqual(audit_overlap(train, test)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
