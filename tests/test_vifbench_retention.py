import json
import tempfile
import unittest
from pathlib import Path

from scripts.camera_detection_retention.vifbench_retention import (
    build_retention_summary,
    evaluate_predictions,
    load_index,
    video_id_from_frame_dir,
)


def prediction(video_id: str, source: str, answer: str) -> dict:
    return {
        "video_id": video_id,
        "aigc_model_name": source,
        "gt": "Real" if source == "real" else "Fake",
        "answer": answer,
        "response": f"<answer>{answer}</answer>",
    }


class VifBenchIndexTest(unittest.TestCase):
    def test_loads_shards_and_skips_full_videos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_root = root / "test_normalized"
            real = frame_root / "real" / "case-1"
            fake = frame_root / "fake-model" / "case-1"
            real.mkdir(parents=True)
            fake.mkdir(parents=True)
            (root / "test_index.rank0.json").write_text(
                json.dumps({"real": [str(real)], "fake-model": [str(fake), "/x/full-videos/y"]}),
                encoding="utf-8",
            )
            result = load_index(root, expected_ranks=1, check_frame_dirs=True)
        self.assertEqual(result["num_expected_videos"], 2)
        self.assertEqual(result["num_skipped_full_videos"], 1)
        self.assertFalse(result["missing_frame_dirs"])
        self.assertEqual(video_id_from_frame_dir(str(fake)), "fake-model/case-1")


class VifBenchRetentionEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = {
            "real/case-1": {"aigc_model_name": "real", "frame_dir": "/tmp/real/case-1"},
            "model-a/case-1": {
                "aigc_model_name": "model-a",
                "frame_dir": "/tmp/model-a/case-1",
            },
            "real/case-2": {"aigc_model_name": "real", "frame_dir": "/tmp/real/case-2"},
            "model-a/case-2": {
                "aigc_model_name": "model-a",
                "frame_dir": "/tmp/model-a/case-2",
            },
        }

    def test_matches_official_paired_metrics_and_reports_format(self) -> None:
        rows = [
            prediction("real/case-1", "real", "Real"),
            prediction("model-a/case-1", "model-a", "Fake"),
            prediction("real/case-2", "real", "Fake"),
            prediction("model-a/case-2", "model-a", "Error"),
        ]
        result = evaluate_predictions(rows, self.expected)
        metrics = result["per_fake_model"]["model-a"]
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["format_valid_rate"], 0.75)
        self.assertEqual(metrics["balanced_accuracy"], 0.75)
        self.assertEqual(metrics["fake_recall"], 1.0)
        self.assertEqual(metrics["invalid_pairs"], 1)
        self.assertEqual(metrics["strict_valid_pair_metrics"]["num_pairs"], 1)

    def test_retention_gate_uses_average_accuracy_and_f1(self) -> None:
        base_rows = [
            prediction("real/case-1", "real", "Real"),
            prediction("model-a/case-1", "model-a", "Fake"),
            prediction("real/case-2", "real", "Real"),
            prediction("model-a/case-2", "model-a", "Fake"),
        ]
        camera_rows = list(base_rows)
        base_eval = evaluate_predictions(base_rows, self.expected)
        camera_eval = evaluate_predictions(camera_rows, self.expected)
        summary = build_retention_summary(base_eval, camera_eval)
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["camera_minus_base"]["balanced_accuracy"], 0.0)
        self.assertTrue(summary["checks"]["same_fake_model_set"])


if __name__ == "__main__":
    unittest.main()
