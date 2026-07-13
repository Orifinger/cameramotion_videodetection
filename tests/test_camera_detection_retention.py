import unittest

from scripts.camera_detection_retention.build_data import build, split_case_ids
from scripts.camera_detection_retention.summarize import compact


def record(case: str, side: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "detect"},
            {"role": "user", "content": "<image>"},
            {"role": "assistant", "content": f"<answer>{side.title()}</answer>"},
        ],
        "images": [f"/tmp/frames_40step_v3/{case}/{side}/000.png"],
    }


class BuildRetentionDataTest(unittest.TestCase):
    def test_builds_complete_fixed_pairs_without_camera_text(self) -> None:
        development = [record("dataA_v1_00001", "real")]
        detection = [
            record("dataA_v1_00001", "real"),
            record("dataA_v1_00001", "fake"),
            record("dataA_v1_00002", "real"),
            record("dataA_v1_00002", "fake"),
        ]
        output, summary = build(
            detection,
            split_case_ids(development),
            check_images=False,
        )
        self.assertEqual(len(output), 2)
        self.assertEqual(summary["num_complete_pairs"], 1)
        self.assertEqual(summary["side_counts"], {"fake": 1, "real": 1})
        self.assertEqual(summary["camera_context_records"], 0)

    def test_rejects_incomplete_pair(self) -> None:
        development = [record("dataA_v1_00001", "real")]
        with self.assertRaisesRegex(ValueError, "incomplete real/fake"):
            build(
                [record("dataA_v1_00001", "real")],
                split_case_ids(development),
                check_images=False,
            )


class RetentionSummaryTest(unittest.TestCase):
    def test_compacts_external_dataa_eval_schema(self) -> None:
        payload = {
            "num_gt_records": 2,
            "num_matched_records": 2,
            "basic": {
                "format_valid_rate": 1.0,
                "accuracy": 0.75,
                "balanced_accuracy": 0.7,
                "fake_recall": 0.8,
                "real_recall": 0.6,
                "fake_f1": 0.72,
            },
            "pair": {"pair_accuracy": 0.5, "num_pairs": 1},
            "iou": {
                "pred_evidence_sample_rate": 0.8,
                "mean_best_temporal_iou": 0.4,
                "mean_best_bbox_iou": 0.3,
                "evidence_hit_t03_b03": 0.2,
                "sample_any_evidence_hit_t03_b03": 0.25,
            },
        }
        result = compact(payload)
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["pair_accuracy"], 0.5)
        self.assertEqual(result["evidence"]["mean_best_bbox_iou"], 0.3)


if __name__ == "__main__":
    unittest.main()
