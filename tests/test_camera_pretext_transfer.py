from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from scripts.camera_pretext_transfer.runtime import (
    find_last_subsequence,
    multilabel_metrics,
    parse_camera_response,
    target_supervision_span,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_camera_pretext_transfer_gate", ROOT / "tools" / "build_camera_pretext_transfer_gate.py"
)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class CameraPretextDataTests(unittest.TestCase):
    def test_canonical_prompt_is_stable_and_image_aligned(self) -> None:
        first = BUILDER.camera_prompt(3, "canonical")
        second = BUILDER.camera_prompt(3, "canonical")
        self.assertEqual(first, second)
        self.assertEqual(first.count("<image>"), 3)
        self.assertNotEqual(first, BUILDER.camera_prompt(3, "paraphrased"))

    def test_semantic_permutation_is_valid_wrong_and_same_length(self) -> None:
        original = ["no-shaking", "complex-motion", "regular-speed", "pan-left"]
        shuffled = BUILDER.permuted_labels(original)
        self.assertEqual(shuffled, ["very-unsteady", "minor-motion", "slow-speed", "pan-right"])
        self.assertNotEqual(original, shuffled)
        self.assertEqual(len(original), len(shuffled))
        self.assertTrue(set(shuffled) <= set(BUILDER.CAMERA_LABEL_ORDER))

    def test_static_is_removed_but_no_motion_is_kept(self) -> None:
        labels, unknown = BUILDER.canonical_labels(["static", "no_motion", "roll-cw"])
        self.assertEqual(labels, ["no-motion", "roll-CW"])
        self.assertEqual(unknown, [])


class CameraPretextRuntimeTests(unittest.TestCase):
    def test_find_last_subsequence(self) -> None:
        self.assertEqual(find_last_subsequence([1, 2, 3, 2, 3], [2, 3]), 3)
        self.assertEqual(find_last_subsequence([1, 2], [3]), -1)

    def test_target_supervision_includes_assistant_terminator(self) -> None:
        start, end = target_supervision_span(
            [0, 0, 10, 20, 21, 99], [0, 0, 1, 1, 1, 1], [20, 21]
        )
        self.assertEqual((start, end), (3, 6))

    def test_response_requires_exact_tagged_json(self) -> None:
        allowed = ["no-motion", "complex-motion"]
        valid = parse_camera_response('<camera_motion>["no-motion"]</camera_motion>', allowed)
        extra = parse_camera_response('answer: <camera_motion>["no-motion"]</camera_motion>', allowed)
        self.assertTrue(valid["format_valid"])
        self.assertFalse(extra["format_valid"])

    def test_multilabel_metrics(self) -> None:
        gold = [
            {"case_id": "a", "camera_labels": ["no-motion"]},
            {"case_id": "b", "camera_labels": ["complex-motion"]},
        ]
        predictions = [
            {"case_id": "a", "response": '<camera_motion>["no-motion"]</camera_motion>'},
            {"case_id": "b", "response": '<camera_motion>["complex-motion"]</camera_motion>'},
        ]
        metrics = multilabel_metrics(gold, predictions, ["no-motion", "complex-motion"])
        self.assertEqual(metrics["micro_f1"], 1.0)
        self.assertEqual(metrics["macro_f1_supported_labels"], 1.0)
        self.assertEqual(metrics["coarse_motion_bucket_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
