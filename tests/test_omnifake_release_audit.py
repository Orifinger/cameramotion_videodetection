from __future__ import annotations

import unittest

from scripts.omnifake_release_audit.audit_release import (
    field_names_matching,
    normalize_label,
    normalized_name,
    overlap_summary,
)


class OmniFakeReleaseAuditTest(unittest.TestCase):
    def test_label_normalization(self) -> None:
        self.assertEqual(normalize_label("Fake"), "full_synthetic")
        self.assertEqual(normalize_label("fully-synthetic"), "full_synthetic")
        self.assertEqual(normalize_label("partially manipulated"), "tampered")

    def test_field_detection_is_case_insensitive(self) -> None:
        columns = ["Video", "Label", "source_id", "Mask_Path"]
        self.assertEqual(field_names_matching(columns, {"source_id"}), ["source_id"])
        self.assertEqual(field_names_matching(columns, {"mask_path"}), ["Mask_Path"])

    def test_overlap_keeps_exact_and_normalized_separate(self) -> None:
        result = overlap_summary(
            ["train/A sample.mp4", "train/unique.mp4"],
            ["test/a_sample.mp4", "test/other.mp4"],
        )
        self.assertEqual(result["exact_basename_overlap_count"], 0)
        self.assertEqual(result["normalized_stem_overlap_count"], 1)
        self.assertEqual(result["status"], "passed")

    def test_exact_overlap_fails(self) -> None:
        result = overlap_summary(["set/same.mp4"], ["ood/same.mp4"])
        self.assertEqual(result["exact_basename_overlap_count"], 1)
        self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main()
