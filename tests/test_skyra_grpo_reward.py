from __future__ import annotations

import math
import unittest

from scripts.skyra_grpo_diagnostics.reward import compute_score


FAKE_ONE = (
    "<think>Artifact <type>Contact Region Artifact</type> in "
    "<t>[0.5, 1.0]</t> at <bbox>[10, 20, 200, 300]</bbox>.</think>"
    "<answer>Fake</answer>"
)
REAL_ONE = (
    "<think>Checked <t>[0.5, 1.0]</t> at <bbox>[10, 20, 200, 300]</bbox>.</think>"
    "<answer>Real</answer>"
)


def score(text: str, truth: str, variant: str) -> dict[str, float | str]:
    return compute_score(
        data_source="datab_skyra_grpo",
        solution_str=text,
        ground_truth=truth,
        extra_info={"duration_seconds": 2.0},
        reward_variant=variant,
    )


class SkyraRewardTests(unittest.TestCase):
    def test_sample_id_is_forwarded_for_rollout_grouping(self):
        result = compute_score(
            "datab",
            REAL_ONE,
            "Real",
            extra_info={"sample_id": "datab_example"},
        )
        self.assertEqual(result["diagnostic_sample_id"], "datab_example")

    def test_paper_reward_correct_fake(self) -> None:
        result = score(FAKE_ONE, "Fake", "paper_asymmetric_inspection")
        self.assertAlmostEqual(result["score"], 0.8 + 0.2 * math.log(2))
        self.assertEqual(result["correct"], 1.0)
        self.assertEqual(result["strict_check_count"], 1.0)

    def test_paper_reward_can_make_false_positive_net_positive(self) -> None:
        block = (
            "<type>Contact Region Artifact</type> in "
            "<t>[0.5, 1.0]</t> at <bbox>[10, 20, 200, 300]</bbox>"
        )
        text = f"<think>{block} {block} {block}</think><answer>Fake</answer>"
        result = score(text, "Real", "paper_asymmetric_inspection")
        self.assertGreater(result["score"], 0.0)
        self.assertEqual(result["false_positive"], 1.0)
        self.assertEqual(result["wrong_positive_reward"], 1.0)

    def test_symmetric_zero_strengthens_false_positive_incentive(self) -> None:
        paper = score(FAKE_ONE, "Real", "paper_asymmetric_inspection")
        symmetric = score(FAKE_ONE, "Real", "symmetric_zero_inspection")
        self.assertGreater(symmetric["score"], paper["score"])

    def test_strict_reward_rejects_zero_and_duplicate_boxes(self) -> None:
        block = (
            "<type>Contact Region Artifact</type> in "
            "<t>[0.5, 1.0]</t> at <bbox>[0, 0, 0, 0]</bbox>"
        )
        text = f"<think>{block} {block}</think><answer>Fake</answer>"
        result = score(text, "Fake", "strict_unique_inspection")
        self.assertEqual(result["raw_check_count"], 2.0)
        self.assertEqual(result["strict_check_count"], 0.0)
        self.assertEqual(result["inspection_reward"], 0.0)
        self.assertAlmostEqual(result["score"], 0.8)

    def test_strict_reward_deduplicates_valid_blocks(self) -> None:
        block = (
            "<type>Contact Region Artifact</type> in "
            "<t>[0.5, 1.0]</t> at <bbox>[10, 20, 200, 300]</bbox>"
        )
        result = score(f"<think>{block} {block}</think><answer>Fake</answer>", "Fake", "strict_unique_inspection")
        self.assertEqual(result["raw_check_count"], 2.0)
        self.assertEqual(result["strict_check_count"], 1.0)
        self.assertEqual(result["duplicate_check_count"], 1.0)

    def test_official_repository_bug_gives_correct_real_zero(self) -> None:
        result = score(REAL_ONE, "Real", "official_repository_bug")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["correct"], 1.0)

    def test_outer_format_requires_complete_contract(self) -> None:
        valid = score(FAKE_ONE, "Fake", "asymmetric_outer_format")
        invalid = score("prefix " + FAKE_ONE, "Fake", "asymmetric_outer_format")
        self.assertEqual(valid["format_valid"], 1.0)
        self.assertEqual(invalid["format_valid"], 0.0)


if __name__ == "__main__":
    unittest.main()
