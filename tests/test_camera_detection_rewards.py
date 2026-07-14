from __future__ import annotations

import unittest

from rl.camera_detection_rewards import (
    CameraBinaryFormatReward,
    CameraBinaryReward,
    CameraExactReward,
    CameraFormatReward,
    CameraSetF1Reward,
    DetectionBinaryReward,
    DetectionFormatReward,
    JointDetectionFormatReward,
    camera_binary_correct,
    camera_binary_format_valid,
    camera_exact_match,
    camera_format_valid,
    camera_set_f1,
    detection_binary_correct,
    detection_format_valid,
    joint_detection_format_valid,
    parse_camera_binary_answer,
    parse_camera_completion,
)


class CameraRewardTests(unittest.TestCase):
    def test_binary_camera_answer_contract(self) -> None:
        self.assertEqual(parse_camera_binary_answer("Yes"), "Yes")
        self.assertEqual(parse_camera_binary_answer("<answer>No</answer>"), "No")
        self.assertEqual(
            parse_camera_binary_answer("<think>\n\n</think>\n\nYes"),
            "Yes",
        )
        self.assertIsNone(parse_camera_binary_answer("Yes, because the camera pans."))
        self.assertIsNone(parse_camera_binary_answer("<think>reasoning</think>Yes"))
        self.assertIsNone(parse_camera_binary_answer("<answer>Yes</answer> extra"))

    def test_binary_camera_reward_batch(self) -> None:
        completions = ["<think></think>Yes", "<answer>No</answer>"]
        solutions = ["Yes", "Yes"]
        self.assertEqual(camera_binary_correct(completions[0], "Yes"), 1.0)
        self.assertEqual(camera_binary_format_valid(completions[1]), 1.0)
        self.assertEqual(
            CameraBinaryReward()(completions, solution=solutions),
            [1.0, 0.0],
        )
        self.assertEqual(CameraBinaryFormatReward()(completions), [1.0, 1.0])

    def test_exact_camera_match(self) -> None:
        pred = '<camera_motion>["no-shaking", "no-motion", "regular-speed"]</camera_motion>'
        truth = ["no-shaking", "no-motion", "regular-speed"]
        self.assertEqual(camera_set_f1(pred, truth), 1.0)
        self.assertEqual(camera_exact_match(pred, truth), 1.0)
        self.assertEqual(camera_format_valid(pred), 1.0)

    def test_order_does_not_change_semantic_reward(self) -> None:
        pred = '<camera_motion>["regular-speed", "no-motion", "no-shaking"]</camera_motion>'
        truth = ["no-shaking", "no-motion", "regular-speed"]
        self.assertEqual(camera_set_f1(pred, truth), 1.0)
        self.assertEqual(camera_exact_match(pred, truth), 1.0)

    def test_missing_label_has_fractional_f1(self) -> None:
        pred = '<camera_motion>["no-shaking", "no-motion"]</camera_motion>'
        truth = ["no-shaking", "no-motion", "regular-speed"]
        self.assertAlmostEqual(camera_set_f1(pred, truth), 0.8)
        self.assertEqual(camera_exact_match(pred, truth), 0.0)

    def test_unknown_label_is_false_positive(self) -> None:
        pred = '<camera_motion>["no-shaking", "invented-motion"]</camera_motion>'
        truth = ["no-shaking"]
        self.assertAlmostEqual(camera_set_f1(pred, truth), 2 / 3)
        self.assertEqual(camera_format_valid(pred), 0.0)

    def test_duplicate_label_invalidates_format_and_exact(self) -> None:
        pred = '<camera_motion>["pan-left", "pan-left"]</camera_motion>'
        parsed = parse_camera_completion(pred)
        self.assertTrue(parsed.duplicate)
        self.assertEqual(camera_format_valid(pred), 0.0)
        self.assertEqual(camera_exact_match(pred, ["pan-left"]), 0.0)

    def test_malformed_camera_output(self) -> None:
        pred = "<camera_motion>pan-left</camera_motion>"
        self.assertEqual(camera_set_f1(pred, ["pan-left"]), 0.0)
        self.assertEqual(camera_format_valid(pred), 0.0)

    def test_static_truth_is_excluded(self) -> None:
        pred = '<camera_motion>["no-motion"]</camera_motion>'
        self.assertEqual(camera_exact_match(pred, ["static", "no-motion"]), 1.0)

    def test_camera_orm_batch(self) -> None:
        completions = [
            '<camera_motion>["pan-left"]</camera_motion>',
            '<camera_motion>["pan-right"]</camera_motion>',
        ]
        labels = [["pan-left"], ["pan-left"]]
        self.assertEqual(CameraSetF1Reward()(completions, camera_labels=labels), [1.0, 0.0])
        self.assertEqual(CameraExactReward()(completions, camera_labels=labels), [1.0, 0.0])
        self.assertEqual(CameraFormatReward()(completions), [1.0, 1.0])


class DetectionRewardTests(unittest.TestCase):
    def test_detection_correctness(self) -> None:
        fake = "<think>There is temporal deformation.</think><answer>Fake</answer>"
        real = "<think>The sequence is coherent.</think><answer>Real</answer>"
        self.assertEqual(detection_binary_correct(fake, "Fake"), 1.0)
        self.assertEqual(detection_binary_correct(real, 0), 1.0)
        self.assertEqual(detection_binary_correct(fake, "Real"), 0.0)

    def test_detection_format(self) -> None:
        valid = "<think>Evidence.</think><answer>Fake</answer>"
        no_think = "<answer>Fake</answer>"
        invalid_answer = "<think>Evidence.</think><answer>Unknown</answer>"
        self.assertEqual(detection_format_valid(valid), 1.0)
        self.assertEqual(detection_format_valid(no_think), 0.0)
        self.assertEqual(detection_format_valid(invalid_answer), 0.0)

    def test_detection_orm_batch(self) -> None:
        completions = [
            "<think>a</think><answer>Fake</answer>",
            "<think>b</think><answer>Real</answer>",
        ]
        labels = ["Fake", "Fake"]
        self.assertEqual(DetectionBinaryReward()(completions, label=labels), [1.0, 0.0])
        self.assertEqual(DetectionFormatReward()(completions), [1.0, 1.0])

    def test_joint_camera_then_detection_contract(self) -> None:
        valid = (
            '<camera_motion>["no-shaking","no-motion"]</camera_motion>\n'
            '<answer>Fake</answer>'
        )
        wrapped = "<think>\n</think>\n" + valid
        extra = valid + " explanation"
        unknown = (
            '<camera_motion>["invented-motion"]</camera_motion>\n'
            '<answer>Fake</answer>'
        )
        self.assertEqual(joint_detection_format_valid(valid), 1.0)
        self.assertEqual(joint_detection_format_valid(wrapped), 1.0)
        self.assertEqual(joint_detection_format_valid(extra), 0.0)
        self.assertEqual(joint_detection_format_valid(unknown), 0.0)
        self.assertEqual(JointDetectionFormatReward()([valid, extra]), [1.0, 0.0])

    def test_joint_detection_reward_uses_explicit_detection_label(self) -> None:
        completions = [
            '<camera_motion>["pan-left"]</camera_motion><answer>Fake</answer>',
            '<camera_motion>["pan-left"]</camera_motion><answer>Real</answer>',
        ]
        self.assertEqual(
            DetectionBinaryReward()(completions, detection_label=["Fake", "Fake"]),
            [1.0, 0.0],
        )

    def test_camera_reward_prefers_control_truth_field(self) -> None:
        completion = '<camera_motion>["pan-right"]</camera_motion><answer>Fake</answer>'
        reward = CameraSetF1Reward()(
            [completion],
            camera_labels=[["pan-left"]],
            camera_labels_reward=[["pan-right"]],
        )
        self.assertEqual(reward, [1.0])


if __name__ == "__main__":
    unittest.main()
