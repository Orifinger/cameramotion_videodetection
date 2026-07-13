#!/usr/bin/env python3
"""Summarize the balanced binary-camera joint-SFT gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.caspr_gate1.runtime import write_json


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def condition(path: str | Path, name: str) -> dict[str, Any]:
    return dict(read_json(path)["conditions"][name])


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "coverage": metrics["coverage"],
        "num_supported_labels": metrics["num_supported_labels"],
        "balanced_accuracy": metrics["overall"]["balanced_accuracy"],
        "average_precision": metrics["overall"]["average_precision"],
        "roc_auc": metrics["overall"]["roc_auc"],
        "macro_balanced_accuracy": metrics["macro"]["balanced_accuracy"],
        "macro_average_precision": metrics["macro"]["average_precision"],
        "macro_roc_auc": metrics["macro"]["roc_auc"],
        "paired_question_accuracy": metrics["paired_question_accuracy"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-only-camera-eval", required=True)
    parser.add_argument("--correct-camera-eval", required=True)
    parser.add_argument("--shuffled-camera-eval", required=True)
    parser.add_argument("--correct-readiness", required=True)
    parser.add_argument("--shuffled-readiness")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-supported-labels", type=int, default=20)
    parser.add_argument("--min-correct-minus-shuffled-macro-ap", type=float, default=0.03)
    parser.add_argument("--min-correct-minus-shuffled-balanced", type=float, default=0.05)
    parser.add_argument("--min-opposite-frame-drop", type=float, default=0.10)
    parser.add_argument("--min-no-frame-drop", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detection_only = condition(args.detection_only_camera_eval, "matched_frames")
    correct = condition(args.correct_camera_eval, "matched_frames")
    shuffled = condition(args.shuffled_camera_eval, "matched_frames")
    opposite = condition(args.correct_camera_eval, "opposite_frames")
    no_frames = condition(args.correct_camera_eval, "no_frames")
    readiness = read_json(args.correct_readiness)
    shuffled_readiness = read_json(args.shuffled_readiness) if args.shuffled_readiness else None

    deltas = {
        "correct_minus_shuffled_macro_average_precision": (
            correct["macro"]["average_precision"] - shuffled["macro"]["average_precision"]
        ),
        "correct_minus_shuffled_balanced_accuracy": (
            correct["overall"]["balanced_accuracy"] - shuffled["overall"]["balanced_accuracy"]
        ),
        "matched_minus_opposite_balanced_accuracy": (
            correct["overall"]["balanced_accuracy"] - opposite["overall"]["balanced_accuracy"]
        ),
        "matched_minus_no_frames_balanced_accuracy": (
            correct["overall"]["balanced_accuracy"] - no_frames["overall"]["balanced_accuracy"]
        ),
    }
    checks = {
        "held_out_coverage": correct["coverage"] >= 0.99,
        "enough_supported_camera_primitives": (
            correct["num_supported_labels"] >= args.min_supported_labels
        ),
        "correct_supervision_beats_flipped_targets": (
            deltas["correct_minus_shuffled_macro_average_precision"]
            >= args.min_correct_minus_shuffled_macro_ap
            or deltas["correct_minus_shuffled_balanced_accuracy"]
            >= args.min_correct_minus_shuffled_balanced
        ),
        "camera_answers_depend_on_visual_frames": (
            deltas["matched_minus_opposite_balanced_accuracy"] >= args.min_opposite_frame_drop
            or deltas["matched_minus_no_frames_balanced_accuracy"] >= args.min_no_frame_drop
        ),
        "rl_exploration_available": readiness.get("status") in {"rl_ready", "borderline"},
    }
    if all(checks.values()):
        status = "passed_for_short_rl"
        next_action = "先做 DataA 与 VIF-Bench 无相机文本检测保留评测，再进入短程 GRPO。"
    elif (
        checks["camera_answers_depend_on_visual_frames"]
        and checks["rl_exploration_available"]
        and checks["enough_supported_camera_primitives"]
    ):
        status = "borderline_short_rl_only"
        next_action = "仅允许预先限定步数的短程 GRPO，并保留翻转监督对照；不能直接做完整 RL。"
    else:
        status = "not_ready_for_rl"
        next_action = "先修复相机任务、视觉依赖或采样奖励信号，再投入 RL。"

    summary = {
        "gate": "同一 16 帧上的二元相机 VQA 与检测 replay 联合 SFT 验收",
        "status": status,
        "what_was_tested": (
            "三个等样本数 LoRA 分支从同一检测 checkpoint 出发并共享检测 replay；"
            "唯一变化是辅助槽使用额外检测样本、正确二元相机标签或逐条翻转的错误标签。"
        ),
        "thresholds": {
            "min_supported_labels": args.min_supported_labels,
            "min_correct_minus_shuffled_macro_ap": args.min_correct_minus_shuffled_macro_ap,
            "min_correct_minus_shuffled_balanced": args.min_correct_minus_shuffled_balanced,
            "min_opposite_frame_drop": args.min_opposite_frame_drop,
            "min_no_frame_drop": args.min_no_frame_drop,
        },
        "checks": checks,
        "camera_eval": {
            "detection_only_matched_frames": compact(detection_only),
            "correct_camera_matched_frames": compact(correct),
            "flipped_camera_matched_frames": compact(shuffled),
            "correct_camera_opposite_frames": compact(opposite),
            "correct_camera_no_frames": compact(no_frames),
        },
        "deltas": deltas,
        "correct_rl_readiness": {
            "status": readiness.get("status"),
            "metrics": readiness.get("metrics"),
        },
        "shuffled_rl_readiness": (
            {"status": shuffled_readiness.get("status"), "metrics": shuffled_readiness.get("metrics")}
            if shuffled_readiness else None
        ),
        "does_not_establish": (
            "本验收只证明相机监督可学、依赖画面且存在可用 RL 奖励；不证明 AIGC 检测提升。"
            "检测结论必须来自留出 DataA 和外部 VIF-Bench，且推理时不提供相机文本。"
        ),
        "next_action": next_action,
    }
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
