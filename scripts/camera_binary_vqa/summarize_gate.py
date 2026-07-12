#!/usr/bin/env python3
"""Summarize the binary camera VQA learning and visual-dependence gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.camera_binary_vqa.runtime import write_json


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def compact(condition: dict[str, Any]) -> dict[str, Any]:
    return {
        "coverage": condition["coverage"],
        "num_supported_labels": condition["num_supported_labels"],
        "balanced_accuracy": condition["overall"]["balanced_accuracy"],
        "average_precision": condition["overall"]["average_precision"],
        "roc_auc": condition["overall"]["roc_auc"],
        "macro_balanced_accuracy": condition["macro"]["balanced_accuracy"],
        "macro_average_precision": condition["macro"]["average_precision"],
        "macro_roc_auc": condition["macro"]["roc_auc"],
        "paired_question_accuracy": condition["paired_question_accuracy"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-eval", type=Path, required=True)
    parser.add_argument("--epoch1-eval", type=Path, required=True)
    parser.add_argument("--final-eval", type=Path, required=True)
    parser.add_argument("--training-state", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-labels", type=int, default=20)
    parser.add_argument("--min-final-macro-ap", type=float, default=0.65)
    parser.add_argument("--min-final-balanced-accuracy", type=float, default=0.60)
    parser.add_argument("--min-paired-question-accuracy", type=float, default=0.35)
    parser.add_argument("--min-macro-ap-over-base", type=float, default=0.08)
    parser.add_argument("--min-balanced-over-no-video", type=float, default=0.08)
    parser.add_argument("--min-balanced-over-opposite-video", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_raw = read_json(args.base_eval)
    epoch1_raw = read_json(args.epoch1_eval)
    final_raw = read_json(args.final_eval)
    training = read_json(args.training_state)
    base = compact(base_raw["conditions"]["matched_video"])
    epoch1 = compact(epoch1_raw["conditions"]["matched_video"])
    final = compact(final_raw["conditions"]["matched_video"])
    opposite = compact(final_raw["conditions"]["opposite_label_video"])
    no_video = compact(final_raw["conditions"]["no_video"])
    deltas = {
        "final_macro_ap_minus_base": final["macro_average_precision"]
        - base["macro_average_precision"],
        "final_balanced_accuracy_minus_no_video": final["balanced_accuracy"]
        - no_video["balanced_accuracy"],
        "final_balanced_accuracy_minus_opposite_video": final["balanced_accuracy"]
        - opposite["balanced_accuracy"],
    }
    checks = {
        "complete_coverage": min(
            base["coverage"], epoch1["coverage"], final["coverage"], opposite["coverage"], no_video["coverage"]
        )
        >= 0.99,
        "enough_supported_labels": final["num_supported_labels"] >= args.min_labels,
        "trained_at_least_one_epoch": float(training.get("effective_epochs", 0.0)) >= 1.0,
        "final_macro_ap": final["macro_average_precision"] >= args.min_final_macro_ap,
        "final_balanced_accuracy": final["balanced_accuracy"]
        >= args.min_final_balanced_accuracy,
        "paired_question_accuracy": final["paired_question_accuracy"]
        >= args.min_paired_question_accuracy,
        "learned_over_base": deltas["final_macro_ap_minus_base"]
        >= args.min_macro_ap_over_base,
        "uses_video_over_no_video": deltas["final_balanced_accuracy_minus_no_video"]
        >= args.min_balanced_over_no_video,
        "responds_to_opposite_video": deltas["final_balanced_accuracy_minus_opposite_video"]
        >= args.min_balanced_over_opposite_video,
    }
    output = {
        "gate": "DataA balanced binary camera VQA learning and visual dependence",
        "status": "passed" if all(checks.values()) else "failed",
        "what_was_tested": (
            "Held-out DataA identities are evaluated with balanced Yes/No questions for each supported "
            "camera primitive. The final model must beat its untrained start and must depend on the video, "
            "as tested by no-video and opposite-label-video controls."
        ),
        "thresholds": {
            "min_labels": args.min_labels,
            "min_final_macro_ap": args.min_final_macro_ap,
            "min_final_balanced_accuracy": args.min_final_balanced_accuracy,
            "min_paired_question_accuracy": args.min_paired_question_accuracy,
            "min_macro_ap_over_base": args.min_macro_ap_over_base,
            "min_balanced_over_no_video": args.min_balanced_over_no_video,
            "min_balanced_over_opposite_video": args.min_balanced_over_opposite_video,
        },
        "checks": checks,
        "training": training,
        "base_matched_video": base,
        "epoch1_matched_video": epoch1,
        "final_matched_video": final,
        "final_opposite_label_video": opposite,
        "final_no_video": no_video,
        "deltas": deltas,
        "next_action": (
            "If passed, compare the generic and detection-checkpoint starts, then design a joint detection "
            "plus camera auxiliary gate without camera text at inference. If failed, do not run that joint gate."
        ),
    }
    write_json(args.output_json, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
