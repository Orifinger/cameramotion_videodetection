#!/usr/bin/env python3
"""Summarize the correct-versus-flipped binary-camera supervision gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.caspr_gate1.runtime import write_json


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def condition(payload: dict[str, Any], name: str) -> dict[str, Any]:
    return dict(payload["conditions"][name])


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


def build_summary(
    correct_payload: dict[str, Any],
    flipped_payload: dict[str, Any],
    min_supported_labels: int,
    min_macro_ap_delta: float,
    min_balanced_delta: float,
    min_opposite_drop: float,
    min_no_frame_drop: float,
) -> dict[str, Any]:
    correct = condition(correct_payload, "matched_frames")
    flipped = condition(flipped_payload, "matched_frames")
    opposite = condition(correct_payload, "opposite_frames")
    no_frames = condition(correct_payload, "no_frames")
    deltas = {
        "correct_minus_flipped_macro_average_precision": (
            correct["macro"]["average_precision"] - flipped["macro"]["average_precision"]
        ),
        "correct_minus_flipped_balanced_accuracy": (
            correct["overall"]["balanced_accuracy"]
            - flipped["overall"]["balanced_accuracy"]
        ),
        "matched_minus_opposite_balanced_accuracy": (
            correct["overall"]["balanced_accuracy"]
            - opposite["overall"]["balanced_accuracy"]
        ),
        "matched_minus_no_frames_balanced_accuracy": (
            correct["overall"]["balanced_accuracy"]
            - no_frames["overall"]["balanced_accuracy"]
        ),
    }
    checks = {
        "held_out_coverage": (
            correct["coverage"] >= 0.99 and flipped["coverage"] >= 0.99
        ),
        "enough_supported_camera_primitives": (
            correct["num_supported_labels"] >= min_supported_labels
            and flipped["num_supported_labels"] >= min_supported_labels
        ),
        "correct_supervision_beats_flipped_targets": (
            deltas["correct_minus_flipped_macro_average_precision"] >= min_macro_ap_delta
            or deltas["correct_minus_flipped_balanced_accuracy"] >= min_balanced_delta
        ),
        "correct_model_depends_on_visual_frames": (
            deltas["matched_minus_opposite_balanced_accuracy"] >= min_opposite_drop
            or deltas["matched_minus_no_frames_balanced_accuracy"] >= min_no_frame_drop
        ),
    }
    passed = all(checks.values())
    return {
        "gate": "held-out correct-versus-flipped binary-camera supervision gate",
        "status": "passed" if passed else "failed",
        "what_was_tested": (
            "The correct and target-flipped LoRA branches were scored on the same held-out "
            "binary camera questions. The correct branch was also tested with opposite-answer "
            "frames and with no frames."
        ),
        "thresholds": {
            "min_supported_labels": min_supported_labels,
            "min_correct_minus_flipped_macro_ap": min_macro_ap_delta,
            "min_correct_minus_flipped_balanced_accuracy": min_balanced_delta,
            "min_matched_minus_opposite_balanced_accuracy": min_opposite_drop,
            "min_matched_minus_no_frames_balanced_accuracy": min_no_frame_drop,
        },
        "checks": checks,
        "camera_eval": {
            "correct_matched_frames": compact(correct),
            "flipped_matched_frames": compact(flipped),
            "correct_opposite_frames": compact(opposite),
            "correct_no_frames": compact(no_frames),
        },
        "deltas": deltas,
        "does_not_establish": (
            "This gate does not establish AIGC detection gains or detection-format retention."
        ),
        "next_action": (
            "Run the sampled RL-readiness audit, then train the detection-only control."
            if passed
            else "Stop before the detection-only control and inspect per-label camera metrics."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correct-camera-eval", required=True)
    parser.add_argument("--flipped-camera-eval", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-supported-labels", type=int, default=20)
    parser.add_argument("--min-correct-minus-flipped-macro-ap", type=float, default=0.03)
    parser.add_argument("--min-correct-minus-flipped-balanced", type=float, default=0.05)
    parser.add_argument("--min-opposite-frame-drop", type=float, default=0.10)
    parser.add_argument("--min-no-frame-drop", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        read_json(args.correct_camera_eval),
        read_json(args.flipped_camera_eval),
        args.min_supported_labels,
        args.min_correct_minus_flipped_macro_ap,
        args.min_correct_minus_flipped_balanced,
        args.min_opposite_frame_drop,
        args.min_no_frame_drop,
    )
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
