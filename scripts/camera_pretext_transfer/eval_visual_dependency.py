#!/usr/bin/env python3
"""Compare one camera model on matched versus deliberately mismatched video frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.caspr_gate1.runtime import write_json


def metrics(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return dict(json.load(handle)["metrics"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-summary", required=True)
    parser.add_argument("--shuffled-frame-summary", required=True)
    parser.add_argument("--frame-control-summary", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-bucket-balanced-accuracy", type=float, default=0.45)
    parser.add_argument("--min-predicted-buckets", type=int, default=3)
    parser.add_argument("--min-micro-f1-drop", type=float, default=0.05)
    parser.add_argument("--min-macro-f1-drop", type=float, default=0.03)
    parser.add_argument("--min-bucket-balanced-drop", type=float, default=0.10)
    return parser.parse_args()


def compact(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "format_valid_rate", "exact_set_accuracy", "micro_f1", "macro_f1_supported_labels",
        "coarse_motion_bucket_accuracy", "coarse_motion_bucket_balanced_accuracy",
        "num_predicted_motion_buckets", "gold_motion_bucket_counts",
        "predicted_motion_bucket_counts", "motion_bucket_confusion",
    )
    return {key: value[key] for key in keys}


def main() -> None:
    args = parse_args()
    matched = metrics(args.matched_summary)
    shuffled = metrics(args.shuffled_frame_summary)
    with Path(args.frame_control_summary).open("r", encoding="utf-8") as handle:
        control = json.load(handle)
    deltas = {
        "micro_f1_drop_after_frame_shuffle": matched["micro_f1"] - shuffled["micro_f1"],
        "macro_f1_drop_after_frame_shuffle": matched["macro_f1_supported_labels"]
        - shuffled["macro_f1_supported_labels"],
        "bucket_balanced_accuracy_drop_after_frame_shuffle": matched[
            "coarse_motion_bucket_balanced_accuracy"
        ] - shuffled["coarse_motion_bucket_balanced_accuracy"],
    }
    checks = {
        "matched_bucket_balanced_accuracy": matched["coarse_motion_bucket_balanced_accuracy"]
        >= args.min_bucket_balanced_accuracy,
        "matched_predicts_all_buckets": matched["num_predicted_motion_buckets"]
        >= args.min_predicted_buckets,
        "micro_f1_depends_on_frames": deltas["micro_f1_drop_after_frame_shuffle"]
        >= args.min_micro_f1_drop,
        "macro_f1_depends_on_frames": deltas["macro_f1_drop_after_frame_shuffle"]
        >= args.min_macro_f1_drop,
        "bucket_prediction_depends_on_frames": deltas[
            "bucket_balanced_accuracy_drop_after_frame_shuffle"
        ] >= args.min_bucket_balanced_drop,
    }
    passed = all(checks.values())
    summary = {
        "gate": "Stage 1B - camera prediction visual-dependence audit",
        "status": "passed" if passed else "failed",
        "what_was_tested": (
            "The same trained model and gold records are evaluated once with matched frames and once "
            "with a one-to-one frame permutation that maximizes motion-bucket mismatch."
        ),
        "thresholds": {
            "min_bucket_balanced_accuracy": args.min_bucket_balanced_accuracy,
            "min_predicted_buckets": args.min_predicted_buckets,
            "min_micro_f1_drop": args.min_micro_f1_drop,
            "min_macro_f1_drop": args.min_macro_f1_drop,
            "min_bucket_balanced_drop": args.min_bucket_balanced_drop,
        },
        "checks": checks,
        "frame_control": control,
        "matched_frames": compact(matched),
        "shuffled_frames": compact(shuffled),
        "matched_minus_shuffled": deltas,
        "next_action": (
            "Stage 1 is visually grounded; run the paraphrased prompt diagnostic before Stage 2."
            if passed else "Do not start Stage 2; the camera-label score is not sufficiently video-dependent."
        ),
    }
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
