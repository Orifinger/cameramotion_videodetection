#!/usr/bin/env python3
"""Compare correct camera SFT against the base and shuffled-label controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.caspr_gate1.runtime import write_json


def load_metrics(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(payload["metrics"])


def compact(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "num_gold", "num_matched", "coverage", "format_valid_rate", "exact_set_accuracy",
        "coarse_motion_bucket_accuracy", "coarse_motion_bucket_balanced_accuracy",
        "num_predicted_motion_buckets", "gold_motion_bucket_counts",
        "predicted_motion_bucket_counts", "micro_f1", "macro_f1_supported_labels",
    )
    return {key: metrics[key] for key in keys}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-summary", required=True)
    parser.add_argument("--correct-summary", required=True)
    parser.add_argument("--shuffled-summary", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--min-format-valid", type=float, default=0.95)
    parser.add_argument("--min-macro-f1-delta", type=float, default=0.10)
    parser.add_argument("--min-coarse-bucket-accuracy", type=float, default=0.50)
    parser.add_argument("--min-coarse-bucket-balanced-accuracy", type=float, default=0.45)
    parser.add_argument("--min-predicted-motion-buckets", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_metrics(args.base_summary)
    correct = load_metrics(args.correct_summary)
    shuffled = load_metrics(args.shuffled_summary)
    correct_macro = float(correct["macro_f1_supported_labels"])
    deltas = {
        "correct_minus_base_macro_f1": correct_macro - float(base["macro_f1_supported_labels"]),
        "correct_minus_shuffled_macro_f1": correct_macro - float(shuffled["macro_f1_supported_labels"]),
        "correct_minus_base_micro_f1": float(correct["micro_f1"]) - float(base["micro_f1"]),
        "correct_minus_shuffled_micro_f1": float(correct["micro_f1"]) - float(shuffled["micro_f1"]),
    }
    checks = {
        "coverage": float(correct["coverage"]) >= args.min_coverage,
        "format_valid": float(correct["format_valid_rate"]) >= args.min_format_valid,
        "macro_f1_beats_base": deltas["correct_minus_base_macro_f1"] >= args.min_macro_f1_delta,
        "macro_f1_beats_shuffled": deltas["correct_minus_shuffled_macro_f1"] >= args.min_macro_f1_delta,
        "coarse_bucket_not_collapsed": float(correct["coarse_motion_bucket_accuracy"])
        >= args.min_coarse_bucket_accuracy,
        "coarse_bucket_balanced_accuracy": float(correct["coarse_motion_bucket_balanced_accuracy"])
        >= args.min_coarse_bucket_balanced_accuracy,
        "all_motion_buckets_predicted": int(correct["num_predicted_motion_buckets"])
        >= args.min_predicted_motion_buckets,
    }
    passed = all(checks.values())
    summary = {
        "gate": "Stage 1 - correct camera ability learning",
        "status": "passed" if passed else "failed",
        "what_was_tested": (
            "The same videos, prompt, steps, optimizer, and per-sample target length are used. "
            "The control applies a fixed valid within-semantic-group permutation to make camera targets wrong."
        ),
        "thresholds": {
            "min_coverage": args.min_coverage,
            "min_format_valid": args.min_format_valid,
            "min_macro_f1_delta": args.min_macro_f1_delta,
            "min_coarse_bucket_accuracy": args.min_coarse_bucket_accuracy,
            "min_coarse_bucket_balanced_accuracy": args.min_coarse_bucket_balanced_accuracy,
            "min_predicted_motion_buckets": args.min_predicted_motion_buckets,
        },
        "checks": checks,
        "base": compact(base),
        "correct": compact(correct),
        "shuffled": compact(shuffled),
        "correct_deltas": deltas,
        "next_action": (
            "Run the paraphrased-prompt diagnostic, then Stage 2 detection transfer."
            if passed else "Stop before Stage 2 and inspect the learning curve and per-label support."
        ),
    }
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
