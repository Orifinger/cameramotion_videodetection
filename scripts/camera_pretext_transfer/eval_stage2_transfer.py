#!/usr/bin/env python3
"""Evaluate correct-camera pretraining against no-pretext and shuffled controls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.caspr_gate1.eval_gate import load_score_dir
from scripts.caspr_gate1.metrics import aggregate_pairs, grouped_metrics, paired_bootstrap_auc_delta
from scripts.caspr_gate1.runtime import write_json


def align(reference: Sequence[Mapping[str, Any]], other: Sequence[Mapping[str, Any]]) -> tuple[list, list, float]:
    left = {str(row["pair_id"]): row for row in reference}
    right = {str(row["pair_id"]): row for row in other}
    common = sorted(left.keys() & right.keys())
    union = left.keys() | right.keys()
    coverage = len(common) / len(union) if union else 0.0
    return [left[key] for key in common], [right[key] for key in common], coverage


def compare(control: Sequence[Mapping[str, Any]], method: Sequence[Mapping[str, Any]], repeats: int, seed: int) -> dict[str, Any]:
    control_common, method_common, coverage = align(control, method)
    control_metrics = aggregate_pairs(control_common)
    method_metrics = aggregate_pairs(method_common)
    control_motion = grouped_metrics(control_common, "motion_bucket")
    method_motion = grouped_metrics(method_common, "motion_bucket")
    control_source = grouped_metrics(control_common, "source_family")
    method_source = grouped_metrics(method_common, "source_family")
    complex_delta = float("nan")
    if "complex-motion" in control_motion and "complex-motion" in method_motion:
        complex_delta = method_motion["complex-motion"]["auc"] - control_motion["complex-motion"]["auc"]
    source_deltas = {
        name: method_source[name]["auc"] - control_source[name]["auc"]
        for name in sorted(control_source.keys() & method_source.keys())
    }
    return {
        "coverage": coverage,
        "control": control_metrics,
        "method": method_metrics,
        "deltas": {
            "overall_auc": method_metrics["auc"] - control_metrics["auc"],
            "pair_accuracy": method_metrics["pair_accuracy_fake_gt_real"]
            - control_metrics["pair_accuracy_fake_gt_real"],
            "complex_motion_auc": complex_delta,
            "by_source_auc": source_deltas,
            "positive_source_count": sum(value > 0 for value in source_deltas.values()),
        },
        "bootstrap_auc_delta": paired_bootstrap_auc_delta(control_common, method_common, repeats, seed),
    }


def passes(result: Mapping[str, Any], args: argparse.Namespace) -> dict[str, bool]:
    deltas = result["deltas"]
    return {
        "coverage": float(result["coverage"]) >= args.min_coverage,
        "overall_auc": float(deltas["overall_auc"]) >= args.min_auc_delta,
        "pair_accuracy": float(deltas["pair_accuracy"]) >= args.min_pair_accuracy_delta,
        "complex_motion_auc": math.isfinite(float(deltas["complex_motion_auc"]))
        and float(deltas["complex_motion_auc"]) >= args.min_complex_auc_delta,
        "gain_not_confined_to_one_source": int(deltas["positive_source_count"]) >= args.min_positive_sources,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pretext-scores", required=True)
    parser.add_argument("--correct-camera-scores", required=True)
    parser.add_argument("--shuffled-camera-scores", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--min-auc-delta", type=float, default=0.02)
    parser.add_argument("--min-pair-accuracy-delta", type=float, default=0.03)
    parser.add_argument("--min-complex-auc-delta", type=float, default=0.02)
    parser.add_argument("--min-positive-sources", type=int, default=2)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    no_pretext = load_score_dir(args.no_pretext_scores)
    correct = load_score_dir(args.correct_camera_scores)
    shuffled = load_score_dir(args.shuffled_camera_scores)
    versus_no = compare(no_pretext, correct, args.bootstrap_repeats, args.seed)
    versus_shuffled = compare(shuffled, correct, args.bootstrap_repeats, args.seed + 1)
    checks = {
        "correct_vs_no_pretext": passes(versus_no, args),
        "correct_vs_shuffled_camera": passes(versus_shuffled, args),
    }
    passed = all(all(group.values()) for group in checks.values())
    summary = {
        "gate": "Stage 2 - camera ability transfer to local-edit detection",
        "status": "dataa_passed_vif_retention_pending" if passed else "failed",
        "what_was_tested": (
            "All branches use the same pair-rank and DataB replay recipe with no camera text at inference. "
            "The only inherited difference is no camera pretext, correct camera pretext, or shuffled camera pretext."
        ),
        "thresholds": {
            "min_coverage": args.min_coverage,
            "min_auc_delta": args.min_auc_delta,
            "min_pair_accuracy_delta": args.min_pair_accuracy_delta,
            "min_complex_auc_delta": args.min_complex_auc_delta,
            "min_positive_sources": args.min_positive_sources,
        },
        "checks": checks,
        "correct_vs_no_pretext": versus_no,
        "correct_vs_shuffled_camera": versus_shuffled,
        "next_action": (
            "Run fixed-prompt VIF-Bench retention; allowed drop is at most 1.5 points."
            if passed else "Stop this transfer recipe; do not add epochs, GRPO, or prompt-side camera text."
        ),
    }
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
