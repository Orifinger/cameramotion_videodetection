#!/usr/bin/env python3
"""Combine ViF-Bench and GenBuster benchmark CTNE Gate 1 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.camera_ctne_gate1.contracts import write_json


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def combine(vif: dict[str, Any], genbuster: dict[str, Any]) -> dict[str, Any]:
    inputs = {"ViF-Bench": vif, "GenBuster benchmark": genbuster}
    deltas = {
        name: {
            "roc_auc": float(value["deltas"]["matched_minus_unconditional_roc_auc"]),
            "generator_macro_balanced_accuracy": float(
                value["deltas"]["matched_minus_unconditional_generator_macro_balanced_accuracy"]
            ),
            "matched_minus_shuffled_auc_ci_lower": float(
                value["deltas"]["matched_minus_shuffled_bootstrap"]["roc_auc_delta"]["ci95_lower"]
            ),
            "matched_minus_shuffled_macro_ci_lower": float(
                value["deltas"]["matched_minus_shuffled_bootstrap"]["generator_macro_balanced_delta"]["ci95_lower"]
            ),
        }
        for name, value in inputs.items()
    }
    gain_on_one = any(
        (value["roc_auc"] >= 0.01 and value["generator_macro_balanced_accuracy"] >= -0.005)
        or (value["generator_macro_balanced_accuracy"] >= 0.01 and value["roc_auc"] >= -0.005)
        for value in deltas.values()
    )
    no_large_cross_benchmark_drop = all(
        value["roc_auc"] >= -0.005 and value["generator_macro_balanced_accuracy"] >= -0.005
        for value in deltas.values()
    )
    matched_beats_shuffled = all(
        value["matched_minus_shuffled_auc_ci_lower"] > 0.0
        or value["matched_minus_shuffled_macro_ci_lower"] > 0.0
        for value in deltas.values()
    )
    camera_only_clean = all(value["checks"]["camera_only_not_suspiciously_high"] for value in inputs.values())
    enough_seeds = all(value["checks"]["all_seed_pairs_present"] for value in inputs.values())
    direction_consistent = all(
        value["checks"]["motion_bucket_direction_consistency"]
        and value["checks"]["generator_direction_consistency"]
        for value in inputs.values()
    )
    checks = {
        "camera_increment_on_at_least_one_benchmark": gain_on_one,
        "no_more_than_half_point_drop_on_other_benchmark": no_large_cross_benchmark_drop,
        "matched_beats_shuffled_on_both_benchmarks": matched_beats_shuffled,
        "camera_only_shortcut_control_clean": camera_only_clean,
        "three_seed_pairs_present": enough_seeds,
        "motion_and_generator_direction_consistency": direction_consistent,
    }
    return {
        "gate": "Gate 1 final decision across ViF-Bench and GenBuster benchmark",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "deltas": deltas,
        "decision": (
            "Gate 1 passed; proceed to frozen-Qwen score fusion (Gate 2)."
            if all(checks.values())
            else "Gate 1 failed; do not start Qwen pose-token, caption-SFT, or RL experiments for the camera claim."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vif-summary", type=Path, required=True)
    parser.add_argument("--genbuster-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = combine(_load(args.vif_summary), _load(args.genbuster_summary))
    write_json(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
