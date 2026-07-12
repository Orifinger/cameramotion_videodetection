#!/usr/bin/env python3
"""Compare matched control and CASPR pair-ranking verdict scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

from scripts.caspr_gate1.metrics import aggregate_pairs, grouped_metrics, paired_bootstrap_auc_delta
from scripts.caspr_gate1.runtime import read_jsonl, write_json


def load_score_dir(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.glob("rank_*.jsonl"))
    rows: list[dict[str, Any]] = []
    for file_path in files:
        rows.extend(read_jsonl(file_path))
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        if not pair_id:
            raise ValueError(f"score row has no pair_id: {row}")
        if pair_id in by_id:
            raise ValueError(f"duplicate pair_id in scores: {pair_id}")
        by_id[pair_id] = row
    return [by_id[pair_id] for pair_id in sorted(by_id)]


def metric_delta(method: Mapping[str, Any], control: Mapping[str, Any], key: str) -> float:
    return float(method[key]) - float(control[key])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-scores", required=True)
    parser.add_argument("--method-scores", required=True)
    parser.add_argument("--baseline-scores")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--min-auc-delta", type=float, default=0.03)
    parser.add_argument("--min-pair-accuracy-delta", type=float, default=0.05)
    parser.add_argument("--min-complex-auc-delta", type=float, default=0.03)
    parser.add_argument("--max-source-auc-drop", type=float, default=0.02)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    control = load_score_dir(args.control_scores)
    method = load_score_dir(args.method_scores)
    baseline = load_score_dir(args.baseline_scores) if args.baseline_scores else []
    control_by_id = {str(row["pair_id"]): row for row in control}
    method_by_id = {str(row["pair_id"]): row for row in method}
    common_ids = sorted(control_by_id.keys() & method_by_id.keys())
    union_ids = control_by_id.keys() | method_by_id.keys()
    coverage = len(common_ids) / len(union_ids) if union_ids else 0.0
    control_common = [control_by_id[pair_id] for pair_id in common_ids]
    method_common = [method_by_id[pair_id] for pair_id in common_ids]
    control_overall = aggregate_pairs(control_common)
    method_overall = aggregate_pairs(method_common)
    control_motion = grouped_metrics(control_common, "motion_bucket")
    method_motion = grouped_metrics(method_common, "motion_bucket")
    control_source = grouped_metrics(control_common, "source_family")
    method_source = grouped_metrics(method_common, "source_family")
    baseline_by_id = {str(row["pair_id"]): row for row in baseline}
    baseline_common = [baseline_by_id[pair_id] for pair_id in common_ids if pair_id in baseline_by_id]
    baseline_metrics = aggregate_pairs(baseline_common) if baseline_common else None
    overall_auc_delta = metric_delta(method_overall, control_overall, "auc")
    pair_accuracy_delta = metric_delta(method_overall, control_overall, "pair_accuracy_fake_gt_real")
    complex_auc_delta = float("nan")
    if "complex-motion" in control_motion and "complex-motion" in method_motion:
        complex_auc_delta = method_motion["complex-motion"]["auc"] - control_motion["complex-motion"]["auc"]
    source_auc_deltas = {
        source: method_source[source]["auc"] - control_source[source]["auc"]
        for source in sorted(control_source.keys() & method_source.keys())
    }
    finite_source_deltas = [value for value in source_auc_deltas.values() if math.isfinite(value)]
    worst_source_delta = min(finite_source_deltas) if finite_source_deltas else float("nan")
    checks = {
        "score_coverage": coverage >= args.min_coverage,
        "overall_auc_delta": overall_auc_delta >= args.min_auc_delta,
        "pair_accuracy_delta": pair_accuracy_delta >= args.min_pair_accuracy_delta,
        "complex_motion_auc_delta": math.isfinite(complex_auc_delta)
        and complex_auc_delta >= args.min_complex_auc_delta,
        "no_source_auc_drop_over_limit": math.isfinite(worst_source_delta)
        and worst_source_delta >= -args.max_source_auc_drop,
    }
    passed = bool(common_ids) and all(checks.values())
    bootstrap = paired_bootstrap_auc_delta(
        control_common, method_common, args.bootstrap_repeats, args.seed
    )
    summary = {
        "gate": "CASPR Gate 1A - independent verdict pair ranking",
        "status": "dataa_passed_vif_retention_pending" if passed else "failed",
        "what_was_tested": (
            "The control and method use the same checkpoint, videos, verdict binary loss, DataB replay, "
            "steps, and prompt. The method alone adds exact DataA fake-vs-real score ranking."
        ),
        "what_was_not_tested": (
            "This stage does not establish a camera-pretraining gain, explanation improvement, or clean final-test performance."
        ),
        "coverage": {
            "control_pairs": len(control),
            "method_pairs": len(method),
            "common_pairs": len(common_ids),
            "common_over_union": coverage,
        },
        "thresholds": {
            "min_coverage": args.min_coverage,
            "min_auc_delta": args.min_auc_delta,
            "min_pair_accuracy_delta": args.min_pair_accuracy_delta,
            "min_complex_auc_delta": args.min_complex_auc_delta,
            "max_source_auc_drop": args.max_source_auc_drop,
        },
        "checks": checks,
        "control": {
            "overall": control_overall,
            "by_motion_bucket": control_motion,
            "by_source_family": control_source,
        },
        "starting_checkpoint_context": {
            "note": "Reported for context only; the matched control is the causal comparison because both trained runs adapt to the verdict prompt.",
            "coverage_pairs": len(baseline_common),
            "overall": baseline_metrics,
        },
        "method": {
            "overall": method_overall,
            "by_motion_bucket": method_motion,
            "by_source_family": method_source,
        },
        "method_minus_control": {
            "overall_auc": overall_auc_delta,
            "pair_accuracy": pair_accuracy_delta,
            "complex_motion_auc": complex_auc_delta,
            "by_source_auc": source_auc_deltas,
            "worst_source_auc": worst_source_delta,
        },
        "bootstrap_auc_delta": bootstrap,
        "next_action": (
            "Merge both adapters and run the fixed VIF-Bench retention comparison."
            if passed
            else "Stop this ranking recipe; do not start camera pretraining or RL."
        ),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "caspr_gate1_dataa_summary.json", summary)
    with (out_dir / "caspr_gate1_pair_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "pair_id", "source_family", "motion_bucket", "control_real_score", "control_fake_score",
            "method_real_score", "method_fake_score", "control_margin", "method_margin",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair_id in common_ids:
            c, m = control_by_id[pair_id], method_by_id[pair_id]
            writer.writerow(
                {
                    "pair_id": pair_id,
                    "source_family": c.get("source_family"),
                    "motion_bucket": c.get("motion_bucket"),
                    "control_real_score": c["real_score"],
                    "control_fake_score": c["fake_score"],
                    "method_real_score": m["real_score"],
                    "method_fake_score": m["fake_score"],
                    "control_margin": c["score_margin_fake_minus_real"],
                    "method_margin": m["score_margin_fake_minus_real"],
                }
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
