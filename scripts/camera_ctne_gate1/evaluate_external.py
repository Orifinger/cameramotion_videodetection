#!/usr/bin/env python3
"""Evaluate a CTNE model bundle on an untouched external benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_ctne_gate1.contracts import MODEL_SCHEMA_VERSION, normalize_path, read_jsonl, write_json
from scripts.camera_ctne_gate1.evaluate import (
    _bootstrap_delta,
    _direction_consistency,
    _discover_models,
    _method_report,
    _score_flow_controls,
)
from scripts.camera_ctne_gate1.preprocessing import CTNEPreprocessor


def _load_camera_probe(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema_version"].item()) != MODEL_SCHEMA_VERSION:
            raise ValueError("camera-only calibration schema mismatch")
        return {
            "mean": np.asarray(archive["scaler_mean"], dtype=np.float32),
            "scale": np.asarray(archive["scaler_scale"], dtype=np.float32),
            "coefficient": np.asarray(archive["coefficient"], dtype=np.float32),
            "intercept": np.asarray(archive["intercept"], dtype=np.float32),
        }


def _camera_probability(summaries: np.ndarray, probe: dict[str, np.ndarray]) -> np.ndarray:
    standardized = (summaries - probe["mean"]) / np.maximum(probe["scale"], 1e-8)
    logits = standardized @ probe["coefficient"] + float(probe["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def evaluate_external(args: argparse.Namespace) -> dict[str, Any]:
    calibration = json.loads((args.calibration_dir / "calibration.json").read_text(encoding="utf-8"))
    thresholds = {key: float(value) for key, value in calibration["thresholds"].items()}
    mean_weight = float(calibration["video_score_mean_weight"])
    shuffle_seed = int(calibration["shuffle_seed"])
    all_rows = read_jsonl(args.test_index_jsonl)
    rows = [row for row in all_rows if args.test_split == "all" or str(row.get("dataset_split")) == args.test_split]
    if not rows:
        raise ValueError("external feature index selected zero rows")
    preprocessor = CTNEPreprocessor.load(args.model_root / "preprocessor.npz")
    scores, by_seed, donors, relaxation, camera_summaries = _score_flow_controls(
        rows,
        model_root=args.model_root,
        preprocessor=preprocessor,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        mean_weight=mean_weight,
        shuffle_seed=shuffle_seed,
    )
    scores["camera_only"] = _camera_probability(
        camera_summaries,
        _load_camera_probe(args.calibration_dir / "camera_only.npz"),
    )
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    reports = {method: _method_report(rows, labels, values, thresholds[method]) for method, values in scores.items()}
    per_seed = {
        method: {
            str(seed): _method_report(rows, labels, values, thresholds[method])
            for seed, values in sorted(seed_values.items())
        }
        for method, seed_values in by_seed.items()
    }
    matched_vs_unconditional = _bootstrap_delta(
        rows,
        labels,
        scores["matched"],
        scores["unconditional"],
        thresholds["matched"],
        thresholds["unconditional"],
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    matched_vs_shuffled = _bootstrap_delta(
        rows,
        labels,
        scores["matched"],
        scores["shuffled"],
        thresholds["matched"],
        thresholds["shuffled"],
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed + 1,
    )
    auc_delta = float(reports["matched"]["roc_auc"] - reports["unconditional"]["roc_auc"])
    macro_delta = float(
        reports["matched"]["generator_macro_balanced_accuracy"]
        - reports["unconditional"]["generator_macro_balanced_accuracy"]
    )
    directions = _direction_consistency(
        rows,
        labels,
        scores["matched"],
        scores["unconditional"],
        thresholds["matched"],
        thresholds["unconditional"],
    )
    checks = {
        "matched_has_increment_over_unconditional": (
            (auc_delta >= args.min_primary_gain and macro_delta >= -args.max_other_drop)
            or (macro_delta >= args.min_primary_gain and auc_delta >= -args.max_other_drop)
        ),
        "matched_beats_shuffled_with_paired_ci": (
            matched_vs_shuffled["roc_auc_delta"]["ci95_lower"] > 0.0
            or matched_vs_shuffled["generator_macro_balanced_delta"]["ci95_lower"] > 0.0
        ),
        "camera_only_not_suspiciously_high": reports["camera_only"]["roc_auc"] <= args.max_camera_only_auc,
        "motion_bucket_direction_consistency": directions["motion_bucket"]["positive_groups"] >= 2,
        "generator_direction_consistency": directions["generator_name"]["positive_rate"] > 0.5,
        "all_seed_pairs_present": len(_discover_models(args.model_root)) >= args.min_seeds,
    }
    result = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "gate": f"Gate 1 external CTNE evaluation - {args.dataset_name}",
        "status": "passed" if all(checks.values()) else "failed",
        "what_was_tested": (
            "Correct continuous camera context, equal-capacity zero context, and evaluation-only shuffled context "
            "were scored on identical variable-length transitions. Thresholds came only from DataB validation."
        ),
        "inputs": {
            "dataset_name": args.dataset_name,
            "test_index_jsonl": normalize_path(args.test_index_jsonl),
            "model_root": normalize_path(args.model_root),
            "calibration_dir": normalize_path(args.calibration_dir),
        },
        "thresholds": {
            "decision_thresholds_from_datab_validation": thresholds,
            "min_primary_gain": args.min_primary_gain,
            "max_other_drop": args.max_other_drop,
            "max_camera_only_auc": args.max_camera_only_auc,
        },
        "checks": checks,
        "settings": {
            "num_samples": len(rows),
            "video_score_mean_weight": mean_weight,
            "video_score_tail_quantile": 0.90,
            "shuffle_seed": shuffle_seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "qwen_used": False,
            "camera_text_used": False,
        },
        "test": {"reports": reports, "shuffle_relaxation_counts": relaxation},
        "deltas": {
            "matched_minus_unconditional_roc_auc": auc_delta,
            "matched_minus_unconditional_generator_macro_balanced_accuracy": macro_delta,
            "matched_minus_unconditional_bootstrap": matched_vs_unconditional,
            "matched_minus_shuffled_bootstrap": matched_vs_shuffled,
        },
        "direction_consistency": directions,
        "per_seed_reports": per_seed,
        "does_not_establish": "This Gate 1 result does not yet establish a Qwen Real/Fake improvement.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "ctne_gate1_summary.json", result)
    with (args.output_dir / "ctne_gate1_items.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "sample_id",
            "label",
            "source_name",
            "generator_name",
            "motion_bucket",
            "frame_count",
            "transition_count",
            "matched_score",
            "unconditional_score",
            "shuffled_score",
            "camera_only_score",
            "shuffled_camera_donor_sample_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    **{key: row.get(key) for key in fields[:7]},
                    "matched_score": scores["matched"][index],
                    "unconditional_score": scores["unconditional"][index],
                    "shuffled_score": scores["shuffled"][index],
                    "camera_only_score": scores["camera_only"][index],
                    "shuffled_camera_donor_sample_id": rows[donors[index]]["sample_id"],
                }
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--test-index-jsonl", type=Path, required=True)
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument("--min-primary-gain", type=float, default=0.01)
    parser.add_argument("--max-other-drop", type=float, default=0.005)
    parser.add_argument("--max-camera-only-auc", type=float, default=0.65)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    evaluate_external(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
