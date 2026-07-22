#!/usr/bin/env python3
"""Evaluate the temporal-causality controls on ViF-Bench development data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.forensic_temporal_expert_gate.contracts import normalize_path, read_json_or_jsonl, write_json
from scripts.forensic_temporal_expert_gate.data import FeatureDataset, collate_features
from scripts.forensic_temporal_expert_gate.metrics import classification_metrics, logit, sigmoid
from scripts.forensic_temporal_expert_gate.model import load_model


CONDITIONS = {
    "static": ("static", "ordered"),
    "ordered": ("ordered", "ordered"),
    "ordered_shuffled_input": ("ordered", "shuffled"),
    "shuffled_trained": ("shuffled", "shuffled"),
}


def discover_models(root: Path) -> dict[int, dict[str, Path]]:
    output: dict[int, dict[str, Path]] = {}
    for seed_dir in sorted((root / "models").glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        modes = {
            mode: seed_dir / mode
            for mode in ("static", "ordered", "shuffled")
            if (seed_dir / mode / "model.pt").is_file()
            and (seed_dir / mode / "config.json").is_file()
        }
        if len(modes) == 3:
            output[seed] = modes
    if not output:
        raise FileNotFoundError(f"no complete three-mode model seeds under {root}")
    return output


@torch.inference_mode()
def score(
    rows: Sequence[Mapping[str, Any]],
    model_dir: Path,
    *,
    order: str,
    order_seed: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    model, config = load_model(model_dir, device)
    dataset = FeatureDataset(rows, order=order, seed=order_seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_features,
    )
    values: list[np.ndarray] = []
    for batch in loader:
        logits = model(
            batch["cls_tokens"].to(device, non_blocking=True),
            batch["patch_tokens"].to(device, non_blocking=True),
            batch["lengths"].to(device, non_blocking=True),
        )
        values.append(logits.float().cpu().numpy())
    threshold = float(config["validation_threshold"])
    return sigmoid(np.concatenate(values)), threshold


def calibrated_ensemble(values: Sequence[np.ndarray], thresholds: Sequence[float]) -> np.ndarray:
    adjusted = [sigmoid(logit(scores) - logit(np.asarray([threshold]))[0]) for scores, threshold in zip(values, thresholds)]
    return np.mean(np.stack(adjusted), axis=0)


def bootstrap_delta(
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row.get("group_id", row["sample_id"]))].append(index)
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        indices = np.asarray([index for key in sampled for index in grouped[str(key)]], dtype=np.int64)
        if np.unique(labels[indices]).size < 2:
            continue
        values.append(float(roc_auc_score(labels[indices], first[indices]) - roc_auc_score(labels[indices], second[indices])))
    array = np.asarray(values, dtype=np.float64)
    return {
        "iterations": int(array.size),
        "mean": float(array.mean()),
        "ci95_lower": float(np.quantile(array, 0.025)),
        "ci95_upper": float(np.quantile(array, 0.975)),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-index-jsonl", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shuffle-seed", type=int, default=20260722)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-records", type=int, default=3160)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--min-primary-gain", type=float, default=0.015)
    parser.add_argument("--max-other-primary-drop", type=float, default=0.01)
    parser.add_argument("--min-order-sensitivity", type=float, default=0.01)
    parser.add_argument("--max-real-recall-drop", type=float, default=0.03)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_json_or_jsonl(args.feature_index_jsonl)
    coverage = len(rows) / args.expected_records if args.expected_records > 0 else 0.0
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    generators = [str(row.get("generator_name", "unknown")) for row in rows]
    models = discover_models(args.model_root)
    device = torch.device(args.device)
    by_seed: dict[str, dict[int, dict[str, Any]]] = {key: {} for key in CONDITIONS}
    ensemble_scores: dict[str, np.ndarray] = {}
    for condition, (mode, order) in CONDITIONS.items():
        values: list[np.ndarray] = []
        thresholds: list[float] = []
        for seed, paths in sorted(models.items()):
            scores, threshold = score(
                rows,
                paths[mode],
                order=order,
                order_seed=args.shuffle_seed,
                batch_size=args.batch_size,
                device=device,
            )
            values.append(scores)
            thresholds.append(threshold)
            by_seed[condition][seed] = classification_metrics(
                labels, scores, threshold, generators
            )
        ensemble_scores[condition] = calibrated_ensemble(values, thresholds)

    reports = {
        condition: classification_metrics(labels, scores, 0.5, generators)
        for condition, scores in ensemble_scores.items()
    }
    primary = ("roc_auc", "generator_macro_balanced_accuracy")

    def deltas(left: str, right: str) -> dict[str, float]:
        return {key: float(reports[left][key] - reports[right][key]) for key in primary}

    ordered_static = deltas("ordered", "static")
    ordered_shuffled = deltas("ordered", "shuffled_trained")
    order_sensitivity = deltas("ordered", "ordered_shuffled_input")
    seed_wins_static = 0
    seed_wins_shuffled = 0
    for seed in models:
        seed_wins_static += int(max(
            float(by_seed["ordered"][seed][key] - by_seed["static"][seed][key])
            for key in primary
        ) > 0)
        seed_wins_shuffled += int(max(
            float(by_seed["ordered"][seed][key] - by_seed["shuffled_trained"][seed][key])
            for key in primary
        ) > 0)

    def primary_pass(delta: Mapping[str, float]) -> bool:
        values = [float(delta[key]) for key in primary]
        return max(values) >= args.min_primary_gain and min(values) >= -args.max_other_primary_drop

    checks = {
        "held_out_feature_coverage": coverage >= args.min_coverage,
        "exactly_three_complete_seeds": len(models) == 3,
        "ordered_beats_static": primary_pass(ordered_static),
        "ordered_beats_equal_capacity_shuffled_training": primary_pass(ordered_shuffled),
        "ordered_model_uses_frame_order": max(order_sensitivity.values()) >= args.min_order_sensitivity,
        "seed_direction_consistency_vs_static": seed_wins_static >= 2,
        "seed_direction_consistency_vs_shuffled_training": seed_wins_shuffled >= 2,
        "real_recall_preserved": float(reports["ordered"]["real_recall"] - reports["static"]["real_recall"]) >= -args.max_real_recall_drop,
    }
    status = "passed" if all(checks.values()) else "failed"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items_path = args.output_dir / "forensic_temporal_expert_vifbench_items.csv"
    with items_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "sample_id", "video_id", "label", "generator_name",
            *[f"{condition}_score" for condition in CONDITIONS],
            *[f"{condition}_prediction" for condition in CONDITIONS],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            item = {
                "sample_id": row["sample_id"],
                "video_id": row.get("video_id", ""),
                "label": row["label"],
                "generator_name": row.get("generator_name", "unknown"),
            }
            for condition in CONDITIONS:
                value = float(ensemble_scores[condition][index])
                item[f"{condition}_score"] = value
                item[f"{condition}_prediction"] = int(value >= 0.5)
            writer.writerow(item)
    summary = {
        "gate": "原生尺度 DINO 时序因果专家门（Gate 1）",
        "status": status,
        "what_was_tested": (
            "Frozen DINOv2 patch tokens on every available frame; static, correctly ordered, "
            "and equal-capacity shuffled-order experts. No camera text, RAFT, or Qwen is used."
        ),
        "development_dataset": "ViF-Bench; no threshold or fusion weight is fitted on its labels",
        "genbuster_closed_benchmark_touched": False,
        "expected_records": args.expected_records,
        "valid_feature_records": len(rows),
        "coverage": coverage,
        "models": {str(seed): {mode: normalize_path(path) for mode, path in paths.items()} for seed, paths in models.items()},
        "thresholds": {
            "min_primary_gain": args.min_primary_gain,
            "max_other_primary_drop": args.max_other_primary_drop,
            "min_order_sensitivity": args.min_order_sensitivity,
            "max_real_recall_drop": args.max_real_recall_drop,
            "min_positive_seeds": 2,
        },
        "checks": checks,
        "metrics": reports,
        "per_seed_metrics": by_seed,
        "deltas": {
            "ordered_minus_static": ordered_static,
            "ordered_minus_shuffled_trained": ordered_shuffled,
            "ordered_minus_ordered_model_shuffled_input": order_sensitivity,
        },
        "bootstrap_ordered_minus_static_roc_auc": bootstrap_delta(
            rows, labels, ensemble_scores["ordered"], ensemble_scores["static"],
            iterations=args.bootstrap_iterations, seed=17,
        ),
        "items_csv": normalize_path(items_path),
        "does_not_establish": "This development gate does not establish final GenBuster Closed Benchmark gains.",
        "next_action": "Run fixed Qwen complementarity Gate 2 only if this Gate passes.",
    }
    write_json(args.output_dir / "forensic_temporal_expert_gate1_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
