#!/usr/bin/env python3
"""Train equal-capacity frozen-feature controls and evaluate them on ViF-Bench."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_flow_probe.contracts import read_jsonl
from scripts.camera_flow_probe.metrics import best_balanced_threshold, binary_metrics, roc_auc
from scripts.camera_geometric_residual_gate.contracts import FEATURE_SCHEMA_VERSION, feature_filename, write_json
from scripts.camera_geometric_residual_gate.features import VARIANT_DIMS


VARIANTS = (
    "appearance",
    "appearance_raw_motion",
    "appearance_geometry_residual",
    "appearance_wrong_geometry",
)
PRIMARY = "appearance_geometry_residual"
CONTROLS = ("appearance_raw_motion", "appearance_wrong_geometry")
MOTION_BUCKETS = ("static/no-motion", "minor-motion", "complex-motion")


def _network(input_dim: int, hidden_dims: Sequence[int]) -> nn.Module:
    layers: list[nn.Module] = []
    current = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(current, int(hidden)), nn.ReLU(), nn.Dropout(0.10)])
        current = int(hidden)
    layers.append(nn.Linear(current, 1))
    return nn.Sequential(*layers)


@dataclass
class ModelState:
    mean: np.ndarray
    std: np.ndarray
    hidden_dims: tuple[int, ...]
    state_dict: dict[str, torch.Tensor]

    def predict(self, values: np.ndarray, *, device: torch.device, batch_size: int = 4096) -> np.ndarray:
        model = _network(self.mean.shape[0], self.hidden_dims).to(device)
        model.load_state_dict(self.state_dict)
        model.eval()
        normalized = ((np.nan_to_num(values) - self.mean) / self.std).astype(np.float32)
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(normalized), batch_size):
                batch = torch.from_numpy(normalized[start : start + batch_size]).to(device)
                outputs.append(torch.sigmoid(model(batch).squeeze(1)).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32)


def _dataset_directory(row: Mapping[str, Any]) -> str:
    return str(row.get("dataset_name", "dataset")).casefold().replace("-", "_")


def _load_matrix(
    rows: Sequence[Mapping[str, Any]],
    feature_root: Path,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, list[Mapping[str, Any]], list[str]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    kept: list[Mapping[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        path = feature_root / _dataset_directory(row) / feature_filename(str(row["sample_id"]))
        if not path.is_file():
            missing.append(str(row["sample_id"]))
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["schema_version"].item()) != FEATURE_SCHEMA_VERSION:
                    raise ValueError("schema mismatch")
                value = np.asarray(archive[variant], dtype=np.float32)
        except Exception:  # noqa: BLE001
            missing.append(str(row["sample_id"]))
            continue
        if value.shape != (VARIANT_DIMS[variant],):
            missing.append(str(row["sample_id"]))
            continue
        features.append(value)
        labels.append(int(row["label"]))
        kept.append(row)
    matrix = np.stack(features) if features else np.zeros((0, VARIANT_DIMS[variant]), dtype=np.float32)
    return matrix, np.asarray(labels, dtype=np.int64), kept, missing


def _sample_weights(rows: Sequence[Mapping[str, Any]], *, max_weight: float) -> np.ndarray:
    keys = [
        (str(row.get("source_name")), str(row.get("answer")), str(row.get("motion_bucket")))
        for row in rows
    ]
    counts = Counter(keys)
    cells = max(1, len(counts))
    raw = np.asarray([len(rows) / (cells * counts[key]) for key in keys], dtype=np.float32)
    raw = np.minimum(raw, float(max_weight))
    return raw / max(float(raw.mean()), 1e-8)


def _train_one(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_w: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    hidden_dims: Sequence[int],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    patience_limit: int,
) -> ModelState:
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean = np.nanmean(train_x, axis=0).astype(np.float32)
    std = np.nanstd(train_x, axis=0).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0

    def normalize(values: np.ndarray) -> np.ndarray:
        return ((np.nan_to_num(values) - mean) / std).astype(np.float32)

    dataset = TensorDataset(
        torch.from_numpy(normalize(train_x)),
        torch.from_numpy(train_y.astype(np.float32)),
        torch.from_numpy(train_w.astype(np.float32)),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    hidden_dims = tuple(int(value) for value in hidden_dims)
    model = _network(train_x.shape[1], hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_auc = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    val_tensor = torch.from_numpy(normalize(val_x)).to(device)
    for _epoch in range(epochs):
        model.train()
        for batch_x, batch_y, batch_w in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device)).squeeze(1)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits,
                batch_y.to(device),
                reduction="none",
            )
            loss = (losses * batch_w.to(device)).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            scores = torch.sigmoid(model(val_tensor).squeeze(1)).cpu().numpy()
        selection = roc_auc(val_y, scores)
        if np.isfinite(selection) and selection > best_auc + 1e-5:
            best_auc = float(selection)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                break
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return ModelState(mean=mean, std=std, hidden_dims=hidden_dims, state_dict=best_state)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores) & np.isin(labels, [0, 1])
    labels, scores = labels[valid], scores[valid]
    positives = int((labels == 1).sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    tp = np.cumsum(ranked == 1)
    precision = tp / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    output = binary_metrics(labels, scores, threshold)
    output["average_precision"] = average_precision(labels, scores)
    return output


def _group_metrics(
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    field: str,
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row.get(field, "unknown"))].append(index)
    output: dict[str, dict[str, float]] = {}
    for name, indices in sorted(groups.items()):
        index = np.asarray(indices, dtype=np.int64)
        output[name] = _metrics(labels[index], scores[index], threshold)
    return output


def _motion_macro(by_motion: Mapping[str, Mapping[str, float]]) -> float:
    values = [
        float(by_motion[bucket]["balanced_accuracy"])
        for bucket in MOTION_BUCKETS
        if bucket in by_motion
        and int(by_motion[bucket].get("num_samples", 0)) > 0
        and np.isfinite(float(by_motion[bucket].get("auc", np.nan)))
    ]
    return float(np.mean(values)) if values else float("nan")


def _paired_auc_bootstrap(
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    real = np.flatnonzero(labels == 0)
    fake = np.flatnonzero(labels == 1)
    observed = float(roc_auc(labels, first) - roc_auc(labels, second))
    deltas: list[float] = []
    for _ in range(iterations):
        indices = np.concatenate(
            [
                rng.choice(real, size=real.size, replace=True),
                rng.choice(fake, size=fake.size, replace=True),
            ]
        )
        delta = roc_auc(labels[indices], first[indices]) - roc_auc(labels[indices], second[indices])
        if np.isfinite(delta):
            deltas.append(float(delta))
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "observed_delta": observed,
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "iterations": int(values.size),
    }


def _source_win_rate(
    primary: Mapping[str, Mapping[str, float]],
    control: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    supported = [
        source
        for source in primary.keys() & control.keys()
        if np.isfinite(float(primary[source].get("auc", np.nan)))
        and np.isfinite(float(control[source].get("auc", np.nan)))
    ]
    wins = sum(primary[source]["balanced_accuracy"] > control[source]["balanced_accuracy"] for source in supported)
    return {
        "supported_sources": len(supported),
        "wins": int(wins),
        "win_rate": float(wins / len(supported)) if supported else float("nan"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default="20260719,20260720,20260721,20260722,20260723")
    parser.add_argument("--hidden-dims", default="64,32")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--max-sample-weight", type=float, default=5.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--min-auc-gain", type=float, default=0.01)
    parser.add_argument("--min-motion-macro-balanced-gain", type=float, default=0.01)
    parser.add_argument("--max-static-balanced-drop", type=float, default=0.01)
    parser.add_argument("--min-source-win-rate", type=float, default=0.60)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.train_manifest)
    test_rows = read_jsonl(args.test_manifest)
    train_rows_only = [row for row in train_rows if row.get("dataset_split") == "train"]
    val_rows_only = [row for row in train_rows if row.get("dataset_split") == "val"]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    hidden_dims = [int(value) for value in args.hidden_dims.split(",") if value.strip()]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results: dict[str, Any] = {}
    prediction_rows: dict[str, dict[str, Any]] = {
        str(row["sample_id"]): {
            "sample_id": row["sample_id"],
            "answer": row["answer"],
            "motion_bucket": row.get("motion_bucket"),
            "source_name": row.get("source_name"),
        }
        for row in test_rows
    }
    test_scores_by_variant: dict[str, np.ndarray] = {}
    kept_test_rows: list[Mapping[str, Any]] | None = None
    kept_test_labels: np.ndarray | None = None
    coverages: dict[str, float] = {}

    for variant in VARIANTS:
        train_x, train_y, kept_train, missing_train = _load_matrix(train_rows_only, args.feature_root, variant)
        val_x, val_y, kept_val, missing_val = _load_matrix(val_rows_only, args.feature_root, variant)
        test_x, test_y, kept_test, missing_test = _load_matrix(test_rows, args.feature_root, variant)
        coverage = min(
            len(kept_train) / max(1, len(train_rows_only)),
            len(kept_val) / max(1, len(val_rows_only)),
            len(kept_test) / max(1, len(test_rows)),
        )
        coverages[variant] = coverage
        if coverage < args.min_coverage:
            raise ValueError(
                f"feature coverage below threshold for {variant}: {coverage:.6f}; "
                f"missing train/val/test={len(missing_train)}/{len(missing_val)}/{len(missing_test)}"
            )
        if kept_test_rows is None:
            kept_test_rows, kept_test_labels = kept_test, test_y
        elif [row["sample_id"] for row in kept_test_rows] != [row["sample_id"] for row in kept_test]:
            raise ValueError(f"test sample order differs for {variant}")
        train_weights = _sample_weights(kept_train, max_weight=args.max_sample_weight)
        val_predictions: list[np.ndarray] = []
        test_predictions: list[np.ndarray] = []
        for seed in seeds:
            state = _train_one(
                train_x,
                train_y,
                train_weights,
                val_x,
                val_y,
                seed=seed,
                device=device,
                hidden_dims=hidden_dims,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                patience_limit=args.patience,
            )
            val_predictions.append(state.predict(val_x, device=device))
            test_predictions.append(state.predict(test_x, device=device))
        val_scores = np.mean(np.stack(val_predictions), axis=0)
        test_scores = np.mean(np.stack(test_predictions), axis=0)
        threshold = best_balanced_threshold(val_y, val_scores)
        by_motion = _group_metrics(kept_test, test_y, test_scores, threshold, "motion_bucket")
        by_source = _group_metrics(kept_test, test_y, test_scores, threshold, "source_name")
        results[variant] = {
            "input_dim": int(train_x.shape[1]),
            "coverage": coverage,
            "train_samples": len(train_y),
            "val_samples": len(val_y),
            "test_samples": len(test_y),
            "threshold_selected_on_datab_val": float(threshold),
            "datab_val": _metrics(val_y, val_scores, threshold),
            "vif_overall": _metrics(test_y, test_scores, threshold),
            "vif_motion_macro_balanced_accuracy": _motion_macro(by_motion),
            "vif_by_motion": by_motion,
            "vif_by_source": by_source,
        }
        test_scores_by_variant[variant] = test_scores
        for row, score in zip(kept_test, test_scores, strict=True):
            prediction_rows[str(row["sample_id"])][variant] = float(score)

    assert kept_test_rows is not None and kept_test_labels is not None
    primary_result = results[PRIMARY]
    comparisons: dict[str, Any] = {}
    for control in CONTROLS:
        primary_overall = primary_result["vif_overall"]
        control_overall = results[control]["vif_overall"]
        bootstrap = _paired_auc_bootstrap(
            kept_test_labels,
            test_scores_by_variant[PRIMARY],
            test_scores_by_variant[control],
            seed=20260719,
            iterations=args.bootstrap_iterations,
        )
        source_wins = _source_win_rate(primary_result["vif_by_source"], results[control]["vif_by_source"])
        comparisons[control] = {
            "auc_delta": primary_overall["auc"] - control_overall["auc"],
            "balanced_accuracy_delta": primary_overall["balanced_accuracy"] - control_overall["balanced_accuracy"],
            "motion_macro_balanced_accuracy_delta": (
                primary_result["vif_motion_macro_balanced_accuracy"]
                - results[control]["vif_motion_macro_balanced_accuracy"]
            ),
            "static_balanced_accuracy_delta": (
                primary_result["vif_by_motion"].get("static/no-motion", {}).get("balanced_accuracy", float("nan"))
                - results[control]["vif_by_motion"].get("static/no-motion", {}).get("balanced_accuracy", float("nan"))
            ),
            "paired_auc_bootstrap": bootstrap,
            "source_balanced_accuracy_wins": source_wins,
        }

    raw_cmp = comparisons["appearance_raw_motion"]
    wrong_cmp = comparisons["appearance_wrong_geometry"]
    supported_source_checks = [
        item["source_balanced_accuracy_wins"]
        for item in (raw_cmp, wrong_cmp)
        if item["source_balanced_accuracy_wins"]["supported_sources"] >= 3
    ]
    checks = {
        "all_feature_coverages": min(coverages.values()) >= args.min_coverage,
        "geometry_auc_beats_raw_motion": raw_cmp["auc_delta"] >= args.min_auc_gain,
        "geometry_auc_beats_wrong_camera": wrong_cmp["auc_delta"] >= args.min_auc_gain,
        "geometry_motion_macro_beats_raw_motion": (
            raw_cmp["motion_macro_balanced_accuracy_delta"] >= args.min_motion_macro_balanced_gain
        ),
        "geometry_motion_macro_beats_wrong_camera": (
            wrong_cmp["motion_macro_balanced_accuracy_delta"] >= args.min_motion_macro_balanced_gain
        ),
        "paired_auc_ci_above_zero_vs_raw": raw_cmp["paired_auc_bootstrap"]["ci95_low"] > 0.0,
        "paired_auc_ci_above_zero_vs_wrong": wrong_cmp["paired_auc_bootstrap"]["ci95_low"] > 0.0,
        "static_not_materially_harmed_vs_raw": (
            not np.isfinite(raw_cmp["static_balanced_accuracy_delta"])
            or raw_cmp["static_balanced_accuracy_delta"] >= -args.max_static_balanced_drop
        ),
        "source_wins_when_supported": all(
            item["win_rate"] >= args.min_source_win_rate for item in supported_source_checks
        ),
    }
    passed = all(checks.values())
    summary = {
        "gate": "DataB-to-ViF camera-conditioned geometric residual gate",
        "status": "passed" if passed else "failed",
        "what_was_tested": (
            "Frozen DINOv2 appearance and RAFT motion use one weighted binary BCE probe. "
            "Camera labels are used only for stratification. The primary geometry residual is compared "
            "with equal-dimensional raw-motion and cyclic wrong-camera controls on the same ViF frames."
        ),
        "data_contract": {
            "dataa_used": False,
            "detection_cot_used_as_target": False,
            "camera_text_is_classifier_input": False,
            "train_manifest": str(args.train_manifest),
            "test_manifest": str(args.test_manifest),
            "threshold_source": "held-out DataB validation only",
        },
        "thresholds": {
            "min_coverage": args.min_coverage,
            "min_auc_gain": args.min_auc_gain,
            "min_motion_macro_balanced_gain": args.min_motion_macro_balanced_gain,
            "max_static_balanced_drop": args.max_static_balanced_drop,
            "min_source_win_rate": args.min_source_win_rate,
        },
        "checks": checks,
        "models": results,
        "comparisons": comparisons,
        "does_not_establish": (
            "A pass establishes an independent frozen-feature camera-geometry signal, not an MLLM gain, "
            "CoT quality, DataA localization quality, or final-test generalization."
        ),
        "next_action": (
            "If passed, inject the frozen geometry block through a small projector into the shared detector and "
            "train with the original detection SFT loss only. If failed, stop this camera-geometry route before MLLM training."
        ),
    }
    write_json(args.output_dir / "camera_geometric_residual_gate_summary.json", summary)
    csv_path = args.output_dir / "camera_geometric_residual_gate_predictions.csv"
    fieldnames = [
        "sample_id",
        "answer",
        "motion_bucket",
        "source_name",
        *VARIANTS,
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id in sorted(prediction_rows):
            writer.writerow(prediction_rows[sample_id])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
