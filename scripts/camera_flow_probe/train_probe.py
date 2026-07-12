#!/usr/bin/env python3
"""Train and evaluate the three frozen-feature camera-flow probe variants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_flow_probe.contracts import read_jsonl
from scripts.camera_flow_probe.metrics import (
    best_balanced_threshold,
    best_f1_threshold,
    binary_metrics,
    roc_auc,
)


VARIANTS = ("global", "local_unaligned", "local_aligned")


def _stable_unit(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _probe_network(input_dim: int, hidden_dims: Sequence[int]) -> nn.Module:
    layers: list[nn.Module] = []
    current = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(current, int(hidden)), nn.ReLU()])
        current = int(hidden)
    layers.append(nn.Linear(current, 1))
    return nn.Sequential(*layers)


@dataclass
class StandardizedProbe:
    mean: np.ndarray
    std: np.ndarray
    hidden_dims: tuple[int, ...]
    state_dict: dict[str, torch.Tensor]

    def predict(self, features: np.ndarray, *, device: torch.device, batch_size: int = 65536) -> np.ndarray:
        model = _probe_network(self.mean.shape[0], self.hidden_dims).to(device)
        model.load_state_dict(self.state_dict)
        model.eval()
        normalized = (np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0) - self.mean) / self.std
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, normalized.shape[0], batch_size):
                values = torch.from_numpy(normalized[start : start + batch_size].astype(np.float32)).to(device)
                outputs.append(torch.sigmoid(model(values).squeeze(1)).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32)


def train_probe_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    hidden_dims: Sequence[int],
) -> StandardizedProbe:
    torch.manual_seed(seed)
    mean = np.nanmean(train_x, axis=0).astype(np.float32)
    std = np.nanstd(train_x, axis=0).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0

    def normalize(values: np.ndarray) -> np.ndarray:
        return ((np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std).astype(np.float32)

    x = torch.from_numpy(normalize(train_x))
    y = torch.from_numpy(train_y.astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=min(batch_size, len(x)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    hidden_dims = tuple(int(value) for value in hidden_dims)
    model = _probe_network(train_x.shape[1], hidden_dims).to(device)
    positives = max(1, int((train_y == 1).sum()))
    negatives = max(1, int((train_y == 0).sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_auc = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    for _epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device)).squeeze(1)
            loss = loss_fn(logits, batch_y.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            values = torch.from_numpy(normalize(val_x)).to(device)
            val_scores = torch.sigmoid(model(values).squeeze(1)).cpu().numpy()
        auc = roc_auc(val_y, val_scores)
        selection = auc if np.isfinite(auc) else -1.0
        if selection > best_auc + 1e-5:
            best_auc = selection
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 6:
                break
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return StandardizedProbe(mean=mean, std=std, hidden_dims=hidden_dims, state_dict=best_state)


def _sample_indices(indices: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if indices.size <= limit:
        return indices
    return np.sort(rng.choice(indices, size=limit, replace=False))


def _load_feature(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _aggregate_top(scores: np.ndarray, fraction: float) -> float:
    valid = np.asarray(scores, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return float("nan")
    count = max(1, int(math.ceil(valid.size * fraction)))
    return float(np.partition(valid, valid.size - count)[-count:].mean())


def _build_global_training(
    rows: Sequence[Mapping[str, Any]],
    loaded: Mapping[str, dict[str, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    partitions: dict[str, list[tuple[np.ndarray, int]]] = defaultdict(list)
    for row in rows:
        case_id = str(row["case_id"])
        values = loaded[case_id]
        partition = "val" if _stable_unit(case_id, 20260712) < 0.15 else "train"
        for role, label in (("real", 0), ("fake", 1)):
            for feature in values[f"{role}_global"]:
                partitions[partition].append((feature.astype(np.float32), label))
    arrays = {}
    for partition in ("train", "val"):
        items = partitions[partition]
        arrays[partition] = (
            np.stack([item[0] for item in items]),
            np.asarray([item[1] for item in items], dtype=np.int64),
        )
    return arrays


def _build_local_training(
    rows: Sequence[Mapping[str, Any]],
    loaded: Mapping[str, dict[str, np.ndarray]],
    *,
    variant: str,
    max_patches_per_group: int,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    partitions: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    rng = np.random.default_rng(seed)
    suffix = "aligned" if variant == "local_aligned" else "unaligned"
    for row in rows:
        case_id = str(row["case_id"])
        values = loaded[case_id]
        partition = "val" if _stable_unit(case_id, 20260712) < 0.15 else "train"
        for role in ("real", "fake"):
            features = values[f"{role}_{variant}"].reshape(-1, values[f"{role}_{variant}"].shape[-1])
            valid = values[f"{role}_valid_{suffix}"].reshape(-1).astype(bool)
            labels = values[f"{role}_label_{suffix}"].reshape(-1).astype(np.int64)
            positive = _sample_indices(np.flatnonzero(valid & (labels == 1)), max_patches_per_group, rng)
            negative = _sample_indices(np.flatnonzero(valid & (labels == 0)), max_patches_per_group, rng)
            selected = np.concatenate([positive, negative])
            if selected.size:
                partitions[partition].append((features[selected].astype(np.float32), labels[selected]))
    output = {}
    for partition in ("train", "val"):
        values = partitions[partition]
        if not values:
            raise ValueError(f"no local {partition} patches for {variant}")
        output[partition] = (np.concatenate([item[0] for item in values]), np.concatenate([item[1] for item in values]))
    return output


def _has_effective_positive(values: Mapping[str, np.ndarray], suffix: str) -> bool:
    labels = values[f"fake_label_{suffix}"].reshape(-1).astype(bool)
    valid = values[f"fake_valid_{suffix}"].reshape(-1).astype(bool)
    return bool((labels & valid).any())


def _common_local_supervision_rows(
    rows: Sequence[Mapping[str, Any]],
    loaded: Mapping[str, dict[str, np.ndarray]],
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    eligible: list[Mapping[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row["case_id"])
        values = loaded[case_id]
        aligned = _has_effective_positive(values, "aligned")
        unaligned = _has_effective_positive(values, "unaligned")
        if aligned and unaligned:
            eligible.append(row)
        else:
            excluded.append(
                {
                    "case_id": case_id,
                    "dataset_split": row.get("dataset_split"),
                    "motion_bucket": row.get("motion_bucket"),
                    "source_name": row.get("source_name"),
                    "aligned_effective_positive": aligned,
                    "unaligned_effective_positive": unaligned,
                }
            )
    return eligible, excluded


def _video_predictions(
    rows: Sequence[Mapping[str, Any]],
    loaded: Mapping[str, dict[str, np.ndarray]],
    *,
    variant: str,
    probe: StandardizedProbe,
    device: torch.device,
    local_top_fraction: float,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    suffix = "aligned" if variant == "local_aligned" else "unaligned"
    for row in rows:
        case_id = str(row["case_id"])
        values = loaded[case_id]
        for role, label in (("real", 0), ("fake", 1)):
            if variant == "global":
                features = values[f"{role}_global"]
                scores = probe.predict(features, device=device)
                video_score = _aggregate_top(scores, 0.25)
            else:
                features = values[f"{role}_{variant}"].reshape(-1, values[f"{role}_{variant}"].shape[-1])
                valid = values[f"{role}_valid_{suffix}"].reshape(-1).astype(bool)
                scores = probe.predict(features[valid], device=device)
                video_score = _aggregate_top(scores, local_top_fraction)
            predictions.append(
                {
                    "case_id": case_id,
                    "role": role,
                    "label": label,
                    "score": video_score,
                    "motion_bucket": row.get("motion_bucket", "unknown"),
                    "source_name": row.get("source_name", ""),
                    "vace_model": row.get("vace_model", ""),
                }
            )
    return predictions


def _localization_metrics(
    rows: Sequence[Mapping[str, Any]],
    loaded: Mapping[str, dict[str, np.ndarray]],
    *,
    variant: str,
    probe: StandardizedProbe,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    suffix = "aligned" if variant == "local_aligned" else "unaligned"
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    intersections = unions = 0
    pointing_hits = pointing_total = 0
    for row in rows:
        values = loaded[str(row["case_id"])]
        features = values[f"fake_{variant}"].reshape(-1, values[f"fake_{variant}"].shape[-1])
        valid = values[f"fake_valid_{suffix}"].reshape(-1).astype(bool)
        labels = values[f"fake_label_{suffix}"].reshape(-1).astype(np.int64)
        scores = np.full(labels.shape, np.nan, dtype=np.float32)
        scores[valid] = probe.predict(features[valid], device=device)
        labels_all.append(labels[valid])
        scores_all.append(scores[valid])
        prediction = (scores >= threshold) & valid
        truth = (labels == 1) & valid
        intersections += int((prediction & truth).sum())
        unions += int((prediction | truth).sum())
        if truth.any() and np.isfinite(scores).any():
            pointing_total += 1
            pointing_hits += int(truth[int(np.nanargmax(scores))])
    labels_flat = np.concatenate(labels_all)
    scores_flat = np.concatenate(scores_all)
    return {
        "patch_auc": float(roc_auc(labels_flat, scores_flat)),
        "patch_iou": float(intersections / unions) if unions else 0.0,
        "pointing_game_accuracy": float(pointing_hits / pointing_total) if pointing_total else 0.0,
        "num_valid_patches": int(labels_flat.size),
        "num_positive_patches": int((labels_flat == 1).sum()),
        "threshold": float(threshold),
    }


def _metric_groups(predictions: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    output: dict[str, Any] = {}
    groups: dict[str, list[Mapping[str, Any]]] = {"overall": list(predictions)}
    buckets = sorted({str(item.get("motion_bucket", "unknown")) for item in predictions})
    for bucket in buckets:
        groups[f"motion:{bucket}"] = [item for item in predictions if item.get("motion_bucket") == bucket]
    for name, items in groups.items():
        labels = np.asarray([item["label"] for item in items], dtype=np.int64)
        scores = np.asarray([item["score"] for item in items], dtype=np.float64)
        metrics = binary_metrics(labels, scores, threshold)
        by_case: dict[str, dict[str, float]] = defaultdict(dict)
        for item in items:
            by_case[str(item["case_id"])][str(item["role"])] = float(item["score"])
        pairs = [values for values in by_case.values() if set(values) == {"real", "fake"}]
        metrics["pair_accuracy_fake_gt_real"] = (
            float(np.mean([values["fake"] > values["real"] for values in pairs])) if pairs else float("nan")
        )
        metrics["num_pairs"] = len(pairs)
        output[name] = metrics
    return output


def _bootstrap_auc_delta(
    aligned: Sequence[Mapping[str, Any]],
    unaligned: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    aligned_map = {(str(item["case_id"]), str(item["role"])): item for item in aligned}
    unaligned_map = {(str(item["case_id"]), str(item["role"])): item for item in unaligned}
    cases = sorted({case for case, _role in aligned_map} & {case for case, _role in unaligned_map})
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        sampled = [rng.choice(cases) for _case in cases]
        labels: list[int] = []
        aligned_scores: list[float] = []
        unaligned_scores: list[float] = []
        for case in sampled:
            for role, label in (("real", 0), ("fake", 1)):
                labels.append(label)
                aligned_scores.append(float(aligned_map[(case, role)]["score"]))
                unaligned_scores.append(float(unaligned_map[(case, role)]["score"]))
        delta = roc_auc(labels, aligned_scores) - roc_auc(labels, unaligned_scores)
        if np.isfinite(delta):
            deltas.append(float(delta))
    return {
        "iterations": len(deltas),
        "mean": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dims", default="64,32")
    parser.add_argument("--max-patches-per-group", type=int, default=256)
    parser.add_argument("--local-top-fraction", type=float, default=0.05)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--min-feature-coverage", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    rows = sorted(read_jsonl(args.manifest_jsonl), key=lambda value: str(value["case_id"]))
    available = [row for row in rows if (args.feature_dir / f"{row['case_id']}.npz").is_file()]
    coverage = len(available) / len(rows) if rows else 0.0
    if coverage < args.min_feature_coverage:
        print(
            f"feature coverage below threshold: {len(available)}/{len(rows)}={coverage:.4f} "
            f"required={args.min_feature_coverage:.4f}",
            file=sys.stderr,
        )
        return 2
    train_rows = [row for row in available if row.get("dataset_split") == "train"]
    test_rows = [row for row in available if row.get("dataset_split") == "test"]
    if not train_rows or not test_rows:
        print("both train and test feature sets are required", file=sys.stderr)
        return 2
    loaded = {
        str(row["case_id"]): _load_feature(args.feature_dir / f"{row['case_id']}.npz")
        for row in (*train_rows, *test_rows)
    }
    local_train_rows, local_train_excluded = _common_local_supervision_rows(train_rows, loaded)
    common_test_rows, common_test_excluded = _common_local_supervision_rows(test_rows, loaded)
    if not local_train_rows or not common_test_rows:
        print("no common aligned/unaligned local supervision rows", file=sys.stderr)
        return 2
    global_sets = _build_global_training(local_train_rows, loaded)

    try:
        hidden_dims = tuple(int(value.strip()) for value in args.hidden_dims.split(",") if value.strip())
    except ValueError:
        print(f"invalid --hidden-dims: {args.hidden_dims}", file=sys.stderr)
        return 2
    probes: dict[str, StandardizedProbe] = {}
    val_thresholds: dict[str, float] = {}
    patch_thresholds: dict[str, float] = {}
    predictions: dict[str, list[dict[str, Any]]] = {}
    localization: dict[str, dict[str, float]] = {}

    global_probe = train_probe_model(
        *global_sets["train"],
        *global_sets["val"],
        device=device,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        hidden_dims=hidden_dims,
    )
    probes["global"] = global_probe

    for variant in ("local_unaligned", "local_aligned"):
        local_sets = _build_local_training(
            local_train_rows,
            loaded,
            variant=variant,
            max_patches_per_group=args.max_patches_per_group,
            seed=args.seed,
        )
        probes[variant] = train_probe_model(
            *local_sets["train"],
            *local_sets["val"],
            device=device,
            seed=args.seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            hidden_dims=hidden_dims,
        )
        val_scores = probes[variant].predict(local_sets["val"][0], device=device)
        patch_thresholds[variant] = best_f1_threshold(local_sets["val"][1], val_scores)

    common_val_rows = [row for row in local_train_rows if _stable_unit(str(row["case_id"]), 20260712) < 0.15]
    full_test_predictions: dict[str, list[dict[str, Any]]] = {}
    for variant in VARIANTS:
        val_predictions = _video_predictions(
            common_val_rows,
            loaded,
            variant=variant,
            probe=probes[variant],
            device=device,
            local_top_fraction=args.local_top_fraction,
        )
        val_thresholds[variant] = best_balanced_threshold(
            np.asarray([item["label"] for item in val_predictions]),
            np.asarray([item["score"] for item in val_predictions]),
        )
        predictions[variant] = _video_predictions(
            common_test_rows,
            loaded,
            variant=variant,
            probe=probes[variant],
            device=device,
            local_top_fraction=args.local_top_fraction,
        )
        full_test_predictions[variant] = _video_predictions(
            test_rows,
            loaded,
            variant=variant,
            probe=probes[variant],
            device=device,
            local_top_fraction=args.local_top_fraction,
        )
        if variant != "global":
            localization[variant] = _localization_metrics(
                common_test_rows,
                loaded,
                variant=variant,
                probe=probes[variant],
                device=device,
                threshold=patch_thresholds[variant],
            )

    metrics = {
        variant: _metric_groups(predictions[variant], val_thresholds[variant])
        for variant in VARIANTS
    }
    full_test_sensitivity_metrics = {
        variant: _metric_groups(full_test_predictions[variant], val_thresholds[variant])
        for variant in VARIANTS
    }
    aligned_auc = metrics["local_aligned"]["overall"]["auc"]
    unaligned_auc = metrics["local_unaligned"]["overall"]["auc"]
    aligned_complex = metrics["local_aligned"].get("motion:complex-motion", {}).get("auc", float("nan"))
    unaligned_complex = metrics["local_unaligned"].get("motion:complex-motion", {}).get("auc", float("nan"))
    aligned_static = metrics["local_aligned"].get("motion:no-motion", {}).get("auc", float("nan"))
    unaligned_static = metrics["local_unaligned"].get("motion:no-motion", {}).get("auc", float("nan"))
    bootstrap = _bootstrap_auc_delta(
        predictions["local_aligned"],
        predictions["local_unaligned"],
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    checks = {
        "feature_coverage": coverage >= args.min_feature_coverage,
        "common_local_train_coverage": len(local_train_rows) / len(train_rows) >= 0.95,
        "common_test_coverage": len(common_test_rows) / len(test_rows) >= 0.95,
        "overall_auc_gain_at_least_3_points": aligned_auc - unaligned_auc >= 0.03,
        "complex_motion_auc_gain_positive": (
            bool(np.isfinite(aligned_complex) and np.isfinite(unaligned_complex) and aligned_complex - unaligned_complex >= 0.03)
        ),
        "no_motion_auc_drop_at_most_2_points": (
            bool(np.isfinite(aligned_static) and np.isfinite(unaligned_static) and aligned_static - unaligned_static >= -0.02)
        ),
        "bootstrap_ci_excludes_zero": bootstrap["ci95_low"] > 0.0,
    }
    if all(checks.values()):
        gate_status = "passed"
    elif aligned_auc <= unaligned_auc or (
        np.isfinite(aligned_complex) and np.isfinite(unaligned_complex) and aligned_complex <= unaligned_complex
    ):
        gate_status = "failed"
    else:
        gate_status = "inconclusive"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "gate": "camera-aligned local trajectory probe",
        "status": gate_status,
        "manifest_jsonl": str(args.manifest_jsonl),
        "feature_dir": str(args.feature_dir),
        "feature_coverage": coverage,
        "num_manifest_cases": len(rows),
        "num_feature_cases": len(available),
        "num_train_cases": len(train_rows),
        "num_test_cases": len(test_rows),
        "local_supervision": {
            "num_common_train_cases": len(local_train_rows),
            "num_excluded_train_cases": len(local_train_excluded),
            "excluded_train_cases": local_train_excluded,
            "num_full_test_cases": len(test_rows),
            "num_common_primary_test_cases": len(common_test_rows),
            "common_primary_test_coverage": len(common_test_rows) / len(test_rows),
            "num_excluded_primary_test_cases": len(common_test_excluded),
            "excluded_primary_test_cases": common_test_excluded,
        },
        "evaluation_protocol": {
            "primary_train_scope": "common effective local supervision cases for all three probes",
            "primary_test_scope": "common effective local supervision cases for all video and localization metrics",
            "full_test_scope": "all held-out cases, reported as sensitivity only",
        },
        "validation_thresholds": val_thresholds,
        "patch_thresholds": patch_thresholds,
        "metrics": metrics,
        "full_test_sensitivity_metrics": full_test_sensitivity_metrics,
        "localization": localization,
        "aligned_minus_unaligned": {
            "overall_auc": aligned_auc - unaligned_auc,
            "complex_motion_auc": aligned_complex - unaligned_complex,
            "no_motion_auc": aligned_static - unaligned_static,
            "bootstrap_auc_delta": bootstrap,
        },
        "checks": checks,
    }
    (args.output_dir / "camera_aligned_local_probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "camera_aligned_local_probe_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = ["variant", "case_id", "role", "label", "score", "motion_bucket", "source_name", "vace_model"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant in VARIANTS:
            for item in predictions[variant]:
                writer.writerow({"variant": variant, **item})
    with (args.output_dir / "camera_aligned_local_probe_full_test_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = ["variant", "case_id", "role", "label", "score", "motion_bucket", "source_name", "vace_model"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant in VARIANTS:
            for item in full_test_predictions[variant]:
                writer.writerow({"variant": variant, **item})
    for variant, probe in probes.items():
        torch.save(
            {
                "mean": probe.mean,
                "std": probe.std,
                "hidden_dims": probe.hidden_dims,
                "state_dict": probe.state_dict,
            },
            args.output_dir / f"{variant}_mlp_probe.pt",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
