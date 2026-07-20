"""Pure evaluation-control helpers (no model runtime imports)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.camera_ctne_gate1.contracts import stable_unit


def shuffled_donor_indices(rows: Sequence[Mapping[str, Any]], seed: int) -> tuple[list[int], dict[str, int]]:
    levels = (
        ("dataset_name", "source_name", "motion_bucket", "frame_count_bin"),
        ("dataset_name", "motion_bucket", "frame_count_bin"),
        ("dataset_name", "frame_count_bin"),
        ("dataset_name",),
    )
    maps: list[dict[tuple[str, ...], list[int]]] = []
    for fields in levels:
        groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            groups[tuple(str(row.get(field, "unknown")) for field in fields)].append(index)
        maps.append(groups)
    donors: list[int] = []
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        candidates: list[int] = []
        level_name = "unavailable"
        for fields, groups in zip(levels, maps):
            key = tuple(str(row.get(field, "unknown")) for field in fields)
            candidates = [candidate for candidate in groups[key] if candidate != index]
            if candidates:
                level_name = "+".join(fields)
                break
        if not candidates:
            raise ValueError(f"no non-self shuffled camera donor for {row.get('sample_id')}")
        candidates.sort(
            key=lambda candidate: stable_unit(
                f"{row.get('sample_id')}->{rows[candidate].get('sample_id')}",
                seed,
            )
        )
        donors.append(candidates[0])
        counts[level_name] += 1
    if any(index == donor for index, donor in enumerate(donors)):
        raise AssertionError("shuffled camera control assigned a sample to itself")
    return donors, dict(counts)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    fake_recall = tp / (tp + fn) if tp + fn else 0.0
    real_recall = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * fake_recall / (precision + fake_recall) if precision + fake_recall else 0.0
    return {
        "num_samples": int(labels.size),
        "roc_auc": float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float("nan"),
        "average_precision": float(average_precision_score(labels, scores)) if np.unique(labels).size == 2 else float("nan"),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float((fake_recall + real_recall) / 2.0),
        "fake_recall": float(fake_recall),
        "real_recall": float(real_recall),
        "fake_f1": float(f1),
        "predicted_fake_rate": float(predictions.mean()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def best_balanced_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    from sklearn.metrics import roc_curve

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    false_positive_rate, true_positive_rate, candidates = roc_curve(labels, scores)
    balanced = 0.5 * (true_positive_rate + 1.0 - false_positive_rate)
    best = np.flatnonzero(balanced == balanced.max())
    if best.size > 1:
        fake_rate = np.asarray([(scores >= candidates[index]).mean() for index in best])
        chosen = int(best[np.argmin(np.abs(fake_rate - labels.mean()))])
    else:
        chosen = int(best[0])
    threshold = float(candidates[chosen])
    if not np.isfinite(threshold):
        threshold = float(np.nextafter(scores.max(), np.inf))
    return threshold, binary_metrics(labels, scores, threshold)
