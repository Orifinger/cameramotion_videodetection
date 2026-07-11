"""Dependency-light binary metrics for the camera-flow probe."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def roc_auc(labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(s) & np.isin(y, [0, 1])
    y, s = y[valid], s[valid]
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.shape[0], dtype=np.float64)
    index = 0
    while index < order.size:
        end = index + 1
        while end < order.size and s[order[end]] == s[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = (np.asarray(scores) >= threshold).astype(np.int64)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if precision + tpr else 0.0
    return {
        "num_samples": int(labels.size),
        "auc": float(roc_auc(labels, scores)),
        "accuracy": float((labels == predictions).mean()) if labels.size else float("nan"),
        "balanced_accuracy": float((tpr + tnr) / 2.0),
        "fake_recall": float(tpr),
        "real_recall": float(tnr),
        "fake_f1": float(f1),
        "threshold": float(threshold),
    }


def best_balanced_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([[0.0], np.asarray(scores), [1.0]]))
    if candidates.size > 2001:
        candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 2001))
    best = max(
        (
            binary_metrics(labels, scores, float(threshold))["balanced_accuracy"],
            -abs(float(threshold) - 0.5),
            float(threshold),
        )
        for threshold in candidates
    )
    return best[2]


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([[0.0], np.asarray(scores), [1.0]]))
    if candidates.size > 2001:
        candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 2001))
    best = max(
        (
            binary_metrics(labels, scores, float(threshold))["fake_f1"],
            -abs(float(threshold) - 0.5),
            float(threshold),
        )
        for threshold in candidates
    )
    return best[2]
