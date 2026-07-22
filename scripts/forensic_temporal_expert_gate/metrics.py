"""Metrics shared by the temporal-expert training and evaluation steps."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def best_balanced_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(np.concatenate([[0.0], scores, [1.0]]))
    if candidates.size > 4096:
        candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 4096))
    best = (float("-inf"), 0.5)
    for threshold in candidates:
        predictions = (scores >= threshold).astype(np.int64)
        real = labels == 0
        fake = labels == 1
        value = 0.5 * (
            float((predictions[real] == 0).mean())
            + float((predictions[fake] == 1).mean())
        )
        candidate = (value, -abs(float(threshold) - 0.5))
        current = (best[0], -abs(best[1] - 0.5))
        if candidate > current:
            best = (value, float(threshold))
    return best[1]


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    generators: Sequence[str] | None = None,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    real = labels == 0
    fake = labels == 1
    real_recall = float((predictions[real] == 0).mean()) if real.any() else None
    fake_recall = float((predictions[fake] == 1).mean()) if fake.any() else None
    true_fake = int(((predictions == 1) & fake).sum())
    predicted_fake = int((predictions == 1).sum())
    precision = true_fake / predicted_fake if predicted_fake else 0.0
    fake_f1 = (
        2.0 * precision * fake_recall / (precision + fake_recall)
        if fake_recall is not None and precision + fake_recall
        else 0.0
    )
    result: dict[str, Any] = {
        "num_samples": int(labels.size),
        "threshold": float(threshold),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": (
            0.5 * (real_recall + fake_recall)
            if real_recall is not None and fake_recall is not None
            else None
        ),
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "fake_precision": float(precision),
        "fake_f1": float(fake_f1),
        "predicted_fake_rate": float(predictions.mean()),
        "roc_auc": (
            float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else None
        ),
        "average_precision": (
            float(average_precision_score(labels, scores))
            if np.unique(labels).size == 2
            else None
        ),
    }
    if generators is not None and real_recall is not None:
        values = np.asarray([str(value) for value in generators], dtype=object)
        by_generator: dict[str, float] = {}
        for generator in sorted(set(values[fake])):
            mask = fake & (values == generator)
            fake_generator_recall = float((predictions[mask] == 1).mean())
            by_generator[generator] = 0.5 * (real_recall + fake_generator_recall)
        result["generator_macro_balanced_accuracy"] = (
            float(np.mean(list(by_generator.values()))) if by_generator else None
        )
        result["per_fake_generator_balanced_accuracy"] = by_generator
    return result


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    output = np.empty_like(values)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def logit(values: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(values / (1.0 - values))
