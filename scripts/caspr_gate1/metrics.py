from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(int(label == 1) for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][0] == ordered[position][0]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        rank_sum += average_rank * sum(int(label == 1) for _, label in ordered[position:end])
        position = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def aggregate_pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels: list[int] = []
    scores: list[float] = []
    real_correct = fake_correct = pair_correct = 0
    margins: list[float] = []
    for row in rows:
        real_score, fake_score = float(row["real_score"]), float(row["fake_score"])
        labels.extend([0, 1])
        scores.extend([real_score, fake_score])
        real_correct += int(real_score <= 0)
        fake_correct += int(fake_score > 0)
        pair_correct += int(fake_score > real_score)
        margins.append(fake_score - real_score)
    count = len(rows)
    real_recall = safe_div(real_correct, count)
    fake_recall = safe_div(fake_correct, count)
    return {
        "num_pairs": count,
        "num_videos": count * 2,
        "auc": binary_auc(labels, scores),
        "balanced_accuracy_at_zero": (real_recall + fake_recall) / 2.0,
        "real_recall_at_zero": real_recall,
        "fake_recall_at_zero": fake_recall,
        "pair_accuracy_fake_gt_real": safe_div(pair_correct, count),
        "mean_pair_margin": safe_div(sum(margins), count),
        "median_pair_margin": sorted(margins)[count // 2] if count else float("nan"),
    }


def grouped_metrics(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    return {name: aggregate_pairs(group) for name, group in sorted(groups.items())}


def paired_bootstrap_auc_delta(
    control: Sequence[Mapping[str, Any]], method: Sequence[Mapping[str, Any]], repeats: int, seed: int
) -> dict[str, Any]:
    control_by_id = {str(row["pair_id"]): row for row in control}
    method_by_id = {str(row["pair_id"]): row for row in method}
    pair_ids = sorted(control_by_id.keys() & method_by_id.keys())
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(repeats):
        sampled = [pair_ids[rng.randrange(len(pair_ids))] for _ in pair_ids]
        control_rows = [control_by_id[pair_id] for pair_id in sampled]
        method_rows = [method_by_id[pair_id] for pair_id in sampled]
        delta = aggregate_pairs(method_rows)["auc"] - aggregate_pairs(control_rows)["auc"]
        if math.isfinite(delta):
            deltas.append(delta)
    deltas.sort()
    if not deltas:
        return {"repeats": 0, "mean": float("nan"), "ci95": [float("nan"), float("nan")]}
    lower = deltas[max(0, round(0.025 * (len(deltas) - 1)))]
    upper = deltas[min(len(deltas) - 1, round(0.975 * (len(deltas) - 1)))]
    return {"repeats": len(deltas), "mean": sum(deltas) / len(deltas), "ci95": [lower, upper]}
