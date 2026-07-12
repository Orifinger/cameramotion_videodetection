#!/usr/bin/env python3
"""Evaluate balanced binary camera VQA scores and paired question accuracy."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.camera_binary_vqa.runtime import read_jsonl, write_json


def named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("value must be NAME=PATH")
    return name, path


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = fp = 0
    previous_recall = 0.0
    ap = 0.0
    cursor = 0
    while cursor < len(ordered):
        score = ordered[cursor][0]
        group_labels: list[int] = []
        while cursor < len(ordered) and ordered[cursor][0] == score:
            group_labels.append(ordered[cursor][1])
            cursor += 1
        tp += sum(group_labels)
        fp += len(group_labels) - sum(group_labels)
        recall = tp / positives
        precision = tp / (tp + fp)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[ordered[position][0]] = average_rank
        cursor = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [int(row["answer_id"]) for row in rows]
    scores = [float(row["yes_minus_no_score"]) for row in rows]
    predictions = [1 if score >= 0.0 else 0 for score in scores]
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    positive_recall = tp / (tp + fn) if tp + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    return {
        "num_samples": len(rows),
        "positive_samples": sum(labels),
        "negative_samples": len(labels) - sum(labels),
        "accuracy": (tp + tn) / len(rows) if rows else 0.0,
        "balanced_accuracy": (positive_recall + negative_recall) / 2.0,
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "average_precision": average_precision(labels, scores),
        "roc_auc": roc_auc(labels, scores),
        "mean_yes_score": (
            sum(score for score, label in zip(scores, labels) if label == 1) / sum(labels)
            if sum(labels)
            else 0.0
        ),
        "mean_no_score": (
            sum(score for score, label in zip(scores, labels) if label == 0)
            / (len(labels) - sum(labels))
            if len(labels) - sum(labels)
            else 0.0
        ),
        "confusion": {"yes_as_yes": tp, "yes_as_no": fn, "no_as_no": tn, "no_as_yes": fp},
    }


def evaluate_condition(
    gold: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    prediction_by_id: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for row in predictions:
        sample_id = str(row.get("sample_id") or "")
        if sample_id in prediction_by_id:
            duplicates.append(sample_id)
        prediction_by_id[sample_id] = row
    if duplicates:
        raise ValueError(f"duplicate prediction sample ids: {sorted(set(duplicates))[:20]}")
    merged: list[dict[str, Any]] = []
    missing: list[str] = []
    for gold_row in gold:
        sample_id = str(gold_row.get("sample_id") or "")
        prediction = prediction_by_id.get(sample_id)
        if prediction is None:
            missing.append(sample_id)
            continue
        if int(prediction.get("answer_id")) != int(gold_row.get("answer_id")):
            raise ValueError(f"prediction/gold answer mismatch for {sample_id}")
        if str(prediction.get("camera_primitive")) != str(gold_row.get("camera_primitive")):
            raise ValueError(f"prediction/gold primitive mismatch for {sample_id}")
        merged.append({**dict(gold_row), **dict(prediction)})
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merged:
        by_label[str(row["camera_primitive"])].append(row)
    per_label = {label: binary_metrics(rows) for label, rows in sorted(by_label.items())}
    macro_keys = ("average_precision", "roc_auc", "balanced_accuracy")
    macro = {
        key: sum(metrics[key] for metrics in per_label.values()) / len(per_label)
        if per_label
        else 0.0
        for key in macro_keys
    }
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merged:
        pairs[str(row["pair_id"])].append(row)
    valid_pairs = [rows for rows in pairs.values() if len(rows) == 2]
    pair_correct = sum(
        all((float(row["yes_minus_no_score"]) >= 0.0) == bool(int(row["answer_id"])) for row in rows)
        for rows in valid_pairs
    )
    return {
        "num_gold": len(gold),
        "num_predictions": len(predictions),
        "num_matched": len(merged),
        "coverage": len(merged) / len(gold) if gold else 0.0,
        "missing_sample_ids": missing,
        "duplicate_prediction_ids": sorted(set(duplicates)),
        "num_supported_labels": len(per_label),
        "overall": binary_metrics(merged),
        "macro": macro,
        "paired_question_accuracy": pair_correct / len(valid_pairs) if valid_pairs else 0.0,
        "num_valid_pairs": len(valid_pairs),
        "per_label": per_label,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", action="append", type=named_path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--model-stage", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_rows: list[dict[str, Any]] = []
    for path in sorted(args.predictions_dir.glob("rank_*.jsonl")):
        prediction_rows.extend(read_jsonl(path))
    if not prediction_rows:
        raise ValueError(f"no rank prediction files found in {args.predictions_dir}")
    output: dict[str, Any] = {
        "schema_version": "dataa_camera_binary_vqa_evaluation_v1",
        "model_stage": args.model_stage,
        "predictions_dir": str(args.predictions_dir),
        "conditions": {},
    }
    for condition_name, gold_path in args.gold:
        gold = read_jsonl(gold_path)
        selected = [row for row in prediction_rows if row.get("condition") == condition_name]
        output["conditions"][condition_name] = evaluate_condition(gold, selected)
    write_json(args.output_json, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
