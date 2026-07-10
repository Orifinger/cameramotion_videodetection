#!/usr/bin/env python3
"""Evaluate predictions for camera_pretext_eval.json."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.camera_detection_rewards import (  # noqa: E402
    CAMERA_LABEL_ORDER,
    camera_exact_match,
    camera_format_valid,
    camera_set_f1,
    normalize_truth_labels,
    parse_camera_completion,
)

DATAA_CASE_RE = re.compile(r"(dataA_v1(?:_[A-Za-z][A-Za-z0-9]*)*_\d+)")
RESPONSE_KEYS = ("response", "prediction", "raw_response", "completion", "generated_text", "text")


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def records_from_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        payload = read_json(path)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, Mapping) and isinstance(payload.get("predictions"), list):
            rows = payload["predictions"]
        elif isinstance(payload, Mapping):
            rows = [payload]
        else:
            rows = []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def prediction_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for candidate in sorted(path.rglob("*")):
        folded = candidate.name.casefold()
        if (candidate.is_file() and candidate.suffix.lower() in {".json", ".jsonl"}
                and "summary" not in folded and "metrics" not in folded):
            yield candidate


def load_predictions(path: Path) -> list[dict[str, Any]]:
    rows = []
    for file_path in prediction_files(path):
        for row in records_from_file(file_path):
            row["_source_file"] = str(file_path)
            rows.append(row)
    return rows


def first_image(record: Mapping[str, Any]) -> str:
    images = record.get("images")
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes)) and images:
        return str(images[0]).replace("\\", "/")
    return ""


def infer_sample_id(record: Mapping[str, Any], index: int) -> str:
    direct = str(record.get("sample_id", "")).strip()
    if direct:
        return direct
    image = first_image(record)
    match = DATAA_CASE_RE.search(image)
    case_id = match.group(1) if match else str(record.get("case_id", "")).strip()
    split = PurePosixPath(image).parent.name.casefold() if image else ""
    if case_id and split in {"real", "fake"}:
        return f"{case_id}:{split}"
    return case_id or f"prediction_{index:06d}"


def assistant_response(record: Mapping[str, Any] | None) -> str:
    if record is None:
        return ""
    for key in RESPONSE_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, Mapping):
            for nested in ("content", "text"):
                if isinstance(value.get(nested), str):
                    return str(value[nested])
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                return str(message.get("content", ""))
    return ""


def load_gt(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list: {path}")
    rows, seen = [], set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ValueError(f"non-object GT record at index {index}")
        row = dict(raw)
        sid = infer_sample_id(row, index)
        if sid in seen:
            raise ValueError(f"duplicate GT sample_id: {sid}")
        labels = normalize_truth_labels(row.get("camera_labels", []))
        if not labels:
            raise ValueError(f"empty camera_labels for GT sample: {sid}")
        seen.add(sid)
        row.update(sample_id=sid, camera_labels=labels)
        rows.append(row)
    return rows


def build_items(gt_rows, predictions):
    pred_by_id = {}
    duplicates = 0
    for index, prediction in enumerate(predictions):
        sid = infer_sample_id(prediction, index)
        duplicates += int(sid in pred_by_id)
        pred_by_id[sid] = prediction
    gt_ids = {str(row["sample_id"]) for row in gt_rows}
    unmatched = sum(sid not in gt_ids for sid in pred_by_id)
    items = []
    for gt in gt_rows:
        sid = str(gt["sample_id"])
        pred = pred_by_id.get(sid)
        response = assistant_response(pred)
        parsed = parse_camera_completion(response)
        truth = list(gt["camera_labels"])
        items.append({
            "sample_id": sid,
            "motion_bucket": str(gt.get("motion_bucket", "unknown")),
            "truth_labels": truth,
            "pred_labels": list(parsed.labels),
            "unknown_labels": list(parsed.unknown),
            "format_valid": bool(camera_format_valid(response)),
            "exact_match": bool(camera_exact_match(response, truth)),
            "set_f1": camera_set_f1(response, truth),
            "missing_prediction": pred is None,
            "source_file": str(pred.get("_source_file", "")) if pred else "",
        })
    return items, unmatched, duplicates


def aggregate_metrics(items):
    total = len(items)
    stats = {label: Counter() for label in CAMERA_LABEL_ORDER}
    tp_total = fp_total = fn_total = unknown_total = predicted_total = 0
    for item in items:
        truth, predicted = set(item["truth_labels"]), set(item["pred_labels"])
        unknown = list(item["unknown_labels"])
        unknown_total += len(unknown)
        predicted_total += len(predicted) + len(unknown)
        tp_total += len(truth & predicted)
        fp_total += len(predicted - truth) + len(unknown)
        fn_total += len(truth - predicted)
        for label in CAMERA_LABEL_ORDER:
            if label in truth and label in predicted:
                stats[label]["tp"] += 1
            elif label not in truth and label in predicted:
                stats[label]["fp"] += 1
            elif label in truth and label not in predicted:
                stats[label]["fn"] += 1
    per_label, supported_f1 = {}, []
    for label, counts in stats.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        support = tp + fn
        if support == 0 and fp == 0:
            continue
        precision, recall = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        if support:
            supported_f1.append(f1)
        per_label[label] = dict(support=support, tp=tp, fp=fp, fn=fn,
                                precision=precision, recall=recall, f1=f1)
    micro_precision = safe_div(tp_total, tp_total + fp_total)
    micro_recall = safe_div(tp_total, tp_total + fn_total)
    return {
        "num_samples": total,
        "num_missing_predictions": sum(bool(item["missing_prediction"]) for item in items),
        "format_valid_rate": safe_div(sum(bool(item["format_valid"]) for item in items), total),
        "exact_set_accuracy": safe_div(sum(bool(item["exact_match"]) for item in items), total),
        "mean_sample_set_f1": safe_div(sum(float(item["set_f1"]) for item in items), total),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall),
        "macro_f1_gt_supported_labels": safe_div(sum(supported_f1), len(supported_f1)),
        "unknown_label_rate": safe_div(unknown_total, predicted_total),
        "unknown_label_count": unknown_total,
        "per_label": per_label,
    }


def write_items(path: Path, items) -> None:
    fields = ["sample_id", "motion_bucket", "truth_labels", "pred_labels", "unknown_labels",
              "format_valid", "exact_match", "set_f1", "missing_prediction", "source_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            row = dict(item)
            for key in ("truth_labels", "pred_labels", "unknown_labels"):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow({key: row.get(key, "") for key in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--output-prefix", default="camera_motion_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path, pred_path = Path(args.gt_json), Path(args.pred_json)
    gt_rows, predictions = load_gt(gt_path), load_predictions(pred_path)
    items, unmatched, duplicates = build_items(gt_rows, predictions)
    overall = aggregate_metrics(items)
    buckets = sorted({str(item["motion_bucket"]) for item in items})
    summary = {
        "gt_json": str(gt_path), "pred_json": str(pred_path),
        "num_gt_records": len(gt_rows), "num_prediction_records_loaded": len(predictions),
        "num_unmatched_predictions": unmatched, "num_duplicate_prediction_ids": duplicates,
        "overall": overall,
        "by_motion_bucket": {bucket: aggregate_metrics(
            [item for item in items if item["motion_bucket"] == bucket]) for bucket in buckets},
    }
    out_dir = Path(args.out_dir) if args.out_dir else (pred_path if pred_path.is_dir() else pred_path.parent)
    summary_path = out_dir / f"{args.output_prefix}_summary.json"
    items_path = out_dir / f"{args.output_prefix}_items.csv"
    write_json(summary_path, summary)
    write_items(items_path, items)
    print("=== Camera Motion Evaluation ===")
    print(f"GT records: {len(gt_rows)}")
    print(f"Predictions loaded: {len(predictions)}")
    print(f"Missing predictions: {overall['num_missing_predictions']}")
    print(f"Format valid rate: {overall['format_valid_rate'] * 100:.2f}%")
    print(f"Exact set accuracy: {overall['exact_set_accuracy'] * 100:.2f}%")
    print(f"Micro F1: {overall['micro_f1'] * 100:.2f}%")
    print(f"Mean sample F1: {overall['mean_sample_set_f1'] * 100:.2f}%")
    print(f"Saved summary: {summary_path}")
    print(f"Saved items: {items_path}")


if __name__ == "__main__":
    main()
