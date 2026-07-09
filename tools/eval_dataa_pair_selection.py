#!/usr/bin/env python3
"""Evaluate DataA A/B local-edit selection predictions.

Ground truth is produced by tools/build_dataa_pair_region_pretext.py with
--task pair. Predictions can be one JSON file or a directory of rank shard JSON
files produced by v4train/eval/infer_dataa.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CASE_RE = re.compile(r"(dataA_v1(?:_[A-Za-z][A-Za-z0-9]*)*_\d+)")
EDITED_TAG_RE = re.compile(r"<edited_video>\s*([AB])\s*</edited_video>", re.IGNORECASE | re.DOTALL)
ANSWER_TAG_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE | re.DOTALL)
LABELS_RE = re.compile(r"<labels>\s*(.*?)\s*</labels>", re.IGNORECASE | re.DOTALL)
TYPE_RE = re.compile(r"<artifact_type>\s*(.*?)\s*</artifact_type>", re.IGNORECASE | re.DOTALL)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def get_assistant_content(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def parse_choice(text: Any) -> tuple[str, str]:
    text = str(text or "").strip()
    for pattern, source in (
        (EDITED_TAG_RE, "edited_video_tag"),
        (ANSWER_TAG_RE, "answer_tag"),
        (re.compile(r"\bedited\s+video\s*(?:is|:)?\s*([AB])\b", re.IGNORECASE), "edited_video_phrase"),
        (re.compile(r"\bvideo\s*([AB])\b", re.IGNORECASE), "video_phrase"),
    ):
        match = pattern.search(text)
        if match:
            return match.group(1).upper(), source
    if re.fullmatch(r"[ABab]", text):
        return text.upper(), "single_letter"
    return "UNKNOWN", "missing"


def parse_case_id(record: Mapping[str, Any], data_index: int) -> str:
    case_id = record.get("case_id")
    if case_id:
        return str(case_id)
    images = record.get("images")
    first_image = str(images[0]).replace("\\", "/") if isinstance(images, list) and images else ""
    match = CASE_RE.search(first_image)
    return match.group(1) if match else f"sample_{data_index:06d}"


def case_family(case_id: str) -> str:
    if case_id.startswith("dataA_v1_dataset_v2_"):
        return "dataset_v2"
    if case_id.startswith("dataA_v1_textedit_reserve_"):
        return "textedit_reserve"
    if case_id.startswith("dataA_v1_"):
        return "dataA_v1"
    return "unknown"


def parse_camera_labels(record: Mapping[str, Any], assistant: str) -> list[str]:
    labels = record.get("camera_labels")
    if isinstance(labels, list):
        return [str(item) for item in labels]
    match = LABELS_RE.search(assistant)
    if not match:
        return []
    raw = match.group(1).strip()
    if not raw or raw.lower() == "unknown":
        return []
    return [part.strip() for part in raw.split(";") if part.strip()]


def camera_bucket(labels: Sequence[str]) -> str:
    text = " ".join(labels).lower()
    if not text:
        return "unknown"
    if "static" in text:
        return "static"
    if "minor" in text:
        return "minor-motion"
    return "complex-motion"


def parse_artifact_type(record: Mapping[str, Any], assistant: str) -> str:
    if record.get("artifact_type"):
        return str(record.get("artifact_type"))
    match = TYPE_RE.search(assistant)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def build_gt_rows(path: str | Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")

    rows: list[dict[str, Any]] = []
    for data_index, record in enumerate(data):
        if not isinstance(record, Mapping):
            continue
        assistant = get_assistant_content(record)
        gt_choice, gt_source = parse_choice(record.get("edited_video") or assistant)
        case_id = parse_case_id(record, data_index)
        labels = parse_camera_labels(record, assistant)
        rows.append(
            {
                "data_index": data_index,
                "case_id": case_id,
                "family": case_family(case_id),
                "gt": gt_choice,
                "gt_parse_source": gt_source,
                "camera_labels": labels,
                "camera_bucket": camera_bucket(labels),
                "artifact_type": parse_artifact_type(record, assistant),
                "video_a_source_split": record.get("video_a_source_split", ""),
                "video_b_source_split": record.get("video_b_source_split", ""),
            }
        )
    return rows


def iter_prediction_files(path: str | Path) -> Iterable[Path]:
    path = Path(path)
    if path.is_file():
        yield path
        return
    for candidate in sorted(path.rglob("*.json")):
        name = candidate.name.lower()
        if "summary" in name or "metrics" in name:
            continue
        yield candidate


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for file_path in iter_prediction_files(path):
        payload = load_json(file_path)
        rows = payload.get("predictions") if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping):
                item = dict(row)
                item["_source_file"] = str(file_path)
                predictions.append(item)
    return predictions


def prediction_case_id(pred: Mapping[str, Any]) -> str:
    if pred.get("case_id"):
        return str(pred.get("case_id"))
    images = pred.get("images")
    first_image = str(images[0]).replace("\\", "/") if isinstance(images, list) and images else ""
    match = CASE_RE.search(first_image)
    return match.group(1) if match else ""


def build_eval_items(gt_rows: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    gt_by_index = {int(row["data_index"]): row for row in gt_rows}
    gt_by_case = {str(row["case_id"]): row for row in gt_rows}
    used_indices: set[int] = set()
    unmatched_predictions = 0
    items: list[dict[str, Any]] = []

    for pred in predictions:
        gt_row = None
        if isinstance(pred.get("data_index"), int):
            gt_row = gt_by_index.get(int(pred["data_index"]))
        if gt_row is None:
            gt_row = gt_by_case.get(prediction_case_id(pred))
        if gt_row is None:
            unmatched_predictions += 1
            continue

        response = str(pred.get("response", pred.get("prediction", pred.get("raw_response", ""))))
        pred_choice, pred_source = parse_choice(response)
        data_index = int(gt_row["data_index"])
        used_indices.add(data_index)
        items.append(
            {
                "data_index": data_index,
                "case_id": gt_row.get("case_id", ""),
                "family": gt_row.get("family", "unknown"),
                "camera_bucket": gt_row.get("camera_bucket", "unknown"),
                "artifact_type": gt_row.get("artifact_type", ""),
                "gt": gt_row.get("gt", "UNKNOWN"),
                "pred": pred_choice,
                "correct": pred_choice == gt_row.get("gt"),
                "pred_parse_source": pred_source,
                "source_file": pred.get("_source_file", ""),
                "response": response,
            }
        )

    missing_gt = len([row for row in gt_rows if int(row["data_index"]) not in used_indices])
    return items, unmatched_predictions, missing_gt


def basic_metrics(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for item in items:
        gt = str(item.get("gt", "UNKNOWN"))
        pred = str(item.get("pred", "UNKNOWN"))
        counts["total"] += 1
        counts["valid"] += int(pred in {"A", "B"})
        counts[f"gt_{gt}"] += 1
        counts[f"pred_{pred}"] += 1
        counts["correct"] += int(pred == gt)
    total = counts["total"]
    return {
        "num_samples": total,
        "num_valid_predictions": counts["valid"],
        "format_valid_rate": safe_div(counts["valid"], total),
        "accuracy": safe_div(counts["correct"], total),
        "gt_A_rate": safe_div(counts["gt_A"], total),
        "pred_A_rate": safe_div(counts["pred_A"], total),
        "accuracy_when_gt_A": safe_div(
            sum(1 for item in items if item.get("gt") == "A" and item.get("pred") == "A"),
            counts["gt_A"],
        ),
        "accuracy_when_gt_B": safe_div(
            sum(1 for item in items if item.get("gt") == "B" and item.get("pred") == "B"),
            counts["gt_B"],
        ),
        "confusion": {
            "A_as_A": sum(1 for item in items if item.get("gt") == "A" and item.get("pred") == "A"),
            "A_as_B": sum(1 for item in items if item.get("gt") == "A" and item.get("pred") == "B"),
            "B_as_B": sum(1 for item in items if item.get("gt") == "B" and item.get("pred") == "B"),
            "B_as_A": sum(1 for item in items if item.get("gt") == "B" and item.get("pred") == "A"),
            "pred_unknown": counts["pred_UNKNOWN"],
        },
    }


def grouped_metrics(items: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item.get(key, "unknown") or "unknown")].append(item)
    return {name: basic_metrics(rows) for name, rows in sorted(groups.items())}


def write_items_csv(path: str | Path, items: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "data_index",
        "case_id",
        "family",
        "camera_bucket",
        "artifact_type",
        "gt",
        "pred",
        "correct",
        "pred_parse_source",
        "source_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow({field: item.get(field, "") for field in fields})


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt_json", required=True, help="Pair-selection JSON produced by the pretext builder.")
    parser.add_argument("--pred_json", required=True, help="Prediction JSON file or rank-output directory.")
    parser.add_argument("--out_dir", default=None, help="Output directory. Defaults to pred_json/eval_pair_selection.")
    parser.add_argument("--output_prefix", default="dataa_pair_selection_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_rows = build_gt_rows(args.gt_json)
    predictions = load_predictions(args.pred_json)
    items, unmatched_predictions, missing_gt = build_eval_items(gt_rows, predictions)
    summary = {
        "gt_json": str(args.gt_json),
        "pred_json": str(args.pred_json),
        "num_gt_records": len(gt_rows),
        "num_prediction_records_loaded": len(predictions),
        "num_matched_records": len(items),
        "num_unmatched_predictions": unmatched_predictions,
        "num_missing_gt_predictions": missing_gt,
        "overall": basic_metrics(items),
        "by_family": grouped_metrics(items, "family"),
        "by_camera_bucket": grouped_metrics(items, "camera_bucket"),
        "by_artifact_type": grouped_metrics(items, "artifact_type"),
    }

    pred_path = Path(args.pred_json)
    default_out = pred_path / "eval_pair_selection" if pred_path.is_dir() else pred_path.parent / "eval_pair_selection"
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    summary_path = out_dir / f"{args.output_prefix}_summary.json"
    csv_path = out_dir / f"{args.output_prefix}_items.csv"
    write_json(summary_path, summary)
    write_items_csv(csv_path, items)

    overall = summary["overall"]
    print("=== DataA Pair-Selection Evaluation ===")
    print(f"GT records: {summary['num_gt_records']}")
    print(f"Predictions loaded: {summary['num_prediction_records_loaded']}")
    print(f"Matched records: {summary['num_matched_records']}")
    print(f"Accuracy: {format_pct(overall['accuracy'])}")
    print(f"Format valid rate: {format_pct(overall['format_valid_rate'])}")
    print(f"GT A rate: {format_pct(overall['gt_A_rate'])}")
    print(f"Pred A rate: {format_pct(overall['pred_A_rate'])}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved items: {csv_path}")


if __name__ == "__main__":
    main()
