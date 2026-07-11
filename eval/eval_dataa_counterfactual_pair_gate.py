#!/usr/bin/env python3
"""Gate 1: evaluate DataA pair choice, A/B swap consistency, and localization."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CHOICE_RE = re.compile(r"<edited_video>\s*([AB])\s*</edited_video>", re.IGNORECASE | re.DOTALL)
BBOX_PATTERNS = [
    re.compile(r"<edit_bbox>\s*\[([^]]+)\]\s*</edit_bbox>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<bbox>\s*\[([^]]+)\]\s*</bbox>", re.IGNORECASE | re.DOTALL),
]


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assistant_text(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                return str(message.get("content", ""))
    return ""


def response_text(record: Mapping[str, Any]) -> str:
    for key in ("response", "prediction", "raw_response", "completion", "generated_text"):
        if isinstance(record.get(key), str):
            return str(record[key])
    return assistant_text(record)


def parse_choice(value: Any) -> str:
    match = CHOICE_RE.search(str(value or ""))
    return match.group(1).upper() if match else "UNKNOWN"


def parse_bbox(value: Any) -> list[float] | None:
    text = str(value or "")
    for pattern in BBOX_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        try:
            box = [float(part.strip()) for part in match.group(1).split(",")]
        except ValueError:
            continue
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return [max(0.0, x1), max(0.0, y1), min(1000.0, x2), min(1000.0, y2)]
    return None


def bbox_iou(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def prediction_files(path: str | Path) -> Iterable[Path]:
    value = Path(path)
    if value.is_file():
        yield value
        return
    for candidate in sorted(value.rglob("*.json")):
        folded = candidate.name.casefold()
        if "summary" not in folded and "metrics" not in folded:
            yield candidate


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    output = []
    for file_path in prediction_files(path):
        payload = read_json(file_path)
        rows = payload.get("predictions") if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping):
                item = dict(row)
                item["_source_file"] = str(file_path)
                output.append(item)
    return output


def first_image_split(record: Mapping[str, Any]) -> str:
    images = record.get("images")
    if not isinstance(images, list) or not images:
        return ""
    path = str(images[0]).replace("\\", "/")
    parent = Path(path).parent.name.casefold()
    return parent if parent in {"real", "fake"} else ""


def prediction_key(record: Mapping[str, Any]) -> tuple[str, str] | None:
    case_id = str(record.get("case_id", "")).strip()
    if not case_id:
        sample_id = str(record.get("sample_id", ""))
        match = re.search(r"(dataA_v1_(?:dataset_v2_|textedit_reserve_)?\d+)", sample_id)
        case_id = match.group(1) if match else ""
    first_split = first_image_split(record)
    if case_id and first_split:
        return case_id, "real_first" if first_split == "real" else "fake_first"
    return None


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def aggregate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(items)
    valid_choice = [item for item in items if item.get("pred_choice") in {"A", "B"}]
    valid_bbox = [item for item in items if item.get("pred_bbox") is not None]
    return {
        "num_samples": total,
        "choice_format_valid_rate": safe_div(len(valid_choice), total),
        "bbox_format_valid_rate": safe_div(len(valid_bbox), total),
        "pair_choice_accuracy": safe_div(sum(bool(item.get("choice_correct")) for item in items), total),
        "pred_A_rate": safe_div(sum(item.get("pred_choice") == "A" for item in items), len(valid_choice)),
        "mean_bbox_iou": safe_div(sum(float(item.get("bbox_iou", 0.0)) for item in items), total),
        "bbox_hit_at_0_3": safe_div(sum(float(item.get("bbox_iou", 0.0)) >= 0.3 for item in items), total),
    }


def physical_prediction(item: Mapping[str, Any]) -> str:
    choice = item.get("pred_choice")
    if choice == "A":
        return str(item.get("video_a_source_split", ""))
    if choice == "B":
        return str(item.get("video_b_source_split", ""))
    return "unknown"


def swap_metrics(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for item in items:
        by_case[str(item.get("case_id", ""))][str(item.get("pair_order", ""))] = item
    complete = [orders for orders in by_case.values() if set(orders) >= {"real_first", "fake_first"}]
    consistent = 0
    both_correct = 0
    flipped = 0
    for orders in complete:
        real_first, fake_first = orders["real_first"], orders["fake_first"]
        physical_a = physical_prediction(real_first)
        physical_b = physical_prediction(fake_first)
        consistent += int(physical_a == physical_b and physical_a in {"real", "fake"})
        both_correct += int(bool(real_first.get("choice_correct")) and bool(fake_first.get("choice_correct")))
        choice_a, choice_b = real_first.get("pred_choice"), fake_first.get("pred_choice")
        flipped += int(choice_a in {"A", "B"} and choice_b in {"A", "B"} and choice_a != choice_b)
    return {
        "num_complete_swap_cases": len(complete),
        "swap_consistency_rate": safe_div(consistent, len(complete)),
        "both_orders_correct_rate": safe_div(both_correct, len(complete)),
        "prediction_flip_rate": safe_div(flipped, len(complete)),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "data_index", "case_id", "pair_order", "motion_bucket", "artifact_type",
        "gt_choice", "pred_choice", "choice_correct", "gt_bbox", "pred_bbox",
        "bbox_iou", "video_a_source_split", "video_b_source_split", "source_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["gt_bbox"] = json.dumps(output.get("gt_bbox"), ensure_ascii=False)
            output["pred_bbox"] = json.dumps(output.get("pred_bbox"), ensure_ascii=False)
            writer.writerow({field: output.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-choice-accuracy", type=float, default=0.70)
    parser.add_argument("--min-swap-consistency", type=float, default=0.85)
    parser.add_argument("--min-pred-a-rate", type=float, default=0.45)
    parser.add_argument("--max-pred-a-rate", type=float, default=0.55)
    parser.add_argument("--min-mean-bbox-iou", type=float, default=0.30)
    parser.add_argument("--min-format-valid-rate", type=float, default=0.95)
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt = read_json(args.gt_json)
    if not isinstance(gt, list):
        raise ValueError(f"expected GT JSON list: {args.gt_json}")
    predictions = load_predictions(args.pred_json)
    by_index = {int(row["data_index"]): row for row in predictions if isinstance(row.get("data_index"), int)}
    by_key = {key: row for row in predictions if (key := prediction_key(row)) is not None}
    items = []
    used_prediction_ids = set()
    for data_index, record in enumerate(gt):
        if not isinstance(record, Mapping):
            continue
        key = (str(record.get("case_id", "")), str(record.get("pair_order", "")))
        prediction = by_index.get(data_index) or by_key.get(key)
        response = response_text(prediction) if prediction else ""
        pred_choice, pred_bbox = parse_choice(response), parse_bbox(response)
        gt_choice = str(record.get("edited_video", "UNKNOWN"))
        gt_bbox = [float(value) for value in record.get("edit_bbox_1000", [])]
        if prediction:
            used_prediction_ids.add(id(prediction))
        items.append(
            {
                "data_index": data_index,
                "case_id": record.get("case_id", ""),
                "pair_order": record.get("pair_order", ""),
                "motion_bucket": record.get("motion_bucket", "unknown"),
                "artifact_type": record.get("artifact_type", ""),
                "gt_choice": gt_choice,
                "pred_choice": pred_choice,
                "choice_correct": pred_choice == gt_choice,
                "gt_bbox": gt_bbox,
                "pred_bbox": pred_bbox,
                "bbox_iou": bbox_iou(gt_bbox, pred_bbox),
                "video_a_source_split": record.get("video_a_source_split", ""),
                "video_b_source_split": record.get("video_b_source_split", ""),
                "source_file": prediction.get("_source_file", "") if prediction else "",
            }
        )

    overall = aggregate(items)
    swap = swap_metrics(items)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item.get("motion_bucket", "unknown"))].append(item)
    checks = {
        "pair_choice_accuracy": overall["pair_choice_accuracy"] >= args.min_choice_accuracy,
        "swap_consistency_rate": swap["swap_consistency_rate"] >= args.min_swap_consistency,
        "pred_A_rate_lower": overall["pred_A_rate"] >= args.min_pred_a_rate,
        "pred_A_rate_upper": overall["pred_A_rate"] <= args.max_pred_a_rate,
        "mean_bbox_iou": overall["mean_bbox_iou"] >= args.min_mean_bbox_iou,
        "choice_format_valid_rate": overall["choice_format_valid_rate"] >= args.min_format_valid_rate,
        "bbox_format_valid_rate": overall["bbox_format_valid_rate"] >= args.min_format_valid_rate,
    }
    passed = bool(items) and swap["num_complete_swap_cases"] > 0 and all(checks.values())
    summary = {
        "gate": "Gate 1 - camera-matched pair choice and localization",
        "gt_json": args.gt_json,
        "pred_json": args.pred_json,
        "status": "passed" if passed else "failed",
        "num_gt_records": len(items),
        "num_prediction_records_loaded": len(predictions),
        "num_missing_predictions": sum(not item["source_file"] for item in items),
        "num_unmatched_predictions": len(predictions) - len(used_prediction_ids),
        "thresholds": {
            "min_choice_accuracy": args.min_choice_accuracy,
            "min_swap_consistency": args.min_swap_consistency,
            "pred_A_rate_range": [args.min_pred_a_rate, args.max_pred_a_rate],
            "min_mean_bbox_iou": args.min_mean_bbox_iou,
            "min_format_valid_rate": args.min_format_valid_rate,
        },
        "checks": checks,
        "overall": overall,
        "swap_control": swap,
        "by_motion_bucket": {name: aggregate(rows) for name, rows in sorted(groups.items())},
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "dataa_counterfactual_pair_gate_summary.json"
    items_path = out_dir / "dataa_counterfactual_pair_gate_items.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(items_path, items)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved items: {items_path}")
    if args.fail_on_gate and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
