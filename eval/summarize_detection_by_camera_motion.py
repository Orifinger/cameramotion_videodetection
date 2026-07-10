#!/usr/bin/env python3
"""Stratify DataA detection evaluation items by camera-motion labels."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.camera_detection_rewards import normalize_truth_labels  # noqa: E402

DATAA_CASE_RE = re.compile(r"(dataA_v1(?:_[A-Za-z][A-Za-z0-9]*)*_\d+)")


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def normalize_detection_label(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if text == "fake":
        return "Fake"
    if text == "real":
        return "Real"
    return "UNKNOWN"


def motion_bucket(labels: Sequence[str]) -> str:
    present = set(labels)
    if "complex-motion" in present:
        return "complex-motion"
    if "minor-motion" in present:
        return "minor-motion"
    if "no-motion" in present:
        return "no-motion"
    return "unknown"


def camera_key(path_value: Any) -> tuple[str, str] | None:
    normalized = str(path_value or "").replace("\\", "/").rstrip("/")
    match = DATAA_CASE_RE.search(normalized)
    if not match:
        return None
    split = PurePosixPath(normalized).name.casefold()
    if split not in {"real", "fake"}:
        return None
    return match.group(1), split


def load_camera(path: Path) -> tuple[dict[tuple[str, str], list[str]], dict[str, list[str]]]:
    exact: dict[tuple[str, str], list[str]] = {}
    per_case: dict[str, list[list[str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, Mapping):
                continue
            key = camera_key(row.get("path"))
            if key is None:
                continue
            labels = normalize_truth_labels(row.get("labels", []))
            exact[key] = labels
            per_case[key[0]].append(labels)
    fallback = {}
    for case_id, candidates in per_case.items():
        canonical = {tuple(labels) for labels in candidates}
        if len(canonical) == 1:
            fallback[case_id] = list(next(iter(canonical)))
    return exact, fallback


def infer_case_and_split(row: Mapping[str, Any]) -> tuple[str, str]:
    case_id = str(row.get("case_id", "")).strip()
    sample_id = str(row.get("sample_id", "")).strip()
    if not case_id:
        match = DATAA_CASE_RE.search(sample_id)
        case_id = match.group(1) if match else ""
    gt = normalize_detection_label(row.get("gt"))
    split = gt.casefold() if gt in {"Real", "Fake"} else ""
    if split not in {"real", "fake"} and sample_id.rsplit(":", 1)[-1].casefold() in {"real", "fake"}:
        split = sample_id.rsplit(":", 1)[-1].casefold()
    return case_id, split


def classification_metrics(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for item in items:
        gt = normalize_detection_label(item.get("gt"))
        pred = normalize_detection_label(item.get("pred"))
        counts["total"] += 1
        counts["valid"] += int(pred in {"Fake", "Real"})
        counts[f"gt_{gt.casefold()}"] += 1
        counts[f"pred_{pred.casefold()}"] += 1
        if gt == "Fake" and pred == "Fake":
            counts["tp"] += 1
        elif gt == "Real" and pred == "Real":
            counts["tn"] += 1
        elif gt == "Real" and pred == "Fake":
            counts["fp"] += 1
        elif gt == "Fake" and pred != "Fake":
            counts["fn"] += 1
        elif gt == "Real" and pred == "UNKNOWN":
            counts["unknown_real"] += 1
    total = counts["total"]
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    precision = safe_div(tp, tp + fp)
    fake_recall = safe_div(tp, counts["gt_fake"])
    real_recall = safe_div(tn, counts["gt_real"])
    return {
        "num_samples": total,
        "num_valid_predictions": counts["valid"],
        "format_valid_rate": safe_div(counts["valid"], total),
        "accuracy": safe_div(tp + tn, total),
        "balanced_accuracy": (fake_recall + real_recall) / 2.0,
        "fake_precision": precision,
        "fake_recall": fake_recall,
        "real_recall": real_recall,
        "fake_f1": safe_div(2 * precision * fake_recall, precision + fake_recall),
        "confusion_fake_positive": {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "unknown_on_real": counts["unknown_real"],
        },
        "label_counts": {
            "gt_fake": counts["gt_fake"], "gt_real": counts["gt_real"],
            "pred_fake": counts["pred_fake"], "pred_real": counts["pred_real"],
            "pred_unknown": counts["pred_unknown"],
        },
    }


def load_eval_items(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def enrich_items(rows, exact_camera, fallback_camera):
    items, unmatched = [], 0
    for row in rows:
        case_id, split = infer_case_and_split(row)
        labels = exact_camera.get((case_id, split))
        if labels is None:
            labels = fallback_camera.get(case_id)
        if labels is None:
            labels = []
            unmatched += 1
        item = dict(row)
        item.update(case_id=case_id, camera_split=split,
                    camera_labels=list(labels), motion_bucket=motion_bucket(labels))
        items.append(item)
    return items, unmatched


def write_items(path: Path, items) -> None:
    base = ["sample_id", "case_id", "camera_split", "gt", "pred", "correct",
            "motion_bucket", "camera_labels", "source_file"]
    extra = sorted({key for item in items for key in item if key not in base})
    fields = base + extra
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            row = dict(item)
            row["camera_labels"] = json.dumps(row.get("camera_labels", []), ensure_ascii=False)
            writer.writerow({key: row.get(key, "") for key in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-items-csv", required=True)
    parser.add_argument("--camera-jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-items-csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_path, camera_path = Path(args.eval_items_csv), Path(args.camera_jsonl)
    rows = load_eval_items(eval_path)
    exact_camera, fallback_camera = load_camera(camera_path)
    items, unmatched = enrich_items(rows, exact_camera, fallback_camera)
    buckets = ["no-motion", "minor-motion", "complex-motion", "unknown"]
    labels = sorted({label for item in items for label in item["camera_labels"]})
    summary = {
        "eval_items_csv": str(eval_path),
        "camera_jsonl": str(camera_path),
        "num_eval_items": len(items),
        "num_items_without_camera_match": unmatched,
        "overall": classification_metrics(items),
        "by_motion_bucket": {
            bucket: classification_metrics([item for item in items if item["motion_bucket"] == bucket])
            for bucket in buckets if any(item["motion_bucket"] == bucket for item in items)
        },
        "by_camera_label": {
            label: classification_metrics([item for item in items if label in item["camera_labels"]])
            for label in labels
        },
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    out_csv = Path(args.out_items_csv) if args.out_items_csv else out_json.with_name(
        f"{out_json.stem}_items.csv")
    write_items(out_csv, items)
    print("=== Detection Metrics by Camera Motion ===")
    print(f"Evaluation items: {len(items)}")
    print(f"Items without camera match: {unmatched}")
    for bucket, metrics in summary["by_motion_bucket"].items():
        print(f"{bucket}: n={metrics['num_samples']}, accuracy={metrics['accuracy'] * 100:.2f}%, "
              f"balanced_accuracy={metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"Saved summary: {out_json}")
    print(f"Saved items: {out_csv}")


if __name__ == "__main__":
    main()
