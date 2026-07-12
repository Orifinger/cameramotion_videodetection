#!/usr/bin/env python3
"""Evaluate generated camera-motion label sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.camera_pretext_transfer.runtime import multilabel_metrics
from scripts.caspr_gate1.runtime import read_jsonl, write_json
from tools.build_camera_pretext_transfer_gate import CAMERA_LABEL_ORDER


def load_predictions(path: str | Path) -> list[dict]:
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.glob("rank_*.jsonl"))
    rows: list[dict] = []
    for file_path in files:
        rows.extend(read_jsonl(file_path))
    case_ids = [str(row.get("case_id")) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case_id in camera predictions")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-jsonl", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold = read_jsonl(args.gold_jsonl)
    predictions = load_predictions(args.predictions)
    metrics = multilabel_metrics(gold, predictions, CAMERA_LABEL_ORDER)
    summary = {
        "gate": "Stage 1 camera-motion ability evaluation",
        "model_name": args.model_name,
        "gold_jsonl": args.gold_jsonl,
        "predictions": args.predictions,
        "metrics": metrics,
    }
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
