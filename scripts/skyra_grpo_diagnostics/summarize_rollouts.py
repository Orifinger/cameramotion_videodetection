#!/usr/bin/env python3
"""Summarize verl rollout JSONL files into compact learning-curve artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ANSWER_RE = re.compile(r"<answer>\s*(Fake|Real)\s*</answer>", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reward-variant", required=True)
    return parser.parse_args()


def numeric_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return statistics.fmean(values) if values else None


def answer(text: Any) -> str | None:
    matches = ANSWER_RE.findall(str(text or ""))
    return matches[0].title() if len(matches) == 1 else None


def summarize_step(step: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        group_key = row.get("diagnostic_sample_id") or row.get("input", "")
        grouped_scores[str(group_key)].append(float(row.get("score", 0.0)))
    group_stds = [statistics.pstdev(values) for values in grouped_scores.values() if values]

    result: dict[str, Any] = {
        "step": step,
        "responses": len(rows),
        "prompt_groups": len(grouped_scores),
        "score_mean": numeric_mean(rows, "score"),
        "zero_std_group_rate": (
            sum(value <= 1e-12 for value in group_stds) / len(group_stds) if group_stds else None
        ),
        "group_reward_std_mean": statistics.fmean(group_stds) if group_stds else None,
    }
    metric_keys = [
        "accuracy_reward",
        "inspection_reward",
        "correct",
        "format_valid",
        "pred_fake",
        "gt_fake",
        "false_positive",
        "false_negative",
        "answer_invalid",
        "raw_check_count",
        "strict_check_count",
        "duplicate_check_count",
        "invalid_check_count",
        "invalid_bbox_count",
        "invalid_time_count",
        "invalid_type_count",
        "wrong_positive_reward",
        "response_chars",
    ]
    for key in metric_keys:
        result[key] = numeric_mean(rows, key)

    if result["correct"] is None:
        truths = [answer(row.get("gts")) or str(row.get("gts", "")).title() for row in rows]
        predictions = [answer(row.get("output")) for row in rows]
        valid = [(truth, prediction) for truth, prediction in zip(truths, predictions) if prediction is not None]
        result["correct"] = (
            sum(truth == prediction for truth, prediction in valid) / len(rows) if rows else None
        )
    return result


def main() -> None:
    args = parse_args()
    rollout_dir = Path(args.rollout_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(rollout_dir.glob("*.jsonl"), key=lambda path: int(path.stem))
    if not files:
        raise FileNotFoundError(f"no rollout JSONL files found under {rollout_dir}")

    curves = []
    for path in files:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        curves.append(summarize_step(int(path.stem), rows))

    keys = list(curves[0])
    csv_path = output_dir / "rollout_learning_curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(curves)

    summary = {
        "status": "completed",
        "reward_variant": args.reward_variant,
        "rollout_dir": str(rollout_dir),
        "steps_found": len(curves),
        "first_step": curves[0],
        "last_step": curves[-1],
        "curve_csv": str(csv_path),
    }
    summary_path = output_dir / "rollout_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
