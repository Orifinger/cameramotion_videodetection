#!/usr/bin/env python3
"""Estimate the unattended run duration from distributed preflight measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.camera_binary_vqa.runtime import write_json


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-summary", type=Path, required=True)
    parser.add_argument("--train-state", type=Path, required=True)
    parser.add_argument("--score-state", type=Path, required=True)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--max-train-wall-seconds", type=float, default=16200.0)
    parser.add_argument("--fixed-buffer-seconds", type=float, default=1200.0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = read_json(args.data_summary)
    train = read_json(args.train_state)
    score = read_json(args.score_state)
    completed_steps = int(train["completed_steps"])
    if completed_steps <= 0:
        raise ValueError("training smoke completed no steps")
    seconds_per_step = float(train["training_elapsed_seconds"]) / completed_steps
    steps_per_epoch = int(data["steps_per_epoch"])
    planned_steps = steps_per_epoch * args.num_epochs
    one_epoch_seconds = steps_per_epoch * seconds_per_step
    unbounded_training_seconds = planned_steps * seconds_per_step
    estimated_training_seconds = max(
        one_epoch_seconds,
        min(unbounded_training_seconds, args.max_train_wall_seconds),
    )

    records_per_second = float(score["aggregate_records_per_second"])
    if records_per_second <= 0:
        raise ValueError("preflight inference throughput is not positive")
    dev_records = int(data["dev_records_per_condition"])
    total_scored_records = dev_records * 5
    inference_compute_seconds = total_scored_records / records_per_second
    inference_model_load_seconds = float(score["model_setup_seconds"]) * 3
    training_model_load_seconds = float(train["model_setup_seconds"])
    total_seconds = (
        estimated_training_seconds
        + inference_compute_seconds
        + inference_model_load_seconds
        + training_model_load_seconds
        + args.fixed_buffer_seconds
    )
    conservative_seconds = total_seconds * 1.25
    output = {
        "estimate": "DataA binary camera VQA full unattended pipeline",
        "basis": "16-GPU four-step training smoke and 32-record distributed inference smoke",
        "data": {
            "train_records": data["train_records"],
            "dev_records_per_condition": dev_records,
            "steps_per_epoch": steps_per_epoch,
            "planned_epochs": args.num_epochs,
            "planned_steps": planned_steps,
            "total_scored_records": total_scored_records,
        },
        "measured": {
            "training_seconds_per_optimizer_step": seconds_per_step,
            "inference_records_per_second_aggregate": records_per_second,
            "training_model_setup_seconds": train["model_setup_seconds"],
            "inference_model_setup_seconds": score["model_setup_seconds"],
        },
        "estimated": {
            "one_epoch_minutes": one_epoch_seconds / 60.0,
            "training_hours_before_wall_guard": unbounded_training_seconds / 3600.0,
            "training_hours_with_wall_guard": estimated_training_seconds / 3600.0,
            "all_inference_hours": (
                inference_compute_seconds + inference_model_load_seconds
            )
            / 3600.0,
            "fixed_eval_and_archive_buffer_minutes": args.fixed_buffer_seconds / 60.0,
            "total_hours": total_seconds / 3600.0,
            "conservative_total_hours": conservative_seconds / 3600.0,
            "fits_eight_hours_conservatively": conservative_seconds <= 8 * 3600,
        },
        "warning": (
            "This is a short-smoke estimate. Video decoding cache, OSS bandwidth, and shared server load can "
            "change throughput; the formal trainer still enforces its own wall-clock guard."
        ),
    }
    write_json(args.output_json, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
