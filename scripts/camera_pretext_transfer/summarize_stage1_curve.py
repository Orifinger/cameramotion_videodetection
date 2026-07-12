#!/usr/bin/env python3
"""Summarize the fixed four-epoch camera-learning gate without best-checkpoint cherry-picking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.caspr_gate1.runtime import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[48, 96, 144, 192])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    curve = []
    earliest_passing_step = None
    for step in args.steps:
        path = eval_dir / f"stage1_gate_clean_step_{step}.json"
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        passed = payload.get("status") == "passed"
        if passed and earliest_passing_step is None:
            earliest_passing_step = step
        curve.append({
            "cumulative_steps": step,
            "approx_epochs": step / 48.0,
            "status": payload.get("status"),
            "checks": payload.get("checks"),
            "correct": payload.get("correct"),
            "shuffled": payload.get("shuffled"),
            "correct_deltas": payload.get("correct_deltas"),
            "source_json": str(path),
        })
    summary = {
        "gate": "Stage 1 fixed clean four-epoch camera-learning curve",
        "status": "passed" if earliest_passing_step is not None else "failed",
        "selection_rule": "Select the earliest checkpoint that passes every pre-registered Stage 1 check.",
        "earliest_passing_step": earliest_passing_step,
        "selected_correct_adapter": (
            f"camera_sft/correct_clean_4epoch/checkpoint-{earliest_passing_step}"
            if earliest_passing_step is not None else None
        ),
        "curve": curve,
        "next_action": (
            "Run the paraphrased prompt diagnostic, then Stage 2 from the selected checkpoint."
            if earliest_passing_step is not None
            else "Stop the camera-label pretext route; do not add more epochs or start Stage 2."
        ),
    }
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
