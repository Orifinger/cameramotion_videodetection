#!/usr/bin/env python3
"""Fail a Camera-PPRL smoke run when most GRPO groups have zero reward variance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


def metric_values(rows: list[Mapping[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        for key, value in row.items():
            if str(key).split("/")[-1] != metric:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
    return values


def load_logging_rows(train_dir: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    paths = sorted(Path(train_dir).rglob("logging.jsonl"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    rows.append(dict(payload))
    return rows, [str(path) for path in paths]


def build_audit(
    rows: list[Mapping[str, Any]],
    logging_files: list[str],
    max_zero_std_rate: float,
    min_log_points: int,
) -> dict[str, Any]:
    zero_std = metric_values(rows, "frac_reward_zero_std")
    reward_std = metric_values(rows, "reward_std")
    mean_zero_std = mean(zero_std) if zero_std else None
    checks = {
        "logging_file_found": bool(logging_files),
        "enough_reward_variance_points": len(zero_std) >= min_log_points,
        "nonzero_advantage_rate": (
            mean_zero_std is not None and mean_zero_std <= max_zero_std_rate
        ),
    }
    return {
        "gate": "Camera-PPRL smoke reward-variance audit",
        "status": "passed" if all(checks.values()) else "failed",
        "thresholds": {
            "max_mean_frac_reward_zero_std": max_zero_std_rate,
            "min_log_points": min_log_points,
        },
        "checks": checks,
        "metrics": {
            "logging_files": logging_files,
            "num_logging_rows": len(rows),
            "num_zero_std_points": len(zero_std),
            "mean_frac_reward_zero_std": mean_zero_std,
            "min_frac_reward_zero_std": min(zero_std) if zero_std else None,
            "max_frac_reward_zero_std": max(zero_std) if zero_std else None,
            "mean_reward_std": mean(reward_std) if reward_std else None,
        },
        "next_action": (
            "Proceed to the fixed 1024-record Camera-PPRL run."
            if all(checks.values())
            else "Stop before the formal run and inspect rollout exploration or log compatibility."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-zero-std-rate", type=float, default=0.80)
    parser.add_argument("--min-log-points", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, logging_files = load_logging_rows(args.train_dir)
    audit = build_audit(
        rows,
        logging_files,
        args.max_zero_std_rate,
        args.min_log_points,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
