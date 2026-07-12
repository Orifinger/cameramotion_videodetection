#!/usr/bin/env python3
"""Monitor mean utilization across all GPUs in fixed two-hour windows."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.camera_binary_vqa.runtime import write_json


STOP_REQUESTED = False


def request_stop(signum: int, frame: Any) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parent_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def query_utilization(expected_gpus: int) -> list[float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi utilization query failed")
    values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if len(values) != expected_gpus:
        raise RuntimeError(f"expected {expected_gpus} GPU values, received {len(values)}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, default=16)
    parser.add_argument("--sample-interval-seconds", type=float, default=60.0)
    parser.add_argument("--window-seconds", type=float, default=7200.0)
    parser.add_argument("--minimum-window-mean", type=float, default=30.0)
    return parser.parse_args()


def summarize_window(
    index: int,
    started_at: str,
    elapsed_seconds: float,
    samples: list[list[float]],
    minimum: float,
    complete: bool,
) -> dict[str, Any]:
    flat = [value for sample in samples for value in sample]
    per_gpu = [
        sum(sample[gpu] for sample in samples) / len(samples)
        for gpu in range(len(samples[0]))
    ] if samples else []
    mean = sum(flat) / len(flat) if flat else 0.0
    return {
        "window_index": index,
        "started_at_utc": started_at,
        "elapsed_seconds": elapsed_seconds,
        "num_samples": len(samples),
        "complete_two_hour_window": complete,
        "mean_utilization_percent": mean,
        "per_gpu_mean_utilization_percent": per_gpu,
        "minimum_required_percent": minimum,
        "passed": (mean >= minimum) if complete else None,
    }


def write_summary(
    path: Path,
    windows: list[dict[str, Any]],
    current: dict[str, Any] | None,
    all_samples: list[list[float]],
    minimum: float,
    running: bool,
    errors: list[str],
) -> None:
    complete = [window for window in windows if window["complete_two_hour_window"]]
    violations = [window for window in complete if not window["passed"]]
    flat = [value for sample in all_samples for value in sample]
    output = {
        "monitor": "mean GPU utilization across all GPUs in fixed two-hour windows",
        "status": "running" if running else (
            "warning" if violations or errors else "completed"
        ),
        "minimum_window_mean_percent": minimum,
        "num_complete_windows": len(complete),
        "num_violations": len(violations),
        "full_compute_phase_mean_percent": sum(flat) / len(flat) if flat else 0.0,
        "windows": windows,
        "current_partial_window": current,
        "errors": errors,
    }
    write_json(path, output)


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    window_index = 0
    window_started_monotonic = time.monotonic()
    window_started_utc = utc_now()
    window_samples: list[list[float]] = []
    all_samples: list[list[float]] = []
    windows: list[dict[str, Any]] = []
    errors: list[str] = []

    with args.output_jsonl.open("a", encoding="utf-8", newline="\n") as handle:
        while not STOP_REQUESTED and parent_exists(args.parent_pid):
            sampled_at = utc_now()
            try:
                values = query_utilization(args.expected_gpus)
                mean = sum(values) / len(values)
                row = {
                    "sampled_at_utc": sampled_at,
                    "mean_utilization_percent": mean,
                    "per_gpu_utilization_percent": values,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                window_samples.append(values)
                all_samples.append(values)
            except Exception as exc:
                errors.append(f"{sampled_at}: {exc!r}")

            elapsed = time.monotonic() - window_started_monotonic
            if elapsed >= args.window_seconds:
                windows.append(
                    summarize_window(
                        window_index,
                        window_started_utc,
                        elapsed,
                        window_samples,
                        args.minimum_window_mean,
                        True,
                    )
                )
                window_index += 1
                window_started_monotonic = time.monotonic()
                window_started_utc = utc_now()
                window_samples = []
            partial = summarize_window(
                window_index,
                window_started_utc,
                time.monotonic() - window_started_monotonic,
                window_samples,
                args.minimum_window_mean,
                False,
            )
            write_summary(
                args.summary_json,
                windows,
                partial,
                all_samples,
                args.minimum_window_mean,
                True,
                errors,
            )
            deadline = time.monotonic() + args.sample_interval_seconds
            while not STOP_REQUESTED and time.monotonic() < deadline:
                time.sleep(min(1.0, deadline - time.monotonic()))

    partial = summarize_window(
        window_index,
        window_started_utc,
        time.monotonic() - window_started_monotonic,
        window_samples,
        args.minimum_window_mean,
        False,
    )
    write_summary(
        args.summary_json,
        windows,
        partial,
        all_samples,
        args.minimum_window_mean,
        False,
        errors,
    )


if __name__ == "__main__":
    main()
