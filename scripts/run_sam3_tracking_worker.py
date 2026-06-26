#!/usr/bin/env python3
"""Run one deterministic, resumable SAM3 shard on one visible GPU.

This is launched by scripts/launch_sam3_parallel.py. Do not start multiple
workers manually with the same worker index: a worker owns one output file and
one non-overlapping modulo shard of the Qwen candidate list.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sam3_tracking_config import (
    SAM3_PARALLEL_RUN_ROOT,
    SAM3_RETRY_FAILURE_RECORDS,
    SAM3_SCHEMA_VERSION,
    SAM3_TRACK_MASK_ROOT,
)
import run_sam3_tracking as core


TERMINAL_STATUSES = {"success", "no_track", "no_valid_candidate"}


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    videos = payload.get("videos") if isinstance(payload, dict) else None
    if not isinstance(videos, list):
        return {}
    return {
        str(record["video_id"]): record
        for record in videos
        if isinstance(record, dict) and record.get("video_id")
    }


def worker_payload(
    run_id: str,
    worker_index: int,
    num_workers: int,
    physical_gpu_id: str,
    assigned_total: int,
    records: list[dict[str, Any]],
    started_at: str,
    finished: bool,
) -> dict[str, Any]:
    statuses: dict[str, int] = defaultdict(int)
    for record in records:
        statuses[str(record.get("status", "unknown"))] += 1
    return {
        "schema_version": SAM3_SCHEMA_VERSION,
        "run_id": run_id,
        "worker_index": worker_index,
        "num_workers": num_workers,
        "physical_gpu_id": physical_gpu_id,
        "assigned_video_count": assigned_total,
        "completed_video_count": len(records),
        "status_totals": dict(statuses),
        "started_at_utc": started_at,
        "updated_at_utc": core.utc_now(),
        "finished_at_utc": core.utc_now() if finished else None,
        "videos": records,
    }


def is_terminal(record: dict[str, Any] | None) -> bool:
    if not isinstance(record, dict):
        return False
    status = record.get("status")
    return status in TERMINAL_STATUSES or (
        status == "failure" and not SAM3_RETRY_FAILURE_RECORDS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker-index", required=True, type=int)
    parser.add_argument("--num-workers", required=True, type=int)
    parser.add_argument("--physical-gpu-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_index < 0 or args.worker_index >= args.num_workers:
        raise ValueError("worker-index must be in [0, num-workers)")

    dataset, input_videos = core.load_input()
    assigned = [
        video for index, video in enumerate(input_videos)
        if index % args.num_workers == args.worker_index
    ]

    run_root = Path(SAM3_PARALLEL_RUN_ROOT) / args.run_id
    worker_root = run_root / "workers"
    logs_root = run_root / "logs"
    worker_path = worker_root / f"worker_{args.worker_index:03d}_tracks.json"
    worker_summary_path = worker_root / f"worker_{args.worker_index:03d}_summary.json"
    logs_root.mkdir(parents=True, exist_ok=True)

    # Isolate mask assets between parallel runs and avoid collisions with earlier
    # single-worker smoke runs. Video shards are disjoint inside a given run.
    core.SAM3_TRACK_MASK_ROOT = (
        Path(SAM3_TRACK_MASK_ROOT) / "parallel_runs" / args.run_id / f"worker_{args.worker_index:03d}"
    )

    existing = load_existing(worker_path)
    todo = [video for video in assigned if not is_terminal(existing.get(str(video["video_id"])))]
    started_at = core.utc_now()
    print(
        f"[worker {args.worker_index:03d}] gpu={args.physical_gpu_id} "
        f"assigned={len(assigned)} resume_skip={len(assigned) - len(todo)} todo={len(todo)}",
        flush=True,
    )

    ordered_results: dict[str, dict[str, Any]] = dict(existing)
    if not todo:
        records = [ordered_results[str(video["video_id"])] for video in assigned if str(video["video_id"]) in ordered_results]
        final_payload = worker_payload(
            args.run_id, args.worker_index, args.num_workers, args.physical_gpu_id,
            len(assigned), records, started_at, True,
        )
        atomic_write_json(worker_path, final_payload)
        atomic_write_json(worker_summary_path, {k: v for k, v in final_payload.items() if k != "videos"})
        print(f"[worker {args.worker_index:03d}] nothing to do", flush=True)
        return

    runner = core.Sam3Runner()
    try:
        for completed, video in enumerate(todo, start=1):
            result = core.process_video(runner, video)
            ordered_results[str(video["video_id"])] = result
            records = [ordered_results[str(item["video_id"])] for item in assigned if str(item["video_id"]) in ordered_results]
            checkpoint = worker_payload(
                args.run_id, args.worker_index, args.num_workers, args.physical_gpu_id,
                len(assigned), records, started_at, False,
            )
            atomic_write_json(worker_path, checkpoint)
            atomic_write_json(worker_summary_path, {k: v for k, v in checkpoint.items() if k != "videos"})
            print(
                f"[worker {args.worker_index:03d}] {completed}/{len(todo)} "
                f"video={video['video_id']} status={result['status']} "
                f"elapsed={result.get('elapsed_seconds')}s",
                flush=True,
            )
    finally:
        runner.shutdown()

    records = [ordered_results[str(video["video_id"])] for video in assigned if str(video["video_id"]) in ordered_results]
    final_payload = worker_payload(
        args.run_id, args.worker_index, args.num_workers, args.physical_gpu_id,
        len(assigned), records, started_at, True,
    )
    atomic_write_json(worker_path, final_payload)
    atomic_write_json(worker_summary_path, {k: v for k, v in final_payload.items() if k != "videos"})
    print(f"[worker {args.worker_index:03d}] finished", flush=True)


if __name__ == "__main__":
    main()
