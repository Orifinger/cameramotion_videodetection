#!/usr/bin/env python3
"""Launch independent SAM3 workers across multiple physical GPUs.

Example full run:
    python scripts/launch_sam3_parallel.py

The launcher creates N_GPU * WORKERS_PER_GPU subprocesses. Each subprocess is
bound to one physical GPU with CUDA_VISIBLE_DEVICES and handles a deterministic
non-overlapping video shard. It only merges results if every worker exits cleanly.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sam3_tracking_config import (
    SAM3_PARALLEL_RUN_ROOT,
    SAM3_PHYSICAL_GPU_IDS,
    SAM3_WORKERS_PER_GPU,
    SAM3_WORKER_STARTUP_WAVE_DELAY_SEC,
)


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("sam3_v4_%Y%m%dT%H%M%SZ")


def parse_gpu_ids(raw: str) -> list[int]:
    ids = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not ids:
        raise ValueError("At least one physical GPU id is required")
    if len(ids) != len(set(ids)):
        raise ValueError("GPU ids must be unique")
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        default=",".join(str(gpu) for gpu in SAM3_PHYSICAL_GPU_IDS),
        help="Comma-separated physical GPU ids.",
    )
    parser.add_argument("--workers-per-gpu", type=int, default=SAM3_WORKERS_PER_GPU)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--startup-wave-delay-sec",
        type=float,
        default=SAM3_WORKER_STARTUP_WAVE_DELAY_SEC,
    )
    parser.add_argument("--qwen-candidates", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--mask-root", type=Path, default=None)
    parser.add_argument("--max-candidates-per-video", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_ids = parse_gpu_ids(args.gpus)
    if args.workers_per_gpu <= 0:
        raise ValueError("workers-per-gpu must be positive")
    if args.startup_wave_delay_sec < 0:
        raise ValueError("startup-wave-delay-sec must be non-negative")
    if args.max_candidates_per_video is not None and args.max_candidates_per_video <= 0:
        raise ValueError("--max-candidates-per-video must be positive")

    run_id = args.run_id or default_run_id()
    num_workers = len(gpu_ids) * args.workers_per_gpu
    parallel_root = Path(args.out_root) / "parallel_runs" if args.out_root is not None else Path(SAM3_PARALLEL_RUN_ROOT)
    run_root = parallel_root / run_id
    logs_root = run_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    # Slot-major ordering: all GPUs start their first model, then second model,
    # avoiding a 64-process simultaneous model/checkpoint load burst.
    jobs: list[tuple[int, int, int]] = []
    worker_index = 0
    for slot in range(args.workers_per_gpu):
        for gpu_id in gpu_ids:
            jobs.append((worker_index, gpu_id, slot))
            worker_index += 1

    print(
        f"[launch] run_id={run_id} gpus={gpu_ids} workers_per_gpu={args.workers_per_gpu} "
        f"total_workers={num_workers}",
        flush=True,
    )
    processes: list[tuple[int, int, Path, subprocess.Popen[bytes], object]] = []
    try:
        for position, (worker_index, gpu_id, slot) in enumerate(jobs, start=1):
            log_path = logs_root / f"worker_{worker_index:03d}_gpu_{gpu_id:02d}_slot_{slot}.log"
            handle = open(log_path, "ab", buffering=0)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONUNBUFFERED"] = "1"
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_sam3_tracking_worker.py"),
                "--run-id", run_id,
                "--worker-index", str(worker_index),
                "--num-workers", str(num_workers),
                "--physical-gpu-id", str(gpu_id),
            ]
            if args.qwen_candidates is not None:
                command.extend(["--qwen-candidates", str(args.qwen_candidates)])
            if args.out_root is not None:
                command.extend(["--out-root", str(args.out_root)])
            if args.mask_root is not None:
                command.extend(["--mask-root", str(args.mask_root)])
            if args.max_candidates_per_video is not None:
                command.extend(["--max-candidates-per-video", str(args.max_candidates_per_video)])
            process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
            processes.append((worker_index, gpu_id, log_path, process, handle))
            print(f"[launch] worker={worker_index:03d} gpu={gpu_id} slot={slot} log={log_path}", flush=True)
            if position % len(gpu_ids) == 0 and position < len(jobs):
                time.sleep(args.startup_wave_delay_sec)

        failed: list[tuple[int, int, int, Path]] = []
        for worker_index, gpu_id, log_path, process, handle in processes:
            return_code = process.wait()
            handle.close()
            if return_code != 0:
                failed.append((worker_index, gpu_id, return_code, log_path))
                print(f"[launch] FAILED worker={worker_index:03d} gpu={gpu_id} rc={return_code} log={log_path}", flush=True)
            else:
                print(f"[launch] finished worker={worker_index:03d} gpu={gpu_id}", flush=True)

        if failed:
            raise RuntimeError(f"{len(failed)} worker processes failed; canonical merge was not started")

        merge_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "merge_sam3_parallel_results.py"),
            "--run-id", run_id,
            "--num-workers", str(num_workers),
        ]
        if args.qwen_candidates is not None:
            merge_command.extend(["--qwen-candidates", str(args.qwen_candidates)])
        if args.out_root is not None:
            merge_command.extend(["--out-root", str(args.out_root)])
        print("[launch] all workers clean; merging canonical track bank", flush=True)
        subprocess.run(merge_command, cwd=str(PROJECT_ROOT), check=True)
        print(f"[launch] complete run_id={run_id}", flush=True)
    except KeyboardInterrupt:
        print("[launch] interrupted; terminating child workers. Existing shard checkpoints remain resumable.", flush=True)
        for _, _, _, process, handle in processes:
            if process.poll() is None:
                process.terminate()
            handle.close()
        raise


if __name__ == "__main__":
    main()
