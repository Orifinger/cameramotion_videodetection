#!/usr/bin/env python3
"""Persistent VACE worker process.

Launched once per worker group via torchrun. Rank 0 reads one shard descriptor
and broadcasts case descriptors to the group; all ranks process the same case in
lockstep. This wrapper intentionally refuses per-case model reload fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.common import DataAError, read_json
from scripts.dataa_v1.config import load_config
from scripts.dataa_v1.run_state import RunPaths, RunState
from scripts.dataa_v1.vace_runtime import PersistentVaceRuntime, VaceJob


def _dist_broadcast_object(obj: Any, src: int = 0) -> Any:
    try:
        import torch.distributed as dist
    except ImportError as exc:
        raise DataAError("torch.distributed is required for persistent VACE worker") from exc
    holder = [obj]
    dist.broadcast_object_list(holder, src=src)
    return holder[0]


def run_worker(*, config_path: Path, shard_path: Path, worker_id: int, run_id: str) -> int:
    config = load_config(config_path)
    shard = read_json(shard_path)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    runtime = PersistentVaceRuntime(config["vace"])
    runtime.initialize_once()
    paths = RunPaths.from_root(Path(config["run"]["tmp_root"]), run_id)
    state = RunState(paths, run_id=run_id, topology=shard.get("topology", {}))
    cases = shard.get("cases", [])
    for case in cases:
        descriptor = case if rank == 0 else None
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            descriptor = _dist_broadcast_object(descriptor, src=0)
        job = VaceJob(**descriptor["vace_job"])
        if rank == 0:
            state.append_status(job.case_id, "generation_started", worker_id=worker_id)
        result = runtime.generate_job(job)
        if rank == 0:
            state.append_status(job.case_id, "generated", worker_id=worker_id, detail=result)
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--shard", required=True, type=Path)
    parser.add_argument("--worker-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run_worker(config_path=args.config, shard_path=args.shard, worker_id=args.worker_id, run_id=args.run_id)
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
