#!/usr/bin/env python3
"""Run a fast all-GPU collective and CUDA compute smoke test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from scripts.camera_binary_vqa.runtime import cleanup_distributed, init_distributed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    rank, local_rank, world_size = init_distributed()
    device = torch.device("cuda", local_rank)
    left = torch.randn((512, 512), device=device, dtype=torch.float32)
    right = torch.randn((512, 512), device=device, dtype=torch.float32)
    product = left @ right
    if not torch.isfinite(product).all():
        raise FloatingPointError(f"non-finite CUDA matmul on rank {rank}")
    collective = torch.tensor(float(rank), device=device)
    dist.all_reduce(collective, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size - 1) / 2.0
    if float(collective.item()) != expected:
        raise RuntimeError(
            f"distributed all-reduce mismatch on rank {rank}: "
            f"expected={expected} actual={float(collective.item())}"
        )
    torch.cuda.synchronize(device)
    if rank == 0:
        output = {
            "smoke": "all-GPU CUDA matmul and distributed all-reduce",
            "status": "passed",
            "world_size": world_size,
            "expected_collective_sum": expected,
            "actual_collective_sum": float(collective.item()),
            "elapsed_seconds": time.time() - started,
        }
        write_json(args.output_json, output)
        print(json.dumps(output, ensure_ascii=False, indent=2))
    cleanup_distributed()


if __name__ == "__main__":
    main()
