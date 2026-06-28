"""Construct persistent worker torchrun commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from .topology import WorkerGroup


def build_worker_command(
    *,
    group: WorkerGroup,
    config_path: Path,
    shard_path: Path,
    run_id: str,
    torchrun_bin: str = "torchrun",
) -> Dict[str, Any]:
    argv = [
        torchrun_bin,
        f"--nproc_per_node={group.nproc_per_node}",
        "scripts/dataa_v1/vace_persistent_worker.py",
        "--config",
        str(config_path),
        "--shard",
        str(shard_path),
        "--worker-id",
        str(group.worker_id),
        "--run-id",
        run_id,
    ]
    vace_flags = {
        "dit_fsdp": True,
        "t5_fsdp": True,
        "ulysses_size": group.ulysses_size,
        "ring_size": group.ring_size,
    }
    return {
        "worker_id": group.worker_id,
        "label": group.label,
        "env": group.env(),
        "argv": argv,
        "vace_distributed_flags": vace_flags,
        "contract": {
            "persistent_process": True,
            "model_loads_per_worker_group": 1,
            "per_case_torchrun_forbidden": True,
        },
    }
