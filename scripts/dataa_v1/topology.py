"""GPU topology and deterministic worker sharding."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

from .common import DataAError


@dataclass(frozen=True)
class WorkerGroup:
    worker_id: int
    cuda_visible_devices: tuple[int, ...]
    nproc_per_node: int
    ulysses_size: int
    ring_size: int = 1

    @property
    def label(self) -> str:
        return f"worker_{self.worker_id:02d}"

    def env(self) -> Dict[str, str]:
        return {"CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in self.cuda_visible_devices)}


@dataclass(frozen=True)
class Topology:
    worker_groups: int
    gpus_per_worker: int
    total_gpus: int
    groups: tuple[WorkerGroup, ...]
    is_fallback: bool = False

    @property
    def name(self) -> str:
        return f"{self.worker_groups}x{self.gpus_per_worker}"


def build_topology(config: Mapping[str, Any], *, available_gpu_count: int | None = None) -> Topology:
    worker_groups = int(config.get("worker_groups", 4))
    gpus_per_worker = int(config.get("gpus_per_worker", 4))
    if worker_groups <= 0 or gpus_per_worker <= 0:
        raise DataAError("gpu.worker_groups and gpu.gpus_per_worker must be positive")
    total = worker_groups * gpus_per_worker
    if available_gpu_count is not None and available_gpu_count < total:
        raise DataAError(f"topology {worker_groups}x{gpus_per_worker} requires {total} GPUs, only {available_gpu_count} visible")
    fallback_specs = {(int(item.get("worker_groups")), int(item.get("gpus_per_worker"))) for item in config.get("fallback_topologies", [])}
    groups: List[WorkerGroup] = []
    for worker_id in range(worker_groups):
        start = worker_id * gpus_per_worker
        devices = tuple(range(start, start + gpus_per_worker))
        groups.append(
            WorkerGroup(
                worker_id=worker_id,
                cuda_visible_devices=devices,
                nproc_per_node=gpus_per_worker,
                ulysses_size=gpus_per_worker,
                ring_size=1,
            )
        )
    return Topology(
        worker_groups=worker_groups,
        gpus_per_worker=gpus_per_worker,
        total_gpus=total,
        groups=tuple(groups),
        is_fallback=(worker_groups, gpus_per_worker) in fallback_specs,
    )


def validate_topology_for_resume(previous: Mapping[str, Any] | None, topology: Topology, *, allow_reshard: bool) -> None:
    if not previous:
        return
    previous_name = previous.get("name") or f"{previous.get('worker_groups')}x{previous.get('gpus_per_worker')}"
    if previous_name != topology.name and not allow_reshard:
        raise DataAError(f"resume topology mismatch: previous={previous_name}, current={topology.name}; pass --allow-reshard to override")


def stable_worker_id(case_id: str, run_id: str, topology: Topology) -> int:
    digest = hashlib.sha256(f"{run_id}:{topology.name}:{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % topology.worker_groups


def shard_cases(case_ids: Iterable[str], run_id: str, topology: Topology) -> Dict[int, List[str]]:
    shards: Dict[int, List[str]] = {group.worker_id: [] for group in topology.groups}
    for case_id in sorted(case_ids):
        shards[stable_worker_id(case_id, run_id, topology)].append(case_id)
    return shards


def topology_payload(topology: Topology) -> Dict[str, Any]:
    return {
        "name": topology.name,
        "worker_groups": topology.worker_groups,
        "gpus_per_worker": topology.gpus_per_worker,
        "total_gpus": topology.total_gpus,
        "is_fallback": topology.is_fallback,
        "groups": [
            {
                "worker_id": group.worker_id,
                "label": group.label,
                "cuda_visible_devices": list(group.cuda_visible_devices),
                "nproc_per_node": group.nproc_per_node,
                "ulysses_size": group.ulysses_size,
                "ring_size": group.ring_size,
            }
            for group in topology.groups
        ],
    }
