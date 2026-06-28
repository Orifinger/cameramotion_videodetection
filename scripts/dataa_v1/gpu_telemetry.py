"""nvidia-smi telemetry parsing and VRAM floor summaries."""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .common import DataAError, utc_now_iso, write_json


NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
    "--format=csv,noheader,nounits",
]


def sample_nvidia_smi(command: str = "nvidia-smi") -> List[Dict[str, Any]]:
    query = [command] + NVIDIA_SMI_QUERY[1:]
    proc = subprocess.run(query, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise DataAError(f"nvidia-smi failed: {proc.stderr.strip()}")
    return parse_nvidia_smi_csv(proc.stdout)


def parse_nvidia_smi_csv(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in csv.reader(StringIO(text)):
        if len(raw) < 5:
            continue
        index, used, total, util, temp = [item.strip() for item in raw[:5]]
        rows.append(
            {
                "timestamp_utc": utc_now_iso(),
                "gpu_index": int(index),
                "memory_used_mib": int(float(used)),
                "memory_total_mib": int(float(total)),
                "utilization_gpu_percent": int(float(util)),
                "temperature_c": int(float(temp)),
            }
        )
    return rows


def aggregate_vram_ratio(samples: Iterable[Mapping[str, Any]]) -> float:
    used = 0
    total = 0
    for row in samples:
        used += int(row.get("memory_used_mib", 0))
        total += int(row.get("memory_total_mib", 0))
    return 0.0 if total <= 0 else used / total


def summarize_telemetry(samples: List[Mapping[str, Any]], *, min_aggregate_vram_ratio: float) -> Dict[str, Any]:
    ratio = aggregate_vram_ratio(samples)
    util_values = [int(row.get("utilization_gpu_percent", 0)) for row in samples]
    return {
        "sample_count": len(samples),
        "aggregate_vram_ratio": ratio,
        "vram_floor": min_aggregate_vram_ratio,
        "vram_floor_met": ratio >= min_aggregate_vram_ratio,
        "average_gpu_compute_utilization_percent": (sum(util_values) / len(util_values)) if util_values else 0.0,
    }


def write_telemetry_summary(path: Path, samples: List[Mapping[str, Any]], *, min_aggregate_vram_ratio: float) -> Dict[str, Any]:
    summary = summarize_telemetry(samples, min_aggregate_vram_ratio=min_aggregate_vram_ratio)
    write_json(path, summary)
    return summary
