"""Production configuration loading for Data A v1 VACE runs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from .common import DataAError, read_json


DEFAULT_CONFIG: Dict[str, Any] = {
    "run": {
        "run_id": None,
        "tmp_root": "/tmp/cameramotion_det/dataA_v1/vace14b",
        "oss_prefix": "oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/dataA_v1/vace14b",
    },
    "execution": {
        "full_execution_plan": "res/dataA_v1/plans/frozen_full_vace_execution_plan.json",
        "resume": True,
        "strict": True,
        "block_invalid_cases": True,
        "allow_reshard": False,
    },
    "vace": {
        "repo_dir": "third_party/VACE",
        "checkpoint_dir": "/home/admin/wan2.1-VACE",
        "model_name": "vace-14B",
        "profile": "production_720",
        "size": "720p",
        "seed": 20260629,
        "ffmpeg_bin": "/input/tmp/ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg",
        "ffprobe_bin": "/input/tmp/ffmpeg/ffmpeg-7.0.2-amd64-static/ffprobe",
        "torchrun_bin": "torchrun",
        "sample_solver": "unipc",
        "sample_steps": 50,
        "sample_shift": 16,
        "sample_guide_scale": 5.0,
        "use_prompt_extend": "plain",
        "force_flash_attn_2": True,
    },
    "gpu": {
        "worker_groups": 4,
        "gpus_per_worker": 4,
        "workers_per_gpu": 1,
        "fallback_topologies": [{"worker_groups": 2, "gpus_per_worker": 8}],
        "telemetry_interval_seconds": 10,
        "min_aggregate_vram_ratio": 0.50,
        "enforce_min_aggregate_vram_ratio": True,
    },
    "upload": {
        "enabled": True,
        "upload_command": "ossutil64",
        "every_completed_cases": 8,
        "every_minutes": 30,
        "delete_local_after_verified_upload": True,
        "tmp_free_space_watermark_gb": 500,
        "retry_backoff_seconds": [30, 120, 300, 900],
    },
}


def _merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise DataAError(f"config file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise DataAError(f"invalid YAML config {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DataAError(f"config must be a mapping: {path}")
    return _merge(DEFAULT_CONFIG, payload)


def apply_cli_overrides(
    config: Dict[str, Any],
    *,
    execution_plan: Path | None = None,
    checkpoint_dir: str | None = None,
    run_id: str | None = None,
    oss_prefix: str | None = None,
    resume: bool | None = None,
    allow_reshard: bool | None = None,
    topology: str | None = None,
    workers_per_gpu: int | None = None,
) -> Dict[str, Any]:
    out = deepcopy(config)
    if execution_plan is not None:
        out["execution"]["full_execution_plan"] = str(execution_plan)
    if checkpoint_dir is not None:
        out["vace"]["checkpoint_dir"] = checkpoint_dir
    if run_id is not None:
        out["run"]["run_id"] = run_id
    if oss_prefix is not None:
        out["run"]["oss_prefix"] = oss_prefix
    if resume is not None:
        out["execution"]["resume"] = bool(resume)
    if allow_reshard is not None:
        out["execution"]["allow_reshard"] = bool(allow_reshard)
    if topology:
        try:
            groups, gpus = topology.lower().split("x", 1)
            out["gpu"]["worker_groups"] = int(groups)
            out["gpu"]["gpus_per_worker"] = int(gpus)
        except Exception as exc:  # noqa: BLE001
            raise DataAError(f"invalid topology '{topology}', expected like 4x4 or 2x8") from exc
    if workers_per_gpu is not None:
        if workers_per_gpu <= 0:
            raise DataAError(f"invalid workers_per_gpu '{workers_per_gpu}', expected a positive integer")
        out["gpu"]["workers_per_gpu"] = int(workers_per_gpu)
        out["gpu"].pop("batch_size", None)
    return out
