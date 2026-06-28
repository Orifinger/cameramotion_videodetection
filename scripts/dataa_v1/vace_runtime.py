"""Persistent VACE runtime adapter.

This wrapper keeps the production contract explicit: a worker process must load
VACE/Wan once and then consume multiple assigned case descriptors. It never
falls back to launching one official CLI per case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .common import DataAError


@dataclass
class VaceJob:
    case_id: str
    source_clip: str
    target_mask_gen_video: str
    model_prompt: str
    output_path: str
    donor_reference: Optional[str] = None
    frame_count: int = 81
    size: str = "720p"
    seed: int = 20260629


class PersistentVaceRuntime:
    """Server-side runtime adapter for one torchrun worker group."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.initialized = False
        self._wan_vace = None

    def initialize_once(self) -> None:
        if self.initialized:
            return
        repo_dir = Path(str(self.config.get("repo_dir", "third_party/VACE")))
        checkpoint_dir = Path(str(self.config.get("checkpoint_dir") or ""))
        if not (repo_dir / "vace" / "vace_wan_inference.py").is_file():
            raise DataAError(f"VACE source missing: {repo_dir}")
        if not checkpoint_dir.is_dir():
            raise DataAError(f"VACE checkpoint_dir missing: {checkpoint_dir}")

        # The official VACE code exposes model construction in vace_wan_inference.py
        # and WanVace.generate, but not a stable queue API. The server adapter is
        # deliberately explicit: integration must wire the imported WanVace object
        # here once, then call generate_job repeatedly. If that import path changes,
        # fail before generation rather than degrading to per-case CLI reloads.
        self.initialized = True

    def generate_job(self, job: VaceJob) -> Dict[str, Any]:
        if not self.initialized:
            raise DataAError("persistent VACE runtime is not initialized")
        raise DataAError(
            "vace_runtime_adapter_not_bound: integrate WanVace.prepare_source/generate/cache_video here on the server; "
            "per-case torchrun/CLI fallback is forbidden"
        )


def build_single_job_args(job: VaceJob, config: Mapping[str, Any]) -> Dict[str, Any]:
    args = {
        "model_name": config.get("model_name", "vace-14B"),
        "size": job.size,
        "frame_num": job.frame_count,
        "ckpt_dir": config.get("checkpoint_dir"),
        "offload_model": False,
        "ulysses_size": int(config.get("ulysses_size", 4)),
        "ring_size": int(config.get("ring_size", 1)),
        "t5_fsdp": True,
        "dit_fsdp": True,
        "save_file": job.output_path,
        "src_video": job.source_clip,
        "src_mask": job.target_mask_gen_video,
        "src_ref_images": job.donor_reference,
        "prompt": job.model_prompt,
        "use_prompt_extend": config.get("use_prompt_extend", "plain"),
        "base_seed": job.seed,
        "sample_solver": config.get("sample_solver", "unipc"),
        "sample_steps": config.get("sample_steps", 50),
        "sample_shift": config.get("sample_shift", 16),
        "sample_guide_scale": config.get("sample_guide_scale", 5.0),
    }
    return args
