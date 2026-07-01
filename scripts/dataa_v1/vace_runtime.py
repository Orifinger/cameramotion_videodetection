"""Persistent VACE runtime adapter.

This wrapper keeps the production contract explicit: a worker process must load
VACE/Wan once and then consume multiple assigned case descriptors. It never
falls back to launching one official CLI per case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys
from typing import Any, Dict, Mapping, Optional

from .common import DataAError, utc_now_iso


@dataclass
class VaceJob:
    case_id: str
    source_clip: str
    target_mask_gen_video: str
    model_prompt: str
    output_path: str
    donor_reference: Optional[str] = None
    frame_count: int = 81
    output_fps: float = 16.0
    size: str = "720p"
    seed: int = 20260629


class PersistentVaceRuntime:
    """Server-side runtime adapter for one torchrun worker group."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.initialized = False
        self._wan_vace = None
        self._cfg = None
        self._size_configs = None
        self._cache_video = None
        self._rank = 0
        self._world_size = 1
        self._device = 0
        self._attention_backend = "default"

    def initialize_once(self) -> None:
        if self.initialized:
            return
        repo_dir = Path(str(self.config.get("repo_dir", "third_party/VACE")))
        checkpoint_dir = Path(str(self.config.get("checkpoint_dir") or ""))
        if not (repo_dir / "vace" / "vace_wan_inference.py").is_file():
            raise DataAError(f"VACE source missing: {repo_dir}")
        if not checkpoint_dir.is_dir():
            raise DataAError(f"VACE checkpoint_dir missing: {checkpoint_dir}")

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        for item in (repo_dir / "vace", repo_dir):
            value = str(item.resolve())
            if value not in sys.path:
                sys.path.insert(0, value)

        try:
            import torch
            import torch.distributed as dist
            from wan.utils.utils import cache_video
            from models.wan import WanVace
            from models.wan.configs import SIZE_CONFIGS, WAN_CONFIGS
        except Exception as exc:  # noqa: BLE001
            raise DataAError(
                "vace_runtime_import_failed: install Wan2.1/VACE runtime dependencies on the server; "
                f"original error: {exc}"
            ) from exc

        if bool(self.config.get("force_flash_attn_2", True)):
            try:
                import wan.modules.attention as wan_attention
            except Exception as exc:  # noqa: BLE001
                raise DataAError(f"vace_runtime_import_failed: cannot inspect Wan attention backend: {exc}") from exc
            if not bool(getattr(wan_attention, "FLASH_ATTN_2_AVAILABLE", False)):
                raise DataAError(
                    "vace_runtime_missing_flash_attn_2: force_flash_attn_2 is enabled but Wan cannot import FlashAttention 2"
                )
            if bool(getattr(wan_attention, "FLASH_ATTN_3_AVAILABLE", False)):
                wan_attention.FLASH_ATTN_3_AVAILABLE = False
                self._attention_backend = "flash_attn_2_forced"
            else:
                self._attention_backend = "flash_attn_2"

        model_name = str(self.config.get("model_name", "vace-14B"))
        if model_name not in WAN_CONFIGS:
            raise DataAError(f"unsupported VACE model_name: {model_name}")
        use_prompt_extend = self.config.get("use_prompt_extend", "plain")
        if use_prompt_extend not in (None, "", "plain"):
            raise DataAError("blocked_prompt_extend: Data A prompts are frozen; VACE prompt extension must stay plain")
        if bool(self.config.get("offload_model", False)):
            raise DataAError("blocked_slow_memory_mode: offload_model must be false for Data A production")
        if bool(self.config.get("t5_cpu", False)):
            raise DataAError("blocked_slow_memory_mode: t5_cpu must be false for Data A production")

        rank = int(os.getenv("RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        ulysses_size = int(self.config.get("ulysses_size", 4))
        ring_size = int(self.config.get("ring_size", 1))
        t5_fsdp = bool(self.config.get("t5_fsdp", True))
        dit_fsdp = bool(self.config.get("dit_fsdp", True))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if world_size > 1:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
        elif t5_fsdp or dit_fsdp or ulysses_size > 1 or ring_size > 1:
            raise DataAError("distributed VACE flags require torchrun WORLD_SIZE > 1")

        if ulysses_size > 1 or ring_size > 1:
            if ulysses_size * ring_size != world_size:
                raise DataAError(
                    f"invalid VACE parallelism: ulysses_size({ulysses_size}) * ring_size({ring_size}) != WORLD_SIZE({world_size})"
                )
            try:
                from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
            except Exception as exc:  # noqa: BLE001
                raise DataAError(f"vace_runtime_import_failed: xfuser is required for Ulysses parallelism: {exc}") from exc
            init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=ring_size,
                ulysses_degree=ulysses_size,
            )

        cfg = WAN_CONFIGS[model_name]
        if ulysses_size > 1 and cfg.num_heads % ulysses_size != 0:
            raise DataAError(f"`num_heads` must be divisible by ulysses_size={ulysses_size}")

        self._wan_vace = WanVace(
            config=cfg,
            checkpoint_dir=str(checkpoint_dir),
            device_id=local_rank,
            rank=rank,
            t5_fsdp=t5_fsdp,
            dit_fsdp=dit_fsdp,
            use_usp=(ulysses_size > 1 or ring_size > 1),
            t5_cpu=bool(self.config.get("t5_cpu", False)),
        )
        self._torch = torch
        self._dist = dist
        self._cfg = cfg
        self._size_configs = SIZE_CONFIGS
        self._cache_video = cache_video
        self._rank = rank
        self._world_size = world_size
        self._device = local_rank
        self.initialized = True

    def generate_job(self, job: VaceJob) -> Dict[str, Any]:
        if not self.initialized:
            raise DataAError("persistent VACE runtime is not initialized")
        assert self._wan_vace is not None
        assert self._cfg is not None
        assert self._size_configs is not None
        assert self._cache_video is not None
        for label, raw_path in (("source_clip", job.source_clip), ("target_mask_gen_video", job.target_mask_gen_video)):
            if not Path(raw_path).is_file():
                raise DataAError(f"blocked_vace_generation_failure: {label} does not exist for {job.case_id}: {raw_path}")
        if job.donor_reference and not Path(job.donor_reference).is_file():
            raise DataAError(f"blocked_vace_generation_failure: donor_reference does not exist for {job.case_id}: {job.donor_reference}")
        if not job.model_prompt:
            raise DataAError(f"blocked_vace_generation_failure: model_prompt is empty for {job.case_id}")
        if job.size not in self._size_configs:
            raise DataAError(f"blocked_vace_generation_failure: unsupported VACE size '{job.size}' for {job.case_id}")
        if job.frame_count <= 0 or job.frame_count % 4 != 1:
            raise DataAError(
                f"blocked_vace_generation_failure: frame_count must be 4n+1 for {job.case_id}, got {job.frame_count}"
            )
        if job.output_fps <= 0:
            raise DataAError(f"blocked_vace_generation_failure: output_fps must be positive for {job.case_id}, got {job.output_fps}")

        refs = None if not job.donor_reference else [job.donor_reference]
        torch = self._torch
        with torch.no_grad():
            src_video, src_mask, src_ref_images = self._wan_vace.prepare_source(
                [job.source_clip],
                [job.target_mask_gen_video],
                [refs],
                job.frame_count,
                self._size_configs[job.size],
                self._device,
            )
            video = self._wan_vace.generate(
                job.model_prompt,
                src_video,
                src_mask,
                src_ref_images,
                size=self._size_configs[job.size],
                frame_num=job.frame_count,
                shift=float(self.config.get("sample_shift", 16)),
                sample_solver=str(self.config.get("sample_solver", "unipc")),
                sampling_steps=int(self.config.get("sample_steps", 50)),
                guide_scale=float(self.config.get("sample_guide_scale", 5.0)),
                seed=int(job.seed),
                offload_model=bool(self.config.get("offload_model", False)),
            )

        output = {
            "status": "generated",
            "case_id": job.case_id,
            "output_path": job.output_path,
            "rank": self._rank,
            "world_size": self._world_size,
            "attention_backend": self._attention_backend,
            "output_fps": float(job.output_fps),
            "completed_at_utc": utc_now_iso(),
        }
        if self._rank == 0:
            Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)
            self._cache_video(
                tensor=video[None],
                save_file=job.output_path,
                fps=float(job.output_fps),
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
        if self._world_size > 1 and self._dist.is_initialized():
            self._dist.barrier()
        return output


def build_single_job_args(job: VaceJob, config: Mapping[str, Any]) -> Dict[str, Any]:
    args = {
        "model_name": config.get("model_name", "vace-14B"),
        "size": job.size,
        "frame_num": job.frame_count,
        "output_fps": job.output_fps,
        "ckpt_dir": config.get("checkpoint_dir"),
        "offload_model": False,
        "ulysses_size": int(config.get("ulysses_size", 4)),
        "ring_size": int(config.get("ring_size", 1)),
        "t5_fsdp": bool(config.get("t5_fsdp", True)),
        "dit_fsdp": bool(config.get("dit_fsdp", True)),
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
