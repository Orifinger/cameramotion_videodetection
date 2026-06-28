"""Generate a VACE command specification without executing it."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def build_vace_command_spec(
    *,
    case_id: str,
    vace_repo_dir: str,
    source_clip: str,
    mask_video: str,
    prompt: str,
    output_dir: str,
    donor_reference: Optional[str] = None,
    dry_run: bool = True,
    frame_num: int = 81,
    size: str = "720p",
    checkpoint_dir: Optional[str] = None,
    model_name: str = "vace-14B",
) -> Dict[str, Any]:
    argv = [
        "python",
        str(Path(vace_repo_dir) / "vace" / "vace_wan_inference.py"),
        "--model_name",
        model_name,
        "--size",
        size,
        "--frame_num",
        str(frame_num),
        "--src_video",
        source_clip,
        "--src_mask",
        mask_video,
        "--prompt",
        prompt,
        "--save_dir",
        output_dir,
        "--save_file",
        str(Path(output_dir) / "generated_raw.mp4"),
        "--dit_fsdp",
        "--t5_fsdp",
        "--ulysses_size",
        "4",
        "--ring_size",
        "1",
    ]
    if checkpoint_dir:
        argv.extend(["--ckpt_dir", checkpoint_dir])
    if donor_reference:
        argv.extend(["--src_ref_images", donor_reference])
    return {
        "case_id": case_id,
        "dry_run": dry_run,
        "will_execute_vace": False,
        "argv": argv,
        "persistent_worker_contract": {
            "default_topology": "4x4",
            "nproc_per_node": 4,
            "ulysses_size": 4,
            "ring_size": 1,
            "dit_fsdp": True,
            "t5_fsdp": True,
            "per_case_torchrun_forbidden": True,
        },
        "blocked_until_server_runtime": [
            "Wan2.1 offline dependency installed",
            "VACE-14B weights available",
            "source_clip.mp4 and target_mask_gen.mp4 physically rendered",
        ],
    }
