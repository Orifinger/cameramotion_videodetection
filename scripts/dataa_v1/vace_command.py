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
) -> Dict[str, Any]:
    argv = [
        "python",
        str(Path(vace_repo_dir) / "scripts" / "vace_inference.py"),
        "--src_video",
        source_clip,
        "--src_mask",
        mask_video,
        "--prompt",
        prompt,
        "--output_dir",
        output_dir,
    ]
    if donor_reference:
        argv.extend(["--src_ref_images", donor_reference])
    return {
        "case_id": case_id,
        "dry_run": dry_run,
        "will_execute_vace": False,
        "argv": argv,
        "blocked_until_server_runtime": [
            "Wan2.1 offline dependency installed",
            "VACE-14B weights available",
            "source_clip.mp4 and target_mask_gen.mp4 physically rendered",
        ],
    }

