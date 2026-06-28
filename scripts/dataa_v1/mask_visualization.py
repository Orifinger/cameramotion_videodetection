"""Mask visualization manifest helpers.

The existing scripts/render_dataa_v1_mask_videos.py remains the visual renderer.
This module records planned visualization paths so Stage P manifests are complete
without silently creating compressed QA media in dry-run mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def planned_mask_visualizations(attempt_dir: Path) -> Dict[str, str]:
    return {
        "target_mask_raw_video": str(attempt_dir / "target_mask_raw.mp4"),
        "target_mask_overlay_video": str(attempt_dir / "target_mask_overlay.mp4"),
        "donor_mask_raw_video": str(attempt_dir / "donor_mask_raw.mp4"),
        "donor_mask_overlay_video": str(attempt_dir / "donor_mask_overlay.mp4"),
        "created": "false_in_dry_run",
    }

