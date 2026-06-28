"""Canonical video planning for Stage P.

Real media creation is deliberately gated by the caller. Dry-run packaging only
records the intended source/canonical paths and frame mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .clip_selection import ClipSelection, VaceProfile


def canonical_video_plan(attempt_dir: Path, clip: ClipSelection, profile: VaceProfile, *, source_video_path: str | None) -> Dict[str, Any]:
    height, width = profile.landscape_size
    return {
        "source_video_path": source_video_path,
        "source_real_raw_path": str(attempt_dir / "source_real_raw.mp4"),
        "source_clip_path": str(attempt_dir / "source_clip.mp4"),
        "native": {
            "start_frame": clip.source_start_frame,
            "end_frame": clip.source_end_frame,
            "source_fps": clip.source_fps,
        },
        "canonical": {
            "fps": clip.canonical_fps,
            "frame_count": clip.canonical_frame_count,
            "height": height,
            "width": width,
            "profile": profile.name,
            "frame_mapping": [
                {"canonical_frame": i, "source_frame_float": frame}
                for i, frame in enumerate(clip.canonical_to_source_frames)
            ],
        },
        "media_created": False,
        "note": "dry-run scaffold records planned canonical media; server execution must render actual mp4 files",
    }

