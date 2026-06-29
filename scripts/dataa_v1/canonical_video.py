"""Canonical video planning for Stage P.

Real media creation is deliberately gated by the caller. Dry-run packaging only
records the intended source/canonical paths and frame mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .clip_selection import ClipSelection, VaceProfile
from .common import DataAError
from .media_io import export_canonical_source_clip, export_source_real_raw, ffprobe_video


def canonical_video_plan(attempt_dir: Path, clip: ClipSelection, profile: VaceProfile, *, source_video_path: str | None) -> Dict[str, Any]:
    height, width = profile.landscape_size
    valid_frame_count = int(clip.selection_meta.get("valid_canonical_frame_count") or clip.canonical_frame_count)
    pad_frame_count = int(clip.selection_meta.get("pad_canonical_frames") or 0)
    pad_mode = str(clip.selection_meta.get("pad_mode") or "none")
    return {
        "source_video_path": source_video_path,
        "source_real_raw_path": str(attempt_dir / "source_real_raw.mp4"),
        "source_clip_path": str(attempt_dir / "source_clip.mp4"),
        "native": {
            "start_frame": clip.source_start_frame,
            "end_frame": clip.source_end_frame,
            "source_fps": clip.source_fps,
            "start_time_sec": float(clip.source_start_frame / clip.source_fps),
            "end_time_sec": float((clip.source_end_frame + 1) / clip.source_fps),
            "duration_seconds": float(clip.duration_seconds),
        },
        "canonical": {
            "fps": float(clip.canonical_fps),
            "generation_fps": float(clip.canonical_fps),
            "frame_count": clip.canonical_frame_count,
            "valid_frame_count": valid_frame_count,
            "pad_frame_count": pad_frame_count,
            "pad_mode": pad_mode,
            "padded_short_clip": bool(clip.selection_meta.get("padded_short_clip", False)),
            "crop_generated_to_valid_frames": pad_frame_count > 0,
            "height": height,
            "width": width,
            "profile": profile.name,
            "source_duration_sec": float(clip.duration_seconds),
            "frame_mapping": [
                {"canonical_frame": i, "source_frame_float": frame}
                for i, frame in enumerate(clip.canonical_to_source_frames)
            ],
        },
        "selection_meta": clip.selection_meta,
        "media_created": False,
        "note": "dry-run scaffold records planned canonical media; server execution must render actual mp4 files",
    }


def export_canonical_videos(
    *,
    attempt_dir: Path,
    source_video_path: str,
    clip: ClipSelection,
    profile: VaceProfile,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> Dict[str, Any]:
    height, width = profile.landscape_size
    raw_path = attempt_dir / "source_real_raw.mp4"
    clip_path = attempt_dir / "source_clip.mp4"
    raw = export_source_real_raw(
        source_video=Path(source_video_path),
        out_path=raw_path,
        start_frame=clip.source_start_frame,
        end_frame=clip.source_end_frame,
        source_fps=clip.source_fps,
        ffmpeg_bin=ffmpeg_bin,
    )
    canonical = export_canonical_source_clip(
        source_video=Path(source_video_path),
        out_path=clip_path,
        start_frame=clip.source_start_frame,
        end_frame=clip.source_end_frame,
        source_fps=clip.source_fps,
        canonical_fps=clip.canonical_fps,
        frame_count=clip.canonical_frame_count,
        height=height,
        width=width,
        ffmpeg_bin=ffmpeg_bin,
    )
    meta = ffprobe_video(clip_path, ffprobe_bin=ffprobe_bin)
    if meta.frame_count != clip.canonical_frame_count or round(meta.fps, 6) != round(clip.canonical_fps, 6) or meta.height != height or meta.width != width:
        raise DataAError(f"canonical source_clip validation failed: {meta}")
    return {"source_real_raw": raw, "source_clip": canonical, "source_clip_meta": meta, "media_created": True}
