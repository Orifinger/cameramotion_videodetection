"""Automatic target clip selection from a SAM3 visible mask tube."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .common import DataAError
from .mask_io import MaskTube


@dataclass(frozen=True)
class VaceProfile:
    name: str = "production_720"
    fps: float = 16.0
    frame_options: tuple[int, ...] = (81, 65, 49)
    landscape_size: tuple[int, int] = (720, 1280)
    portrait_size: tuple[int, int] = (1280, 720)

    @property
    def seconds_by_frames(self) -> Dict[int, int]:
        return {49: 3, 65: 4, 81: 5}


def profile_from_name(name: str) -> VaceProfile:
    if name == "production_720":
        return VaceProfile()
    if name == "production_480":
        return VaceProfile(name="production_480", frame_options=(81, 65, 49), landscape_size=(480, 832), portrait_size=(832, 480))
    if name == "smoke_480":
        return VaceProfile(name="smoke_480", frame_options=(81, 65, 49), landscape_size=(480, 832), portrait_size=(832, 480))
    raise DataAError(f"unknown VACE profile: {name}")


@dataclass
class ClipSelection:
    source_start_frame: int
    source_end_frame: int
    duration_seconds: float
    canonical_fps: float
    canonical_frame_count: int
    source_fps: float
    canonical_to_source_frames: List[float]
    selection_meta: Dict[str, Any]


def contiguous_runs(frame_indices: np.ndarray, *, max_gap: int = 1) -> List[tuple[int, int, int]]:
    if len(frame_indices) == 0:
        return []
    runs: List[tuple[int, int, int]] = []
    start = prev = int(frame_indices[0])
    count = 1
    for value in frame_indices[1:]:
        current = int(value)
        if current - prev <= max_gap:
            count += 1
        else:
            runs.append((start, prev, count))
            start, count = current, 1
        prev = current
    runs.append((start, prev, count))
    return runs


def _mask_area_score(masks: np.ndarray) -> float:
    areas = masks.reshape(masks.shape[0], -1).mean(axis=1)
    return float(areas.mean() + 0.5 * areas.min())


def next_4n_plus_1(frame_count: int) -> int:
    if frame_count <= 0:
        raise DataAError(f"frame_count must be positive for 4n+1 padding, got {frame_count}")
    remainder = (frame_count - 1) % 4
    if remainder == 0:
        return frame_count
    return frame_count + (4 - remainder)


def select_clip(
    tube: MaskTube,
    *,
    source_fps: float,
    profile: VaceProfile | None = None,
    max_gap: int = 1,
    min_padded_visible_seconds: float = 1.0,
) -> ClipSelection:
    if source_fps <= 0:
        raise DataAError(f"source_fps must be positive, got {source_fps}")
    profile = profile or VaceProfile()
    if tube.frame_indices.size == 0:
        raise DataAError("blocked_low_visibility: no visible target frames")
    source_start = int(tube.frame_indices[0])
    source_end = int(tube.frame_indices[-1])
    source_span = source_end - source_start + 1
    if source_span <= 0:
        raise DataAError("blocked_low_visibility: invalid visible target envelope")
    source_duration = float(source_span / source_fps)
    if source_duration <= 5.0:
        valid_frames = max(1, int(round(source_duration * float(profile.fps))))
        canonical_frames = next_4n_plus_1(valid_frames)
        generation_fps = float(profile.fps)
        clip_policy = "first_visible_to_last_visible_pad_to_nearest_4n_plus_1"
        pad_mode = "repeat_last_frame" if canonical_frames > valid_frames else "none"
    else:
        canonical_frames = 81
        valid_frames = 81
        generation_fps = float(canonical_frames / source_duration)
        clip_policy = "first_visible_to_last_visible_uniform_81"
        pad_mode = "none"
    runs = contiguous_runs(tube.frame_indices, max_gap=max_gap)
    chosen = {
        "source_start_frame": source_start,
        "source_end_frame": source_end,
        "source_start_time_sec": float(source_start / source_fps),
        "source_end_time_sec": float((source_end + 1) / source_fps),
        "duration_seconds": source_duration,
        "canonical_frame_count": canonical_frames,
        "valid_canonical_frame_count": valid_frames,
        "pad_canonical_frames": canonical_frames - valid_frames,
        "pad_mode": pad_mode,
        "padded_short_clip": canonical_frames > valid_frames,
        "visible_first_frame": source_start,
        "visible_last_frame": source_end,
        "visible_source_frames": int(tube.frame_indices.shape[0]),
        "envelope_source_frames": int(source_span),
        "visibility_gap_frame_count": int(source_span - tube.frame_indices.shape[0]),
        "visible_runs": [
            {"start_frame": int(run_start), "end_frame": int(run_end), "visible_frame_count": int(count)}
            for run_start, run_end, count in runs
        ],
        "max_gap": max_gap,
        "score": float(_mask_area_score(tube.masks)),
        "hard_cut_check": "not_evaluated_in_synthetic_scaffold",
        "clip_policy": clip_policy,
        "generation_fps": generation_fps,
    }
    source_start = int(chosen["source_start_frame"])
    source_end = int(chosen["source_end_frame"])
    canonical_frames = int(chosen["canonical_frame_count"])
    valid_frames = int(chosen.get("valid_canonical_frame_count") or canonical_frames)
    if canonical_frames % 4 != 1:
        raise DataAError(f"internal clip selection error: canonical_frame_count is not 4n+1: {canonical_frames}")
    if valid_frames > canonical_frames:
        raise DataAError(
            "internal clip selection error: "
            f"valid_canonical_frame_count({valid_frames}) exceeds canonical_frame_count({canonical_frames})"
        )
    if valid_frames <= 1:
        valid_mapping = np.array([source_start], dtype=np.float64)
    else:
        valid_mapping = np.linspace(source_start, source_end, valid_frames, dtype=np.float64)
    if valid_frames < canonical_frames:
        pad = np.full(canonical_frames - valid_frames, source_end, dtype=np.float64)
        canonical_to_source = np.concatenate([valid_mapping, pad])
    else:
        canonical_to_source = valid_mapping
    return ClipSelection(
        source_start_frame=source_start,
        source_end_frame=source_end,
        duration_seconds=float(chosen["duration_seconds"]),
        canonical_fps=float(chosen["generation_fps"]),
        canonical_frame_count=canonical_frames,
        source_fps=float(source_fps),
        canonical_to_source_frames=[float(v) for v in canonical_to_source],
        selection_meta=chosen,
    )
