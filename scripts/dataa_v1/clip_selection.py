"""Automatic target clip selection from a SAM3 visible mask tube."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np

from .common import DataAError
from .mask_io import MaskTube


@dataclass(frozen=True)
class VaceProfile:
    name: str = "smoke_480"
    fps: int = 16
    frame_options: tuple[int, ...] = (49, 65, 81)
    landscape_size: tuple[int, int] = (480, 832)
    portrait_size: tuple[int, int] = (832, 480)

    @property
    def seconds_by_frames(self) -> Dict[int, int]:
        return {49: 3, 65: 4, 81: 5}


@dataclass
class ClipSelection:
    source_start_frame: int
    source_end_frame: int
    duration_seconds: int
    canonical_fps: int
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


def select_clip(
    tube: MaskTube,
    *,
    source_fps: float,
    profile: VaceProfile | None = None,
    max_gap: int = 1,
) -> ClipSelection:
    if source_fps <= 0:
        raise DataAError(f"source_fps must be positive, got {source_fps}")
    profile = profile or VaceProfile()
    lookup = {int(frame): pos for pos, frame in enumerate(tube.frame_indices)}
    runs = sorted(contiguous_runs(tube.frame_indices, max_gap=max_gap), key=lambda item: (item[2], item[1] - item[0]), reverse=True)
    if not runs:
        raise DataAError("blocked_low_visibility: no visible target frames")

    candidates: List[tuple[float, Dict[str, Any]]] = []
    frame_options = sorted(profile.frame_options, key=lambda frames: profile.seconds_by_frames.get(frames, 0), reverse=True)
    for frame_count in frame_options:
        seconds = profile.seconds_by_frames.get(frame_count)
        if seconds is None:
            continue
        required_source_span = max(1, int(round(seconds * source_fps)))
        for run_start, run_end, _count in runs:
            if run_end - run_start + 1 < required_source_span:
                continue
            source_start = run_start
            source_end = source_start + required_source_span - 1
            if source_end > run_end:
                continue
            positions = [lookup[f] for f in range(source_start, source_end + 1) if f in lookup]
            if len(positions) != required_source_span:
                continue
            score = float(seconds * 100.0 + _mask_area_score(tube.masks[positions]))
            candidates.append(
                (
                    score,
                    {
                        "source_start_frame": source_start,
                        "source_end_frame": source_end,
                        "duration_seconds": seconds,
                        "canonical_frame_count": frame_count,
                        "run_start_frame": run_start,
                        "run_end_frame": run_end,
                        "visible_source_frames": required_source_span,
                        "max_gap": max_gap,
                        "score": score,
                        "hard_cut_check": "not_evaluated_in_synthetic_scaffold",
                    },
                )
            )
        if candidates:
            break

    if not candidates:
        raise DataAError("blocked_low_visibility: no 3-5 second fully visible window")

    chosen = max(candidates, key=lambda item: item[0])[1]
    source_start = int(chosen["source_start_frame"])
    source_end = int(chosen["source_end_frame"])
    canonical_frames = int(chosen["canonical_frame_count"])
    canonical_to_source = np.linspace(source_start, source_end, canonical_frames, dtype=np.float64)
    return ClipSelection(
        source_start_frame=source_start,
        source_end_frame=source_end,
        duration_seconds=int(chosen["duration_seconds"]),
        canonical_fps=profile.fps,
        canonical_frame_count=canonical_frames,
        source_fps=float(source_fps),
        canonical_to_source_frames=[float(v) for v in canonical_to_source],
        selection_meta=chosen,
    )

