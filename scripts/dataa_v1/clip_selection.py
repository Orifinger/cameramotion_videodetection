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
    fps: int = 16
    frame_options: tuple[int, ...] = (81, 65, 49)
    landscape_size: tuple[int, int] = (720, 1280)
    portrait_size: tuple[int, int] = (1280, 720)

    @property
    def seconds_by_frames(self) -> Dict[int, int]:
        return {49: 3, 65: 4, 81: 5}


def profile_from_name(name: str) -> VaceProfile:
    if name == "production_720":
        return VaceProfile()
    if name == "smoke_480":
        return VaceProfile(name="smoke_480", frame_options=(81, 65, 49), landscape_size=(480, 832), portrait_size=(832, 480))
    raise DataAError(f"unknown VACE profile: {name}")


@dataclass
class ClipSelection:
    source_start_frame: int
    source_end_frame: int
    duration_seconds: float
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
                        "valid_canonical_frame_count": frame_count,
                        "pad_canonical_frames": 0,
                        "pad_mode": "none",
                        "padded_short_clip": False,
                        "run_start_frame": run_start,
                        "run_end_frame": run_end,
                        "visible_source_frames": required_source_span,
                        "max_gap": max_gap,
                        "score": score,
                        "hard_cut_check": "not_evaluated_in_synthetic_scaffold",
                        "clip_policy": "full_visible_window",
                    },
                )
            )
        if candidates:
            break

    if not candidates:
        min_source_span = max(1, int(round(min_padded_visible_seconds * source_fps)))
        viable_runs = [run for run in runs if run[2] >= min_source_span]
        if not viable_runs:
            raise DataAError(
                "blocked_low_visibility: "
                f"no >={min_padded_visible_seconds:.2f} second visible window for padded VACE input"
            )
        run_start, run_end, run_count = viable_runs[0]
        positions = [lookup[f] for f in range(run_start, run_end + 1) if f in lookup]
        if len(positions) != run_count:
            raise DataAError("blocked_low_visibility: visible run has missing frame indices")
        valid_canonical_frames = max(1, int(round((run_count / source_fps) * profile.fps)))
        padded_canonical_frames = next_4n_plus_1(valid_canonical_frames)
        score = float(10.0 + _mask_area_score(tube.masks[positions]))
        candidates.append(
            (
                score,
                {
                    "source_start_frame": run_start,
                    "source_end_frame": run_end,
                    "duration_seconds": float(run_count / source_fps),
                    "canonical_frame_count": padded_canonical_frames,
                    "valid_canonical_frame_count": valid_canonical_frames,
                    "pad_canonical_frames": padded_canonical_frames - valid_canonical_frames,
                    "pad_mode": "repeat_last_frame",
                    "padded_short_clip": True,
                    "run_start_frame": run_start,
                    "run_end_frame": run_end,
                    "visible_source_frames": run_count,
                    "max_gap": max_gap,
                    "score": score,
                    "hard_cut_check": "not_evaluated_in_synthetic_scaffold",
                    "clip_policy": "short_visible_segment_pad_to_nearest_4n_plus_1",
                    "short_clip_min_visible_seconds": float(min_padded_visible_seconds),
                    "short_clip_reason": "no 3-5 second fully visible window",
                },
            )
        )

    chosen = max(candidates, key=lambda item: item[0])[1]
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
        canonical_fps=profile.fps,
        canonical_frame_count=canonical_frames,
        source_fps=float(source_fps),
        canonical_to_source_frames=[float(v) for v in canonical_to_source],
        selection_meta=chosen,
    )
