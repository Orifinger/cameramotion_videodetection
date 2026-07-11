"""Deterministic dense sampling from paired full videos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    fps: float
    frame_count: int
    height: int
    width: int

    @property
    def duration(self) -> float:
        return float(self.frame_count) / self.fps if self.fps > 0 else 0.0


def probe_video(path: Path) -> VideoMeta:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        return VideoMeta(
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
            height=int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            width=int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        )
    finally:
        capture.release()


def dense_sample_indices(meta: VideoMeta, *, target_fps: float) -> list[int]:
    if meta.fps <= 0 or meta.frame_count <= 0 or target_fps <= 0:
        raise ValueError(f"invalid sampling metadata: {meta}, target_fps={target_fps}")
    effective_fps = min(meta.fps, target_fps)
    duration = float(meta.frame_count - 1) / meta.fps
    sample_times = np.arange(0.0, duration + 1e-9, 1.0 / effective_fps)
    indices = sorted({min(meta.frame_count - 1, int(round(value * meta.fps))) for value in sample_times})
    if indices[-1] != meta.frame_count - 1:
        indices.append(meta.frame_count - 1)
    return indices


def read_video_frames(path: Path, frame_indices: Sequence[int]) -> np.ndarray:
    requested = [int(value) for value in frame_indices]
    if not requested:
        raise ValueError("frame_indices must not be empty")
    if requested != sorted(set(requested)):
        raise ValueError("frame_indices must be sorted and unique")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    requested_set = set(requested)
    final_index = requested[-1]
    index = 0
    try:
        while index <= final_index:
            ok, frame = capture.read()
            if not ok:
                break
            if index in requested_set:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            index += 1
    finally:
        capture.release()
    if len(frames) != len(requested):
        raise ValueError(f"decoded frame count mismatch for {path}: expected={len(requested)} got={len(frames)}")
    return np.stack(frames, axis=0)


def paired_dense_frames(
    real_video: Path,
    fake_video: Path,
    *,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, VideoMeta]:
    real_meta = probe_video(real_video)
    fake_meta = probe_video(fake_video)
    if (real_meta.height, real_meta.width) != (fake_meta.height, fake_meta.width):
        raise ValueError(f"paired video shape mismatch: real={real_meta} fake={fake_meta}")
    if abs(real_meta.fps - fake_meta.fps) > 1e-3:
        raise ValueError(f"paired video FPS mismatch: real={real_meta.fps} fake={fake_meta.fps}")
    frame_count = min(real_meta.frame_count, fake_meta.frame_count)
    shared_meta = VideoMeta(real_meta.fps, frame_count, real_meta.height, real_meta.width)
    indices = dense_sample_indices(shared_meta, target_fps=target_fps)
    timestamps = np.asarray(indices, dtype=np.float64) / shared_meta.fps
    return (
        read_video_frames(real_video, indices),
        read_video_frames(fake_video, indices),
        timestamps,
        shared_meta,
    )


def sliding_windows(
    frame_count: int,
    *,
    window_frames: int,
    stride_frames: int,
    max_windows: int | None = None,
) -> list[tuple[int, int]]:
    if frame_count < 2:
        return []
    window = min(max(2, window_frames), frame_count)
    stride = max(1, stride_frames)
    starts = list(range(0, max(1, frame_count - window + 1), stride))
    final_start = frame_count - window
    if starts[-1] != final_start:
        starts.append(final_start)
    windows = [(start, start + window) for start in sorted(set(starts))]
    if max_windows is not None and max_windows > 0 and len(windows) > max_windows:
        positions = np.linspace(0, len(windows) - 1, max_windows)
        windows = [windows[int(round(position))] for position in positions]
    return windows
