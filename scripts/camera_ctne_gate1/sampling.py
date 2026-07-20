"""Pure variable-length sampling helpers (no model runtime imports)."""

from __future__ import annotations

import numpy as np


def uniform_frame_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count < 0 or max_frames < 0:
        raise ValueError("frame_count and max_frames must be non-negative")
    if max_frames == 0 or frame_count <= max_frames:
        return list(range(frame_count))
    if max_frames < 3:
        raise ValueError("an explicit max_frames must be at least 3")
    values = np.linspace(0, frame_count - 1, num=max_frames)
    indices = [int(round(value)) for value in values]
    if len(set(indices)) != max_frames:
        raise RuntimeError(f"uniform sampling produced duplicate indices: n={frame_count} max={max_frames}")
    return indices


def frame_chunks(frame_count: int, chunk_frames: int) -> list[tuple[int, int]]:
    if frame_count < 2:
        return []
    if chunk_frames < 2:
        raise ValueError("chunk_frames must be at least 2")
    output: list[tuple[int, int]] = []
    start = 0
    while start < frame_count - 1:
        end = min(frame_count, start + chunk_frames)
        output.append((start, end))
        if end == frame_count:
            break
        start = end - 1
    if sum(end - start - 1 for start, end in output) != frame_count - 1:
        raise AssertionError("chunking changed the number of adjacent transitions")
    return output
