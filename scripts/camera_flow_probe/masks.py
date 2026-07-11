"""Map VACE canonical mask tubes onto full-video timestamps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


@dataclass
class MaskTube:
    times: np.ndarray
    masks: np.ndarray
    max_time_error: float

    def sample(self, timestamp: float, *, height: int, width: int) -> np.ndarray:
        if self.times.size == 0:
            return np.zeros((height, width), dtype=np.uint8)
        index = int(np.argmin(np.abs(self.times - float(timestamp))))
        if abs(float(self.times[index]) - float(timestamp)) > self.max_time_error:
            return np.zeros((height, width), dtype=np.uint8)
        mask = self.masks[index].astype(np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8)


def _read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"case manifest must be an object: {path}")
    return value


def _canonical_times(manifest: Mapping[str, Any], canonical_frames: Sequence[int]) -> tuple[np.ndarray, float]:
    source_clip = manifest.get("source_clip") if isinstance(manifest.get("source_clip"), Mapping) else {}
    native = source_clip.get("native") if isinstance(source_clip.get("native"), Mapping) else {}
    canonical = source_clip.get("canonical") if isinstance(source_clip.get("canonical"), Mapping) else {}
    source_fps = float(native.get("source_fps") or 0.0)
    generation_fps = float(canonical.get("generation_fps") or canonical.get("fps") or 0.0)
    start_time = float(native.get("start_time_sec") or 0.0)
    mapping = canonical.get("frame_mapping") if isinstance(canonical.get("frame_mapping"), list) else []
    by_frame = {
        int(item["canonical_frame"]): item
        for item in mapping
        if isinstance(item, Mapping) and item.get("canonical_frame") is not None
    }
    times: list[float] = []
    for frame in canonical_frames:
        item = by_frame.get(int(frame))
        if item is not None and source_fps > 0 and item.get("source_frame_float") is not None:
            times.append(float(item["source_frame_float"]) / source_fps)
        elif generation_fps > 0:
            times.append(start_time + float(frame) / generation_fps)
        else:
            times.append(float(frame))
    cadence = 1.0 / generation_fps if generation_fps > 0 else 1.0
    return np.asarray(times, dtype=np.float64), max(0.51 * cadence, 1e-4)


def load_mask_tube(mask_npz: Path, case_manifest: Path) -> MaskTube:
    manifest = _read_manifest(case_manifest)
    with np.load(mask_npz, allow_pickle=False) as archive:
        if "masks" not in archive:
            raise ValueError(f"mask NPZ is missing masks: {mask_npz}")
        masks = (archive["masks"] > 0).astype(np.uint8)
        frames = (
            archive["frame_indices"].astype(np.int64)
            if "frame_indices" in archive
            else np.arange(masks.shape[0], dtype=np.int64)
        )
    if masks.ndim != 3 or frames.ndim != 1 or frames.shape[0] != masks.shape[0]:
        raise ValueError(f"invalid mask tube shape: masks={masks.shape} frames={frames.shape}")
    source_clip = manifest.get("source_clip") if isinstance(manifest.get("source_clip"), Mapping) else {}
    canonical = source_clip.get("canonical") if isinstance(source_clip.get("canonical"), Mapping) else {}
    valid_count = int(canonical.get("valid_frame_count") or masks.shape[0])
    valid_count = max(0, min(valid_count, masks.shape[0]))
    masks = masks[:valid_count]
    frames = frames[:valid_count]
    times, tolerance = _canonical_times(manifest, frames.tolist())
    return MaskTube(times=times, masks=masks, max_time_error=tolerance)
