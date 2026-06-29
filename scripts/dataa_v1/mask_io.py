"""Lossless SAM3 mask tube loading and canonical frame alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from .common import DataAError
from .path_resolver import ResolvedPath


@dataclass
class MaskTube:
    frame_indices: np.ndarray
    masks: np.ndarray
    path: str

    @property
    def height(self) -> int:
        return int(self.masks.shape[1])

    @property
    def width(self) -> int:
        return int(self.masks.shape[2])


def inspect_mask_npz(resolved: ResolvedPath) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "checked": False,
        "valid": False,
        "reason": None,
        "frame_indices_count": None,
        "mask_count": None,
        "height": None,
        "width": None,
        "frame_indices_dtype": None,
        "dtype": None,
        "frame_indices_strictly_increasing": None,
        "mask_nonempty_ratio": None,
    }
    if resolved.state not in {"readable_persistent", "readable_volatile"} or not resolved.resolved_path:
        result["reason"] = f"not locally readable ({resolved.state})"
        return result
    try:
        tube = load_mask_tube(Path(resolved.resolved_path))
    except Exception as exc:  # noqa: BLE001 - report exact validation reason
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result

    nonempty = np.any(tube.masks > 0, axis=(1, 2))
    result.update(
        checked=True,
        valid=bool(nonempty.all()),
        reason="ok" if bool(nonempty.all()) else "at least one visible-frame mask is empty",
        frame_indices_count=int(tube.frame_indices.shape[0]),
        mask_count=int(tube.masks.shape[0]),
        height=tube.height,
        width=tube.width,
        frame_indices_dtype=str(tube.frame_indices.dtype),
        dtype=str(tube.masks.dtype),
        frame_indices_strictly_increasing=True,
        mask_nonempty_ratio=float(nonempty.mean()),
    )
    return result


def load_mask_tube(path: Path) -> MaskTube:
    if not path.is_file():
        raise DataAError(f"mask npz does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "frame_indices" not in archive or "masks" not in archive:
            raise DataAError("npz must contain frame_indices and masks")
        frame_indices = archive["frame_indices"]
        masks = archive["masks"]

    if frame_indices.ndim != 1:
        raise DataAError(f"frame_indices must be 1D, got {frame_indices.shape}")
    if frame_indices.dtype != np.int32:
        raise DataAError(f"frame_indices must be int32, got {frame_indices.dtype}")
    if masks.ndim != 3:
        raise DataAError(f"masks must have [N,H,W], got {masks.shape}")
    if masks.dtype != np.uint8:
        raise DataAError(f"masks must be uint8, got {masks.dtype}")
    if frame_indices.shape[0] != masks.shape[0]:
        raise DataAError("frame_indices and masks have different N")
    if masks.shape[0] == 0:
        raise DataAError("empty mask tube")
    if masks.shape[1] <= 0 or masks.shape[2] <= 0:
        raise DataAError(f"invalid mask raster shape: {masks.shape}")
    if len(frame_indices) > 1 and not bool(np.all(np.diff(frame_indices.astype(np.int64)) > 0)):
        raise DataAError("frame_indices are not strictly increasing")
    return MaskTube(frame_indices=frame_indices, masks=(masks > 0).astype(np.uint8), path=str(path))


def frame_lookup(tube: MaskTube) -> Dict[int, np.ndarray]:
    return {int(idx): tube.masks[pos] for pos, idx in enumerate(tube.frame_indices)}


def nearest_source_frame(source_frame: float, visible_frames: np.ndarray) -> int:
    pos = int(np.argmin(np.abs(visible_frames.astype(np.float64) - float(source_frame))))
    return int(visible_frames[pos])


def align_masks_to_canonical(
    tube: MaskTube,
    canonical_to_source_frames: Iterable[float],
    *,
    zero_missing: bool = True,
) -> tuple[np.ndarray, Dict[str, Any]]:
    lookup = frame_lookup(tube)
    aligned = []
    mapping = []
    zero_filled = []
    for canonical_index, source_frame in enumerate(canonical_to_source_frames):
        rounded = int(round(float(source_frame)))
        if rounded in lookup:
            aligned.append(lookup[rounded])
            mask_source_frame = rounded
            status = "visible"
        elif zero_missing:
            aligned.append(np.zeros((tube.height, tube.width), dtype=np.uint8))
            mask_source_frame = None
            status = "zero_filled_invisible_gap"
            zero_filled.append(int(canonical_index))
        else:
            nearest = nearest_source_frame(float(source_frame), tube.frame_indices)
            if nearest not in lookup:
                raise DataAError(f"nearest visible source frame missing from lookup: {nearest}")
            aligned.append(lookup[nearest])
            mask_source_frame = nearest
            status = "nearest_visible_fallback"
        mapping.append(
            {
                "canonical_frame": int(canonical_index),
                "source_frame_float": float(source_frame),
                "source_frame_index": int(rounded),
                "mask_source_frame_index": None if mask_source_frame is None else int(mask_source_frame),
                "mask_alignment_status": status,
            }
        )
    return np.stack(aligned, axis=0).astype(np.uint8), {
        "frame_mapping": mapping,
        "zero_missing": bool(zero_missing),
        "zero_filled_gap_frames": zero_filled,
        "zero_filled_gap_frame_count": int(len(zero_filled)),
    }


def save_mask_npz(path: Path, masks: np.ndarray, *, frame_indices: np.ndarray | None = None, kind: str = "mask") -> None:
    if masks.ndim != 3:
        raise DataAError(f"{kind} masks must be [N,H,W], got {masks.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame_indices is None:
        frame_indices = np.arange(masks.shape[0], dtype=np.int32)
    np.savez_compressed(path, frame_indices=frame_indices.astype(np.int32), masks=masks)
