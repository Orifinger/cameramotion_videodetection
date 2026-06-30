"""Create M_raw, M_edit, M_gen and M_alpha from aligned SAM3 masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np
from scipy import ndimage

from .common import DataAError


@dataclass(frozen=True)
class MaskProcessingConfig:
    min_component_area_px: int = 8
    closing_radius_px: int = 1
    dilation_radius_px: int = 2
    alpha_feather_sigma_px: float = 1.0


def disk_structure(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    labels, count = ndimage.label(mask > 0)
    if count == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_area)
    keep[0] = False
    return keep[labels].astype(np.uint8)


def process_masks(aligned_raw: np.ndarray, config: MaskProcessingConfig | None = None) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if aligned_raw.ndim != 3:
        raise DataAError(f"aligned_raw must be [N,H,W], got {aligned_raw.shape}")
    config = config or MaskProcessingConfig()
    m_raw = (aligned_raw > 0).astype(np.uint8)
    edit_frames = []
    gen_frames = []
    alpha_frames = []
    close_kernel = disk_structure(config.closing_radius_px)
    dilate_kernel = disk_structure(config.dilation_radius_px)
    for frame in m_raw:
        cleaned = remove_small_components(frame, config.min_component_area_px)
        closed = ndimage.binary_closing(cleaned > 0, structure=close_kernel).astype(np.uint8)
        edit = closed
        gen = ndimage.binary_dilation(edit > 0, structure=dilate_kernel).astype(np.uint8)
        if config.alpha_feather_sigma_px > 0:
            alpha = ndimage.gaussian_filter(gen.astype(np.float32), sigma=config.alpha_feather_sigma_px)
            alpha = np.clip(alpha, 0.0, 1.0)
        else:
            alpha = gen.astype(np.float32)
        edit_frames.append(edit)
        gen_frames.append(gen)
        alpha_frames.append(alpha.astype(np.float32))
    params = {
        "min_component_area_px": config.min_component_area_px,
        "closing_radius_px": config.closing_radius_px,
        "dilation_radius_px": config.dilation_radius_px,
        "alpha_feather_sigma_px": config.alpha_feather_sigma_px,
        "implementation": "scipy.ndimage binary morphology",
    }
    return {
        "M_raw": m_raw,
        "M_edit": np.stack(edit_frames).astype(np.uint8),
        "M_gen": np.stack(gen_frames).astype(np.uint8),
        "M_alpha": np.stack(alpha_frames).astype(np.float32),
    }, params


def _binary_alpha(masks: np.ndarray, sigma: float) -> np.ndarray:
    frames = []
    for frame in (masks > 0).astype(np.uint8):
        if sigma > 0:
            alpha = ndimage.gaussian_filter(frame.astype(np.float32), sigma=sigma)
            frames.append(np.clip(alpha, 0.0, 1.0).astype(np.float32))
        else:
            frames.append(frame.astype(np.float32))
    return np.stack(frames).astype(np.float32)


def _dilate_masks(masks: np.ndarray, radius: int) -> np.ndarray:
    kernel = disk_structure(radius)
    return np.stack([ndimage.binary_dilation(frame > 0, structure=kernel).astype(np.uint8) for frame in masks])


def _erode_masks(masks: np.ndarray, radius: int) -> tuple[np.ndarray, int]:
    kernel = disk_structure(radius)
    frames = []
    recovered = 0
    for frame in (masks > 0).astype(np.uint8):
        eroded = ndimage.binary_erosion(frame > 0, structure=kernel).astype(np.uint8)
        if frame.any() and not eroded.any():
            eroded = frame
            recovered += 1
        frames.append(eroded)
    return np.stack(frames).astype(np.uint8), recovered


def _close_masks(masks: np.ndarray, radius: int) -> np.ndarray:
    kernel = disk_structure(radius)
    return np.stack([ndimage.binary_closing(frame > 0, structure=kernel).astype(np.uint8) for frame in masks])


def _bbox_xywh(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1 - x0, y1 - y0


def _expanded_bbox_mask(masks: np.ndarray, expand_ratio: float) -> np.ndarray:
    if expand_ratio <= 0:
        raise DataAError(f"bbox_expand_ratio must be positive, got {expand_ratio}")
    out = np.zeros_like(masks, dtype=np.uint8)
    height, width = int(masks.shape[1]), int(masks.shape[2])
    for index, frame in enumerate(masks):
        bbox = _bbox_xywh(frame)
        if bbox is None:
            continue
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        new_w = max(1.0, w * float(expand_ratio))
        new_h = max(1.0, h * float(expand_ratio))
        x0 = max(0, int(round(cx - new_w / 2.0)))
        y0 = max(0, int(round(cy - new_h / 2.0)))
        x1 = min(width, int(round(cx + new_w / 2.0)))
        y1 = min(height, int(round(cy + new_h / 2.0)))
        if x1 > x0 and y1 > y0:
            out[index, y0:y1, x0:x1] = 1
    return out


def _area_stats(masks: np.ndarray) -> Dict[str, float]:
    areas = (masks > 0).reshape(masks.shape[0], -1).mean(axis=1)
    if areas.size == 0:
        return {"mean": 0.0, "median": 0.0, "p20": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(areas)),
        "median": float(np.median(areas)),
        "p20": float(np.quantile(areas, 0.20)),
        "min": float(np.min(areas)),
        "max": float(np.max(areas)),
    }


def _bbox_tube(masks: np.ndarray) -> list[Dict[str, int | None]]:
    tube: list[Dict[str, int | None]] = []
    for index, frame in enumerate(masks):
        bbox = _bbox_xywh(frame)
        if bbox is None:
            tube.append({"frame_index": int(index), "x": None, "y": None, "w": None, "h": None})
        else:
            x, y, w, h = bbox
            tube.append({"frame_index": int(index), "x": int(x), "y": int(y), "w": int(w), "h": int(h)})
    return tube


def apply_effective_mask_policy(
    masks: Dict[str, np.ndarray],
    policy: Mapping[str, Any],
    config: MaskProcessingConfig | None = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Apply a plan-frozen effective mask policy to M_gen/M_alpha only."""
    if "M_raw" not in masks or "M_edit" not in masks or "M_gen" not in masks:
        raise DataAError("mask policy requires M_raw, M_edit and M_gen")
    if not policy:
        raise DataAError("blocked_missing_frozen_mask_policy: sampling_meta.mask_policy is required")
    config = config or MaskProcessingConfig()
    variant = str(policy.get("variant_type") or "")
    if variant not in {"sam3_shape", "dilated", "expanded_bbox", "closing", "erode_then_dilate"}:
        raise DataAError(f"blocked_invalid_mask_policy: unsupported variant_type={variant}")
    if variant == "expanded_bbox" and bool(policy.get("person_bbox_disabled")):
        raise DataAError("blocked_invalid_mask_policy: expanded_bbox is disabled for person route")

    base_gen = (masks["M_gen"] > 0).astype(np.uint8)
    empty_erosion_recovered_frames = 0
    if variant == "sam3_shape":
        effective = base_gen
    elif variant == "dilated":
        radius = int(policy.get("dilation_radius_px") or config.dilation_radius_px)
        effective = _dilate_masks((masks["M_edit"] > 0).astype(np.uint8), radius)
    elif variant == "closing":
        radius = int(policy.get("closing_radius_px") or config.closing_radius_px)
        effective = _close_masks((masks["M_edit"] > 0).astype(np.uint8), radius)
    elif variant == "erode_then_dilate":
        erosion_radius = int(policy.get("erosion_radius_px") or 1)
        dilation_radius = int(policy.get("dilation_radius_px") or config.dilation_radius_px)
        eroded, empty_erosion_recovered_frames = _erode_masks((masks["M_edit"] > 0).astype(np.uint8), erosion_radius)
        effective = _dilate_masks(eroded, dilation_radius)
    else:
        expand_ratio = float(policy.get("bbox_expand_ratio") or 1.15)
        effective = _expanded_bbox_mask(base_gen, expand_ratio)

    updated = dict(masks)
    updated["M_gen"] = effective.astype(np.uint8)
    updated["M_alpha"] = _binary_alpha(updated["M_gen"], float(config.alpha_feather_sigma_px))
    params = {
        "mask_policy": dict(policy),
        "variant_type": variant,
        "original_mask_area_stats": _area_stats((masks["M_raw"] > 0).astype(np.uint8)),
        "base_gen_area_stats": _area_stats(base_gen),
        "effective_mask_area_stats": _area_stats(updated["M_gen"]),
        "original_bbox_tube": _bbox_tube((masks["M_raw"] > 0).astype(np.uint8)),
        "base_gen_bbox_tube": _bbox_tube(base_gen),
        "effective_bbox_tube": _bbox_tube(updated["M_gen"]),
        "alpha_feather_sigma_px": float(config.alpha_feather_sigma_px),
        "empty_erosion_recovered_frames": int(empty_erosion_recovered_frames),
        "implementation": "plan-frozen effective mask policy",
    }
    return updated, params
