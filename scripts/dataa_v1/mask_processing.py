"""Create M_raw, M_edit, M_gen and M_alpha from aligned SAM3 masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

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

