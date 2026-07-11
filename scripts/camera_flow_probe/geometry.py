"""Geometry helpers for dense-flow camera estimation and feature alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np


@dataclass(frozen=True)
class CanvasGeometry:
    source_height: int
    source_width: int
    canvas_height: int
    canvas_width: int
    scale: float
    pad_top: int
    pad_left: int

    @property
    def source_to_canvas(self) -> np.ndarray:
        return np.array(
            [
                [self.scale, 0.0, float(self.pad_left)],
                [0.0, self.scale, float(self.pad_top)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def canvas_geometry(height: int, width: int, *, long_side: int, multiple: int) -> CanvasGeometry:
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid source shape: {(height, width)}")
    if long_side <= 0 or multiple <= 0:
        raise ValueError("long_side and multiple must be positive")
    scale = float(long_side) / float(max(height, width))
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    canvas_height = int(np.ceil(resized_height / multiple) * multiple)
    canvas_width = int(np.ceil(resized_width / multiple) * multiple)
    pad_top = (canvas_height - resized_height) // 2
    pad_left = (canvas_width - resized_width) // 2
    return CanvasGeometry(
        source_height=height,
        source_width=width,
        canvas_height=canvas_height,
        canvas_width=canvas_width,
        scale=scale,
        pad_top=pad_top,
        pad_left=pad_left,
    )


def resize_and_pad(image: np.ndarray, geometry: CanvasGeometry, *, value: int = 0) -> np.ndarray:
    resized_width = int(round(geometry.source_width * geometry.scale))
    resized_height = int(round(geometry.source_height * geometry.scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    bottom = geometry.canvas_height - geometry.pad_top - resized_height
    right = geometry.canvas_width - geometry.pad_left - resized_width
    return cv2.copyMakeBorder(
        resized,
        geometry.pad_top,
        bottom,
        geometry.pad_left,
        right,
        cv2.BORDER_CONSTANT,
        value=value,
    )


def project_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    projected = homogeneous @ np.asarray(transform, dtype=np.float64).T
    denominator = projected[:, 2:3]
    denominator[np.abs(denominator) < 1e-9] = np.nan
    return projected[:, :2] / denominator


def transform_flow_to_source(transform: np.ndarray, geometry: CanvasGeometry) -> np.ndarray:
    canvas_to_source = np.linalg.inv(geometry.source_to_canvas)
    return canvas_to_source @ np.asarray(transform, dtype=np.float64) @ geometry.source_to_canvas


def dense_transform_flow(transform: np.ndarray, height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    points = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
    projected = project_points(points, transform)
    return (projected - points).reshape(height, width, 2).astype(np.float32)


def forward_backward_error(forward: np.ndarray, backward: np.ndarray) -> np.ndarray:
    if forward.shape != backward.shape or forward.ndim != 3 or forward.shape[2] != 2:
        raise ValueError(f"flow shapes must match [H,W,2], got {forward.shape} and {backward.shape}")
    height, width = forward.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    map_x = xx + forward[..., 0].astype(np.float32)
    map_y = yy + forward[..., 1].astype(np.float32)
    sampled_x = cv2.remap(
        backward[..., 0].astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan
    )
    sampled_y = cv2.remap(
        backward[..., 1].astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan
    )
    return np.sqrt((forward[..., 0] + sampled_x) ** 2 + (forward[..., 1] + sampled_y) ** 2)


def _translation_fallback(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = np.nanmedian(target - source, axis=0)
    return np.array(
        [[1.0, 0.0, float(delta[0])], [0.0, 1.0, float(delta[1])], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def fit_global_camera_transform(
    flow: np.ndarray,
    *,
    fb_error: np.ndarray | None = None,
    grid_step: int = 8,
    max_fb_error: float = 2.0,
    ransac_threshold: float = 2.0,
    model: Literal["homography", "affine"] = "homography",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit dominant camera motion without using any edit mask."""

    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"flow must have shape [H,W,2], got {flow.shape}")
    height, width = flow.shape[:2]
    margin = max(2, grid_step)
    ys = np.arange(margin, max(margin + 1, height - margin), grid_step)
    xs = np.arange(margin, max(margin + 1, width - margin), grid_step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    source_all = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    vectors = flow[yy, xx].reshape(-1, 2)
    finite_flow = np.isfinite(vectors).all(axis=1)
    valid = finite_flow.copy()
    fb_filter_relaxed = False
    if fb_error is not None:
        sampled_error = np.asarray(fb_error)[yy, xx].reshape(-1)
        valid &= np.isfinite(sampled_error) & (sampled_error <= max_fb_error)
        if int(valid.sum()) < 12:
            valid = finite_flow
            fb_filter_relaxed = True
    source = source_all[valid]
    target = source + vectors[valid]
    finite = np.isfinite(target).all(axis=1)
    source, target = source[finite], target[finite]
    if source.shape[0] < 12:
        raise ValueError(f"not enough reliable flow correspondences: {source.shape[0]}")

    selected_model = model
    inlier_mask: np.ndarray | None = None
    transform: np.ndarray | None = None
    if model == "homography":
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        transform, inlier_mask = cv2.findHomography(
            source,
            target,
            method=method,
            ransacReprojThreshold=ransac_threshold,
            maxIters=5000,
            confidence=0.999,
        )
    if transform is None:
        affine, inlier_mask = cv2.estimateAffine2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=5000,
            confidence=0.999,
            refineIters=10,
        )
        if affine is not None:
            transform = np.vstack([affine, [0.0, 0.0, 1.0]])
            selected_model = "affine" if model == "homography" else model
    if transform is None or not np.isfinite(transform).all():
        transform = _translation_fallback(source, target)
        selected_model = "translation"
        inlier_mask = np.ones((source.shape[0], 1), dtype=np.uint8)

    predicted = project_points(source, transform)
    errors = np.linalg.norm(predicted - target, axis=1)
    inliers = (
        np.asarray(inlier_mask).reshape(-1).astype(bool)
        if inlier_mask is not None and np.asarray(inlier_mask).size == source.shape[0]
        else errors <= ransac_threshold
    )
    stats = {
        "model": selected_model,
        "sample_count": int(source.shape[0]),
        "inlier_count": int(inliers.sum()),
        "inlier_rate": float(inliers.mean()),
        "median_reprojection_error": float(np.nanmedian(errors[inliers] if inliers.any() else errors)),
        "p90_reprojection_error": float(np.nanpercentile(errors[inliers] if inliers.any() else errors, 90)),
        "forward_backward_filter_relaxed": fb_filter_relaxed,
    }
    return np.asarray(transform, dtype=np.float64), stats


def compose_transforms(pair_transforms: list[np.ndarray]) -> list[np.ndarray]:
    """Return transforms from frame zero to every frame in a window."""

    cumulative = [np.eye(3, dtype=np.float64)]
    current = cumulative[0]
    for transform in pair_transforms:
        current = np.asarray(transform, dtype=np.float64) @ current
        current = current / current[2, 2]
        cumulative.append(current.copy())
    return cumulative
