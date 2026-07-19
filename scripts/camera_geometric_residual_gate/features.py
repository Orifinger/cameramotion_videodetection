"""Compact appearance, raw-motion, and camera-geometry residual features."""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from scripts.camera_flow_probe.features import GLOBAL_FEATURE_DIM, global_restrav_features
from scripts.camera_flow_probe.geometry import dense_transform_flow, fit_global_camera_transform
from scripts.camera_flow_probe.models import DinoV2Extractor, TorchvisionRaft


PAIR_DISTRIBUTION_DIM = 6
SERIES_SUMMARY_DIM = 22
COMMON_GEOMETRY_DIM = 20
MOTION_BLOCK_DIM = SERIES_SUMMARY_DIM * 2 + COMMON_GEOMETRY_DIM
VARIANT_DIMS = {
    "appearance": GLOBAL_FEATURE_DIM,
    "appearance_raw_motion": GLOBAL_FEATURE_DIM + MOTION_BLOCK_DIM,
    "appearance_geometry_residual": GLOBAL_FEATURE_DIM + MOTION_BLOCK_DIM,
    "appearance_wrong_geometry": GLOBAL_FEATURE_DIM + MOTION_BLOCK_DIM,
}


def _finite(values: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def distribution_stats(values: np.ndarray | Sequence[float]) -> np.ndarray:
    valid = _finite(values)
    if valid.size == 0:
        return np.zeros(PAIR_DISTRIBUTION_DIM, dtype=np.float32)
    return np.asarray(
        [
            valid.mean(),
            valid.std(),
            np.quantile(valid, 0.10),
            np.quantile(valid, 0.50),
            np.quantile(valid, 0.90),
            valid.max(),
        ],
        dtype=np.float32,
    )


def summarize_pair_values(values: Sequence[np.ndarray | Sequence[float]]) -> np.ndarray:
    if not values:
        return np.zeros(SERIES_SUMMARY_DIM, dtype=np.float32)
    pair_stats = np.stack([distribution_stats(value) for value in values])
    aggregate = np.concatenate(
        [
            pair_stats.mean(axis=0),
            pair_stats.std(axis=0),
            pair_stats.max(axis=0),
        ]
    )
    medians = pair_stats[:, 3]
    second = np.abs(np.diff(medians, n=2)) if medians.size >= 3 else np.zeros(0, dtype=np.float32)
    if second.size:
        dynamics = np.asarray(
            [second.mean(), second.std(), np.quantile(second, 0.90), second.max()],
            dtype=np.float32,
        )
    else:
        dynamics = np.zeros(4, dtype=np.float32)
    output = np.concatenate([aggregate, dynamics]).astype(np.float32)
    if output.shape != (SERIES_SUMMARY_DIM,):
        raise AssertionError(output.shape)
    return output


def _series_four(values: Sequence[float]) -> np.ndarray:
    valid = _finite(values)
    if valid.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.asarray([valid.mean(), valid.std(), valid.min(), valid.max()], dtype=np.float32)


def _flow_correspondences(
    flow: np.ndarray,
    fb_error: np.ndarray | None,
    *,
    grid_step: int,
    max_fb_error: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = flow.shape[:2]
    margin = max(2, int(grid_step))
    ys = np.arange(margin, max(margin + 1, height - margin), grid_step)
    xs = np.arange(margin, max(margin + 1, width - margin), grid_step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    source = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    vectors = np.asarray(flow, dtype=np.float32)[yy, xx].reshape(-1, 2)
    target = source + vectors
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    valid &= (
        (target[:, 0] >= 0.0)
        & (target[:, 0] < width)
        & (target[:, 1] >= 0.0)
        & (target[:, 1] < height)
    )
    fb_values = np.full(source.shape[0], np.nan, dtype=np.float32)
    if fb_error is not None:
        fb_values = np.asarray(fb_error, dtype=np.float32)[yy, xx].reshape(-1)
        strict = valid & np.isfinite(fb_values) & (fb_values <= max_fb_error)
        if int(strict.sum()) >= 16:
            valid = strict
    return source[valid], target[valid], vectors[valid], fb_values[valid]


def _fit_fundamental(source: np.ndarray, target: np.ndarray, *, threshold: float) -> tuple[np.ndarray | None, dict[str, float]]:
    if source.shape[0] < 16:
        return None, {"valid": 0.0, "inlier_rate": 0.0, "median_error": float("nan")}
    method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
    try:
        matrix, mask = cv2.findFundamentalMat(
            source,
            target,
            method,
            threshold,
            0.999,
            10000,
        )
    except TypeError:
        matrix, mask = cv2.findFundamentalMat(source, target, method, threshold, 0.999)
    if matrix is None or np.asarray(matrix).shape != (3, 3) or not np.isfinite(matrix).all():
        return None, {"valid": 0.0, "inlier_rate": 0.0, "median_error": float("nan")}
    matrix = np.asarray(matrix, dtype=np.float64)
    errors = sampson_errors(source, target, matrix)
    inliers = np.asarray(mask).reshape(-1).astype(bool) if mask is not None and np.asarray(mask).size == source.shape[0] else errors <= threshold
    return matrix, {
        "valid": 1.0,
        "inlier_rate": float(inliers.mean()),
        "median_error": float(np.nanmedian(errors[inliers] if inliers.any() else errors)),
    }


def sampson_errors(source: np.ndarray, target: np.ndarray, fundamental: np.ndarray | None) -> np.ndarray:
    if fundamental is None or source.size == 0:
        return np.zeros(0, dtype=np.float32)
    ones = np.ones((source.shape[0], 1), dtype=np.float64)
    first = np.concatenate([np.asarray(source, dtype=np.float64), ones], axis=1)
    second = np.concatenate([np.asarray(target, dtype=np.float64), ones], axis=1)
    fx = first @ np.asarray(fundamental, dtype=np.float64).T
    ftx = second @ np.asarray(fundamental, dtype=np.float64)
    numerator = np.sum(second * fx, axis=1)
    denominator = fx[:, 0] ** 2 + fx[:, 1] ** 2 + ftx[:, 0] ** 2 + ftx[:, 1] ** 2
    values = np.sqrt(np.square(numerator) / np.maximum(denominator, 1e-12))
    return values.astype(np.float32)


def _sample_transform_residual(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    if source.size == 0:
        return np.zeros(0, dtype=np.float32)
    homogeneous = np.concatenate([source.astype(np.float64), np.ones((source.shape[0], 1))], axis=1)
    projected = homogeneous @ np.asarray(transform, dtype=np.float64).T
    denominator = projected[:, 2:3]
    denominator[np.abs(denominator) < 1e-9] = np.nan
    projected = projected[:, :2] / denominator
    return np.linalg.norm(projected - target, axis=1).astype(np.float32)


def _sample_dense_flow_magnitude(flow: np.ndarray, source: np.ndarray) -> np.ndarray:
    if source.size == 0:
        return np.zeros(0, dtype=np.float32)
    x = np.clip(np.rint(source[:, 0]).astype(int), 0, flow.shape[1] - 1)
    y = np.clip(np.rint(source[:, 1]).astype(int), 0, flow.shape[0] - 1)
    return np.linalg.norm(np.asarray(flow)[y, x], axis=1).astype(np.float32)


def _common_geometry_features(
    homography_stats: Sequence[dict[str, Any]],
    fundamental_stats: Sequence[dict[str, float]],
    global_magnitudes: Sequence[np.ndarray],
    fb_values: Sequence[np.ndarray],
    *,
    diagonal: float,
) -> np.ndarray:
    values = np.concatenate(
        [
            _series_four([float(item.get("inlier_rate", 0.0)) for item in homography_stats]),
            _series_four([float(item.get("median_reprojection_error", np.nan)) / diagonal for item in homography_stats]),
            _series_four([float(item.get("inlier_rate", 0.0)) for item in fundamental_stats]),
            _series_four([float(np.nanmedian(value)) / diagonal if _finite(value).size else np.nan for value in global_magnitudes]),
            _series_four([float(np.nanmedian(value)) / diagonal if _finite(value).size else np.nan for value in fb_values]),
        ]
    ).astype(np.float32)
    if values.shape != (COMMON_GEOMETRY_DIM,):
        raise AssertionError(values.shape)
    return values


def build_motion_blocks(
    forward: np.ndarray,
    backward: np.ndarray | None,
    *,
    grid_step: int = 8,
    max_fb_error: float = 2.0,
    ransac_threshold: float = 2.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from scripts.camera_flow_probe.geometry import forward_backward_error

    transforms: list[np.ndarray] = []
    fundamentals: list[np.ndarray | None] = []
    homography_stats: list[dict[str, Any]] = []
    fundamental_stats: list[dict[str, float]] = []
    correspondences: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    global_flows: list[np.ndarray] = []
    diagonal = float(np.hypot(forward.shape[1], forward.shape[2]))
    for index, flow in enumerate(forward):
        fb = None if backward is None else forward_backward_error(flow, backward[index])
        source, target, vectors, sampled_fb = _flow_correspondences(
            flow,
            fb,
            grid_step=grid_step,
            max_fb_error=max_fb_error,
        )
        transform, h_stats = fit_global_camera_transform(
            flow,
            fb_error=fb,
            grid_step=grid_step,
            max_fb_error=max_fb_error,
            ransac_threshold=ransac_threshold,
            model="homography",
        )
        fundamental, f_stats = _fit_fundamental(source, target, threshold=ransac_threshold)
        transforms.append(transform)
        fundamentals.append(fundamental)
        homography_stats.append(h_stats)
        fundamental_stats.append(f_stats)
        correspondences.append((source, target, vectors, sampled_fb))
        global_flows.append(dense_transform_flow(transform, flow.shape[0], flow.shape[1]))

    count = len(transforms)
    wrong_transforms = [transforms[(index + 1) % count] for index in range(count)]
    wrong_fundamentals = [fundamentals[(index + 1) % count] for index in range(count)]
    raw_values: list[np.ndarray] = []
    global_values: list[np.ndarray] = []
    correct_h_values: list[np.ndarray] = []
    correct_f_values: list[np.ndarray] = []
    wrong_h_values: list[np.ndarray] = []
    wrong_f_values: list[np.ndarray] = []
    fb_values: list[np.ndarray] = []
    for index, (source, target, vectors, sampled_fb) in enumerate(correspondences):
        del vectors
        raw_values.append(_sample_dense_flow_magnitude(forward[index], source) / diagonal)
        global_values.append(_sample_dense_flow_magnitude(global_flows[index], source) / diagonal)
        correct_h_values.append(_sample_transform_residual(source, target, transforms[index]) / diagonal)
        correct_f_values.append(sampson_errors(source, target, fundamentals[index]) / diagonal)
        wrong_h_values.append(_sample_transform_residual(source, target, wrong_transforms[index]) / diagonal)
        wrong_f_values.append(sampson_errors(source, target, wrong_fundamentals[index]) / diagonal)
        fb_values.append(sampled_fb)

    common = _common_geometry_features(
        homography_stats,
        fundamental_stats,
        global_values,
        fb_values,
        diagonal=diagonal,
    )
    blocks = {
        "raw_motion": np.concatenate(
            [summarize_pair_values(raw_values), summarize_pair_values(global_values), common]
        ).astype(np.float32),
        "geometry_residual": np.concatenate(
            [summarize_pair_values(correct_h_values), summarize_pair_values(correct_f_values), common]
        ).astype(np.float32),
        "wrong_geometry": np.concatenate(
            [summarize_pair_values(wrong_h_values), summarize_pair_values(wrong_f_values), common]
        ).astype(np.float32),
    }
    for key, value in blocks.items():
        if value.shape != (MOTION_BLOCK_DIM,):
            raise AssertionError(f"{key}: {value.shape}")
    quality = {
        "num_frame_pairs": count,
        "homography_inlier_rate_mean": float(np.mean([item.get("inlier_rate", 0.0) for item in homography_stats])),
        "fundamental_valid_rate": float(np.mean([item.get("valid", 0.0) for item in fundamental_stats])),
        "fundamental_inlier_rate_mean": float(np.mean([item.get("inlier_rate", 0.0) for item in fundamental_stats])),
    }
    return blocks, quality


def extract_features(
    frames: np.ndarray,
    *,
    raft: TorchvisionRaft,
    dino: DinoV2Extractor,
    grid_step: int = 8,
    max_fb_error: float = 2.0,
    ransac_threshold: float = 2.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cls_features, _patch_features, _geometry = dino.extract(frames)
    appearance = global_restrav_features(cls_features)
    forward, backward, _flow_geometry = raft.infer_pairs(frames, backward=True)
    blocks, quality = build_motion_blocks(
        forward,
        backward,
        grid_step=grid_step,
        max_fb_error=max_fb_error,
        ransac_threshold=ransac_threshold,
    )
    output = {
        "appearance": appearance,
        "appearance_raw_motion": np.concatenate([appearance, blocks["raw_motion"]]).astype(np.float32),
        "appearance_geometry_residual": np.concatenate([appearance, blocks["geometry_residual"]]).astype(np.float32),
        "appearance_wrong_geometry": np.concatenate([appearance, blocks["wrong_geometry"]]).astype(np.float32),
    }
    for key, expected in VARIANT_DIMS.items():
        if output[key].shape != (expected,):
            raise AssertionError(f"{key}: {output[key].shape}, expected {(expected,)}")
    return output, quality
