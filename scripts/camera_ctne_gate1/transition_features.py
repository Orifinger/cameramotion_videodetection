"""Per-transition camera context and temporal forensic evidence."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from scripts.camera_flow_probe.geometry import fit_global_camera_transform, forward_backward_error


CAMERA_FEATURE_NAMES = (
    "translation_x_over_width",
    "translation_y_over_height",
    "rotation_radians",
    "log_scale_x",
    "log_scale_y",
    "shear_cosine",
    "perspective_x_times_width",
    "perspective_y_times_height",
    "log_abs_affine_determinant",
    "affine_determinant_sign",
    "inlier_rate",
    "median_reprojection_over_diagonal",
    "p90_reprojection_over_diagonal",
    "median_forward_backward_over_diagonal",
    "p90_forward_backward_over_diagonal",
    "median_flow_x_over_width",
    "median_flow_y_over_height",
    "mean_flow_magnitude_over_diagonal",
    "median_flow_magnitude_over_diagonal",
    "p90_flow_magnitude_over_diagonal",
    "model_is_homography",
    "model_is_affine",
    "model_is_translation",
    "forward_backward_filter_relaxed",
)

PATCH_STAT_NAMES = tuple(
    f"patch_{kind}_{stat}"
    for kind in ("cosine_distance", "l2")
    for stat in ("mean", "std", "q10", "q50", "q90")
)
FLOW_STAT_NAMES = (
    "flow_x_mean_over_width",
    "flow_x_std_over_width",
    "flow_x_q10_over_width",
    "flow_x_q50_over_width",
    "flow_x_q90_over_width",
    "flow_y_mean_over_height",
    "flow_y_std_over_height",
    "flow_y_q10_over_height",
    "flow_y_q50_over_height",
    "flow_y_q90_over_height",
    "flow_magnitude_mean_over_diagonal",
    "flow_magnitude_std_over_diagonal",
    "flow_magnitude_q10_over_diagonal",
    "flow_magnitude_q50_over_diagonal",
    "flow_magnitude_q90_over_diagonal",
    "divergence_mean",
    "divergence_std",
    "divergence_q10",
    "divergence_q50",
    "divergence_q90",
    "curl_mean",
    "curl_std",
    "curl_q10",
    "curl_q50",
    "curl_q90",
    "forward_backward_mean_over_diagonal",
    "forward_backward_std_over_diagonal",
    "forward_backward_q50_over_diagonal",
    "forward_backward_q90_over_diagonal",
) + tuple(f"flow_grid_{row}_{column}_mean_over_diagonal" for row in range(4) for column in range(4))


def evidence_feature_names(dino_dim: int) -> tuple[str, ...]:
    return (
        tuple(f"dino_first_delta_{index}" for index in range(dino_dim))
        + tuple(f"dino_second_delta_{index}" for index in range(dino_dim))
        + (
            "dino_adjacent_cosine_distance",
            "dino_first_delta_l2",
            "dino_second_delta_l2",
        )
        + PATCH_STAT_NAMES
        + FLOW_STAT_NAMES
    )


def _finite(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).reshape(-1)
    return output[np.isfinite(output)]


def _stats5(values: np.ndarray) -> np.ndarray:
    finite = _finite(values)
    if finite.size == 0:
        return np.zeros(5, dtype=np.float32)
    return np.asarray(
        [
            finite.mean(),
            finite.std(),
            np.quantile(finite, 0.10),
            np.quantile(finite, 0.50),
            np.quantile(finite, 0.90),
        ],
        dtype=np.float32,
    )


def _translation_fallback(flow: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    vectors = np.asarray(flow, dtype=np.float64).reshape(-1, 2)
    vectors = vectors[np.isfinite(vectors).all(axis=1)]
    delta = np.median(vectors, axis=0) if vectors.size else np.zeros(2, dtype=np.float64)
    transform = np.array(
        [[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return transform, {
        "model": "translation",
        "sample_count": int(vectors.shape[0]),
        "inlier_count": int(vectors.shape[0]),
        "inlier_rate": 1.0 if vectors.shape[0] else 0.0,
        "median_reprojection_error": 0.0,
        "p90_reprojection_error": 0.0,
        "forward_backward_filter_relaxed": True,
        "fit_exception_fallback": True,
    }


def safe_camera_fit(
    flow: np.ndarray,
    backward: np.ndarray | None,
    *,
    grid_step: int,
    max_fb_error: float,
    global_model: str,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    fb = forward_backward_error(flow, backward) if backward is not None else None
    try:
        transform, stats = fit_global_camera_transform(
            flow,
            fb_error=fb,
            grid_step=grid_step,
            max_fb_error=max_fb_error,
            model="affine" if global_model == "affine" else "homography",
        )
        stats["fit_exception_fallback"] = False
    except (cv2.error, ValueError, FloatingPointError, np.linalg.LinAlgError):
        transform, stats = _translation_fallback(flow)
    return transform, stats, fb


def _decompose_camera(
    transform: np.ndarray,
    stats: dict[str, Any],
    flow: np.ndarray,
    fb_error: np.ndarray | None,
) -> np.ndarray:
    height, width = flow.shape[:2]
    diagonal = max(float(np.hypot(height, width)), 1.0)
    matrix = np.asarray(transform, dtype=np.float64)
    if abs(matrix[2, 2]) > 1e-9:
        matrix = matrix / matrix[2, 2]
    affine = matrix[:2, :2]
    scale_x = max(float(np.linalg.norm(affine[:, 0])), 1e-8)
    scale_y = max(float(np.linalg.norm(affine[:, 1])), 1e-8)
    shear = float(np.dot(affine[:, 0], affine[:, 1]) / (scale_x * scale_y))
    rotation = float(np.arctan2(affine[1, 0] - affine[0, 1], affine[0, 0] + affine[1, 1]))
    determinant = float(np.linalg.det(affine))
    vectors = np.asarray(flow, dtype=np.float64)
    magnitude = np.linalg.norm(vectors, axis=2)
    fb = _finite(fb_error) / diagonal if fb_error is not None else np.empty(0, dtype=np.float64)
    model = str(stats.get("model", "translation"))
    values = np.asarray(
        [
            matrix[0, 2] / max(width, 1),
            matrix[1, 2] / max(height, 1),
            rotation,
            np.log(scale_x),
            np.log(scale_y),
            np.clip(shear, -1.0, 1.0),
            matrix[2, 0] * width,
            matrix[2, 1] * height,
            np.log(max(abs(determinant), 1e-8)),
            np.sign(determinant),
            float(stats.get("inlier_rate", 0.0)),
            float(stats.get("median_reprojection_error", 0.0)) / diagonal,
            float(stats.get("p90_reprojection_error", 0.0)) / diagonal,
            float(np.median(fb)) if fb.size else 0.0,
            float(np.quantile(fb, 0.90)) if fb.size else 0.0,
            float(np.nanmedian(vectors[..., 0])) / max(width, 1),
            float(np.nanmedian(vectors[..., 1])) / max(height, 1),
            float(np.nanmean(magnitude)) / diagonal,
            float(np.nanmedian(magnitude)) / diagonal,
            float(np.nanquantile(magnitude, 0.90)) / diagonal,
            float(model == "homography"),
            float(model == "affine"),
            float(model == "translation"),
            float(bool(stats.get("forward_backward_filter_relaxed", False))),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(values, nan=0.0, posinf=1e4, neginf=-1e4)


def _patch_pair_stats(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_vectors = np.moveaxis(np.asarray(first, dtype=np.float32), 0, -1).reshape(-1, first.shape[0])
    second_vectors = np.moveaxis(np.asarray(second, dtype=np.float32), 0, -1).reshape(-1, second.shape[0])
    first_norm = np.linalg.norm(first_vectors, axis=1)
    second_norm = np.linalg.norm(second_vectors, axis=1)
    cosine = np.sum(first_vectors * second_vectors, axis=1) / np.maximum(first_norm * second_norm, 1e-8)
    cosine_distance = 1.0 - np.clip(cosine, -1.0, 1.0)
    l2 = np.linalg.norm(second_vectors - first_vectors, axis=1) / np.sqrt(max(first.shape[0], 1))
    return np.concatenate([_stats5(cosine_distance), _stats5(l2)]).astype(np.float32)


def _grid_means(values: np.ndarray, rows: int = 4, columns: int = 4) -> np.ndarray:
    height, width = values.shape
    output: list[float] = []
    for y_indices in np.array_split(np.arange(height), rows):
        for x_indices in np.array_split(np.arange(width), columns):
            if y_indices.size == 0 or x_indices.size == 0:
                output.append(0.0)
            else:
                output.append(float(np.nanmean(values[np.ix_(y_indices, x_indices)])))
    return np.nan_to_num(np.asarray(output, dtype=np.float32))


def _flow_stats(flow: np.ndarray, fb_error: np.ndarray | None) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)
    height, width = flow.shape[:2]
    diagonal = max(float(np.hypot(height, width)), 1.0)
    x = flow[..., 0] / max(width, 1)
    y = flow[..., 1] / max(height, 1)
    magnitude = np.linalg.norm(flow, axis=2) / diagonal
    dx_dx = np.gradient(flow[..., 0], axis=1) / max(width, 1)
    dy_dy = np.gradient(flow[..., 1], axis=0) / max(height, 1)
    dy_dx = np.gradient(flow[..., 1], axis=1) / max(width, 1)
    dx_dy = np.gradient(flow[..., 0], axis=0) / max(height, 1)
    divergence = dx_dx + dy_dy
    curl = dy_dx - dx_dy
    if fb_error is None:
        fb = np.zeros(4, dtype=np.float32)
    else:
        values = _finite(fb_error) / diagonal
        fb = (
            np.asarray([values.mean(), values.std(), np.quantile(values, 0.50), np.quantile(values, 0.90)], dtype=np.float32)
            if values.size
            else np.zeros(4, dtype=np.float32)
        )
    return np.concatenate(
        [
            _stats5(x),
            _stats5(y),
            _stats5(magnitude),
            _stats5(divergence),
            _stats5(curl),
            fb,
            _grid_means(magnitude),
        ]
    ).astype(np.float32)


def build_transition_features(
    cls_features: np.ndarray,
    patch_features: np.ndarray,
    forward_flow: np.ndarray,
    backward_flow: np.ndarray | None,
    *,
    previous_delta: np.ndarray | None = None,
    grid_step: int = 8,
    max_fb_error: float = 2.0,
    global_model: str = "homography",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    cls_features = np.asarray(cls_features, dtype=np.float32)
    patch_features = np.asarray(patch_features, dtype=np.float32)
    forward_flow = np.asarray(forward_flow, dtype=np.float32)
    transitions = cls_features.shape[0] - 1
    if transitions < 1:
        raise ValueError("at least two frames are required")
    if patch_features.shape[0] != cls_features.shape[0] or forward_flow.shape[0] != transitions:
        raise ValueError(
            f"transition input mismatch: cls={cls_features.shape} patches={patch_features.shape} flow={forward_flow.shape}"
        )
    if backward_flow is not None and np.asarray(backward_flow).shape != forward_flow.shape:
        raise ValueError("backward flow must match forward flow")
    norms = np.linalg.norm(cls_features, axis=1, keepdims=True)
    normalized_cls = cls_features / np.maximum(norms, 1e-8)
    deltas = normalized_cls[1:] - normalized_cls[:-1]
    camera_rows: list[np.ndarray] = []
    evidence_rows: list[np.ndarray] = []
    quality: list[dict[str, Any]] = []
    prior = None if previous_delta is None else np.asarray(previous_delta, dtype=np.float32)
    for index in range(transitions):
        backward = None if backward_flow is None else np.asarray(backward_flow[index], dtype=np.float32)
        transform, fit_stats, fb_error = safe_camera_fit(
            forward_flow[index],
            backward,
            grid_step=grid_step,
            max_fb_error=max_fb_error,
            global_model=global_model,
        )
        first_delta = deltas[index]
        second_delta = np.zeros_like(first_delta) if prior is None else first_delta - prior
        prior = first_delta
        adjacent_cosine = 1.0 - float(np.clip(np.dot(normalized_cls[index], normalized_cls[index + 1]), -1.0, 1.0))
        evidence = np.concatenate(
            [
                first_delta,
                second_delta,
                np.asarray(
                    [adjacent_cosine, np.linalg.norm(first_delta), np.linalg.norm(second_delta)],
                    dtype=np.float32,
                ),
                _patch_pair_stats(patch_features[index], patch_features[index + 1]),
                _flow_stats(forward_flow[index], fb_error),
            ]
        ).astype(np.float32)
        camera_rows.append(_decompose_camera(transform, fit_stats, forward_flow[index], fb_error))
        evidence_rows.append(np.nan_to_num(evidence, nan=0.0, posinf=1e4, neginf=-1e4))
        quality.append(
            {
                **fit_stats,
                "transition_index": index,
                "median_forward_backward_error": (
                    float(np.nanmedian(fb_error)) if fb_error is not None and np.isfinite(fb_error).any() else None
                ),
            }
        )
    camera = np.stack(camera_rows).astype(np.float32)
    evidence = np.stack(evidence_rows).astype(np.float32)
    if camera.shape[1] != len(CAMERA_FEATURE_NAMES):
        raise AssertionError(f"camera feature contract changed: {camera.shape[1]} != {len(CAMERA_FEATURE_NAMES)}")
    if evidence.shape[1] != len(evidence_feature_names(cls_features.shape[1])):
        raise AssertionError("evidence feature contract changed")
    return camera, evidence, prior, quality
