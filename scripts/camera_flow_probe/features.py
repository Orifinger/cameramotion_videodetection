"""ReStraV-style global and local patch trajectory features."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from scripts.camera_flow_probe.geometry import CanvasGeometry, project_points


GLOBAL_FEATURE_DIM = 21
LOCAL_TRAJECTORY_DIM = 9
FLOW_SUMMARY_DIM = 4
LOCAL_FEATURE_DIM = LOCAL_TRAJECTORY_DIM + FLOW_SUMMARY_DIM


def _pad_prefix(values: np.ndarray, length: int) -> np.ndarray:
    output = np.zeros(length, dtype=np.float32)
    count = min(length, int(values.shape[0]))
    if count:
        output[:count] = values[:count]
    return output


def global_restrav_features(cls_features: np.ndarray) -> np.ndarray:
    features = np.asarray(cls_features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError(f"CLS trajectory must be [T,D] with T>=2, got {features.shape}")
    norm = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    distances = 1.0 - np.sum(norm[:-1] * norm[1:], axis=1)
    deltas = np.diff(norm, axis=0)
    deltas /= np.maximum(np.linalg.norm(deltas, axis=1, keepdims=True), 1e-8)
    angles = 1.0 - np.sum(deltas[:-1] * deltas[1:], axis=1) if deltas.shape[0] >= 2 else np.zeros(0, dtype=np.float32)

    def stats(values: np.ndarray) -> list[float]:
        if values.size == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(values.mean()), float(values.min()), float(values.max()), float(values.var())]

    output = np.concatenate(
        [
            _pad_prefix(distances, 7),
            _pad_prefix(angles, 6),
            np.asarray(stats(distances) + stats(angles), dtype=np.float32),
        ]
    )
    if output.shape != (GLOBAL_FEATURE_DIM,):
        raise AssertionError(output.shape)
    return output.astype(np.float32)


def _grid_for_transform(
    transform_source: np.ndarray,
    *,
    geometry: CanvasGeometry,
    patch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid_height = geometry.canvas_height // patch_size
    grid_width = geometry.canvas_width // patch_size
    yy, xx = np.mgrid[0:grid_height, 0:grid_width]
    canvas_points = np.stack(
        [(xx.ravel() + 0.5) * patch_size, (yy.ravel() + 0.5) * patch_size],
        axis=1,
    )
    source_points = project_points(canvas_points, np.linalg.inv(geometry.source_to_canvas))
    target_source = project_points(source_points, transform_source)
    target_canvas = project_points(target_source, geometry.source_to_canvas)
    normalized = np.stack(
        [
            2.0 * target_canvas[:, 0] / geometry.canvas_width - 1.0,
            2.0 * target_canvas[:, 1] / geometry.canvas_height - 1.0,
        ],
        axis=1,
    ).reshape(grid_height, grid_width, 2)
    target_source_grid = target_source.reshape(grid_height, grid_width, 2)
    source_grid = source_points.reshape(grid_height, grid_width, 2)
    valid = (
        np.isfinite(normalized).all(axis=2)
        & (np.abs(normalized[..., 0]) <= 1.0)
        & (np.abs(normalized[..., 1]) <= 1.0)
        & (source_grid[..., 0] >= 0.0)
        & (source_grid[..., 0] < geometry.source_width)
        & (source_grid[..., 1] >= 0.0)
        & (source_grid[..., 1] < geometry.source_height)
        & (target_source_grid[..., 0] >= 0.0)
        & (target_source_grid[..., 0] < geometry.source_width)
        & (target_source_grid[..., 1] >= 0.0)
        & (target_source_grid[..., 1] < geometry.source_height)
    )
    return (
        torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0).to(device),
        torch.from_numpy(valid).to(device),
    )


def align_patch_sequence(
    patch_features: np.ndarray,
    transforms_anchor_to_frame: Sequence[np.ndarray],
    *,
    geometry: CanvasGeometry,
    patch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    features = torch.from_numpy(np.asarray(patch_features, dtype=np.float32)).to(device)
    if features.ndim != 4 or features.shape[0] != len(transforms_anchor_to_frame):
        raise ValueError(
            f"patch trajectory/transform mismatch: features={tuple(features.shape)} "
            f"transforms={len(transforms_anchor_to_frame)}"
        )
    aligned: list[torch.Tensor] = []
    valid: list[torch.Tensor] = []
    for index, transform in enumerate(transforms_anchor_to_frame):
        grid, frame_valid = _grid_for_transform(
            transform,
            geometry=geometry,
            patch_size=patch_size,
            device=device,
        )
        sampled = F.grid_sample(
            features[index : index + 1],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0]
        aligned.append(sampled)
        valid.append(frame_valid)
    return torch.stack(aligned).cpu().numpy(), torch.stack(valid).cpu().numpy()


def _nan_summary(values: torch.Tensor, *, dimension: int) -> torch.Tensor:
    finite = torch.isfinite(values)
    count = finite.sum(dim=dimension).clamp_min(1)
    safe = torch.where(finite, values, torch.zeros_like(values))
    mean = safe.sum(dim=dimension) / count
    centered = torch.where(finite, values - mean.unsqueeze(dimension), torch.zeros_like(values))
    std = torch.sqrt((centered.square().sum(dim=dimension) / count).clamp_min(0.0))
    maximum = torch.where(finite, values, torch.full_like(values, float("-inf"))).amax(dim=dimension)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    quantile_input = torch.where(finite, values, torch.full_like(values, float("nan")))
    q90 = torch.nanquantile(quantile_input, 0.9, dim=dimension)
    q90 = torch.nan_to_num(q90)
    return torch.stack([mean, std, maximum, q90], dim=-1)


def local_trajectory_features(sequence: np.ndarray, valid: np.ndarray) -> np.ndarray:
    features = torch.from_numpy(np.asarray(sequence, dtype=np.float32))
    validity = torch.from_numpy(np.asarray(valid, dtype=bool))
    if features.ndim != 4 or validity.shape != (features.shape[0], features.shape[2], features.shape[3]):
        raise ValueError(f"invalid local trajectory shapes: features={features.shape} valid={validity.shape}")
    norm = F.normalize(features, dim=1, eps=1e-8)
    pair_valid = validity[:-1] & validity[1:]
    distances = 1.0 - (norm[:-1] * norm[1:]).sum(dim=1)
    distances = torch.where(pair_valid, distances, torch.full_like(distances, float("nan")))
    distance_stats = _nan_summary(distances, dimension=0)

    delta = F.normalize(norm[1:] - norm[:-1], dim=1, eps=1e-8)
    if delta.shape[0] >= 2:
        angle_valid = pair_valid[:-1] & pair_valid[1:]
        angles = 1.0 - (delta[:-1] * delta[1:]).sum(dim=1)
        angles = torch.where(angle_valid, angles, torch.full_like(angles, float("nan")))
        angle_stats = _nan_summary(angles, dimension=0)
    else:
        angle_stats = torch.zeros((*features.shape[2:], 4), dtype=torch.float32)
    valid_fraction = validity.float().mean(dim=0, keepdim=False).unsqueeze(-1)
    output = torch.cat([distance_stats, angle_stats, valid_fraction], dim=-1)
    return output.numpy().astype(np.float32)


def append_flow_summary(local_features: np.ndarray, flow_values: np.ndarray) -> np.ndarray:
    local = np.asarray(local_features, dtype=np.float32)
    flow = np.asarray(flow_values, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[1:] != local.shape[:2]:
        raise ValueError(f"flow/local shape mismatch: flow={flow.shape} local={local.shape}")
    summary = np.stack(
        [
            np.nanmean(flow, axis=0),
            np.nanstd(flow, axis=0),
            np.nanmax(flow, axis=0),
            np.nanquantile(flow, 0.9, axis=0),
        ],
        axis=-1,
    ).astype(np.float32)
    summary = np.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0)
    output = np.concatenate([local, summary], axis=-1)
    if output.shape[-1] != LOCAL_FEATURE_DIM:
        raise AssertionError(output.shape)
    return output
