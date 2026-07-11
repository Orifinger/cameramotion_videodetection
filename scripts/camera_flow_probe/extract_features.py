#!/usr/bin/env python3
"""Extract global, local-unaligned, and camera-aligned Data A probe features."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_flow_probe.contracts import read_jsonl
from scripts.camera_flow_probe.features import (
    LOCAL_FEATURE_DIM,
    align_patch_sequence,
    append_flow_summary,
    global_restrav_features,
    local_trajectory_features,
)
from scripts.camera_flow_probe.geometry import (
    CanvasGeometry,
    compose_transforms,
    dense_transform_flow,
    fit_global_camera_transform,
    forward_backward_error,
    project_points,
    transform_flow_to_source,
)
from scripts.camera_flow_probe.masks import MaskTube, load_mask_tube
from scripts.camera_flow_probe.models import DinoV2Extractor, TorchvisionRaft
from scripts.camera_flow_probe.video import paired_dense_frames, sliding_windows


FEATURE_SCHEMA_VERSION = "dataA_camera_flow_probe_features_v1"


def _source_scalar_map(values: np.ndarray, geometry: CanvasGeometry, *, vector_scale: bool) -> np.ndarray:
    resized_height = int(round(geometry.source_height * geometry.scale))
    resized_width = int(round(geometry.source_width * geometry.scale))
    top, left = geometry.pad_top, geometry.pad_left
    cropped = np.asarray(values, dtype=np.float32)[top : top + resized_height, left : left + resized_width]
    source = cv2.resize(cropped, (geometry.source_width, geometry.source_height), interpolation=cv2.INTER_LINEAR)
    if vector_scale:
        source = source / geometry.scale
    return source.astype(np.float32)


def _patch_source_coordinates(geometry: CanvasGeometry, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    grid_height = geometry.canvas_height // patch_size
    grid_width = geometry.canvas_width // patch_size
    yy, xx = np.mgrid[0:grid_height, 0:grid_width]
    canvas_points = np.stack(
        [(xx.ravel() + 0.5) * patch_size, (yy.ravel() + 0.5) * patch_size],
        axis=1,
    )
    source = project_points(canvas_points, np.linalg.inv(geometry.source_to_canvas))
    return (
        source[:, 0].reshape(grid_height, grid_width).astype(np.float32),
        source[:, 1].reshape(grid_height, grid_width).astype(np.float32),
    )


def _sample_source_to_patches(values: np.ndarray, geometry: CanvasGeometry, patch_size: int) -> np.ndarray:
    map_x, map_y = _patch_source_coordinates(geometry, patch_size)
    return cv2.remap(
        np.asarray(values, dtype=np.float32),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )


def _mask_patch_coverage(mask: np.ndarray, geometry: CanvasGeometry, patch_size: int) -> np.ndarray:
    resized_height = int(round(geometry.source_height * geometry.scale))
    resized_width = int(round(geometry.source_width * geometry.scale))
    resized = cv2.resize(mask.astype(np.float32), (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((geometry.canvas_height, geometry.canvas_width), dtype=np.float32)
    canvas[
        geometry.pad_top : geometry.pad_top + resized_height,
        geometry.pad_left : geometry.pad_left + resized_width,
    ] = resized
    grid_height = geometry.canvas_height // patch_size
    grid_width = geometry.canvas_width // patch_size
    return cv2.resize(canvas, (grid_width, grid_height), interpolation=cv2.INTER_AREA).astype(np.float32)


def _warp_to_anchor(values: np.ndarray, anchor_to_frame: np.ndarray) -> np.ndarray:
    height, width = values.shape[:2]
    return cv2.warpPerspective(
        np.asarray(values, dtype=np.float32),
        np.linalg.inv(anchor_to_frame),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )


def _warp_mask_to_anchor(mask: np.ndarray, anchor_to_frame: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    return cv2.warpPerspective(
        mask.astype(np.float32),
        np.linalg.inv(anchor_to_frame),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _estimate_motion(
    frames: np.ndarray,
    raft: TorchvisionRaft,
    *,
    global_model: str,
    grid_step: int,
    max_fb_error: float,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    forward, backward, flow_geometry = raft.infer_pairs(frames, backward=True)
    transforms: list[np.ndarray] = []
    raw_source: list[np.ndarray] = []
    residual_source: list[np.ndarray] = []
    stats: list[dict[str, Any]] = []
    for index, flow in enumerate(forward):
        fb = None if backward is None else forward_backward_error(flow, backward[index])
        transform_canvas, fit_stats = fit_global_camera_transform(
            flow,
            fb_error=fb,
            grid_step=grid_step,
            max_fb_error=max_fb_error,
            model="affine" if global_model == "affine" else "homography",
        )
        global_flow = dense_transform_flow(transform_canvas, flow.shape[0], flow.shape[1])
        raw_magnitude = np.linalg.norm(flow, axis=2)
        residual_magnitude = np.linalg.norm(flow - global_flow, axis=2)
        transforms.append(transform_flow_to_source(transform_canvas, flow_geometry))
        raw_source.append(_source_scalar_map(raw_magnitude, flow_geometry, vector_scale=True))
        residual_source.append(_source_scalar_map(residual_magnitude, flow_geometry, vector_scale=True))
        if fb is not None:
            fit_stats["median_forward_backward_error"] = float(np.nanmedian(fb))
        fit_stats["median_raw_flow_source_px"] = float(np.nanmedian(raw_source[-1]))
        fit_stats["median_residual_flow_source_px"] = float(np.nanmedian(residual_source[-1]))
        fit_stats["transform_source"] = transforms[-1].tolist()
        stats.append(fit_stats)
    return transforms, np.stack(raw_source), np.stack(residual_source), stats


def _window_mask_maps(
    tube: MaskTube | None,
    timestamps: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    if tube is None:
        return np.zeros((timestamps.shape[0], height, width), dtype=np.uint8)
    return np.stack([tube.sample(float(timestamp), height=height, width=width) for timestamp in timestamps])


def _paired_camera_consistency(
    real_quality: Mapping[str, Any],
    fake_quality: Mapping[str, Any],
    *,
    height: int,
    width: int,
) -> dict[str, float]:
    real_stats = list(real_quality.get("camera_fit") or [])
    fake_stats = list(fake_quality.get("camera_fit") or [])
    count = min(len(real_stats), len(fake_stats))
    if count == 0:
        return {"num_frame_pairs": 0, "median_corner_error_px": float("nan"), "median_corner_error_normalized": float("nan")}
    corners = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0], [width - 1.0, height - 1.0], [width / 2.0, height / 2.0]],
        dtype=np.float64,
    )
    errors: list[float] = []
    for index in range(count):
        real_transform = np.asarray(real_stats[index]["transform_source"], dtype=np.float64)
        fake_transform = np.asarray(fake_stats[index]["transform_source"], dtype=np.float64)
        difference = np.linalg.norm(
            project_points(corners, real_transform) - project_points(corners, fake_transform),
            axis=1,
        )
        errors.append(float(np.nanmedian(difference)))
    median = float(np.nanmedian(errors))
    diagonal = float(np.hypot(height, width))
    return {
        "num_frame_pairs": count,
        "median_corner_error_px": median,
        "p90_corner_error_px": float(np.nanpercentile(errors, 90)),
        "median_corner_error_normalized": median / diagonal if diagonal > 0 else float("nan"),
    }


def _window_features(
    *,
    cls_features: np.ndarray,
    patch_features: np.ndarray,
    dino_geometry: CanvasGeometry,
    patch_size: int,
    pair_transforms_source: Sequence[np.ndarray],
    raw_flow_source: np.ndarray,
    residual_flow_source: np.ndarray,
    masks_source: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    cumulative = compose_transforms(list(pair_transforms_source))
    aligned_sequence, aligned_valid = align_patch_sequence(
        patch_features,
        cumulative,
        geometry=dino_geometry,
        patch_size=patch_size,
        device=device,
    )
    identity_transforms = [np.eye(3, dtype=np.float64) for _ in range(patch_features.shape[0])]
    unaligned_sequence, unaligned_valid = align_patch_sequence(
        patch_features,
        identity_transforms,
        geometry=dino_geometry,
        patch_size=patch_size,
        device=device,
    )
    aligned_local = local_trajectory_features(aligned_sequence, aligned_valid)
    unaligned_local = local_trajectory_features(unaligned_sequence, unaligned_valid)

    aligned_residual: list[np.ndarray] = []
    unaligned_raw: list[np.ndarray] = []
    for pair_index in range(raw_flow_source.shape[0]):
        unaligned_raw.append(_sample_source_to_patches(raw_flow_source[pair_index], dino_geometry, patch_size))
        aligned_map = _warp_to_anchor(residual_flow_source[pair_index], cumulative[pair_index])
        aligned_residual.append(_sample_source_to_patches(aligned_map, dino_geometry, patch_size))
    aligned_local = append_flow_summary(aligned_local, np.stack(aligned_residual))
    unaligned_local = append_flow_summary(unaligned_local, np.stack(unaligned_raw))

    unaligned_masks = [_mask_patch_coverage(mask, dino_geometry, patch_size) for mask in masks_source]
    aligned_masks = [
        _mask_patch_coverage(_warp_mask_to_anchor(mask, cumulative[index]), dino_geometry, patch_size)
        for index, mask in enumerate(masks_source)
    ]
    return {
        "global": global_restrav_features(cls_features),
        "local_unaligned": unaligned_local.reshape(-1, LOCAL_FEATURE_DIM),
        "local_aligned": aligned_local.reshape(-1, LOCAL_FEATURE_DIM),
        "valid_unaligned": (unaligned_valid.mean(axis=0).reshape(-1) >= 0.5),
        "valid_aligned": (aligned_valid.mean(axis=0).reshape(-1) >= 0.5),
        "mask_unaligned": np.max(np.stack(unaligned_masks), axis=0).reshape(-1).astype(np.float32),
        "mask_aligned": np.max(np.stack(aligned_masks), axis=0).reshape(-1).astype(np.float32),
    }


def _role_features(
    frames: np.ndarray,
    timestamps: np.ndarray,
    *,
    raft: TorchvisionRaft,
    dino: DinoV2Extractor,
    mask_tube: MaskTube | None,
    target_fps: float,
    window_frames: int,
    stride_frames: int,
    max_windows: int | None,
    global_model: str,
    grid_step: int,
    max_fb_error: float,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    del target_fps
    cls_features, patch_features, dino_geometry = dino.extract(frames)
    transforms, raw_flow, residual_flow, motion_stats = _estimate_motion(
        frames,
        raft,
        global_model=global_model,
        grid_step=grid_step,
        max_fb_error=max_fb_error,
    )
    windows = sliding_windows(
        frames.shape[0],
        window_frames=window_frames,
        stride_frames=stride_frames,
        max_windows=max_windows,
    )
    if not windows:
        raise ValueError("video produced no valid trajectory windows")
    outputs: dict[str, list[np.ndarray]] = {
        "global": [],
        "local_unaligned": [],
        "local_aligned": [],
        "valid_unaligned": [],
        "valid_aligned": [],
        "mask_unaligned": [],
        "mask_aligned": [],
    }
    window_times: list[list[float]] = []
    for start, end in windows:
        masks = _window_mask_maps(
            mask_tube,
            timestamps[start:end],
            height=frames.shape[1],
            width=frames.shape[2],
        )
        values = _window_features(
            cls_features=cls_features[start:end],
            patch_features=patch_features[start:end],
            dino_geometry=dino_geometry,
            patch_size=dino.patch_size,
            pair_transforms_source=transforms[start : end - 1],
            raw_flow_source=raw_flow[start : end - 1],
            residual_flow_source=residual_flow[start : end - 1],
            masks_source=masks,
            device=device,
        )
        for key, value in values.items():
            outputs[key].append(value)
        window_times.append([float(timestamps[start]), float(timestamps[end - 1])])
    stacked = {key: np.stack(values) for key, values in outputs.items()}
    quality = {
        "window_times_sec": window_times,
        "num_sampled_frames": int(frames.shape[0]),
        "num_windows": len(windows),
        "patch_grid": [int(patch_features.shape[2]), int(patch_features.shape[3])],
        "camera_fit": motion_stats,
        "median_camera_inlier_rate": float(np.median([value["inlier_rate"] for value in motion_stats])),
        "median_camera_reprojection_error": float(
            np.median([value["median_reprojection_error"] for value in motion_stats])
        ),
    }
    return stacked, quality


def extract_case(
    row: Mapping[str, Any],
    *,
    raft: TorchvisionRaft,
    dino: DinoV2Extractor,
    output_dir: Path,
    target_fps: float,
    window_frames: int,
    stride_frames: int,
    max_windows: int | None,
    max_sampled_frames: int | None,
    global_model: str,
    grid_step: int,
    max_fb_error: float,
    mask_positive_threshold: float,
    device: torch.device,
    overwrite: bool,
) -> dict[str, Any]:
    case_id = str(row["case_id"])
    feature_path = output_dir / "features" / f"{case_id}.npz"
    metadata_path = output_dir / "features" / f"{case_id}.json"
    if feature_path.is_file() and metadata_path.is_file() and not overwrite:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    started = time.time()
    real_frames, fake_frames, timestamps, meta = paired_dense_frames(
        Path(str(row["real_video"])),
        Path(str(row["fake_video"])),
        target_fps=target_fps,
    )
    if max_sampled_frames is not None and max_sampled_frames >= 2 and timestamps.size > max_sampled_frames:
        time_range = row.get("edit_time_range_source_sec") or []
        if isinstance(time_range, Sequence) and not isinstance(time_range, (str, bytes)) and len(time_range) >= 2:
            center_time = (float(time_range[0]) + float(time_range[1])) / 2.0
            center_index = int(np.argmin(np.abs(timestamps - center_time)))
        else:
            center_index = timestamps.size // 2
        start = max(0, min(timestamps.size - max_sampled_frames, center_index - max_sampled_frames // 2))
        end = start + max_sampled_frames
        real_frames = real_frames[start:end]
        fake_frames = fake_frames[start:end]
        timestamps = timestamps[start:end]
    tube = load_mask_tube(Path(str(row["mask_npz"])), Path(str(row["case_manifest"])))
    common = {
        "raft": raft,
        "dino": dino,
        "target_fps": target_fps,
        "window_frames": window_frames,
        "stride_frames": stride_frames,
        "max_windows": max_windows,
        "global_model": global_model,
        "grid_step": grid_step,
        "max_fb_error": max_fb_error,
        "device": device,
    }
    real, real_quality = _role_features(real_frames, timestamps, mask_tube=None, **common)
    fake, fake_quality = _role_features(fake_frames, timestamps, mask_tube=tube, **common)
    arrays: dict[str, np.ndarray] = {
        "timestamps_sec": timestamps.astype(np.float32),
    }
    for role, values in (("real", real), ("fake", fake)):
        for key, value in values.items():
            arrays[f"{role}_{key}"] = value
    arrays["fake_label_unaligned"] = (arrays["fake_mask_unaligned"] >= mask_positive_threshold).astype(np.uint8)
    arrays["fake_label_aligned"] = (arrays["fake_mask_aligned"] >= mask_positive_threshold).astype(np.uint8)
    arrays["real_label_unaligned"] = np.zeros_like(arrays["real_valid_unaligned"], dtype=np.uint8)
    arrays["real_label_aligned"] = np.zeros_like(arrays["real_valid_aligned"], dtype=np.uint8)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = feature_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(feature_path)
    metadata = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "case_id": case_id,
        "dataset_split": row.get("dataset_split"),
        "motion_bucket": row.get("motion_bucket"),
        "source_name": row.get("source_name"),
        "vace_model": row.get("vace_model"),
        "operation": row.get("operation"),
        "feature_path": str(feature_path),
        "video_meta": {
            "fps": meta.fps,
            "frame_count": meta.frame_count,
            "height": meta.height,
            "width": meta.width,
        },
        "sampling": {
            "target_fps": target_fps,
            "window_frames": window_frames,
            "stride_frames": stride_frames,
            "max_windows": max_windows,
            "max_sampled_frames": max_sampled_frames,
        },
        "real_quality": real_quality,
        "fake_quality": fake_quality,
        "paired_camera_consistency": _paired_camera_consistency(
            real_quality,
            fake_quality,
            height=meta.height,
            width=meta.width,
        ),
        "elapsed_sec": time.time() - started,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raft-checkpoint", type=Path, default=Path("/home/admin/raft_large_C_T_SKHT_V2-ff5fadd5.pth"))
    parser.add_argument("--dinov2-model", type=Path, default=Path("/home/admin/dinov2-small"))
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--window-frames", type=int, default=16)
    parser.add_argument("--stride-frames", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--max-sampled-frames", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--flow-long-side", type=int, default=512)
    parser.add_argument("--dino-long-side", type=int, default=518)
    parser.add_argument("--raft-batch-size", type=int, default=4)
    parser.add_argument("--dino-batch-size", type=int, default=16)
    parser.add_argument("--global-model", choices=("homography", "affine"), default="homography")
    parser.add_argument("--flow-grid-step", type=int, default=8)
    parser.add_argument("--max-forward-backward-error", type=float, default=2.0)
    parser.add_argument("--mask-positive-threshold", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if args.device.startswith("cuda"):
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)
    rows = sorted(read_jsonl(args.manifest_jsonl), key=lambda value: str(value.get("case_id")))
    if args.split != "all":
        rows = [row for row in rows if row.get("dataset_split") == args.split]
    if args.max_cases > 0:
        rows = rows[: args.max_cases]
    rows = [row for index, row in enumerate(rows) if index % world_size == rank]
    raft = TorchvisionRaft(
        args.raft_checkpoint,
        device=device,
        long_side=args.flow_long_side,
        batch_size=args.raft_batch_size,
    )
    dino = DinoV2Extractor(
        args.dinov2_model,
        device=device,
        long_side=args.dino_long_side,
        batch_size=args.dino_batch_size,
    )
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        try:
            metadata = extract_case(
                row,
                raft=raft,
                dino=dino,
                output_dir=args.output_dir,
                target_fps=args.target_fps,
                window_frames=args.window_frames,
                stride_frames=args.stride_frames,
                max_windows=args.max_windows if args.max_windows > 0 else None,
                max_sampled_frames=args.max_sampled_frames if args.max_sampled_frames > 0 else None,
                global_model=args.global_model,
                grid_step=args.flow_grid_step,
                max_fb_error=args.max_forward_backward_error,
                mask_positive_threshold=args.mask_positive_threshold,
                device=device,
                overwrite=bool(args.overwrite),
            )
            completed.append(metadata)
            print(f"[{rank}/{world_size}] OK {row['case_id']} elapsed={metadata.get('elapsed_sec', 0.0):.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            failure = {"case_id": row.get("case_id"), "type": type(exc).__name__, "error": str(exc)}
            failures.append(failure)
            print(f"[{rank}/{world_size}] FAILED {failure}", file=sys.stderr, flush=True)
    rank_dir = args.output_dir / "ranks"
    rank_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "rank": rank,
        "world_size": world_size,
        "assigned_cases": len(rows),
        "completed_cases": len(completed),
        "failed_cases": len(failures),
        "failures": failures,
    }
    (rank_dir / f"rank_{rank:03d}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
