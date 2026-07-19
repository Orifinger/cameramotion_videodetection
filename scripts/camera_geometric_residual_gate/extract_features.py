#!/usr/bin/env python3
"""Extract compact frozen DINOv2/RAFT geometry features on distributed GPUs."""

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
from scripts.camera_flow_probe.models import DinoV2Extractor, TorchvisionRaft
from scripts.camera_geometric_residual_gate.contracts import FEATURE_SCHEMA_VERSION, feature_filename, write_json
from scripts.camera_geometric_residual_gate.features import VARIANT_DIMS, extract_features


def _uniform_indices(length: int, count: int) -> np.ndarray:
    if length <= count:
        return np.arange(length, dtype=np.int64)
    return np.rint(np.linspace(0, length - 1, count)).astype(np.int64)


def _load_frames(paths: Sequence[str], max_frames: int) -> tuple[np.ndarray, list[str]]:
    selected = [paths[index] for index in _uniform_indices(len(paths), max_frames)]
    frames: list[np.ndarray] = []
    target_shape: tuple[int, int] | None = None
    for path in selected:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot decode frame: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if target_shape is None:
            target_shape = image.shape[:2]
        elif image.shape[:2] != target_shape:
            image = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
        frames.append(image)
    if len(frames) < 3:
        raise ValueError(f"need at least 3 frames, got {len(frames)}")
    return np.stack(frames), selected


def _extract_one(
    row: Mapping[str, Any],
    *,
    output_dir: Path,
    raft: TorchvisionRaft,
    dino: DinoV2Extractor,
    max_frames: int,
    grid_step: int,
    max_fb_error: float,
    ransac_threshold: float,
    overwrite: bool,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    dataset = str(row.get("dataset_name", "dataset")).casefold().replace("-", "_")
    feature_path = output_dir / dataset / feature_filename(sample_id)
    if feature_path.is_file() and not overwrite:
        with np.load(feature_path, allow_pickle=False) as archive:
            if str(archive["schema_version"].item()) == FEATURE_SCHEMA_VERSION:
                return {"sample_id": sample_id, "feature_path": str(feature_path), "reused": True}
    started = time.time()
    frames, selected_paths = _load_frames([str(value) for value in row.get("frame_paths") or []], max_frames)
    values, quality = extract_features(
        frames,
        raft=raft,
        dino=dino,
        grid_step=grid_step,
        max_fb_error=max_fb_error,
        ransac_threshold=ransac_threshold,
    )
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = feature_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(FEATURE_SCHEMA_VERSION),
        sample_id=np.asarray(sample_id),
        label=np.asarray(int(row["label"]), dtype=np.int64),
        **values,
    )
    temporary.replace(feature_path)
    return {
        "sample_id": sample_id,
        "feature_path": str(feature_path),
        "reused": False,
        "selected_frames": len(selected_paths),
        "quality": quality,
        "elapsed_sec": time.time() - started,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raft-checkpoint", type=Path, default=Path("/home/admin/raft_large_C_T_SKHT_V2-ff5fadd5.pth"))
    parser.add_argument("--dinov2-model", type=Path, default=Path("/home/admin/dinov2-small"))
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--flow-long-side", type=int, default=512)
    parser.add_argument("--dino-long-side", type=int, default=518)
    parser.add_argument("--raft-batch-size", type=int, default=4)
    parser.add_argument("--dino-batch-size", type=int, default=16)
    parser.add_argument("--grid-step", type=int, default=8)
    parser.add_argument("--max-forward-backward-error", type=float, default=2.0)
    parser.add_argument("--ransac-threshold", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    rows = sorted(read_jsonl(args.manifest_jsonl), key=lambda row: str(row["sample_id"]))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    assigned = [row for index, row in enumerate(rows) if index % world_size == rank]
    if args.device.startswith("cuda"):
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)
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
    for row in assigned:
        try:
            result = _extract_one(
                row,
                output_dir=args.output_dir,
                raft=raft,
                dino=dino,
                max_frames=args.max_frames,
                grid_step=args.grid_step,
                max_fb_error=args.max_forward_backward_error,
                ransac_threshold=args.ransac_threshold,
                overwrite=args.overwrite,
            )
            completed.append(result)
            print(f"[{rank}/{world_size}] OK {row['sample_id']} {result.get('elapsed_sec', 0.0):.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            failure = {"sample_id": row.get("sample_id"), "type": type(exc).__name__, "error": str(exc)}
            failures.append(failure)
            print(f"[{rank}/{world_size}] FAILED {failure}", file=sys.stderr, flush=True)
    rank_dir = args.output_dir / "ranks"
    summary = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "variant_dims": VARIANT_DIMS,
        "rank": rank,
        "world_size": world_size,
        "assigned_samples": len(assigned),
        "completed_samples": len(completed),
        "failed_samples": len(failures),
        "mean_elapsed_sec": float(np.mean([item.get("elapsed_sec", 0.0) for item in completed])) if completed else 0.0,
        "failures": failures,
    }
    write_json(rank_dir / f"rank_{rank:03d}_summary.json", summary)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
