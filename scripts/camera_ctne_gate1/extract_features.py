#!/usr/bin/env python3
"""Extract variable-length CTNE camera/evidence transition features."""

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

from scripts.camera_ctne_gate1.contracts import (
    FEATURE_SCHEMA_VERSION,
    dataset_slug,
    feature_filename,
    normalize_path,
    read_jsonl,
    validate_feature_archive,
    write_json,
    write_jsonl,
)
from scripts.camera_ctne_gate1.sampling import frame_chunks, uniform_frame_indices
from scripts.camera_ctne_gate1.transition_features import (
    CAMERA_FEATURE_NAMES,
    build_transition_features,
    evidence_feature_names,
)
from scripts.camera_flow_probe.models import DinoV2Extractor, TorchvisionRaft


def _decode_rgb(path: str, reference_shape: tuple[int, int] | None) -> tuple[np.ndarray, bool]:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to decode image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = False
    if reference_shape is not None and image.shape[:2] != reference_shape:
        height, width = reference_shape
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        resized = True
    return image, resized


def _feature_paths(output_dir: Path, row: Mapping[str, Any]) -> tuple[Path, Path]:
    slug = dataset_slug(row.get("dataset_slug") or row.get("dataset_name"))
    stem = Path(feature_filename(str(row["sample_id"]))).stem
    directory = output_dir / "features" / slug
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def extract_sample(
    row: Mapping[str, Any],
    *,
    raft: TorchvisionRaft,
    dino: DinoV2Extractor,
    output_dir: Path,
    max_frames: int,
    chunk_frames: int,
    grid_step: int,
    max_fb_error: float,
    global_model: str,
    overwrite: bool,
) -> dict[str, Any]:
    feature_path, metadata_path = _feature_paths(output_dir, row)
    if feature_path.is_file() and metadata_path.is_file() and not overwrite:
        validation = validate_feature_archive(feature_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["resumed"] = True
        metadata["validation"] = validation
        return metadata

    frame_paths = [normalize_path(value) for value in row.get("frame_paths") or []]
    original_count = len(frame_paths)
    if original_count < 3:
        raise ValueError(f"ctne_unavailable: fewer than 3 listed frames ({original_count})")
    selected_indices = uniform_frame_indices(original_count, max_frames)
    selected_paths = [frame_paths[index] for index in selected_indices]
    started = time.time()
    first, _ = _decode_rgb(selected_paths[0], None)
    reference_shape = (int(first.shape[0]), int(first.shape[1]))
    camera_parts: list[np.ndarray] = []
    evidence_parts: list[np.ndarray] = []
    quality: list[dict[str, Any]] = []
    previous_delta: np.ndarray | None = None
    resized_frame_references = 0
    transition_offset = 0

    for start, end in frame_chunks(len(selected_paths), chunk_frames):
        frames: list[np.ndarray] = []
        for local_index, path in enumerate(selected_paths[start:end]):
            if start == 0 and local_index == 0:
                image, resized = first, False
            else:
                image, resized = _decode_rgb(path, reference_shape)
            frames.append(image)
            resized_frame_references += int(resized)
        array = np.stack(frames)
        cls_features, patch_features, _ = dino.extract(array)
        forward, backward, _ = raft.infer_pairs(array, backward=True)
        camera, evidence, previous_delta, chunk_quality = build_transition_features(
            cls_features,
            patch_features,
            forward,
            backward,
            previous_delta=previous_delta,
            grid_step=grid_step,
            max_fb_error=max_fb_error,
            global_model=global_model,
        )
        for item in chunk_quality:
            item["transition_index"] = int(item["transition_index"]) + transition_offset
        transition_offset += camera.shape[0]
        camera_parts.append(camera)
        evidence_parts.append(evidence)
        quality.extend(chunk_quality)

    camera_context = np.concatenate(camera_parts, axis=0).astype(np.float32)
    temporal_evidence = np.concatenate(evidence_parts, axis=0).astype(np.float32)
    if camera_context.shape[0] != len(selected_paths) - 1:
        raise AssertionError("feature extraction lost or duplicated adjacent transitions")
    if not np.isfinite(camera_context).all() or not np.isfinite(temporal_evidence).all():
        raise ValueError("non-finite CTNE feature values")

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = feature_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(FEATURE_SCHEMA_VERSION),
        sample_id=np.asarray(str(row["sample_id"])),
        dataset_name=np.asarray(str(row.get("dataset_name", ""))),
        label=np.asarray(int(row["label"]), dtype=np.int64),
        frame_count=np.asarray(original_count, dtype=np.int64),
        selected_frame_count=np.asarray(len(selected_paths), dtype=np.int64),
        selected_frame_indices=np.asarray(selected_indices, dtype=np.int64),
        camera_context=camera_context,
        temporal_evidence=temporal_evidence,
    )
    temporary.replace(feature_path)
    model_counts: dict[str, int] = {}
    for item in quality:
        key = str(item.get("model", "unknown"))
        model_counts[key] = model_counts.get(key, 0) + 1
    metadata = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "sample_id": str(row["sample_id"]),
        "dataset_name": row.get("dataset_name"),
        "dataset_split": row.get("dataset_split"),
        "label_name": row.get("label_name"),
        "source_name": row.get("source_name"),
        "generator_name": row.get("generator_name"),
        "motion_bucket": row.get("motion_bucket"),
        "feature_path": normalize_path(feature_path),
        "frame_contract": {
            "original_frame_count": original_count,
            "selected_frame_count": len(selected_paths),
            "selected_frame_indices": selected_indices,
            "max_frames": max_frames,
            "chunk_frames": chunk_frames,
            "transition_count": int(camera_context.shape[0]),
            "reference_height": reference_shape[0],
            "reference_width": reference_shape[1],
            "resized_frame_references": resized_frame_references,
        },
        "feature_contract": {
            "camera_dim": int(camera_context.shape[1]),
            "evidence_dim": int(temporal_evidence.shape[1]),
            "camera_feature_names": list(CAMERA_FEATURE_NAMES),
            "evidence_feature_names": list(evidence_feature_names((temporal_evidence.shape[1] - 58) // 2)),
            "camera_and_evidence_are_separate": True,
            "camera_residual_subtraction_used": False,
        },
        "quality": {
            "camera_fit_model_counts": model_counts,
            "fit_exception_fallback_count": sum(bool(item.get("fit_exception_fallback")) for item in quality),
            "median_camera_inlier_rate": float(np.median([float(item.get("inlier_rate", 0.0)) for item in quality])),
            "median_reprojection_error": float(
                np.median([float(item.get("median_reprojection_error", 0.0)) for item in quality])
            ),
        },
        "elapsed_sec": time.time() - started,
        "resumed": False,
    }
    write_json(metadata_path, metadata)
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raft-checkpoint", type=Path, default=Path("/home/admin/raft_large_C_T_SKHT_V2-ff5fadd5.pth"))
    parser.add_argument("--dinov2-model", type=Path, default=Path("/home/admin/dinov2-small"))
    parser.add_argument("--split", default="all", help="all or comma-separated split names")
    parser.add_argument("--max-frames", type=int, default=0, help="0 uses every listed frame; formal default")
    parser.add_argument("--chunk-frames", type=int, default=32)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--flow-long-side", type=int, default=512)
    parser.add_argument("--dino-long-side", type=int, default=518)
    parser.add_argument("--raft-batch-size", type=int, default=4)
    parser.add_argument("--dino-batch-size", type=int, default=16)
    parser.add_argument("--global-model", choices=("homography", "affine"), default="homography")
    parser.add_argument("--flow-grid-step", type=int, default=8)
    parser.add_argument("--max-forward-backward-error", type=float, default=2.0)
    parser.add_argument("--max-failure-rate", type=float, default=0.02)
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
    rows = sorted(read_jsonl(args.manifest_jsonl), key=lambda value: str(value["sample_id"]))
    if args.split != "all":
        allowed = {value.strip() for value in args.split.split(",") if value.strip()}
        rows = [row for row in rows if str(row.get("dataset_split")) in allowed]
    if args.max_cases > 0:
        rows = rows[: args.max_cases]
    assigned = [row for index, row in enumerate(rows) if index % world_size == rank]
    eligible: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for row in assigned:
        if bool(row.get("ctne_available", int(row.get("frame_count", 0)) >= 3)):
            eligible.append(row)
        else:
            unavailable.append(
                {"sample_id": row.get("sample_id"), "frame_count": int(row.get("frame_count", 0))}
            )
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
    for row in eligible:
        try:
            metadata = extract_sample(
                row,
                raft=raft,
                dino=dino,
                output_dir=args.output_dir,
                max_frames=args.max_frames,
                chunk_frames=args.chunk_frames,
                grid_step=args.flow_grid_step,
                max_fb_error=args.max_forward_backward_error,
                global_model=args.global_model,
                overwrite=args.overwrite,
            )
            completed.append(metadata)
            print(
                f"[{rank}/{world_size}] OK {row['sample_id']} "
                f"frames={row.get('frame_count')} elapsed={metadata.get('elapsed_sec', 0.0):.1f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failure = {
                "sample_id": row.get("sample_id"),
                "type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(f"[{rank}/{world_size}] FAILED {failure}", file=sys.stderr, flush=True)
    rank_dir = args.output_dir / "ranks"
    rank_dir.mkdir(parents=True, exist_ok=True)
    failure_rate = len(failures) / len(eligible) if eligible else 0.0
    summary = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "rank": rank,
        "world_size": world_size,
        "assigned_records": len(assigned),
        "eligible_records": len(eligible),
        "unavailable_records": unavailable,
        "completed_records": len(completed),
        "failed_records": len(failures),
        "failure_rate": failure_rate,
        "max_failure_rate": args.max_failure_rate,
        "max_frames": args.max_frames,
        "formal_variable_length_contract": args.max_frames == 0,
        "failures": failures,
    }
    write_json(rank_dir / f"rank_{rank:03d}_summary.json", summary)
    write_jsonl(rank_dir / f"rank_{rank:03d}_completed.jsonl", completed)
    return 0 if failure_rate <= args.max_failure_rate else 2


if __name__ == "__main__":
    raise SystemExit(main())
