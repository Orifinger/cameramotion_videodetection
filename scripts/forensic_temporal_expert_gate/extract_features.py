#!/usr/bin/env python3
"""Extract frozen, native-aspect DINOv2 frame tokens on distributed GPUs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.forensic_temporal_expert_gate import FEATURE_SCHEMA_VERSION
from scripts.forensic_temporal_expert_gate.contracts import (
    feature_filename,
    normalize_path,
    read_json_or_jsonl,
    resized_shape,
    write_json,
    write_jsonl,
)


def resolve_model_root(path: Path) -> Path:
    if (path / "config.json").is_file():
        return path
    candidates = [item.parent for item in path.rglob("config.json")]
    candidates = [
        item
        for item in candidates
        if list(item.glob("*.safetensors")) or list(item.glob("pytorch_model*.bin"))
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one local Hugging Face model under {path}")
    return candidates[0]


def load_frame(
    path: str,
    *,
    patch_size: int,
    max_pixels: int,
    max_side: int,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        original = (image.width, image.height)
        target = resized_shape(
            image.width,
            image.height,
            patch_size=patch_size,
            max_pixels=max_pixels,
            max_side=max_side,
        )
        if image.size != target:
            image = image.resize(target, Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
    return tensor, original, target


class NativeDinoExtractor:
    def __init__(self, path: Path, device: torch.device, grid_size: int) -> None:
        from transformers import AutoModel

        self.root = resolve_model_root(path)
        self.device = device
        self.grid_size = grid_size
        self.model = AutoModel.from_pretrained(
            self.root, local_files_only=True, trust_remote_code=False
        ).to(device).eval()
        self.patch_size = int(getattr(self.model.config, "patch_size", 14))
        self.hidden_size = int(getattr(self.model.config, "hidden_size"))
        self.register_tokens = int(
            getattr(self.model.config, "num_register_tokens", 0) or 0
        )
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.inference_mode()
    def run(self, values: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        normalized = (values - self.mean) / self.std
        normalized = normalized.to(self.device, non_blocking=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            try:
                output = self.model(
                    pixel_values=normalized, interpolate_pos_encoding=True
                )
            except TypeError:
                output = self.model(pixel_values=normalized)
        hidden = output.last_hidden_state.float()
        grid_height = values.shape[-2] // self.patch_size
        grid_width = values.shape[-1] // self.patch_size
        expected = grid_height * grid_width
        patches = hidden[:, -expected:]
        if patches.shape[1] != expected:
            raise ValueError(
                f"DINO patch mismatch: expected={expected}, actual={patches.shape[1]}"
            )
        patches = patches.transpose(1, 2).reshape(
            -1, hidden.shape[-1], grid_height, grid_width
        )
        patches = functional.adaptive_avg_pool2d(
            patches, (self.grid_size, self.grid_size)
        )
        patches = patches.flatten(2).transpose(1, 2)
        return (
            hidden[:, 0].cpu().numpy().astype(np.float16),
            patches.cpu().numpy().astype(np.float16),
        )


def output_path(root: Path, sample_id: str) -> Path:
    return root / "features" / feature_filename(sample_id)


def validate_existing(path: Path, expected_frames: int, grid_tokens: int) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            cls = archive["cls_tokens"]
            patches = archive["patch_tokens"]
            return (
                str(archive["schema_version"].item()) == FEATURE_SCHEMA_VERSION
                and cls.ndim == 2
                and patches.ndim == 3
                and cls.shape[0] == expected_frames
                and patches.shape[:2] == (expected_frames, grid_tokens)
                and cls.shape[-1] == patches.shape[-1]
                and np.isfinite(cls).all()
                and np.isfinite(patches).all()
            )
    except Exception:
        return False


def extract_one(
    row: Mapping[str, Any],
    *,
    extractor: NativeDinoExtractor,
    root: Path,
    batch_size: int,
    max_pixels: int,
    max_side: int,
    overwrite: bool,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    frames = [normalize_path(value) for value in row.get("frame_paths") or []]
    if len(frames) < 2:
        raise ValueError(f"fewer than two frames: {sample_id}")
    path = output_path(root, sample_id)
    grid_tokens = extractor.grid_size * extractor.grid_size
    if path.is_file() and not overwrite and validate_existing(path, len(frames), grid_tokens):
        return {"sample_id": sample_id, "feature_path": normalize_path(path), "resumed": True}

    started = time.time()
    tensors: list[torch.Tensor] = []
    originals: list[tuple[int, int]] = []
    processed: list[tuple[int, int]] = []
    for frame in frames:
        tensor, original, target = load_frame(
            frame,
            patch_size=extractor.patch_size,
            max_pixels=max_pixels,
            max_side=max_side,
        )
        tensors.append(tensor)
        originals.append(original)
        processed.append(target)

    by_shape: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for index, tensor in enumerate(tensors):
        by_shape[(int(tensor.shape[-2]), int(tensor.shape[-1]))].append(index)
    cls_output = np.empty((len(frames), extractor.hidden_size), dtype=np.float16)
    patch_output = np.empty(
        (len(frames), grid_tokens, extractor.hidden_size), dtype=np.float16
    )
    for indices in by_shape.values():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            cls, patches = extractor.run(torch.stack([tensors[index] for index in selected]))
            cls_output[selected] = cls
            patch_output[selected] = patches

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            schema_version=np.asarray(FEATURE_SCHEMA_VERSION),
            sample_id=np.asarray(sample_id),
            cls_tokens=cls_output,
            patch_tokens=patch_output,
            original_sizes=np.asarray(originals, dtype=np.int32),
            processed_sizes=np.asarray(processed, dtype=np.int32),
        )
    temporary.replace(path)
    return {
        "sample_id": sample_id,
        "feature_path": normalize_path(path),
        "frame_count": len(frames),
        "hidden_size": extractor.hidden_size,
        "grid_tokens": grid_tokens,
        "processed_size_counts": dict(
            sorted((f"{width}x{height}", processed.count((width, height))) for width, height in set(processed))
        ),
        "elapsed_sec": time.time() - started,
        "resumed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dinov2-model", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--max-side", type=int, default=672)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-failure-rate", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    rows = read_json_or_jsonl(args.manifest_jsonl)
    if args.max_cases > 0:
        rows = rows[: args.max_cases]
    assigned = [row for index, row in enumerate(rows) if index % world_size == rank]
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    extractor = NativeDinoExtractor(args.dinov2_model, device, args.grid_size)
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in assigned:
        try:
            item = extract_one(
                row,
                extractor=extractor,
                root=args.output_dir,
                batch_size=args.batch_size,
                max_pixels=args.max_pixels,
                max_side=args.max_side,
                overwrite=args.overwrite,
            )
            completed.append(item)
            print(
                f"[{rank}/{world_size}] OK {row['sample_id']} "
                f"frames={row.get('frame_count')} elapsed={item.get('elapsed_sec', 0):.2f}s",
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
    write_jsonl(rank_dir / f"rank_{rank:03d}_completed.jsonl", completed)
    failure_rate = len(failures) / len(assigned) if assigned else 0.0
    summary = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "rank": rank,
        "world_size": world_size,
        "assigned_records": len(assigned),
        "completed_records": len(completed),
        "failed_records": len(failures),
        "failure_rate": failure_rate,
        "failures": failures,
    }
    write_json(rank_dir / f"rank_{rank:03d}_summary.json", summary)
    return 0 if failure_rate <= args.max_failure_rate else 2


if __name__ == "__main__":
    raise SystemExit(main())
