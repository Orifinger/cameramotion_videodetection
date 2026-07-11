#!/usr/bin/env python3
"""Validate all offline model files needed by the camera-flow probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_flow_probe.contracts import sha256


def _resolve_file(path: Path, name: str) -> Path | None:
    if path.is_file() and path.name == name:
        return path
    direct = path / name
    if direct.is_file():
        return direct
    matches = list(path.rglob(name)) if path.is_dir() else []
    return matches[0] if len(matches) == 1 else None


def inspect_weights(
    *,
    raft_checkpoint: Path,
    dinov2_model: Path,
    sea_raft_model: Path | None,
) -> dict[str, Any]:
    errors: list[str] = []
    raft_digest = ""
    if not raft_checkpoint.is_file():
        errors.append(f"missing RAFT checkpoint: {raft_checkpoint}")
    else:
        raft_digest = sha256(raft_checkpoint)
        if not raft_digest.startswith("ff5fadd5"):
            errors.append(
                "RAFT checkpoint SHA256 does not start with ff5fadd5: "
                f"{raft_checkpoint}: {raft_digest}"
            )

    dino_weight = _resolve_file(dinov2_model, "model.safetensors")
    dino_config = _resolve_file(dinov2_model, "config.json")
    dino_preprocessor = _resolve_file(dinov2_model, "preprocessor_config.json")
    if dino_weight is None:
        errors.append(f"cannot resolve DINOv2 model.safetensors under: {dinov2_model}")
    if dino_config is None:
        errors.append(f"cannot resolve DINOv2 config.json under: {dinov2_model}")
    if dino_preprocessor is None:
        errors.append(f"cannot resolve DINOv2 preprocessor_config.json under: {dinov2_model}")

    sea_weight = None
    if sea_raft_model is not None:
        sea_weight = _resolve_file(sea_raft_model, "model.safetensors")
        if sea_weight is None:
            errors.append(f"cannot resolve SEA-RAFT model.safetensors under: {sea_raft_model}")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "raft": {
            "checkpoint": str(raft_checkpoint),
            "sha256": raft_digest,
            "exists": raft_checkpoint.is_file(),
        },
        "dinov2": {
            "root": str(dinov2_model),
            "model": "" if dino_weight is None else str(dino_weight),
            "config": "" if dino_config is None else str(dino_config),
            "preprocessor": "" if dino_preprocessor is None else str(dino_preprocessor),
        },
        "sea_raft": {
            "requested": sea_raft_model is not None,
            "root": "" if sea_raft_model is None else str(sea_raft_model),
            "model": "" if sea_weight is None else str(sea_weight),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raft-checkpoint",
        type=Path,
        default=Path("/home/admin/raft_large_C_T_SKHT_V2-ff5fadd5.pth"),
    )
    parser.add_argument("--dinov2-model", type=Path, default=Path("/home/admin/dinov2-small"))
    parser.add_argument("--sea-raft-model", type=Path, default=Path("/home/admin/MemorySlices/Tartan-C-T-TSKH-spring540x960-M"))
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = inspect_weights(
        raft_checkpoint=args.raft_checkpoint,
        dinov2_model=args.dinov2_model,
        sea_raft_model=args.sea_raft_model,
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
