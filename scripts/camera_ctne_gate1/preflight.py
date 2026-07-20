#!/usr/bin/env python3
"""Fast file, dependency, and GPU preflight for CTNE Gate 1."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from scripts.camera_ctne_gate1.contracts import normalize_path, write_json
from scripts.camera_flow_probe.models import resolve_hf_model_root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("extract", "train", "evaluate"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, default=16)
    parser.add_argument("--raft-checkpoint", type=Path, default=Path("/home/admin/raft_large_C_T_SKHT_V2-ff5fadd5.pth"))
    parser.add_argument("--dinov2-model", type=Path, default=Path("/home/admin/dinov2-small"))
    parser.add_argument("--required-file", type=Path, action="append", default=[])
    parser.add_argument("--required-dir", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    required_modules = ["numpy"]
    if args.role == "extract":
        required_modules.extend(["cv2", "transformers", "torchvision"])
    else:
        required_modules.extend(["sklearn", "nflows"])
    for module in required_modules:
        try:
            imported = importlib.import_module(module)
            checks[f"import_{module}"] = True
            details[f"{module}_version"] = str(getattr(imported, "__version__", "unknown"))
        except Exception as exc:  # noqa: BLE001
            checks[f"import_{module}"] = False
            details[f"{module}_error"] = f"{type(exc).__name__}: {exc}"
    checks["expected_gpu_count"] = torch.cuda.device_count() == args.expected_gpus
    details["gpu_count"] = torch.cuda.device_count()
    details["gpu_names"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if args.role == "extract":
        checks["raft_checkpoint_exists"] = args.raft_checkpoint.is_file()
        try:
            dino_root = resolve_hf_model_root(args.dinov2_model)
            checks["dinov2_local_model_complete"] = True
            details["dinov2_resolved_root"] = normalize_path(dino_root)
        except Exception as exc:  # noqa: BLE001
            checks["dinov2_local_model_complete"] = False
            details["dinov2_error"] = f"{type(exc).__name__}: {exc}"
    for index, path in enumerate(args.required_file):
        checks[f"required_file_{index}"] = path.is_file()
        details[f"required_file_{index}"] = normalize_path(path)
    for index, path in enumerate(args.required_dir):
        checks[f"required_dir_{index}"] = path.is_dir()
        details[f"required_dir_{index}"] = normalize_path(path)
    result = {
        "gate": "CTNE fast environment preflight",
        "role": args.role,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "details": details,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    write_json(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
