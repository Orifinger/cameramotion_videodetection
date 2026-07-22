#!/usr/bin/env python3
"""Fast offline dependency and local-file preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from scripts.forensic_temporal_expert_gate.contracts import normalize_path, write_json
from scripts.forensic_temporal_expert_gate.extract_features import resolve_model_root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov2-model", type=Path, required=True)
    parser.add_argument("--required-file", type=Path, action="append", default=[])
    parser.add_argument("--required-dir", type=Path, action="append", default=[])
    parser.add_argument("--expected-gpus", type=int, default=16)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from transformers import AutoConfig

    args = parse_args(argv)
    root = resolve_model_root(args.dinov2_model)
    config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    checks = {
        "dinov2_config": (root / "config.json").is_file(),
        "dinov2_weights": bool(list(root.glob("*.safetensors")) or list(root.glob("pytorch_model*.bin"))),
        "cuda_available": torch.cuda.is_available(),
        "expected_gpu_count": torch.cuda.device_count() == args.expected_gpus,
        "required_files": all(path.is_file() for path in args.required_file),
        "required_dirs": all(path.is_dir() for path in args.required_dir),
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "dinov2_root": normalize_path(root),
        "dinov2_model_type": str(getattr(config, "model_type", "unknown")),
        "hidden_size": int(getattr(config, "hidden_size", 0)),
        "patch_size": int(getattr(config, "patch_size", 0)),
        "torch": torch.__version__,
        "gpus": torch.cuda.device_count(),
        "required_files": [normalize_path(path) for path in args.required_file],
        "required_dirs": [normalize_path(path) for path in args.required_dir],
    }
    write_json(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
