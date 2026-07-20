#!/usr/bin/env python3
"""Fast dependency, file, model, and GPU preflight."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_ctne_gate1.contracts import normalize_path, read_jsonl, write_json


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("train", "evaluate"), required=True)
    parser.add_argument("--feature-index-jsonl", type=Path, required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--expected-gpus", type=int, default=16)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    for module in ("numpy", "sklearn", "torch"):
        try:
            imported = importlib.import_module(module)
            checks[f"import_{module}"] = True
            details[f"{module}_version"] = str(getattr(imported, "__version__", "unknown"))
        except Exception as exc:  # noqa: BLE001
            checks[f"import_{module}"] = False
            details[f"{module}_error"] = f"{type(exc).__name__}: {exc}"
    checks["expected_gpu_count"] = torch.cuda.device_count() == args.expected_gpus
    details["gpu_count"] = torch.cuda.device_count()
    checks["feature_index_exists"] = args.feature_index_jsonl.is_file()
    details["feature_index_jsonl"] = normalize_path(args.feature_index_jsonl)
    rows = read_jsonl(args.feature_index_jsonl) if args.feature_index_jsonl.is_file() else []
    checks["feature_index_nonempty"] = bool(rows)
    missing = [row.get("feature_path") for row in rows if not Path(str(row.get("feature_path", ""))).is_file()]
    checks["all_local_feature_archives_exist"] = not missing
    details["feature_rows"] = len(rows)
    details["missing_feature_archives"] = len(missing)
    details["first_missing_feature_archives"] = missing[:20]
    if args.role == "evaluate":
        checks["model_preprocessor_exists"] = bool(
            args.model_root and (args.model_root / "preprocessor.npz").is_file()
        )
        checks["nine_models_exist"] = bool(
            args.model_root and len(list((args.model_root / "models").glob("seed_*/*/model.pt"))) == 9
        )
        checks["calibration_exists"] = bool(
            args.calibration_dir and (args.calibration_dir / "calibration.json").is_file()
        )
    result = {
        "gate": "Supervised camera interaction fast preflight",
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
