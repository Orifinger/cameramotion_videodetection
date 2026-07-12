#!/usr/bin/env python3
"""Audit files, GPU runtime, storage, and OSS access before an unattended run."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from scripts.camera_binary_vqa.runtime import write_json


REQUIRED_MODULES = ("torch", "torchvision", "transformers", "peft", "qwen_vl_utils")
REQUIRED_PROJECT_FILES = (
    "scripts/camera_binary_vqa/build_data.py",
    "scripts/camera_binary_vqa/runtime.py",
    "scripts/camera_binary_vqa/train.py",
    "scripts/camera_binary_vqa/score.py",
    "scripts/camera_binary_vqa/evaluate.py",
    "scripts/camera_binary_vqa/summarize_gate.py",
    "scripts/camera_binary_vqa/preflight_environment.py",
    "scripts/camera_binary_vqa/distributed_smoke.py",
    "scripts/camera_binary_vqa/monitor_gpu_utilization.py",
    "scripts/camera_binary_vqa/run_unattended.sh",
)


def module_version(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    return {
        "available": True,
        "version": str(getattr(module, "__version__", "unknown")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--oss-uri", required=True)
    parser.add_argument("--expected-gpus", type=int, default=16)
    parser.add_argument("--minimum-free-gb", type=float, default=100.0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "modules": {name: module_version(name) for name in REQUIRED_MODULES},
    }
    checks["required_python_modules"] = all(
        value["available"] for value in details["modules"].values()
    )

    required_paths = {
        "project_root": args.project_root,
        "model_path": args.model_path,
        "model_config": args.model_path / "config.json",
        "manifest_jsonl": args.manifest_jsonl,
        **{
            f"project_file:{relative}": args.project_root / relative
            for relative in REQUIRED_PROJECT_FILES
        },
    }
    path_status = {name: path.exists() for name, path in required_paths.items()}
    details["required_paths"] = {
        name: {"path": str(required_paths[name]), "exists": exists}
        for name, exists in path_status.items()
    }
    checks["required_paths"] = all(path_status.values())

    args.tmp_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(args.tmp_root)
    free_gb = usage.free / (1024**3)
    details["tmp_storage"] = {
        "path": str(args.tmp_root),
        "total_gb": usage.total / (1024**3),
        "used_gb": usage.used / (1024**3),
        "free_gb": free_gb,
        "minimum_free_gb": args.minimum_free_gb,
    }
    checks["tmp_free_space"] = free_gb >= args.minimum_free_gb

    nas_probe = args.persistent_root / f".preflight_write_{uuid.uuid4().hex}.tmp"
    try:
        args.persistent_root.mkdir(parents=True, exist_ok=True)
        nas_probe.write_text("ok\n", encoding="utf-8")
        nas_write_ok = nas_probe.read_text(encoding="utf-8") == "ok\n"
    except OSError as exc:
        nas_write_ok = False
        details["nas_error"] = repr(exc)
    finally:
        try:
            nas_probe.unlink(missing_ok=True)
        except OSError:
            pass
    checks["persistent_nas_writable"] = nas_write_ok
    details["persistent_root"] = str(args.persistent_root)

    torch_info: dict[str, Any] = {}
    try:
        import torch

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        torch_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu_count": gpu_count,
            "gpus": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_gb": torch.cuda.get_device_properties(index).total_memory / (1024**3),
                }
                for index in range(gpu_count)
            ],
        }
        checks["gpu_count"] = gpu_count == args.expected_gpus
        checks["gpu_memory"] = gpu_count > 0 and min(
            item["memory_gb"] for item in torch_info["gpus"]
        ) >= 80.0
    except Exception as exc:
        torch_info = {"error": repr(exc)}
        checks["gpu_count"] = False
        checks["gpu_memory"] = False
    details["torch_runtime"] = torch_info

    nvidia_smi = shutil.which("nvidia-smi")
    details["nvidia_smi"] = {"path": nvidia_smi}
    nvidia_query_ok = False
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            nvidia_query_ok = result.returncode == 0 and len(values) == args.expected_gpus
            details["nvidia_smi"].update(
                {
                    "returncode": result.returncode,
                    "gpu_values": values,
                    "stderr_tail": result.stderr[-1000:],
                }
            )
        except Exception as exc:
            details["nvidia_smi"]["error"] = repr(exc)
    checks["nvidia_smi_utilization_query"] = nvidia_query_ok

    ossutil = shutil.which("ossutil64")
    details["ossutil64"] = {"path": ossutil, "uri": args.oss_uri}
    checks["ossutil64_available"] = ossutil is not None
    oss_access = False
    if ossutil:
        try:
            result = subprocess.run(
                [ossutil, "ls", args.oss_uri],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            oss_access = result.returncode == 0
            details["ossutil64"].update(
                {
                    "returncode": result.returncode,
                    "stdout_tail": result.stdout[-1000:],
                    "stderr_tail": result.stderr[-1000:],
                }
            )
        except Exception as exc:
            details["ossutil64"]["error"] = repr(exc)
    checks["oss_read_access"] = oss_access

    output = {
        "audit": "DataA binary camera VQA unattended environment preflight",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "details": details,
        "environment": {
            "cwd": os.getcwd(),
            "expected_gpus": args.expected_gpus,
        },
    }
    write_json(args.output_json, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if output["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
