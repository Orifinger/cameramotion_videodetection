"""Server runtime preflight checks for production VACE runs."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

from .common import DataAError
from .topology import Topology


def _which(name: str) -> str | None:
    return shutil.which(name)


def check_runtime(config: Mapping[str, Any], topology: Topology, *, execution_plan: Path | None, strict: bool = True) -> Dict[str, Any]:
    checks: Dict[str, Any] = {
        "python_version": sys.version,
        "topology": topology.name,
        "checks": {},
        "ok": True,
    }

    def record(name: str, ok: bool, detail: Any) -> None:
        checks["checks"][name] = {"ok": ok, "detail": detail}
        if not ok:
            checks["ok"] = False

    vace_cfg = config.get("vace", {})
    upload_cfg = config.get("upload", {})
    repo_dir = Path(str(vace_cfg.get("repo_dir", "third_party/VACE")))
    checkpoint_dir = Path(str(vace_cfg.get("checkpoint_dir", "")))
    record("vace_vendored_source", (repo_dir / "vace" / "vace_wan_inference.py").is_file(), str(repo_dir))
    record("checkpoint_dir", checkpoint_dir.is_dir(), str(checkpoint_dir))
    record("ffmpeg", _which(str(vace_cfg.get("ffmpeg_bin", "ffmpeg"))) is not None, vace_cfg.get("ffmpeg_bin", "ffmpeg"))
    record("ffprobe", _which(str(vace_cfg.get("ffprobe_bin", "ffprobe"))) is not None, vace_cfg.get("ffprobe_bin", "ffprobe"))
    record("torchrun", _which(str(vace_cfg.get("torchrun_bin", "torchrun"))) is not None, vace_cfg.get("torchrun_bin", "torchrun"))
    record("nvidia_smi", _which("nvidia-smi") is not None, "nvidia-smi")
    if upload_cfg.get("enabled", True):
        record("ossutil64", _which(str(upload_cfg.get("upload_command", "ossutil64"))) is not None, upload_cfg.get("upload_command", "ossutil64"))
    if execution_plan is not None:
        record("execution_plan", execution_plan.is_file(), str(execution_plan))
    tmp_root = Path(str(config.get("run", {}).get("tmp_root", "/tmp/cameramotion_det/dataA_v1/vace14b")))
    try:
        tmp_root.mkdir(parents=True, exist_ok=True)
        probe = tmp_root / ".write_probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        record("tmp_root_writable", True, str(tmp_root))
    except Exception as exc:  # noqa: BLE001
        record("tmp_root_writable", False, f"{tmp_root}: {type(exc).__name__}: {exc}")

    if strict and not checks["ok"]:
        failed = [name for name, item in checks["checks"].items() if not item["ok"]]
        raise DataAError(f"runtime preflight failed before generation: {failed}")
    return checks
