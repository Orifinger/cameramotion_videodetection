"""OSS upload inventory, receipt and retry-safe command helpers."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .common import DataAError, utc_now_iso, write_json


READY_MARKER = "READY_TO_UPLOAD"
RECEIPT_NAME = "upload_receipt.json"


def artifact_inventory(root: Path) -> Dict[str, Any]:
    if not root.is_dir():
        raise DataAError(f"artifact root does not exist: {root}")
    files: List[Dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == RECEIPT_NAME:
            continue
        rel = path.relative_to(root).as_posix()
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        files.append({"relative_path": rel, "size": len(data), "sha256": sha})
        digest.update(rel.encode("utf-8"))
        digest.update(str(len(data)).encode("ascii"))
        digest.update(sha.encode("ascii"))
    return {"root": str(root), "file_count": len(files), "files": files, "inventory_digest": digest.hexdigest()}


def mark_ready_to_upload(attempt_dir: Path) -> Path:
    marker = attempt_dir / READY_MARKER
    marker.write_text(utc_now_iso() + "\n", encoding="utf-8")
    return marker


def build_oss_cp_command(upload_command: str, local_dir: Path, oss_dest: str) -> List[str]:
    return [upload_command, "cp", "-r", str(local_dir), oss_dest.rstrip("/")]


def upload_case_bundle(
    *,
    attempt_dir: Path,
    oss_dest: str,
    upload_command: str = "ossutil64",
    run_id: str,
    case_id: str,
    worker_id: int,
    code_commit: str | None = None,
    execute: bool = False,
) -> Dict[str, Any]:
    if not (attempt_dir / READY_MARKER).is_file():
        raise DataAError(f"case is not ready to upload, missing {READY_MARKER}: {attempt_dir}")
    inventory = artifact_inventory(attempt_dir)
    start = utc_now_iso()
    command = build_oss_cp_command(upload_command, attempt_dir, oss_dest)
    return_code = None
    stderr = ""
    stdout = ""
    if execute:
        proc = subprocess.run(command, text=True, capture_output=True, check=False)
        return_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
        if proc.returncode != 0:
            return {
                "status": "upload_failed_retry_queued",
                "case_id": case_id,
                "run_id": run_id,
                "worker_id": worker_id,
                "oss_dest": oss_dest,
                "command": command,
                "return_code": return_code,
                "stdout": stdout,
                "stderr": stderr,
                "inventory": inventory,
                "upload_start_utc": start,
                "upload_end_utc": utc_now_iso(),
            }
    receipt = {
        "status": "uploaded_verified" if execute else "upload_planned",
        "case_id": case_id,
        "run_id": run_id,
        "worker_id": worker_id,
        "oss_dest": oss_dest,
        "command": command,
        "return_code": 0 if execute else None,
        "inventory_digest": inventory["inventory_digest"],
        "inventory": inventory,
        "upload_start_utc": start,
        "upload_end_utc": utc_now_iso(),
        "code_commit": code_commit,
    }
    if execute:
        write_json(attempt_dir / RECEIPT_NAME, receipt)
    return receipt


def should_trigger_upload(*, completed_since_upload: int, minutes_since_upload: float, every_completed_cases: int, every_minutes: int) -> bool:
    return completed_since_upload >= every_completed_cases or minutes_since_upload >= every_minutes
