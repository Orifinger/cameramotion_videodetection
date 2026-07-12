#!/usr/bin/env python3
"""Upload each completed epoch checkpoint to OSS while training is still running."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STOP_REQUESTED = False


def request_stop(signum: int, frame: Any) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parent_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--oss-uri", required=True)
    parser.add_argument("--log-jsonl", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def completed_checkpoints(train_dir: Path) -> list[Path]:
    output: list[Path] = []
    candidates = sorted(train_dir.glob("checkpoint-epoch-*"))
    if (train_dir / "final").is_dir():
        candidates.append(train_dir / "final")
    for path in candidates:
        if not path.is_dir() or (path / ".oss_uploaded").exists():
            continue
        if not (path / "adapter_model.safetensors").is_file():
            continue
        if not (path / "camera_binary_vqa_training_state.json").is_file():
            continue
        output.append(path)
    return output


def upload(path: Path, oss_uri: str) -> dict[str, Any]:
    destination = f"{oss_uri.rstrip('/')}/train/{path.name}/"
    started = time.time()
    result = subprocess.run(
        ["ossutil64", "cp", "-r", f"{path}/", destination],
        capture_output=True,
        text=True,
        check=False,
    )
    record = {
        "timestamp_utc": utc_now(),
        "checkpoint": str(path),
        "destination": destination,
        "returncode": result.returncode,
        "elapsed_seconds": time.time() - started,
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    }
    if result.returncode == 0:
        (path / ".oss_uploaded").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return record


def scan_and_upload(args: argparse.Namespace, handle: Any) -> None:
    for checkpoint in completed_checkpoints(args.train_dir):
        record = upload(checkpoint, args.oss_uri)
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.log_jsonl.open("a", encoding="utf-8", newline="\n") as handle:
        while not STOP_REQUESTED and parent_exists(args.parent_pid):
            scan_and_upload(args, handle)
            deadline = time.monotonic() + args.poll_seconds
            while not STOP_REQUESTED and time.monotonic() < deadline:
                time.sleep(min(1.0, deadline - time.monotonic()))
        scan_and_upload(args, handle)


if __name__ == "__main__":
    main()
