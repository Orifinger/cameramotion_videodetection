"""Append-only run state and resume helpers."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from .common import utc_now_iso, write_json


TERMINAL_STATUSES = {
    "accepted",
    "rejected_generation_failure",
    "rejected_global_drift",
    "rejected_wrong_edit",
    "rejected_low_visibility",
    "needs_manual_review",
    "uploaded_verified",
    "blocked_missing_mask",
    "blocked_volatile_mask",
    "blocked_mapped_but_unverified",
    "blocked_invalid_mask_npz",
    "blocked_clip_selection_failure",
    "blocked_mask_video_mismatch",
    "blocked_donor_reference_failure",
    "blocked_vace_generation_failure",
    "blocked_packaging_failure",
    "blocked_plan_validation_failure",
}


@dataclass
class RunPaths:
    run_root: Path
    coordinator_dir: Path

    @classmethod
    def from_root(cls, tmp_root: Path, run_id: str) -> "RunPaths":
        run_root = tmp_root / run_id
        return cls(run_root=run_root, coordinator_dir=run_root / "coordinator")

    @property
    def run_state_path(self) -> Path:
        return self.coordinator_dir / "run_state.json"

    @property
    def case_status_path(self) -> Path:
        return self.coordinator_dir / "case_status.jsonl"

    @property
    def batch_summary_path(self) -> Path:
        return self.coordinator_dir / "batch_summary.json"

    @property
    def telemetry_jsonl_path(self) -> Path:
        return self.coordinator_dir / "gpu_telemetry.jsonl"

    @property
    def run_state_lock_path(self) -> Path:
        return self.coordinator_dir / "run_state.lock"

    def worker_dir(self, worker_id: int) -> Path:
        return self.run_root / f"worker_{worker_id:02d}"

    def attempt_dir(self, worker_id: int, case_id: str) -> Path:
        return self.worker_dir(worker_id) / "attempts" / case_id


class RunState:
    def __init__(self, paths: RunPaths, *, run_id: str, topology: Mapping[str, Any]) -> None:
        self.paths = paths
        self.run_id = run_id
        self.topology = dict(topology)
        self.paths.coordinator_dir.mkdir(parents=True, exist_ok=True)
        (self.paths.coordinator_dir / "logs").mkdir(parents=True, exist_ok=True)

    def _default_state(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at_utc": utc_now_iso(),
            "updated_at_utc": None,
            "topology": self.topology,
            "cases": {},
        }

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.paths.coordinator_dir.mkdir(parents=True, exist_ok=True)
        with self.paths.run_state_lock_path.open("a+b") as handle:
            if os.name == "nt":
                import msvcrt

                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load_from_status_log_unlocked(self) -> Dict[str, Any]:
        state = self._default_state()
        state["recovered_from_case_status_jsonl"] = True
        if not self.paths.case_status_path.is_file():
            return state
        with self.paths.case_status_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                case_id = event.get("case_id")
                if not case_id:
                    continue
                state.setdefault("cases", {})[str(case_id)] = {
                    "status": event.get("status"),
                    "worker_id": event.get("worker_id"),
                    "updated_at_utc": event.get("timestamp_utc"),
                    "detail": event.get("detail") or {},
                }
        return state

    def _load_unlocked(self) -> Dict[str, Any]:
        if not self.paths.run_state_path.is_file():
            if self.paths.case_status_path.is_file():
                return self._load_from_status_log_unlocked()
            return self._default_state()
        try:
            return json.loads(self.paths.run_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            suffix = utc_now_iso().replace(":", "").replace("+", "_").replace(".", "_")
            backup = self.paths.run_state_path.with_name(f"run_state.invalid.{suffix}.{os.getpid()}.json")
            try:
                os.replace(self.paths.run_state_path, backup)
            except OSError:
                pass
            state = self._load_from_status_log_unlocked()
            state["invalid_run_state_backup"] = str(backup)
            state["invalid_run_state_error"] = str(exc)
            return state

    def load(self) -> Dict[str, Any]:
        with self._locked():
            return self._load_unlocked()

    def save(self, state: Mapping[str, Any]) -> None:
        with self._locked():
            self._save_unlocked(state)

    def _save_unlocked(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        payload["updated_at_utc"] = utc_now_iso()
        write_json(self.paths.run_state_path, payload)

    def append_status(self, case_id: str, status: str, *, worker_id: int | None = None, detail: Mapping[str, Any] | None = None) -> None:
        event = {
            "timestamp_utc": utc_now_iso(),
            "run_id": self.run_id,
            "case_id": case_id,
            "worker_id": worker_id,
            "status": status,
            "detail": dict(detail or {}),
        }
        with self._locked():
            self.paths.case_status_path.parent.mkdir(parents=True, exist_ok=True)
            with self.paths.case_status_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=False) + "\n")
            state = self._load_unlocked()
            state.setdefault("cases", {})[case_id] = {
                "status": status,
                "worker_id": worker_id,
                "updated_at_utc": event["timestamp_utc"],
                "detail": event["detail"],
            }
            self._save_unlocked(state)

    def should_skip_case(self, case_id: str) -> bool:
        state = self.load()
        case_state = state.get("cases", {}).get(case_id) or {}
        status = case_state.get("status")
        if status not in TERMINAL_STATUSES:
            return False
        receipt = case_state.get("detail", {}).get("upload_receipt")
        return status == "uploaded_verified" or bool(receipt)


def summarize_statuses(events: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in events:
        status = str(event.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return counts
