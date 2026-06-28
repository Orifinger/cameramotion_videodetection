"""Append-only run state and resume helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .common import read_json, utc_now_iso, write_json


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

    def load(self) -> Dict[str, Any]:
        if self.paths.run_state_path.is_file():
            return read_json(self.paths.run_state_path)
        return {
            "run_id": self.run_id,
            "created_at_utc": utc_now_iso(),
            "updated_at_utc": None,
            "topology": self.topology,
            "cases": {},
        }

    def save(self, state: Mapping[str, Any]) -> None:
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
        self.paths.case_status_path.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.case_status_path.open("a", encoding="utf-8") as handle:
            import json

            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=False) + "\n")
        state = self.load()
        state.setdefault("cases", {})[case_id] = {
            "status": status,
            "worker_id": worker_id,
            "updated_at_utc": event["timestamp_utc"],
            "detail": event["detail"],
        }
        self.save(state)

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
