"""Automatic QA report helpers for generated Real/Fake pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from .common import DataAError, write_json


def compare_video_metadata(real_meta: Mapping[str, Any], fake_meta: Mapping[str, Any]) -> Dict[str, Any]:
    keys = ["fps", "frame_count", "height", "width"]
    matches = {key: real_meta.get(key) == fake_meta.get(key) for key in keys}
    return {"matches": matches, "compatible": all(matches.values())}


def mask_outside_difference_score(real: np.ndarray, fake: np.ndarray, mask: np.ndarray) -> float:
    if real.shape != fake.shape:
        raise DataAError(f"real/fake shape mismatch: {real.shape} vs {fake.shape}")
    if mask.ndim == 3 and real.ndim == 4:
        mask_expanded = mask[:, :, :, None] > 0
    else:
        mask_expanded = mask > 0
    outside = ~mask_expanded
    if not np.any(outside):
        return 0.0
    diff = np.abs(real.astype(np.float32) - fake.astype(np.float32))
    return float(diff[outside].mean())


def classify_qa(meta_check: Mapping[str, Any], *, outside_diff_score: float | None = None, thresholds: Mapping[str, float] | None = None) -> str:
    thresholds = thresholds or {"outside_diff_warn": 8.0}
    if not meta_check.get("compatible", False):
        return "rejected_generation_failure"
    if outside_diff_score is not None and outside_diff_score > float(thresholds["outside_diff_warn"]):
        return "needs_manual_review"
    return "accepted"


def write_qa_report(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, dict(payload))


def batch_summary(case_results: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    rejected: Dict[str, int] = {}
    total = 0
    for result in case_results:
        total += 1
        status = str(result.get("status"))
        counts[status] = counts.get(status, 0) + 1
        if status.startswith("rejected_"):
            rejected[status] = rejected.get(status, 0) + 1
    return {
        "full_frozen_execution_plan_count": total,
        "status_counts": counts,
        "accepted_count": counts.get("accepted", 0),
        "rejected_count_by_reason": rejected,
        "uploaded_verified_count": counts.get("uploaded_verified", 0),
    }
