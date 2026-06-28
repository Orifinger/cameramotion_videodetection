"""Frozen full VACE execution-plan loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .common import DataAError, read_json
from .schema import CanonicalCaseSpec, build_track_index, normalize_cases
from .path_resolver import PathResolver


SUPPORTED_OPERATIONS = {
    "object_swap",
    "person_appearance_swap",
    "surface_content_edit",
    "object_attribute_edit",
    "surface_attribute_edit",
}

ROUTE_OPERATION_MAP = {
    "vace14b_masktrack_reference_swap": {
        "object_swap",
        "person_appearance_swap",
        "surface_content_edit",
    },
    "vace14b_masktrack_text_edit": {
        "object_attribute_edit",
        "surface_attribute_edit",
    },
}


class MissingFrozenFullExecutionPlan(DataAError):
    """Raised when --execute is requested without a frozen full execution plan."""


@dataclass
class ExecutionPlan:
    source_path: Path
    cases: List[CanonicalCaseSpec]
    raw: Any
    validation: Dict[str, Any]


def discover_full_execution_plan(root: Path) -> Optional[Path]:
    candidates = [
        root / "res" / "dataA_v1" / "plans" / "frozen_full_vace_execution_plan.json",
        root / "res" / "dataA_v1" / "plans" / "vace14b_full_execution_plan.json",
        root / "res" / "dataA_v1" / "plans" / "dataA_v1_vace14b_full_execution_plan.json",
        root / "res" / "dataA_v1" / "registries" / "frozen_full_vace_execution_plan.json",
        root / "res" / "dataA_v1" / "manifests" / "frozen_full_vace_execution_plan.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_execution_plan(
    *,
    execution_plan_path: Path,
    track_bank_path: Path | None,
    path_mapping_path: Path | None,
) -> ExecutionPlan:
    payload = read_json(execution_plan_path)
    mapping = read_json(path_mapping_path) if path_mapping_path else {}
    if track_bank_path and track_bank_path.is_file():
        track_bank = read_json(track_bank_path)
        track_index = build_track_index(track_bank)
        cases = normalize_cases(payload, track_index, PathResolver(mapping), str(execution_plan_path))
    else:
        # Full plans are allowed to carry complete target/donor records inline.
        cases = normalize_cases(payload, {"track_id": {}, "candidate_id": {}}, PathResolver(mapping), str(execution_plan_path))
    validation = validate_execution_cases(cases)
    return ExecutionPlan(source_path=execution_plan_path, cases=cases, raw=payload, validation=validation)


def validate_execution_cases(cases: Iterable[CanonicalCaseSpec], *, donor_reuse_limit: int | None = None) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    seen_case_ids: set[str] = set()
    target_videos: Dict[str, str] = {}
    donor_counts: Dict[str, int] = {}
    operation_counts: Dict[str, int] = {}
    case_count = 0
    for case in cases:
        case_count += 1
        if case.case_id in seen_case_ids:
            errors.append(f"duplicate_case_id:{case.case_id}")
        seen_case_ids.add(case.case_id)
        if case.operation not in SUPPORTED_OPERATIONS:
            errors.append(f"unsupported_operation:{case.case_id}:{case.operation}")
        operation_counts[str(case.operation)] = operation_counts.get(str(case.operation), 0) + 1
        allowed_ops = ROUTE_OPERATION_MAP.get(str(case.generator_route))
        if not allowed_ops:
            errors.append(f"unsupported_generator_route:{case.case_id}:{case.generator_route}")
        elif case.operation not in allowed_ops:
            errors.append(f"route_operation_mismatch:{case.case_id}:{case.generator_route}:{case.operation}")
        if not case.target or not case.target.video_id or not case.target.track_id:
            errors.append(f"missing_target_ref:{case.case_id}")
        elif case.target.video_id in target_videos:
            errors.append(f"target_video_reused:{case.case_id}:{case.target.video_id}:first={target_videos[case.target.video_id]}")
        else:
            target_videos[case.target.video_id] = case.case_id
        if case.generator_route == "vace14b_masktrack_reference_swap" and not case.donor:
            errors.append(f"missing_donor_for_reference_route:{case.case_id}")
        if case.donor:
            donor_key = case.donor.track_id or case.donor.video_id or "<unknown>"
            donor_counts[donor_key] = donor_counts.get(donor_key, 0) + 1
            if case.target.video_id and case.donor.video_id and case.target.video_id == case.donor.video_id:
                errors.append(f"target_and_donor_same_video:{case.case_id}:{case.target.video_id}")
    if donor_reuse_limit is not None:
        for donor, count in donor_counts.items():
            if count > donor_reuse_limit:
                errors.append(f"donor_reuse_limit_exceeded:{donor}:{count}>{donor_reuse_limit}")
    if case_count == 0:
        errors.append("empty_execution_plan")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "case_count": case_count,
        "target_video_count": len(target_videos),
        "donor_reuse_counts": donor_counts,
        "operation_counts": operation_counts,
    }


def require_full_plan_for_execute(path: Path | None) -> Path:
    if path is None or not path.is_file():
        raise MissingFrozenFullExecutionPlan(
            "missing_frozen_full_execution_plan: provide --execution-plan pointing to a frozen full VACE execution plan "
            "with case_id, operation, generator_route, target track/video, donor track/video when required, and sampling_meta"
        )
    return path
