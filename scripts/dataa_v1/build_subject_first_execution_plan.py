#!/usr/bin/env python3
"""Build a subject-first Data A v1 VACE execution plan from existing tracks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.common import DataAError, read_json, utc_now_iso, write_json
from scripts.dataa_v1.execution_plan import validate_execution_cases
from scripts.dataa_v1.path_resolver import PathResolver
from scripts.dataa_v1.schema import build_track_index, normalize_cases
from scripts.dataa_v1.subject_selection import (
    audit_record,
    evaluate_tracks,
    load_selection_config,
    load_track_bank_records,
    metric_payload,
    select_subjects_by_video,
)


DEFAULT_OUT_CATALOG = Path("res/dataA_v1/catalogs/subject_first_target_catalog.json")
DEFAULT_OUT_AUDIT_JSON = Path("res/dataA_v1/audits/subject_first_selection_audit.json")
DEFAULT_OUT_AUDIT_CSV = Path("res/dataA_v1/audits/subject_first_selection_audit.csv")
DEFAULT_OUT_PLAN = Path("res/dataA_v1/plans/frozen_subject_first_vace_execution_plan.json")
DEFAULT_OUT_COVERAGE_PLAN = Path("res/dataA_v1/plans/frozen_subject_first_vace_execution_plan_coverage.json")
DEFAULT_OUT_RESERVE_PLAN = Path("res/dataA_v1/plans/frozen_subject_first_vace_execution_plan_reserve.json")
DEFAULT_OUT_VACE14B_PLAN = Path("res/dataA_v1/plans/frozen_subject_first_vace14b_execution_plan.json")
DEFAULT_OUT_VACE13B_PLAN = Path("res/dataA_v1/plans/frozen_subject_first_vace13b_execution_plan.json")

REFERENCE_ROUTE = "vace14b_masktrack_reference_swap"
TEXT_ROUTE = "vace14b_masktrack_text_edit"
SURFACE_LABELS = {"display_screen", "sign_or_poster", "framed_art", "paper_book_map", "screen", "poster", "sign", "billboard", "book", "map", "paper"}
PERSON_LABELS = {"human", "person", "people", "man", "woman", "child", "face", "body"}


def _default_num_workers() -> int:
    return max(1, min(96, (os.cpu_count() or 2) // 2))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_project_path(path: Path | str | None) -> Path | None:
    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else _repo_root() / value


def _track_payload(record: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "track_id",
        "candidate_id",
        "video_id",
        "source_video_id",
        "video_path",
        "source_video_path",
        "mask_tube_path",
        "mask_path",
        "mask_npz_path",
        "mask_tube_npz_path",
        "sam3_mask_path",
        "sam3_mask_npz_path",
        "track_mask_path",
        "track_mask_npz_path",
        "npz_path",
        "candidate_class",
        "canonical_concept",
        "display_phrase",
        "region_family",
        "content_domain",
        "style_domain",
        "bbox_tube_xywh",
        "bbox_tube",
        "bboxes",
        "track_quality_score",
        "source_fps",
    )
    return {key: record[key] for key in keys if key in record and record[key] not in (None, "")}


def _donor_payload(donor: Any) -> Optional[Dict[str, Any]]:
    if donor is None:
        return None
    if hasattr(donor, "record"):
        return _track_payload(donor.record)
    return {
        "track_id": donor.track_id,
        "candidate_id": donor.candidate_id,
        "video_id": donor.video_id,
        "video_path": donor.video_path,
        "mask_tube_path": donor.mask_tube_path,
        "candidate_class": donor.candidate_class,
        "canonical_concept": donor.canonical_concept,
        "display_phrase": donor.display_phrase,
        "region_family": donor.region_family,
        "content_domain": donor.content_domain,
        "style_domain": donor.style_domain,
        "bbox_tube_xywh": donor.bbox_tube_xywh,
    }


def _selection_meta(
    selected: Any,
    config: Mapping[str, Any],
    *,
    quality_tier: str = "clean",
    risk_tags: Sequence[str] | None = None,
    donor_repair: Mapping[str, Any] | None = None,
    operation_gate: Mapping[str, Any] | None = None,
    target_repair: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    meta = {
        "selection_role": selected.selection_role,
        "selection_mode": selected.selection_mode,
        "primary_probability": float(config.get("primary_probability", 0.85)),
        "secondary_pool_size": int(selected.secondary_pool_size),
        "random_seed": int(config.get("random_seed", 20260629)),
        "selection_random_value": float(selected.selection_random_value or 0.0),
        "subject_score": selected.subject_score,
        "quality_tier": quality_tier,
    }
    if risk_tags:
        meta["risk_tags"] = sorted(set(str(tag) for tag in risk_tags))
    if donor_repair:
        meta["donor_repair"] = dict(donor_repair)
    if operation_gate:
        meta["operation_gate"] = dict(operation_gate)
    if target_repair:
        meta["target_repair"] = dict(target_repair)
    return meta


def _quantiles(values: list[float]) -> Dict[str, float]:
    if not values:
        return {"p20": 0.0, "median": 0.0, "p80": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"p20": float(np.quantile(arr, 0.20)), "median": float(np.median(arr)), "p80": float(np.quantile(arr, 0.80))}


def _reason_category(reason: str) -> str:
    if "mask npz does not exist:" in reason:
        return "mask_npz_does_not_exist"
    if "; path=" in reason:
        return reason.split("; path=", 1)[0]
    if " -> " in reason:
        return reason.split("=", 1)[0] + " -> " + reason.split(" -> ", 1)[1].split(":", 1)[0]
    return reason


def _summary(audit_records: list[Dict[str, Any]], plan_cases: list[Dict[str, Any]]) -> Dict[str, Any]:
    selected = [row for row in audit_records if row["selection_status"] == "selected"]
    videos = {row["video_id"] for row in audit_records}
    videos_with_primary = {row["video_id"] for row in audit_records if row["selection_status"] in {"selected", "primary_subject"}}
    rejection_tags = Counter(tag for row in audit_records for tag in (row.get("rejection_tags") or []))
    rejection_reasons = Counter(
        _reason_category(str(reason))
        for row in audit_records
        for reason in (row.get("rejection_reasons") or [])
    )
    return {
        "videos_total": len(videos),
        "videos_with_primary": len(videos_with_primary),
        "videos_without_eligible_target": len(videos - videos_with_primary),
        "selected_primary_count": sum(1 for row in selected if row.get("selection_role") in {"primary_subject", "fallback_primary"}),
        "selected_secondary_count": sum(1 for row in selected if row.get("selection_role") == "eligible_secondary"),
        "ineligible_count": sum(1 for row in audit_records if row["selection_status"] == "ineligible_small_or_weak"),
        "selection_role_counts": dict(Counter(str(row.get("selection_role") or "<none>") for row in selected)),
        "candidate_class_counts_before_selection": dict(Counter(str(row.get("candidate_class") or "<missing>") for row in audit_records)),
        "candidate_class_counts_after_selection": dict(Counter(str(row.get("candidate_class") or "<missing>") for row in selected)),
        "rejection_tag_counts": dict(rejection_tags.most_common(20)),
        "rejection_reason_counts": dict(rejection_reasons.most_common(20)),
        "area_ratio_quantiles_before_selection": _quantiles([float(row["median_mask_area_ratio"]) for row in audit_records]),
        "area_ratio_quantiles_after_selection": _quantiles([float(row["median_mask_area_ratio"]) for row in selected]),
        "operation_counts": dict(Counter(str(case.get("operation") or "<missing>") for case in plan_cases)),
        "route_counts": dict(Counter(str(case.get("generator_route") or "<missing>") for case in plan_cases)),
    }


def _write_audit_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video_id",
        "track_id",
        "candidate_class",
        "canonical_concept",
        "selection_status",
        "selection_role",
        "subject_score",
        "median_mask_area_ratio",
        "p20_mask_area_ratio",
        "median_bbox_short_side_720",
        "temporal_visibility_seconds",
        "frame_centrality",
        "track_quality",
        "secondary_pool_size",
        "rejection_tags",
        "rejection_reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = dict(row)
            clean["rejection_tags"] = ";".join(row.get("rejection_tags") or [])
            clean["rejection_reasons"] = ";".join(row.get("rejection_reasons") or [])
            writer.writerow({field: clean.get(field) for field in fields})


def _track_text(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    return " ".join(str(record.get(key) or "").strip().lower() for key in keys if record.get(key))


def _stable_seed(*parts: Any) -> int:
    text = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def _stable_unit(*parts: Any) -> float:
    return _stable_seed(*parts) / float(0xFFFFFFFFFFFF)


def _track_label_text(track: Any) -> str:
    if hasattr(track, "record"):
        raw = track.record
    elif isinstance(track, Mapping):
        raw = track
    elif isinstance(getattr(track, "raw", None), Mapping):
        raw = track.raw
    else:
        raw = {
            "candidate_class": getattr(track, "candidate_class", None),
            "canonical_concept": getattr(track, "canonical_concept", None),
            "display_phrase": getattr(track, "display_phrase", None),
            "region_family": getattr(track, "region_family", None),
            "content_domain": getattr(track, "content_domain", None),
            "style_domain": getattr(track, "style_domain", None),
        }
    keys = ("candidate_class", "canonical_concept", "display_phrase", "region_family", "content_domain", "style_domain")
    return _track_text(raw, keys)


def _has_any_label(track: Any, labels: set[str]) -> bool:
    text = _track_label_text(track)
    return any(label in text for label in labels)


def _is_person_track(track: Any) -> bool:
    return _has_any_label(track, PERSON_LABELS)


def _is_surface_track(track: Any) -> bool:
    return _has_any_label(track, SURFACE_LABELS)


def _is_object_track(track: Any) -> bool:
    return not _is_person_track(track) and not _is_surface_track(track)


def _operation_compatible(operation: str | None, track: Any) -> bool:
    if operation == "person_appearance_swap":
        return _is_person_track(track)
    if operation in {"surface_content_edit", "surface_attribute_edit"}:
        return _is_surface_track(track)
    if operation in {"object_swap", "object_attribute_edit"}:
        return _is_object_track(track)
    return True


def _operation_for_track(track: Any, *, prefer_reference_route: bool) -> tuple[str, str] | None:
    if _is_person_track(track):
        return "person_appearance_swap", REFERENCE_ROUTE
    if _is_surface_track(track):
        if prefer_reference_route:
            return "surface_content_edit", REFERENCE_ROUTE
        return "surface_attribute_edit", TEXT_ROUTE
    if prefer_reference_route:
        return "object_swap", REFERENCE_ROUTE
    return "object_attribute_edit", TEXT_ROUTE


def _operation_gate(operation: str | None, config: Mapping[str, Any]) -> Mapping[str, Any]:
    gates = config.get("operation_gates") or {}
    return gates.get(str(operation), config.get("universal_gate") or {})


def _operation_gate_report(track: Any, operation: str | None, config: Mapping[str, Any]) -> Dict[str, Any]:
    gate = _operation_gate(operation, config)
    metrics = track.metrics
    if metrics is None:
        return {"operation": operation, "pass": False, "thresholds": dict(gate), "failures": ["missing_metrics"]}
    checks = {
        "median_mask_area_ratio": (
            float(metrics.median_mask_area_ratio),
            float(gate.get("min_median_mask_area_ratio", 0.0)),
        ),
        "p20_mask_area_ratio": (
            float(metrics.p20_mask_area_ratio),
            float(gate.get("min_p20_mask_area_ratio", 0.0)),
        ),
        "median_bbox_short_side_720": (
            float(metrics.median_bbox_short_side_720),
            float(gate.get("min_median_bbox_short_side_720", 0.0)),
        ),
    }
    failures = [key for key, (value, threshold) in checks.items() if value < threshold]
    return {
        "operation": operation,
        "pass": not failures,
        "thresholds": dict(gate),
        "values": {key: value for key, (value, _threshold) in checks.items()},
        "failures": failures,
    }


def _largest_track_key(track: Any) -> tuple[float, float, float, float, str]:
    metrics = track.metrics
    if metrics is None:
        return (0.0, 0.0, 0.0, 0.0, str(track.track_id))
    return (
        float(metrics.median_mask_area_ratio),
        float(metrics.p20_mask_area_ratio),
        float(metrics.median_bbox_short_side_720),
        float(track.subject_score),
        str(track.track_id),
    )


def _best_relaxed_target(candidates: Sequence[Any]) -> Optional[Any]:
    viable = [item for item in candidates if item.metrics is not None]
    if not viable:
        return None
    return max(
        viable,
        key=lambda item: (
            item.subject_score,
            item.metrics.median_bbox_short_side_720 if item.metrics else 0.0,
            item.track_id,
        ),
    )


def _build_relaxed_targets(selections: Mapping[str, Any]) -> Dict[str, Any]:
    relaxed: Dict[str, Any] = {}
    for video_id, choice in selections.items():
        if choice.selected is not None:
            relaxed[video_id] = choice.selected
            continue
        best = _best_relaxed_target(choice.candidates)
        if best is not None:
            relaxed[video_id] = best
    return relaxed


def _freeze_mask_policy(case_id: str, operation: str | None, track: Any, config: Mapping[str, Any]) -> Dict[str, Any]:
    seed = _stable_seed("mask_policy", config.get("random_seed", 20260629), case_id, operation, track.track_id)
    unit = _stable_unit("mask_policy_variant", seed)
    person = operation == "person_appearance_swap" or _is_person_track(track)
    if person:
        if unit < 0.60:
            variant = "sam3_shape"
        elif unit < 0.85:
            variant = "dilated"
        elif unit < 0.95:
            variant = "closing"
        else:
            variant = "erode_then_dilate"
    elif unit < 0.70:
        variant = "sam3_shape"
    elif unit < 0.90:
        variant = "dilated"
    else:
        variant = "expanded_bbox"
    metrics = track.metrics
    area = 0.0 if metrics is None else float(metrics.median_mask_area_ratio)
    if area < 0.02:
        radius = 24
    elif area < 0.05:
        radius = 16
    else:
        radius = 8
    return {
        "schema_version": "dataA_v1_mask_policy_v1",
        "variant_type": variant,
        "seed": int(seed),
        "dilation_radius_px": int(radius),
        "erosion_radius_px": 2 if person else 0,
        "closing_radius_px": 4 if person else 1,
        "bbox_expand_ratio": 1.15,
        "person_bbox_disabled": bool(person),
        "base_dilation_radius_px": 2,
        "selection_unit_interval_value": float(unit),
        "trigger_reason": "person_shape_preserving_policy" if person else "plan_time_deterministic_policy",
    }


def _artifact_policy(case_id: str, operation: str | None, track: Any, config: Mapping[str, Any]) -> Dict[str, Any] | None:
    if operation != "surface_content_edit" or not _is_surface_track(track):
        return None
    unit = _stable_unit("surface_artifact", config.get("random_seed", 20260629), case_id, track.track_id)
    if unit >= 0.30:
        return None
    return {
        "schema_version": "dataA_v1_artifact_policy_v1",
        "artifact_type": "surface_text_degradation",
        "seed": int(_stable_seed("surface_artifact", case_id, track.track_id)),
        "selection_unit_interval_value": float(unit),
        "probability": 0.30,
        "allowed_fields_only": True,
    }


def _vace_model_plan(operation: str | None, quality_tier: str) -> Dict[str, Any]:
    if operation in {"surface_content_edit", "surface_attribute_edit"} or quality_tier.startswith("coverage_only"):
        return {
            "model_name": "vace-1.3B",
            "size": "480p",
            "profile": "production_480",
            "reason": "surface_or_coverage_simple_edit",
            "prompt_extension": "plain",
            "offload_model": False,
            "t5_cpu": False,
        }
    return {
        "model_name": "vace-14B",
        "size": "720p",
        "profile": "production_720",
        "reason": "person_or_object_primary_edit",
        "prompt_extension": "plain",
        "offload_model": False,
        "t5_cpu": False,
    }


def _donor_match_score(target: Any, donor: Any) -> tuple[int, str]:
    target_raw = target.record
    donor_raw = donor.record
    for key, score, level in (
        ("canonical_concept", 100, "same_canonical_concept"),
        ("candidate_class", 80, "same_candidate_class"),
        ("region_family", 60, "same_region_family"),
        ("content_domain", 40, "same_content_domain"),
        ("style_domain", 30, "same_style_domain"),
    ):
        target_value = str(target_raw.get(key) or "").strip().lower()
        donor_value = str(donor_raw.get(key) or "").strip().lower()
        if target_value and donor_value and target_value == donor_value:
            return score, level
    target_text = _track_text(target_raw, ("canonical_concept", "candidate_class", "region_family", "content_domain", "style_domain"))
    donor_text = _track_text(donor_raw, ("canonical_concept", "candidate_class", "region_family", "content_domain", "style_domain"))
    if target_text and donor_text and set(target_text.split()).intersection(donor_text.split()):
        return 15, "similar_text_overlap"
    return 1, "best_cross_video"


def _repair_donor(target: Any, donor_pool: Sequence[Any]) -> tuple[Optional[Any], Optional[Dict[str, Any]]]:
    candidates = [item for item in donor_pool if item.metrics is not None and item.video_id != target.video_id and item.track_id != target.track_id]
    if not candidates:
        return None, None
    scored = []
    for donor in candidates:
        score, level = _donor_match_score(target, donor)
        scored.append((score, donor.subject_score, donor.track_id, level, donor))
    score, _subject_score, _track_id, level, donor = max(scored, key=lambda item: (item[0], item[1], item[2]))
    return donor, {
        "enabled": True,
        "match_level": level,
        "score": int(score),
        "donor_track_id": donor.track_id,
        "donor_video_id": donor.video_id,
    }


def _select_case_target(
    case: Any,
    selection: Any,
    config: Mapping[str, Any],
    donor_pool: Sequence[Any],
) -> tuple[Optional[Any], Optional[str], Optional[str], Any, str, list[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    candidates = [item for item in selection.candidates if item.metrics is not None]
    compatible = [item for item in candidates if _operation_compatible(case.operation, item)]
    passing = [item for item in compatible if _operation_gate_report(item, case.operation, config)["pass"]]
    preferred = selection.selected if selection.selected is not None else None
    donor = case.donor
    donor_repair = None
    operation_repair = None
    target_repair = None
    risk_tags: list[str] = []
    quality_tier = "clean"

    if preferred in passing:
        selected = preferred
    elif passing:
        selected = max(passing, key=_largest_track_key)
        target_repair = {
            "enabled": True,
            "reason": "preferred_target_failed_operation_gate_or_compatibility",
            "original_track_id": None if preferred is None else preferred.track_id,
            "repaired_track_id": selected.track_id,
            "operation": case.operation,
        }
    elif compatible:
        selected = max(compatible, key=_largest_track_key)
        quality_tier = "area_gate_fallback_largest"
        risk_tags = sorted(set(selected.rejection_tags + ["area_gate_fallback_largest"]))
        target_repair = {
            "enabled": True,
            "reason": "no_operation_compatible_track_passed_area_gate",
            "original_track_id": None if preferred is None else preferred.track_id,
            "repaired_track_id": selected.track_id,
            "operation": case.operation,
        }
    else:
        selected = _best_relaxed_target(candidates)
        if selected is None:
            return None, None, None, None, quality_tier, risk_tags, None, None, None
        repaired = _operation_for_track(selected, prefer_reference_route=case.generator_route == REFERENCE_ROUTE)
        if repaired is None:
            return None, None, None, None, quality_tier, risk_tags, None, None, None
        repaired_operation, repaired_route = repaired
        quality_tier = "operation_repair"
        risk_tags = sorted(set(selected.rejection_tags + ["operation_repair"]))
        operation_repair = {
            "enabled": True,
            "reason": "no_target_compatible_with_original_operation",
            "original_operation": case.operation,
            "original_generator_route": case.generator_route,
            "repaired_operation": repaired_operation,
            "repaired_generator_route": repaired_route,
            "repaired_track_id": selected.track_id,
        }
        if repaired_route == REFERENCE_ROUTE:
            donor, donor_repair = _repair_donor(selected, donor_pool)
            if donor is None:
                return None, None, None, None, quality_tier, risk_tags, None, operation_repair, None
            donor_repair["reason"] = "operation_repair_requires_reference_donor"
        else:
            donor = None
        return selected, repaired_operation, repaired_route, donor, quality_tier, risk_tags, donor_repair, operation_repair, target_repair

    if case.generator_route == REFERENCE_ROUTE:
        if donor is None or (donor.video_id and donor.video_id == selected.video_id):
            donor, donor_repair = _repair_donor(selected, donor_pool)
            if donor is None:
                return None, None, None, None, quality_tier, risk_tags, None, operation_repair, target_repair
            donor_repair["reason"] = "missing_donor" if case.donor is None else "same_video_donor"
    return selected, case.operation, case.generator_route, donor, quality_tier, risk_tags, donor_repair, operation_repair, target_repair


def _case_from_template(
    case: Any,
    selected: Any,
    config: Mapping[str, Any],
    *,
    operation: str | None = None,
    generator_route: str | None = None,
    donor: Any = None,
    quality_tier: str = "clean",
    risk_tags: Sequence[str] | None = None,
    donor_repair: Mapping[str, Any] | None = None,
    operation_repair: Mapping[str, Any] | None = None,
    target_repair: Mapping[str, Any] | None = None,
) -> Optional[Dict[str, Any]]:
    operation = operation or case.operation
    generator_route = generator_route or case.generator_route
    donor = case.donor if donor is None else donor
    if generator_route != REFERENCE_ROUTE:
        donor = None
    if generator_route == REFERENCE_ROUTE and donor is None:
        return None
    if donor and donor.video_id and donor.video_id == selected.video_id:
        return None
    sampling_meta = dict(case.sampling_meta or {})
    gate = _operation_gate_report(selected, operation, config)
    sampling_meta["target_selection"] = _selection_meta(
        selected,
        config,
        quality_tier=quality_tier,
        risk_tags=risk_tags,
        donor_repair=donor_repair,
        operation_gate=gate,
        target_repair=target_repair,
    )
    sampling_meta["target_saliency"] = metric_payload(selected)
    sampling_meta["mask_policy"] = _freeze_mask_policy(case.case_id, operation, selected, config)
    artifact = _artifact_policy(case.case_id, operation, selected, config)
    if artifact:
        sampling_meta["artifact_policy"] = artifact
    if operation_repair:
        sampling_meta["operation_repair"] = dict(operation_repair)
    sampling_meta["vace_model_plan"] = _vace_model_plan(operation, quality_tier)
    sampling_meta["subject_first_source"] = "scripts/dataa_v1/build_subject_first_execution_plan.py"
    sampling_meta["frozen"] = True
    return {
        "case_id": case.case_id,
        "operation": operation,
        "generator_route": generator_route,
        "target": _track_payload(selected.record),
        "donor": _donor_payload(donor),
        "sampling_meta": sampling_meta,
    }


def _case_from_template_coverage(
    case: Any,
    selected: Any,
    config: Mapping[str, Any],
    *,
    quality_tier: str,
    risk_tags: Sequence[str],
    donor_pool: Sequence[Any],
) -> Optional[Dict[str, Any]]:
    operation = case.operation
    generator_route = case.generator_route
    donor = case.donor
    donor_repair = None
    if generator_route == REFERENCE_ROUTE:
        needs_repair = donor is None or (donor.video_id and donor.video_id == selected.video_id)
        if needs_repair:
            repaired_donor, donor_repair = _repair_donor(selected, donor_pool)
            if repaired_donor is None:
                return None
            donor = repaired_donor
            donor_repair["reason"] = "missing_donor" if case.donor is None else "same_video_donor"

    sampling_meta = dict(case.sampling_meta or {})
    gate = _operation_gate_report(selected, operation, config)
    sampling_meta["target_selection"] = _selection_meta(
        selected,
        config,
        quality_tier=quality_tier,
        risk_tags=risk_tags,
        donor_repair=donor_repair,
        operation_gate=gate,
    )
    sampling_meta["target_saliency"] = metric_payload(selected)
    sampling_meta["mask_policy"] = _freeze_mask_policy(case.case_id, operation, selected, config)
    artifact = _artifact_policy(case.case_id, operation, selected, config)
    if artifact:
        sampling_meta["artifact_policy"] = artifact
    sampling_meta["vace_model_plan"] = _vace_model_plan(operation, quality_tier)
    sampling_meta["subject_first_source"] = "scripts/dataa_v1/build_subject_first_execution_plan.py"
    sampling_meta["frozen"] = True
    sampling_meta["coverage_plan"] = quality_tier != "clean"
    return {
        "case_id": case.case_id,
        "operation": operation,
        "generator_route": generator_route,
        "target": _track_payload(selected.record),
        "donor": _donor_payload(donor),
        "sampling_meta": sampling_meta,
    }


def _minimal_case(
    index: int,
    selected: Any,
    config: Mapping[str, Any],
    *,
    case_id_prefix: str = "dataA_v1_subject_first",
    quality_tier: str = "clean",
    risk_tags: Sequence[str] | None = None,
) -> Optional[Dict[str, Any]]:
    repaired = _operation_for_track(selected, prefer_reference_route=False)
    if repaired is None:
        return None
    operation, generator_route = repaired
    if operation == "person_appearance_swap":
        return None
    case_id = f"{case_id_prefix}_{index:05d}"
    return {
        "case_id": case_id,
        "operation": operation,
        "generator_route": generator_route,
        "target": _track_payload(selected.record),
        "donor": None,
        "sampling_meta": {
            "target_selection": _selection_meta(
                selected,
                config,
                quality_tier=quality_tier,
                risk_tags=risk_tags,
                operation_gate=_operation_gate_report(selected, operation, config),
            ),
            "target_saliency": metric_payload(selected),
            "mask_policy": _freeze_mask_policy(case_id, operation, selected, config),
            "vace_model_plan": _vace_model_plan(operation, quality_tier),
            "subject_first_source": "scripts/dataa_v1/build_subject_first_execution_plan.py",
            "frozen": True,
            "coverage_plan": quality_tier != "clean",
        },
    }


def _plan_payload(
    *,
    schema_version: str,
    track_bank: Path,
    base_plan: Path | None,
    path_mapping: Path | None,
    selection_summary: Mapping[str, Any],
    validation: Mapping[str, Any],
    cases: list[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": schema_version,
        "generated_at_utc": utc_now_iso(),
        "track_bank": str(track_bank),
        "base_plan": str(base_plan) if base_plan else None,
        "path_mapping": str(path_mapping) if path_mapping else None,
        "case_count": len(cases),
        "selection_summary": dict(selection_summary),
        "validation": dict(validation),
        "cases": cases,
    }


def _case_model_name(case: Mapping[str, Any]) -> str:
    meta = case.get("sampling_meta") or {}
    model = meta.get("vace_model_plan") or {}
    return str(model.get("model_name") or "vace-14B")


def _validate_execution_cases_allow_empty(cases: Sequence[Any]) -> Dict[str, Any]:
    validation = validate_execution_cases(cases)
    if validation["case_count"] == 0 and validation["errors"] == ["empty_execution_plan"]:
        validation = dict(validation)
        validation["valid"] = True
        validation["warnings"] = [*validation.get("warnings", []), "empty_model_split_plan"]
        validation["errors"] = []
    return validation


def build_subject_first_plan(
    *,
    track_bank: Path,
    selection_config: Path | None,
    base_plan: Path | None,
    path_mapping: Path | None = None,
    out_catalog: Path,
    out_audit_json: Path,
    out_audit_csv: Path,
    out_plan: Path,
    out_coverage_plan: Path = DEFAULT_OUT_COVERAGE_PLAN,
    out_reserve_plan: Path = DEFAULT_OUT_RESERVE_PLAN,
    out_vace14b_plan: Path | None = None,
    out_vace13b_plan: Path | None = None,
    ffprobe_bin: str = "ffprobe",
    seed: int | None = None,
    dry_run: bool = False,
    progress_every: int = 0,
    num_workers: int = 1,
    config_overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = dict(config_overrides or {})
    if seed is not None:
        overrides["random_seed"] = int(seed)
    config = load_selection_config(selection_config, overrides=overrides)
    out_vace14b_plan = out_vace14b_plan or out_plan.with_name(DEFAULT_OUT_VACE14B_PLAN.name)
    out_vace13b_plan = out_vace13b_plan or out_plan.with_name(DEFAULT_OUT_VACE13B_PLAN.name)
    records = load_track_bank_records(track_bank)
    resolver = PathResolver(read_json(path_mapping) if path_mapping else {})
    evaluated = evaluate_tracks(
        records,
        config,
        ffprobe_bin=ffprobe_bin,
        path_resolver=resolver,
        progress_every=progress_every,
        num_workers=num_workers,
    )
    selections = select_subjects_by_video(evaluated, config)
    selected_by_video = {video_id: choice.selected for video_id, choice in selections.items() if choice.selected is not None}
    coverage_by_video = _build_relaxed_targets(selections)
    donor_pool = [item for item in evaluated if item.metrics is not None]

    audit_rows = [audit_record(item) for item in evaluated]
    plan_cases: list[Dict[str, Any]] = []
    coverage_cases: list[Dict[str, Any]] = []
    coverage_skipped_templates: list[Dict[str, Any]] = []
    skipped_templates: list[Dict[str, Any]] = []
    if base_plan is not None:
        track_index = build_track_index({"tracks": records})
        base_cases = normalize_cases(read_json(base_plan), track_index, resolver, str(base_plan))
        used_videos: set[str] = set()
        used_coverage_videos: set[str] = set()
        for case in base_cases:
            video_id = case.target.video_id
            video_selection = selections.get(str(video_id)) if video_id else None
            selected = selected_by_video.get(str(video_id)) if video_id else None
            coverage_selected = coverage_by_video.get(str(video_id)) if video_id else None
            clean_case_added = False
            if video_selection is None:
                skipped_templates.append({"case_id": case.case_id, "reason": "no_subject_first_target_for_video", "video_id": video_id})
            else:
                (
                    selected,
                    operation,
                    generator_route,
                    donor,
                    quality_tier,
                    risk_tags,
                    donor_repair,
                    operation_repair,
                    target_repair,
                ) = _select_case_target(case, video_selection, config, donor_pool)
                if selected is not None and selected.video_id in used_videos:
                    skipped_templates.append({"case_id": case.case_id, "reason": "target_video_already_used", "video_id": selected.video_id})
                    selected = None
                new_case = None if selected is None else _case_from_template(
                    case,
                    selected,
                    config,
                    operation=operation,
                    generator_route=generator_route,
                    donor=donor,
                    quality_tier=quality_tier,
                    risk_tags=risk_tags,
                    donor_repair=donor_repair,
                    operation_repair=operation_repair,
                    target_repair=target_repair,
                )
                if new_case is not None:
                    used_videos.add(selected.video_id)
                    used_coverage_videos.add(selected.video_id)
                    plan_cases.append(new_case)
                    coverage_cases.append(new_case)
                    clean_case_added = True
                else:
                    skipped_templates.append({"case_id": case.case_id, "reason": "operation_target_or_donor_repair_failed", "video_id": video_id})

            if clean_case_added:
                continue
            if coverage_selected is None:
                coverage_skipped_templates.append({"case_id": case.case_id, "reason": "no_readable_target_for_coverage", "video_id": video_id})
                continue
            if coverage_selected.video_id in used_coverage_videos:
                coverage_skipped_templates.append({"case_id": case.case_id, "reason": "coverage_target_video_already_used", "video_id": coverage_selected.video_id})
                continue
            if video_selection is None:
                coverage_skipped_templates.append({"case_id": case.case_id, "reason": "missing_video_selection_for_coverage", "video_id": video_id})
                continue
            (
                coverage_selected,
                operation,
                generator_route,
                donor,
                quality_tier,
                risk_tags,
                donor_repair,
                operation_repair,
                target_repair,
            ) = _select_case_target(case, video_selection, config, donor_pool)
            if coverage_selected is None:
                coverage_skipped_templates.append({"case_id": case.case_id, "reason": "coverage_operation_target_or_donor_repair_failed", "video_id": video_id})
                continue
            if quality_tier == "clean":
                quality_tier = "clean_donor_repair" if donor_repair else "coverage_clean_repair"
            if selected is None:
                risk_tags = sorted(set([*risk_tags, "relaxed_target_gate"]))
                if quality_tier == "clean_donor_repair":
                    quality_tier = "relaxed_rescue"
            coverage_case = _case_from_template(
                case,
                coverage_selected,
                config,
                operation=operation,
                generator_route=generator_route,
                donor=donor,
                quality_tier=quality_tier,
                risk_tags=risk_tags,
                donor_repair=donor_repair,
                operation_repair=operation_repair,
                target_repair=target_repair,
            )
            if coverage_case is None:
                coverage_skipped_templates.append({"case_id": case.case_id, "reason": "coverage_donor_repair_failed", "video_id": coverage_selected.video_id})
                continue
            used_coverage_videos.add(coverage_selected.video_id)
            coverage_cases.append(coverage_case)

        extra_index = 0
        for video_id, selected in sorted(coverage_by_video.items()):
            if video_id in used_coverage_videos:
                continue
            quality_tier = "coverage_only_clean" if video_id in selected_by_video else "coverage_only_relaxed"
            risk_tags = list(selected.rejection_tags)
            if video_id not in selected_by_video:
                risk_tags.append("relaxed_target_gate")
            minimal = _minimal_case(
                extra_index,
                selected,
                config,
                case_id_prefix="dataA_v1_subject_first_coverage",
                quality_tier=quality_tier,
                risk_tags=risk_tags,
            )
            if minimal is None:
                coverage_skipped_templates.append({"case_id": f"dataA_v1_subject_first_coverage_{extra_index:05d}", "reason": "coverage_only_no_text_route_compatible_operation", "video_id": video_id})
                continue
            coverage_cases.append(minimal)
            used_coverage_videos.add(video_id)
            extra_index += 1
    else:
        for index, selected in enumerate(sorted(selected_by_video.values(), key=lambda item: item.video_id)):
            case = _minimal_case(index, selected, config)
            if case is not None:
                plan_cases.append(case)
                coverage_cases.append(case)
        clean_videos = set(selected_by_video)
        extra_index = 0
        for video_id, selected in sorted(coverage_by_video.items()):
            if video_id in clean_videos:
                continue
            minimal = _minimal_case(
                extra_index,
                selected,
                config,
                case_id_prefix="dataA_v1_subject_first_coverage",
                quality_tier="coverage_only_relaxed",
                risk_tags=[*selected.rejection_tags, "relaxed_target_gate"],
            )
            if minimal is not None:
                coverage_cases.append(minimal)
            extra_index += 1

    normalized_plan_cases = normalize_cases({"cases": plan_cases}, build_track_index({"tracks": records}), resolver, str(out_plan))
    normalized_coverage_cases = normalize_cases({"cases": coverage_cases}, build_track_index({"tracks": records}), resolver, str(out_coverage_plan))
    clean_case_ids = {str(case.get("case_id")) for case in plan_cases}
    reserve_cases = [case for case in coverage_cases if str(case.get("case_id")) not in clean_case_ids]
    normalized_reserve_cases = normalize_cases({"cases": reserve_cases}, build_track_index({"tracks": records}), resolver, str(out_reserve_plan))
    vace14b_cases = [case for case in coverage_cases if _case_model_name(case) == "vace-14B"]
    vace13b_cases = [case for case in coverage_cases if _case_model_name(case) == "vace-1.3B"]
    normalized_vace14b_cases = normalize_cases({"cases": vace14b_cases}, build_track_index({"tracks": records}), resolver, str(out_vace14b_plan))
    normalized_vace13b_cases = normalize_cases({"cases": vace13b_cases}, build_track_index({"tracks": records}), resolver, str(out_vace13b_plan))
    validation = validate_execution_cases(normalized_plan_cases)
    coverage_validation = validate_execution_cases(normalized_coverage_cases)
    reserve_validation = validate_execution_cases(normalized_reserve_cases)
    vace14b_validation = _validate_execution_cases_allow_empty(normalized_vace14b_cases)
    vace13b_validation = _validate_execution_cases_allow_empty(normalized_vace13b_cases)
    summary = _summary(audit_rows, plan_cases)
    coverage_summary = _summary(audit_rows, coverage_cases)
    reserve_summary = _summary(audit_rows, reserve_cases)
    catalog = {
        "schema_version": "dataA_v1_subject_first_target_catalog_v1",
        "generated_at_utc": utc_now_iso(),
        "track_bank": str(track_bank),
        "base_plan": str(base_plan) if base_plan else None,
        "path_mapping": str(path_mapping) if path_mapping else None,
        "selection_config": config,
        "selected_targets": [audit_record(item) for item in selected_by_video.values()],
        "skipped_templates": skipped_templates,
        "coverage_skipped_templates": coverage_skipped_templates,
        "summary": summary,
        "coverage_summary": coverage_summary,
        "reserve_summary": reserve_summary,
    }
    audit = {
        "schema_version": "dataA_v1_subject_first_selection_audit_v1",
        "generated_at_utc": utc_now_iso(),
        "track_bank": str(track_bank),
        "base_plan": str(base_plan) if base_plan else None,
        "path_mapping": str(path_mapping) if path_mapping else None,
        "summary": summary,
        "coverage_summary": coverage_summary,
        "reserve_summary": reserve_summary,
        "candidates": audit_rows,
    }
    plan_payload = _plan_payload(
        schema_version="dataA_v1_frozen_subject_first_vace_execution_plan_v1",
        track_bank=track_bank,
        base_plan=base_plan,
        path_mapping=path_mapping,
        selection_summary=summary,
        validation=validation,
        cases=plan_cases,
    )
    coverage_payload = _plan_payload(
        schema_version="dataA_v1_frozen_subject_first_vace_execution_plan_coverage_v1",
        track_bank=track_bank,
        base_plan=base_plan,
        path_mapping=path_mapping,
        selection_summary=coverage_summary,
        validation=coverage_validation,
        cases=coverage_cases,
    )
    reserve_payload = _plan_payload(
        schema_version="dataA_v1_frozen_subject_first_vace_execution_plan_reserve_v1",
        track_bank=track_bank,
        base_plan=base_plan,
        path_mapping=path_mapping,
        selection_summary=reserve_summary,
        validation=reserve_validation,
        cases=reserve_cases,
    )
    vace14b_payload = _plan_payload(
        schema_version="dataA_v1_frozen_subject_first_vace14b_execution_plan_v1",
        track_bank=track_bank,
        base_plan=base_plan,
        path_mapping=path_mapping,
        selection_summary=_summary(audit_rows, vace14b_cases),
        validation=vace14b_validation,
        cases=vace14b_cases,
    )
    vace13b_payload = _plan_payload(
        schema_version="dataA_v1_frozen_subject_first_vace13b_execution_plan_v1",
        track_bank=track_bank,
        base_plan=base_plan,
        path_mapping=path_mapping,
        selection_summary=_summary(audit_rows, vace13b_cases),
        validation=vace13b_validation,
        cases=vace13b_cases,
    )

    write_json(out_catalog, catalog)
    write_json(out_audit_json, audit)
    _write_audit_csv(out_audit_csv, audit_rows)
    if not dry_run:
        write_json(out_plan, plan_payload)
        write_json(out_coverage_plan, coverage_payload)
        write_json(out_reserve_plan, reserve_payload)
        write_json(out_vace14b_plan, vace14b_payload)
        write_json(out_vace13b_plan, vace13b_payload)
    return {
        "dry_run": dry_run,
        "out_catalog": str(out_catalog),
        "out_audit_json": str(out_audit_json),
        "out_audit_csv": str(out_audit_csv),
        "out_plan": None if dry_run else str(out_plan),
        "out_coverage_plan": None if dry_run else str(out_coverage_plan),
        "out_reserve_plan": None if dry_run else str(out_reserve_plan),
        "out_vace14b_plan": None if dry_run else str(out_vace14b_plan),
        "out_vace13b_plan": None if dry_run else str(out_vace13b_plan),
        "summary": summary,
        "coverage_summary": coverage_summary,
        "reserve_summary": reserve_summary,
        "validation": validation,
        "coverage_validation": coverage_validation,
        "reserve_validation": reserve_validation,
        "vace14b_validation": vace14b_validation,
        "vace13b_validation": vace13b_validation,
        "skipped_templates": skipped_templates,
        "coverage_skipped_templates": coverage_skipped_templates,
    }


def _threshold_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    universal: Dict[str, Any] = {}
    secondary: Dict[str, Any] = {}
    if args.primary_probability is not None:
        overrides: Dict[str, Any] = {"primary_probability": args.primary_probability}
    else:
        overrides = {}
    if args.min_contiguous_visible_seconds is not None:
        universal["min_contiguous_visible_seconds"] = args.min_contiguous_visible_seconds
    if args.min_median_mask_area_ratio is not None:
        universal["min_median_mask_area_ratio"] = args.min_median_mask_area_ratio
    if args.min_p20_mask_area_ratio is not None:
        universal["min_p20_mask_area_ratio"] = args.min_p20_mask_area_ratio
    if args.min_median_bbox_short_side_720 is not None:
        universal["min_median_bbox_short_side_720"] = args.min_median_bbox_short_side_720
    if args.secondary_min_median_mask_area_ratio is not None:
        secondary["min_median_mask_area_ratio"] = args.secondary_min_median_mask_area_ratio
    if args.secondary_min_p20_mask_area_ratio is not None:
        secondary["min_p20_mask_area_ratio"] = args.secondary_min_p20_mask_area_ratio
    if args.secondary_min_median_bbox_short_side_720 is not None:
        secondary["min_median_bbox_short_side_720"] = args.secondary_min_median_bbox_short_side_720
    if universal:
        overrides["universal_gate"] = universal
    if secondary:
        overrides["secondary_gate"] = secondary
    return overrides


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-bank", required=True, type=Path)
    parser.add_argument("--base-plan", type=Path, default=None)
    parser.add_argument("--path-mapping", type=Path, default=None)
    parser.add_argument("--selection-config", type=Path, default=Path("configs/dataa_v1/subject_selection_v1.json"))
    parser.add_argument("--out-catalog", type=Path, default=DEFAULT_OUT_CATALOG)
    parser.add_argument("--out-audit-json", type=Path, default=DEFAULT_OUT_AUDIT_JSON)
    parser.add_argument("--out-audit-csv", type=Path, default=DEFAULT_OUT_AUDIT_CSV)
    parser.add_argument("--out-plan", type=Path, default=DEFAULT_OUT_PLAN)
    parser.add_argument("--out-coverage-plan", type=Path, default=DEFAULT_OUT_COVERAGE_PLAN)
    parser.add_argument("--out-reserve-plan", type=Path, default=DEFAULT_OUT_RESERVE_PLAN)
    parser.add_argument("--out-vace14b-plan", type=Path, default=DEFAULT_OUT_VACE14B_PLAN)
    parser.add_argument("--out-vace13b-plan", type=Path, default=DEFAULT_OUT_VACE13B_PLAN)
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--num-workers", type=int, default=_default_num_workers())
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--primary-probability", type=float, default=None)
    parser.add_argument("--min-contiguous-visible-seconds", type=float, default=None)
    parser.add_argument("--min-median-mask-area-ratio", type=float, default=None)
    parser.add_argument("--min-p20-mask-area-ratio", type=float, default=None)
    parser.add_argument("--min-median-bbox-short-side-720", type=float, default=None)
    parser.add_argument("--secondary-min-median-mask-area-ratio", type=float, default=None)
    parser.add_argument("--secondary-min-p20-mask-area-ratio", type=float, default=None)
    parser.add_argument("--secondary-min-median-bbox-short-side-720", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_subject_first_plan(
            track_bank=_resolve_project_path(args.track_bank) or args.track_bank,
            selection_config=_resolve_project_path(args.selection_config),
            base_plan=_resolve_project_path(args.base_plan),
            path_mapping=_resolve_project_path(args.path_mapping),
            out_catalog=_resolve_project_path(args.out_catalog) or args.out_catalog,
            out_audit_json=_resolve_project_path(args.out_audit_json) or args.out_audit_json,
            out_audit_csv=_resolve_project_path(args.out_audit_csv) or args.out_audit_csv,
            out_plan=_resolve_project_path(args.out_plan) or args.out_plan,
            out_coverage_plan=_resolve_project_path(args.out_coverage_plan) or args.out_coverage_plan,
            out_reserve_plan=_resolve_project_path(args.out_reserve_plan) or args.out_reserve_plan,
            out_vace14b_plan=_resolve_project_path(args.out_vace14b_plan) or args.out_vace14b_plan,
            out_vace13b_plan=_resolve_project_path(args.out_vace13b_plan) or args.out_vace13b_plan,
            ffprobe_bin=args.ffprobe_bin,
            seed=args.seed,
            dry_run=bool(args.dry_run),
            progress_every=max(0, int(args.progress_every)),
            num_workers=max(1, int(args.num_workers)),
            config_overrides=_threshold_overrides(args),
        )
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    reserve_ok = summary["reserve_validation"]["case_count"] == 0 or summary["reserve_validation"]["valid"]
    print(
        "subject_first_plan "
        f"dry_run={summary['dry_run']} "
        f"videos_total={summary['summary']['videos_total']} "
        f"videos_with_primary={summary['summary']['videos_with_primary']} "
        f"cases={summary['validation']['case_count']} "
        f"coverage_cases={summary['coverage_validation']['case_count']} "
        f"reserve_cases={summary['reserve_validation']['case_count']} "
        f"vace14b_cases={summary['vace14b_validation']['case_count']} "
        f"vace13b_cases={summary['vace13b_validation']['case_count']} "
        f"valid={summary['validation']['valid']} "
        f"coverage_valid={summary['coverage_validation']['valid']} "
        f"reserve_valid={reserve_ok}"
    )
    if not summary["validation"]["valid"] or not summary["coverage_validation"]["valid"] or not reserve_ok:
        rejection_tags = summary["summary"].get("rejection_tag_counts") or {}
        rejection_reasons = summary["summary"].get("rejection_reason_counts") or {}
        print(f"validation_errors={summary['validation']['errors']}", file=sys.stderr)
        print(f"coverage_validation_errors={summary['coverage_validation']['errors']}", file=sys.stderr)
        print(f"reserve_validation_errors={summary['reserve_validation']['errors']}", file=sys.stderr)
        if rejection_tags:
            print(f"top_rejection_tags={rejection_tags}", file=sys.stderr)
        if rejection_reasons:
            print(f"top_rejection_reasons={rejection_reasons}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
