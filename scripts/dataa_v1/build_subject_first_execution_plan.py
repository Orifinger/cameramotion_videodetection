#!/usr/bin/env python3
"""Build a subject-first Data A v1 VACE execution plan from existing tracks."""

from __future__ import annotations

import argparse
import csv
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


def _case_from_template(case: Any, selected: Any, config: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if case.generator_route == "vace14b_masktrack_reference_swap" and case.donor is None:
        return None
    if case.donor and case.donor.video_id and case.donor.video_id == selected.video_id:
        return None
    sampling_meta = dict(case.sampling_meta or {})
    sampling_meta["target_selection"] = _selection_meta(selected, config)
    sampling_meta["target_saliency"] = metric_payload(selected)
    sampling_meta["subject_first_source"] = "scripts/dataa_v1/build_subject_first_execution_plan.py"
    sampling_meta["frozen"] = True
    return {
        "case_id": case.case_id,
        "operation": case.operation,
        "generator_route": case.generator_route,
        "target": _track_payload(selected.record),
        "donor": _donor_payload(case.donor),
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
    donor = case.donor
    donor_repair = None
    if case.generator_route == "vace14b_masktrack_reference_swap":
        needs_repair = donor is None or (donor.video_id and donor.video_id == selected.video_id)
        if needs_repair:
            repaired_donor, donor_repair = _repair_donor(selected, donor_pool)
            if repaired_donor is None:
                return None
            donor = repaired_donor
            donor_repair["reason"] = "missing_donor" if case.donor is None else "same_video_donor"

    sampling_meta = dict(case.sampling_meta or {})
    sampling_meta["target_selection"] = _selection_meta(
        selected,
        config,
        quality_tier=quality_tier,
        risk_tags=risk_tags,
        donor_repair=donor_repair,
    )
    sampling_meta["target_saliency"] = metric_payload(selected)
    sampling_meta["subject_first_source"] = "scripts/dataa_v1/build_subject_first_execution_plan.py"
    sampling_meta["frozen"] = True
    sampling_meta["coverage_plan"] = quality_tier != "clean"
    return {
        "case_id": case.case_id,
        "operation": case.operation,
        "generator_route": case.generator_route,
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
) -> Dict[str, Any]:
    return {
        "case_id": f"{case_id_prefix}_{index:05d}",
        "operation": "object_attribute_edit",
        "generator_route": "vace14b_masktrack_text_edit",
        "target": _track_payload(selected.record),
        "donor": None,
        "sampling_meta": {
            "target_selection": _selection_meta(selected, config, quality_tier=quality_tier, risk_tags=risk_tags),
            "target_saliency": metric_payload(selected),
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
            selected = selected_by_video.get(str(video_id)) if video_id else None
            coverage_selected = coverage_by_video.get(str(video_id)) if video_id else None
            clean_case_added = False
            if selected is None:
                skipped_templates.append({"case_id": case.case_id, "reason": "no_subject_first_target_for_video", "video_id": video_id})
            elif selected.video_id in used_videos:
                skipped_templates.append({"case_id": case.case_id, "reason": "target_video_already_used", "video_id": selected.video_id})
            else:
                new_case = _case_from_template(case, selected, config)
                if new_case is not None:
                    used_videos.add(selected.video_id)
                    used_coverage_videos.add(selected.video_id)
                    plan_cases.append(new_case)
                    coverage_cases.append(new_case)
                    clean_case_added = True
                else:
                    skipped_templates.append({"case_id": case.case_id, "reason": "invalid_reference_route_or_same_video_donor", "video_id": selected.video_id})

            if clean_case_added:
                continue
            if coverage_selected is None:
                coverage_skipped_templates.append({"case_id": case.case_id, "reason": "no_readable_target_for_coverage", "video_id": video_id})
                continue
            if coverage_selected.video_id in used_coverage_videos:
                coverage_skipped_templates.append({"case_id": case.case_id, "reason": "coverage_target_video_already_used", "video_id": coverage_selected.video_id})
                continue
            quality_tier = "clean_donor_repair" if selected is not None else "relaxed_rescue"
            risk_tags = list(coverage_selected.rejection_tags)
            if selected is None:
                risk_tags.append("relaxed_target_gate")
            coverage_case = _case_from_template_coverage(
                case,
                coverage_selected,
                config,
                quality_tier=quality_tier,
                risk_tags=risk_tags,
                donor_pool=donor_pool,
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
            coverage_cases.append(
                _minimal_case(
                    extra_index,
                    selected,
                    config,
                    case_id_prefix="dataA_v1_subject_first_coverage",
                    quality_tier=quality_tier,
                    risk_tags=risk_tags,
                )
            )
            used_coverage_videos.add(video_id)
            extra_index += 1
    else:
        for index, selected in enumerate(sorted(selected_by_video.values(), key=lambda item: item.video_id)):
            case = _minimal_case(index, selected, config)
            plan_cases.append(case)
            coverage_cases.append(case)
        clean_videos = set(selected_by_video)
        extra_index = 0
        for video_id, selected in sorted(coverage_by_video.items()):
            if video_id in clean_videos:
                continue
            coverage_cases.append(
                _minimal_case(
                    extra_index,
                    selected,
                    config,
                    case_id_prefix="dataA_v1_subject_first_coverage",
                    quality_tier="coverage_only_relaxed",
                    risk_tags=[*selected.rejection_tags, "relaxed_target_gate"],
                )
            )
            extra_index += 1

    normalized_plan_cases = normalize_cases({"cases": plan_cases}, build_track_index({"tracks": records}), resolver, str(out_plan))
    normalized_coverage_cases = normalize_cases({"cases": coverage_cases}, build_track_index({"tracks": records}), resolver, str(out_coverage_plan))
    clean_case_ids = {str(case.get("case_id")) for case in plan_cases}
    reserve_cases = [case for case in coverage_cases if str(case.get("case_id")) not in clean_case_ids]
    normalized_reserve_cases = normalize_cases({"cases": reserve_cases}, build_track_index({"tracks": records}), resolver, str(out_reserve_plan))
    validation = validate_execution_cases(normalized_plan_cases)
    coverage_validation = validate_execution_cases(normalized_coverage_cases)
    reserve_validation = validate_execution_cases(normalized_reserve_cases)
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

    write_json(out_catalog, catalog)
    write_json(out_audit_json, audit)
    _write_audit_csv(out_audit_csv, audit_rows)
    if not dry_run:
        write_json(out_plan, plan_payload)
        write_json(out_coverage_plan, coverage_payload)
        write_json(out_reserve_plan, reserve_payload)
    return {
        "dry_run": dry_run,
        "out_catalog": str(out_catalog),
        "out_audit_json": str(out_audit_json),
        "out_audit_csv": str(out_audit_csv),
        "out_plan": None if dry_run else str(out_plan),
        "out_coverage_plan": None if dry_run else str(out_coverage_plan),
        "out_reserve_plan": None if dry_run else str(out_reserve_plan),
        "summary": summary,
        "coverage_summary": coverage_summary,
        "reserve_summary": reserve_summary,
        "validation": validation,
        "coverage_validation": coverage_validation,
        "reserve_validation": reserve_validation,
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
