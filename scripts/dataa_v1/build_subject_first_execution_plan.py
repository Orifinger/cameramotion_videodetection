#!/usr/bin/env python3
"""Build a subject-first Data A v1 VACE execution plan from existing tracks."""

from __future__ import annotations

import argparse
import csv
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


def _selection_meta(selected: Any, config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "selection_role": selected.selection_role,
        "selection_mode": selected.selection_mode,
        "primary_probability": float(config.get("primary_probability", 0.85)),
        "secondary_pool_size": int(selected.secondary_pool_size),
        "random_seed": int(config.get("random_seed", 20260629)),
        "selection_random_value": float(selected.selection_random_value or 0.0),
        "subject_score": selected.subject_score,
    }


def _quantiles(values: list[float]) -> Dict[str, float]:
    if not values:
        return {"p20": 0.0, "median": 0.0, "p80": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"p20": float(np.quantile(arr, 0.20)), "median": float(np.median(arr)), "p80": float(np.quantile(arr, 0.80))}


def _summary(audit_records: list[Dict[str, Any]], plan_cases: list[Dict[str, Any]]) -> Dict[str, Any]:
    selected = [row for row in audit_records if row["selection_status"] == "selected"]
    videos = {row["video_id"] for row in audit_records}
    videos_with_primary = {row["video_id"] for row in audit_records if row["selection_status"] in {"selected", "primary_subject"}}
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
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = dict(row)
            clean["rejection_tags"] = ";".join(row.get("rejection_tags") or [])
            writer.writerow({field: clean.get(field) for field in fields})


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


def _minimal_case(index: int, selected: Any, config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": f"dataA_v1_subject_first_{index:05d}",
        "operation": "object_attribute_edit",
        "generator_route": "vace14b_masktrack_text_edit",
        "target": _track_payload(selected.record),
        "donor": None,
        "sampling_meta": {
            "target_selection": _selection_meta(selected, config),
            "target_saliency": metric_payload(selected),
            "subject_first_source": "scripts/dataa_v1/build_subject_first_execution_plan.py",
            "frozen": True,
        },
    }


def build_subject_first_plan(
    *,
    track_bank: Path,
    selection_config: Path | None,
    base_plan: Path | None,
    out_catalog: Path,
    out_audit_json: Path,
    out_audit_csv: Path,
    out_plan: Path,
    ffprobe_bin: str = "ffprobe",
    seed: int | None = None,
    dry_run: bool = False,
    config_overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = dict(config_overrides or {})
    if seed is not None:
        overrides["random_seed"] = int(seed)
    config = load_selection_config(selection_config, overrides=overrides)
    records = load_track_bank_records(track_bank)
    evaluated = evaluate_tracks(records, config, ffprobe_bin=ffprobe_bin)
    selections = select_subjects_by_video(evaluated, config)
    selected_by_video = {video_id: choice.selected for video_id, choice in selections.items() if choice.selected is not None}

    audit_rows = [audit_record(item) for item in evaluated]
    plan_cases: list[Dict[str, Any]] = []
    skipped_templates: list[Dict[str, Any]] = []
    if base_plan is not None:
        track_index = build_track_index({"tracks": records})
        base_cases = normalize_cases(read_json(base_plan), track_index, PathResolver({}), str(base_plan))
        used_videos: set[str] = set()
        for case in base_cases:
            video_id = case.target.video_id
            selected = selected_by_video.get(str(video_id)) if video_id else None
            if selected is None:
                skipped_templates.append({"case_id": case.case_id, "reason": "no_subject_first_target_for_video", "video_id": video_id})
                continue
            if selected.video_id in used_videos:
                skipped_templates.append({"case_id": case.case_id, "reason": "target_video_already_used", "video_id": selected.video_id})
                continue
            new_case = _case_from_template(case, selected, config)
            if new_case is None:
                skipped_templates.append({"case_id": case.case_id, "reason": "invalid_reference_route_or_same_video_donor", "video_id": selected.video_id})
                continue
            used_videos.add(selected.video_id)
            plan_cases.append(new_case)
    else:
        for index, selected in enumerate(sorted(selected_by_video.values(), key=lambda item: item.video_id)):
            plan_cases.append(_minimal_case(index, selected, config))

    normalized_plan_cases = normalize_cases({"cases": plan_cases}, build_track_index({"tracks": records}), PathResolver({}), str(out_plan))
    validation = validate_execution_cases(normalized_plan_cases)
    summary = _summary(audit_rows, plan_cases)
    catalog = {
        "schema_version": "dataA_v1_subject_first_target_catalog_v1",
        "generated_at_utc": utc_now_iso(),
        "track_bank": str(track_bank),
        "base_plan": str(base_plan) if base_plan else None,
        "selection_config": config,
        "selected_targets": [audit_record(item) for item in selected_by_video.values()],
        "skipped_templates": skipped_templates,
        "summary": summary,
    }
    audit = {
        "schema_version": "dataA_v1_subject_first_selection_audit_v1",
        "generated_at_utc": utc_now_iso(),
        "track_bank": str(track_bank),
        "base_plan": str(base_plan) if base_plan else None,
        "summary": summary,
        "candidates": audit_rows,
    }
    plan_payload = {
        "schema_version": "dataA_v1_frozen_subject_first_vace_execution_plan_v1",
        "generated_at_utc": utc_now_iso(),
        "track_bank": str(track_bank),
        "base_plan": str(base_plan) if base_plan else None,
        "case_count": len(plan_cases),
        "selection_summary": summary,
        "validation": validation,
        "cases": plan_cases,
    }

    write_json(out_catalog, catalog)
    write_json(out_audit_json, audit)
    _write_audit_csv(out_audit_csv, audit_rows)
    if not dry_run:
        write_json(out_plan, plan_payload)
    return {
        "dry_run": dry_run,
        "out_catalog": str(out_catalog),
        "out_audit_json": str(out_audit_json),
        "out_audit_csv": str(out_audit_csv),
        "out_plan": None if dry_run else str(out_plan),
        "summary": summary,
        "validation": validation,
        "skipped_templates": skipped_templates,
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
    parser.add_argument("--selection-config", type=Path, default=Path("configs/dataa_v1/subject_selection_v1.json"))
    parser.add_argument("--out-catalog", type=Path, default=DEFAULT_OUT_CATALOG)
    parser.add_argument("--out-audit-json", type=Path, default=DEFAULT_OUT_AUDIT_JSON)
    parser.add_argument("--out-audit-csv", type=Path, default=DEFAULT_OUT_AUDIT_CSV)
    parser.add_argument("--out-plan", type=Path, default=DEFAULT_OUT_PLAN)
    parser.add_argument("--ffprobe-bin", default="ffprobe")
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
            out_catalog=_resolve_project_path(args.out_catalog) or args.out_catalog,
            out_audit_json=_resolve_project_path(args.out_audit_json) or args.out_audit_json,
            out_audit_csv=_resolve_project_path(args.out_audit_csv) or args.out_audit_csv,
            out_plan=_resolve_project_path(args.out_plan) or args.out_plan,
            ffprobe_bin=args.ffprobe_bin,
            seed=args.seed,
            dry_run=bool(args.dry_run),
            config_overrides=_threshold_overrides(args),
        )
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        "subject_first_plan "
        f"dry_run={summary['dry_run']} "
        f"videos_total={summary['summary']['videos_total']} "
        f"videos_with_primary={summary['summary']['videos_with_primary']} "
        f"cases={summary['validation']['case_count']} "
        f"valid={summary['validation']['valid']}"
    )
    if not summary["validation"]["valid"]:
        print(f"validation_errors={summary['validation']['errors']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
