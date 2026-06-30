#!/usr/bin/env python3
"""Build a continuation VACE plan while preserving already generated cases."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.build_subject_first_execution_plan import (
    REFERENCE_ROUTE,
    _case_from_template,
    _case_model_name,
    _donor_match_score,
    _is_person_track,
    _is_surface_track,
    _largest_track_key,
    _operation_compatible,
    _operation_gate_report,
    _plan_payload,
    _repo_root,
    _summary,
    _validate_execution_cases_allow_empty,
)
from scripts.dataa_v1.common import DataAError, read_json, utc_now_iso, write_json
from scripts.dataa_v1.path_resolver import PathResolver
from scripts.dataa_v1.schema import build_track_index, normalize_cases
from scripts.dataa_v1.subject_selection import (
    audit_record,
    evaluate_tracks,
    load_selection_config,
    load_track_bank_records,
    select_subjects_by_video,
)


DEFAULT_OUT_PLAN = Path("res/dataA_v1/plans/frozen_subject_first_vace_continuation_plan.json")
DEFAULT_OUT_VACE14B_PLAN = Path("res/dataA_v1/plans/frozen_subject_first_vace14b_continuation_plan.json")
DEFAULT_OUT_VACE13B_PLAN = Path("res/dataA_v1/plans/frozen_subject_first_vace13b_continuation_plan.json")
DEFAULT_OUT_RESERVE_PLAN = Path("res/dataA_v1/plans/frozen_subject_first_vace_continuation_reserve_plan.json")
DEFAULT_OUT_VACE14B_RESERVE_PLAN = Path("res/dataA_v1/plans/frozen_subject_first_vace14b_continuation_reserve_plan.json")
DEFAULT_OUT_VACE13B_RESERVE_PLAN = Path("res/dataA_v1/plans/frozen_subject_first_vace13b_continuation_reserve_plan.json")
DEFAULT_OUT_AUDIT = Path("res/dataA_v1/audits/subject_first_continuation_plan_audit.json")
DEFAULT_OUT_RERUN_MANIFEST = Path("res/dataA_v1/audits/subject_first_qwen_sam3_rerun_manifest.json")
GOOD_PAIR_MATCH_LEVELS = {
    "same_canonical_concept",
    "same_candidate_class",
    "same_region_family",
    "same_content_domain",
}


@dataclass(frozen=True)
class CompletedCase:
    case_id: str
    video_id: str | None
    source: str
    status: str


def _resolve_project_path(path: Path | str | None) -> Path | None:
    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else _repo_root() / value


def _safe_read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except DataAError:
        return None


def _case_video_id_from_manifest(manifest: Mapping[str, Any] | None) -> str | None:
    if not manifest:
        return None
    target = manifest.get("target") or {}
    if isinstance(target, Mapping):
        value = target.get("video_id")
        return None if value in (None, "") else str(value)
    return None


def _is_generated_attempt(attempt_dir: Path) -> tuple[bool, str]:
    result = _safe_read_json(attempt_dir / "generation_result.json")
    manifest = _safe_read_json(attempt_dir / "case_manifest.json")
    full_video = {}
    if isinstance(result, Mapping):
        full_video = result.get("full_video") or {}
    if not full_video and isinstance(manifest, Mapping):
        full_video = manifest.get("full_video") or {}
    status = str(result.get("status") if isinstance(result, Mapping) else "")
    full_status = str(full_video.get("status") if isinstance(full_video, Mapping) else "")
    local_full_pair = (attempt_dir / "full_real.mp4").is_file() and (attempt_dir / "full_fake.mp4").is_file()
    if status == "generated" and (full_status == "ok" or local_full_pair):
        return True, "generated_full_video"
    if full_status == "ok" and local_full_pair:
        return True, "full_video_files_present"
    return False, status or full_status or "not_completed"


def _completed_from_attempts(run_root: Path) -> Dict[str, CompletedCase]:
    completed: Dict[str, CompletedCase] = {}
    for manifest_path in run_root.rglob("case_manifest.json"):
        attempt_dir = manifest_path.parent
        manifest = _safe_read_json(manifest_path)
        if not isinstance(manifest, Mapping):
            continue
        case_id = str(manifest.get("case_id") or attempt_dir.name)
        ok, status = _is_generated_attempt(attempt_dir)
        if not ok:
            continue
        completed[case_id] = CompletedCase(
            case_id=case_id,
            video_id=_case_video_id_from_manifest(manifest),
            source=str(attempt_dir),
            status=status,
        )
    return completed


def _completed_from_state(run_root: Path) -> Dict[str, CompletedCase]:
    completed: Dict[str, CompletedCase] = {}
    state = _safe_read_json(run_root / "coordinator" / "run_state.json")
    if isinstance(state, Mapping):
        for case_id, info in (state.get("cases") or {}).items():
            if not isinstance(info, Mapping):
                continue
            status = str(info.get("status") or "")
            detail = info.get("detail") or {}
            receipt = detail.get("upload_receipt") if isinstance(detail, Mapping) else None
            if status in {"generated", "uploaded_verified", "accepted"} or receipt:
                completed[str(case_id)] = CompletedCase(str(case_id), None, str(run_root / "coordinator" / "run_state.json"), status)
    jsonl = run_root / "coordinator" / "case_status.jsonl"
    if jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = str(event.get("status") or "")
            case_id = str(event.get("case_id") or "")
            detail = event.get("detail") or {}
            receipt = detail.get("upload_receipt") if isinstance(detail, Mapping) else None
            if case_id and (status in {"generated", "uploaded_verified", "accepted"} or receipt):
                completed.setdefault(case_id, CompletedCase(case_id, None, str(jsonl), status))
    return completed


def discover_completed_cases(run_roots: Sequence[Path]) -> Dict[str, CompletedCase]:
    completed: Dict[str, CompletedCase] = {}
    for run_root in run_roots:
        if not run_root:
            continue
        completed.update(_completed_from_state(run_root))
        completed.update(_completed_from_attempts(run_root))
    return completed


def _compatible_donor(target: Any, operation: str, donor_pool: Sequence[Any]) -> tuple[Any | None, Dict[str, Any] | None]:
    candidates = [
        donor
        for donor in donor_pool
        if donor.metrics is not None
        and donor.video_id != target.video_id
        and donor.track_id != target.track_id
        and _operation_compatible(operation, donor)
    ]
    if not candidates:
        return None, None
    scored = []
    for donor in candidates:
        score, level = _donor_match_score(target, donor)
        scored.append((score, donor.subject_score, donor.track_id, level, donor))
    score, _subject_score, _track_id, level, donor = max(scored, key=lambda item: (item[0], item[1], item[2]))
    return donor, {
        "enabled": True,
        "reason": "continuation_person_preferred_requires_compatible_donor",
        "match_level": level,
        "score": int(score),
        "donor_track_id": donor.track_id,
        "donor_video_id": donor.video_id,
    }


def _track_kind(track: Any | None) -> str:
    if track is None:
        return "none"
    if _is_person_track(track):
        return "person"
    if _is_surface_track(track):
        return "surface"
    return "object"


def _track_raw(track: Any | None) -> Dict[str, Any]:
    if track is None:
        return {}
    if hasattr(track, "record"):
        return dict(track.record)
    if isinstance(track, Mapping):
        return dict(track)
    raw = getattr(track, "raw", None)
    if isinstance(raw, Mapping):
        merged = dict(raw)
    else:
        merged = {}
    for key in ("track_id", "video_id", "candidate_class", "canonical_concept", "display_phrase", "region_family", "content_domain", "style_domain"):
        value = getattr(track, key, None)
        if value not in (None, ""):
            merged.setdefault(key, value)
    return merged


def _simple_pair_match(target: Any | None, donor: Any | None) -> Dict[str, Any]:
    if donor is None:
        return {"score": 0, "match_level": "missing_donor", "good_pair": False}
    target_raw = _track_raw(target)
    donor_raw = _track_raw(donor)
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
            return {"score": score, "match_level": level, "good_pair": level in GOOD_PAIR_MATCH_LEVELS}
    target_words = set(
        " ".join(str(target_raw.get(key) or "").lower() for key in ("canonical_concept", "candidate_class", "region_family", "content_domain")).split()
    )
    donor_words = set(
        " ".join(str(donor_raw.get(key) or "").lower() for key in ("canonical_concept", "candidate_class", "region_family", "content_domain")).split()
    )
    if target_words and donor_words and target_words.intersection(donor_words):
        return {"score": 15, "match_level": "similar_text_overlap", "good_pair": False}
    return {"score": 1, "match_level": "weak_cross_category", "good_pair": False}


def _candidate_by_track_id(selection: Any, track_id: str | None) -> Any | None:
    if not track_id:
        return None
    for item in selection.candidates:
        if item.metrics is not None and str(item.track_id) == str(track_id):
            return item
    return None


def _mask_not_small(track: Any | None, operation: str | None, config: Mapping[str, Any]) -> bool:
    if track is None or getattr(track, "metrics", None) is None:
        return False
    return bool(_operation_gate_report(track, operation, config)["pass"])


def _eligible_person_targets(selection: Any, config: Mapping[str, Any]) -> list[Any]:
    candidates = [item for item in selection.candidates if item.metrics is not None and _is_person_track(item)]
    return [item for item in candidates if _operation_gate_report(item, "person_appearance_swap", config)["pass"]]


def _base_pair_keep_case(
    *,
    case: Any,
    selection: Any,
    config: Mapping[str, Any],
) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
    target = _candidate_by_track_id(selection, case.target.track_id)
    gate = _operation_gate_report(target, case.operation, config) if target is not None else {"pass": False, "failures": ["missing_usable_target_track"]}
    pair = _simple_pair_match(target or case.target, case.donor)
    is_person_swap = case.operation == "person_appearance_swap"
    text_route_keep = case.generator_route != REFERENCE_ROUTE and target is not None and bool(gate["pass"])
    reference_keep = (
        case.generator_route == REFERENCE_ROUTE
        and target is not None
        and case.donor is not None
        and bool(gate["pass"])
        and (bool(pair["good_pair"]) or is_person_swap)
    )
    keep = bool(reference_keep or text_route_keep)
    reason = {
        "case_id": case.case_id,
        "video_id": case.target.video_id,
        "operation": case.operation,
        "generator_route": case.generator_route,
        "target_track_id": case.target.track_id,
        "donor_track_id": None if case.donor is None else case.donor.track_id,
        "target_kind": _track_kind(target or case.target),
        "donor_kind": _track_kind(case.donor),
        "pair_match": pair,
        "operation_gate": gate,
        "keep": keep,
        "reason": "keep_person_swap_or_good_pair" if keep else "needs_qwen_sam3_rerun_and_repair",
    }
    if not keep:
        return None, reason
    payload = _case_from_template(
        case,
        target,
        config,
        operation=case.operation,
        generator_route=case.generator_route,
        donor=case.donor,
        quality_tier="continuation_keep_person_swap" if is_person_swap else "continuation_keep_good_pair",
        risk_tags=["continuation_keep_existing_pair"],
        donor_repair=None,
        operation_repair=None,
        target_repair=None,
    )
    if payload is None:
        reason["keep"] = False
        reason["reason"] = "keepable_case_build_failed"
        return None, reason
    _add_continuation_meta(payload, base_case=case, strategy="keep_person_swap" if is_person_swap else "keep_good_base_pair")
    return payload, reason


def _add_continuation_meta(
    case_payload: Dict[str, Any],
    *,
    base_case: Any,
    strategy: str,
    completed_source: str | None = None,
) -> None:
    meta = case_payload.setdefault("sampling_meta", {})
    meta["continuation"] = {
        "schema_version": "dataA_v1_continuation_plan_v1",
        "generated_at_utc": utc_now_iso(),
        "source": "scripts/dataa_v1/build_continuation_execution_plan.py",
        "strategy": strategy,
        "base_case_id": base_case.case_id,
        "base_operation": base_case.operation,
        "base_generator_route": base_case.generator_route,
        "base_target_track_id": base_case.target.track_id,
        "base_target_video_id": base_case.target.video_id,
        "completed_source": completed_source,
    }


def _person_preferred_case(
    *,
    case: Any,
    selection: Any,
    config: Mapping[str, Any],
    donor_pool: Sequence[Any],
) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    people = _eligible_person_targets(selection, config)
    if not people:
        return None, None
    person = max(people, key=_largest_track_key)
    donor, donor_repair = _compatible_donor(person, "person_appearance_swap", donor_pool)
    if donor is None:
        return None, {
            "case_id": case.case_id,
            "video_id": case.target.video_id,
            "reason": "eligible_person_without_compatible_donor",
            "person_track_id": person.track_id,
        }
    operation_repair = None
    if case.operation != "person_appearance_swap" or case.generator_route != REFERENCE_ROUTE or case.target.track_id != person.track_id:
        operation_repair = {
            "enabled": True,
            "reason": "continuation_foreground_person_preferred_for_aigc_artifact",
            "original_operation": case.operation,
            "original_generator_route": case.generator_route,
            "repaired_operation": "person_appearance_swap",
            "repaired_generator_route": REFERENCE_ROUTE,
            "repaired_track_id": person.track_id,
        }
    target_repair = None
    if case.target.track_id != person.track_id:
        target_repair = {
            "enabled": True,
            "reason": "continuation_eligible_person_preferred_over_base_target",
            "original_track_id": case.target.track_id,
            "repaired_track_id": person.track_id,
            "operation": "person_appearance_swap",
        }
    case_payload = _case_from_template(
        case,
        person,
        config,
        operation="person_appearance_swap",
        generator_route=REFERENCE_ROUTE,
        donor=donor,
        quality_tier="continuation_person_preferred",
        risk_tags=["continuation_person_preferred"],
        donor_repair=donor_repair,
        operation_repair=operation_repair,
        target_repair=target_repair,
    )
    if case_payload is None:
        return None, {
            "case_id": case.case_id,
            "video_id": case.target.video_id,
            "reason": "person_preferred_case_build_failed",
            "person_track_id": person.track_id,
        }
    policy = case_payload["sampling_meta"]["mask_policy"]
    policy["person_bbox_disabled"] = True
    if policy.get("variant_type") == "expanded_bbox":
        policy["variant_type"] = "sam3_shape"
        policy["trigger_reason"] = "person_bbox_policy_blocked"
    _add_continuation_meta(case_payload, base_case=case, strategy="person_preferred")
    return case_payload, None


def _continuation_case_from_base(
    *,
    case: Any,
    selection: Any,
    config: Mapping[str, Any],
    donor_pool: Sequence[Any],
) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    person_case, person_skip = _person_preferred_case(case=case, selection=selection, config=config, donor_pool=donor_pool)
    if person_case is not None:
        return person_case, None

    keep_case, keep_reason = _base_pair_keep_case(case=case, selection=selection, config=config)
    if keep_case is not None:
        return keep_case, None

    keep_reason["person_preference_skip"] = person_skip
    return None, keep_reason


def _case_index_row(case_payload: Mapping[str, Any]) -> Dict[str, Any]:
    target = case_payload.get("target") or {}
    donor = case_payload.get("donor") or {}
    meta = case_payload.get("sampling_meta") or {}
    continuation = meta.get("continuation") or {}
    mask_policy = meta.get("mask_policy") or {}
    return {
        "case_id": case_payload.get("case_id"),
        "video_id": target.get("video_id") if isinstance(target, Mapping) else None,
        "operation": case_payload.get("operation"),
        "generator_route": case_payload.get("generator_route"),
        "target_track_id": target.get("track_id") if isinstance(target, Mapping) else None,
        "donor_track_id": donor.get("track_id") if isinstance(donor, Mapping) else None,
        "continuation_strategy": continuation.get("strategy") if isinstance(continuation, Mapping) else None,
        "base_case_id": continuation.get("base_case_id") if isinstance(continuation, Mapping) else None,
        "mask_variant_type": mask_policy.get("variant_type") if isinstance(mask_policy, Mapping) else None,
        "mask_trigger_reason": mask_policy.get("trigger_reason") if isinstance(mask_policy, Mapping) else None,
    }


def _mark_reserve_case(case_payload: Dict[str, Any], *, reason: str, occupied_video_id: str | None) -> None:
    meta = case_payload.setdefault("sampling_meta", {})
    meta["continuation_reserve"] = {
        "schema_version": "dataA_v1_continuation_reserve_v1",
        "generated_at_utc": utc_now_iso(),
        "reason": reason,
        "occupied_video_id": occupied_video_id,
        "policy": "hold_for_later_plan_with_qwen_sam3_rerun_results",
    }


def _rerun_record(case: Any, *, reason: str, extra: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "case_id": case.case_id,
        "video_id": case.target.video_id,
        "operation": case.operation,
        "generator_route": case.generator_route,
        "target_track_id": case.target.track_id,
        "donor_track_id": None if case.donor is None else case.donor.track_id,
        "reason": reason,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def build_continuation_plan(
    *,
    track_bank: Path,
    base_plan: Path,
    run_roots: Sequence[Path],
    selection_config: Path | None,
    path_mapping: Path | None = None,
    out_plan: Path = DEFAULT_OUT_PLAN,
    out_vace14b_plan: Path = DEFAULT_OUT_VACE14B_PLAN,
    out_vace13b_plan: Path = DEFAULT_OUT_VACE13B_PLAN,
    out_reserve_plan: Path = DEFAULT_OUT_RESERVE_PLAN,
    out_vace14b_reserve_plan: Path = DEFAULT_OUT_VACE14B_RESERVE_PLAN,
    out_vace13b_reserve_plan: Path = DEFAULT_OUT_VACE13B_RESERVE_PLAN,
    out_audit: Path = DEFAULT_OUT_AUDIT,
    out_rerun_manifest: Path = DEFAULT_OUT_RERUN_MANIFEST,
    ffprobe_bin: str = "ffprobe",
    num_workers: int = 1,
    progress_every: int = 0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    config = load_selection_config(selection_config)
    records = load_track_bank_records(track_bank)
    resolver = PathResolver(read_json(path_mapping) if path_mapping else {})
    evaluated = evaluate_tracks(
        records,
        config,
        ffprobe_bin=ffprobe_bin,
        path_resolver=resolver,
        num_workers=num_workers,
        progress_every=progress_every,
    )
    selections = select_subjects_by_video(evaluated, config)
    donor_pool = [item for item in evaluated if item.metrics is not None]
    track_index = build_track_index({"tracks": records})
    base_cases = normalize_cases(read_json(base_plan), track_index, resolver, str(base_plan))
    completed = discover_completed_cases(run_roots)
    completed_case_ids = set(completed)
    completed_video_ids = {item.video_id for item in completed.values() if item.video_id}

    cases: list[Dict[str, Any]] = []
    reserve_cases: list[Dict[str, Any]] = []
    skipped: list[Dict[str, Any]] = []
    rerun_candidates: list[Dict[str, Any]] = []
    completed_skips: list[Dict[str, Any]] = []
    reserve_skips: list[Dict[str, Any]] = []
    used_videos: set[str] = set()
    for case in base_cases:
        video_id = case.target.video_id
        if case.case_id in completed_case_ids:
            skip = {
                "case_id": case.case_id,
                "video_id": video_id,
                "reason": "already_completed_case",
                "source": completed[case.case_id].source,
            }
            completed_skips.append(skip)
            skipped.append(skip)
            continue
        if video_id and video_id in completed_video_ids:
            skip = {"case_id": case.case_id, "video_id": video_id, "reason": "already_completed_video"}
            completed_skips.append(skip)
            skipped.append(skip)
            continue
        selection = selections.get(str(video_id)) if video_id else None
        if selection is None:
            rerun = _rerun_record(case, reason="no_track_bank_selection_for_video")
            rerun_candidates.append(rerun)
            skipped.append(rerun)
            continue
        new_case, skip = _continuation_case_from_base(case=case, selection=selection, config=config, donor_pool=donor_pool)
        if new_case is None:
            rerun = skip or {"case_id": case.case_id, "video_id": video_id, "reason": "continuation_case_failed"}
            rerun_candidates.append(rerun)
            skipped.append(rerun)
            continue
        new_video_id = str((new_case.get("target") or {}).get("video_id") or video_id)
        if new_video_id in completed_video_ids:
            skip = {"case_id": case.case_id, "video_id": new_video_id, "reason": "continuation_repaired_video_already_completed"}
            completed_skips.append(skip)
            skipped.append(skip)
            continue
        if (video_id and video_id in used_videos) or new_video_id in used_videos:
            reason = "continuation_target_video_already_used" if video_id and video_id in used_videos else "continuation_repaired_video_already_used"
            _mark_reserve_case(new_case, reason=reason, occupied_video_id=str(video_id or new_video_id))
            reserve_cases.append(new_case)
            reserve = {
                "case_id": case.case_id,
                "video_id": new_video_id,
                "reason": reason,
                "target_track_id": (new_case.get("target") or {}).get("track_id"),
                "operation": new_case.get("operation"),
                "generator_route": new_case.get("generator_route"),
            }
            reserve_skips.append(reserve)
            skipped.append(reserve)
            continue
        used_videos.add(new_video_id)
        cases.append(new_case)

    normalized_cases = normalize_cases({"cases": cases}, track_index, resolver, str(out_plan))
    validation = _validate_execution_cases_allow_empty(normalized_cases)
    vace14b_cases = [case for case in cases if _case_model_name(case) == "vace-14B"]
    vace13b_cases = [case for case in cases if _case_model_name(case) == "vace-1.3B"]
    normalized_vace14b = normalize_cases({"cases": vace14b_cases}, track_index, resolver, str(out_vace14b_plan))
    normalized_vace13b = normalize_cases({"cases": vace13b_cases}, track_index, resolver, str(out_vace13b_plan))
    vace14b_validation = _validate_execution_cases_allow_empty(normalized_vace14b)
    vace13b_validation = _validate_execution_cases_allow_empty(normalized_vace13b)
    normalized_reserve = normalize_cases({"cases": reserve_cases}, track_index, resolver, str(out_reserve_plan))
    reserve_validation = _validate_execution_cases_allow_empty(normalized_reserve)
    reserve_vace14b_cases = [case for case in reserve_cases if _case_model_name(case) == "vace-14B"]
    reserve_vace13b_cases = [case for case in reserve_cases if _case_model_name(case) == "vace-1.3B"]
    normalized_reserve_vace14b = normalize_cases({"cases": reserve_vace14b_cases}, track_index, resolver, str(out_vace14b_reserve_plan))
    normalized_reserve_vace13b = normalize_cases({"cases": reserve_vace13b_cases}, track_index, resolver, str(out_vace13b_reserve_plan))
    reserve_vace14b_validation = _validate_execution_cases_allow_empty(normalized_reserve_vace14b)
    reserve_vace13b_validation = _validate_execution_cases_allow_empty(normalized_reserve_vace13b)
    audit_rows = [audit_record(item) for item in evaluated]
    summary = _summary(audit_rows, cases)
    summary["completed_case_count"] = len(completed_case_ids)
    summary["completed_video_count"] = len(completed_video_ids)
    summary["completed_skip_count"] = len(completed_skips)
    summary["reserve_case_count"] = len(reserve_cases)
    summary["reserve_reason_counts"] = dict(Counter(str(item.get("reason") or "<missing>") for item in reserve_skips))
    summary["continuation_operation_counts"] = dict(Counter(str(case.get("operation") or "<missing>") for case in cases))
    summary["continuation_model_counts"] = dict(Counter(_case_model_name(case) for case in cases))
    summary["reserve_operation_counts"] = dict(Counter(str(case.get("operation") or "<missing>") for case in reserve_cases))
    summary["reserve_model_counts"] = dict(Counter(_case_model_name(case) for case in reserve_cases))
    summary["qwen_sam3_rerun_candidate_count"] = len(rerun_candidates)
    summary["qwen_sam3_rerun_reason_counts"] = dict(Counter(str(item.get("reason") or "<missing>") for item in rerun_candidates))
    kept_cases = [_case_index_row(case) for case in cases]
    kept_video_ids = sorted({str(item["video_id"]) for item in kept_cases if item.get("video_id")})
    reserved_cases = [_case_index_row(case) for case in reserve_cases]
    reserved_video_ids = sorted({str(item["video_id"]) for item in reserved_cases if item.get("video_id")})

    payload = _plan_payload(
        schema_version="dataA_v1_frozen_subject_first_vace_continuation_plan_v1",
        track_bank=track_bank,
        base_plan=base_plan,
        path_mapping=path_mapping,
        selection_summary=summary,
        validation=validation,
        cases=cases,
    )
    payload["run_roots"] = [str(path) for path in run_roots]
    payload["completed_case_ids"] = sorted(completed_case_ids)
    payload["completed_video_ids"] = sorted(completed_video_ids)
    payload["reserved_same_video_case_ids"] = [str(case.get("case_id")) for case in reserve_cases if case.get("case_id")]

    vace14b_payload = _plan_payload(
        schema_version="dataA_v1_frozen_subject_first_vace14b_continuation_plan_v1",
        track_bank=track_bank,
        base_plan=base_plan,
        path_mapping=path_mapping,
        selection_summary=_summary(audit_rows, vace14b_cases),
        validation=vace14b_validation,
        cases=vace14b_cases,
    )
    vace13b_payload = _plan_payload(
        schema_version="dataA_v1_frozen_subject_first_vace13b_continuation_plan_v1",
        track_bank=track_bank,
        base_plan=base_plan,
        path_mapping=path_mapping,
        selection_summary=_summary(audit_rows, vace13b_cases),
        validation=vace13b_validation,
        cases=vace13b_cases,
    )
    reserve_payload = _plan_payload(
        schema_version="dataA_v1_frozen_subject_first_vace_continuation_reserve_plan_v1",
        track_bank=track_bank,
        base_plan=base_plan,
        path_mapping=path_mapping,
        selection_summary={**summary, "plan_role": "same_video_reserve"},
        validation=reserve_validation,
        cases=reserve_cases,
    )
    reserve_payload["run_roots"] = [str(path) for path in run_roots]
    reserve_vace14b_payload = _plan_payload(
        schema_version="dataA_v1_frozen_subject_first_vace14b_continuation_reserve_plan_v1",
        track_bank=track_bank,
        base_plan=base_plan,
        path_mapping=path_mapping,
        selection_summary={**_summary(audit_rows, reserve_vace14b_cases), "plan_role": "same_video_reserve"},
        validation=reserve_vace14b_validation,
        cases=reserve_vace14b_cases,
    )
    reserve_vace13b_payload = _plan_payload(
        schema_version="dataA_v1_frozen_subject_first_vace13b_continuation_reserve_plan_v1",
        track_bank=track_bank,
        base_plan=base_plan,
        path_mapping=path_mapping,
        selection_summary={**_summary(audit_rows, reserve_vace13b_cases), "plan_role": "same_video_reserve"},
        validation=reserve_vace13b_validation,
        cases=reserve_vace13b_cases,
    )
    audit = {
        "schema_version": "dataA_v1_subject_first_continuation_audit_v1",
        "generated_at_utc": utc_now_iso(),
        "track_bank": str(track_bank),
        "base_plan": str(base_plan),
        "run_roots": [str(path) for path in run_roots],
        "summary": summary,
        "validation": validation,
        "vace14b_validation": vace14b_validation,
        "vace13b_validation": vace13b_validation,
        "reserve_validation": reserve_validation,
        "reserve_vace14b_validation": reserve_vace14b_validation,
        "reserve_vace13b_validation": reserve_vace13b_validation,
        "skipped": skipped,
        "completed_skips": completed_skips,
        "same_video_reserve_skips": reserve_skips,
        "qwen_sam3_rerun_candidates": rerun_candidates,
        "kept_cases": kept_cases,
        "reserved_same_video_cases": reserved_cases,
        "completed": [completed_case.__dict__ for completed_case in completed.values()],
        "candidate_count": len(audit_rows),
    }
    rerun_manifest = {
        "schema_version": "dataA_v1_qwen_sam3_rerun_manifest_v1",
        "generated_at_utc": utc_now_iso(),
        "track_bank": str(track_bank),
        "base_plan": str(base_plan),
        "run_roots": [str(path) for path in run_roots],
        "policy": {
            "completed_cases_are_excluded": True,
            "keep_if_person_swap": True,
            "keep_if_good_pair_and_mask_not_small": True,
            "rerun_other_unfinished_cases": True,
            "same_video_conflicts_go_to_reserve_plan": True,
            "rerun_scope": "unfinished_cases_only",
        },
        "summary": {
            "rerun_candidate_count": len(rerun_candidates),
            "rerun_video_count": len({str(item.get("video_id")) for item in rerun_candidates if item.get("video_id")}),
            "kept_case_count": len(kept_cases),
            "kept_video_count": len(kept_video_ids),
            "reserved_same_video_case_count": len(reserved_cases),
            "reserved_same_video_count": len(reserved_video_ids),
            "completed_case_count": len(completed_case_ids),
            "completed_video_count": len(completed_video_ids),
            "reason_counts": dict(Counter(str(item.get("reason") or "<missing>") for item in rerun_candidates)),
        },
        "completed_case_ids": sorted(completed_case_ids),
        "completed_video_ids": sorted(completed_video_ids),
        "kept_cases": kept_cases,
        "kept_video_ids": kept_video_ids,
        "reserved_same_video_cases": reserved_cases,
        "reserved_same_video_ids": reserved_video_ids,
        "rerun_candidates": rerun_candidates,
        "rerun_video_ids": sorted({str(item.get("video_id")) for item in rerun_candidates if item.get("video_id")}),
    }
    if not dry_run:
        write_json(out_plan, payload)
        write_json(out_vace14b_plan, vace14b_payload)
        write_json(out_vace13b_plan, vace13b_payload)
        write_json(out_reserve_plan, reserve_payload)
        write_json(out_vace14b_reserve_plan, reserve_vace14b_payload)
        write_json(out_vace13b_reserve_plan, reserve_vace13b_payload)
        write_json(out_audit, audit)
        write_json(out_rerun_manifest, rerun_manifest)
    return {
        "dry_run": dry_run,
        "out_plan": str(out_plan),
        "out_vace14b_plan": str(out_vace14b_plan),
        "out_vace13b_plan": str(out_vace13b_plan),
        "out_reserve_plan": str(out_reserve_plan),
        "out_vace14b_reserve_plan": str(out_vace14b_reserve_plan),
        "out_vace13b_reserve_plan": str(out_vace13b_reserve_plan),
        "out_audit": str(out_audit),
        "out_rerun_manifest": str(out_rerun_manifest),
        "summary": summary,
        "validation": validation,
        "vace14b_validation": vace14b_validation,
        "vace13b_validation": vace13b_validation,
        "reserve_validation": reserve_validation,
        "reserve_vace14b_validation": reserve_vace14b_validation,
        "reserve_vace13b_validation": reserve_vace13b_validation,
        "skipped_count": len(skipped),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a continuation Data A v1 VACE plan from unfinished cases.")
    parser.add_argument("--track-bank", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, action="append", required=True, help="Existing run root to preserve; repeatable.")
    parser.add_argument("--selection-config", type=Path, default=Path("configs/dataa_v1/subject_selection_v1.json"))
    parser.add_argument("--path-mapping", type=Path, default=None)
    parser.add_argument("--out-plan", type=Path, default=DEFAULT_OUT_PLAN)
    parser.add_argument("--out-vace14b-plan", type=Path, default=DEFAULT_OUT_VACE14B_PLAN)
    parser.add_argument("--out-vace13b-plan", type=Path, default=DEFAULT_OUT_VACE13B_PLAN)
    parser.add_argument("--out-reserve-plan", type=Path, default=DEFAULT_OUT_RESERVE_PLAN)
    parser.add_argument("--out-vace14b-reserve-plan", type=Path, default=DEFAULT_OUT_VACE14B_RESERVE_PLAN)
    parser.add_argument("--out-vace13b-reserve-plan", type=Path, default=DEFAULT_OUT_VACE13B_RESERVE_PLAN)
    parser.add_argument("--out-audit", type=Path, default=DEFAULT_OUT_AUDIT)
    parser.add_argument("--out-rerun-manifest", type=Path, default=DEFAULT_OUT_RERUN_MANIFEST)
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_continuation_plan(
            track_bank=_resolve_project_path(args.track_bank) or args.track_bank,
            base_plan=_resolve_project_path(args.base_plan) or args.base_plan,
            run_roots=[_resolve_project_path(path) or path for path in args.run_root],
            selection_config=_resolve_project_path(args.selection_config),
            path_mapping=_resolve_project_path(args.path_mapping),
            out_plan=_resolve_project_path(args.out_plan) or args.out_plan,
            out_vace14b_plan=_resolve_project_path(args.out_vace14b_plan) or args.out_vace14b_plan,
            out_vace13b_plan=_resolve_project_path(args.out_vace13b_plan) or args.out_vace13b_plan,
            out_reserve_plan=_resolve_project_path(args.out_reserve_plan) or args.out_reserve_plan,
            out_vace14b_reserve_plan=_resolve_project_path(args.out_vace14b_reserve_plan) or args.out_vace14b_reserve_plan,
            out_vace13b_reserve_plan=_resolve_project_path(args.out_vace13b_reserve_plan) or args.out_vace13b_reserve_plan,
            out_audit=_resolve_project_path(args.out_audit) or args.out_audit,
            out_rerun_manifest=_resolve_project_path(args.out_rerun_manifest) or args.out_rerun_manifest,
            ffprobe_bin=args.ffprobe_bin,
            num_workers=max(1, int(args.num_workers)),
            progress_every=max(0, int(args.progress_every)),
            dry_run=bool(args.dry_run),
        )
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        "continuation_plan "
        f"dry_run={summary['dry_run']} "
        f"cases={summary['validation']['case_count']} "
        f"vace14b_cases={summary['vace14b_validation']['case_count']} "
        f"vace13b_cases={summary['vace13b_validation']['case_count']} "
        f"reserve_cases={summary['reserve_validation']['case_count']} "
        f"completed_cases={summary['summary']['completed_case_count']} "
        f"completed_videos={summary['summary']['completed_video_count']} "
        f"skipped={summary['skipped_count']} "
        f"rerun_candidates={summary['summary']['qwen_sam3_rerun_candidate_count']} "
        f"valid={summary['validation']['valid']} "
        f"reserve_valid={summary['reserve_validation']['valid']}"
    )
    if not summary["validation"]["valid"] or not summary["reserve_validation"]["valid"]:
        print(f"validation_errors={summary['validation']['errors']}", file=sys.stderr)
        print(f"reserve_validation_errors={summary['reserve_validation']['errors']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
