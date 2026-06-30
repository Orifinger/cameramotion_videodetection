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
    _select_case_target,
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
DEFAULT_OUT_AUDIT = Path("res/dataA_v1/audits/subject_first_continuation_plan_audit.json")


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


def _eligible_person_targets(selection: Any, config: Mapping[str, Any]) -> list[Any]:
    candidates = [item for item in selection.candidates if item.metrics is not None and _is_person_track(item)]
    return [item for item in candidates if _operation_gate_report(item, "person_appearance_swap", config)["pass"]]


def _force_bbox_policy(case_payload: Dict[str, Any], *, reason: str, bbox_expand_ratio: float) -> None:
    meta = case_payload.setdefault("sampling_meta", {})
    policy = dict(meta.get("mask_policy") or {})
    if policy.get("person_bbox_disabled") or case_payload.get("operation") == "person_appearance_swap":
        return
    policy["variant_type"] = "expanded_bbox"
    policy["bbox_expand_ratio"] = float(bbox_expand_ratio)
    policy["dilation_radius_px"] = max(16, int(policy.get("dilation_radius_px") or 0))
    policy["trigger_reason"] = reason
    policy["shape_bias_mitigation"] = True
    meta["mask_policy"] = policy


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

    selected, operation, route, donor, quality_tier, risk_tags, donor_repair, operation_repair, target_repair = _select_case_target(
        case, selection, config, donor_pool
    )
    if selected is None:
        return None, {
            "case_id": case.case_id,
            "video_id": case.target.video_id,
            "reason": "no_continuation_target_or_donor",
            "person_preference_skip": person_skip,
        }
    case_payload = _case_from_template(
        case,
        selected,
        config,
        operation=operation,
        generator_route=route,
        donor=donor,
        quality_tier=quality_tier,
        risk_tags=[*risk_tags, "continuation_remaining_case"],
        donor_repair=donor_repair,
        operation_repair=operation_repair,
        target_repair=target_repair,
    )
    if case_payload is None:
        return None, {
            "case_id": case.case_id,
            "video_id": case.target.video_id,
            "reason": "continuation_case_build_failed",
            "person_preference_skip": person_skip,
        }
    _add_continuation_meta(case_payload, base_case=case, strategy="operation_aware_remaining")

    if operation != "person_appearance_swap" and quality_tier == "area_gate_fallback_largest":
        _force_bbox_policy(case_payload, reason="small_target_no_better_candidate", bbox_expand_ratio=1.35)
    if operation != "person_appearance_swap" and route == REFERENCE_ROUTE and donor is not None:
        target_kind = _track_kind(selected)
        donor_kind = _track_kind(donor)
        if target_kind != donor_kind:
            _force_bbox_policy(case_payload, reason="target_donor_shape_mismatch", bbox_expand_ratio=1.25)
    return case_payload, None


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
    out_audit: Path = DEFAULT_OUT_AUDIT,
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
    skipped: list[Dict[str, Any]] = []
    used_videos: set[str] = set()
    for case in base_cases:
        video_id = case.target.video_id
        if case.case_id in completed_case_ids:
            skipped.append({
                "case_id": case.case_id,
                "video_id": video_id,
                "reason": "already_completed_case",
                "source": completed[case.case_id].source,
            })
            continue
        if video_id and video_id in completed_video_ids:
            skipped.append({"case_id": case.case_id, "video_id": video_id, "reason": "already_completed_video"})
            continue
        if video_id and video_id in used_videos:
            skipped.append({"case_id": case.case_id, "video_id": video_id, "reason": "continuation_target_video_already_used"})
            continue
        selection = selections.get(str(video_id)) if video_id else None
        if selection is None:
            skipped.append({"case_id": case.case_id, "video_id": video_id, "reason": "no_track_bank_selection_for_video"})
            continue
        new_case, skip = _continuation_case_from_base(case=case, selection=selection, config=config, donor_pool=donor_pool)
        if new_case is None:
            skipped.append(skip or {"case_id": case.case_id, "video_id": video_id, "reason": "continuation_case_failed"})
            continue
        new_video_id = str((new_case.get("target") or {}).get("video_id") or video_id)
        if new_video_id in used_videos or new_video_id in completed_video_ids:
            skipped.append({"case_id": case.case_id, "video_id": new_video_id, "reason": "continuation_repaired_video_already_used"})
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
    audit_rows = [audit_record(item) for item in evaluated]
    summary = _summary(audit_rows, cases)
    summary["completed_case_count"] = len(completed_case_ids)
    summary["completed_video_count"] = len(completed_video_ids)
    summary["continuation_operation_counts"] = dict(Counter(str(case.get("operation") or "<missing>") for case in cases))
    summary["continuation_model_counts"] = dict(Counter(_case_model_name(case) for case in cases))

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
        "skipped": skipped,
        "completed": [completed_case.__dict__ for completed_case in completed.values()],
        "candidate_count": len(audit_rows),
    }
    if not dry_run:
        write_json(out_plan, payload)
        write_json(out_vace14b_plan, vace14b_payload)
        write_json(out_vace13b_plan, vace13b_payload)
        write_json(out_audit, audit)
    return {
        "dry_run": dry_run,
        "out_plan": str(out_plan),
        "out_vace14b_plan": str(out_vace14b_plan),
        "out_vace13b_plan": str(out_vace13b_plan),
        "out_audit": str(out_audit),
        "summary": summary,
        "validation": validation,
        "vace14b_validation": vace14b_validation,
        "vace13b_validation": vace13b_validation,
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
    parser.add_argument("--out-audit", type=Path, default=DEFAULT_OUT_AUDIT)
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
            out_audit=_resolve_project_path(args.out_audit) or args.out_audit,
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
        f"completed_cases={summary['summary']['completed_case_count']} "
        f"completed_videos={summary['summary']['completed_video_count']} "
        f"skipped={summary['skipped_count']} "
        f"valid={summary['validation']['valid']}"
    )
    if not summary["validation"]["valid"]:
        print(f"validation_errors={summary['validation']['errors']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
