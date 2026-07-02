#!/usr/bin/env python3
"""Build a no-donor text-edit reserve VACE plan from unused inventory-v2 tracks."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.build_pairing_dataset import (
    _candidate_id,
    _completed_usage,
    _entity_for_track,
    _group_for_label,
    _load_inventory_entities,
    _mask_path,
    _policy_for_target,
    _taxonomy_label,
    _track_id,
    _video_id,
    _video_path,
)
from scripts.dataa_v1.common import DataAError, read_json, utc_now_iso, write_json
from scripts.dataa_v1.execution_plan import validate_execution_cases
from scripts.dataa_v1.path_resolver import PathResolver
from scripts.dataa_v1.schema import TRACK_LIST_KEYS, as_records, normalize_cases


DEFAULT_INVENTORY = Path("res/qwen_inventory_v2/qwen_inventory_entities.json")
DEFAULT_TRACK_BANK = Path("res/sam_track_bank/inventory_v2/parallel_runs/sam3_quality_tracks.json")
DEFAULT_COMPATIBILITY = Path("configs/dataa_v1/compatibility_matrix_v2.json")
DEFAULT_PAIRING_INDEX = Path("res/dataA_v1/dataset_v2/pairing_dataset_index.json")
DEFAULT_OUT_PLAN = Path("res/dataA_v1/plans/frozen_dataset_v2_textedit_reserve_vace13b_plan.json")
DEFAULT_OUT_AUDIT = Path("res/dataA_v1/audits/textedit_reserve_v2_audit.json")
TEXT_ROUTE = "vace14b_masktrack_text_edit"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_project_path(path: Path | str | None) -> Path | None:
    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else _repo_root() / value


def _clean(value: Any, *, max_len: int = 240) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())[:max_len]


def _used_targets_from_pairing_index(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    payload = read_json(path)
    used: set[str] = set()
    for pair in payload.get("pairs") or []:
        if not isinstance(pair, Mapping):
            continue
        target = pair.get("target") or {}
        if isinstance(target, Mapping):
            video_id = _clean(target.get("video_id"))
            if video_id:
                used.add(video_id)
    return used


def _target_name(track: Mapping[str, Any], entity: Mapping[str, Any], label: str) -> str:
    for value in (
        entity.get("display_phrase"),
        entity.get("sam3_prompt_phrase"),
        entity.get("fine_type_raw"),
        track.get("display_phrase"),
        track.get("canonical_concept"),
    ):
        text = _clean(value, max_len=120)
        if text:
            return text
    return label.replace(".", " ")


def _visual_domain(label: str, entity: Mapping[str, Any]) -> str:
    explicit = _clean(entity.get("visual_domain")).lower()
    if explicit:
        return explicit
    if label.startswith("person.cartoon"):
        return "cartoon"
    if label.startswith("person.3d_character"):
        return "3d"
    return "real"


def _operation_for_label(label: str) -> str:
    return "surface_attribute_edit" if label.startswith("surface.") else "object_attribute_edit"


def _model_prompt(label: str, target_name: str, entity: Mapping[str, Any]) -> str:
    visual_domain = _visual_domain(label, entity)
    if label.startswith("person."):
        person_style = {
            "cartoon": "cartoon character",
            "3d": "3D character",
            "3d_render": "3D character",
        }.get(visual_domain, "person")
        return (
            f"The masked {target_name} is regenerated as a visibly different {person_style} with changed appearance, "
            "clothing, hair, facial or body details, while preserving the original pose, action, timing, camera motion, "
            "lighting, and scene geometry."
        )
    if label.startswith("surface."):
        return (
            f"The masked {target_name} keeps the original plane, perspective, and motion, but its visible text, graphics, "
            "color, texture, or printed details are altered with locally generated imperfect AIGC artifacts."
        )
    if label.startswith("vehicle."):
        return (
            f"The masked {target_name} is regenerated as a visibly altered vehicle of the same broad type, with changed "
            "color, surface details, material, markings, or local shape while preserving motion and perspective."
        )
    if label.startswith("animal."):
        return (
            f"The masked {target_name} is regenerated with different visible appearance, fur, feather, skin, color, or "
            "surface detail while preserving its original pose, motion, and scene context."
        )
    return (
        f"The masked {target_name} is regenerated as a visibly altered object of the same broad kind, with changed "
        "color, material, texture, details, or local shape while the surrounding video remains unchanged."
    )


def _control_prompt() -> str:
    return (
        "Use only the text prompt and the target mask tube for this edit. Edit inside the mask and a reasonable boundary "
        "band only. Preserve all non-edit regions, camera motion, timing, lighting, shadows, and temporal continuity. "
        "No donor RGB, reference image, or external object crop is used."
    )


def _track_score(track: Mapping[str, Any], entity: Mapping[str, Any], label: str) -> tuple[int, float, float, str]:
    salience = _clean(entity.get("salience")).lower()
    foreground = _clean(entity.get("foreground_status")).lower()
    size = _clean(entity.get("size_level")).lower()
    suitability = _clean(entity.get("edit_suitability")).lower()
    visibility = _clean(entity.get("visibility")).lower()
    score = 0
    if label.startswith("person.real.") and size != "tiny":
        score += 220
    elif label.startswith("person.") and size != "tiny":
        score += 180
    score += {"primary": 45, "secondary": 20, "background": -30}.get(salience, 0)
    score += {"foreground": 30, "midground": 12, "background": -30}.get(foreground, 0)
    score += {"large": 30, "medium": 18, "small": 2, "tiny": -80}.get(size, 0)
    score += {"good": 20, "maybe": 5, "bad": -80}.get(suitability, 0)
    score += {"complete": 16, "partial": 4, "occluded": -20, "truncated": -30}.get(visibility, 0)
    median_area = float(track.get("median_area_ratio") or track.get("mean_area_ratio") or 0.0)
    quality = float(track.get("track_quality_score") or 0.0)
    return score, median_area, quality, _track_id(track)


def _target_record(track: Mapping[str, Any], entity: Mapping[str, Any], label: str, group: str) -> dict[str, Any]:
    return {
        "track_id": _track_id(track),
        "candidate_id": _candidate_id(track),
        "video_id": _video_id(track),
        "video_path": str(_video_path(track)),
        "mask_tube_path": str(_mask_path(track)),
        "candidate_class": track.get("candidate_class"),
        "canonical_concept": track.get("canonical_concept") or entity.get("fine_type_raw") or label,
        "display_phrase": track.get("display_phrase") or entity.get("display_phrase") or entity.get("sam3_prompt_phrase"),
        "region_family": track.get("region_family"),
        "content_domain": track.get("content_domain"),
        "style_domain": track.get("style_domain") or _visual_domain(label, entity),
        "taxonomy_label": label,
        "compatibility_group": group,
        "inventory_entity": dict(entity),
    }


def build_textedit_reserve_plan(
    *,
    inventory: Path,
    track_bank: Path,
    compatibility_path: Path,
    pairing_index: Path | None,
    completed_run_roots: Sequence[Path],
    out_plan: Path,
    out_audit: Path,
    model_name: str,
    profile: str,
    size: str,
    max_cases: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    entities = _load_inventory_entities(inventory)
    tracks = as_records(read_json(track_bank), TRACK_LIST_KEYS, "SAM3 track-bank")
    compatibility = read_json(compatibility_path)
    completed = _completed_usage(completed_run_roots)
    used_target_videos = set(completed["used_target_videos"]) | _used_targets_from_pairing_index(pairing_index)

    tracks_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped_tracks: Counter[str] = Counter()
    for track in tracks:
        video_id = _video_id(track)
        track_id = _track_id(track)
        if not video_id or not track_id:
            skipped_tracks["missing_video_or_track_id"] += 1
            continue
        if video_id in used_target_videos:
            skipped_tracks["target_video_already_used"] += 1
            continue
        if _mask_path(track) is None or _video_path(track) is None:
            skipped_tracks["missing_mask_or_video_path"] += 1
            continue
        entity = _entity_for_track(track, entities)
        label = _taxonomy_label(track, entity)
        group = _group_for_label(label, compatibility)
        if not group:
            skipped_tracks["missing_compatibility_group"] += 1
            continue
        edit_suitability = _clean(entity.get("edit_suitability")).lower()
        if edit_suitability == "bad":
            skipped_tracks["edit_suitability_bad"] += 1
            continue
        item = dict(track)
        item["_entity"] = entity
        item["_taxonomy_label"] = label
        item["_compatibility_group"] = group
        tracks_by_video[video_id].append(item)

    cases: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for video_id in sorted(tracks_by_video):
        candidates = sorted(
            tracks_by_video[video_id],
            key=lambda item: _track_score(item, item.get("_entity") or {}, item["_taxonomy_label"]),
            reverse=True,
        )
        if not candidates:
            continue
        selected = candidates[0]
        entity = selected.get("_entity") or {}
        label = selected["_taxonomy_label"]
        group = selected["_compatibility_group"]
        operation = _operation_for_label(label)
        case_id = f"dataA_v1_textedit_reserve_{len(cases) + 1:06d}"
        target_name = _target_name(selected, entity, label)
        mask_policy = _policy_for_target(case_id=case_id, group_name=group, entity=entity, compatibility=compatibility)
        text_edit_policy = {
            "schema_version": "dataA_v1_text_edit_policy_v1",
            "route": "text_driven_mask_edit",
            "prompt_source": "qwen_inventory_v2_fields_only",
            "target_name": target_name,
            "taxonomy_label": label,
            "visual_domain": _visual_domain(label, entity),
            "model_prompt": _model_prompt(label, target_name, entity),
            "control_prompt": _control_prompt(),
        }
        sampling_meta = {
            "schema_version": "dataA_v1_textedit_reserve_sampling_meta",
            "subject_first_source": "inventory_v2_textedit_reserve",
            "used_video_exclusion": {
                "completed_target_video_count": len(completed["used_target_videos"]),
                "paired_target_video_count": len(_used_targets_from_pairing_index(pairing_index)),
                "policy": "exclude completed VACE target videos and paired dataset target videos",
            },
            "taxonomy": {
                "target_label": label,
                "target_group": group,
                "donor_label": None,
                "donor_group": None,
            },
            "mask_policy": mask_policy,
            "text_edit_policy": text_edit_policy,
            "vace_model_plan": {
                "model_name": model_name,
                "profile": profile,
                "size": size,
                "route": "inventory_v2_textedit_reserve",
            },
        }
        if label.startswith("surface."):
            sampling_meta["artifact_policy"] = {
                "artifact_type": "surface_text_degradation",
                "policy_source": "inventory_v2_textedit_reserve",
                "description": "degrade fine text, graphics, markings, or printed surface details",
            }
        case = {
            "case_id": case_id,
            "operation": operation,
            "generator_route": TEXT_ROUTE,
            "target": _target_record(selected, entity, label, group),
            "donor": None,
            "sampling_meta": sampling_meta,
        }
        cases.append(case)
        audit_rows.append(
            {
                "video_id": video_id,
                "case_id": case_id,
                "status": "selected",
                "operation": operation,
                "target_track_id": _track_id(selected),
                "taxonomy_label": label,
                "score": _track_score(selected, entity, label)[:3],
                "candidate_count_for_video": len(candidates),
            }
        )
        if max_cases is not None and len(cases) >= max_cases:
            break

    normalized = normalize_cases({"cases": cases}, {"track_id": {}, "candidate_id": {}}, PathResolver({}), str(out_plan))
    validation = validate_execution_cases(normalized)
    summary = {
        "case_count": len(cases),
        "operation_counts": dict(Counter(case["operation"] for case in cases)),
        "target_taxonomy_counts": dict(Counter((case["target"] or {}).get("taxonomy_label") for case in cases).most_common()),
        "skipped_track_counts": dict(skipped_tracks),
        "used_completed_target_video_count": len(completed["used_target_videos"]),
        "used_pairing_target_video_count": len(_used_targets_from_pairing_index(pairing_index)),
        "input_track_count": len(tracks),
        "candidate_video_count": len(tracks_by_video),
    }
    payload = {
        "schema_version": "dataA_v1_frozen_textedit_reserve_vace_execution_plan",
        "generated_at_utc": utc_now_iso(),
        "inventory": str(inventory),
        "track_bank": str(track_bank),
        "compatibility_matrix": str(compatibility_path),
        "pairing_index": str(pairing_index) if pairing_index else None,
        "completed_run_roots": [str(path) for path in completed_run_roots],
        "model_name": model_name,
        "profile": profile,
        "size": size,
        "selection_summary": summary,
        "validation": validation,
        "cases": cases,
    }
    audit = {
        "schema_version": "dataA_v1_textedit_reserve_v2_audit",
        "generated_at_utc": utc_now_iso(),
        "summary": summary,
        "validation": validation,
        "audit_rows": audit_rows,
    }
    if not dry_run:
        write_json(out_plan, payload)
        write_json(out_audit, audit)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--track-bank", type=Path, default=DEFAULT_TRACK_BANK)
    parser.add_argument("--compatibility", type=Path, default=DEFAULT_COMPATIBILITY)
    parser.add_argument("--pairing-index", type=Path, default=DEFAULT_PAIRING_INDEX)
    parser.add_argument("--completed-run-root", type=Path, action="append", default=[])
    parser.add_argument("--out-plan", type=Path, default=DEFAULT_OUT_PLAN)
    parser.add_argument("--out-audit", type=Path, default=DEFAULT_OUT_AUDIT)
    parser.add_argument("--model-name", default="vace-1.3B")
    parser.add_argument("--profile", default="production_480")
    parser.add_argument("--size", default="480p")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_textedit_reserve_plan(
            inventory=_resolve_project_path(args.inventory) or args.inventory,
            track_bank=_resolve_project_path(args.track_bank) or args.track_bank,
            compatibility_path=_resolve_project_path(args.compatibility) or args.compatibility,
            pairing_index=_resolve_project_path(args.pairing_index) or args.pairing_index,
            completed_run_roots=[Path(path) for path in args.completed_run_root],
            out_plan=_resolve_project_path(args.out_plan) or args.out_plan,
            out_audit=_resolve_project_path(args.out_audit) or args.out_audit,
            model_name=str(args.model_name),
            profile=str(args.profile),
            size=str(args.size),
            max_cases=args.max_cases,
            dry_run=bool(args.dry_run),
        )
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    validation = payload["validation"]
    summary = payload["selection_summary"]
    print(
        "textedit_reserve_plan "
        f"dry_run={args.dry_run} cases={validation['case_count']} "
        f"operations={summary['operation_counts']} valid={validation['valid']} out={args.out_plan}"
    )
    if not validation["valid"]:
        print(f"validation_errors={validation['errors'][:20]}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
