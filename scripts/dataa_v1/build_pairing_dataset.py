#!/usr/bin/env python3
"""Build a materialized Data A v1 pairing dataset from inventory-v2 SAM3 tracks."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.common import DataAError, read_json, utc_now_iso, write_json
from scripts.dataa_v1.donor_reference import export_donor_reference_from_video
from scripts.dataa_v1.mask_io import load_mask_tube, save_mask_npz
from scripts.dataa_v1.mask_processing import MaskProcessingConfig, apply_effective_mask_policy, process_masks
from scripts.dataa_v1.mask_video import write_mask_video_ffmpeg
from scripts.dataa_v1.media_io import ffprobe_video
from scripts.dataa_v1.schema import TRACK_LIST_KEYS, as_records


DEFAULT_INVENTORY = Path("res/qwen_inventory_v2/qwen_inventory_v2_normalized.json")
DEFAULT_TRACK_BANK = Path("res/sam_track_bank/inventory_v2/sam3_quality_tracks.json")
DEFAULT_TAXONOMY = Path("configs/dataa_v1/taxonomy_v2_seed.json")
DEFAULT_COMPATIBILITY = Path("configs/dataa_v1/compatibility_matrix_v2.json")
DEFAULT_DATASET_ROOT = Path("/tmp/camerabenchtrain/dataset")
DEFAULT_OUT_INDEX = Path("res/dataA_v1/dataset_v2/pairing_dataset_index.json")
DEFAULT_OUT_AUDIT = Path("res/dataA_v1/audits/pairing_dataset_v2_audit.json")
REFERENCE_ROUTE = "vace14b_masktrack_reference_swap"
TEXT_ROUTE = "vace14b_masktrack_text_edit"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_project_path(path: Path | str | None) -> Path | None:
    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else _repo_root() / value


def _clean(value: Any, *, max_len: int = 200) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())[:max_len]


def _optional_path(value: Any) -> Path | None:
    text = _clean(value, max_len=1000)
    return Path(text) if text else None


def _load_inventory_entities(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = read_json(path)
    entities: list[dict[str, Any]] = []
    if isinstance(payload, Mapping) and isinstance(payload.get("entities"), list):
        entities.extend(dict(item) for item in payload["entities"] if isinstance(item, Mapping))
    elif isinstance(payload, Mapping) and isinstance(payload.get("videos"), list):
        for video in payload["videos"]:
            if not isinstance(video, Mapping):
                continue
            video_id = _clean(video.get("video_id"))
            for entity in video.get("entities") or []:
                if isinstance(entity, Mapping):
                    item = dict(entity)
                    item.setdefault("video_id", video_id)
                    item.setdefault("video_path", video.get("video_path"))
                    item.setdefault("relative_path", video.get("relative_path"))
                    entities.append(item)
    else:
        raise DataAError("inventory JSON must contain videos[] or entities[]")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for entity in entities:
        video_id = _clean(entity.get("video_id"))
        entity_id = _clean(entity.get("entity_id"))
        if video_id and entity_id:
            by_key[(video_id, entity_id)] = entity
    return by_key


def _track_id(track: Mapping[str, Any]) -> str:
    return _clean(track.get("track_id"), max_len=300)


def _video_id(track: Mapping[str, Any]) -> str:
    return _clean(track.get("video_id") or track.get("source_video_id"), max_len=200)


def _candidate_id(track: Mapping[str, Any]) -> str:
    return _clean(track.get("candidate_id"), max_len=200)


def _mask_path(track: Mapping[str, Any]) -> Path | None:
    for key in ("mask_tube_path", "mask_path", "mask_npz_path", "npz_path"):
        path = _optional_path(track.get(key))
        if path is not None:
            return path
    return None


def _video_path(track: Mapping[str, Any]) -> Path | None:
    return _optional_path(track.get("video_path") or track.get("source_video_path"))


def _entity_for_track(track: Mapping[str, Any], entities: Mapping[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    return dict(entities.get((_video_id(track), _candidate_id(track))) or {})


def _taxonomy_label(track: Mapping[str, Any], entity: Mapping[str, Any]) -> str:
    return _clean(entity.get("taxonomy_label"), max_len=120).lower() or _clean(track.get("taxonomy_label"), max_len=120).lower() or "unknown"


def _group_for_label(label: str, compatibility: Mapping[str, Any]) -> str | None:
    groups = compatibility.get("groups") or {}
    for group_name, group in groups.items():
        if not isinstance(group, Mapping):
            continue
        for prefix in group.get("label_prefixes") or []:
            if label.startswith(str(prefix)):
                return str(group_name)
    return None


def _hard_block_reason(target_label: str, donor_label: str, compatibility: Mapping[str, Any]) -> str | None:
    for rule in compatibility.get("hard_blocks") or []:
        if not isinstance(rule, Mapping):
            continue
        target_prefix = str(rule.get("target_prefix") or "")
        donor_prefix = str(rule.get("donor_prefix") or "")
        if target_prefix and not target_label.startswith(target_prefix):
            continue
        if donor_prefix and not donor_label.startswith(donor_prefix):
            continue
        return str(rule.get("reason") or "hard_block")
    return None


def _compatible(target_label: str, donor_label: str, compatibility: Mapping[str, Any]) -> tuple[bool, str]:
    blocked = _hard_block_reason(target_label, donor_label, compatibility)
    if blocked:
        return False, blocked
    target_group = _group_for_label(target_label, compatibility)
    donor_group = _group_for_label(donor_label, compatibility)
    if not target_group or not donor_group:
        return False, "missing_compatibility_group"
    group = (compatibility.get("groups") or {}).get(target_group) or {}
    allowed = set(str(item) for item in group.get("allowed_with") or [])
    if donor_group == target_group or donor_group in allowed:
        return True, "compatible"
    return False, f"group_mismatch:{target_group}!={donor_group}"


def _operation_for_group(group_name: str | None, compatibility: Mapping[str, Any]) -> tuple[str, str, bool]:
    group = (compatibility.get("groups") or {}).get(str(group_name)) or {}
    operation = str(group.get("preferred_operation") or "object_swap")
    reference_required = bool(group.get("reference_required", operation not in {"surface_attribute_edit", "object_attribute_edit"}))
    route = REFERENCE_ROUTE if reference_required else TEXT_ROUTE
    return operation, route, reference_required


def _policy_for_target(
    *,
    case_id: str,
    group_name: str | None,
    entity: Mapping[str, Any],
    compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    groups = compatibility.get("groups") or {}
    group = groups.get(str(group_name)) or {}
    policy_name = str(group.get("mask_policy") or "similar_object")
    size_level = _clean(entity.get("size_level")).lower()
    if policy_name == "similar_object" and size_level in {"small", "tiny"}:
        policy_name = "bbox_for_small_or_irregular"
    policy = dict((compatibility.get("mask_policies") or {}).get(policy_name) or {})
    if not policy:
        policy = {"variant_type": "sam3_shape", "person_bbox_disabled": False}
    policy.update(
        {
            "schema_version": "dataA_v1_mask_policy_v2",
            "policy_name": policy_name,
            "case_id": case_id,
            "trigger_reason": f"dataset_v2_group={group_name};size={size_level or 'unknown'}",
        }
    )
    return policy


def _track_score(track: Mapping[str, Any], entity: Mapping[str, Any], label: str) -> tuple[int, float, float, str]:
    salience = _clean(entity.get("salience")).lower()
    foreground = _clean(entity.get("foreground_status")).lower()
    size = _clean(entity.get("size_level")).lower()
    suitability = _clean(entity.get("edit_suitability")).lower()
    person_bonus = 100 if label.startswith("person.real.") and size != "tiny" else 0
    salience_bonus = {"primary": 30, "secondary": 15, "background": -20}.get(salience, 0)
    foreground_bonus = {"foreground": 20, "midground": 8, "background": -20}.get(foreground, 0)
    size_bonus = {"large": 20, "medium": 12, "small": 0, "tiny": -30}.get(size, 0)
    suitability_bonus = {"good": 15, "maybe": 5, "bad": -40}.get(suitability, 0)
    median_area = float(track.get("median_area_ratio") or track.get("mean_area_ratio") or 0.0)
    quality = float(track.get("track_quality_score") or 0.0)
    return person_bonus + salience_bonus + foreground_bonus + size_bonus + suitability_bonus, median_area, quality, _track_id(track)


def _donor_score(track: Mapping[str, Any], entity: Mapping[str, Any]) -> tuple[int, float, float, str]:
    suitability = _clean(entity.get("donor_suitability")).lower()
    visibility = _clean(entity.get("visibility")).lower()
    size = _clean(entity.get("size_level")).lower()
    score = {"good": 40, "maybe": 10, "bad": -100}.get(suitability, 0)
    score += {"complete": 20, "partial": 5, "occluded": -20, "truncated": -30}.get(visibility, 0)
    score += {"large": 20, "medium": 12, "small": 0, "tiny": -40}.get(size, 0)
    median_area = float(track.get("median_area_ratio") or track.get("mean_area_ratio") or 0.0)
    quality = float(track.get("track_quality_score") or 0.0)
    return score, median_area, quality, _track_id(track)


def _copy_or_link(src: Path, dst: Path, *, prefer_hardlink: bool) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return {"path": str(dst), "mode": "already_exists"}
    if prefer_hardlink:
        try:
            os.link(src, dst)
            return {"path": str(dst), "mode": "hardlink"}
        except OSError:
            pass
    shutil.copy2(src, dst)
    return {"path": str(dst), "mode": "copy"}


def _materialize_masks(
    *,
    pair_dir: Path,
    target_track: Mapping[str, Any],
    donor_track: Mapping[str, Any] | None,
    mask_policy: Mapping[str, Any],
    ffmpeg_bin: str,
    source_fps: float,
) -> dict[str, Any]:
    target_mask_path = _mask_path(target_track)
    if target_mask_path is None:
        raise DataAError(f"target_missing_mask_path:{_track_id(target_track)}")
    target_tube = load_mask_tube(target_mask_path)
    raw_path = pair_dir / "target_mask_raw.npz"
    effective_path = pair_dir / "target_mask_effective.npz"
    _copy_or_link(target_mask_path, raw_path, prefer_hardlink=True)
    masks, base_params = process_masks(target_tube.masks, MaskProcessingConfig())
    masks, effective_params = apply_effective_mask_policy(masks, mask_policy, MaskProcessingConfig())
    save_mask_npz(effective_path, masks["M_gen"], frame_indices=target_tube.frame_indices, kind="target_effective")
    vis_path = pair_dir / "target_mask_vis.mp4"
    mask_video = write_mask_video_ffmpeg(vis_path, masks["M_gen"], fps=source_fps, ffmpeg_bin=ffmpeg_bin)
    donor_mask_result = None
    if donor_track is not None:
        donor_mask = _mask_path(donor_track)
        if donor_mask is None:
            raise DataAError(f"donor_missing_mask_path:{_track_id(donor_track)}")
        donor_mask_result = _copy_or_link(donor_mask, pair_dir / "donor_mask_raw.npz", prefer_hardlink=True)
    return {
        "target_mask_raw": str(raw_path),
        "target_mask_effective": str(effective_path),
        "target_mask_vis": str(vis_path),
        "target_mask_video": mask_video,
        "donor_mask_raw": donor_mask_result,
        "mask_processing": {"base": base_params, "effective": effective_params},
    }


def _materialize_pair_assets(
    *,
    pair: Mapping[str, Any],
    pair_dir: Path,
    source_video_path: Path,
    source_video_dataset: Path,
    reference_path: Path,
    reference_alpha_path: Path,
    selected: Mapping[str, Any],
    donor: Mapping[str, Any] | None,
    mask_policy: Mapping[str, Any],
    prefer_hardlink: bool,
    ffmpeg_bin: str,
    ffprobe_bin: str,
) -> dict[str, Any]:
    source_link = _copy_or_link(source_video_path, source_video_dataset, prefer_hardlink=prefer_hardlink)
    video_meta = ffprobe_video(source_video_path, ffprobe_bin=ffprobe_bin)
    materialized: dict[str, Any] = {
        "status": "materialized",
        "source_video": source_link,
        "source_video_meta": {
            "fps": video_meta.fps,
            "frame_count": video_meta.frame_count,
            "height": video_meta.height,
            "width": video_meta.width,
            "duration": video_meta.duration,
        },
    }
    materialized["masks"] = _materialize_masks(
        pair_dir=pair_dir,
        target_track=selected,
        donor_track=donor,
        mask_policy=mask_policy,
        ffmpeg_bin=ffmpeg_bin,
        source_fps=video_meta.fps,
    )
    if donor is not None:
        donor_video = _video_path(donor)
        donor_mask = _mask_path(donor)
        if donor_video is None or donor_mask is None:
            raise DataAError(f"blocked_donor_reference_missing_input:{_track_id(donor)}")
        donor_ref = export_donor_reference_from_video(
            out_dir=pair_dir,
            tube=load_mask_tube(donor_mask),
            donor_video_path=str(donor_video),
            ffmpeg_bin=ffmpeg_bin,
        )
        os.replace(pair_dir / "donor_reference.png", reference_path)
        os.replace(pair_dir / "donor_reference_alpha.png", reference_alpha_path)
        donor_ref["donor_reference"] = str(reference_path)
        donor_ref["donor_reference_alpha"] = str(reference_alpha_path)
        materialized["donor_reference"] = donor_ref
    output = dict(pair)
    output["materialization"] = materialized
    write_json(pair_dir / "pair_meta.json", output)
    write_json(pair_dir / "vace_case_spec.json", output)
    return output


def _completed_usage(run_roots: Sequence[Path]) -> dict[str, Any]:
    used_target_videos: set[str] = set()
    used_donor_videos: Counter[str] = Counter()
    used_donor_tracks: set[str] = set()
    completed_cases: list[str] = []
    for root in run_roots:
        if not root.is_dir():
            continue
        for manifest_path in root.rglob("case_manifest.json"):
            pair_dir = manifest_path.parent
            if not ((pair_dir / "full_fake.mp4").is_file() or (pair_dir / "generated_raw.mp4").is_file()):
                continue
            try:
                manifest = read_json(manifest_path)
            except DataAError:
                continue
            case_id = _clean(manifest.get("case_id") or pair_dir.name)
            target = manifest.get("target") or {}
            donor = manifest.get("donor") or {}
            target_video = _clean(target.get("video_id") if isinstance(target, Mapping) else "")
            donor_video = _clean(donor.get("video_id") if isinstance(donor, Mapping) else "")
            donor_track = _clean(donor.get("track_id") if isinstance(donor, Mapping) else "")
            if target_video:
                used_target_videos.add(target_video)
            if donor_video:
                used_donor_videos[donor_video] += 1
            if donor_track:
                used_donor_tracks.add(donor_track)
            if case_id:
                completed_cases.append(case_id)
    return {
        "used_target_videos": used_target_videos,
        "used_donor_videos": used_donor_videos,
        "used_donor_tracks": used_donor_tracks,
        "completed_cases": completed_cases,
    }


def build_pairing_dataset(
    *,
    inventory: Path,
    track_bank: Path,
    taxonomy: Path,
    compatibility_path: Path,
    dataset_root: Path,
    out_index: Path,
    out_audit: Path,
    completed_run_roots: Sequence[Path] = (),
    execute: bool = False,
    prefer_hardlink: bool = True,
    max_pairs: int | None = None,
    num_workers: int = 1,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> dict[str, Any]:
    if num_workers < 1:
        raise DataAError("--num-workers must be >= 1")
    entities = _load_inventory_entities(inventory)
    tracks = as_records(read_json(track_bank), TRACK_LIST_KEYS, "SAM3 track-bank")
    _ = read_json(taxonomy)
    compatibility = read_json(compatibility_path)
    limits = compatibility.get("limits") or {}
    target_video_max_use = int(limits.get("target_video_max_use") or 1)
    donor_track_max_use = int(limits.get("donor_track_max_use") or 1)
    donor_video_max_use = int(limits.get("donor_video_max_use") or 2)
    completed = _completed_usage(completed_run_roots)
    used_target_videos: Counter[str] = Counter({video_id: target_video_max_use for video_id in completed["used_target_videos"]})
    used_donor_videos: Counter[str] = Counter(completed["used_donor_videos"])
    used_donor_tracks: Counter[str] = Counter({track_id: donor_track_max_use for track_id in completed["used_donor_tracks"]})

    enriched_tracks: list[dict[str, Any]] = []
    skipped_tracks: Counter[str] = Counter()
    for track in tracks:
        video_id = _video_id(track)
        track_id = _track_id(track)
        if not video_id or not track_id:
            skipped_tracks["missing_video_or_track_id"] += 1
            continue
        mask_path = _mask_path(track)
        video_path = _video_path(track)
        if mask_path is None or video_path is None:
            skipped_tracks["missing_mask_or_video_path"] += 1
            continue
        entity = _entity_for_track(track, entities)
        label = _taxonomy_label(track, entity)
        group = _group_for_label(label, compatibility)
        if not group:
            skipped_tracks["missing_compatibility_group"] += 1
            continue
        item = dict(track)
        item["_entity"] = entity
        item["_taxonomy_label"] = label
        item["_compatibility_group"] = group
        enriched_tracks.append(item)

    tracks_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    donors_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track in enriched_tracks:
        tracks_by_video[_video_id(track)].append(track)
        donors_by_group[track["_compatibility_group"]].append(track)

    pairs: list[dict[str, Any]] = []
    materialize_jobs: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for video_id in sorted(tracks_by_video):
        if used_target_videos[video_id] >= target_video_max_use:
            audit_rows.append({"video_id": video_id, "status": "skipped_target_video_already_used"})
            continue
        candidates = sorted(
            tracks_by_video[video_id],
            key=lambda track: _track_score(track, track.get("_entity") or {}, track["_taxonomy_label"]),
            reverse=True,
        )
        if not candidates:
            continue
        selected = candidates[0]
        target_label = selected["_taxonomy_label"]
        target_group = selected["_compatibility_group"]
        operation, route, reference_required = _operation_for_group(target_group, compatibility)
        case_id = f"dataA_v1_dataset_v2_{len(pairs) + 1:06d}"
        donor: dict[str, Any] | None = None
        donor_match = {"reference_required": reference_required, "match_level": "not_required"}
        if reference_required:
            donor_candidates = []
            for maybe in enriched_tracks:
                donor_video = _video_id(maybe)
                donor_track = _track_id(maybe)
                if donor_video == video_id:
                    continue
                if used_donor_tracks[donor_track] >= donor_track_max_use:
                    continue
                if used_donor_videos[donor_video] >= donor_video_max_use:
                    continue
                ok, reason = _compatible(target_label, maybe["_taxonomy_label"], compatibility)
                if not ok:
                    continue
                donor_candidates.append((_donor_score(maybe, maybe.get("_entity") or {}), reason, maybe))
            if donor_candidates:
                _score, reason, donor = max(donor_candidates, key=lambda item: item[0])
                donor_match = {"reference_required": True, "match_level": reason}
            else:
                audit_rows.append(
                    {
                        "video_id": video_id,
                        "target_track_id": _track_id(selected),
                        "status": "blocked_no_compatible_donor",
                        "target_taxonomy_label": target_label,
                        "target_group": target_group,
                    }
                )
                continue
        mask_policy = _policy_for_target(case_id=case_id, group_name=target_group, entity=selected.get("_entity") or {}, compatibility=compatibility)
        operation_dir = dataset_root / "pairs" / operation
        pair_dir = operation_dir / f"{video_id}__{case_id}"
        source_video_path = _video_path(selected)
        if source_video_path is None:
            audit_rows.append({"video_id": video_id, "status": "blocked_missing_source_video_path"})
            continue
        target_mask_path = _mask_path(selected)
        if target_mask_path is None:
            audit_rows.append({"video_id": video_id, "status": "blocked_missing_target_mask_path"})
            continue
        materialized: dict[str, Any] = {"status": "planned"}
        source_video_dataset = pair_dir / "source_video.mp4"
        reference_path = pair_dir / "reference.png"
        reference_alpha_path = pair_dir / "reference_alpha.png"
        pair = {
            "case_id": case_id,
            "operation": operation,
            "generator_route": route,
            "pair_dir": str(pair_dir),
            "target": {
                "track_id": _track_id(selected),
                "candidate_id": _candidate_id(selected),
                "video_id": video_id,
                "video_path_original": str(source_video_path),
                "video_path": str(source_video_dataset if execute else source_video_path),
                "mask_tube_path_original": str(target_mask_path),
                "mask_tube_path": str(pair_dir / "target_mask_raw.npz") if execute else str(target_mask_path),
                "candidate_class": selected.get("candidate_class"),
                "canonical_concept": selected.get("canonical_concept"),
                "display_phrase": selected.get("display_phrase"),
                "region_family": selected.get("region_family"),
                "taxonomy_label": target_label,
                "compatibility_group": target_group,
                "inventory_entity": selected.get("_entity") or {},
            },
            "donor": None
            if donor is None
            else {
                "track_id": _track_id(donor),
                "candidate_id": _candidate_id(donor),
                "video_id": _video_id(donor),
                "video_path": str(_video_path(donor)),
                "mask_tube_path": str(_mask_path(donor)),
                "candidate_class": donor.get("candidate_class"),
                "canonical_concept": donor.get("canonical_concept"),
                "display_phrase": donor.get("display_phrase"),
                "region_family": donor.get("region_family"),
                "taxonomy_label": donor["_taxonomy_label"],
                "compatibility_group": donor["_compatibility_group"],
                "inventory_entity": donor.get("_entity") or {},
            },
            "reference_image_path": str(reference_path) if donor is not None else None,
            "reference_alpha_path": str(reference_alpha_path) if donor is not None else None,
            "mask_policy": mask_policy,
            "donor_match": donor_match,
            "materialization": materialized,
        }
        if execute:
            if num_workers == 1:
                pair = _materialize_pair_assets(
                    pair=pair,
                    pair_dir=pair_dir,
                    source_video_path=source_video_path,
                    source_video_dataset=source_video_dataset,
                    reference_path=reference_path,
                    reference_alpha_path=reference_alpha_path,
                    selected=selected,
                    donor=donor,
                    mask_policy=mask_policy,
                    prefer_hardlink=prefer_hardlink,
                    ffmpeg_bin=ffmpeg_bin,
                    ffprobe_bin=ffprobe_bin,
                )
            else:
                materialize_jobs.append(
                    {
                        "pair_index": len(pairs),
                        "pair": pair,
                        "pair_dir": pair_dir,
                        "source_video_path": source_video_path,
                        "source_video_dataset": source_video_dataset,
                        "reference_path": reference_path,
                        "reference_alpha_path": reference_alpha_path,
                        "selected": selected,
                        "donor": donor,
                        "mask_policy": mask_policy,
                    }
                )
        pairs.append(pair)
        used_target_videos[video_id] += 1
        if donor is not None:
            used_donor_tracks[_track_id(donor)] += 1
            used_donor_videos[_video_id(donor)] += 1
        if max_pairs is not None and len(pairs) >= max_pairs:
            break

    if execute and num_workers > 1 and materialize_jobs:
        print(f"pairing_dataset materialize workers={num_workers} jobs={len(materialize_jobs)}", flush=True)
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(
                    _materialize_pair_assets,
                    pair=job["pair"],
                    pair_dir=job["pair_dir"],
                    source_video_path=job["source_video_path"],
                    source_video_dataset=job["source_video_dataset"],
                    reference_path=job["reference_path"],
                    reference_alpha_path=job["reference_alpha_path"],
                    selected=job["selected"],
                    donor=job["donor"],
                    mask_policy=job["mask_policy"],
                    prefer_hardlink=prefer_hardlink,
                    ffmpeg_bin=ffmpeg_bin,
                    ffprobe_bin=ffprobe_bin,
                ): int(job["pair_index"])
                for job in materialize_jobs
            }
            for completed_count, future in enumerate(as_completed(futures), start=1):
                pair_index = futures[future]
                pairs[pair_index] = future.result()
                if completed_count == 1 or completed_count % 25 == 0 or completed_count == len(materialize_jobs):
                    print(
                        f"pairing_dataset materialized {completed_count}/{len(materialize_jobs)}",
                        flush=True,
                    )

    summary = {
        "execute": execute,
        "pair_count": len(pairs),
        "operation_counts": dict(Counter(pair["operation"] for pair in pairs)),
        "target_taxonomy_counts": dict(Counter(pair["target"]["taxonomy_label"] for pair in pairs).most_common()),
        "donor_taxonomy_counts": dict(Counter((pair.get("donor") or {}).get("taxonomy_label") or "<none>" for pair in pairs).most_common()),
        "skipped_track_counts": dict(skipped_tracks),
        "audit_status_counts": dict(Counter(row.get("status") for row in audit_rows)),
        "completed_case_count": len(completed["completed_cases"]),
        "completed_target_video_count": len(completed["used_target_videos"]),
        "completed_donor_track_count": len(completed["used_donor_tracks"]),
    }
    index = {
        "schema_version": "dataA_v1_pairing_dataset_v2",
        "generated_at_utc": utc_now_iso(),
        "dataset_root": str(dataset_root),
        "inventory": str(inventory),
        "track_bank": str(track_bank),
        "taxonomy": str(taxonomy),
        "compatibility_matrix": str(compatibility_path),
        "completed_run_roots": [str(path) for path in completed_run_roots],
        "summary": summary,
        "pairs": pairs,
        "ledgers": {
            "used_target_videos": dict(used_target_videos),
            "used_donor_videos": dict(used_donor_videos),
            "used_donor_tracks": dict(used_donor_tracks),
        },
    }
    audit = {
        "schema_version": "dataA_v1_pairing_dataset_v2_audit",
        "generated_at_utc": utc_now_iso(),
        "summary": summary,
        "audit_rows": audit_rows,
    }
    write_json(out_index, index)
    write_json(out_audit, audit)
    if execute:
        write_json(dataset_root / "ledgers" / "donor_usage.json", dict(used_donor_tracks))
        write_json(dataset_root / "ledgers" / "target_usage.json", dict(used_target_videos))
        write_json(dataset_root / "ledgers" / "video_role_usage.json", {"donor_videos": dict(used_donor_videos), "target_videos": dict(used_target_videos)})
    return index


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--track-bank", type=Path, default=DEFAULT_TRACK_BANK)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--compatibility", type=Path, default=DEFAULT_COMPATIBILITY)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-index", type=Path, default=DEFAULT_OUT_INDEX)
    parser.add_argument("--out-audit", type=Path, default=DEFAULT_OUT_AUDIT)
    parser.add_argument("--completed-run-root", type=Path, action="append", default=[])
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--copy-instead-of-hardlink", action="store_true")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        index = build_pairing_dataset(
            inventory=_resolve_project_path(args.inventory) or args.inventory,
            track_bank=_resolve_project_path(args.track_bank) or args.track_bank,
            taxonomy=_resolve_project_path(args.taxonomy) or args.taxonomy,
            compatibility_path=_resolve_project_path(args.compatibility) or args.compatibility,
            dataset_root=args.dataset_root,
            out_index=_resolve_project_path(args.out_index) or args.out_index,
            out_audit=_resolve_project_path(args.out_audit) or args.out_audit,
            completed_run_roots=[Path(path) for path in args.completed_run_root],
            execute=bool(args.execute),
            prefer_hardlink=not bool(args.copy_instead_of_hardlink),
            max_pairs=args.max_pairs,
            num_workers=int(args.num_workers),
            ffmpeg_bin=str(args.ffmpeg_bin),
            ffprobe_bin=str(args.ffprobe_bin),
        )
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    summary = index["summary"]
    print(
        "pairing_dataset "
        f"execute={summary['execute']} pairs={summary['pair_count']} "
        f"operations={summary['operation_counts']} dataset_root={index['dataset_root']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
