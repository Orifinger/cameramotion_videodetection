#!/usr/bin/env python3
"""Package one frozen Data A v1 case for VACE Stage P.

Default mode is dry-run: it writes JSON specs only and never calls VACE.
Synthetic tests can enable media creation for mask-video round-trip validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.audit_vace_stage1 import audit_case, load_cases_for_audit
from scripts.dataa_v1.canonical_video import canonical_video_plan, export_canonical_videos
from scripts.dataa_v1.clip_selection import VaceProfile, profile_from_name, select_clip
from scripts.dataa_v1.common import DataAError, utc_now_iso, write_json
from scripts.dataa_v1.donor_reference import export_donor_reference_from_video, export_synthetic_donor_reference
from scripts.dataa_v1.execution_plan import load_execution_plan
from scripts.dataa_v1.manifest import build_case_manifest, write_manifest
from scripts.dataa_v1.mask_io import align_masks_to_canonical, load_mask_tube, save_mask_npz
from scripts.dataa_v1.mask_processing import MaskProcessingConfig, apply_effective_mask_policy, process_masks
from scripts.dataa_v1.mask_video import validate_mask_video_roundtrip, write_mask_video, write_mask_video_ffmpeg
from scripts.dataa_v1.media_io import ffprobe_video
from scripts.dataa_v1.mask_visualization import planned_mask_visualizations
from scripts.dataa_v1.prompt_builder import build_prompts
from scripts.dataa_v1.schema import serialize_case
from scripts.dataa_v1.vace_command import build_vace_command_spec
from scripts.dataa_v1.vace_condition import export_vace_condition_video, resize_masks_nearest


def _find_case(cases: list[Any], case_id: str) -> Any:
    for case in cases:
        if case.case_id == case_id:
            return case
    raise DataAError(f"case_id not found in normalized plan: {case_id}")


def _load_cases(plan: Path, track_bank: Optional[Path], path_mapping: Optional[Path]) -> list[Any]:
    if track_bank is not None:
        return load_cases_for_audit(plan, track_bank, path_mapping)
    execution_plan = load_execution_plan(execution_plan_path=plan, track_bank_path=None, path_mapping_path=path_mapping)
    return execution_plan.cases


def _video_shape(meta: Any) -> dict[str, Any]:
    return {"frame_count": meta.frame_count, "fps": meta.fps, "height": meta.height, "width": meta.width}


def _validate_vace_input_media(
    *,
    source_clip_path: Path,
    target_mask_gen_path: Path,
    source_vace_condition_path: Path,
    ffprobe_bin: str,
) -> dict[str, Any]:
    metas = {
        "source_clip": ffprobe_video(source_clip_path, ffprobe_bin=ffprobe_bin),
        "target_mask_gen": ffprobe_video(target_mask_gen_path, ffprobe_bin=ffprobe_bin),
        "source_vace_condition": ffprobe_video(source_vace_condition_path, ffprobe_bin=ffprobe_bin),
    }
    shapes = {label: _video_shape(meta) for label, meta in metas.items()}
    signatures = {
        label: (shape["frame_count"], round(float(shape["fps"]), 6), shape["height"], shape["width"])
        for label, shape in shapes.items()
    }
    if len(set(signatures.values())) != 1:
        raise DataAError(
            "blocked_vace_input_shape_mismatch: "
            f"source_clip={shapes['source_clip']} "
            f"target_mask_gen={shapes['target_mask_gen']} "
            f"source_vace_condition={shapes['source_vace_condition']}"
        )
    return {"status": "ok", "shapes": shapes}


def package_case(
    *,
    plan: Path,
    track_bank: Optional[Path],
    case_id: str,
    out_dir: Path,
    path_mapping: Optional[Path] = None,
    source_fps: float = 30.0,
    profile_name: str = "production_720",
    dry_run: bool = True,
    synthetic_media: bool = False,
    execute_media: bool = False,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> dict[str, Any]:
    cases = _load_cases(plan, track_bank, path_mapping)
    case = _find_case(cases, case_id)
    preflight = audit_case(case)
    attempt_dir = out_dir / case.case_id
    attempt_dir.mkdir(parents=True, exist_ok=True)
    write_json(attempt_dir / "preflight_report.json", preflight)

    if preflight["stage_status"] != "preflight_passed":
        prompts = build_prompts(case)
        command = build_vace_command_spec(
            case_id=case.case_id,
            vace_repo_dir="third_party/VACE",
            source_clip=str(attempt_dir / "source_clip.mp4"),
            mask_video=str(attempt_dir / "target_mask_gen.mp4"),
            prompt=prompts["model_prompt"],
            output_dir=str(attempt_dir),
            donor_reference=str(attempt_dir / "donor_reference.png") if case.donor else None,
            dry_run=True,
        )
        manifest = build_case_manifest(
            case_id=case.case_id,
            stage_status=preflight["stage_status"],
            operation=case.operation,
            generator_route=case.generator_route,
            target=preflight["target"],
            donor=preflight["donor"],
            source_clip={"status": "not_selected_due_to_preflight_blocker"},
            canonical_vace_profile={"name": "smoke_480", "fps": 16, "frame_options": [49, 65, 81]},
            mask_layers={
                "M_raw": str(attempt_dir / "target_mask_raw.npz"),
                "M_edit": str(attempt_dir / "target_mask_edit.npz"),
                "M_gen": str(attempt_dir / "target_mask_gen.npz"),
                "M_alpha": str(attempt_dir / "target_mask_alpha.npz"),
            },
            mask_processing_parameters={},
            prompt=prompts,
            vace_command=command,
            preflight=preflight,
        )
        write_json(attempt_dir / "vace_command.json", command)
        write_manifest(attempt_dir / "case_manifest.json", manifest)
        return manifest

    target_path = case.target.path.resolved_path if case.target.path else None
    if not target_path:
        raise DataAError("blocked_missing_mask: target resolved path is absent")
    target_tube = load_mask_tube(Path(target_path))
    plan_model = (case.sampling_meta or {}).get("vace_model_plan") or {}
    plan_profile = plan_model.get("profile")
    if plan_profile and profile_name != str(plan_profile):
        raise DataAError(
            f"blocked_model_plan_mismatch: case requires profile={plan_profile}, package profile={profile_name}"
        )
    profile = profile_from_name(profile_name)
    if execute_media:
        if not case.target.video_path:
            raise DataAError("blocked_clip_selection_failure: target video_path is absent")
        video_meta = ffprobe_video(Path(case.target.video_path), ffprobe_bin=ffprobe_bin)
        source_fps = video_meta.fps
    try:
        clip = select_clip(target_tube, source_fps=source_fps, profile=profile)
    except DataAError as exc:
        raise DataAError(f"blocked_clip_selection_failure: {exc}") from exc

    aligned_raw, alignment_meta = align_masks_to_canonical(target_tube, clip.canonical_to_source_frames, zero_missing=True)
    processing_config = MaskProcessingConfig()
    masks, mask_params = process_masks(aligned_raw, processing_config)
    height, width = profile.landscape_size
    masks = {
        name: resize_masks_nearest(mask, height=height, width=width)
        for name, mask in masks.items()
        if name != "M_alpha"
    }
    mask_policy = (case.sampling_meta or {}).get("mask_policy")
    masks, effective_params = apply_effective_mask_policy(masks, mask_policy, processing_config)
    mask_params["effective_mask_policy"] = effective_params
    mask_params["canonical_resize"] = {
        "height": height,
        "width": width,
        "method": "nearest",
        "scope": "case_intermediate_masks_only",
    }
    mask_layers = {
        "M_raw": str(attempt_dir / "target_mask_raw.npz"),
        "M_edit": str(attempt_dir / "target_mask_edit.npz"),
        "M_gen": str(attempt_dir / "target_mask_gen.npz"),
        "M_alpha": str(attempt_dir / "target_mask_alpha.npz"),
    }
    if synthetic_media or execute_media:
        save_mask_npz(Path(mask_layers["M_raw"]), masks["M_raw"])
        save_mask_npz(Path(mask_layers["M_edit"]), masks["M_edit"])
        save_mask_npz(Path(mask_layers["M_gen"]), masks["M_gen"])
        save_mask_npz(Path(mask_layers["M_alpha"]), masks["M_alpha"])

    mask_video_path = attempt_dir / "target_mask_gen.mp4"
    roundtrip = {"status": "not_run_in_dry_run"}
    mask_video_write = {"status": "not_run_in_dry_run"}
    if execute_media:
        mask_video_write = write_mask_video_ffmpeg(mask_video_path, masks["M_gen"], fps=clip.canonical_fps, ffmpeg_bin=ffmpeg_bin)
        roundtrip = validate_mask_video_roundtrip(mask_video_path, masks["M_gen"], ffmpeg_bin=ffmpeg_bin)
        if roundtrip["status"] != "ok":
            raise DataAError("blocked_mask_video_mismatch")
    elif synthetic_media:
        mask_video_write = write_mask_video(mask_video_path, masks["M_gen"], fps=clip.canonical_fps)
        roundtrip = validate_mask_video_roundtrip(mask_video_path, masks["M_gen"])
        if roundtrip["status"] != "ok":
            raise DataAError("blocked_mask_video_mismatch")

    donor_meta = None
    donor_reference_path = None
    if case.donor:
        if not (case.donor.path and case.donor.path.resolved_path):
            raise DataAError("blocked_donor_reference_failure: donor mask path is unresolved")
        donor_tube = load_mask_tube(Path(case.donor.path.resolved_path))
        if execute_media:
            if not case.donor.video_path:
                raise DataAError("blocked_donor_reference_failure: donor video_path is absent")
            donor_meta = export_donor_reference_from_video(out_dir=attempt_dir, tube=donor_tube, donor_video_path=case.donor.video_path, ffmpeg_bin=ffmpeg_bin)
        elif synthetic_media:
            donor_meta = export_synthetic_donor_reference(attempt_dir, donor_tube)
        else:
            donor_meta = {
                "status": "planned",
                "note": "server execution will crop donor RGB; dry-run does not read donor video pixels",
            }
        donor_reference_path = str(attempt_dir / "donor_reference.png")

    prompts = build_prompts(case)
    source_clip = canonical_video_plan(attempt_dir, clip, profile, source_video_path=case.target.video_path)
    source_vace_condition_path = attempt_dir / "source_vace_condition.mp4"
    source_clip["source_vace_condition_path"] = str(source_vace_condition_path)
    source_clip["vace_input_path"] = str(source_vace_condition_path)
    source_clip["vace_condition"] = {"status": "not_run_in_dry_run"}
    if execute_media and case.target.video_path:
        source_clip.update(
            export_canonical_videos(
                attempt_dir=attempt_dir,
                source_video_path=case.target.video_path,
                clip=clip,
                profile=profile,
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
            )
        )
        source_clip["vace_condition"] = export_vace_condition_video(
            source_clip=Path(source_clip["source_clip_path"]),
            masks=masks["M_gen"],
            out_path=source_vace_condition_path,
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
        )
        source_clip["vace_input_validation"] = _validate_vace_input_media(
            source_clip_path=Path(source_clip["source_clip_path"]),
            target_mask_gen_path=mask_video_path,
            source_vace_condition_path=source_vace_condition_path,
            ffprobe_bin=ffprobe_bin,
        )
    command = build_vace_command_spec(
        case_id=case.case_id,
        vace_repo_dir="third_party/VACE",
        source_clip=source_clip["vace_input_path"],
        mask_video=str(mask_video_path),
        prompt=prompts["model_prompt"],
        output_dir=str(attempt_dir),
        donor_reference=donor_reference_path,
        dry_run=dry_run,
        frame_num=clip.canonical_frame_count,
        size=str(plan_model.get("size") or ("480p" if profile.name in {"smoke_480", "production_480"} else "720p")),
        model_name=str(plan_model.get("model_name") or "vace-14B"),
    )
    manifest = build_case_manifest(
        case_id=case.case_id,
        stage_status="packed" if (synthetic_media or execute_media) else "planned",
        operation=case.operation,
        generator_route=case.generator_route,
        target=serialize_case(case)["target"],
        donor=serialize_case(case)["donor"],
        source_clip=source_clip,
        canonical_vace_profile={
            "name": profile.name,
            "fps": float(clip.canonical_fps),
            "source_duration_sec": float(clip.duration_seconds),
            "frame_count": int(clip.canonical_frame_count),
            "frame_options": list(profile.frame_options),
        },
        mask_layers=mask_layers,
        mask_processing_parameters=mask_params,
        prompt=prompts,
        reference_metadata=donor_meta,
        mask_video={"target_mask_gen_video": str(mask_video_path), "write": mask_video_write, "roundtrip": roundtrip},
        mask_alignment=alignment_meta,
        visualizations=planned_mask_visualizations(attempt_dir),
        seed=None,
        model_version={"vace_upstream_commit": "48eb44f1c4be87cc65a98bff985a26976841e9f3", "wan_commit": None, "weight_revision": None},
        code_commit=None,
        vace_command=command,
        preflight=preflight,
        sampling_meta=case.sampling_meta,
        generated_at_utc=utc_now_iso(),
    )
    write_json(attempt_dir / "vace_command.json", command)
    write_manifest(attempt_dir / "case_manifest.json", manifest)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--track-bank", type=Path, default=None)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--path-mapping", type=Path, default=None)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--profile", default="production_720")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--synthetic-media", action="store_true", help="Only for synthetic tests; writes tiny mask/video/png artifacts.")
    parser.add_argument("--execute-media", action="store_true", help="Use ffprobe/ffmpeg path for real media packaging.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        manifest = package_case(
            plan=args.plan,
            track_bank=args.track_bank,
            case_id=args.case_id,
            out_dir=args.out_dir,
            path_mapping=args.path_mapping,
            source_fps=args.source_fps,
            profile_name=args.profile,
            dry_run=args.dry_run,
            synthetic_media=args.synthetic_media,
            execute_media=args.execute_media,
            ffmpeg_bin=args.ffmpeg_bin,
            ffprobe_bin=args.ffprobe_bin,
        )
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"{manifest['case_id']}: {manifest['stage_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
