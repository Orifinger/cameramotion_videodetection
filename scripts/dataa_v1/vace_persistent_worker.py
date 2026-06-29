#!/usr/bin/env python3
"""Persistent VACE worker process.

Launched once per worker group via torchrun. Rank 0 reads one shard descriptor
and broadcasts case descriptors to the group; all ranks process the same case in
lockstep. This wrapper intentionally refuses per-case model reload fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.common import DataAError, read_json, write_json
from scripts.dataa_v1.config import load_config
from scripts.dataa_v1.full_video import reassemble_full_video_pair
from scripts.dataa_v1.manifest import write_manifest
from scripts.dataa_v1.media_io import crop_video_frames
from scripts.dataa_v1.oss_sync import mark_ready_to_upload, upload_case_bundle
from scripts.dataa_v1.package_vace_case import package_case
from scripts.dataa_v1.run_state import RunPaths, RunState
from scripts.dataa_v1.vace_runtime import PersistentVaceRuntime, VaceJob


def _dist_broadcast_object(obj: Any, src: int = 0) -> Any:
    try:
        import torch.distributed as dist
    except ImportError as exc:
        raise DataAError("torch.distributed is required for persistent VACE worker") from exc
    holder = [obj]
    dist.broadcast_object_list(holder, src=src)
    return holder[0]


def _status_from_error(exc: Exception) -> str:
    text = str(exc)
    prefix = text.split(":", 1)[0]
    if prefix.startswith("blocked_"):
        return prefix
    return "blocked_packaging_failure"


def _frame_count_from_manifest(manifest: Mapping[str, Any]) -> int:
    source_clip = manifest.get("source_clip") or {}
    canonical = source_clip.get("canonical") or {}
    return int(canonical.get("frame_count") or 81)


def _postprocess_from_manifest(manifest: Mapping[str, Any], *, attempt_dir: Path) -> Dict[str, Any]:
    source_clip = manifest.get("source_clip") or {}
    canonical = source_clip.get("canonical") or {}
    frame_count = int(canonical.get("frame_count") or 81)
    valid_frame_count = int(canonical.get("valid_frame_count") or frame_count)
    pad_frame_count = int(canonical.get("pad_frame_count") or max(0, frame_count - valid_frame_count))
    crop_after_generation = bool(canonical.get("crop_generated_to_valid_frames") or valid_frame_count < frame_count)
    return {
        "frame_count": frame_count,
        "valid_frame_count": valid_frame_count,
        "pad_frame_count": pad_frame_count,
        "pad_mode": str(canonical.get("pad_mode") or "none"),
        "crop_after_generation": crop_after_generation,
        "fps": float(canonical.get("generation_fps") or canonical.get("fps") or 16),
        "raw_output_path": str(attempt_dir / "generated_raw.mp4"),
        "trimmed_output_path": str(attempt_dir / "generated_trimmed.mp4"),
    }


def _validate_model_plan(manifest: Mapping[str, Any], *, config: Mapping[str, Any], profile: str, size: str) -> None:
    model_plan = ((manifest.get("sampling_meta") or {}).get("vace_model_plan") or {})
    if not model_plan:
        raise DataAError(f"blocked_model_plan_mismatch: missing sampling_meta.vace_model_plan for {manifest.get('case_id')}")
    expected = {
        "model_name": str(config.get("model_name") or ""),
        "profile": str(profile),
        "size": str(size),
    }
    actual = {
        "model_name": str(model_plan.get("model_name") or ""),
        "profile": str(model_plan.get("profile") or ""),
        "size": str(model_plan.get("size") or ""),
    }
    if actual != expected:
        raise DataAError(f"blocked_model_plan_mismatch: case={actual} worker={expected}")


def _job_from_manifest(manifest: Mapping[str, Any], *, attempt_dir: Path, size: str, seed: int) -> VaceJob:
    source_clip = manifest.get("source_clip") or {}
    mask_video = manifest.get("mask_video") or {}
    prompt = manifest.get("prompt") or {}
    canonical = source_clip.get("canonical") or {}
    target_mask = mask_video.get("target_mask_gen_video") or str(attempt_dir / "target_mask_gen.mp4")
    vace_input = (
        source_clip.get("vace_input_path")
        or source_clip.get("source_vace_condition_path")
        or str(attempt_dir / "source_vace_condition.mp4")
    )
    donor_reference = None
    donor = manifest.get("donor")
    planned_reference = attempt_dir / "donor_reference.png"
    if donor and planned_reference.is_file():
        donor_reference = str(planned_reference)
    return VaceJob(
        case_id=str(manifest["case_id"]),
        source_clip=str(vace_input),
        target_mask_gen_video=str(target_mask),
        model_prompt=str(prompt.get("model_prompt") or ""),
        output_path=str(attempt_dir / "generated_raw.mp4"),
        donor_reference=donor_reference,
        frame_count=_frame_count_from_manifest(manifest),
        output_fps=float(canonical.get("generation_fps") or canonical.get("fps") or 16),
        size=size,
        seed=seed,
    )


def _postprocess_generation(result: Dict[str, Any], postprocess: Mapping[str, Any], *, ffmpeg_bin: str) -> Dict[str, Any]:
    detail = dict(postprocess)
    if not bool(detail.get("crop_after_generation")):
        detail["status"] = "not_required"
        detail["final_output_path"] = result["output_path"]
        result["postprocess"] = detail
        result["final_output_path"] = result["output_path"]
        return result
    crop = crop_video_frames(
        source_video=Path(str(result["output_path"])),
        out_path=Path(str(detail["trimmed_output_path"])),
        frame_count=int(detail["valid_frame_count"]),
        fps=float(detail["fps"]),
        ffmpeg_bin=ffmpeg_bin,
    )
    detail["status"] = "cropped_to_valid_frames"
    detail["crop"] = crop
    detail["final_output_path"] = crop["path"]
    result["postprocess"] = detail
    result["final_output_path"] = crop["path"]
    return result


def run_worker(*, config_path: Path, shard_path: Path, worker_id: int, run_id: str) -> int:
    config = load_config(config_path)
    shard = read_json(shard_path)
    rank = int(os.environ.get("RANK", "0"))
    group_cfg = next(
        (item for item in (shard.get("topology", {}).get("groups") or []) if int(item.get("worker_id", -1)) == worker_id),
        {},
    )
    config["vace"]["ulysses_size"] = int(group_cfg.get("ulysses_size", config["vace"].get("ulysses_size", 4)))
    config["vace"]["ring_size"] = int(group_cfg.get("ring_size", config["vace"].get("ring_size", 1)))
    config["vace"]["t5_fsdp"] = True
    config["vace"]["dit_fsdp"] = True
    if bool(config["vace"].get("offload_model", False)) or bool(config["vace"].get("t5_cpu", False)):
        raise DataAError("blocked_slow_memory_mode: offload_model and t5_cpu must both be false")
    runtime = PersistentVaceRuntime(config["vace"])
    runtime.initialize_once()
    paths = RunPaths.from_root(Path(config["run"]["tmp_root"]), run_id)
    state = RunState(paths, run_id=run_id, topology=shard.get("topology", {}))
    cases = shard.get("cases", [])
    attempts_root = paths.worker_dir(worker_id) / "attempts"
    execution_plan = Path(shard["execution_plan"])
    track_bank = Path(shard["track_bank"]) if shard.get("track_bank") else None
    path_mapping = Path(shard["path_mapping"]) if shard.get("path_mapping") else None
    profile = str(shard.get("profile") or config["vace"].get("profile", "production_720"))
    ffmpeg_bin = str(shard.get("ffmpeg_bin") or config["vace"].get("ffmpeg_bin", "ffmpeg"))
    ffprobe_bin = str(shard.get("ffprobe_bin") or config["vace"].get("ffprobe_bin", "ffprobe"))
    vace_size = str(shard.get("vace_size") or config["vace"].get("size") or ("480p" if profile == "smoke_480" else "720p"))
    seed = int(config["vace"].get("seed", 20260629))
    for case in cases:
        case_id = str(case["case_id"])
        package_result = None
        if rank == 0:
            if state.should_skip_case(case_id):
                package_result = {"status": "skipped_existing_terminal", "case_id": case_id, "skip_generation": True}
            else:
                try:
                    state.append_status(case_id, "packaging_started", worker_id=worker_id)
                    manifest = package_case(
                        plan=execution_plan,
                        track_bank=track_bank,
                        case_id=case_id,
                        out_dir=attempts_root,
                        path_mapping=path_mapping,
                        profile_name=profile,
                        dry_run=False,
                        synthetic_media=False,
                        execute_media=True,
                        ffmpeg_bin=ffmpeg_bin,
                        ffprobe_bin=ffprobe_bin,
                    )
                    attempt_dir = attempts_root / case_id
                    if manifest.get("stage_status") != "packed":
                        status = str(manifest.get("stage_status") or "blocked_packaging_failure")
                        state.append_status(case_id, status, worker_id=worker_id, detail={"manifest": str(attempt_dir / "case_manifest.json")})
                        package_result = {"status": status, "case_id": case_id, "skip_generation": True}
                    else:
                        _validate_model_plan(manifest, config=config["vace"], profile=profile, size=vace_size)
                        job = _job_from_manifest(manifest, attempt_dir=attempt_dir, size=vace_size, seed=seed)
                        postprocess = _postprocess_from_manifest(manifest, attempt_dir=attempt_dir)
                        state.append_status(case_id, "packed", worker_id=worker_id, detail={"manifest": str(attempt_dir / "case_manifest.json")})
                        package_result = {"status": "packed", "case_id": case_id, "vace_job": job.__dict__, "postprocess": postprocess, "manifest": manifest}
                except DataAError as exc:
                    status = _status_from_error(exc)
                    state.append_status(case_id, status, worker_id=worker_id, detail={"error": str(exc)})
                    package_result = {"status": status, "case_id": case_id, "skip_generation": True, "error": str(exc)}
        descriptor = package_result if rank == 0 else None
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            descriptor = _dist_broadcast_object(descriptor, src=0)
        if descriptor.get("skip_generation"):
            continue
        job = VaceJob(**descriptor["vace_job"])
        if rank == 0:
            state.append_status(job.case_id, "generation_started", worker_id=worker_id)
        try:
            result = runtime.generate_job(job)
        except DataAError as exc:
            if rank == 0:
                state.append_status(job.case_id, "blocked_vace_generation_failure", worker_id=worker_id, detail={"error": str(exc)})
            continue
        if rank == 0:
            attempt_dir = Path(job.output_path).parent
            try:
                result = _postprocess_generation(result, descriptor.get("postprocess") or {}, ffmpeg_bin=ffmpeg_bin)
                manifest_payload = descriptor.get("manifest") or read_json(attempt_dir / "case_manifest.json")
                full_video = reassemble_full_video_pair(
                    manifest=manifest_payload,
                    attempt_dir=attempt_dir,
                    generated_raw_video=Path(str(result["output_path"])),
                    final_generated_video=Path(str(result["final_output_path"])),
                    ffmpeg_bin=ffmpeg_bin,
                    ffprobe_bin=ffprobe_bin,
                )
                result["full_video"] = full_video
                manifest_payload["full_video"] = full_video
                write_manifest(attempt_dir / "case_manifest.json", manifest_payload)
            except DataAError as exc:
                state.append_status(job.case_id, "blocked_generation_postprocess_failure", worker_id=worker_id, detail={"error": str(exc)})
                continue
            write_json(attempt_dir / "generation_result.json", result)
            state.append_status(job.case_id, "generated", worker_id=worker_id, detail=result)
            mark_ready_to_upload(attempt_dir)
            upload_cfg = config.get("upload", {})
            oss_prefix = str(config["run"].get("oss_prefix", "")).rstrip("/")
            if oss_prefix:
                receipt = upload_case_bundle(
                    attempt_dir=attempt_dir,
                    oss_dest=f"{oss_prefix}/{run_id}/worker_{worker_id:02d}/attempts/{job.case_id}",
                    upload_command=str(upload_cfg.get("upload_command", "ossutil64")),
                    run_id=run_id,
                    case_id=job.case_id,
                    worker_id=worker_id,
                    execute=bool(upload_cfg.get("enabled", True)),
                )
                write_json(attempt_dir / "upload_receipt.json", receipt)
                state.append_status(job.case_id, str(receipt["status"]), worker_id=worker_id, detail={"upload_receipt": receipt})
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--shard", required=True, type=Path)
    parser.add_argument("--worker-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run_worker(config_path=args.config, shard_path=args.shard, worker_id=args.worker_id, run_id=args.run_id)
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
