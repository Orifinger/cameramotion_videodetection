#!/usr/bin/env python3
"""Repair existing full_real/full_fake pairs with a bounded reassembly end skew."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.common import DataAError, read_json, utc_now_iso, write_json
from scripts.dataa_v1.full_video import repair_one_frame_full_pair_mismatch
from scripts.dataa_v1.media_io import ffprobe_video
from scripts.dataa_v1.oss_sync import mark_ready_to_upload, upload_case_bundle
from scripts.dataa_v1.run_state import RunPaths, RunState


def _worker_id_from_attempt(attempt_dir: Path) -> int:
    worker_name = attempt_dir.parent.parent.name
    if not worker_name.startswith("worker_"):
        return -1
    try:
        return int(worker_name.split("_", 1)[1])
    except ValueError:
        return -1


def _find_attempt_dir(run_root: Path, case_id: str) -> Path | None:
    matches = list(run_root.glob(f"worker_*/attempts/{case_id}"))
    if len(matches) == 1:
        return matches[0]
    return None


def _candidate_case_ids(state: Mapping[str, Any], *, include_all: bool) -> list[str]:
    cases = state.get("cases") or {}
    output: list[str] = []
    for case_id, info in cases.items():
        if include_all:
            output.append(str(case_id))
            continue
        if isinstance(info, Mapping) and info.get("status") == "blocked_generation_postprocess_failure":
            detail = info.get("detail") or {}
            error = str(detail.get("error") if isinstance(detail, Mapping) else detail)
            if "blocked_full_video_reassembly_mismatch" in error:
                output.append(str(case_id))
    return sorted(output)


def _patch_manifest(attempt_dir: Path, repair: Mapping[str, Any], *, ffprobe_bin: str) -> dict[str, Any]:
    manifest_path = attempt_dir / "case_manifest.json"
    manifest = read_json(manifest_path)
    real_meta = ffprobe_video(attempt_dir / "full_real.mp4", ffprobe_bin=ffprobe_bin)
    fake_meta = ffprobe_video(attempt_dir / "full_fake.mp4", ffprobe_bin=ffprobe_bin)
    if (
        round(real_meta.fps, 6) != round(fake_meta.fps, 6)
        or real_meta.frame_count != fake_meta.frame_count
        or real_meta.height != fake_meta.height
        or real_meta.width != fake_meta.width
    ):
        raise DataAError(
            "repair verification failed after bounded end trim: "
            f"full_real={{'fps': {real_meta.fps}, 'frame_count': {real_meta.frame_count}, 'height': {real_meta.height}, 'width': {real_meta.width}}} "
            f"full_fake={{'fps': {fake_meta.fps}, 'frame_count': {fake_meta.frame_count}, 'height': {fake_meta.height}, 'width': {fake_meta.width}}}"
        )
    full_video = dict(manifest.get("full_video") or {})
    full_video.update(
        {
            "status": "ok",
            "full_real_path": str(attempt_dir / "full_real.mp4"),
            "full_fake_path": str(attempt_dir / "full_fake.mp4"),
            "shape": {
                "fps": real_meta.fps,
                "frame_count": real_meta.frame_count,
                "height": real_meta.height,
                "width": real_meta.width,
            },
            "alignment_repair": dict(repair),
            "repaired_at_utc": utc_now_iso(),
            "donor_rgb_used": False,
        }
    )
    manifest["full_video"] = full_video
    manifest.setdefault("postprocess_repairs", []).append(
        {
            "type": "bounded_full_video_pair_alignment",
            "repair": dict(repair),
            "repaired_at_utc": full_video["repaired_at_utc"],
        }
    )
    write_json(manifest_path, manifest)
    return manifest


def repair_run(
    *,
    run_root: Path,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    oss_prefix: str | None,
    upload_command: str,
    execute_upload: bool,
    include_all: bool,
    execute: bool,
) -> dict[str, Any]:
    if not run_root.is_dir():
        raise DataAError(f"run root does not exist: {run_root}")
    paths = RunPaths(run_root=run_root, coordinator_dir=run_root / "coordinator")
    state_payload = read_json(paths.run_state_path)
    state = RunState(paths, run_id=run_root.name, topology=state_payload.get("topology") or {})
    case_ids = _candidate_case_ids(state_payload, include_all=include_all)
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for case_id in case_ids:
        attempt_dir = _find_attempt_dir(run_root, case_id)
        if attempt_dir is None:
            counts["skipped_attempt_dir_not_found"] += 1
            continue
        full_real = attempt_dir / "full_real.mp4"
        full_fake = attempt_dir / "full_fake.mp4"
        if not full_real.is_file() or not full_fake.is_file():
            counts["skipped_missing_full_pair"] += 1
            continue
        repair = repair_one_frame_full_pair_mismatch(
            full_real=full_real,
            full_fake=full_fake,
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
            execute=execute,
        )
        status = str(repair.get("status"))
        if status == "not_repairable":
            counts[f"not_repairable:{repair.get('reason')}"] += 1
            if len(examples) < 10:
                examples.append({"case_id": case_id, "repair": repair})
            continue
        if status == "already_aligned":
            counts[status] += 1
            continue
        if execute:
            manifest = _patch_manifest(attempt_dir, repair, ffprobe_bin=ffprobe_bin)
            mark_ready_to_upload(attempt_dir)
            detail: dict[str, Any] = {
                "repair": repair,
                "manifest": str(attempt_dir / "case_manifest.json"),
                "full_video": manifest.get("full_video") or {},
            }
            if oss_prefix:
                worker_id = _worker_id_from_attempt(attempt_dir)
                receipt = upload_case_bundle(
                    attempt_dir=attempt_dir,
                    oss_dest=f"{oss_prefix.rstrip('/')}/{run_root.name}/worker_{worker_id:02d}/attempts/{case_id}",
                    upload_command=upload_command,
                    run_id=run_root.name,
                    case_id=case_id,
                    worker_id=worker_id,
                    execute=execute_upload,
                )
                write_json(attempt_dir / "upload_receipt.json", receipt)
                detail["upload_receipt"] = receipt
                state.append_status(case_id, str(receipt["status"]), worker_id=worker_id, detail=detail)
            else:
                state.append_status(case_id, "generated", worker_id=_worker_id_from_attempt(attempt_dir), detail=detail)
        counts[status] += 1
        if len(examples) < 10:
            examples.append({"case_id": case_id, "attempt_dir": str(attempt_dir), "repair": repair})

    summary = {
        "run_root": str(run_root),
        "candidate_case_count": len(case_ids),
        "execute": execute,
        "execute_upload": execute_upload,
        "counts": dict(counts),
        "examples": examples,
    }
    write_json(run_root / "coordinator" / "one_frame_repair_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--oss-prefix", default=None)
    parser.add_argument("--upload-command", default="ossutil64")
    parser.add_argument("--execute-upload", action="store_true")
    parser.add_argument("--include-all", action="store_true", help="Scan all run_state cases, not only blocked one-frame mismatches.")
    parser.add_argument("--execute", action="store_true", help="Apply repairs. Without this flag, only writes a dry-run summary.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = repair_run(
            run_root=args.run_root,
            ffmpeg_bin=str(args.ffmpeg_bin),
            ffprobe_bin=str(args.ffprobe_bin),
            oss_prefix=str(args.oss_prefix) if args.oss_prefix else None,
            upload_command=str(args.upload_command),
            execute_upload=bool(args.execute_upload),
            include_all=bool(args.include_all),
            execute=bool(args.execute),
        )
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        "one_frame_repair "
        f"execute={summary['execute']} execute_upload={summary['execute_upload']} "
        f"candidates={summary['candidate_case_count']} counts={summary['counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
