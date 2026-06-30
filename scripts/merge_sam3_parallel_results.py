#!/usr/bin/env python3
"""Merge SAM3 parallel worker JSON shards into the canonical track bank."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.sam3_tracking_config import (
    QWEN_INPUT_SCHEMA_VERSION,
    QWEN_SAM3_CANDIDATES_PATH,
    SAM3_FAILURES_PATH,
    SAM3_PARALLEL_RUN_ROOT,
    SAM3_QUALITY_TRACKS_PATH,
    SAM3_RUN_SUMMARY_PATH,
    SAM3_SCHEMA_VERSION,
    SAM3_TRACKS_ALL_PATH,
)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--num-workers", required=True, type=int)
    parser.add_argument("--qwen-candidates", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    return parser.parse_args()


def collect_quality(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for video in results:
        for candidate in video.get("candidate_results", []):
            if not isinstance(candidate, dict):
                continue
            for track in candidate.get("tracks", []):
                if not isinstance(track, dict) or track.get("quality_status") != "pass":
                    continue
                output.append({
                    "video_id": video["video_id"],
                    "relative_path": video.get("relative_path"),
                    "video_path": video["video_path"],
                    "candidate_id": candidate.get("candidate_id"),
                    "candidate_class": candidate.get("candidate_class"),
                    "canonical_concept": candidate.get("canonical_concept"),
                    "display_phrase": candidate.get("display_phrase"),
                    "sam_prompt": candidate.get("sam_prompt"),
                    **track,
                })
    return output


def main() -> None:
    args = parse()
    qwen_candidates_path = Path(args.qwen_candidates) if args.qwen_candidates is not None else Path(QWEN_SAM3_CANDIDATES_PATH)
    out_root = Path(args.out_root) if args.out_root is not None else None
    parallel_run_root = out_root / "parallel_runs" if out_root is not None else Path(SAM3_PARALLEL_RUN_ROOT)
    tracks_all_path = out_root / "sam3_tracks_all.json" if out_root is not None else Path(SAM3_TRACKS_ALL_PATH)
    quality_tracks_path = out_root / "sam3_quality_tracks.json" if out_root is not None else Path(SAM3_QUALITY_TRACKS_PATH)
    failures_path = out_root / "sam3_failures.json" if out_root is not None else Path(SAM3_FAILURES_PATH)
    summary_path = out_root / "sam3_run_summary.json" if out_root is not None else Path(SAM3_RUN_SUMMARY_PATH)

    qwen = read(qwen_candidates_path)
    if not isinstance(qwen, dict) or qwen.get("schema_version") != QWEN_INPUT_SCHEMA_VERSION:
        raise ValueError("Qwen v4 candidate input is invalid")
    input_videos = qwen.get("videos")
    if not isinstance(input_videos, list):
        raise ValueError("Qwen candidate input has no videos list")
    expected_ids = [str(item["video_id"]) for item in input_videos if isinstance(item, dict) and item.get("video_id")]

    worker_root = parallel_run_root / args.run_id / "workers"
    by_video: dict[str, dict[str, Any]] = {}
    shard_info: list[dict[str, Any]] = []
    missing_shards: list[str] = []
    for index in range(args.num_workers):
        path = worker_root / f"worker_{index:03d}_tracks.json"
        if not path.is_file():
            missing_shards.append(str(path))
            continue
        shard = read(path)
        if not isinstance(shard, dict) or shard.get("run_id") != args.run_id:
            raise ValueError(f"Invalid worker shard: {path}")
        records = shard.get("videos")
        if not isinstance(records, list):
            raise ValueError(f"Worker shard has no videos list: {path}")
        shard_info.append({
            "worker_index": index,
            "physical_gpu_id": shard.get("physical_gpu_id"),
            "assigned_video_count": shard.get("assigned_video_count"),
            "completed_video_count": shard.get("completed_video_count"),
            "status_totals": shard.get("status_totals"),
            "finished_at_utc": shard.get("finished_at_utc"),
        })
        for record in records:
            if not isinstance(record, dict) or not record.get("video_id"):
                continue
            key = str(record["video_id"])
            if key in by_video:
                raise ValueError(f"Duplicate result for video_id={key}")
            by_video[key] = record

    missing_videos = [key for key in expected_ids if key not in by_video]
    if missing_shards or missing_videos:
        raise RuntimeError(
            f"Incomplete parallel run: missing_shards={len(missing_shards)} missing_videos={len(missing_videos)}"
        )

    results = [by_video[key] for key in expected_ids]
    quality = collect_quality(results)
    failures = [result for result in results if result.get("status") == "failure"]
    status_totals: dict[str, int] = defaultdict(int)
    candidate_totals: dict[str, int] = defaultdict(int)
    all_tracks = 0
    for result in results:
        status_totals[str(result.get("status", "unknown"))] += 1
        for candidate in result.get("candidate_results", []):
            if not isinstance(candidate, dict):
                continue
            candidate_totals[str(candidate.get("status", "unknown"))] += 1
            tracks = candidate.get("tracks")
            if isinstance(tracks, list):
                all_tracks += len(tracks)

    summary = {
        "schema_version": SAM3_SCHEMA_VERSION,
        "created_at_utc": timestamp(),
        "run_id": args.run_id,
        "input_file": str(qwen_candidates_path),
        "input_video_records": len(expected_ids),
        "processed_videos": len(results),
        "parallel_worker_count": args.num_workers,
        "worker_shards": shard_info,
        "video_status_totals": dict(status_totals),
        "candidate_status_totals": dict(candidate_totals),
        "track_totals": {"all_tracks": all_tracks, "quality_pass_tracks": len(quality)},
    }
    write(tracks_all_path, {
        "schema_version": SAM3_SCHEMA_VERSION,
        "created_at_utc": timestamp(),
        "run_id": args.run_id,
        "input_file": str(qwen_candidates_path),
        "worker_shards": shard_info,
        "videos": results,
    })
    write(quality_tracks_path, {
        "schema_version": SAM3_SCHEMA_VERSION,
        "created_at_utc": timestamp(),
        "run_id": args.run_id,
        "tracks": quality,
    })
    write(failures_path, {
        "schema_version": SAM3_SCHEMA_VERSION,
        "created_at_utc": timestamp(),
        "run_id": args.run_id,
        "failures": failures,
    })
    write(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
