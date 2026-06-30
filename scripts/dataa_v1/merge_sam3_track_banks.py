#!/usr/bin/env python3
"""Merge an old SAM3 track bank with rerun results for selected videos."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.build_subject_first_execution_plan import _repo_root
from scripts.dataa_v1.common import DataAError, read_json, utc_now_iso, write_json
from scripts.dataa_v1.schema import TRACK_ID_KEYS, TRACK_LIST_KEYS, VIDEO_ID_KEYS, as_records, first_value


DEFAULT_OUT_TRACK_BANK = Path("res/sam_track_bank/sam3_quality_tracks_enriched_merged_v001.json")
DEFAULT_OUT_ACTIVE_TRACK_BANK = Path("res/sam_track_bank/sam3_quality_tracks_enriched_active.json")
DEFAULT_OUT_MANIFEST = Path("res/dataA_v1/audits/sam3_track_bank_merge_manifest.json")


def _resolve_project_path(path: Path | str | None) -> Path | None:
    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else _repo_root() / value


def _optional_str(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _record_video_id(record: Mapping[str, Any]) -> str | None:
    return _optional_str(first_value(record, VIDEO_ID_KEYS))


def _record_track_id(record: Mapping[str, Any]) -> str | None:
    return _optional_str(first_value(record, TRACK_ID_KEYS))


def _records_by_video(records: Sequence[Mapping[str, Any]]) -> tuple[Dict[str, list[Dict[str, Any]]], list[Dict[str, Any]], list[str]]:
    by_video: Dict[str, list[Dict[str, Any]]] = {}
    without_video: list[Dict[str, Any]] = []
    order: list[str] = []
    for record in records:
        item = dict(record)
        video_id = _record_video_id(item)
        if not video_id:
            without_video.append(item)
            continue
        if video_id not in by_video:
            by_video[video_id] = []
            order.append(video_id)
        by_video[video_id].append(item)
    return by_video, without_video, order


def _rerun_video_ids(manifest: Mapping[str, Any]) -> set[str]:
    explicit = manifest.get("rerun_video_ids")
    if isinstance(explicit, list):
        return {str(item) for item in explicit if item not in (None, "")}
    ids: set[str] = set()
    for item in manifest.get("rerun_candidates") or []:
        if isinstance(item, Mapping) and item.get("video_id"):
            ids.add(str(item["video_id"]))
    return ids


def _check_duplicate_track_ids(records: Sequence[Mapping[str, Any]]) -> None:
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for index, record in enumerate(records):
        track_id = _record_track_id(record)
        if not track_id:
            continue
        if track_id in seen:
            duplicates.append(track_id)
        seen[track_id] = index
    if duplicates:
        sample = ", ".join(sorted(set(duplicates))[:20])
        raise DataAError(f"merged_track_bank_duplicate_track_id: {sample}")


def merge_track_banks(
    *,
    old_track_bank: Path,
    rerun_track_bank: Path,
    rerun_manifest: Path,
    out_track_bank: Path = DEFAULT_OUT_TRACK_BANK,
    out_active_track_bank: Path | None = DEFAULT_OUT_ACTIVE_TRACK_BANK,
    out_manifest: Path = DEFAULT_OUT_MANIFEST,
    dry_run: bool = False,
) -> Dict[str, Any]:
    old_payload = read_json(old_track_bank)
    rerun_payload = read_json(rerun_track_bank)
    rerun_payload_manifest = read_json(rerun_manifest)
    if not isinstance(rerun_payload_manifest, Mapping):
        raise DataAError("rerun_manifest_must_be_object")

    old_records = as_records(old_payload, TRACK_LIST_KEYS, "old track-bank")
    rerun_records = as_records(rerun_payload, TRACK_LIST_KEYS, "rerun track-bank")
    old_by_video, old_without_video, old_video_order = _records_by_video(old_records)
    rerun_by_video, rerun_without_video, rerun_video_order = _records_by_video(rerun_records)
    requested_rerun_videos = _rerun_video_ids(rerun_payload_manifest)

    merged_records: list[Dict[str, Any]] = []
    video_sources: list[Dict[str, Any]] = []
    replaced_video_ids: list[str] = []
    preserved_old_video_ids: list[str] = []
    added_rerun_video_ids: list[str] = []
    rerun_failed_video_ids: list[str] = []

    for video_id in old_video_order:
        if video_id in requested_rerun_videos:
            replacement = rerun_by_video.get(video_id)
            if replacement:
                merged_records.extend(replacement)
                replaced_video_ids.append(video_id)
                source = "rerun"
                track_count = len(replacement)
            else:
                merged_records.extend(old_by_video[video_id])
                rerun_failed_video_ids.append(video_id)
                source = "old_rerun_missing"
                track_count = len(old_by_video[video_id])
        else:
            merged_records.extend(old_by_video[video_id])
            preserved_old_video_ids.append(video_id)
            source = "old"
            track_count = len(old_by_video[video_id])
        video_sources.append({"video_id": video_id, "source": source, "track_count": track_count})

    old_video_ids = set(old_by_video)
    for video_id in rerun_video_order:
        if video_id in old_video_ids or video_id not in requested_rerun_videos:
            continue
        merged_records.extend(rerun_by_video[video_id])
        added_rerun_video_ids.append(video_id)
        video_sources.append({"video_id": video_id, "source": "rerun_added", "track_count": len(rerun_by_video[video_id])})
    for video_id in sorted(requested_rerun_videos - old_video_ids - set(rerun_by_video)):
        rerun_failed_video_ids.append(video_id)
        video_sources.append({"video_id": video_id, "source": "missing_from_old_and_rerun", "track_count": 0})

    merged_records.extend(old_without_video)
    if rerun_without_video:
        video_sources.append({"video_id": None, "source": "rerun_without_video_ignored", "track_count": len(rerun_without_video)})
    _check_duplicate_track_ids(merged_records)

    source_counts = Counter(item["source"] for item in video_sources)
    summary = {
        "old_track_count": len(old_records),
        "rerun_track_count": len(rerun_records),
        "merged_track_count": len(merged_records),
        "requested_rerun_video_count": len(requested_rerun_videos),
        "replaced_video_count": len(replaced_video_ids),
        "added_rerun_video_count": len(added_rerun_video_ids),
        "rerun_failed_video_count": len(rerun_failed_video_ids),
        "preserved_old_video_count": len(preserved_old_video_ids),
        "source_counts": dict(source_counts),
    }
    merged_payload = {
        "schema_version": "dataA_v1_sam3_quality_tracks_enriched_merged_v1",
        "generated_at_utc": utc_now_iso(),
        "source_track_banks": {
            "old_track_bank": str(old_track_bank),
            "rerun_track_bank": str(rerun_track_bank),
        },
        "rerun_manifest": str(rerun_manifest),
        "merge_policy": {
            "replace_only_requested_rerun_video_ids": True,
            "preserve_old_for_completed_or_kept_videos": True,
            "preserve_old_when_rerun_video_missing": True,
            "ignore_rerun_videos_not_requested": True,
        },
        "summary": summary,
        "tracks": merged_records,
    }
    manifest = {
        "schema_version": "dataA_v1_sam3_track_bank_merge_manifest_v1",
        "generated_at_utc": utc_now_iso(),
        "old_track_bank": str(old_track_bank),
        "rerun_track_bank": str(rerun_track_bank),
        "rerun_manifest": str(rerun_manifest),
        "out_track_bank": str(out_track_bank),
        "out_active_track_bank": None if out_active_track_bank is None else str(out_active_track_bank),
        "summary": summary,
        "replaced_video_ids": sorted(replaced_video_ids),
        "added_rerun_video_ids": sorted(added_rerun_video_ids),
        "rerun_failed_video_ids": sorted(rerun_failed_video_ids),
        "preserved_old_video_ids": sorted(preserved_old_video_ids),
        "video_sources": video_sources,
    }
    if not dry_run:
        write_json(out_track_bank, merged_payload)
        if out_active_track_bank is not None:
            write_json(out_active_track_bank, merged_payload)
        write_json(out_manifest, manifest)
    return {
        "dry_run": dry_run,
        "out_track_bank": str(out_track_bank),
        "out_active_track_bank": None if out_active_track_bank is None else str(out_active_track_bank),
        "out_manifest": str(out_manifest),
        "summary": summary,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge old SAM3 track-bank records with rerun records for rerun-manifest videos.")
    parser.add_argument("--old-track-bank", type=Path, required=True)
    parser.add_argument("--rerun-track-bank", type=Path, required=True)
    parser.add_argument("--rerun-manifest", type=Path, required=True)
    parser.add_argument("--out-track-bank", type=Path, default=DEFAULT_OUT_TRACK_BANK)
    parser.add_argument("--out-active-track-bank", type=Path, default=DEFAULT_OUT_ACTIVE_TRACK_BANK)
    parser.add_argument("--no-active-copy", action="store_true")
    parser.add_argument("--out-manifest", type=Path, default=DEFAULT_OUT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = merge_track_banks(
            old_track_bank=_resolve_project_path(args.old_track_bank) or args.old_track_bank,
            rerun_track_bank=_resolve_project_path(args.rerun_track_bank) or args.rerun_track_bank,
            rerun_manifest=_resolve_project_path(args.rerun_manifest) or args.rerun_manifest,
            out_track_bank=_resolve_project_path(args.out_track_bank) or args.out_track_bank,
            out_active_track_bank=None
            if args.no_active_copy
            else (_resolve_project_path(args.out_active_track_bank) or args.out_active_track_bank),
            out_manifest=_resolve_project_path(args.out_manifest) or args.out_manifest,
            dry_run=bool(args.dry_run),
        )
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    summary = result["summary"]
    print(
        "merge_sam3_track_banks "
        f"dry_run={result['dry_run']} "
        f"merged_tracks={summary['merged_track_count']} "
        f"replaced_videos={summary['replaced_video_count']} "
        f"rerun_failed_videos={summary['rerun_failed_video_count']} "
        f"out_track_bank={result['out_track_bank']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
