#!/usr/bin/env python3
"""Build a filtered video manifest for one Qwen/SAM3 recovery rerun."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.build_subject_first_execution_plan import _repo_root
from scripts.dataa_v1.common import DataAError, read_json, utc_now_iso, write_json


DEFAULT_BASE_VIDEO_MANIFEST = Path("data/cambench_videos.json")
DEFAULT_RERUN_MANIFEST = Path("res/dataA_v1/audits/subject_first_qwen_sam3_rerun_manifest.json")
DEFAULT_OUT_MANIFEST = Path("data/cambench_videos_qwen_sam3_rerun_v001.json")


def _resolve_project_path(path: Path | str | None) -> Path | None:
    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else _repo_root() / value


def _rerun_video_ids(manifest: Mapping[str, Any]) -> set[str]:
    explicit = manifest.get("rerun_video_ids")
    if isinstance(explicit, list):
        return {str(item) for item in explicit if item not in (None, "")}
    ids: set[str] = set()
    for item in manifest.get("rerun_candidates") or []:
        if isinstance(item, Mapping) and item.get("video_id"):
            ids.add(str(item["video_id"]))
    return ids


def build_rerun_video_manifest(
    *,
    base_video_manifest: Path = DEFAULT_BASE_VIDEO_MANIFEST,
    rerun_manifest: Path = DEFAULT_RERUN_MANIFEST,
    out_manifest: Path = DEFAULT_OUT_MANIFEST,
    allow_missing: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    base = read_json(base_video_manifest)
    rerun = read_json(rerun_manifest)
    if not isinstance(base, Mapping) or not isinstance(base.get("videos"), list):
        raise DataAError(f"base_video_manifest_invalid: {base_video_manifest}")
    if not isinstance(rerun, Mapping):
        raise DataAError(f"rerun_manifest_invalid: {rerun_manifest}")

    requested = _rerun_video_ids(rerun)
    by_video: dict[str, dict[str, Any]] = {}
    for video in base["videos"]:
        if isinstance(video, Mapping) and video.get("video_id"):
            by_video[str(video["video_id"])] = dict(video)
    missing = sorted(requested - set(by_video))
    if missing and not allow_missing:
        raise DataAError(f"rerun_video_ids_missing_from_base_manifest: {missing[:20]}")

    selected = [dict(video) for video in base["videos"] if isinstance(video, Mapping) and str(video.get("video_id")) in requested]
    payload = {
        "schema_version": base.get("schema_version", "cambench_video_manifest_v1"),
        "created_at_utc": utc_now_iso(),
        "source_manifest": str(base_video_manifest),
        "rerun_manifest": str(rerun_manifest),
        "rerun_policy": {
            "scope": "qwen_sam3_recovery_round_1_only",
            "no_auto_second_rerun": True,
            "missing_video_policy": "allow_missing" if allow_missing else "block",
        },
        "video_root": base.get("video_root"),
        "num_videos": len(selected),
        "requested_video_count": len(requested),
        "missing_video_ids": missing,
        "videos": selected,
    }
    if not dry_run:
        write_json(out_manifest, payload)
    return {
        "dry_run": dry_run,
        "out_manifest": str(out_manifest),
        "requested_video_count": len(requested),
        "selected_video_count": len(selected),
        "missing_video_count": len(missing),
        "missing_video_ids": missing,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a filtered video manifest for Qwen/SAM3 rerun videos.")
    parser.add_argument("--base-video-manifest", type=Path, default=DEFAULT_BASE_VIDEO_MANIFEST)
    parser.add_argument("--rerun-manifest", type=Path, default=DEFAULT_RERUN_MANIFEST)
    parser.add_argument("--out-manifest", type=Path, default=DEFAULT_OUT_MANIFEST)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = build_rerun_video_manifest(
            base_video_manifest=_resolve_project_path(args.base_video_manifest) or args.base_video_manifest,
            rerun_manifest=_resolve_project_path(args.rerun_manifest) or args.rerun_manifest,
            out_manifest=_resolve_project_path(args.out_manifest) or args.out_manifest,
            allow_missing=bool(args.allow_missing),
            dry_run=bool(args.dry_run),
        )
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        "rerun_video_manifest "
        f"dry_run={result['dry_run']} "
        f"requested={result['requested_video_count']} "
        f"selected={result['selected_video_count']} "
        f"missing={result['missing_video_count']} "
        f"out={result['out_manifest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
