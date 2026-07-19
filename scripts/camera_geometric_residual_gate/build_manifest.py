#!/usr/bin/env python3
"""Build de-duplicated DataB and ViF manifests for the geometric-residual gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_geometric_residual_gate.contracts import (
    SCHEMA_VERSION,
    assistant_answer,
    camera_bucket,
    compact_counts,
    frame_paths_in_directory,
    label_from_vif_path,
    normalize_path,
    path_key,
    read_json_or_jsonl,
    source_from_datab_path,
    stable_unit,
    write_json,
    write_jsonl,
)
from tools.prepare_vifbench_camera_context import read_vif_index


def _camera_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        key = path_key(row.get("path"))
        if not key:
            raise ValueError(f"camera row {index} has no path")
        if key in output:
            raise ValueError(f"duplicate camera path: {row.get('path')}")
        labels = [str(value).strip() for value in row.get("labels") or [] if str(value).strip()]
        caption = str(row.get("caption") or "").strip()
        if not labels or not caption:
            raise ValueError(f"camera row {index} has empty labels/caption")
        output[key] = {
            "path": normalize_path(row.get("path")),
            "labels": labels,
            "caption": caption,
            "motion_bucket": camera_bucket(labels),
        }
    return output


def _assign_stratified_split(rows: list[dict[str, Any]], *, val_ratio: float, seed: int) -> None:
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(row["source_name"], row["answer"], row["motion_bucket"])].append(row)
    for values in strata.values():
        values.sort(key=lambda row: stable_unit(str(row["sample_id"]), seed))
        count = int(round(len(values) * val_ratio))
        if len(values) >= 8:
            count = max(1, count)
        count = min(count, max(0, len(values) - 1))
        for index, row in enumerate(values):
            row["dataset_split"] = "val" if index < count else "train"


def _camera_shortcut_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    class_counts = Counter(str(row["answer"]) for row in rows)
    for row in rows:
        by_bucket[str(row["motion_bucket"])][str(row["answer"])] += 1
    mapping = {
        bucket: counts.most_common(1)[0][0]
        for bucket, counts in by_bucket.items()
        if counts
    }
    recalls: dict[str, float] = {}
    for answer in ("Real", "Fake"):
        total = class_counts[answer]
        correct = sum(
            1
            for row in rows
            if row["answer"] == answer and mapping.get(str(row["motion_bucket"])) == answer
        )
        recalls[answer] = correct / total if total else 0.0
    return {
        "majority_mapping": mapping,
        "real_recall": recalls.get("Real", 0.0),
        "fake_recall": recalls.get("Fake", 0.0),
        "balanced_accuracy": (recalls.get("Real", 0.0) + recalls.get("Fake", 0.0)) / 2.0,
        "bucket_answer_counts": {
            bucket: dict(sorted(counts.items())) for bucket, counts in sorted(by_bucket.items())
        },
    }


def build_datab(args: argparse.Namespace) -> dict[str, Any]:
    detection = read_json_or_jsonl(args.detection_json)
    camera_rows = read_json_or_jsonl(args.camera_jsonl)
    camera = _camera_map(camera_rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unmatched_rows = 0
    for index, row in enumerate(detection):
        images = [normalize_path(value) for value in row.get("images") or []]
        if len(images) < args.min_frames:
            raise ValueError(f"DataB detection row {index} has only {len(images)} images")
        frame_dir = normalize_path(Path(images[0]).parent)
        camera_row = camera.get(path_key(frame_dir))
        if camera_row is None:
            unmatched_rows += 1
            continue
        grouped[path_key(frame_dir)].append(
            {
                "row_index": index,
                "answer": assistant_answer(row),
                "images": images,
                "frame_dir": frame_dir,
                "camera": camera_row,
            }
        )

    manifest: list[dict[str, Any]] = []
    duplicate_groups = 0
    for key, values in grouped.items():
        answers = {value["answer"] for value in values}
        image_lists = {tuple(path_key(path) for path in value["images"]) for value in values}
        if len(answers) != 1:
            raise ValueError(f"duplicate DataB frame directory has conflicting answers: {values[0]['frame_dir']}")
        if len(image_lists) != 1:
            raise ValueError(f"duplicate DataB frame directory has different frame lists: {values[0]['frame_dir']}")
        duplicate_groups += int(len(values) > 1)
        value = values[0]
        camera_row = value["camera"]
        sample_id = f"datab:{key}"
        if args.check_files:
            missing = [path for path in value["images"] if not Path(path).is_file()]
            if missing:
                raise FileNotFoundError(f"DataB sample has missing frames: {missing[:3]}")
        manifest.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "dataset_name": "DataB",
                "dataset_split": "pending",
                "answer": next(iter(answers)),
                "label": 1 if next(iter(answers)) == "Fake" else 0,
                "frame_dir_path": value["frame_dir"],
                "frame_paths": value["images"],
                "source_name": source_from_datab_path(value["frame_dir"]),
                "generator_name": "unknown",
                "camera_labels": camera_row["labels"],
                "camera_caption": camera_row["caption"],
                "motion_bucket": camera_row["motion_bucket"],
                "camera_annotation_kind": "CameraBench model prediction; stratification only",
                "detection_row_indices": [item["row_index"] for item in values],
                "duplicate_detection_rows": len(values),
            }
        )
    _assign_stratified_split(manifest, val_ratio=args.val_ratio, seed=args.seed)
    manifest.sort(key=lambda row: str(row["sample_id"]))
    write_jsonl(args.output_jsonl, manifest)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "DataB",
        "detection_rows": len(detection),
        "camera_rows": len(camera_rows),
        "matched_detection_rows": sum(len(values) for values in grouped.values()),
        "unmatched_detection_rows": unmatched_rows,
        "unique_manifest_samples": len(manifest),
        "duplicate_frame_directory_groups": duplicate_groups,
        "split_counts": compact_counts(manifest, "dataset_split"),
        "answer_counts": compact_counts(manifest, "answer"),
        "motion_bucket_counts": compact_counts(manifest, "motion_bucket"),
        "source_counts": compact_counts(manifest, "source_name"),
        "camera_bucket_shortcut": _camera_shortcut_summary(manifest),
        "camera_is_classifier_input": False,
        "dataa_is_used": False,
        "status": "passed" if manifest else "failed",
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_vif(args: argparse.Namespace) -> dict[str, Any]:
    expected = read_vif_index(args.index_dir, args.expected_ranks)
    camera_rows = read_json_or_jsonl(args.canonical_camera_jsonl)
    camera_by_id = {str(row.get("video_id")): row for row in camera_rows}
    if len(camera_by_id) != len(camera_rows):
        raise ValueError("canonical ViF camera sidecar contains duplicate video_id values")
    manifest: list[dict[str, Any]] = []
    missing_camera: list[str] = []
    for item in expected:
        video_id = str(item["video_id"])
        camera_row = camera_by_id.get(video_id)
        if camera_row is None:
            missing_camera.append(video_id)
            continue
        frame_dir = Path(str(item["frame_dir_path"]))
        if args.check_files and not frame_dir.is_dir():
            raise FileNotFoundError(f"missing ViF frame directory: {frame_dir}")
        frames = frame_paths_in_directory(frame_dir) if frame_dir.is_dir() else []
        if args.check_files and len(frames) < args.min_frames:
            raise ValueError(f"ViF sample has only {len(frames)} frames: {frame_dir}")
        labels = [str(value).strip() for value in camera_row.get("labels") or [] if str(value).strip()]
        answer = label_from_vif_path(str(frame_dir))
        manifest.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": f"vif:{video_id}",
                "dataset_name": "ViF-Bench",
                "dataset_split": "test",
                "answer": answer,
                "label": 1 if answer == "Fake" else 0,
                "frame_dir_path": normalize_path(frame_dir),
                "frame_paths": frames,
                "source_name": str(item["aigc_model_name"]),
                "generator_name": str(item["aigc_model_name"]),
                "camera_labels": labels,
                "camera_caption": str(camera_row.get("caption") or "").strip(),
                "motion_bucket": camera_bucket(labels),
                "camera_annotation_kind": "CameraBench model prediction; stratification only",
            }
        )
    coverage = len(manifest) / len(expected)
    manifest.sort(key=lambda row: str(row["sample_id"]))
    write_jsonl(args.output_jsonl, manifest)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "ViF-Bench",
        "expected_samples": len(expected),
        "camera_rows": len(camera_rows),
        "manifest_samples": len(manifest),
        "coverage": coverage,
        "missing_camera_count": len(missing_camera),
        "first_missing_camera": missing_camera[:30],
        "answer_counts": compact_counts(manifest, "answer"),
        "motion_bucket_counts": compact_counts(manifest, "motion_bucket"),
        "source_counts": compact_counts(manifest, "source_name"),
        "camera_bucket_shortcut": _camera_shortcut_summary(manifest),
        "camera_is_classifier_input": False,
        "dataa_is_used": False,
        "status": "passed" if coverage >= args.min_coverage else "failed",
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "passed":
        raise SystemExit(2)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    datab = subparsers.add_parser("datab")
    datab.add_argument("--detection-json", type=Path, required=True)
    datab.add_argument("--camera-jsonl", type=Path, required=True)
    datab.add_argument("--output-jsonl", type=Path, required=True)
    datab.add_argument("--summary-json", type=Path, required=True)
    datab.add_argument("--val-ratio", type=float, default=0.15)
    datab.add_argument("--seed", type=int, default=20260719)
    datab.add_argument("--min-frames", type=int, default=8)
    datab.add_argument("--check-files", action="store_true")
    datab.set_defaults(func=build_datab)

    vif = subparsers.add_parser("vif")
    vif.add_argument("--index-dir", type=Path, required=True)
    vif.add_argument("--canonical-camera-jsonl", type=Path, required=True)
    vif.add_argument("--output-jsonl", type=Path, required=True)
    vif.add_argument("--summary-json", type=Path, required=True)
    vif.add_argument("--expected-ranks", type=int, default=16)
    vif.add_argument("--min-coverage", type=float, default=1.0)
    vif.add_argument("--min-frames", type=int, default=8)
    vif.add_argument("--check-files", action="store_true")
    vif.set_defaults(func=build_vif)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = args.func(args)
    return 0 if summary.get("status") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
