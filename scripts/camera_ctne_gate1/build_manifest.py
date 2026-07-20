#!/usr/bin/env python3
"""Build variable-length manifests for CTNE Gate 1."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_ctne_gate1.contracts import (
    MANIFEST_SCHEMA_VERSION,
    assistant_answer,
    camera_sidecar_map,
    compact_counts,
    dataset_slug,
    frame_count_bin,
    frame_count_summary,
    frame_paths_in_directory,
    label_from_path,
    lookup_camera_row,
    normalize_path,
    path_key,
    read_json_or_jsonl,
    read_vif_index,
    source_from_datab_path,
    source_from_labeled_path,
    stable_unit,
    write_json,
    write_jsonl,
)


def _common_parent(images: Sequence[str]) -> str:
    if not images:
        raise ValueError("sample has no image paths")
    parents = {path_key(PurePosixPath(normalize_path(value)).parent) for value in images}
    if len(parents) != 1:
        raise ValueError(f"sample spans multiple frame directories: {sorted(parents)[:3]}")
    return normalize_path(PurePosixPath(normalize_path(images[0])).parent)


def _camera_fields(camera_map: Mapping[str, dict[str, Any]], frame_dir: str) -> dict[str, Any]:
    row = lookup_camera_row(camera_map, frame_dir)
    if row is None:
        return {
            "camera_sidecar_available": False,
            "camera_labels": [],
            "camera_caption": "",
            "motion_bucket": "unknown",
        }
    return {
        "camera_sidecar_available": True,
        "camera_labels": list(row["labels"]),
        "camera_caption": str(row["caption"]),
        "motion_bucket": str(row["motion_bucket"]),
    }


def _check_images(images: Sequence[str]) -> list[str]:
    return [value for value in images if not Path(value).is_file()]


def _assign_stratified_split(
    rows: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
) -> None:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if val_ratio == 0.0:
        for row in rows:
            row["dataset_split"] = "train"
        return
    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("source_name", "unknown")),
            str(row.get("label_name", "unknown")),
            str(row.get("motion_bucket", "unknown")),
            str(row.get("frame_count_bin", "unknown")),
        )
        strata[key].append(row)
    for values in strata.values():
        values.sort(key=lambda row: stable_unit(str(row["sample_id"]), seed))
        count = len(values)
        if count < 2:
            val_count = 0
        else:
            val_count = max(1, min(count - 1, int(round(count * val_ratio))))
        for index, row in enumerate(values):
            row["dataset_split"] = "val" if index < val_count else "train"


def _deduplicate(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        key = tuple(path_key(value) for value in row["frame_paths"])
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = row
            output.append(row)
            continue
        duplicates += 1
        if previous["label_name"] != row["label_name"]:
            raise ValueError(f"duplicate frame sequence has conflicting labels: {row['frame_dir_path']}")
        previous.setdefault("source_row_indices", []).extend(row.get("source_row_indices", []))
    return output, duplicates


def _finalize(
    rows: list[dict[str, Any]],
    *,
    output_jsonl: Path,
    summary_json: Path,
    source: Mapping[str, Any],
    duplicate_rows_removed: int,
    check_files: bool,
) -> dict[str, Any]:
    missing: list[dict[str, Any]] = []
    if check_files:
        for row in rows:
            values = _check_images(row["frame_paths"])
            if values:
                missing.append({"sample_id": row["sample_id"], "missing": values[:5], "count": len(values)})
    rows.sort(key=lambda row: str(row["sample_id"]))
    write_jsonl(output_jsonl, rows)
    eligible = sum(bool(row["ctne_available"]) for row in rows)
    summary = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "passed" if not missing else "failed",
        "manifest_jsonl": normalize_path(output_jsonl),
        "source": dict(source),
        "records": len(rows),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "ctne_available_records": eligible,
        "ctne_unavailable_records": len(rows) - eligible,
        "ctne_available_rate": eligible / len(rows) if rows else 0.0,
        "camera_sidecar_coverage": (
            sum(bool(row["camera_sidecar_available"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "frame_counts": frame_count_summary(rows),
        "split_counts": compact_counts(rows, "dataset_split"),
        "label_counts": compact_counts(rows, "label_name"),
        "source_counts": compact_counts(rows, "source_name"),
        "generator_counts": compact_counts(rows, "generator_name"),
        "motion_bucket_counts": compact_counts(rows, "motion_bucket"),
        "missing_image_references": len(missing),
        "first_missing": missing[:20],
    }
    write_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_datab(args: argparse.Namespace) -> int:
    detection = read_json_or_jsonl(args.detection_json)
    camera_rows = read_json_or_jsonl(args.camera_jsonl) if args.camera_jsonl else []
    camera_map = camera_sidecar_map(camera_rows)
    raw_rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(detection):
        images = [normalize_path(value) for value in source_row.get("images") or []]
        frame_dir = _common_parent(images)
        label_name = assistant_answer(source_row)
        source_name = source_from_datab_path(frame_dir)
        generator_name = source_from_labeled_path(frame_dir) if label_name == "Fake" else "real"
        count = len(images)
        raw_rows.append(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "sample_id": f"datab:{frame_dir}",
                "dataset_name": "DataB",
                "dataset_slug": dataset_slug("DataB"),
                "source_name": source_name,
                "generator_name": generator_name,
                "label_name": label_name,
                "label": int(label_name == "Fake"),
                "frame_dir_path": frame_dir,
                "frame_paths": images,
                "frame_count": count,
                "frame_count_bin": frame_count_bin(count),
                "ctne_available": count >= 3,
                "source_row_indices": [index],
                **_camera_fields(camera_map, frame_dir),
            }
        )
    rows, duplicate_count = _deduplicate(raw_rows)
    _assign_stratified_split(rows, val_ratio=args.val_ratio, seed=args.seed)
    summary = _finalize(
        rows,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        source={
            "kind": "datab_detection_json",
            "detection_json": normalize_path(args.detection_json),
            "camera_jsonl": normalize_path(args.camera_jsonl) if args.camera_jsonl else None,
            "raw_detection_rows": len(detection),
            "raw_camera_rows": len(camera_rows),
            "split_seed": args.seed,
            "val_ratio": args.val_ratio,
        },
        duplicate_rows_removed=duplicate_count,
        check_files=args.check_files,
    )
    return 0 if summary["status"] == "passed" else 2


def _directory_row(
    *,
    dataset_name: str,
    frame_dir: str,
    frame_paths: Sequence[str],
    label_name: str,
    source_name: str,
    generator_name: str,
    split: str,
    camera_map: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    count = len(frame_paths)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "sample_id": f"{dataset_slug(dataset_name)}:{frame_dir}",
        "dataset_name": dataset_name,
        "dataset_slug": dataset_slug(dataset_name),
        "dataset_split": split,
        "source_name": source_name,
        "generator_name": generator_name,
        "label_name": label_name,
        "label": int(label_name == "Fake"),
        "frame_dir_path": frame_dir,
        "frame_paths": list(frame_paths),
        "frame_count": count,
        "frame_count_bin": frame_count_bin(count),
        "ctne_available": count >= 3,
        **_camera_fields(camera_map, frame_dir),
    }


def build_vif(args: argparse.Namespace) -> int:
    expected = read_vif_index(args.index_dir, args.expected_ranks)
    camera_rows = read_json_or_jsonl(args.camera_jsonl) if args.camera_jsonl else []
    camera_map = camera_sidecar_map(camera_rows)
    rows: list[dict[str, Any]] = []
    missing_dirs: list[str] = []
    for item in expected:
        frame_dir = normalize_path(item["frame_dir_path"])
        directory = Path(frame_dir)
        if directory.is_dir():
            frame_paths = frame_paths_in_directory(directory)
        else:
            frame_paths = []
            missing_dirs.append(frame_dir)
        label_name = label_from_path(frame_dir)
        source_name = str(item.get("source_name") or source_from_labeled_path(frame_dir))
        generator_name = source_name if label_name == "Fake" else "real"
        rows.append(
            _directory_row(
                dataset_name="ViF-Bench",
                frame_dir=frame_dir,
                frame_paths=frame_paths,
                label_name=label_name,
                source_name=source_name,
                generator_name=generator_name,
                split="test",
                camera_map=camera_map,
            )
        )
    summary = _finalize(
        rows,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        source={
            "kind": "vif_index",
            "index_dir": normalize_path(args.index_dir),
            "camera_jsonl": normalize_path(args.camera_jsonl) if args.camera_jsonl else None,
            "index_records": len(expected),
            "raw_camera_rows": len(camera_rows),
            "missing_frame_directories": len(missing_dirs),
            "first_missing_frame_directories": missing_dirs[:20],
        },
        duplicate_rows_removed=0,
        check_files=args.check_files,
    )
    if missing_dirs:
        summary["status"] = "failed"
        write_json(args.summary_json, summary)
    return 0 if summary["status"] == "passed" else 2


def _iter_frame_directories(root: Path) -> Iterable[tuple[Path, list[str]]]:
    for directory, _, files in __import__("os").walk(root):
        selected = [name for name in files if Path(name).suffix.casefold() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        if not selected:
            continue
        path = Path(directory)
        yield path, frame_paths_in_directory(path)


def build_tree(args: argparse.Namespace) -> int:
    camera_rows = read_json_or_jsonl(args.camera_jsonl) if args.camera_jsonl else []
    camera_map = camera_sidecar_map(camera_rows)
    raw_rows: list[dict[str, Any]] = []
    label_failures: list[str] = []
    for directory, frame_paths in _iter_frame_directories(args.frame_root):
        frame_dir = normalize_path(directory)
        try:
            label_name = label_from_path(frame_dir)
        except ValueError:
            label_failures.append(frame_dir)
            continue
        generator_name = source_from_labeled_path(frame_dir) if label_name == "Fake" else "real"
        raw_rows.append(
            _directory_row(
                dataset_name=args.dataset_name,
                frame_dir=frame_dir,
                frame_paths=frame_paths,
                label_name=label_name,
                source_name=generator_name if label_name == "Fake" else "real",
                generator_name=generator_name,
                split=args.split,
                camera_map=camera_map,
            )
        )
    rows, duplicate_count = _deduplicate(raw_rows)
    summary = _finalize(
        rows,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        source={
            "kind": "frame_tree",
            "frame_root": normalize_path(args.frame_root),
            "camera_jsonl": normalize_path(args.camera_jsonl) if args.camera_jsonl else None,
            "raw_camera_rows": len(camera_rows),
            "directories_without_unambiguous_label": len(label_failures),
            "first_label_failures": label_failures[:20],
        },
        duplicate_rows_removed=duplicate_count,
        check_files=args.check_files,
    )
    return 0 if summary["status"] == "passed" else 2


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--camera-jsonl", type=Path)
    parser.add_argument("--check-files", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    datab = subparsers.add_parser("datab", help="build from the DataB detection JSON")
    datab.add_argument("--detection-json", type=Path, required=True)
    datab.add_argument("--val-ratio", type=float, default=0.2)
    datab.add_argument("--seed", type=int, default=20260720)
    _add_output_args(datab)
    datab.set_defaults(func=build_datab)

    vif = subparsers.add_parser("vif", help="build from the 16 ViF index shards")
    vif.add_argument("--index-dir", type=Path, required=True)
    vif.add_argument("--expected-ranks", type=int, default=16)
    _add_output_args(vif)
    vif.set_defaults(func=build_vif)

    tree = subparsers.add_parser("tree", help="scan a Real/Fake frame directory tree")
    tree.add_argument("--frame-root", type=Path, required=True)
    tree.add_argument("--dataset-name", required=True)
    tree.add_argument("--split", default="test")
    _add_output_args(tree)
    tree.set_defaults(func=build_tree)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
