#!/usr/bin/env python3
"""Build full DataB and ViF development manifests without a 5524-row filter."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.forensic_temporal_expert_gate import SCHEMA_VERSION
from scripts.forensic_temporal_expert_gate.contracts import (
    assistant_answer,
    common_frame_directory,
    compact_counts,
    frame_paths_in_directory,
    generator_from_labeled_path,
    group_identity,
    label_from_path,
    normalize_path,
    path_key,
    read_json_or_jsonl,
    read_vif_index,
    source_and_split_from_datab,
    stable_hash,
    video_id_from_frame_dir,
    write_json,
    write_jsonl,
)


def assign_group_folds(rows: list[dict[str, Any]], folds: int, seed: int) -> None:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["group_id"])].append(row)
    strata: defaultdict[tuple[Any, ...], list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        signature = (
            tuple(sorted({str(row["label_name"]) for row in members})),
            tuple(sorted({str(row["source_dataset"]) for row in members})),
            tuple(sorted({str(row["generator_name"]) for row in members})),
        )
        strata[signature].append(group_id)
    for group_ids in strata.values():
        group_ids.sort(key=lambda value: stable_hash(value, seed))
        for index, group_id in enumerate(group_ids):
            fold = index % folds
            for row in groups[group_id]:
                row["fold"] = fold


def _deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    seen: dict[tuple[str, ...], dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        key = tuple(path_key(value) for value in row["frame_paths"])
        previous = seen.get(key)
        if previous is None:
            seen[key] = row
            output.append(row)
            continue
        if previous["label"] != row["label"]:
            raise ValueError(f"duplicate frames have conflicting labels: {row['sample_id']}")
        previous.setdefault("source_row_indices", []).extend(row["source_row_indices"])
        duplicates += 1
    return output, duplicates


def _finish(
    rows: list[dict[str, Any]],
    *,
    output_jsonl: Path,
    summary_json: Path,
    source: Mapping[str, Any],
    expected_records: int,
    check_files: bool,
) -> dict[str, Any]:
    missing: list[dict[str, Any]] = []
    if check_files:
        for row in rows:
            absent = [value for value in row["frame_paths"] if not Path(value).is_file()]
            if absent:
                missing.append(
                    {"sample_id": row["sample_id"], "count": len(absent), "first": absent[:3]}
                )
    rows.sort(key=lambda row: str(row["sample_id"]))
    write_jsonl(output_jsonl, rows)
    group_folds: defaultdict[str, set[int]] = defaultdict(set)
    for row in rows:
        if "fold" in row:
            group_folds[str(row["group_id"])].add(int(row["fold"]))
    split_groups = [key for key, values in group_folds.items() if len(values) > 1]
    frame_counts = Counter(int(row["frame_count"]) for row in rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed"
            if len(rows) == expected_records and not missing and not split_groups
            else "failed"
        ),
        "manifest_jsonl": normalize_path(output_jsonl),
        "source": dict(source),
        "records": len(rows),
        "expected_records": expected_records,
        "all_records_retained": len(rows) == expected_records,
        "explicitly_excluded_original_splits": [],
        "label_counts": compact_counts(rows, "label_name"),
        "source_counts": compact_counts(rows, "source_dataset"),
        "original_split_counts": compact_counts(rows, "source_split"),
        "fold_counts": compact_counts(rows, "fold") if rows and "fold" in rows[0] else {},
        "frame_count_distribution": dict(sorted(frame_counts.items())),
        "group_count": len(group_folds) if group_folds else len(rows),
        "groups_crossing_folds": len(split_groups),
        "missing_image_references": sum(item["count"] for item in missing),
        "first_missing": missing[:20],
    }
    write_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_datab(args: argparse.Namespace) -> int:
    payload = read_json_or_jsonl(args.detection_json)
    raw_rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        images = [normalize_path(value) for value in item.get("images") or []]
        frame_dir = common_frame_directory(images)
        label_name = assistant_answer(item)
        path_label = label_from_path(frame_dir)
        if path_label != label_name:
            raise ValueError(
                f"answer/path label mismatch at row {index}: {label_name} vs {path_label}"
            )
        source_dataset, source_split = source_and_split_from_datab(frame_dir)
        generator = generator_from_labeled_path(frame_dir, label_name)
        raw_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": f"datab:{frame_dir}",
                "video_id": video_id_from_frame_dir(frame_dir),
                "dataset_name": "DataB",
                "source_dataset": source_dataset,
                "source_split": source_split,
                "generator_name": generator,
                "label_name": label_name,
                "label": int(label_name == "Fake"),
                "group_id": group_identity(frame_dir, source_dataset),
                "frame_dir_path": frame_dir,
                "frame_paths": images,
                "frame_count": len(images),
                "source_row_indices": [index],
            }
        )
    rows, duplicates = _deduplicate(raw_rows)
    if duplicates:
        raise ValueError(
            f"DataB contains {duplicates} duplicate rows; refusing to silently reduce 6766"
        )
    assign_group_folds(rows, args.folds, args.seed)
    summary = _finish(
        rows,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        source={
            "kind": "full_datab_detection_json",
            "detection_json": normalize_path(args.detection_json),
            "raw_records": len(payload),
            "folds": args.folds,
            "fold_seed": args.seed,
            "important": (
                "All 6766 rows are retained. Original GenBuster train/test is metadata only, "
                "not an exclusion rule."
            ),
        },
        expected_records=args.expected_records,
        check_files=args.check_files,
    )
    return 0 if summary["status"] == "passed" else 2


def build_vif(args: argparse.Namespace) -> int:
    expected = read_vif_index(args.index_dir, args.expected_ranks)
    rows: list[dict[str, Any]] = []
    for item in expected:
        frame_dir = normalize_path(item["frame_dir_path"])
        directory = Path(frame_dir)
        frames = frame_paths_in_directory(directory) if directory.is_dir() else []
        label_name = label_from_path(frame_dir)
        generator = str(item["generator_name"]) if label_name == "Fake" else "real"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": f"vifbench:{frame_dir}",
                "video_id": str(item["video_id"]),
                "dataset_name": "ViF-Bench",
                "source_dataset": "ViF-Bench",
                "source_split": "development",
                "generator_name": generator,
                "label_name": label_name,
                "label": int(label_name == "Fake"),
                "group_id": group_identity(frame_dir, "ViF-Bench"),
                "frame_dir_path": frame_dir,
                "frame_paths": frames,
                "frame_count": len(frames),
            }
        )
    summary = _finish(
        rows,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        source={
            "kind": "vifbench_development_index",
            "index_dir": normalize_path(args.index_dir),
            "development_only": True,
            "genbuster_closed_benchmark_touched": False,
        },
        expected_records=args.expected_records,
        check_files=args.check_files,
    )
    return 0 if summary["status"] == "passed" else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    datab = subparsers.add_parser("datab")
    datab.add_argument("--detection-json", type=Path, required=True)
    datab.add_argument("--output-jsonl", type=Path, required=True)
    datab.add_argument("--summary-json", type=Path, required=True)
    datab.add_argument("--expected-records", type=int, default=6766)
    datab.add_argument("--folds", type=int, default=5)
    datab.add_argument("--seed", type=int, default=20260722)
    datab.add_argument("--check-files", action="store_true")
    datab.set_defaults(func=build_datab)

    vif = subparsers.add_parser("vif")
    vif.add_argument("--index-dir", type=Path, required=True)
    vif.add_argument("--output-jsonl", type=Path, required=True)
    vif.add_argument("--summary-json", type=Path, required=True)
    vif.add_argument("--expected-ranks", type=int, default=16)
    vif.add_argument("--expected-records", type=int, default=3160)
    vif.add_argument("--check-files", action="store_true")
    vif.set_defaults(func=build_vif)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
