#!/usr/bin/env python3
"""Validate and register binary camera-route detection datasets in LlamaFactory."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from tools.install_camera_hard_route_gate import (
    dataset_entry,
    read_json,
    sha256,
    smoke_subset,
    validate_record,
    write_json,
)


DATASETS = {
    "camera_binary_route_shared": "binary_route_shared.json",
    "camera_binary_route_no_motion": "binary_route_no_motion.json",
    "camera_binary_route_motion": "binary_route_motion.json",
}


def record_id(row: Mapping[str, Any]) -> str:
    return str(row.get("route_record_id") or row.get("sample_id") or "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--llamafactory-data-dir", type=Path, required=True)
    parser.add_argument("--smoke-samples", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--check-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.llamafactory_data_dir.resolve()
    dataset_info_path = destination / "dataset_info.json"
    if not dataset_info_path.is_file():
        raise FileNotFoundError(f"verified LlamaFactory dataset_info.json not found: {dataset_info_path}")
    dataset_info = read_json(dataset_info_path)
    if not isinstance(dataset_info, dict):
        raise ValueError(f"dataset_info must be an object: {dataset_info_path}")

    summaries: dict[str, Any] = {}
    identifiers: dict[str, set[str]] = {}
    for offset, (dataset_name, source_name) in enumerate(DATASETS.items()):
        source = args.source_dir / source_name
        payload = read_json(source)
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"dataset must be a non-empty JSON list: {source}")
        records = [dict(row) for row in payload if isinstance(row, Mapping)]
        if len(records) != len(payload):
            raise ValueError(f"non-object records found: {source}")
        errors: list[str] = []
        expected_bucket = {
            "camera_binary_route_no_motion": "no-motion",
            "camera_binary_route_motion": "motion",
        }.get(dataset_name)
        for index, record in enumerate(records):
            errors.extend(validate_record(record, index))
            if record.get("gate_task") != "detection":
                errors.append(f"record[{index}]: binary route datasets must contain detection records")
            bucket = record.get("binary_route_bucket")
            if bucket not in {"no-motion", "motion"}:
                errors.append(f"record[{index}]: invalid binary_route_bucket={bucket!r}")
            if expected_bucket and bucket != expected_bucket:
                errors.append(
                    f"record[{index}]: expected binary_route_bucket={expected_bucket!r}, got {bucket!r}"
                )
            if len(errors) >= 20:
                break
        if errors:
            raise ValueError("invalid binary-route SFT data:\n" + "\n".join(errors[:20]))

        record_ids = [record_id(row) for row in records]
        if any(not value for value in record_ids) or len(record_ids) != len(set(record_ids)):
            raise ValueError(f"missing or duplicate route record ids in {source}")
        identifiers[dataset_name] = set(record_ids)

        unique_images = sorted({str(path) for row in records for path in row.get("images", [])})
        if args.check_images:
            missing = [path for path in unique_images if not Path(path).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"missing {len(missing)}/{len(unique_images)} images for {dataset_name}; "
                    f"first={missing[0]}"
                )

        destination_name = f"{dataset_name}.json"
        destination_path = destination / destination_name
        shutil.copy2(source, destination_path)
        smoke_name = f"{dataset_name}_smoke"
        smoke_path = destination / f"{smoke_name}.json"
        smoke = smoke_subset(records, args.smoke_samples, args.seed + offset)
        write_json(smoke_path, smoke)
        dataset_info[dataset_name] = dataset_entry(destination_name)
        dataset_info[smoke_name] = dataset_entry(smoke_path.name)
        summaries[dataset_name] = {
            "source": str(source),
            "destination": str(destination_path),
            "records": len(records),
            "smoke_records": len(smoke),
            "binary_route_buckets": dict(
                Counter(str(row.get("binary_route_bucket")) for row in records)
            ),
            "three_class_buckets": dict(Counter(str(row.get("route_bucket")) for row in records)),
            "detection_labels": dict(Counter(str(row.get("detection_label")) for row in records)),
            "unique_images": len(unique_images),
            "sha256": sha256(destination_path),
            "smoke_sha256": sha256(smoke_path),
        }

    shared = identifiers["camera_binary_route_shared"]
    no_motion = identifiers["camera_binary_route_no_motion"]
    motion = identifiers["camera_binary_route_motion"]
    if no_motion & motion or shared != no_motion | motion:
        raise AssertionError("binary shared dataset is not the exact disjoint expert union")

    write_json(dataset_info_path, dataset_info)
    summary = {
        "schema_version": "camera_binary_route_llamafactory_install_v1",
        "source_dir": str(args.source_dir),
        "llamafactory_data_dir": str(destination),
        "dataset_info": str(dataset_info_path),
        "shared_is_exact_disjoint_binary_expert_union": True,
        "images_checked": bool(args.check_images),
        "datasets": summaries,
    }
    summary_path = args.source_dir / "llamafactory_install_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
