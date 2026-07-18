#!/usr/bin/env python3
"""Validate and register the paired DataB explicit-camera datasets in LlamaFactory."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.build_datab_explicit_camera_sft import (
    CAMERA_OPEN,
    answer_label,
    read_json,
    records_equal_except_camera_user,
    write_json,
)


DATASETS = {
    "datab_explicit_camera_no_camera": "datab_sft_no_camera_5739.json",
    "datab_explicit_camera_labels_caption": "datab_sft_with_camera_labels_caption_5739.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }


def balanced_indices(records: Sequence[Mapping[str, Any]], count: int, seed: int) -> list[int]:
    count = min(max(1, count), len(records))
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[answer_label(record)].append(index)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[int] = []
    keys = sorted(groups)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def validate_record(record: Mapping[str, Any], index: int) -> None:
    messages = record.get("messages")
    images = record.get("images")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"record[{index}] has no messages")
    if not isinstance(images, list) or not images:
        raise ValueError(f"record[{index}] has no images")
    roles = [message.get("role") for message in messages if isinstance(message, Mapping)]
    if not roles or roles[-1] != "assistant":
        raise ValueError(f"record[{index}] must end with assistant")
    image_tokens = sum(
        str(message.get("content", "")).count("<image>")
        for message in messages
        if isinstance(message, Mapping)
    )
    if image_tokens != len(images):
        raise ValueError(
            f"record[{index}] image token/path mismatch: {image_tokens} != {len(images)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--llamafactory-data-dir", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=5739)
    parser.add_argument("--smoke-samples", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--check-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.llamafactory_data_dir.resolve()
    dataset_info_path = destination / "dataset_info.json"
    if not dataset_info_path.is_file():
        raise FileNotFoundError(f"missing LlamaFactory dataset_info.json: {dataset_info_path}")
    dataset_info = read_json(dataset_info_path)
    if not isinstance(dataset_info, dict):
        raise ValueError(f"dataset_info must be an object: {dataset_info_path}")

    payloads: dict[str, list[dict[str, Any]]] = {}
    for dataset_name, file_name in DATASETS.items():
        path = args.source_dir / file_name
        payload = read_json(path)
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"dataset must be a non-empty list: {path}")
        records = [dict(record) for record in payload if isinstance(record, Mapping)]
        if len(records) != len(payload):
            raise ValueError(f"non-object records found: {path}")
        if args.expected_records > 0 and len(records) != args.expected_records:
            raise ValueError(
                f"unexpected records in {dataset_name}: {len(records)} != {args.expected_records}"
            )
        for index, record in enumerate(records):
            validate_record(record, index)
        payloads[dataset_name] = records

    baseline = payloads["datab_explicit_camera_no_camera"]
    conditioned = payloads["datab_explicit_camera_labels_caption"]
    if len(baseline) != len(conditioned):
        raise ValueError("paired datasets have different record counts")
    for index, (plain, camera) in enumerate(zip(baseline, conditioned)):
        if CAMERA_OPEN in str(plain.get("messages", "")):
            raise ValueError(f"no-camera record[{index}] contains camera context")
        if not records_equal_except_camera_user(plain, camera):
            raise ValueError(f"paired record[{index}] differs outside camera user suffix")

    unique_images = sorted({str(path) for row in baseline for path in row.get("images", [])})
    if args.check_images:
        missing = [path for path in unique_images if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"missing {len(missing)}/{len(unique_images)} images; first={missing[0]}"
            )

    smoke_indices = balanced_indices(baseline, args.smoke_samples, args.seed)
    summaries: dict[str, Any] = {}
    for dataset_name, file_name in DATASETS.items():
        source = args.source_dir / file_name
        destination_path = destination / file_name
        shutil.copy2(source, destination_path)
        smoke_name = f"{dataset_name}_smoke"
        smoke_path = destination / f"{smoke_name}.json"
        smoke = [payloads[dataset_name][index] for index in smoke_indices]
        write_json(smoke_path, smoke)
        dataset_info[dataset_name] = dataset_entry(file_name)
        dataset_info[smoke_name] = dataset_entry(smoke_path.name)
        summaries[dataset_name] = {
            "source": str(source),
            "destination": str(destination_path),
            "records": len(payloads[dataset_name]),
            "smoke_records": len(smoke),
            "sha256": sha256(destination_path),
            "smoke_sha256": sha256(smoke_path),
        }

    write_json(dataset_info_path, dataset_info)
    summary = {
        "schema_version": "datab_explicit_camera_sft_install_v1",
        "source_dir": str(args.source_dir),
        "llamafactory_data_dir": str(destination),
        "dataset_info": str(dataset_info_path),
        "paired_record_count": len(baseline),
        "paired_integrity": True,
        "shared_smoke_indices": smoke_indices,
        "unique_images": len(unique_images),
        "images_checked": bool(args.check_images),
        "datasets": summaries,
    }
    summary_path = args.source_dir / "llamafactory_install_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
