#!/usr/bin/env python3
"""Validate and register the camera joint-SFT gate in LlamaFactory."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


DATASETS = {
    "camera_joint_detection_only": "joint_sft_detection_only.json",
    "camera_joint_correct_camera": "joint_sft_correct_camera.json",
    "camera_joint_shuffled_camera": "joint_sft_shuffled_camera.json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(record: Mapping[str, Any], index: int) -> list[str]:
    errors: list[str] = []
    prefix = f"record[{index}]"
    messages = record.get("messages")
    images = record.get("images")
    if not isinstance(messages, list) or not messages:
        return [f"{prefix}: messages must be a non-empty list"]
    if not isinstance(images, list) or not images:
        errors.append(f"{prefix}: images must be a non-empty list")
        images = []
    roles = [message.get("role") for message in messages if isinstance(message, Mapping)]
    if not roles or roles[-1] != "assistant":
        errors.append(f"{prefix}: the last message must be the assistant target")
    image_tokens = sum(
        str(message.get("content", "")).count("<image>")
        for message in messages
        if isinstance(message, Mapping)
    )
    if image_tokens != len(images):
        errors.append(f"{prefix}: image tokens={image_tokens}, image paths={len(images)}")
    if record.get("gate_task") not in {"camera", "detection"}:
        errors.append(f"{prefix}: invalid gate_task={record.get('gate_task')!r}")
    return errors


def balanced_smoke(records: Sequence[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    count = min(max(1, count), len(records))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record.get("gate_task")), str(record.get("gate_source", "camera")))].append(record)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    output: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(output) < count:
        progressed = False
        for key in keys:
            if groups[key] and len(output) < count:
                output.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    rng.shuffle(output)
    return output


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--llamafactory-data-dir", type=Path, required=True)
    parser.add_argument("--smoke-samples", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--check-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.llamafactory_data_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    dataset_info_path = destination / "dataset_info.json"
    dataset_info = read_json(dataset_info_path) if dataset_info_path.is_file() else {}
    if not isinstance(dataset_info, dict):
        raise ValueError(f"dataset_info must be an object: {dataset_info_path}")

    summaries: dict[str, Any] = {}
    expected_size: int | None = None
    for offset, (dataset_name, source_name) in enumerate(DATASETS.items()):
        source = args.source_dir / source_name
        payload = read_json(source)
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"dataset must be a non-empty JSON list: {source}")
        records = [dict(row) for row in payload if isinstance(row, Mapping)]
        if len(records) != len(payload):
            raise ValueError(f"non-object records found: {source}")
        errors: list[str] = []
        for index, record in enumerate(records):
            errors.extend(validate_record(record, index))
            if len(errors) >= 20:
                break
        if errors:
            raise ValueError("invalid SFT data:\n" + "\n".join(errors[:20]))
        if expected_size is None:
            expected_size = len(records)
        elif len(records) != expected_size:
            raise ValueError(f"three branches must have equal sizes: {len(records)} != {expected_size}")

        unique_images = sorted({str(path) for row in records for path in row.get("images", [])})
        if args.check_images:
            missing = [path for path in unique_images if not Path(path).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"missing {len(missing)}/{len(unique_images)} images for {dataset_name}; first={missing[0]}"
                )

        destination_name = f"{dataset_name}.json"
        destination_path = destination / destination_name
        shutil.copy2(source, destination_path)
        smoke_name = f"{dataset_name}_smoke"
        smoke_path = destination / f"{smoke_name}.json"
        smoke = balanced_smoke(records, args.smoke_samples, args.seed + offset)
        write_json(smoke_path, smoke)
        dataset_info[dataset_name] = dataset_entry(destination_name)
        dataset_info[smoke_name] = dataset_entry(smoke_path.name)
        summaries[dataset_name] = {
            "source": str(source),
            "destination": str(destination_path),
            "records": len(records),
            "smoke_records": len(smoke),
            "task_counts": dict(Counter(str(row.get("gate_task")) for row in records)),
            "source_counts": dict(Counter(str(row.get("gate_source", "camera")) for row in records)),
            "unique_images": len(unique_images),
            "sha256": sha256(destination_path),
            "smoke_sha256": sha256(smoke_path),
        }

    write_json(dataset_info_path, dataset_info)
    summary = {
        "schema_version": "camera_joint_sft_llamafactory_install_v1",
        "source_dir": str(args.source_dir),
        "llamafactory_data_dir": str(destination),
        "dataset_info": str(dataset_info_path),
        "equal_branch_sizes": len({value["records"] for value in summaries.values()}) == 1,
        "images_checked": bool(args.check_images),
        "datasets": summaries,
    }
    summary_path = args.source_dir / "llamafactory_install_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
