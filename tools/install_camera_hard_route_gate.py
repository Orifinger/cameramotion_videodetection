#!/usr/bin/env python3
"""Validate and register hard-route detection datasets in LlamaFactory."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


DATASETS = {
    "camera_hard_route_shared": "hard_route_shared.json",
    "camera_hard_route_no_motion": "hard_route_no_motion.json",
    "camera_hard_route_minor_motion": "hard_route_minor_motion.json",
    "camera_hard_route_complex_motion": "hard_route_complex_motion.json",
    "camera_hard_route_router": "hard_route_router_train.json",
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
        errors.append(f"{prefix}: last message must be the assistant target")
    image_tokens = sum(
        str(message.get("content", "")).count("<image>")
        for message in messages
        if isinstance(message, Mapping)
    )
    if image_tokens != len(images):
        errors.append(f"{prefix}: image tokens={image_tokens}, image paths={len(images)}")
    gate_task = record.get("gate_task")
    if gate_task not in {"detection", "camera_route"}:
        errors.append(f"{prefix}: invalid gate_task={gate_task!r}")
    if record.get("route_bucket") not in {"no-motion", "minor-motion", "complex-motion"}:
        errors.append(f"{prefix}: invalid route_bucket={record.get('route_bucket')!r}")
    if gate_task == "detection" and record.get("detection_label") not in {"Real", "Fake"}:
        errors.append(f"{prefix}: invalid detection_label={record.get('detection_label')!r}")
    if gate_task == "camera_route" and record.get("answer") not in {"Yes", "No"}:
        errors.append(f"{prefix}: invalid camera route answer={record.get('answer')!r}")
    return errors


def smoke_subset(records: Sequence[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    count = min(max(1, count), len(records))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        target = record.get("detection_label") or record.get("answer")
        key = (str(record.get("route_bucket")), str(target))
        groups.setdefault(key, []).append(record)
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)
    output: list[dict[str, Any]] = []
    while len(output) < count:
        progressed = False
        for key in sorted(groups):
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
    source_record_ids: dict[str, set[str]] = {}
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
            raise ValueError("invalid hard-route SFT data:\n" + "\n".join(errors[:20]))
        record_ids = [
            str(row.get("route_record_id") or row.get("sample_id") or "") for row in records
        ]
        if not all(record_ids) or len(record_ids) != len(set(record_ids)):
            raise ValueError(f"missing or duplicate route_record_id in {source}")
        source_record_ids[dataset_name] = set(record_ids)

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
            "route_buckets": dict(Counter(str(row["route_bucket"]) for row in records)),
            "detection_labels": dict(
                Counter(str(row.get("detection_label")) for row in records if row.get("detection_label"))
            ),
            "answers": dict(Counter(str(row.get("answer")) for row in records)),
            "domains": dict(Counter(str(row.get("route_domain")) for row in records)),
            "unique_images": len(unique_images),
            "sha256": sha256(destination_path),
            "smoke_sha256": sha256(smoke_path),
        }

    shared = source_record_ids["camera_hard_route_shared"]
    expert_union = set().union(
        source_record_ids["camera_hard_route_no_motion"],
        source_record_ids["camera_hard_route_minor_motion"],
        source_record_ids["camera_hard_route_complex_motion"],
    )
    expert_total = sum(
        len(source_record_ids[name])
        for name in (
            "camera_hard_route_no_motion",
            "camera_hard_route_minor_motion",
            "camera_hard_route_complex_motion",
        )
    )
    if shared != expert_union or expert_total != len(expert_union):
        raise AssertionError("shared dataset is not the disjoint union of the three expert datasets")

    write_json(dataset_info_path, dataset_info)
    summary = {
        "schema_version": "camera_hard_route_llamafactory_install_v1",
        "source_dir": str(args.source_dir),
        "llamafactory_data_dir": str(destination),
        "dataset_info": str(dataset_info_path),
        "shared_is_exact_disjoint_expert_union": True,
        "images_checked": bool(args.check_images),
        "datasets": summaries,
    }
    summary_path = args.source_dir / "llamafactory_install_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
