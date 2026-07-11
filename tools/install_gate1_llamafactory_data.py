#!/usr/bin/env python3
"""Validate and register Gate 1 multimodal preference data in LlamaFactory."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(record: Mapping[str, Any], index: int) -> list[str]:
    errors: list[str] = []
    messages = record.get("messages")
    images = record.get("images")
    chosen = record.get("chosen")
    rejected = record.get("rejected")
    prefix = f"record[{index}]"
    if not isinstance(messages, list) or not messages:
        errors.append(f"{prefix}: messages must be a non-empty list")
        messages = []
    if not isinstance(images, list) or not images:
        errors.append(f"{prefix}: images must be a non-empty list")
        images = []
    image_tokens = sum(str(message.get("content", "")).count("<image>") for message in messages if isinstance(message, Mapping))
    if image_tokens != len(images):
        errors.append(f"{prefix}: image tokens={image_tokens}, image paths={len(images)}")
    for name, answer in (("chosen", chosen), ("rejected", rejected)):
        if not isinstance(answer, Mapping):
            errors.append(f"{prefix}: {name} must be a message object")
            continue
        if answer.get("role") != "assistant" or not str(answer.get("content", "")).strip():
            errors.append(f"{prefix}: {name} must contain a non-empty assistant response")
    if isinstance(chosen, Mapping) and isinstance(rejected, Mapping):
        if chosen.get("content") == rejected.get("content"):
            errors.append(f"{prefix}: chosen and rejected are identical")
    if record.get("pair_order") not in {"real_first", "fake_first"}:
        errors.append(f"{prefix}: invalid pair_order={record.get('pair_order')!r}")
    if record.get("preference_kind") not in {"video_choice", "localization"}:
        errors.append(f"{prefix}: invalid preference_kind={record.get('preference_kind')!r}")
    return errors


def balanced_smoke(records: Sequence[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    count = min(max(1, count), len(records))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record.get("pair_order")), str(record.get("preference_kind")))].append(record)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    keys = sorted(groups)
    selected: list[dict[str, Any]] = []
    base, remainder = divmod(count, len(keys))
    for index, key in enumerate(keys):
        selected.extend(groups[key][: base + (1 if index < remainder else 0)])
    if len(selected) < count:
        selected_ids = {id(record) for record in selected}
        remaining = [record for record in records if id(record) not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    rng.shuffle(selected)
    return selected


def dataset_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "messages",
            "chosen": "chosen",
            "rejected": "rejected",
            "images": "images",
        },
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
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--llamafactory-data-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="dataa_counterfactual_dpo_local_only")
    parser.add_argument("--smoke-dataset-name", default="dataa_counterfactual_dpo_local_only_smoke")
    parser.add_argument("--smoke-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--check-image-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = read_json(args.source_json)
    if not isinstance(payload, list) or not payload:
        raise ValueError("source preference JSON must be a non-empty list")
    records = [dict(record) for record in payload if isinstance(record, Mapping)]
    if len(records) != len(payload):
        raise ValueError("every preference record must be a JSON object")

    errors: list[str] = []
    for index, record in enumerate(records):
        errors.extend(validate_record(record, index))
        if len(errors) >= 20:
            break
    if errors:
        raise ValueError("invalid preference data:\n" + "\n".join(errors[:20]))

    unique_images = sorted({str(path) for record in records for path in record.get("images", [])})
    missing_images = [path for path in unique_images if args.check_image_files and not Path(path).is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"missing {len(missing_images)}/{len(unique_images)} unique images; first={missing_images[0]}"
        )

    data_dir = args.llamafactory_data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    full_name = f"{args.dataset_name}.json"
    smoke_name = f"{args.smoke_dataset_name}.json"
    full_path = data_dir / full_name
    smoke_path = data_dir / smoke_name
    shutil.copy2(args.source_json, full_path)
    smoke = balanced_smoke(records, args.smoke_samples, args.seed)
    write_json(smoke_path, smoke)

    dataset_info_path = data_dir / "dataset_info.json"
    dataset_info = read_json(dataset_info_path) if dataset_info_path.is_file() else {}
    if not isinstance(dataset_info, dict):
        raise ValueError(f"dataset_info must be a JSON object: {dataset_info_path}")
    dataset_info[args.dataset_name] = dataset_entry(full_name)
    dataset_info[args.smoke_dataset_name] = dataset_entry(smoke_name)
    write_json(dataset_info_path, dataset_info)

    summary = {
        "source_json": str(args.source_json),
        "dataset_info": str(dataset_info_path),
        "dataset_name": args.dataset_name,
        "smoke_dataset_name": args.smoke_dataset_name,
        "num_records": len(records),
        "num_smoke_records": len(smoke),
        "num_unique_images": len(unique_images),
        "image_files_checked": bool(args.check_image_files),
        "all_images_exist": not missing_images if args.check_image_files else None,
        "pair_order": dict(Counter(str(record.get("pair_order")) for record in records)),
        "preference_kind": dict(Counter(str(record.get("preference_kind")) for record in records)),
        "edited_video": dict(Counter(str(record.get("edited_video")) for record in records)),
        "smoke_pair_order": dict(Counter(str(record.get("pair_order")) for record in smoke)),
        "smoke_preference_kind": dict(Counter(str(record.get("preference_kind")) for record in smoke)),
        "smoke_edited_video": dict(Counter(str(record.get("edited_video")) for record in smoke)),
        "full_sha256": sha256(full_path),
        "smoke_sha256": sha256(smoke_path),
    }
    summary_path = data_dir / "gate1_pair_dpo_install_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
