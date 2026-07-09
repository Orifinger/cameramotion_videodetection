#!/usr/bin/env python3
"""Build prompt-matched SFT train/eval sets for camera-context validation.

This is the small supervised smoke-test counterpart of
``build_dataa_camera_context_ablation.py``. It keeps the detection target
unchanged, uses only samples with matched camera labels, and writes:

  * train no-camera records
  * train gold-camera records
  * eval no/gold/shuffled/null-camera records

The intended comparison is:

  M0: continue SFT on ``*_train_no_camera.json``
  M1: continue SFT on ``*_train_gold_camera.json``

Then evaluate M0/M1 on the generated eval variants. This tests whether camera
context becomes useful after prompt-matched training, rather than only probing a
zero-shot prompt perturbation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from build_dataa_camera_context_ablation import (
    build_variant,
    conflict_key,
    deranged_camera_items,
    load_camera_jsonl,
    load_json,
    lookup_camera,
    record_camera_key,
    write_json,
)


DEFAULT_TRAIN_VARIANTS = ["no_camera", "gold_camera"]
DEFAULT_EVAL_VARIANTS = ["no_camera", "gold_camera", "shuffled_camera", "null_camera"]
VALID_VARIANTS = set(DEFAULT_EVAL_VARIANTS)


def answer_label(record: Mapping[str, Any]) -> str:
    """Best-effort Real/Fake label from the assistant answer."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        return "unknown"
    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = str(message.get("content", ""))
        lower = content.lower()
        if "<answer>fake</answer>" in lower:
            return "Fake"
        if "<answer>real</answer>" in lower:
            return "Real"
        if "fake" in lower and "real" not in lower:
            return "Fake"
        if "real" in lower and "fake" not in lower:
            return "Real"
    return "unknown"


def group_records(records: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for index, record in enumerate(records):
        key = conflict_key(record_camera_key(record))
        if not key:
            key = f"record-{index}"
        groups.setdefault(key, []).append(record)
    return groups


def flatten_groups(groups: Iterable[list[Mapping[str, Any]]]) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for group in groups:
        out.extend(group)
    return out


def group_label(group: list[Mapping[str, Any]]) -> str:
    counts = Counter(answer_label(record) for record in group)
    labels = [label for label in counts if label != "unknown"]
    if len(labels) > 1:
        return "mixed"
    if labels:
        return labels[0]
    return counts.most_common(1)[0][0] if counts else "unknown"


def take_group_prefix(
    groups: list[list[Mapping[str, Any]]],
    target_records: int,
) -> tuple[list[list[Mapping[str, Any]]], list[list[Mapping[str, Any]]]]:
    if target_records <= 0:
        return [], groups

    taken: list[list[Mapping[str, Any]]] = []
    remaining: list[list[Mapping[str, Any]]] = []
    count = 0
    taking = True
    for group in groups:
        if taking:
            taken.append(group)
            count += len(group)
            if count >= target_records:
                taking = False
        else:
            remaining.append(group)
    return taken, remaining


def split_eval_groups(
    groups: list[list[Mapping[str, Any]]],
    eval_count: int,
) -> tuple[list[list[Mapping[str, Any]]], list[list[Mapping[str, Any]]]]:
    """Split eval groups with light label stratification."""
    if eval_count <= 0:
        return [], groups

    total_records = sum(len(group) for group in groups)
    buckets: dict[str, list[list[Mapping[str, Any]]]] = {}
    for group in groups:
        buckets.setdefault(group_label(group), []).append(group)

    if len(buckets) <= 1:
        return take_group_prefix(groups, eval_count)

    bucket_sizes = {label: sum(len(group) for group in bucket) for label, bucket in buckets.items()}
    raw_targets = {
        label: eval_count * size / total_records
        for label, size in bucket_sizes.items()
    }
    targets = {label: int(raw_targets[label]) for label in buckets}
    remaining = eval_count - sum(targets.values())
    fractions = sorted(
        ((label, raw_targets[label] - targets[label]) for label in buckets),
        key=lambda item: item[1],
        reverse=True,
    )
    for label, _fraction in fractions:
        if remaining <= 0:
            break
        targets[label] += 1
        remaining -= 1

    eval_groups: list[list[Mapping[str, Any]]] = []
    train_groups: list[list[Mapping[str, Any]]] = []
    for label, bucket in buckets.items():
        target = targets[label]
        if target <= 0 and eval_count >= len(buckets):
            target = 1
        taken, rest = take_group_prefix(bucket, target)
        eval_groups.extend(taken)
        train_groups.extend(rest)

    return eval_groups, train_groups


def variant_camera_items(
    records: list[Mapping[str, Any]],
    camera: Mapping[str, dict[str, Any]],
    variant: str,
    seed: int,
) -> list[dict[str, Any] | None]:
    keys = [record_camera_key(record) for record in records]
    gold_items = [lookup_camera(camera, key) for key in keys]
    if variant in {"no_camera", "gold_camera"}:
        return gold_items
    if variant == "shuffled_camera":
        return deranged_camera_items(keys, gold_items, seed)
    if variant == "null_camera":
        return [None for _ in records]
    raise ValueError(f"unsupported variant: {variant}")


def build_and_write_variants(
    records: list[Mapping[str, Any]],
    camera: Mapping[str, dict[str, Any]],
    out_dir: Path,
    prefix: str,
    split_name: str,
    variants: list[str],
    seed: int,
) -> dict[str, str]:
    files: dict[str, str] = {}
    for variant in variants:
        camera_items = variant_camera_items(records, camera, variant, seed)
        variant_records = build_variant(records, camera_items, variant)
        out_path = out_dir / f"{prefix}_{split_name}_{variant}.json"
        write_json(out_path, variant_records)
        files[variant] = str(out_path)
    return files


def dataset_info_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images",
        },
    }


def write_dataset_info_snippet(
    out_dir: Path,
    prefix: str,
    files_by_split: Mapping[str, Mapping[str, str]],
) -> str:
    snippet: dict[str, Any] = {}
    for split_name, files in files_by_split.items():
        for variant, path in files.items():
            path_obj = Path(path)
            dataset_name = f"{prefix}_{split_name}_{variant}"
            snippet[dataset_name] = dataset_info_entry(path_obj.name)

    out_path = out_dir / f"{prefix}_llamafactory_dataset_info_snippet.json"
    write_json(out_path, snippet)
    return str(out_path)


def count_labels(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(answer_label(record) for record in records))


def validate_variants(values: list[str], arg_name: str) -> list[str]:
    bad = [value for value in values if value not in VALID_VARIANTS]
    if bad:
        raise ValueError(f"{arg_name} contains unsupported variants: {bad}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, help="Detection SFT JSON.")
    parser.add_argument("--camera-jsonl", required=True, help="Camera labels JSONL.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--prefix", default="datab_camera", help="Output filename and dataset prefix.")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--eval-count", type=int, default=1000, help="Approximate eval record count.")
    parser.add_argument(
        "--train-count",
        type=int,
        default=0,
        help="Optional approximate train record count after eval split. 0 means all remaining.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional approximate cap on camera-covered records before train/eval split. 0 means all.",
    )
    parser.add_argument(
        "--train-variants",
        nargs="+",
        default=DEFAULT_TRAIN_VARIANTS,
        help="Training variants to write.",
    )
    parser.add_argument(
        "--eval-variants",
        nargs="+",
        default=DEFAULT_EVAL_VARIANTS,
        help="Evaluation variants to write.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep input group order instead of deterministic shuffling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_variants = validate_variants(list(args.train_variants), "--train-variants")
    eval_variants = validate_variants(list(args.eval_variants), "--eval-variants")

    data = load_json(args.input_json)
    if not isinstance(data, list):
        raise ValueError(f"expected a list in {args.input_json}")

    camera = load_camera_jsonl(args.camera_jsonl)
    eligible_records = [
        record
        for record in data
        if lookup_camera(camera, record_camera_key(record)) is not None
    ]
    groups = list(group_records(eligible_records).values())
    if not groups:
        raise ValueError("no records matched camera labels")

    if not args.no_shuffle:
        random.Random(args.seed).shuffle(groups)

    if args.max_records and args.max_records > 0:
        capped_groups, _ = take_group_prefix(groups, args.max_records)
        groups = capped_groups

    total_records = sum(len(group) for group in groups)
    if args.eval_count >= total_records:
        raise ValueError(
            f"eval-count ({args.eval_count}) must be smaller than available camera-covered records ({total_records})"
        )

    eval_groups, train_groups = split_eval_groups(groups, args.eval_count)
    if args.train_count and args.train_count > 0:
        train_groups, _ = take_group_prefix(train_groups, args.train_count)

    train_records = flatten_groups(train_groups)
    eval_records = flatten_groups(eval_groups)
    if not train_records or not eval_records:
        raise ValueError("both train and eval splits must be non-empty")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files = build_and_write_variants(
        train_records,
        camera,
        out_dir,
        args.prefix,
        "train",
        train_variants,
        args.seed,
    )
    eval_files = build_and_write_variants(
        eval_records,
        camera,
        out_dir,
        args.prefix,
        "eval",
        eval_variants,
        args.seed + 17,
    )
    snippet_path = write_dataset_info_snippet(
        out_dir,
        args.prefix,
        {"train": train_files, "eval": eval_files},
    )

    summary = {
        "input_json": str(args.input_json),
        "camera_jsonl": str(args.camera_jsonl),
        "out_dir": str(out_dir),
        "prefix": args.prefix,
        "seed": args.seed,
        "original_num_records": len(data),
        "camera_rows_loaded": len(camera),
        "camera_covered_records": len(eligible_records),
        "missing_camera_records": len(data) - len(eligible_records),
        "num_groups_after_filtering": len(groups),
        "max_records": args.max_records,
        "requested_eval_count": args.eval_count,
        "requested_train_count": args.train_count,
        "num_train_records": len(train_records),
        "num_eval_records": len(eval_records),
        "train_label_counts": count_labels(train_records),
        "eval_label_counts": count_labels(eval_records),
        "train_variants": train_files,
        "eval_variants": eval_files,
        "llamafactory_dataset_info_snippet": snippet_path,
    }
    summary_path = out_dir / f"{args.prefix}_camera_context_sft_sets_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
