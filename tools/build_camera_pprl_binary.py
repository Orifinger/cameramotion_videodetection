#!/usr/bin/env python3
"""Convert the validated binary-camera SFT records into a balanced GRPO set."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_json(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or any(not isinstance(row, Mapping) for row in payload):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return [dict(row) for row in payload]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pair(pair_id: str, rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(rows) != 2:
        raise ValueError(f"camera pair {pair_id!r} has {len(rows)} records instead of two")
    by_answer = {str(row.get("answer")): dict(row) for row in rows}
    if set(by_answer) != {"Yes", "No"}:
        raise ValueError(f"camera pair {pair_id!r} is not a balanced Yes/No pair")
    yes = by_answer["Yes"]
    no = by_answer["No"]
    if yes.get("camera_primitive") != no.get("camera_primitive"):
        raise ValueError(f"camera pair {pair_id!r} mixes primitives")
    return yes, no


def select_balanced_pairs(
    rows: Sequence[Mapping[str, Any]], max_records: int, seed: int
) -> list[dict[str, Any]]:
    if max_records <= 0 or max_records > len(rows):
        max_records = len(rows)
    if max_records % 2:
        raise ValueError("max-records must be even so every selected question keeps its Yes/No pair")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        if not pair_id:
            raise ValueError("camera record is missing pair_id")
        grouped[pair_id].append(row)

    by_primitive: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair_id, pair_rows in grouped.items():
        yes, no = validate_pair(pair_id, pair_rows)
        primitive = str(yes.get("camera_primitive") or "")
        if not primitive:
            raise ValueError(f"camera pair {pair_id!r} is missing camera_primitive")
        by_primitive[primitive].append((yes, no))

    rng = random.Random(seed)
    for primitive, pairs in by_primitive.items():
        pairs.sort(key=lambda pair: str(pair[0].get("pair_id")))
        rng.shuffle(pairs)

    target_pairs = max_records // 2
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    positions: Counter[str] = Counter()
    primitives = sorted(by_primitive)
    while len(selected) < target_pairs:
        progressed = False
        for primitive in primitives:
            position = positions[primitive]
            if position < len(by_primitive[primitive]) and len(selected) < target_pairs:
                selected.append(by_primitive[primitive][position])
                positions[primitive] += 1
                progressed = True
        if not progressed:
            break
    if len(selected) != target_pairs:
        raise ValueError(
            f"only {len(selected) * 2} balanced records are available; requested {max_records}"
        )

    output = [row for pair in selected for row in pair]
    rng.shuffle(output)
    return output


def convert_record(source: Mapping[str, Any]) -> dict[str, Any]:
    answer = str(source.get("answer") or "")
    if answer not in {"Yes", "No"}:
        raise ValueError(f"invalid camera answer: {answer!r}")
    messages = source.get("messages")
    images = source.get("images")
    if not isinstance(messages, list) or not isinstance(images, list) or not images:
        raise ValueError("camera record must contain messages and at least one image")
    prompt_messages = [
        copy.deepcopy(dict(message))
        for message in messages
        if isinstance(message, Mapping) and message.get("role") != "assistant"
    ]
    if not prompt_messages or prompt_messages[-1].get("role") != "user":
        raise ValueError("camera GRPO prompt must end in a user message")
    if any(message.get("role") == "assistant" for message in prompt_messages):
        raise AssertionError("assistant answer leaked into the camera GRPO prompt")
    token_count = sum(str(message.get("content", "")).count("<image>") for message in prompt_messages)
    if token_count != len(images):
        raise ValueError(
            f"image token/path mismatch for {source.get('sample_id')}: {token_count} != {len(images)}"
        )
    return {
        "messages": prompt_messages,
        "images": [str(path) for path in images],
        "solution": answer,
        "answer": answer,
        "answer_id": 1 if answer == "Yes" else 0,
        "camera_primitive": source.get("camera_primitive"),
        "case_id": source.get("case_id"),
        "pair_id": source.get("pair_id"),
        "sample_id": source.get("sample_id"),
        "source_family": source.get("source_family"),
        "motion_bucket": source.get("motion_bucket"),
        "task_type": "camera_binary_pprl",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--smoke-json", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-records", type=int, default=1024)
    parser.add_argument("--smoke-records", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--check-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = read_json(args.input_json)
    selected = select_balanced_pairs(source, args.max_records, args.seed)
    records = [convert_record(row) for row in selected]
    smoke_selected = select_balanced_pairs(selected, args.smoke_records, args.seed + 1)
    smoke_records = [convert_record(row) for row in smoke_selected]

    if args.check_images:
        missing = sorted(
            {
                path
                for row in records
                for path in row["images"]
                if not Path(path).is_file()
            }
        )
        if missing:
            raise FileNotFoundError(f"missing {len(missing)} image files; first={missing[0]}")

    output_path = Path(args.output_json)
    smoke_path = Path(args.smoke_json)
    summary_path = Path(args.summary_json)
    write_json(output_path, records)
    write_json(smoke_path, smoke_records)

    pair_counts = Counter(str(row["pair_id"]) for row in records)
    primitive_pairs = Counter(
        str(row["camera_primitive"])
        for row in records
        if row["answer"] == "Yes"
    )
    train_case_ids = {str(row["case_id"]) for row in records}
    summary = {
        "schema_version": "camera_binary_pprl_data_v1",
        "question": (
            "Can phase-level GRPO on visually grounded, balanced binary camera questions improve "
            "AIGC detection without camera text at detection inference?"
        ),
        "input_json": args.input_json,
        "seed": args.seed,
        "selection": "round-robin over camera primitives while keeping complete Yes/No pairs",
        "records": len(records),
        "pairs": len(pair_counts),
        "unique_cases": len(train_case_ids),
        "supported_primitives": len(primitive_pairs),
        "answer_counts": dict(Counter(row["answer"] for row in records)),
        "primitive_pair_counts": dict(sorted(primitive_pairs.items())),
        "frame_count_distribution": dict(sorted(Counter(len(row["images"]) for row in records).items())),
        "integrity": {
            "all_pairs_complete": all(count == 2 for count in pair_counts.values()),
            "answers_balanced": Counter(row["answer"] for row in records)["Yes"]
            == Counter(row["answer"] for row in records)["No"],
            "assistant_absent_from_prompts": all(
                all(message.get("role") != "assistant" for message in row["messages"])
                for row in records
            ),
            "solutions_are_binary": all(row["solution"] in {"Yes", "No"} for row in records),
            "images_checked": bool(args.check_images),
        },
        "outputs": {
            "train": {"path": str(output_path), "records": len(records), "sha256": sha256(output_path)},
            "smoke": {
                "path": str(smoke_path),
                "records": len(smoke_records),
                "sha256": sha256(smoke_path),
            },
        },
    }
    required_integrity = {
        key: value
        for key, value in summary["integrity"].items()
        if key != "images_checked"
    }
    if not all(required_integrity.values()):
        raise AssertionError(f"camera PPRL data integrity failed: {summary['integrity']}")
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
