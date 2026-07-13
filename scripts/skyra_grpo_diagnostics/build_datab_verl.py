#!/usr/bin/env python3
"""Convert the DataB ShareGPT JSON into deterministic verl Parquet splits."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ANSWER_RE = re.compile(r"<answer>\s*(Fake|Real)\s*</answer>", re.IGNORECASE)
TIMESTAMP_RE = re.compile(r"\[T=([0-9]+(?:\.[0-9]+)?)s\]")
TARGET_FRAMES = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-per-class", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--skip-image-check", action="store_true")
    return parser.parse_args()


def message_by_role(messages: list[dict[str, Any]], role: str) -> dict[str, Any]:
    matches = [message for message in messages if message.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"expected one {role!r} message, found {len(matches)}")
    if not isinstance(matches[0].get("content"), str):
        raise TypeError(f"{role!r} message content must be a string")
    return matches[0]


def extract_label(messages: list[dict[str, Any]]) -> str:
    assistant = message_by_role(messages, "assistant")["content"]
    matches = ANSWER_RE.findall(assistant)
    if len(matches) != 1:
        raise ValueError(f"expected one Fake/Real answer, found {len(matches)}")
    return matches[0].title()


def uniform_indices(source_count: int, target_count: int = TARGET_FRAMES) -> list[int]:
    if source_count < target_count:
        raise ValueError("cannot select more frames than are available")
    if target_count == 1:
        return [0]
    result = [round(i * (source_count - 1) / (target_count - 1)) for i in range(target_count)]
    if len(set(result)) != target_count:
        raise RuntimeError(f"uniform selection produced duplicate indices: {result}")
    return result


def select_user_frame_lines(user_text: str, selected_indices: list[int], source_count: int) -> str:
    lines = user_text.splitlines(keepends=True)
    frame_lines = [index for index, line in enumerate(lines) if "<image>" in line]
    if len(frame_lines) != source_count:
        raise ValueError(
            f"expected {source_count} one-image lines in user prompt, found {len(frame_lines)}"
        )
    if any(lines[index].count("<image>") != 1 for index in frame_lines):
        raise ValueError("each frame line must contain exactly one <image> placeholder")

    keep_slots = set(selected_indices)
    slot = 0
    selected_lines: list[str] = []
    for line in lines:
        if "<image>" not in line:
            selected_lines.append(line)
            continue
        if slot in keep_slots:
            selected_lines.append(line)
        slot += 1

    result = "".join(selected_lines)
    if result.count("<image>") != len(selected_indices):
        raise RuntimeError("rewritten user prompt has an unexpected image placeholder count")
    return result


def stable_sample_id(images: list[str], original_index: int) -> str:
    payload = f"{original_index}\n" + "\n".join(images)
    return "datab_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def as_file_uri(path: str) -> str:
    if path.startswith("file://"):
        return path
    if path.startswith("/"):
        return "file://" + path
    return path


def normalize_row(
    row: dict[str, Any],
    original_index: int,
    min_pixels: int,
    max_pixels: int,
    check_images: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    messages = row.get("messages")
    images = row.get("images")
    if not isinstance(messages, list) or not isinstance(images, list):
        return None, "missing_messages_or_images"
    if len(images) < TARGET_FRAMES:
        return None, f"fewer_than_{TARGET_FRAMES}_frames"
    if any(not isinstance(path, str) or not path for path in images):
        return None, "invalid_image_path"
    if check_images and any(not Path(path).is_file() for path in images):
        return None, "missing_image_file"

    try:
        label = extract_label(messages)
        system_text = message_by_role(messages, "system")["content"]
        user_text = message_by_role(messages, "user")["content"]
    except (TypeError, ValueError) as exc:
        return None, f"message_contract:{exc}"

    if user_text.count("<image>") != len(images):
        return None, "placeholder_image_count_mismatch"

    selected = uniform_indices(len(images))
    selected_images = [images[index] for index in selected]
    selected_user_text = select_user_frame_lines(user_text, selected, len(images))
    timestamps = [float(value) for value in TIMESTAMP_RE.findall(selected_user_text)]
    duration_seconds = max(timestamps) if timestamps else None
    sample_id = stable_sample_id(selected_images, original_index)

    converted = {
        "data_source": "datab_skyra_grpo",
        "prompt": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": selected_user_text},
        ],
        "images": [
            {
                "image": as_file_uri(path),
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
            for path in selected_images
        ],
        "ability": "aigc_video_detection",
        "reward_model": {"style": "rule", "ground_truth": label},
        "extra_info": {
            "sample_id": sample_id,
            "original_index": original_index,
            "label": label,
            "duration_seconds": duration_seconds,
            "original_frame_count": len(images),
            "selected_frame_indices": selected,
        },
    }
    return converted, None


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise TypeError("input JSON must contain a list")

    normalized: list[dict[str, Any]] = []
    dropped_reasons: Counter[str] = Counter()
    dropped_items: list[dict[str, Any]] = []
    original_frame_counts: Counter[int] = Counter()
    for index, row in enumerate(raw_rows):
        original_frame_counts[len(row.get("images", []))] += 1
        converted, reason = normalize_row(
            row=row,
            original_index=index,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            check_images=not args.skip_image_check,
        )
        if converted is None:
            reason = reason or "unknown"
            dropped_reasons[reason] += 1
            dropped_items.append({"original_index": index, "reason": reason})
        else:
            normalized.append(converted)

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        by_label[row["reward_model"]["ground_truth"]].append(row)
    if set(by_label) != {"Fake", "Real"}:
        raise RuntimeError(f"expected Fake and Real rows, found {sorted(by_label)}")

    balanced_count = min(len(by_label["Fake"]), len(by_label["Real"]))
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    balance_drops: list[dict[str, Any]] = []
    for offset, label in enumerate(("Fake", "Real")):
        rows = copy.deepcopy(by_label[label])
        random.Random(args.seed + offset).shuffle(rows)
        for row in rows[balanced_count:]:
            balance_drops.append(
                {"sample_id": row["extra_info"]["sample_id"], "label": label, "reason": "class_balance"}
            )
        rows = rows[:balanced_count]
        if args.validation_per_class >= len(rows):
            raise ValueError("validation-per-class leaves no training rows")
        label_val = rows[: args.validation_per_class]
        label_train = rows[args.validation_per_class :]
        for row in label_val:
            row["extra_info"]["split"] = "validation"
        for row in label_train:
            row["extra_info"]["split"] = "train"
        val_rows.extend(label_val)
        train_rows.extend(label_train)

    random.Random(args.seed).shuffle(train_rows)
    random.Random(args.seed).shuffle(val_rows)

    train_path = output_dir / "datab_grpo_train.parquet"
    val_path = output_dir / "datab_grpo_validation.parquet"
    write_parquet(train_rows, train_path)
    write_parquet(val_rows, val_path)

    manifest_path = output_dir / "datab_grpo_split_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for split, rows in (("train", train_rows), ("validation", val_rows)):
            for row in rows:
                handle.write(
                    json.dumps(
                        {
                            "sample_id": row["extra_info"]["sample_id"],
                            "split": split,
                            "label": row["reward_model"]["ground_truth"],
                            "original_index": row["extra_info"]["original_index"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    audit = {
        "status": "passed",
        "input_json": str(input_path),
        "seed": args.seed,
        "target_frames": TARGET_FRAMES,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "input_records": len(raw_rows),
        "original_frame_counts": dict(sorted(original_frame_counts.items())),
        "normalized_records": len(normalized),
        "normalized_label_counts": dict(Counter(row["reward_model"]["ground_truth"] for row in normalized)),
        "balanced_per_class": balanced_count,
        "train_records": len(train_rows),
        "train_label_counts": dict(Counter(row["reward_model"]["ground_truth"] for row in train_rows)),
        "validation_records": len(val_rows),
        "validation_label_counts": dict(Counter(row["reward_model"]["ground_truth"] for row in val_rows)),
        "dropped_reasons": dict(dropped_reasons),
        "dropped_items": dropped_items,
        "class_balance_drops": balance_drops,
        "outputs": {
            "train_parquet": str(train_path),
            "validation_parquet": str(val_path),
            "split_manifest": str(manifest_path),
        },
        "integrity_note": (
            "The validation split is held out from GRPO updates but was already seen by the inherited DataB SFT checkpoint. "
            "It is a policy-change diagnostic, not a true held-out generalization test."
        ),
    }
    audit_path = output_dir / "datab_grpo_data_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
