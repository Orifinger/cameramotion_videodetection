#!/usr/bin/env python3
"""Build detection JSON variants for camera-context causal ablation.

The generated files keep the original detection task unchanged except for a
short camera-context block appended to the user message.

Variants:
  no_camera       exact copy of the input records
  gold_camera     append the matching camera labels/caption
  shuffled_camera append camera labels/caption from a different sample
  null_camera     append an explicit camera-not-provided block

The matcher is path based: a camera JSONL row's ``path`` should be the frame
directory, while each detection record contains image paths under that directory.
This works for both DataA and DataB style files.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import PurePosixPath, Path
from typing import Any, Mapping


DATAA_CASE_RE = re.compile(r"(dataA_v1_(?:dataset_v2_|textedit_reserve_)?\d+)")

CAMERA_BLOCK_RE = re.compile(
    r"\n*--- Camera Motion Context ---\n.*?\n--- End Camera Motion Context ---\n*",
    re.DOTALL,
)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def norm_path(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip()
    while len(text) > 1 and text.endswith("/"):
        text = text[:-1]
    return text


def parent_dir(path: str) -> str:
    path = norm_path(path)
    if not path:
        return ""
    return str(PurePosixPath(path).parent)


def first_image(record: Mapping[str, Any]) -> str:
    images = record.get("images")
    if isinstance(images, list) and images:
        return norm_path(images[0])
    return ""


def record_camera_key(record: Mapping[str, Any]) -> str:
    return parent_dir(first_image(record))


def conflict_key(key: str) -> str:
    match = DATAA_CASE_RE.search(norm_path(key))
    if match:
        return match.group(1)
    return norm_path(key)


def load_camera_jsonl(path: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            camera_path = norm_path(item.get("path", ""))
            if camera_path:
                out[camera_path] = item
    return out


def lookup_camera(camera: Mapping[str, dict[str, Any]], key: str) -> dict[str, Any] | None:
    key = norm_path(key)
    if not key:
        return None
    if key in camera:
        return camera[key]
    # Defensive fallback: walk upward in case frames are nested one level deeper.
    cur = PurePosixPath(key)
    for _ in range(4):
        cur = cur.parent
        cur_text = str(cur)
        if cur_text in camera:
            return camera[cur_text]
    return None


def normalize_labels(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value).strip()] if str(value).strip() else []


def camera_context_text(camera_item: Mapping[str, Any] | None, variant: str) -> str:
    if variant == "null_camera":
        body = "<camera_motion>\n<labels>not provided</labels>\n</camera_motion>"
    elif not camera_item:
        body = "<camera_motion>\n<labels>unknown</labels>\n</camera_motion>"
    else:
        labels = normalize_labels(camera_item.get("labels"))
        label_text = "; ".join(labels) if labels else "unknown"
        caption = str(camera_item.get("caption", "")).strip()
        lines = ["<camera_motion>", f"<labels>{label_text}</labels>"]
        if caption:
            lines.append(f"<caption>{caption}</caption>")
        lines.append("</camera_motion>")
        body = "\n".join(lines)

    return "\n".join(
        [
            "--- Camera Motion Context ---",
            body,
            "Use camera motion only as context for local artifact checks; do not treat it as direct Real/Fake evidence.",
            "--- End Camera Motion Context ---",
        ]
    )


def strip_existing_camera_context(text: str) -> str:
    return CAMERA_BLOCK_RE.sub("\n", text).rstrip()


def get_message_index(record: Mapping[str, Any], role: str) -> int | None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    for index, message in enumerate(messages):
        if isinstance(message, Mapping) and message.get("role") == role:
            return index
    return None


def append_to_user(record: dict[str, Any], addition: str) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record has no messages list")
    index = get_message_index(record, "user")
    if index is None:
        raise ValueError("record has no user message")
    content = str(messages[index].get("content", ""))
    content = strip_existing_camera_context(content)
    messages[index]["content"] = content + "\n\n" + addition


def deranged_camera_items(
    keys: list[str],
    gold_items: list[dict[str, Any] | None],
    seed: int,
) -> list[dict[str, Any] | None]:
    indexed = [(idx, key, conflict_key(key), item) for idx, (key, item) in enumerate(zip(keys, gold_items)) if key and item]
    if len(indexed) < 2:
        return gold_items[:]

    rng = random.Random(seed)
    donors = indexed[:]
    for _ in range(100):
        rng.shuffle(donors)
        if all(donor_group != group for (_, _key, group, _), (_, _donor_key, donor_group, _) in zip(indexed, donors)):
            break
    else:
        donors = donors[1:] + donors[:1]

    out = gold_items[:]
    for (target_idx, _target_key, _target_group, _), (_donor_idx, _donor_key, _donor_group, donor_item) in zip(indexed, donors):
        out[target_idx] = donor_item
    return out


def build_variant(
    records: list[Mapping[str, Any]],
    gold_items: list[dict[str, Any] | None],
    variant: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record, camera_item in zip(records, gold_items):
        item = copy.deepcopy(record)
        item["camera_context_variant"] = variant
        if variant != "no_camera":
            append_to_user(item, camera_context_text(camera_item, variant))
        out.append(item)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, help="Detection SFT/test JSON.")
    parser.add_argument("--camera-jsonl", required=True, help="Camera labels JSONL.")
    parser.add_argument("--out-dir", required=True, help="Directory for variant JSON files.")
    parser.add_argument("--prefix", default="detection_test", help="Output filename prefix.")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--drop-missing-camera", action="store_true", help="Keep only records with matched camera labels.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap after camera filtering, for quick pilots.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["no_camera", "gold_camera", "shuffled_camera", "null_camera"],
        choices=["no_camera", "gold_camera", "shuffled_camera", "null_camera"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.input_json)
    if not isinstance(data, list):
        raise ValueError(f"expected a list in {args.input_json}")

    camera = load_camera_jsonl(args.camera_jsonl)
    original_num_records = len(data)
    if args.drop_missing_camera:
        data = [record for record in data if lookup_camera(camera, record_camera_key(record)) is not None]
    if args.max_records and args.max_records > 0:
        data = data[: args.max_records]

    keys = [record_camera_key(record) for record in data]
    gold_items = [lookup_camera(camera, key) for key in keys]
    shuffled_items = deranged_camera_items(keys, gold_items, args.seed)
    camera_items_by_variant = {
        "no_camera": gold_items,
        "gold_camera": gold_items,
        "shuffled_camera": shuffled_items,
        "null_camera": [None for _ in data],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for variant in args.variants:
        variant_records = build_variant(data, camera_items_by_variant[variant], variant)
        out_path = out_dir / f"{args.prefix}_{variant}.json"
        write_json(out_path, variant_records)
        files[variant] = str(out_path)

    missing_camera = sum(1 for item in gold_items if item is None)
    same_shuffled_key = 0
    same_shuffled_conflict_key = 0
    for key, shuffled_item in zip(keys, shuffled_items):
        donor_key = norm_path((shuffled_item or {}).get("path", ""))
        same_shuffled_key += int(bool(key) and key == donor_key)
        same_shuffled_conflict_key += int(bool(key) and conflict_key(key) == conflict_key(donor_key))

    summary = {
        "input_json": str(args.input_json),
        "camera_jsonl": str(args.camera_jsonl),
        "out_dir": str(out_dir),
        "original_num_records": original_num_records,
        "num_records": len(data),
        "drop_missing_camera": bool(args.drop_missing_camera),
        "max_records": args.max_records,
        "camera_rows_loaded": len(camera),
        "missing_camera_records": missing_camera,
        "same_key_in_shuffled_records": same_shuffled_key,
        "same_conflict_key_in_shuffled_records": same_shuffled_conflict_key,
        "variants": files,
    }
    write_json(out_dir / f"{args.prefix}_camera_ablation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
