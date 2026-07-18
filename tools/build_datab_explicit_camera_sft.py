#!/usr/bin/env python3
"""Build paired DataB detection SFT sets with and without camera context.

Both outputs contain the same camera-covered source rows in the original order.
The camera branch changes only the first user message by appending a structured
``labels + caption`` block. System messages, assistant targets, image lists,
and every other field remain byte-for-byte equivalent as Python values.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


CAMERA_OPEN = "<camera_motion>"
CAMERA_CLOSE = "</camera_motion>"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_path(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip()
    while len(text) > 1 and text.endswith("/"):
        text = text[:-1]
    return text


def record_frame_dir(record: Mapping[str, Any]) -> str:
    images = record.get("images")
    if not isinstance(images, list) or not images:
        return ""
    return str(PurePosixPath(normalize_path(images[0])).parent)


def load_camera_rows(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    camera: dict[str, dict[str, Any]] = {}
    rows = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            item = json.loads(line)
            if not isinstance(item, Mapping):
                raise ValueError(f"camera row {line_number} is not an object")
            key = normalize_path(item.get("path"))
            labels = item.get("labels")
            caption = str(item.get("caption", "")).strip()
            if not key:
                raise ValueError(f"camera row {line_number} has no path")
            if not isinstance(labels, list) or not any(str(label).strip() for label in labels):
                raise ValueError(f"camera row {line_number} has no labels: {key}")
            if not caption:
                raise ValueError(f"camera row {line_number} has no caption: {key}")
            normalized = {
                "path": key,
                "labels": [str(label).strip() for label in labels if str(label).strip()],
                "caption": caption,
            }
            if key in camera and camera[key] != normalized:
                raise ValueError(f"conflicting duplicate camera rows for {key}")
            camera[key] = normalized
    return camera, rows


def lookup_camera(
    camera: Mapping[str, dict[str, Any]], frame_dir: str
) -> tuple[str, dict[str, Any]] | None:
    key = normalize_path(frame_dir)
    if not key:
        return None
    if key in camera:
        return key, camera[key]
    current = PurePosixPath(key)
    for _ in range(4):
        current = current.parent
        parent = str(current)
        if parent in camera:
            return parent, camera[parent]
    return None


def role_indices(record: Mapping[str, Any], role: str) -> list[int]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    return [
        index
        for index, message in enumerate(messages)
        if isinstance(message, Mapping) and message.get("role") == role
    ]


def answer_label(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return "Unknown"
    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = str(message.get("content", "")).lower()
        if "<answer>fake</answer>" in content:
            return "Fake"
        if "<answer>real</answer>" in content:
            return "Real"
    return "Unknown"


def camera_block(camera_item: Mapping[str, Any]) -> str:
    labels = "; ".join(str(label).strip() for label in camera_item["labels"])
    caption = str(camera_item["caption"]).strip()
    return "\n".join(
        [
            CAMERA_OPEN,
            f"<labels>{labels}</labels>",
            f"<caption>{caption}</caption>",
            CAMERA_CLOSE,
        ]
    )


def append_camera_context(record: Mapping[str, Any], camera_item: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(record)
    messages = output.get("messages")
    user_indices = role_indices(output, "user")
    if not isinstance(messages, list) or len(user_indices) != 1:
        raise ValueError(f"expected exactly one user message, found {len(user_indices)}")
    user_index = user_indices[0]
    original = str(messages[user_index].get("content", ""))
    if CAMERA_OPEN in original or CAMERA_CLOSE in original:
        raise ValueError("source user prompt already contains a camera_motion block")
    messages[user_index]["content"] = original.rstrip() + "\n\n" + camera_block(camera_item)
    return output


def records_equal_except_camera_user(
    baseline: Mapping[str, Any], camera_record: Mapping[str, Any]
) -> bool:
    base = copy.deepcopy(dict(baseline))
    conditioned = copy.deepcopy(dict(camera_record))
    base_messages = base.get("messages")
    conditioned_messages = conditioned.get("messages")
    base_users = role_indices(base, "user")
    conditioned_users = role_indices(conditioned, "user")
    if (
        not isinstance(base_messages, list)
        or not isinstance(conditioned_messages, list)
        or len(base_users) != 1
        or base_users != conditioned_users
    ):
        return False
    user_index = base_users[0]
    original_user = str(base_messages[user_index].get("content", "")).rstrip()
    conditioned_user = str(conditioned_messages[user_index].get("content", ""))
    if not conditioned_user.startswith(original_user + "\n\n" + CAMERA_OPEN):
        return False
    conditioned_messages[user_index]["content"] = base_messages[user_index].get("content", "")
    return base == conditioned


def build_paired_datasets(
    detection_records: Sequence[Mapping[str, Any]],
    camera: Mapping[str, dict[str, Any]],
    check_images: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    no_camera: list[dict[str, Any]] = []
    with_camera: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    missing_dirs: list[str] = []
    matched_camera_paths: set[str] = set()
    matched_frame_dirs: set[str] = set()
    unique_images: set[str] = set()

    for source_index, source in enumerate(detection_records):
        if not isinstance(source, Mapping):
            raise ValueError(f"detection record {source_index} is not an object")
        user_count = len(role_indices(source, "user"))
        assistant_count = len(role_indices(source, "assistant"))
        if user_count != 1 or assistant_count < 1:
            raise ValueError(
                f"detection record {source_index} requires one user and at least one assistant; "
                f"found user={user_count}, assistant={assistant_count}"
            )
        frame_dir = record_frame_dir(source)
        matched = lookup_camera(camera, frame_dir)
        if matched is None:
            missing_dirs.append(frame_dir)
            continue
        camera_path, camera_item = matched
        baseline = copy.deepcopy(dict(source))
        conditioned = append_camera_context(source, camera_item)
        if not records_equal_except_camera_user(baseline, conditioned):
            raise AssertionError(f"paired integrity failed at source index {source_index}")
        no_camera.append(baseline)
        with_camera.append(conditioned)
        matched_camera_paths.add(camera_path)
        matched_frame_dirs.add(frame_dir)
        images = source.get("images", [])
        if isinstance(images, list):
            unique_images.update(str(path) for path in images)
        manifest.append(
            {
                "pair_index": len(no_camera) - 1,
                "source_index": source_index,
                "frame_dir": frame_dir,
                "camera_path": camera_path,
                "answer": answer_label(source),
                "labels": list(camera_item["labels"]),
            }
        )

    if not no_camera:
        raise ValueError("no DataB detection records matched camera labels")
    if check_images:
        missing_images = [path for path in sorted(unique_images) if not Path(path).is_file()]
        if missing_images:
            raise FileNotFoundError(
                f"missing {len(missing_images)}/{len(unique_images)} matched images; "
                f"first={missing_images[0]}"
            )

    paired_integrity = all(
        records_equal_except_camera_user(baseline, conditioned)
        for baseline, conditioned in zip(no_camera, with_camera)
    )
    summary = {
        "detection_records": len(detection_records),
        "matched_records": len(no_camera),
        "missing_records": len(detection_records) - len(no_camera),
        "coverage": len(no_camera) / len(detection_records),
        "unique_detection_frame_dirs": len({record_frame_dir(row) for row in detection_records}),
        "matched_unique_frame_dirs": len(matched_frame_dirs),
        "matched_unique_camera_paths": len(matched_camera_paths),
        "missing_unique_frame_dirs": len(set(missing_dirs)),
        "answer_counts": dict(Counter(answer_label(row) for row in no_camera)),
        "unique_images": len(unique_images),
        "paired_integrity": paired_integrity,
        "no_camera_prompts_contain_camera_block": any(
            CAMERA_OPEN in str(row.get("messages", "")) for row in no_camera
        ),
        "camera_prompts_with_exactly_one_block": all(
            str(row.get("messages", "")).count(CAMERA_OPEN) == 1
            and str(row.get("messages", "")).count(CAMERA_CLOSE) == 1
            for row in with_camera
        ),
        "first_missing_frame_dirs": list(dict.fromkeys(missing_dirs))[:20],
    }
    if not paired_integrity:
        raise AssertionError("paired branches differ outside the appended camera user block")
    return no_camera, with_camera, manifest, summary


def enforce_expected(name: str, actual: int, expected: int) -> None:
    if expected > 0 and actual != expected:
        raise ValueError(f"unexpected {name}: expected {expected}, got {actual}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-json", type=Path, required=True)
    parser.add_argument("--camera-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-detection-records", type=int, default=6766)
    parser.add_argument("--expected-camera-records", type=int, default=5639)
    parser.add_argument("--expected-matched-records", type=int, default=5739)
    parser.add_argument("--check-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detection = read_json(args.detection_json)
    if not isinstance(detection, list):
        raise ValueError(f"detection JSON must be a list: {args.detection_json}")
    camera, camera_rows = load_camera_rows(args.camera_jsonl)
    enforce_expected("detection records", len(detection), args.expected_detection_records)
    enforce_expected("camera records", camera_rows, args.expected_camera_records)

    no_camera, with_camera, manifest, counts = build_paired_datasets(
        detection, camera, check_images=args.check_images
    )
    enforce_expected("matched records", len(no_camera), args.expected_matched_records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    no_camera_path = args.out_dir / "datab_sft_no_camera_5739.json"
    with_camera_path = args.out_dir / "datab_sft_with_camera_labels_caption_5739.json"
    manifest_path = args.out_dir / "datab_sft_pair_manifest.jsonl"
    write_json(no_camera_path, no_camera)
    write_json(with_camera_path, with_camera)
    write_jsonl(manifest_path, manifest)

    summary = {
        "schema_version": "datab_explicit_camera_sft_v1",
        "question": (
            "Does appending matched CameraBench labels and caption to the unchanged DataB "
            "detection user prompt improve final Real/Fake detection?"
        ),
        "inputs": {
            "detection_json": str(args.detection_json),
            "camera_jsonl": str(args.camera_jsonl),
            "detection_sha256": sha256(args.detection_json),
            "camera_sha256": sha256(args.camera_jsonl),
        },
        "outputs": {
            "no_camera_json": str(no_camera_path),
            "with_camera_json": str(with_camera_path),
            "pair_manifest_jsonl": str(manifest_path),
            "no_camera_sha256": sha256(no_camera_path),
            "with_camera_sha256": sha256(with_camera_path),
            "pair_manifest_sha256": sha256(manifest_path),
        },
        "counts": {**counts, "camera_records": camera_rows, "unique_camera_paths": len(camera)},
        "single_changed_factor": (
            "Only the matched labels+caption camera_motion block is appended to the first user "
            "message. System/assistant/images and all other fields are unchanged."
        ),
        "camera_context_template": (
            "<camera_motion>\\n<labels>...</labels>\\n<caption>...</caption>\\n"
            "</camera_motion>"
        ),
        "images_checked": bool(args.check_images),
    }
    summary_path = args.out_dir / "datab_explicit_camera_sft_data_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
