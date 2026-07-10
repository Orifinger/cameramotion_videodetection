#!/usr/bin/env python3
"""Build leakage-audited datasets for the camera-pretext GRPO gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from build_dataa_camera_context_ablation import (
    conflict_key,
    load_camera_jsonl,
    load_json,
    lookup_camera,
    norm_path,
    normalize_labels,
    record_camera_key,
)

CAMERA_LABEL_ORDER = [
    "very-unsteady",
    "unsteady",
    "minimal-shaking",
    "no-shaking",
    "complex-motion",
    "minor-motion",
    "no-motion",
    "fast-speed",
    "regular-speed",
    "slow-speed",
    "dolly-in",
    "dolly-out",
    "truck-left",
    "truck-right",
    "pedestal-up",
    "pedestal-down",
    "pan-left",
    "pan-right",
    "tilt-up",
    "tilt-down",
    "roll-CW",
    "roll-CCW",
    "zoom-in",
    "zoom-out",
    "arc-CW",
    "arc-CCW",
    "side-tracking",
    "lead-tracking",
    "tail-tracking",
    "aerial-tracking",
    "arc-tracking",
    "pan-tracking",
    "tilt-tracking",
]
CAMERA_LABEL_LOOKUP = {label.casefold().replace("_", "-"): label for label in CAMERA_LABEL_ORDER}
EXCLUDED_CAMERA_LABELS = {"static"}
MOTION_BUCKET_LABELS = ("no-motion", "minor-motion", "complex-motion")
DATAA_CASE_RE = re.compile(r"(dataA_v1_(?:dataset_v2_|textedit_reserve_)?\d+)")

CAMERA_SYSTEM_PROMPT = (
    "You are a camera-motion analyst. Infer only the global camera behavior "
    "visible across the ordered frames. Do not judge whether the video is real or fake."
)
CAMERA_PROMPT_TEMPLATES = [
    "Inspect the ordered frames and identify every applicable camera-motion label.",
    "Determine the global camera movement, steadiness, speed, and tracking primitives.",
    "Classify the camera behavior shown across this frame sequence.",
    "Recognize all camera-motion primitives that are supported by the ordered frames.",
    "Analyze how the camera moves through the sequence and return the matching labels.",
    "Identify the camera trajectory and motion attributes, ignoring object authenticity.",
    "Read the temporal frame sequence and classify its global camera motion.",
    "Return the complete set of camera-motion labels supported by this video.",
]
CAMERA_OUTPUT_INSTRUCTION = (
    'Return exactly one JSON list inside <camera_motion>...</camera_motion>, for example: '
    '<camera_motion>["no-shaking", "no-motion", "regular-speed"]</camera_motion>. '
    "Use only labels from the requested taxonomy and do not add an explanation."
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list: {path}")
    records = [item for item in payload if isinstance(item, Mapping)]
    if len(records) != len(payload):
        raise ValueError(f"non-object records found in {path}")
    return [dict(item) for item in records]


def first_image(record: Mapping[str, Any]) -> str:
    images = record.get("images")
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes)) and images:
        return norm_path(images[0])
    return ""


def record_split(record: Mapping[str, Any]) -> str:
    key = record_camera_key(record)
    name = PurePosixPath(key).name.lower() if key else ""
    if name in {"real", "fake"}:
        return name
    answer = detection_label(record).lower()
    return answer if answer in {"real", "fake"} else "unknown"


def source_group(record: Mapping[str, Any], index: int = 0) -> str:
    key = conflict_key(record_camera_key(record))
    return key or f"record-{index:06d}"


def sample_id(record: Mapping[str, Any], index: int = 0) -> str:
    group = source_group(record, index)
    split = record_split(record)
    return f"{group}:{split}"


def get_messages(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record is missing messages")
    return [dict(message) for message in messages if isinstance(message, Mapping)]


def assistant_content(record: Mapping[str, Any]) -> str:
    for message in reversed(get_messages(record)):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def detection_label(record: Mapping[str, Any]) -> str:
    split = PurePosixPath(record_camera_key(record)).name.lower() if record_camera_key(record) else ""
    if split == "fake":
        return "Fake"
    if split == "real":
        return "Real"
    text = assistant_content(record).lower()
    fake = "<answer>fake</answer>" in text
    real = "<answer>real</answer>" in text
    if fake and not real:
        return "Fake"
    if real and not fake:
        return "Real"
    return "UNKNOWN"


def prompt_messages(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    messages = get_messages(record)
    kept = [message for message in messages if message.get("role") != "assistant"]
    if not kept:
        raise ValueError("record has no prompt messages")
    return kept


def stable_template_index(value: str, seed: int) -> int:
    token = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(token[:4], "big") % len(CAMERA_PROMPT_TEMPLATES)


def camera_user_prompt(num_images: int, key: str, seed: int) -> str:
    if num_images <= 0:
        raise ValueError("camera pretext record has no images")
    template = CAMERA_PROMPT_TEMPLATES[stable_template_index(key, seed)]
    frames = "\n".join(f"Frame {index + 1}: <image>" for index in range(num_images))
    taxonomy = ", ".join(CAMERA_LABEL_ORDER)
    return (
        f"{template}\n\nOrdered frames:\n{frames}\n\n"
        f"Allowed labels: {taxonomy}\n\n{CAMERA_OUTPUT_INSTRUCTION}"
    )


def canonical_camera_labels(value: Any) -> tuple[list[str], list[str], int]:
    canonical: set[str] = set()
    unknown: list[str] = []
    excluded = 0
    for raw in normalize_labels(value):
        cleaned = raw.strip()
        folded = cleaned.casefold().replace("_", "-")
        if folded in {item.casefold() for item in EXCLUDED_CAMERA_LABELS}:
            excluded += 1
            continue
        label = CAMERA_LABEL_LOOKUP.get(folded)
        if label is None:
            unknown.append(cleaned)
            continue
        canonical.add(label)
    ordered = [label for label in CAMERA_LABEL_ORDER if label in canonical]
    return ordered, unknown, excluded


def motion_bucket(labels: Sequence[str]) -> str:
    present = set(labels)
    for label in reversed(MOTION_BUCKET_LABELS):
        if label in present:
            return label
    return "unknown"


def group_records(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[source_group(record, index)].append(dict(record))
    return dict(grouped)


def derive_group_split(
    records: Sequence[Mapping[str, Any]],
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = list(group_records(records).items())
    random.Random(seed).shuffle(groups)
    num_test = max(1, round(len(groups) * test_ratio))
    test_keys = {key for key, _ in groups[:num_test]}
    train = [dict(record) for key, rows in groups if key not in test_keys for record in rows]
    test = [dict(record) for key, rows in groups if key in test_keys for record in rows]
    return train, test


def resolve_dataa_split(
    full_records: list[dict[str, Any]],
    train_json: str | None,
    test_json: str | None,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    train = load_records(train_json) if train_json else None
    test = load_records(test_json) if test_json else None
    full_groups = group_records(full_records)

    if train is None and test is None:
        train, test = derive_group_split(full_records, test_ratio, seed)
        source = f"derived_group_split:{test_ratio}"
    elif train is None:
        assert test is not None
        test_groups = set(group_records(test))
        train = [record for key, rows in full_groups.items() if key not in test_groups for record in rows]
        source = "derived_train_from_full_minus_test"
    elif test is None:
        train_groups = set(group_records(train))
        test = [record for key, rows in full_groups.items() if key not in train_groups for record in rows]
        source = "derived_test_from_full_minus_train"
    else:
        source = "explicit_train_and_test"

    assert train is not None and test is not None
    train_groups = set(group_records(train))
    test_groups = set(group_records(test))
    overlap = sorted(train_groups & test_groups)
    if overlap:
        raise ValueError(f"DataA train/test source-group leakage: {overlap[:20]}")
    if not train or not test:
        raise ValueError("DataA train and test must both be non-empty")
    return train, test, source


def balanced_take(
    records: Sequence[dict[str, Any]],
    count: int,
    bucket_fn,
    seed: int,
) -> list[dict[str, Any]]:
    rows = [dict(record) for record in records]
    if count <= 0 or count >= len(rows):
        random.Random(seed).shuffle(rows)
        return rows

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in rows:
        buckets[str(bucket_fn(record))].append(record)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    names = sorted(buckets)
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for name in names:
            if buckets[name] and len(selected) < count:
                selected.append(buckets[name].pop())
                progressed = True
        if not progressed:
            break
    return selected


def build_camera_record(
    record: Mapping[str, Any],
    camera_item: Mapping[str, Any],
    index: int,
    seed: int,
) -> tuple[dict[str, Any], list[str], int]:
    images = [str(path) for path in record.get("images", [])]
    labels, unknown, excluded = canonical_camera_labels(camera_item.get("labels"))
    if unknown:
        raise ValueError(f"unknown camera labels for {record_camera_key(record)}: {unknown}")
    if not labels:
        raise ValueError(f"empty camera target after filtering: {record_camera_key(record)}")
    sid = sample_id(record, index)
    solution = f"<camera_motion>{json.dumps(labels, ensure_ascii=False)}</camera_motion>"
    output = {
        "messages": [
            {"role": "system", "content": CAMERA_SYSTEM_PROMPT},
            {"role": "user", "content": camera_user_prompt(len(images), sid, seed)},
        ],
        "images": images,
        "solution": solution,
        "camera_labels": labels,
        "task_type": "camera_motion",
        "sample_id": sid,
        "source_group": source_group(record, index),
        "motion_bucket": motion_bucket(labels),
    }
    return output, labels, excluded


def build_detection_record(
    record: Mapping[str, Any],
    index: int,
    camera: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    label = detection_label(record)
    if label not in {"Real", "Fake"}:
        raise ValueError(f"unable to infer detection label: {first_image(record)}")
    labels: list[str] = []
    if camera is not None:
        item = lookup_camera(camera, record_camera_key(record))
        if item:
            labels, _unknown, _excluded = canonical_camera_labels(item.get("labels"))
    return {
        "messages": prompt_messages(record),
        "images": [str(path) for path in record.get("images", [])],
        "solution": f"<answer>{label}</answer>",
        "label": label,
        "task_type": "detection",
        "sample_id": sample_id(record, index),
        "source_group": source_group(record, index),
        "motion_bucket": motion_bucket(labels),
    }


def label_counts(records: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get(key, "unknown")) for record in records).items()))


def camera_label_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        counts.update(record.get("camera_labels", []))
    return {label: counts[label] for label in CAMERA_LABEL_ORDER if counts[label]}


def missing_camera_examples(
    records: Sequence[Mapping[str, Any]],
    camera: Mapping[str, dict[str, Any]],
    limit: int = 20,
) -> list[str]:
    out = []
    for record in records:
        key = record_camera_key(record)
        if lookup_camera(camera, key) is None:
            out.append(key)
            if len(out) >= limit:
                break
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataa-detection-json", required=True)
    parser.add_argument("--dataa-camera-jsonl", required=True)
    parser.add_argument("--datab-detection-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataa-train-json")
    parser.add_argument("--dataa-test-json")
    parser.add_argument("--derived-test-ratio", type=float, default=0.30)
    parser.add_argument("--camera-max-samples", type=int, default=0)
    parser.add_argument("--dataa-detection-max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--check-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.derived_test_ratio < 1.0:
        raise ValueError("--derived-test-ratio must be in (0, 1)")

    dataa_full = load_records(args.dataa_detection_json)
    datab_full = load_records(args.datab_detection_json)
    camera = load_camera_jsonl(args.dataa_camera_jsonl)
    dataa_train, dataa_test, split_source = resolve_dataa_split(
        dataa_full,
        args.dataa_train_json,
        args.dataa_test_json,
        args.derived_test_ratio,
        args.seed,
    )

    train_groups = set(group_records(dataa_train))
    test_groups = set(group_records(dataa_test))
    if train_groups & test_groups:
        raise AssertionError("DataA source-group overlap survived split resolution")

    camera_train_candidates: list[dict[str, Any]] = []
    camera_eval_candidates: list[dict[str, Any]] = []
    for rows, target in ((dataa_train, camera_train_candidates), (dataa_test, camera_eval_candidates)):
        for record in rows:
            if record_split(record) != "real":
                continue
            if lookup_camera(camera, record_camera_key(record)) is not None:
                target.append(record)

    if not camera_train_candidates or not camera_eval_candidates:
        raise ValueError("camera pretext train/eval records are empty")

    camera_train_candidates = balanced_take(
        camera_train_candidates,
        args.camera_max_samples,
        lambda record: motion_bucket(
            canonical_camera_labels(lookup_camera(camera, record_camera_key(record)).get("labels"))[0]
        ),
        args.seed + 1,
    )

    camera_train: list[dict[str, Any]] = []
    camera_eval: list[dict[str, Any]] = []
    excluded_static = 0
    for index, record in enumerate(camera_train_candidates):
        built, _labels, excluded = build_camera_record(
            record, lookup_camera(camera, record_camera_key(record)), index, args.seed
        )
        camera_train.append(built)
        excluded_static += excluded
    for index, record in enumerate(camera_eval_candidates):
        built, _labels, excluded = build_camera_record(
            record, lookup_camera(camera, record_camera_key(record)), index, args.seed + 1000
        )
        camera_eval.append(built)
        excluded_static += excluded

    dataa_common_candidates = balanced_take(
        dataa_train,
        args.dataa_detection_max_samples,
        detection_label,
        args.seed + 2,
    )
    dataa_common = [
        build_detection_record(record, index, camera)
        for index, record in enumerate(dataa_common_candidates)
    ]

    replay_candidates = [
        record for record in datab_full if detection_label(record) in {"Real", "Fake"}
    ]
    replay_selected = balanced_take(
        replay_candidates,
        len(camera_train),
        detection_label,
        args.seed + 3,
    )
    if len(replay_selected) != len(camera_train):
        raise ValueError(
            f"DataB replay count {len(replay_selected)} != camera train count {len(camera_train)}"
        )
    datab_replay = [
        build_detection_record(record, index)
        for index, record in enumerate(replay_selected)
    ]

    if args.check_images:
        missing_images = []
        for record in camera_train + camera_eval + dataa_common + datab_replay:
            for image in record.get("images", []):
                if not Path(image).exists():
                    missing_images.append(image)
                    if len(missing_images) >= 20:
                        break
            if len(missing_images) >= 20:
                break
        if missing_images:
            raise FileNotFoundError(f"missing image examples: {missing_images}")

    train_pretext_groups = {record["source_group"] for record in camera_train}
    eval_pretext_groups = {record["source_group"] for record in camera_eval}
    if train_pretext_groups & eval_pretext_groups:
        raise AssertionError("camera pretext train/eval source-group leakage")
    if any(not str(record["sample_id"]).endswith(":real") for record in camera_train):
        raise AssertionError("camera pretext training contains non-real records")
    if len(datab_replay) != len(camera_train):
        raise AssertionError("first-stage record counts are not matched")

    out_dir = Path(args.out_dir)
    outputs = {
        "camera_pretext_grpo_train": out_dir / "camera_pretext_grpo_train.json",
        "camera_pretext_eval": out_dir / "camera_pretext_eval.json",
        "detection_replay_control_grpo": out_dir / "detection_replay_control_grpo.json",
        "dataa_detection_common_grpo": out_dir / "dataa_detection_common_grpo.json",
    }
    payloads = {
        "camera_pretext_grpo_train": camera_train,
        "camera_pretext_eval": camera_eval,
        "detection_replay_control_grpo": datab_replay,
        "dataa_detection_common_grpo": dataa_common,
    }
    for name, path in outputs.items():
        write_json(path, payloads[name])

    summary = {
        "schema_version": "camera_pretext_grpo_gate_v1",
        "seed": args.seed,
        "inputs": {
            "dataa_detection_json": str(args.dataa_detection_json),
            "dataa_camera_jsonl": str(args.dataa_camera_jsonl),
            "datab_detection_json": str(args.datab_detection_json),
            "dataa_train_json": args.dataa_train_json,
            "dataa_test_json": args.dataa_test_json,
        },
        "split_source": split_source,
        "counts": {
            "dataa_full": len(dataa_full),
            "dataa_train": len(dataa_train),
            "dataa_test": len(dataa_test),
            "dataa_camera_rows": len(camera),
            "camera_train_real_only": len(camera_train),
            "camera_eval_real_only": len(camera_eval),
            "datab_replay": len(datab_replay),
            "dataa_detection_common": len(dataa_common),
            "excluded_static_labels": excluded_static,
        },
        "leakage_audit": {
            "dataa_train_test_group_overlap": sorted(train_groups & test_groups),
            "camera_train_eval_group_overlap": sorted(train_pretext_groups & eval_pretext_groups),
            "camera_train_all_real": all(
                str(record["sample_id"]).endswith(":real") for record in camera_train
            ),
            "first_stage_counts_matched": len(datab_replay) == len(camera_train),
        },
        "camera_train_motion_buckets": label_counts(camera_train, "motion_bucket"),
        "camera_eval_motion_buckets": label_counts(camera_eval, "motion_bucket"),
        "camera_train_label_counts": camera_label_counts(camera_train),
        "camera_eval_label_counts": camera_label_counts(camera_eval),
        "dataa_detection_label_counts": label_counts(dataa_common, "label"),
        "dataa_detection_motion_buckets": label_counts(dataa_common, "motion_bucket"),
        "datab_replay_label_counts": label_counts(datab_replay, "label"),
        "missing_camera_examples_train": missing_camera_examples(dataa_train, camera),
        "missing_camera_examples_test": missing_camera_examples(dataa_test, camera),
        "outputs": {
            name: {
                "path": str(path),
                "records": len(payloads[name]),
                "sha256": file_sha256(path),
            }
            for name, path in outputs.items()
        },
    }
    summary_path = out_dir / "camera_pretext_grpo_sets_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
