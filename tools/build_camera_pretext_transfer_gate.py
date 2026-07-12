#!/usr/bin/env python3
"""Build the correct/shuffled camera-pretext SFT gate from final DataA records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CASE_RE = re.compile(r"(dataA_v1_(?:dataset_v2_|textedit_reserve_)?\d+)/(real|fake)")
CAMERA_LABEL_ORDER = [
    "very-unsteady", "unsteady", "minimal-shaking", "no-shaking",
    "complex-motion", "minor-motion", "no-motion",
    "fast-speed", "regular-speed", "slow-speed",
    "dolly-in", "dolly-out", "truck-left", "truck-right",
    "pedestal-up", "pedestal-down", "pan-left", "pan-right",
    "tilt-up", "tilt-down", "roll-CW", "roll-CCW", "zoom-in", "zoom-out",
    "arc-CW", "arc-CCW", "side-tracking", "lead-tracking", "tail-tracking",
    "aerial-tracking", "arc-tracking", "pan-tracking", "tilt-tracking",
]
LABEL_LOOKUP = {label.casefold().replace("_", "-"): label for label in CAMERA_LABEL_ORDER}
EXCLUDED_LABELS = {"static"}
MOTION_BUCKETS = ("complex-motion", "minor-motion", "no-motion")
SHUFFLED_LABEL_MAP = {
    "very-unsteady": "unsteady", "unsteady": "minimal-shaking",
    "minimal-shaking": "no-shaking", "no-shaking": "very-unsteady",
    "complex-motion": "minor-motion", "minor-motion": "no-motion", "no-motion": "complex-motion",
    "fast-speed": "regular-speed", "regular-speed": "slow-speed", "slow-speed": "fast-speed",
    "dolly-in": "dolly-out", "dolly-out": "dolly-in",
    "truck-left": "truck-right", "truck-right": "truck-left",
    "pedestal-up": "pedestal-down", "pedestal-down": "pedestal-up",
    "pan-left": "pan-right", "pan-right": "pan-left",
    "tilt-up": "tilt-down", "tilt-down": "tilt-up",
    "roll-CW": "roll-CCW", "roll-CCW": "roll-CW",
    "zoom-in": "zoom-out", "zoom-out": "zoom-in",
    "arc-CW": "arc-CCW", "arc-CCW": "arc-CW",
    "side-tracking": "lead-tracking", "lead-tracking": "tail-tracking",
    "tail-tracking": "aerial-tracking", "aerial-tracking": "arc-tracking",
    "arc-tracking": "pan-tracking", "pan-tracking": "tilt-tracking",
    "tilt-tracking": "side-tracking",
}
CAMERA_SYSTEM_PROMPT = (
    "You are a camera-motion analyst. Infer only the global camera behavior visible across "
    "the ordered video frames. Do not judge whether the video is real or fake."
)
CANONICAL_REQUEST = "Analyze the ordered frames and identify every applicable global camera-motion label."
PARAPHRASED_REQUEST = "Review this frame sequence and classify the global behavior of the camera."


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, Mapping):
                rows.append(dict(row))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_path(value: Any) -> str:
    return str(value or "").replace("\\", "/")


def first_image(record: Mapping[str, Any]) -> str:
    images = record.get("images")
    return normalized_path(images[0]) if isinstance(images, list) and images else ""


def identity_from_path(value: Any) -> tuple[str, str] | None:
    match = CASE_RE.search(normalized_path(value))
    return (match.group(1), match.group(2)) if match else None


def identity(record: Mapping[str, Any]) -> tuple[str, str] | None:
    return identity_from_path(first_image(record))


def canonical_labels(values: Any) -> tuple[list[str], list[str]]:
    if isinstance(values, str):
        raw_values: Sequence[Any] = [values]
    elif isinstance(values, Sequence):
        raw_values = values
    else:
        raw_values = []
    selected: set[str] = set()
    unknown: list[str] = []
    for raw in raw_values:
        cleaned = str(raw).strip()
        folded = cleaned.casefold().replace("_", "-")
        if not cleaned or folded in EXCLUDED_LABELS:
            continue
        label = LABEL_LOOKUP.get(folded)
        if label is None:
            unknown.append(cleaned)
        else:
            selected.add(label)
    return [label for label in CAMERA_LABEL_ORDER if label in selected], unknown


def motion_bucket(labels: Sequence[str]) -> str:
    present = set(labels)
    for label in MOTION_BUCKETS:
        if label in present:
            return label
    return "unknown"


def source_family(case_id: str) -> str:
    if "textedit_reserve" in case_id:
        return "vace13b_textedit_40step_v3"
    if "dataset_v2" in case_id:
        return "vace13b_dataset_40step_v3"
    return "vace14b_reused"


def load_real_detection(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list: {path}")
    output: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, Mapping):
            continue
        item_identity = identity(row)
        if item_identity and item_identity[1] == "real":
            case_id = item_identity[0]
            if case_id in output:
                raise ValueError(f"duplicate real detection record: {case_id}")
            output[case_id] = dict(row)
    if not output:
        raise ValueError("no DataA real records found")
    return output


def load_dev_ids(path: str | Path) -> set[str]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list: {path}")
    output = {item[0] for row in payload if isinstance(row, Mapping) and (item := identity(row))}
    if not output:
        raise ValueError("no DataA identities found in development split")
    return output


def load_camera_labels(path: str | Path) -> tuple[dict[str, list[str]], list[str]]:
    output: dict[str, list[str]] = {}
    unknown: list[str] = []
    for row in read_jsonl(path):
        item_identity = identity_from_path(row.get("path"))
        if not item_identity or item_identity[1] != "real":
            continue
        labels, row_unknown = canonical_labels(row.get("labels"))
        unknown.extend(row_unknown)
        if labels:
            existing = output.get(item_identity[0])
            if existing is not None and existing != labels:
                raise ValueError(
                    f"conflicting real camera labels for {item_identity[0]}: {existing} vs {labels}"
                )
            output[item_identity[0]] = labels
    return output, sorted(set(unknown))


def camera_prompt(num_frames: int, variant: str) -> str:
    request = CANONICAL_REQUEST if variant == "canonical" else PARAPHRASED_REQUEST
    frames = "\n".join(f"Frame {index + 1}: <image>" for index in range(num_frames))
    taxonomy = ", ".join(CAMERA_LABEL_ORDER)
    return (
        f"{request}\n\nOrdered frames:\n{frames}\n\nAllowed labels: {taxonomy}\n\n"
        "Return exactly one JSON list inside <camera_motion>...</camera_motion>. "
        "Use only allowed labels and do not add an explanation."
    )


def target_text(labels: Sequence[str]) -> str:
    return f"<camera_motion>{json.dumps(list(labels), ensure_ascii=False)}</camera_motion>"


def build_record(
    case_id: str,
    detection: Mapping[str, Any],
    labels: Sequence[str],
    prompt_variant: str,
    target_kind: str,
) -> dict[str, Any]:
    images = [normalized_path(path) for path in detection.get("images", [])]
    if not images:
        raise ValueError(f"record has no images: {case_id}")
    return {
        "messages": [
            {"role": "system", "content": CAMERA_SYSTEM_PROMPT},
            {"role": "user", "content": camera_prompt(len(images), prompt_variant)},
        ],
        "images": images,
        "target_text": target_text(labels),
        "camera_labels": list(labels),
        "assistant_prefix": "",
        "case_id": case_id,
        "sample_id": f"{case_id}:real:{prompt_variant}",
        "source_family": source_family(case_id),
        "motion_bucket": motion_bucket(labels),
        "prompt_variant": prompt_variant,
        "target_kind": target_kind,
    }


def permuted_labels(labels: Sequence[str]) -> list[str]:
    mapped = {SHUFFLED_LABEL_MAP[label] for label in labels}
    return [label for label in CAMERA_LABEL_ORDER if label in mapped]


def counts(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "unknown")) for row in rows).items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataa-detection-json", required=True)
    parser.add_argument("--dataa-camera-jsonl", required=True)
    parser.add_argument("--dataa-dev-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real_records = load_real_detection(args.dataa_detection_json)
    dev_ids = load_dev_ids(args.dataa_dev_json)
    labels_by_case, unknown_labels = load_camera_labels(args.dataa_camera_jsonl)
    if unknown_labels:
        raise ValueError(f"unknown camera labels: {unknown_labels}")
    missing_dev = sorted(dev_ids - real_records.keys())
    if missing_dev:
        raise ValueError(f"development identities absent from final detection data: {missing_dev[:20]}")
    all_ids = set(real_records)
    train_ids = sorted(all_ids - dev_ids)
    dev_ids_sorted = sorted(dev_ids)
    missing_camera_train = sorted(set(train_ids) - labels_by_case.keys())
    missing_camera_dev = sorted(set(dev_ids_sorted) - labels_by_case.keys())
    eligible_train = [case_id for case_id in train_ids if case_id in labels_by_case]
    eligible_dev = [case_id for case_id in dev_ids_sorted if case_id in labels_by_case]
    correct = [
        build_record(case_id, real_records[case_id], labels_by_case[case_id], "canonical", "correct")
        for case_id in eligible_train
    ]
    shuffled_targets = [permuted_labels(row["camera_labels"]) for row in correct]
    shuffled = [
        build_record(row["case_id"], real_records[row["case_id"]], labels, "canonical", "shuffled")
        for row, labels in zip(correct, shuffled_targets)
    ]
    dev_canonical = [
        build_record(case_id, real_records[case_id], labels_by_case[case_id], "canonical", "gold")
        for case_id in eligible_dev
    ]
    dev_paraphrased = [
        build_record(case_id, real_records[case_id], labels_by_case[case_id], "paraphrased", "gold")
        for case_id in eligible_dev
    ]
    if args.check_images:
        missing_images = [
            image for row in correct + dev_canonical for image in row["images"] if not Path(image).is_file()
        ]
        if missing_images:
            raise FileNotFoundError(f"missing image examples: {missing_images[:20]}")
    if {row["case_id"] for row in correct} & {row["case_id"] for row in dev_canonical}:
        raise AssertionError("camera train/dev identity leakage")
    if any(left["camera_labels"] == right["camera_labels"] for left, right in zip(correct, shuffled)):
        raise AssertionError("shuffled control retained a correct target")
    if any(len(left["camera_labels"]) != len(right["camera_labels"]) for left, right in zip(correct, shuffled)):
        raise AssertionError("shuffled control changed the number of labels in a sample")

    out_dir = Path(args.out_dir)
    payloads = {
        "camera_train_correct.jsonl": correct,
        "camera_train_shuffled.jsonl": shuffled,
        "camera_dev_canonical.jsonl": dev_canonical,
        "camera_dev_paraphrased.jsonl": dev_paraphrased,
    }
    output_info: dict[str, Any] = {}
    for name, rows in payloads.items():
        path = out_dir / name
        write_jsonl(path, rows)
        output_info[name] = {"path": str(path), "records": len(rows), "sha256": file_sha256(path)}
    summary = {
        "schema_version": "camera_pretext_transfer_gate_v1",
        "seed": args.seed,
        "inputs": {
            "dataa_detection_json": args.dataa_detection_json,
            "dataa_camera_jsonl": args.dataa_camera_jsonl,
            "dataa_dev_json": args.dataa_dev_json,
        },
        "counts": {
            "complete_real_cases": len(all_ids),
            "train_complement_cases": len(train_ids),
            "dev_cases": len(dev_ids_sorted),
            "eligible_camera_train": len(correct),
            "eligible_camera_dev": len(dev_canonical),
        },
        "missing_camera_train": missing_camera_train,
        "missing_camera_dev": missing_camera_dev,
        "train_dev_overlap": [],
        "correct_shuffled_same_cases_and_images": all(
            left["case_id"] == right["case_id"] and left["images"] == right["images"]
            for left, right in zip(correct, shuffled)
        ),
        "correct_shuffled_label_count_per_sample_equal": all(
            len(left["camera_labels"]) == len(right["camera_labels"])
            for left, right in zip(correct, shuffled)
        ),
        "correct_shuffled_target_match_count": sum(
            left["camera_labels"] == right["camera_labels"] for left, right in zip(correct, shuffled)
        ),
        "shuffled_control": {
            "kind": "fixed_within_semantic_group_label_permutation",
            "mapping": SHUFFLED_LABEL_MAP,
            "note": (
                "Exact whole-set derangement while preserving the whole-set multiset is impossible when "
                "one camera label set occurs in more than half of the videos. This control keeps valid labels "
                "and per-sample label count while making the semantic target wrong."
            ),
        },
        "train_motion_buckets": counts(correct, "motion_bucket"),
        "dev_motion_buckets": counts(dev_canonical, "motion_bucket"),
        "train_sources": counts(correct, "source_family"),
        "dev_sources": counts(dev_canonical, "source_family"),
        "outputs": output_info,
    }
    write_json(out_dir / "camera_pretext_transfer_data_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
