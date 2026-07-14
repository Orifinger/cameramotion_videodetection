#!/usr/bin/env python3
"""Build the three-way binary-camera/detection joint-SFT validation gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.camera_binary_vqa.build_data import CAMERA_QUESTIONS


CASE_RE = re.compile(r"(dataA_v1_(?:dataset_v2_|textedit_reserve_)?\d+)/(real|fake)")
ANSWER_RE = re.compile(r"<answer>\s*(Fake|Real)\s*</answer>", re.IGNORECASE)
CAMERA_LABEL_ORDER = tuple(label for label, _ in CAMERA_QUESTIONS)
QUESTION_BY_LABEL = dict(CAMERA_QUESTIONS)
LABEL_LOOKUP = {label.casefold().replace("_", "-"): label for label in CAMERA_LABEL_ORDER}
MOTION_BUCKETS = ("complex-motion", "minor-motion", "no-motion")
CAMERA_SYSTEM_PROMPT = (
    "You are a camera-motion analyst. Answer exactly Yes or No. Judge global camera "
    "behavior across the ordered frames, not object motion or video authenticity."
)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(dict(value))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
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


def dataa_identity(record: Mapping[str, Any]) -> tuple[str, str] | None:
    return identity_from_path(first_image(record))


def source_family(case_id: str) -> str:
    if "textedit_reserve" in case_id:
        return "vace13b_textedit_40step_v3"
    if "dataset_v2" in case_id:
        return "vace13b_dataset_40step_v3"
    return "vace14b_reused"


def canonical_labels(values: Any) -> tuple[str, ...]:
    if isinstance(values, str):
        raw_values: Sequence[Any] = [values]
    elif isinstance(values, Sequence):
        raw_values = values
    else:
        raw_values = []
    selected: set[str] = set()
    unknown: list[str] = []
    for value in raw_values:
        cleaned = str(value).strip()
        folded = cleaned.casefold().replace("_", "-")
        if not cleaned or folded == "static":
            continue
        label = LABEL_LOOKUP.get(folded)
        if label is None:
            unknown.append(cleaned)
        else:
            selected.add(label)
    if unknown:
        raise ValueError(f"unknown camera labels: {sorted(set(unknown))}")
    return tuple(label for label in CAMERA_LABEL_ORDER if label in selected)


def motion_bucket(labels: Sequence[str]) -> str:
    present = set(labels)
    return next((label for label in MOTION_BUCKETS if label in present), "unknown")


def assistant_text(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                return str(message.get("content", ""))
    return ""


def detection_label(record: Mapping[str, Any]) -> str:
    path = first_image(record).casefold()
    if "/fake/" in path:
        return "Fake"
    if "/real/" in path:
        return "Real"
    match = ANSWER_RE.search(assistant_text(record))
    return match.group(1).title() if match else "UNKNOWN"


def load_dataa_pairs(path: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list: {path}")
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for raw in payload:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        identity = dataa_identity(row)
        if identity is None:
            continue
        case_id, kind = identity
        if kind in pairs[case_id]:
            raise ValueError(f"duplicate DataA {kind} record: {case_id}")
        images = row.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError(f"DataA record has no images: {case_id}:{kind}")
        pairs[case_id][kind] = row
    incomplete = sorted(case_id for case_id, pair in pairs.items() if set(pair) != {"real", "fake"})
    if incomplete:
        raise ValueError(f"incomplete DataA real/fake pairs: {incomplete[:20]}")
    return dict(pairs)


def load_camera(path: str | Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        identity = identity_from_path(row.get("path"))
        if identity is None:
            continue
        case_id, kind = identity
        labels = canonical_labels(row.get("labels"))
        caption = str(row.get("caption") or "").strip()
        candidate = {"labels": labels, "caption": caption, "source_kind": kind}
        previous = output.get(case_id)
        if previous is None or kind == "real":
            output[case_id] = candidate
        elif previous["labels"] != labels:
            raise ValueError(f"conflicting camera labels for {case_id}")
    if not output:
        raise ValueError(f"no DataA camera annotations found: {path}")
    return output


def stratified_case_split(
    case_ids: Sequence[str],
    camera: Mapping[str, Mapping[str, Any]],
    test_ratio: float,
    seed: int,
) -> tuple[set[str], set[str], dict[str, Any]]:
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for case_id in case_ids:
        labels = camera.get(case_id, {}).get("labels", ())
        strata[(source_family(case_id), motion_bucket(labels))].append(case_id)
    target_test = round(len(case_ids) * test_ratio)
    allocations: dict[tuple[str, str], int] = {}
    fractional: list[tuple[float, tuple[str, str]]] = []
    for key, values in strata.items():
        exact = len(values) * test_ratio
        count = math.floor(exact)
        if len(values) > 1:
            count = max(1, min(len(values) - 1, count))
        else:
            count = 0
        allocations[key] = count
        fractional.append((exact - math.floor(exact), key))
    difference = target_test - sum(allocations.values())
    order = sorted(fractional, key=lambda item: (-item[0], item[1]))
    while difference:
        progressed = False
        for _, key in (order if difference > 0 else list(reversed(order))):
            size = len(strata[key])
            lower = 1 if size > 1 else 0
            upper = size - 1 if size > 1 else 0
            if difference > 0 and allocations[key] < upper:
                allocations[key] += 1
                difference -= 1
                progressed = True
            elif difference < 0 and allocations[key] > lower:
                allocations[key] -= 1
                difference += 1
                progressed = True
            if difference == 0:
                break
        if not progressed:
            raise ValueError("could not satisfy the requested stratified test size")

    train: set[str] = set()
    test: set[str] = set()
    summary: dict[str, Any] = {}
    for key, values in sorted(strata.items()):
        ordered = sorted(
            values,
            key=lambda case_id: hashlib.sha256(
                f"{seed}:{key[0]}:{key[1]}:{case_id}".encode("utf-8")
            ).hexdigest(),
        )
        count = allocations[key]
        test.update(ordered[:count])
        train.update(ordered[count:])
        summary["|".join(key)] = {"total": len(values), "train": len(values) - count, "test": count}
    if train & test or train | test != set(case_ids):
        raise AssertionError("invalid DataA case split")
    return train, test, summary


def stable_case_order(case_ids: Sequence[str], seed: int, salt: str) -> list[str]:
    return sorted(
        case_ids,
        key=lambda case_id: hashlib.sha256(f"{seed}:{salt}:{case_id}".encode()).hexdigest(),
    )


def camera_prompt(num_frames: int, question: str) -> str:
    frames = "\n".join(f"Frame {index + 1}: <image>" for index in range(num_frames))
    return f"Ordered frames:\n{frames}\n\nQuestion: {question}\nAnswer exactly Yes or No."


def align_camera_prompt_frames(row: dict[str, Any]) -> None:
    """Keep visual placeholders aligned after a control swaps or removes frames."""
    messages = row.get("messages")
    images = row.get("images")
    if not isinstance(messages, list) or not isinstance(images, list):
        raise ValueError("camera record must contain messages and images lists")
    candidates = [
        message
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "user"
        and str(message.get("content", "")).startswith("Ordered frames:\n")
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one ordered-frame user message, found {len(candidates)}")
    message = candidates[0]
    content = str(message.get("content", ""))
    separator = "\n\nQuestion:"
    if separator not in content:
        raise ValueError("camera user message has no Question separator")
    _, suffix = content.split(separator, 1)
    frames = "\n".join(f"Frame {index + 1}: <image>" for index in range(len(images)))
    message["content"] = f"Ordered frames:\n{frames}{separator}{suffix}"


def image_token_count(row: Mapping[str, Any]) -> int:
    messages = row.get("messages", [])
    if not isinstance(messages, list):
        return -1
    return sum(
        str(message.get("content", "")).count("<image>")
        for message in messages
        if isinstance(message, Mapping)
    )


def make_binary_camera_record(
    case_id: str,
    visual_record: Mapping[str, Any],
    primitive: str,
    answer: str,
    pair_id: str,
    dataset_split: str,
    target_kind: str,
    include_assistant: bool,
) -> dict[str, Any]:
    images = [normalized_path(path) for path in visual_record.get("images", [])]
    messages = [
        {"role": "system", "content": CAMERA_SYSTEM_PROMPT},
        {"role": "user", "content": camera_prompt(len(images), QUESTION_BY_LABEL[primitive])},
    ]
    if include_assistant:
        messages.append({"role": "assistant", "content": answer})
    return {
        "messages": messages,
        "images": images,
        "target_text": answer,
        "answer": answer,
        "answer_id": 1 if answer == "Yes" else 0,
        "camera_primitive": primitive,
        "case_id": case_id,
        "visual_source_case_id": case_id,
        "pair_id": pair_id,
        "sample_id": f"camera:{dataset_split}:{primitive}:{pair_id}:{answer}",
        "dataset_split": dataset_split,
        "source_family": source_family(case_id),
        "motion_bucket": None,
        "target_kind": target_kind,
        "visual_condition": "matched_frames",
        "assistant_prefix": "",
        "gate_task": "camera",
    }


def balanced_binary_camera_records(
    case_ids: Sequence[str],
    pairs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    camera: Mapping[str, Mapping[str, Any]],
    dataset_split: str,
    seed: int,
    max_per_answer: int,
    min_per_answer: int,
    include_assistant: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    eligible = [case_id for case_id in case_ids if case_id in camera]
    for primitive, _ in CAMERA_QUESTIONS:
        positives = [case_id for case_id in eligible if primitive in set(camera[case_id]["labels"])]
        negatives = [case_id for case_id in eligible if primitive not in set(camera[case_id]["labels"])]
        positives = stable_case_order(positives, seed, f"{dataset_split}:{primitive}:yes")
        negatives = stable_case_order(negatives, seed, f"{dataset_split}:{primitive}:no")
        selected = min(len(positives), len(negatives))
        if max_per_answer > 0:
            selected = min(selected, max_per_answer)
        supported = selected >= min_per_answer
        stats[primitive] = {
            "available_yes": len(positives),
            "available_no": len(negatives),
            "selected_per_answer": selected if supported else 0,
            "supported": supported,
        }
        if not supported:
            continue
        for index, (positive, negative) in enumerate(zip(positives[:selected], negatives[:selected])):
            pair_id = f"{primitive}:{index:04d}"
            output.append(
                make_binary_camera_record(
                    positive, pairs[positive]["real"], primitive, "Yes", pair_id,
                    dataset_split, "correct", include_assistant,
                )
            )
            output.append(
                make_binary_camera_record(
                    negative, pairs[negative]["real"], primitive, "No", pair_id,
                    dataset_split, "correct", include_assistant,
                )
            )
    random.Random(seed + 97).shuffle(output)
    return output, stats


def flip_binary_targets(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = copy.deepcopy(dict(source))
        answer = "No" if row["answer"] == "Yes" else "Yes"
        row["answer"] = answer
        row["answer_id"] = 1 if answer == "Yes" else 0
        row["target_text"] = answer
        row["messages"][-1]["content"] = answer
        row["target_kind"] = "flipped_binary_control"
        row["sample_id"] = f"{row['sample_id']}:flipped"
        output.append(row)
    return output


def opposite_frame_controls(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(row)
    output: list[dict[str, Any]] = []
    for pair_id, pair_rows in sorted(by_pair.items()):
        if len(pair_rows) != 2 or {row["answer"] for row in pair_rows} != {"Yes", "No"}:
            raise ValueError(f"invalid binary development pair: {pair_id}")
        yes = next(row for row in pair_rows if row["answer"] == "Yes")
        no = next(row for row in pair_rows if row["answer"] == "No")
        for original, donor in ((yes, no), (no, yes)):
            controlled = copy.deepcopy(dict(original))
            controlled["images"] = list(donor["images"])
            align_camera_prompt_frames(controlled)
            controlled["visual_source_case_id"] = donor["case_id"]
            controlled["visual_condition"] = "opposite_answer_frames"
            controlled["sample_id"] = str(original["sample_id"])
            output.append(controlled)
    return output


def no_frame_controls(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = copy.deepcopy(dict(source))
        row["images"] = []
        align_camera_prompt_frames(row)
        row["visual_source_case_id"] = None
        row["visual_condition"] = "no_frames"
        output.append(row)
    return output


def mark_detection(record: Mapping[str, Any], source: str) -> dict[str, Any]:
    output = copy.deepcopy(dict(record))
    output["gate_task"] = "detection"
    output["gate_source"] = source
    output["detection_label"] = detection_label(output)
    return output


def load_detection_records(path: str | Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list: {path}")
    records = [dict(row) for row in payload if isinstance(row, Mapping)]
    if len(records) != len(payload):
        raise ValueError(f"non-object detection records found: {path}")
    return records


def data_parent(record: Mapping[str, Any]) -> str:
    path = first_image(record)
    return path.rsplit("/", 1)[0].rstrip("/") if "/" in path else path.rstrip("/")


def load_camera_by_path(path: str | Path) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for row in read_jsonl(path):
        key = normalized_path(row.get("path")).rstrip("/")
        if key:
            output[key] = canonical_labels(row.get("labels"))
    return output


def coarse_signature(labels: Sequence[str]) -> tuple[str, str, str]:
    present = set(labels)

    def choose(order: Sequence[str], default: str) -> str:
        return next((label for label in order if label in present), default)

    return (
        choose(MOTION_BUCKETS, "motion-unknown"),
        choose(("fast-speed", "regular-speed", "slow-speed"), "speed-unknown"),
        choose(
            ("very-unsteady", "unsteady", "minimal-shaking", "no-shaking"),
            "steadiness-unknown",
        ),
    )


def sample_datab_replay(
    records: Sequence[dict[str, Any]],
    camera_by_path: Mapping[str, Sequence[str]],
    target_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strata: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"Real": [], "Fake": []}
    )
    skipped = 0
    for record in records:
        label = detection_label(record)
        labels = camera_by_path.get(data_parent(record), ())
        if label not in {"Real", "Fake"} or not labels:
            skipped += 1
            continue
        strata[coarse_signature(labels)][label].append(record)
    rng = random.Random(seed)
    pools: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for signature, classes in strata.items():
        rng.shuffle(classes["Real"])
        rng.shuffle(classes["Fake"])
        count = min(len(classes["Real"]), len(classes["Fake"]))
        if count:
            pools[signature] = list(zip(classes["Real"][:count], classes["Fake"][:count]))
    selected_pairs: list[tuple[tuple[str, str, str], dict[str, Any], dict[str, Any]]] = []
    positions: Counter[tuple[str, str, str]] = Counter()
    signatures = sorted(pools)
    rng.shuffle(signatures)
    requested_pairs = target_count // 2
    while len(selected_pairs) < requested_pairs:
        progressed = False
        for signature in signatures:
            position = positions[signature]
            if position < len(pools[signature]) and len(selected_pairs) < requested_pairs:
                real, fake = pools[signature][position]
                selected_pairs.append((signature, real, fake))
                positions[signature] += 1
                progressed = True
        if not progressed:
            break
    selected: list[dict[str, Any]] = []
    selected_strata: Counter[str] = Counter()
    for signature, real, fake in selected_pairs:
        selected.extend([mark_detection(real, "datab_replay"), mark_detection(fake, "datab_replay")])
        selected_strata["|".join(signature)] += 2
    rng.shuffle(selected)
    audit = {
        "requested_records": target_count,
        "selected_records": len(selected),
        "target_met": len(selected) == target_count - target_count % 2,
        "balanced_real_fake": dict(Counter(row["detection_label"] for row in selected)),
        "unmatched_or_unlabeled": skipped,
        "selected_strata": dict(selected_strata),
    }
    return selected, audit


def repeated_detection_control(
    records: Sequence[dict[str, Any]], count: int, seed: int, source: str
) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("cannot build detection-only control from an empty pool")
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    output: list[dict[str, Any]] = []
    for index in range(count):
        row = copy.deepcopy(records[order[index % len(order)]])
        row["gate_source"] = source
        row["control_repeat_index"] = index
        output.append(row)
    return output


def stable_shuffle(rows: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    output = list(rows)
    random.Random(seed).shuffle(output)
    return output


def camera_text_in_detection_prompt(record: Mapping[str, Any]) -> bool:
    if record.get("gate_task") != "detection":
        return False
    messages = record.get("messages") or []
    text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, Mapping) and message.get("role") != "assistant"
    ).casefold()
    return "camera_motion" in text or "camera caption" in text or "camera context" in text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataa-detection-json", required=True)
    parser.add_argument("--dataa-camera-jsonl", required=True)
    parser.add_argument("--datab-detection-json", required=True)
    parser.add_argument("--datab-camera-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--test-ratio", type=float, default=0.30)
    parser.add_argument("--expected-dataa-cases", type=int, default=1080)
    parser.add_argument("--datab-replay-records", type=int, default=0)
    parser.add_argument("--max-train-per-answer", type=int, default=0)
    parser.add_argument("--max-dev-per-answer", type=int, default=16)
    parser.add_argument("--min-per-answer", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--check-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_ratio < 1.0:
        raise ValueError("test-ratio must be between zero and one")
    pairs = load_dataa_pairs(args.dataa_detection_json)
    if args.expected_dataa_cases > 0 and len(pairs) != args.expected_dataa_cases:
        raise ValueError(
            f"DataA case count mismatch: expected={args.expected_dataa_cases}, actual={len(pairs)}"
        )
    camera = load_camera(args.dataa_camera_jsonl)
    train_ids, test_ids, split_strata = stratified_case_split(
        sorted(pairs), camera, args.test_ratio, args.seed
    )
    missing_camera_train = sorted(train_ids - camera.keys())
    missing_camera_test = sorted(test_ids - camera.keys())

    dataa_train_detection = [
        mark_detection(pairs[case_id][kind], "dataa_train")
        for case_id in sorted(train_ids)
        for kind in ("real", "fake")
    ]
    dataa_test_detection = [
        mark_detection(pairs[case_id][kind], "dataa_test")
        for case_id in sorted(test_ids)
        for kind in ("real", "fake")
    ]
    correct_camera, train_camera_stats = balanced_binary_camera_records(
        sorted(train_ids), pairs, camera, "train", args.seed + 11,
        args.max_train_per_answer, args.min_per_answer, True,
    )
    shuffled_camera = flip_binary_targets(correct_camera)
    camera_dev, dev_camera_stats = balanced_binary_camera_records(
        sorted(test_ids), pairs, camera, "test", args.seed + 23,
        args.max_dev_per_answer, args.min_per_answer, False,
    )
    camera_dev_opposite = opposite_frame_controls(camera_dev)
    camera_dev_no_frames = no_frame_controls(camera_dev)
    camera_dev_conditions = camera_dev + camera_dev_opposite + camera_dev_no_frames
    token_path_mismatches = [
        str(row.get("sample_id"))
        for row in camera_dev_conditions
        if image_token_count(row) != len(row.get("images", []))
    ]
    if token_path_mismatches:
        raise AssertionError(
            "camera dev image token/path mismatches; first=" + token_path_mismatches[0]
        )

    if args.check_images:
        all_images = {
            normalized_path(path)
            for row in dataa_train_detection + dataa_test_detection
            for path in row.get("images", [])
        }
        missing_images = [path for path in sorted(all_images) if not Path(path).is_file()]
        if missing_images:
            raise FileNotFoundError(
                f"missing {len(missing_images)}/{len(all_images)} DataA images; first={missing_images[0]}"
            )

    datab = load_detection_records(args.datab_detection_json)
    datab_camera = load_camera_by_path(args.datab_camera_jsonl)
    replay_target = args.datab_replay_records or len(dataa_train_detection)
    replay_target -= replay_target % 2
    datab_replay, replay_audit = sample_datab_replay(
        datab, datab_camera, replay_target, args.seed + 31
    )
    if not replay_audit["target_met"]:
        raise ValueError(f"DataB replay target was not met: {replay_audit}")

    base_detection_pool = dataa_train_detection + datab_replay
    detection_padding = repeated_detection_control(
        base_detection_pool,
        max(0, len(correct_camera) - len(base_detection_pool)),
        args.seed + 37,
        "equal_ratio_detection_repeat",
    )
    detection_pool = stable_shuffle(base_detection_pool + detection_padding, args.seed + 41)
    detection_extra = repeated_detection_control(
        detection_pool,
        len(correct_camera),
        args.seed + 43,
        "detection_only_auxiliary_control",
    )
    branches = {
        "joint_sft_detection_only.json": stable_shuffle(detection_pool + detection_extra, args.seed + 51),
        "joint_sft_correct_camera.json": stable_shuffle(detection_pool + correct_camera, args.seed + 51),
        "joint_sft_shuffled_camera.json": stable_shuffle(detection_pool + shuffled_camera, args.seed + 51),
    }
    if len({len(rows) for rows in branches.values()}) != 1:
        raise AssertionError("joint-SFT branch sizes differ")
    if any(camera_text_in_detection_prompt(row) for rows in branches.values() for row in rows):
        raise AssertionError("camera text leaked into a detection prompt")
    if Counter(row["answer"] for row in correct_camera) != Counter(row["answer"] for row in shuffled_camera):
        raise AssertionError("flipped control changed the Yes/No marginal distribution")
    if any(
        left["images"] != right["images"]
        or left["messages"][:-1] != right["messages"][:-1]
        or left["answer"] == right["answer"]
        for left, right in zip(correct_camera, shuffled_camera)
    ):
        raise AssertionError("correct/flipped camera control is not input-matched")

    out_dir = Path(args.out_dir)
    json_payloads: dict[str, list[dict[str, Any]]] = {
        "dataa_train_detection.json": dataa_train_detection,
        "dataa_test_detection.json": dataa_test_detection,
        "datab_detection_replay.json": datab_replay,
        "camera_train_correct.json": correct_camera,
        "camera_train_shuffled.json": shuffled_camera,
        **branches,
    }
    outputs: dict[str, Any] = {}
    for name, rows in json_payloads.items():
        path = out_dir / name
        write_json(path, rows)
        outputs[name] = {"path": str(path), "records": len(rows), "sha256": sha256(path)}
    for name, rows in {
        "camera_dev_matched_frames.jsonl": camera_dev,
        "camera_dev_opposite_frames.jsonl": camera_dev_opposite,
        "camera_dev_no_frames.jsonl": camera_dev_no_frames,
    }.items():
        path = out_dir / name
        write_jsonl(path, rows)
        outputs[name] = {"path": str(path), "records": len(rows), "sha256": sha256(path)}

    split_rows = [
        {
            "case_id": case_id,
            "dataset_split": "train" if case_id in train_ids else "test",
            "source_family": source_family(case_id),
            "motion_bucket": motion_bucket(camera.get(case_id, {}).get("labels", ())),
            "camera_labels": list(camera.get(case_id, {}).get("labels", ())),
            "camera_caption": camera.get(case_id, {}).get("caption", ""),
            "camera_annotation_available": case_id in camera,
            "real_frame_dir": first_image(pairs[case_id]["real"]).rsplit("/", 1)[0],
            "fake_frame_dir": first_image(pairs[case_id]["fake"]).rsplit("/", 1)[0],
        }
        for case_id in sorted(pairs)
    ]
    split_path = out_dir / "dataa_40step_v3_split_manifest.jsonl"
    write_jsonl(split_path, split_rows)
    outputs[split_path.name] = {
        "path": str(split_path), "records": len(split_rows), "sha256": sha256(split_path),
    }

    summary = {
        "schema_version": "camera_joint_binary_sft_gate_v2",
        "seed": args.seed,
        "question": (
            "Does balanced binary camera supervision on the same 16 frames add visually grounded "
            "camera ability while detection replay preserves local/global AIGC detection?"
        ),
        "inputs": {
            "dataa_detection_json": args.dataa_detection_json,
            "dataa_camera_jsonl": args.dataa_camera_jsonl,
            "datab_detection_json": args.datab_detection_json,
            "datab_camera_jsonl": args.datab_camera_jsonl,
        },
        "split": {
            "kind": "case-level stratified by source family and coarse motion bucket",
            "test_ratio": args.test_ratio,
            "total_cases": len(pairs),
            "train_cases": len(train_ids),
            "test_cases": len(test_ids),
            "train_test_overlap": sorted(train_ids & test_ids),
            "strata": split_strata,
        },
        "camera_supervision": {
            "kind": "balanced binary VQA with one canonical question per camera primitive",
            "caption_used_as_training_target": False,
            "train_records": len(correct_camera),
            "dev_records_per_condition": len(camera_dev),
            "train_supported_labels": sum(value["supported"] for value in train_camera_stats.values()),
            "dev_supported_labels": sum(value["supported"] for value in dev_camera_stats.values()),
            "train_answer_counts": dict(Counter(row["answer"] for row in correct_camera)),
            "dev_answer_counts": dict(Counter(row["answer"] for row in camera_dev)),
            "train_per_label": train_camera_stats,
            "dev_per_label": dev_camera_stats,
            "missing_camera_train": missing_camera_train,
            "missing_camera_test": missing_camera_test,
        },
        "shuffled_target_control": {
            "kind": "flip every balanced binary Yes/No target while keeping frames and question fixed",
            "correct_and_shuffled_same_inputs": True,
            "every_target_is_wrong": True,
            "answer_marginal_preserved": (
                Counter(row["answer"] for row in correct_camera)
                == Counter(row["answer"] for row in shuffled_camera)
            ),
        },
        "visual_controls": {
            "opposite_frames_records": len(camera_dev_opposite),
            "no_frames_records": len(camera_dev_no_frames),
            "opposite_frames_keep_question_and_gold_answer": True,
            "matched_frame_count_distribution": dict(
                sorted(Counter(len(row["images"]) for row in camera_dev).items())
            ),
        },
        "detection_replay": replay_audit,
        "joint_task_ratio": {
            "base_detection_records": len(base_detection_pool),
            "detection_padding_records": len(detection_padding),
            "final_detection_records": len(detection_pool),
            "camera_records": len(correct_camera),
            "camera_to_detection_ratio": (
                len(correct_camera) / len(detection_pool) if detection_pool else None
            ),
        },
        "branch_counts": {
            name: {
                "records": len(rows),
                "task_counts": dict(Counter(row["gate_task"] for row in rows)),
                "detection_sources": dict(
                    Counter(
                        str(row.get("gate_source"))
                        for row in rows if row["gate_task"] == "detection"
                    )
                ),
            }
            for name, rows in branches.items()
        },
        "integrity": {
            "all_dataa_pairs_complete": True,
            "real_fake_pair_never_crosses_split": True,
            "branch_record_counts_equal": True,
            "camera_text_absent_from_detection_prompts": True,
            "camera_dev_has_no_assistant_gold_in_prompt": all(
                row["messages"][-1]["role"] == "user" for row in camera_dev
            ),
            "camera_dev_image_tokens_match_paths": not token_path_mismatches,
            "datab_is_replay_not_held_out": True,
        },
        "outputs": outputs,
    }
    write_json(out_dir / "camera_joint_sft_data_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
