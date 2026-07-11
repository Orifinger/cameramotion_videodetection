#!/usr/bin/env python3
"""Build paired DataA training data and motion-matched DataB replay.

DataA cases are kept as complete Real/Fake pairs.  DataB is matched within a
coarse camera signature (motion dynamics, speed, steadiness), with equal Real
and Fake counts in every retained stratum.  Camera text is never injected into
the detection prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

DATAA_RE = re.compile(r"(dataA_v1_(?:dataset_v2_|textedit_reserve_)?\d+)/(real|fake)")
ANSWER_RE = re.compile(r"<answer>\s*(Fake|Real)\s*</answer>", re.IGNORECASE)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
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
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_path(value: Any) -> str:
    return str(value or "").replace("\\", "/")


def first_image(record: Mapping[str, Any]) -> str:
    images = record.get("images")
    return normalize_path(images[0]) if isinstance(images, list) and images else ""


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


def dataa_case(record: Mapping[str, Any]) -> tuple[str, str] | None:
    match = DATAA_RE.search(first_image(record))
    return (match.group(1), match.group(2)) if match else None


def load_records(path: str | Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def split_case_ids(path: str | None) -> set[str] | None:
    if not path:
        return None
    output = set()
    for record in load_records(path):
        identity = dataa_case(record)
        if identity:
            output.add(identity[0])
    return output


def resolve_dataa_train(records: Sequence[Mapping[str, Any]], train_json: str | None, test_json: str | None, seed: int) -> tuple[set[str], set[str], str]:
    all_cases = {identity[0] for record in records if (identity := dataa_case(record))}
    train, test = split_case_ids(train_json), split_case_ids(test_json)
    if train is None and test is None:
        ordered = sorted(all_cases)
        random.Random(seed).shuffle(ordered)
        cut = max(1, round(len(ordered) * 0.7))
        train, test, source = set(ordered[:cut]), set(ordered[cut:]), "derived_case_split_70_30"
    elif train is None:
        train, source = all_cases - set(test or ()), "full_minus_explicit_test"
    elif test is None:
        test, source = all_cases - set(train), "full_minus_explicit_train"
    else:
        source = "explicit_train_and_test"
    train, test = set(train or ()) & all_cases, set(test or ()) & all_cases
    if train & test:
        raise ValueError(f"DataA train/test leakage: {sorted(train & test)[:20]}")
    return train, test, source


def load_camera(path: str | Path) -> dict[str, list[str]]:
    output = {}
    for row in read_jsonl(path):
        key = normalize_path(row.get("path")).rstrip("/")
        if key:
            output[key] = [str(label) for label in row.get("labels", [])]
    return output


def camera_labels(record: Mapping[str, Any], camera: Mapping[str, list[str]]) -> list[str]:
    path = first_image(record)
    parent = path.rsplit("/", 1)[0] if "/" in path else path
    return list(camera.get(parent.rstrip("/"), []))


def choose_one(labels: set[str], order: Sequence[str], default: str) -> str:
    for candidate in order:
        if candidate in labels:
            return candidate
    return default


def coarse_signature(labels: Sequence[str]) -> tuple[str, str, str]:
    present = {str(label).strip().casefold().replace("_", "-") for label in labels}
    motion = choose_one(present, ("complex-motion", "minor-motion", "no-motion"), "motion-unknown")
    speed = choose_one(present, ("fast-speed", "regular-speed", "slow-speed"), "speed-unknown")
    steadiness = choose_one(
        present,
        ("very-unsteady", "unsteady", "minimal-shaking", "no-shaking"),
        "steadiness-unknown",
    )
    return motion, speed, steadiness


def sample_motion_matched(
    records: Sequence[dict[str, Any]],
    camera: Mapping[str, list[str]],
    target_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strata: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"Real": [], "Fake": []})
    unmatched = 0
    for record in records:
        label = detection_label(record)
        labels = camera_labels(record, camera)
        if label not in {"Real", "Fake"} or not labels:
            unmatched += 1
            continue
        strata[coarse_signature(labels)][label].append(record)

    rng = random.Random(seed)
    pair_pools: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for signature, classes in strata.items():
        rng.shuffle(classes["Real"])
        rng.shuffle(classes["Fake"])
        count = min(len(classes["Real"]), len(classes["Fake"]))
        if count:
            pair_pools[signature] = list(zip(classes["Real"][:count], classes["Fake"][:count]))

    signatures = sorted(pair_pools)
    rng.shuffle(signatures)
    selected_pairs = []
    target_pairs = max(1, target_count // 2)
    positions = Counter()
    while len(selected_pairs) < target_pairs:
        progressed = False
        for signature in signatures:
            position = positions[signature]
            pool = pair_pools[signature]
            if position < len(pool) and len(selected_pairs) < target_pairs:
                selected_pairs.append((signature, *pool[position]))
                positions[signature] += 1
                progressed = True
        if not progressed:
            break

    selected = []
    selected_strata = Counter()
    for signature, real, fake in selected_pairs:
        selected.extend([real, fake])
        selected_strata["|".join(signature)] += 2
    rng.shuffle(selected)
    audit = {
        "camera_matched_records_available": 2 * sum(len(pool) for pool in pair_pools.values()),
        "selected_records": len(selected),
        "requested_records": target_count,
        "target_met": len(selected) >= min(target_count - target_count % 2, target_count),
        "unmatched_or_unlabeled_records": unmatched,
        "num_retained_strata": len(pair_pools),
        "selected_strata": dict(selected_strata),
    }
    return selected, audit


def interleave(a: Sequence[dict[str, Any]], b: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    left, right = list(a), list(b)
    rng = random.Random(seed)
    rng.shuffle(left)
    rng.shuffle(right)
    output = []
    for index in range(max(len(left), len(right))):
        if index < len(left):
            output.append(left[index])
        if index < len(right):
            output.append(right[index])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataa-detection-json", required=True)
    parser.add_argument("--datab-detection-json", required=True)
    parser.add_argument("--datab-camera-jsonl", required=True)
    parser.add_argument("--dataa-train-json")
    parser.add_argument("--dataa-test-json")
    parser.add_argument("--datab-target-samples", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--require-target", action="store_true")
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataa = load_records(args.dataa_detection_json)
    datab = load_records(args.datab_detection_json)
    train_cases, test_cases, split_source = resolve_dataa_train(
        dataa, args.dataa_train_json, args.dataa_test_json, args.seed
    )
    dataa_train = [
        record for record in dataa if (identity := dataa_case(record)) and identity[0] in train_cases
    ]
    counts_by_case = Counter(identity[0] for record in dataa_train if (identity := dataa_case(record)))
    broken_pairs = sorted(case_id for case_id, count in counts_by_case.items() if count != 2)
    if broken_pairs:
        raise ValueError(f"DataA train contains incomplete pairs: {broken_pairs[:20]}")

    target = args.datab_target_samples or len(dataa_train)
    datab_camera = load_camera(args.datab_camera_jsonl)
    datab_replay, replay_audit = sample_motion_matched(datab, datab_camera, target, args.seed + 1)
    if args.require_target and not replay_audit["target_met"]:
        raise ValueError(f"motion-matched DataB target not met: {replay_audit}")
    mixed = interleave(dataa_train, datab_replay, args.seed + 2)

    out_dir = Path(args.out_dir)
    outputs = {
        "dataa_paired_train": out_dir / "dataa_detection_train_paired.json",
        "datab_motion_matched_replay": out_dir / "datab_detection_replay_motion_matched.json",
        "mixed_local_global_replay": out_dir / "mixed_local_global_detection_replay.json",
    }
    write_json(outputs["dataa_paired_train"], dataa_train)
    write_json(outputs["datab_motion_matched_replay"], datab_replay)
    write_json(outputs["mixed_local_global_replay"], mixed)
    summary = {
        "schema_version": "local_global_detection_replay_v1",
        "seed": args.seed,
        "split_source": split_source,
        "counts": {
            "dataa_train_cases": len(train_cases),
            "dataa_test_cases": len(test_cases),
            "dataa_train_records": len(dataa_train),
            "datab_replay_records": len(datab_replay),
            "mixed_records": len(mixed),
        },
        "leakage_audit": {
            "dataa_train_test_case_overlap": sorted(train_cases & test_cases),
            "dataa_all_train_cases_complete_pairs": not broken_pairs,
        },
        "dataa_labels": dict(Counter(detection_label(record) for record in dataa_train)),
        "datab_replay_labels": dict(Counter(detection_label(record) for record in datab_replay)),
        "datab_replay_audit": replay_audit,
        "camera_usage": "stratified sampling only; no camera text injected into prompts",
        "outputs": {
            name: {"path": str(path), "records": len(read_json(path)), "sha256": sha256(path)}
            for name, path in outputs.items()
        },
    }
    summary_path = out_dir / "local_global_detection_replay_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
