#!/usr/bin/env python3
"""Build camera-bucketed detection data for the hard-routing validation gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_camera_joint_sft_gate import (  # noqa: E402
    CAMERA_SYSTEM_PROMPT,
    QUESTION_BY_LABEL,
    camera_text_in_detection_prompt,
    data_parent,
    detection_label,
    first_image,
    identity_from_path,
    load_dataa_pairs,
    load_detection_records,
    mark_detection,
    normalized_path,
    source_family,
    stratified_case_split,
)


ROUTE_BUCKETS = ("no-motion", "minor-motion", "complex-motion")
ROUTE_PRIORITY = ("complex-motion", "minor-motion", "no-motion")
STATIC_ALIASES = {"static", "no-motion", "no-camera-motion"}
BUCKET_DATASET_NAMES = {
    "no-motion": "camera_hard_route_no_motion",
    "minor-motion": "camera_hard_route_minor_motion",
    "complex-motion": "camera_hard_route_complex_motion",
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_label(value: Any) -> str:
    return str(value or "").strip().casefold().replace("_", "-")


def route_bucket(values: Any) -> str:
    if isinstance(values, str):
        labels = {normalize_label(values)}
    elif isinstance(values, Sequence):
        labels = {normalize_label(value) for value in values}
    else:
        labels = set()
    if "complex-motion" in labels:
        return "complex-motion"
    if "minor-motion" in labels:
        return "minor-motion"
    if labels & STATIC_ALIASES:
        return "no-motion"
    return "unknown"


def load_dataa_camera(path: str | Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        identity = identity_from_path(row.get("path"))
        if identity is None:
            continue
        case_id, kind = identity
        candidate = {
            "route_bucket": route_bucket(row.get("labels")),
            "labels": [str(value) for value in row.get("labels") or []],
            "caption": str(row.get("caption") or "").strip(),
            "source_kind": kind,
        }
        previous = output.get(case_id)
        if previous is None or kind == "real":
            output[case_id] = candidate
        elif previous["route_bucket"] != candidate["route_bucket"]:
            raise ValueError(f"conflicting DataA route buckets for {case_id}")
    if not output:
        raise ValueError(f"no DataA camera annotations found: {path}")
    return output


def load_camera_by_path(path: str | Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = normalized_path(row.get("path")).rstrip("/")
        if not key:
            continue
        bucket = route_bucket(row.get("labels"))
        output[key] = {
            "route_bucket": bucket,
            "labels": [str(value) for value in row.get("labels") or []],
            "caption": str(row.get("caption") or "").strip(),
        }
    return output


def camera_for_split(camera: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        case_id: {
            **dict(value),
            "labels": ()
            if value.get("route_bucket") == "unknown"
            else (str(value["route_bucket"]),),
        }
        for case_id, value in camera.items()
    }


def stable_key(value: str, seed: int, salt: str) -> str:
    return hashlib.sha256(f"{seed}:{salt}:{value}".encode("utf-8")).hexdigest()


def record_identity(record: Mapping[str, Any]) -> str:
    return f"{first_image(record)}|{detection_label(record)}"


def attach_route_metadata(
    record: Mapping[str, Any],
    bucket: str,
    domain: str,
    source: str,
    unique_key: str | None = None,
) -> dict[str, Any]:
    output = mark_detection(record, source)
    output["route_bucket"] = bucket
    output["route_domain"] = domain
    content_identity = record_identity(output)
    output["route_content_id"] = hashlib.sha256(
        content_identity.encode("utf-8")
    ).hexdigest()[:24]
    output["route_record_id"] = hashlib.sha256(
        f"{domain}|{unique_key or content_identity}".encode("utf-8")
    ).hexdigest()[:24]
    return output


def balanced_datab_records(
    records: Sequence[dict[str, Any]],
    camera_by_path: Mapping[str, Mapping[str, Any]],
    seed: int,
    max_per_bucket_per_class: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    pools: dict[str, dict[str, list[dict[str, Any]]]] = {
        bucket: {"Real": [], "Fake": []} for bucket in ROUTE_BUCKETS
    }
    reasons: Counter[str] = Counter()
    for source_index, record in enumerate(records):
        label = detection_label(record)
        if label not in {"Real", "Fake"}:
            reasons["unknown_detection_label"] += 1
            continue
        camera = camera_by_path.get(data_parent(record))
        if camera is None:
            reasons["camera_path_unmatched"] += 1
            continue
        bucket = str(camera.get("route_bucket", "unknown"))
        if bucket not in ROUTE_BUCKETS:
            reasons["unknown_route_bucket"] += 1
            continue
        indexed_record = copy.deepcopy(record)
        indexed_record["_route_source_index"] = source_index
        pools[bucket][label].append(indexed_record)

    selected: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in ROUTE_BUCKETS}
    bucket_audit: dict[str, Any] = {}
    for offset, bucket in enumerate(ROUTE_BUCKETS):
        classes = pools[bucket]
        for label in ("Real", "Fake"):
            classes[label] = sorted(
                classes[label],
                key=lambda row: stable_key(record_identity(row), seed + offset, f"{bucket}:{label}"),
            )
        count = min(len(classes["Real"]), len(classes["Fake"]))
        if max_per_bucket_per_class > 0:
            count = min(count, max_per_bucket_per_class)
        rows = [
            attach_route_metadata(
                {key: value for key, value in record.items() if key != "_route_source_index"},
                bucket,
                "datab",
                "datab_camera_matched",
                unique_key=f"datab:{record['_route_source_index']}",
            )
            for label in ("Real", "Fake")
            for record in classes[label][:count]
        ]
        random.Random(seed + 101 + offset).shuffle(rows)
        selected[bucket] = rows
        bucket_audit[bucket] = {
            "available_real": len(classes["Real"]),
            "available_fake": len(classes["Fake"]),
            "selected_per_class": count,
            "selected_records": len(rows),
        }
    return selected, {
        "skipped": dict(reasons),
        "buckets": bucket_audit,
        "selected_records": sum(len(rows) for rows in selected.values()),
    }


def camera_prompt(num_frames: int, question: str) -> str:
    frames = "\n".join(f"Frame {index + 1}: <image>" for index in range(num_frames))
    return f"Ordered frames:\n{frames}\n\nQuestion: {question}\nAnswer exactly Yes or No."


def route_question_records(
    case_id: str,
    kind: str,
    visual_record: Mapping[str, Any],
    gold_bucket: str,
) -> list[dict[str, Any]]:
    images = [normalized_path(path) for path in visual_record.get("images", [])]
    video_id = f"{case_id}:{kind}"
    output: list[dict[str, Any]] = []
    for bucket in ROUTE_BUCKETS:
        output.append(
            {
                "messages": [
                    {"role": "system", "content": CAMERA_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": camera_prompt(len(images), QUESTION_BY_LABEL[bucket]),
                    },
                ],
                "images": images,
                "assistant_prefix": "",
                "sample_id": f"dataa-route:{video_id}:{bucket}",
                "video_id": video_id,
                "case_id": case_id,
                "visual_source_case_id": case_id,
                "camera_primitive": bucket,
                "route_gold_bucket": gold_bucket,
                "source_family": source_family(case_id),
                "visual_kind": kind,
                "frame_dir": first_image(visual_record).rsplit("/", 1)[0],
            }
        )
    return output


def router_training_record(
    case_id: str,
    visual_record: Mapping[str, Any],
    question_bucket: str,
    gold_bucket: str,
    answer: str,
    pair_id: str,
) -> dict[str, Any]:
    images = [normalized_path(path) for path in visual_record.get("images", [])]
    return {
        "messages": [
            {"role": "system", "content": CAMERA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": camera_prompt(len(images), QUESTION_BY_LABEL[question_bucket]),
            },
            {"role": "assistant", "content": answer},
        ],
        "images": images,
        "target_text": answer,
        "answer": answer,
        "answer_id": 1 if answer == "Yes" else 0,
        "camera_primitive": question_bucket,
        "case_id": case_id,
        "visual_source_case_id": case_id,
        "pair_id": pair_id,
        "sample_id": f"camera-route:train:{question_bucket}:{pair_id}:{answer}",
        "dataset_split": "train",
        "source_family": source_family(case_id),
        "route_bucket": gold_bucket,
        "route_gold_bucket": gold_bucket,
        "visual_condition": "matched_real_frames",
        "assistant_prefix": "",
        "gate_task": "camera_route",
    }


def balanced_router_training_records(
    train_ids: Sequence[str],
    pairs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    camera: Mapping[str, Mapping[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    known = [
        case_id
        for case_id in train_ids
        if camera.get(case_id, {}).get("route_bucket") in ROUTE_BUCKETS
    ]
    candidates: dict[str, tuple[list[str], list[str]]] = {}
    for offset, question_bucket in enumerate(ROUTE_BUCKETS):
        positives = [
            case_id
            for case_id in known
            if camera[case_id]["route_bucket"] == question_bucket
        ]
        negatives = [
            case_id
            for case_id in known
            if camera[case_id]["route_bucket"] != question_bucket
        ]
        positives = sorted(
            positives,
            key=lambda case_id: stable_key(case_id, seed + offset, f"router:{question_bucket}:yes"),
        )
        negatives = sorted(
            negatives,
            key=lambda case_id: stable_key(case_id, seed + offset, f"router:{question_bucket}:no"),
        )
        candidates[question_bucket] = (positives, negatives)
    selected_per_answer = min(
        min(len(positives), len(negatives))
        for positives, negatives in candidates.values()
    )
    if selected_per_answer == 0:
        raise ValueError("cannot build an equal-frequency three-question camera router")

    records: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for question_bucket in ROUTE_BUCKETS:
        positives, negatives = candidates[question_bucket]
        if not positives or not negatives:
            raise ValueError(f"cannot balance camera router question: {question_bucket}")
        for index, (positive, negative) in enumerate(
            zip(positives[:selected_per_answer], negatives[:selected_per_answer])
        ):
            pair_id = f"{question_bucket}:{index:04d}"
            records.append(
                router_training_record(
                    positive,
                    pairs[positive]["real"],
                    question_bucket,
                    str(camera[positive]["route_bucket"]),
                    "Yes",
                    pair_id,
                )
            )
            records.append(
                router_training_record(
                    negative,
                    pairs[negative]["real"],
                    question_bucket,
                    str(camera[negative]["route_bucket"]),
                    "No",
                    pair_id,
                )
            )
        audit[question_bucket] = {
            "available_yes": len(positives),
            "available_no": len(negatives),
            "selected_per_answer": selected_per_answer,
        }
    audit["equal_frequency_contract"] = {
        "questions": len(ROUTE_BUCKETS),
        "selected_per_answer_per_question": selected_per_answer,
        "records_per_question": selected_per_answer * 2,
    }
    random.Random(seed + 73).shuffle(records)
    return records, audit


def validate_partition(experts: Mapping[str, Sequence[Mapping[str, Any]]], shared: Sequence[Mapping[str, Any]]) -> None:
    expert_ids: list[str] = []
    for bucket in ROUTE_BUCKETS:
        rows = experts[bucket]
        labels = Counter(str(row.get("detection_label")) for row in rows)
        if not rows or set(labels) != {"Real", "Fake"}:
            raise ValueError(f"route expert {bucket} is empty or not binary-balanced: {dict(labels)}")
        if any(row.get("route_bucket") != bucket for row in rows):
            raise AssertionError(f"route expert {bucket} contains another bucket")
        expert_ids.extend(str(row["route_record_id"]) for row in rows)
    shared_ids = [str(row["route_record_id"]) for row in shared]
    if len(expert_ids) != len(set(expert_ids)):
        raise AssertionError("a detection record appears in more than one route expert")
    if Counter(expert_ids) != Counter(shared_ids):
        raise AssertionError("shared data is not the exact union of the three expert datasets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataa-detection-json", required=True)
    parser.add_argument("--dataa-camera-jsonl", required=True)
    parser.add_argument("--datab-detection-json", required=True)
    parser.add_argument("--datab-camera-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--test-ratio", type=float, default=0.30)
    parser.add_argument("--expected-dataa-cases", type=int, default=1080)
    parser.add_argument("--max-datab-per-bucket-per-class", type=int, default=0)
    parser.add_argument("--minimum-records-per-expert", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260715)
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
    dataa_camera = load_dataa_camera(args.dataa_camera_jsonl)
    split_camera = camera_for_split(dataa_camera)
    train_ids, test_ids, split_strata = stratified_case_split(
        sorted(pairs), split_camera, args.test_ratio, args.seed
    )

    experts: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in ROUTE_BUCKETS}
    excluded_dataa: Counter[str] = Counter()
    dataa_test_detection: list[dict[str, Any]] = []
    dataa_route_dev: list[dict[str, Any]] = []
    dataa_route_gold: list[dict[str, Any]] = []
    for case_id in sorted(pairs):
        camera = dataa_camera.get(case_id, {})
        bucket = str(camera.get("route_bucket", "unknown"))
        if case_id in train_ids:
            if bucket not in ROUTE_BUCKETS:
                excluded_dataa["train_unknown_or_missing_route"] += 1
                continue
            for kind in ("real", "fake"):
                experts[bucket].append(
                    attach_route_metadata(
                        pairs[case_id][kind],
                        bucket,
                        "dataa",
                        "dataa_train_pair",
                        unique_key=f"dataa:{case_id}:{kind}",
                    )
                )
        else:
            for kind in ("real", "fake"):
                row = attach_route_metadata(
                    pairs[case_id][kind],
                    bucket if bucket in ROUTE_BUCKETS else "shared",
                    "dataa",
                    "dataa_test",
                    unique_key=f"dataa-test:{case_id}:{kind}",
                )
                dataa_test_detection.append(row)
                dataa_route_gold.append(
                    {
                        "video_id": f"{case_id}:{kind}",
                        "case_id": case_id,
                        "visual_kind": kind,
                        "route_gold_bucket": bucket,
                        "source_family": source_family(case_id),
                    }
                )
                if bucket in ROUTE_BUCKETS:
                    dataa_route_dev.extend(
                        route_question_records(case_id, kind, pairs[case_id][kind], bucket)
                    )
                else:
                    excluded_dataa["test_unknown_or_missing_route"] += 1

    router_train, router_train_audit = balanced_router_training_records(
        sorted(train_ids), pairs, dataa_camera, args.seed + 17
    )

    datab_records = load_detection_records(args.datab_detection_json)
    datab_camera = load_camera_by_path(args.datab_camera_jsonl)
    datab_selected, datab_audit = balanced_datab_records(
        datab_records,
        datab_camera,
        args.seed + 31,
        args.max_datab_per_bucket_per_class,
    )
    for bucket in ROUTE_BUCKETS:
        experts[bucket].extend(datab_selected[bucket])
        random.Random(args.seed + ROUTE_BUCKETS.index(bucket) + 211).shuffle(experts[bucket])
        if len(experts[bucket]) < args.minimum_records_per_expert:
            raise ValueError(
                f"route expert {bucket} has only {len(experts[bucket])} records; "
                f"minimum={args.minimum_records_per_expert}"
            )

    shared = [copy.deepcopy(row) for bucket in ROUTE_BUCKETS for row in experts[bucket]]
    random.Random(args.seed + 307).shuffle(shared)
    validate_partition(experts, shared)
    if any(camera_text_in_detection_prompt(row) for row in shared):
        raise AssertionError("camera text leaked into a hard-route detection prompt")

    if args.check_images:
        paths = {
            normalized_path(path)
            for row in shared + dataa_test_detection
            for path in row.get("images", [])
        }
        missing = [path for path in sorted(paths) if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"missing {len(missing)}/{len(paths)} referenced images; first={missing[0]}"
            )

    out_dir = Path(args.out_dir)
    payloads = {
        "hard_route_shared.json": shared,
        "hard_route_no_motion.json": experts["no-motion"],
        "hard_route_minor_motion.json": experts["minor-motion"],
        "hard_route_complex_motion.json": experts["complex-motion"],
        "hard_route_router_train.json": router_train,
        "dataa_test_detection.json": dataa_test_detection,
    }
    outputs: dict[str, Any] = {}
    for name, rows in payloads.items():
        path = out_dir / name
        write_json(path, rows)
        outputs[name] = {"path": str(path), "records": len(rows), "sha256": sha256(path)}
    for name, rows in {
        "dataa_route_dev_questions.jsonl": dataa_route_dev,
        "dataa_route_dev_gold.jsonl": dataa_route_gold,
    }.items():
        path = out_dir / name
        write_jsonl(path, rows)
        outputs[name] = {"path": str(path), "records": len(rows), "sha256": sha256(path)}

    split_rows = [
        {
            "case_id": case_id,
            "dataset_split": "train" if case_id in train_ids else "test",
            "source_family": source_family(case_id),
            "route_bucket": dataa_camera.get(case_id, {}).get("route_bucket", "unknown"),
            "camera_labels": dataa_camera.get(case_id, {}).get("labels", []),
            "camera_caption": dataa_camera.get(case_id, {}).get("caption", ""),
            "real_frame_dir": first_image(pairs[case_id]["real"]).rsplit("/", 1)[0],
            "fake_frame_dir": first_image(pairs[case_id]["fake"]).rsplit("/", 1)[0],
        }
        for case_id in sorted(pairs)
    ]
    split_path = out_dir / "dataa_hard_route_split_manifest.jsonl"
    write_jsonl(split_path, split_rows)
    outputs[split_path.name] = {
        "path": str(split_path),
        "records": len(split_rows),
        "sha256": sha256(split_path),
    }

    expert_summary = {}
    for bucket, rows in experts.items():
        expert_summary[bucket] = {
            "records": len(rows),
            "detection_labels": dict(Counter(str(row["detection_label"]) for row in rows)),
            "domains": dict(Counter(str(row["route_domain"]) for row in rows)),
            "sources": dict(Counter(str(row["gate_source"]) for row in rows)),
        }
    summary = {
        "schema_version": "camera_hard_route_gate_v1",
        "question": (
            "Does camera-motion-specific detection specialization improve Real/Fake detection when "
            "the route is inferred from the same frames and no camera text enters the detector?"
        ),
        "seed": args.seed,
        "route_buckets": list(ROUTE_BUCKETS),
        "route_priority_for_multilabel_annotations": list(ROUTE_PRIORITY),
        "static_alias_maps_to": "no-motion",
        "inputs": {
            "dataa_detection_json": args.dataa_detection_json,
            "dataa_camera_jsonl": args.dataa_camera_jsonl,
            "datab_detection_json": args.datab_detection_json,
            "datab_camera_jsonl": args.datab_camera_jsonl,
        },
        "dataa_split": {
            "kind": "case-level stratified by source family and three-class route bucket",
            "test_ratio": args.test_ratio,
            "total_cases": len(pairs),
            "train_cases": len(train_ids),
            "test_cases": len(test_ids),
            "train_test_overlap": sorted(train_ids & test_ids),
            "strata": split_strata,
            "excluded": dict(excluded_dataa),
        },
        "datab_selection": datab_audit,
        "experts": expert_summary,
        "shared_control": {
            "records": len(shared),
            "is_exact_expert_union": True,
            "unique_record_ids": len({str(row["route_record_id"]) for row in shared}),
            "unique_content_ids": len({str(row["route_content_id"]) for row in shared}),
            "duplicate_content_occurrences": len(shared)
            - len({str(row["route_content_id"]) for row in shared}),
        },
        "route_development": {
            "router_training_records": len(router_train),
            "router_training_answer_counts": dict(
                Counter(str(row["answer"]) for row in router_train)
            ),
            "router_training_per_question": router_train_audit,
            "router_static_and_no_motion_are_one_positive_class": True,
            "router_uses_real_frames_only": True,
            "heldout_dataa_videos": len(dataa_route_gold),
            "heldout_dataa_videos_with_known_route": len(dataa_route_dev) // len(ROUTE_BUCKETS),
            "three_binary_questions_per_known_video": True,
            "question_records": len(dataa_route_dev),
            "uses_real_and_fake_variants": True,
        },
        "integrity": {
            "camera_text_absent_from_all_detection_prompts": True,
            "dataa_real_fake_pairs_never_cross_split": True,
            "each_training_record_belongs_to_exactly_one_expert": True,
            "combined_expert_records_equal_shared_records": True,
            "datab_is_replay_not_heldout": True,
        },
        "outputs": outputs,
    }
    summary_path = out_dir / "camera_hard_route_data_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
