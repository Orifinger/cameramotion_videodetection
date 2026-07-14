#!/usr/bin/env python3
"""Build a detection-dominant camera-intermediate SFT/GRPO validation set."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_camera_joint_sft_gate import (  # noqa: E402
    CAMERA_LABEL_ORDER,
    canonical_labels,
    coarse_signature,
    data_parent,
    dataa_identity,
    detection_label,
    load_camera,
    load_camera_by_path,
    motion_bucket,
    normalized_path,
    read_json,
    source_family,
)


JOINT_SYSTEM_PROMPT = (
    "You are an AI-generated video detector. Infer global camera motion from the ordered "
    "frames, then classify the video. Camera motion is context, never direct evidence of "
    "authenticity. Output exactly two lines: "
    "<camera_motion>[\"label\", ...]</camera_motion> and "
    "<answer>Real</answer> or <answer>Fake</answer>. Use only these camera labels: "
    + ", ".join(CAMERA_LABEL_ORDER)
    + ". Do not output any other text."
)
JOINT_USER_SUFFIX = (
    "First infer the global camera motion across these ordered frames. Then decide whether "
    "the video is Real or Fake from visual and temporal evidence. Camera motion alone is not "
    "evidence of authenticity."
)
IMAGE_RE = re.compile(r"<image>")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, salt: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{salt}:{value}".encode("utf-8")).hexdigest()


def assistant_text(labels: Sequence[str], answer: str) -> str:
    payload = json.dumps(list(labels), ensure_ascii=False, separators=(",", ":"))
    return f"<camera_motion>{payload}</camera_motion>\n<answer>{answer}</answer>"


def frame_prompt(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    content = ""
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, Mapping) and message.get("role") == "user":
                content = str(message.get("content", ""))
                break
    images = record.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("joint record has no images")
    if content.count("<image>") != len(images):
        content = "Ordered frames:\n" + "\n".join(
            f"Frame {index + 1}: <image>" for index in range(len(images))
        )
    else:
        last_image = content.rfind("<image>")
        content = content[: last_image + len("<image>")].rstrip()
    return f"{content}\n\n{JOINT_USER_SUFFIX}"


def make_record(
    source: Mapping[str, Any],
    *,
    camera_labels: Sequence[str],
    source_dataset: str,
    sample_id: str,
    case_id: str,
    include_assistant: bool,
) -> dict[str, Any]:
    answer = detection_label(source)
    if answer not in {"Real", "Fake"}:
        raise ValueError(f"invalid detection label for {sample_id}: {answer!r}")
    labels = list(camera_labels)
    if not labels:
        raise ValueError(f"empty camera labels for {sample_id}")
    images = [normalized_path(path) for path in source.get("images", [])]
    messages: list[dict[str, str]] = [
        {"role": "system", "content": JOINT_SYSTEM_PROMPT},
        {"role": "user", "content": frame_prompt(source)},
    ]
    if include_assistant:
        messages.append({"role": "assistant", "content": assistant_text(labels, answer)})
    record = {
        "messages": messages,
        "images": images,
        "camera_labels": labels,
        "camera_labels_gold": labels,
        "camera_labels_reward": labels,
        "detection_label": answer,
        "label": answer,
        "sample_id": sample_id,
        "case_id": case_id,
        "source_dataset": source_dataset,
        "motion_bucket": motion_bucket(labels),
        "camera_signature": list(coarse_signature(labels)),
        "task_type": "camera_intermediate_detection",
    }
    token_count = sum(message["content"].count("<image>") for message in messages)
    if token_count != len(images):
        raise ValueError(f"image token/path mismatch for {sample_id}: {token_count} != {len(images)}")
    return record


def dataa_records(
    rows: Sequence[Mapping[str, Any]],
    camera_jsonl: str | Path,
    target_records: int,
    seed: int,
    *,
    include_assistant: bool,
) -> list[dict[str, Any]]:
    if target_records % 2:
        raise ValueError("DataA target records must be even")
    camera = load_camera(camera_jsonl)
    pairs: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        identity = dataa_identity(row)
        if identity is None:
            continue
        case_id, kind = identity
        pairs[case_id][kind] = row
    incomplete = [case_id for case_id, pair in pairs.items() if set(pair) != {"real", "fake"}]
    if incomplete:
        raise ValueError(f"incomplete DataA train pairs: {incomplete[:10]}")

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for case_id in sorted(pairs):
        annotation = camera.get(case_id)
        if annotation and annotation.get("labels"):
            labels = tuple(annotation["labels"])
            strata[(source_family(case_id), motion_bucket(labels))].append(case_id)
    for key, values in strata.items():
        values.sort(key=lambda value: stable_key(seed, "dataa:" + "|".join(key), value))

    target_cases = target_records // 2
    selected: list[str] = []
    positions: Counter[tuple[str, str]] = Counter()
    keys = sorted(strata)
    while len(selected) < target_cases:
        progressed = False
        for key in keys:
            position = positions[key]
            if position < len(strata[key]) and len(selected) < target_cases:
                selected.append(strata[key][position])
                positions[key] += 1
                progressed = True
        if not progressed:
            break
    if len(selected) != target_cases:
        raise ValueError(f"requested {target_cases} DataA cases, selected {len(selected)}")

    output: list[dict[str, Any]] = []
    for case_id in selected:
        labels = tuple(camera[case_id]["labels"])
        for kind in ("real", "fake"):
            output.append(
                make_record(
                    pairs[case_id][kind],
                    camera_labels=labels,
                    source_dataset="dataa_local_edit",
                    sample_id=f"dataa:{case_id}:{kind}",
                    case_id=case_id,
                    include_assistant=include_assistant,
                )
            )
    return output


def round_robin_sample(
    strata: Mapping[tuple[str, ...], Sequence[tuple[Mapping[str, Any], Sequence[str], str]]],
    target: int,
) -> list[tuple[Mapping[str, Any], Sequence[str], str]]:
    output: list[tuple[Mapping[str, Any], Sequence[str], str]] = []
    positions: Counter[tuple[str, ...]] = Counter()
    keys = sorted(strata)
    while len(output) < target:
        progressed = False
        for key in keys:
            position = positions[key]
            values = strata[key]
            if position < len(values) and len(output) < target:
                output.append(values[position])
                positions[key] += 1
                progressed = True
        if not progressed:
            break
    return output


def datab_records(
    rows: Sequence[Mapping[str, Any]],
    camera_jsonl: str | Path,
    target_records: int,
    seed: int,
    *,
    include_assistant: bool,
) -> list[dict[str, Any]]:
    if target_records % 2:
        raise ValueError("DataB target records must be even")
    camera = load_camera_by_path(camera_jsonl)
    by_label: dict[str, dict[tuple[str, ...], list[tuple[Mapping[str, Any], Sequence[str], str]]]] = {
        "Real": defaultdict(list),
        "Fake": defaultdict(list),
    }
    for index, row in enumerate(rows):
        answer = detection_label(row)
        labels = tuple(camera.get(data_parent(row), ()))
        if answer not in by_label or not labels:
            continue
        case_id = data_parent(row)
        signature = tuple(coarse_signature(labels))
        by_label[answer][signature].append((row, labels, case_id))
    per_label = target_records // 2
    selected: list[tuple[Mapping[str, Any], Sequence[str], str]] = []
    for answer in ("Real", "Fake"):
        for signature, values in by_label[answer].items():
            values.sort(
                key=lambda item: stable_key(seed, f"datab:{answer}:{'|'.join(signature)}", item[2])
            )
        chosen = round_robin_sample(by_label[answer], per_label)
        if len(chosen) != per_label:
            raise ValueError(f"requested {per_label} DataB {answer} records, selected {len(chosen)}")
        selected.extend(chosen)

    output: list[dict[str, Any]] = []
    for index, (row, labels, case_id) in enumerate(selected):
        output.append(
            make_record(
                row,
                camera_labels=labels,
                source_dataset="datab_full_generation_replay",
                sample_id=f"datab:{detection_label(row).lower()}:{index:05d}",
                case_id=case_id,
                include_assistant=include_assistant,
            )
        )
    return output


def best_rotation(label_sets: Sequence[tuple[str, ...]]) -> int:
    if len(label_sets) < 2:
        raise ValueError("cannot shuffle fewer than two camera annotations")
    scores = []
    for shift in range(1, len(label_sets)):
        changed = sum(
            label_sets[index] != label_sets[(index + shift) % len(label_sets)]
            for index in range(len(label_sets))
        )
        scores.append((changed, -shift, shift))
    return max(scores)[2]


def shuffled_reward_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = [copy.deepcopy(dict(row)) for row in records]
    dataa_by_case: dict[str, list[int]] = defaultdict(list)
    datab_by_answer: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(output):
        if row["source_dataset"] == "dataa_local_edit":
            dataa_by_case[str(row["case_id"])].append(index)
        else:
            datab_by_answer[str(row["detection_label"])].append(index)

    case_ids = sorted(dataa_by_case)
    case_labels = [tuple(output[dataa_by_case[case_id][0]]["camera_labels_gold"]) for case_id in case_ids]
    shift = best_rotation(case_labels)
    for position, case_id in enumerate(case_ids):
        donor = list(case_labels[(position + shift) % len(case_labels)])
        for index in dataa_by_case[case_id]:
            output[index]["camera_labels_reward"] = donor
            output[index]["camera_reward_source"] = "shuffled_within_dataa_case_units"

    for answer, indices in datab_by_answer.items():
        indices.sort(key=lambda index: str(output[index]["sample_id"]))
        labels = [tuple(output[index]["camera_labels_gold"]) for index in indices]
        shift = best_rotation(labels)
        for position, index in enumerate(indices):
            output[index]["camera_labels_reward"] = list(labels[(position + shift) % len(labels)])
            output[index]["camera_reward_source"] = f"shuffled_within_datab_{answer.lower()}"
    return output


def without_assistant(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in records:
        row = copy.deepcopy(dict(source))
        row["messages"] = [
            dict(message)
            for message in row["messages"]
            if isinstance(message, Mapping) and message.get("role") != "assistant"
        ]
        output.append(row)
    return output


def prompt_contract(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("messages"),
        row.get("images"),
        row.get("detection_label"),
        row.get("sample_id"),
    )


def select_smoke(records: Sequence[Mapping[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0 or count > len(records):
        count = len(records)
    ordered = sorted(
        records,
        key=lambda row: stable_key(seed, "smoke", str(row.get("sample_id"))),
    )
    selected = [copy.deepcopy(dict(row)) for row in ordered[:count]]
    if len({row["detection_label"] for row in selected}) < 2:
        raise ValueError("smoke split must contain both Real and Fake")
    return selected


def load_object_list(path: str | Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list) or any(not isinstance(row, Mapping) for row in payload):
        raise ValueError(f"expected JSON object list: {path}")
    return [dict(row) for row in payload]


def build_dataa_eval(
    rows: Sequence[Mapping[str, Any]], camera_jsonl: str | Path
) -> list[dict[str, Any]]:
    camera = load_camera(camera_jsonl)
    output: list[dict[str, Any]] = []
    for row in rows:
        identity = dataa_identity(row)
        if identity is None:
            continue
        case_id, kind = identity
        annotation = camera.get(case_id)
        if not annotation or not annotation.get("labels"):
            continue
        output.append(
            make_record(
                row,
                camera_labels=annotation["labels"],
                source_dataset="dataa_local_edit_test",
                sample_id=f"dataa-test:{case_id}:{kind}",
                case_id=case_id,
                include_assistant=True,
            )
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataa-train-json", required=True)
    parser.add_argument("--dataa-test-json", required=True)
    parser.add_argument("--dataa-camera-jsonl", required=True)
    parser.add_argument("--datab-replay-json", required=True)
    parser.add_argument("--datab-camera-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataa-records", type=int, default=512)
    parser.add_argument("--datab-records", type=int, default=512)
    parser.add_argument("--smoke-records", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--check-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataa_train = load_object_list(args.dataa_train_json)
    dataa_test = load_object_list(args.dataa_test_json)
    datab_replay = load_object_list(args.datab_replay_json)

    warm_records = dataa_records(
        dataa_train,
        args.dataa_camera_jsonl,
        args.dataa_records,
        args.seed + 11,
        include_assistant=True,
    ) + datab_records(
        datab_replay,
        args.datab_camera_jsonl,
        args.datab_records,
        args.seed + 23,
        include_assistant=True,
    )
    random.Random(args.seed + 31).shuffle(warm_records)
    correct = without_assistant(warm_records)
    shuffled = shuffled_reward_records(correct)
    detection_only = [copy.deepcopy(row) for row in correct]
    for row in detection_only:
        row["camera_reward_source"] = "not_rewarded"
    dataa_eval = build_dataa_eval(dataa_test, args.dataa_camera_jsonl)

    if [prompt_contract(row) for row in correct] != [prompt_contract(row) for row in shuffled]:
        raise AssertionError("correct and shuffled GRPO prompts differ")
    if [prompt_contract(row) for row in correct] != [prompt_contract(row) for row in detection_only]:
        raise AssertionError("correct and detection-only GRPO prompts differ")
    if any(any(message.get("role") == "assistant" for message in row["messages"]) for row in correct):
        raise AssertionError("assistant target leaked into GRPO prompts")

    changed = sum(
        row["camera_labels_reward"] != row["camera_labels_gold"] for row in shuffled
    )
    changed_rate = changed / len(shuffled) if shuffled else 0.0
    if changed_rate < 0.80:
        raise ValueError(f"shuffled camera labels changed only {changed_rate:.3f} of records")
    for source in {row["source_dataset"] for row in shuffled}:
        for answer in {row["detection_label"] for row in shuffled if row["source_dataset"] == source}:
            correct_marginal = Counter(
                tuple(row["camera_labels_gold"])
                for row in correct
                if row["source_dataset"] == source and row["detection_label"] == answer
            )
            shuffled_marginal = Counter(
                tuple(row["camera_labels_reward"])
                for row in shuffled
                if row["source_dataset"] == source and row["detection_label"] == answer
            )
            if correct_marginal != shuffled_marginal:
                raise AssertionError(f"camera marginal changed for {source}:{answer}")

    if args.check_images:
        missing = sorted(
            {
                path
                for row in warm_records + dataa_eval
                for path in row["images"]
                if not Path(path).is_file()
            }
        )
        if missing:
            raise FileNotFoundError(f"missing {len(missing)} image files; first={missing[0]}")

    outputs: dict[str, tuple[list[dict[str, Any]], str]] = {
        "joint_sft_warmup.json": (warm_records, "common format/coupling SFT"),
        "joint_sft_warmup_smoke.json": (
            select_smoke(warm_records, args.smoke_records, args.seed + 37),
            "common format/coupling SFT smoke",
        ),
        "joint_grpo_correct_camera.json": (correct, "correct camera reward"),
        "joint_grpo_shuffled_camera.json": (shuffled, "shuffled camera reward"),
        "joint_grpo_detection_only.json": (detection_only, "detection-only reward"),
        "dataa_test_joint_detection.json": (dataa_eval, "held-out DataA Real/Fake evaluation"),
    }
    for name, records in list(outputs.items()):
        if name.startswith("joint_grpo_"):
            outputs[name.replace(".json", "_smoke.json")] = (
                select_smoke(records[0], args.smoke_records, args.seed + 41),
                records[1] + " smoke",
            )

    out_dir = Path(args.out_dir)
    output_summary: dict[str, Any] = {}
    for name, (records, purpose) in outputs.items():
        path = out_dir / name
        write_json(path, records)
        output_summary[name] = {
            "path": str(path),
            "purpose": purpose,
            "records": len(records),
            "sha256": sha256(path),
        }

    dataa_case_ids = {row["case_id"] for row in correct if row["source_dataset"] == "dataa_local_edit"}
    eval_case_ids = {row["case_id"] for row in dataa_eval}
    source_counts = Counter(row["source_dataset"] for row in correct)
    answer_counts = Counter(row["detection_label"] for row in correct)
    summary = {
        "schema_version": "camera_detection_joint_grpo_v1",
        "question": (
            "Does a camera intermediate supervised inside the same rollout improve Real/Fake "
            "detection beyond equal-compute detection-only and shuffled-camera rewards?"
        ),
        "seed": args.seed,
        "inputs": {
            "dataa_train_json": args.dataa_train_json,
            "dataa_test_json": args.dataa_test_json,
            "dataa_camera_jsonl": args.dataa_camera_jsonl,
            "datab_replay_json": args.datab_replay_json,
            "datab_camera_jsonl": args.datab_camera_jsonl,
        },
        "prompt_contract": {
            "camera_text_provided_as_input": False,
            "camera_intermediate_precedes_detection_answer": True,
            "explanation_cot_trained": False,
            "system_prompt": JOINT_SYSTEM_PROMPT,
            "user_suffix": JOINT_USER_SUFFIX,
        },
        "selection": {
            "records": len(correct),
            "source_counts": dict(source_counts),
            "answer_counts": dict(answer_counts),
            "dataa_complete_pairs": source_counts["dataa_local_edit"] // 2,
            "dataa_train_eval_overlap": sorted(dataa_case_ids & eval_case_ids),
        },
        "controls": {
            "all_grpo_prompts_images_and_detection_labels_identical": True,
            "shuffled_camera_labels_changed_rate": changed_rate,
            "shuffled_preserves_camera_marginal_within_source_and_detection_label": True,
            "dataa_real_fake_pair_receives_same_camera_labels": all(
                len({tuple(correct[index]["camera_labels_gold"]) for index in indices}) == 1
                for indices in (
                    [i for i, row in enumerate(correct) if row["case_id"] == case_id]
                    for case_id in dataa_case_ids
                )
            ),
        },
        "reward_contract": {
            "correct_camera": {
                "detection_accuracy": 0.65,
                "camera_set_f1": 0.30,
                "strict_joint_format": 0.05,
            },
            "shuffled_camera": {
                "detection_accuracy": 0.65,
                "shuffled_camera_set_f1": 0.30,
                "strict_joint_format": 0.05,
            },
            "detection_only": {
                "detection_accuracy": 0.95,
                "strict_joint_format": 0.05,
            },
            "detection_is_lexicographically_dominant": 0.65 > 0.30 + 0.05,
        },
        "integrity": {
            "assistant_absent_from_all_grpo_prompts": True,
            "real_fake_balanced": answer_counts["Real"] == answer_counts["Fake"],
            "dataa_case_level_holdout": not (dataa_case_ids & eval_case_ids),
            "images_checked": bool(args.check_images),
        },
        "outputs": output_summary,
    }
    if not all(value for key, value in summary["integrity"].items() if key != "images_checked"):
        raise AssertionError(f"joint data integrity failed: {summary['integrity']}")
    summary_path = out_dir / "camera_detection_joint_grpo_data_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
