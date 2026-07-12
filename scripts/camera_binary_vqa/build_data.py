#!/usr/bin/env python3
"""Build balanced held-out binary camera-motion VQA records from DataA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CAMERA_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("very-unsteady", "Does the camera show very strong unsteadiness or shaking?"),
    ("unsteady", "Does the camera show noticeable unsteadiness or shaking?"),
    ("minimal-shaking", "Does the camera show only minimal shaking or wobble?"),
    ("no-shaking", "Is the camera free from visible shaking or wobble?"),
    ("complex-motion", "Does the camera have noticeable motion beyond minor shake or wobble?"),
    ("minor-motion", "Does the camera have only minor movement, such as slight shake or wobble?"),
    ("no-motion", "Is the camera completely still without visible movement?"),
    ("fast-speed", "Does the camera move at a fast speed?"),
    ("regular-speed", "Does the camera move at a regular speed?"),
    ("slow-speed", "Does the camera move at a slow speed?"),
    ("dolly-in", "Does the camera dolly forward into the scene?"),
    ("dolly-out", "Does the camera dolly backward out of the scene?"),
    ("truck-left", "Does the camera move laterally to the left?"),
    ("truck-right", "Does the camera move laterally to the right?"),
    ("pedestal-up", "Does the camera move vertically upward?"),
    ("pedestal-down", "Does the camera move vertically downward?"),
    ("pan-left", "Does the camera pan to the left?"),
    ("pan-right", "Does the camera pan to the right?"),
    ("tilt-up", "Does the camera tilt upward?"),
    ("tilt-down", "Does the camera tilt downward?"),
    ("roll-CW", "Does the camera roll clockwise?"),
    ("roll-CCW", "Does the camera roll counterclockwise?"),
    ("zoom-in", "Does the camera zoom in?"),
    ("zoom-out", "Does the camera zoom out?"),
    ("arc-CW", "Does the camera move in a clockwise arc around the scene?"),
    ("arc-CCW", "Does the camera move in a counterclockwise arc around the scene?"),
    ("side-tracking", "Does the camera track the subject from the side?"),
    ("lead-tracking", "Does the camera move ahead of and track the subject?"),
    ("tail-tracking", "Does the camera follow and track the subject from behind?"),
    ("aerial-tracking", "Does an aerial camera track the subject?"),
    ("arc-tracking", "Does the camera track the subject along an arc?"),
    ("pan-tracking", "Does the camera pan to keep a moving subject in view?"),
    ("tilt-tracking", "Does the camera tilt to keep a moving subject in view?"),
)
QUESTION_BY_LABEL = dict(CAMERA_QUESTIONS)
ALLOWED_LABELS = set(QUESTION_BY_LABEL)
SYSTEM_PROMPT = (
    "You are a camera-motion analyst. Answer each question with exactly Yes or No. "
    "Judge global camera behavior rather than object motion."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
            rows.append(dict(value))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_order(rows: Sequence[Mapping[str, Any]], salt: str) -> list[dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> str:
        case_id = str(row.get("case_id", ""))
        return hashlib.sha256(f"{salt}:{case_id}".encode("utf-8")).hexdigest()

    return [dict(row) for row in sorted(rows, key=key)]


def normalize_manifest(rows: Sequence[Mapping[str, Any]], check_videos: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        split = str(row.get("dataset_split") or "").strip()
        video = str(row.get("real_video") or "").strip()
        labels = {str(value).strip() for value in row.get("camera_labels") or [] if str(value).strip()}
        unknown = sorted(labels - ALLOWED_LABELS - {"static"})
        if unknown:
            raise ValueError(f"unknown camera labels for {case_id}: {unknown}")
        if not case_id or case_id in seen:
            raise ValueError(f"missing or duplicate case_id: {case_id!r}")
        if split not in {"train", "test"}:
            raise ValueError(f"invalid dataset_split for {case_id}: {split!r}")
        if not video:
            raise ValueError(f"missing real_video for {case_id}")
        if check_videos and not Path(video).is_file():
            raise FileNotFoundError(f"missing real video for {case_id}: {video}")
        seen.add(case_id)
        output.append(
            {
                "case_id": case_id,
                "dataset_split": split,
                "real_video": video,
                "camera_labels": sorted(labels & ALLOWED_LABELS),
                "source_family": row.get("source_name"),
                "motion_bucket": row.get("motion_bucket"),
            }
        )
    return output


def make_record(
    row: Mapping[str, Any], label: str, answer: str, pair_id: str, split: str
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"<video>{QUESTION_BY_LABEL[label]} Answer only Yes or No.",
            },
        ],
        "videos": [str(row["real_video"])],
        "target_text": answer,
        "answer": answer,
        "answer_id": 1 if answer == "Yes" else 0,
        "camera_primitive": label,
        "case_id": row["case_id"],
        "visual_source_case_id": row["case_id"],
        "pair_id": pair_id,
        "sample_id": f"{split}:{label}:{pair_id}:{answer}",
        "dataset_split": split,
        "source_family": row.get("source_family"),
        "motion_bucket": row.get("motion_bucket"),
        "visual_condition": "matched_video",
        "assistant_prefix": "",
    }


def opposite_video_controls(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pair_id"]), []).append(dict(row))
    output: list[dict[str, Any]] = []
    for pair_id, pair_rows in sorted(by_pair.items()):
        if len(pair_rows) != 2 or {row["answer"] for row in pair_rows} != {"Yes", "No"}:
            raise ValueError(f"invalid held-out answer pair: {pair_id}")
        yes = next(row for row in pair_rows if row["answer"] == "Yes")
        no = next(row for row in pair_rows if row["answer"] == "No")
        for original, donor in ((yes, no), (no, yes)):
            controlled = dict(original)
            controlled["videos"] = list(donor["videos"])
            controlled["visual_source_case_id"] = donor["case_id"]
            controlled["visual_condition"] = "opposite_label_video"
            output.append(controlled)
    return output


def no_video_controls(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        controlled = dict(row)
        controlled["messages"] = [dict(message) for message in row["messages"]]
        controlled["messages"][-1]["content"] = str(controlled["messages"][-1]["content"]).replace(
            "<video>", ""
        )
        controlled["videos"] = []
        controlled["visual_source_case_id"] = None
        controlled["visual_condition"] = "no_video"
        output.append(controlled)
    return output


def balanced_pairs(
    rows: Sequence[Mapping[str, Any]],
    split: str,
    seed: int,
    max_per_class: int,
    minimum_per_class: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for label, _ in CAMERA_QUESTIONS:
        positives = [row for row in rows if label in set(row["camera_labels"])]
        negatives = [row for row in rows if label not in set(row["camera_labels"])]
        positives = stable_order(positives, f"{seed}:{split}:{label}:yes")
        negatives = stable_order(negatives, f"{seed}:{split}:{label}:no")
        selected = min(len(positives), len(negatives))
        if max_per_class > 0:
            selected = min(selected, max_per_class)
        eligible = selected >= minimum_per_class
        stats[label] = {
            "available_positive": len(positives),
            "available_negative": len(negatives),
            "selected_per_answer": selected if eligible else 0,
            "eligible": eligible,
        }
        if not eligible:
            continue
        for index, (positive, negative) in enumerate(zip(positives[:selected], negatives[:selected])):
            pair_id = f"{label}:{index:04d}"
            records.append(make_record(positive, label, "Yes", pair_id, split))
            records.append(make_record(negative, label, "No", pair_id, split))
    records = stable_order(records, f"{seed}:{split}:all")
    return records, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-per-class", type=int, default=0)
    parser.add_argument("--max-dev-per-class", type=int, default=64)
    parser.add_argument("--min-train-per-class", type=int, default=8)
    parser.add_argument("--min-dev-per-class", type=int, default=4)
    parser.add_argument("--minimum-eligible-labels", type=int, default=20)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--check-videos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = normalize_manifest(read_jsonl(args.manifest_jsonl), args.check_videos)
    train_cases = [row for row in manifest if row["dataset_split"] == "train"]
    dev_cases = [row for row in manifest if row["dataset_split"] == "test"]
    train, train_stats = balanced_pairs(
        train_cases,
        "train",
        args.seed,
        args.max_train_per_class,
        args.min_train_per_class,
    )
    dev, dev_stats = balanced_pairs(
        dev_cases,
        "dev",
        args.seed + 1,
        args.max_dev_per_class,
        args.min_dev_per_class,
    )
    train_labels = {row["camera_primitive"] for row in train}
    dev_labels = {row["camera_primitive"] for row in dev}
    eligible_labels = train_labels & dev_labels
    if len(eligible_labels) < args.minimum_eligible_labels:
        raise ValueError(
            f"only {len(eligible_labels)} labels have train/dev support; "
            f"minimum={args.minimum_eligible_labels}"
        )
    train = [row for row in train if row["camera_primitive"] in eligible_labels]
    dev = [row for row in dev if row["camera_primitive"] in eligible_labels]
    train_case_ids = {row["case_id"] for row in train}
    dev_case_ids = {row["case_id"] for row in dev}
    overlap = sorted(train_case_ids & dev_case_ids)
    if overlap:
        raise AssertionError(f"train/dev case leakage: {overlap[:20]}")

    payloads = {
        "train_balanced.jsonl": train,
        "dev_matched_video.jsonl": dev,
        "dev_opposite_label_video.jsonl": opposite_video_controls(dev),
        "dev_no_video.jsonl": no_video_controls(dev),
    }
    outputs: dict[str, Any] = {}
    for name, rows in payloads.items():
        path = args.output_dir / name
        write_jsonl(path, rows)
        outputs[name] = {"path": str(path), "records": len(rows), "sha256": sha256(path)}

    answer_counts = Counter(row["answer"] for row in train)
    steps_per_epoch = math.ceil(len(train) / args.world_size)
    summary = {
        "schema_version": "dataa_camera_binary_vqa_v1",
        "manifest_jsonl": str(args.manifest_jsonl),
        "seed": args.seed,
        "case_counts": {"train": len(train_cases), "dev": len(dev_cases)},
        "train_dev_case_overlap": overlap,
        "eligible_labels": sorted(eligible_labels),
        "num_eligible_labels": len(eligible_labels),
        "excluded_labels": sorted(ALLOWED_LABELS - eligible_labels),
        "train_records": len(train),
        "dev_records_per_condition": len(dev),
        "train_answer_counts": dict(answer_counts),
        "world_size": args.world_size,
        "steps_per_epoch": steps_per_epoch,
        "train_per_label": train_stats,
        "dev_per_label": dev_stats,
        "outputs": outputs,
    }
    write_json(args.output_dir / "data_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
