#!/usr/bin/env python3
"""Build leakage-audited DataA counterfactual DPO and evaluation sets.

The preferred region source is the VACE grounded-CoT input index, whose
``mask_npz`` points to the real generation mask.  Falling back to the
auto-authored evidence bbox is allowed only for diagnostic runs and is marked
explicitly in the output summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

CASE_RE = re.compile(r"(dataA_v1_(?:dataset_v2_|textedit_reserve_)?\d+)/(real|fake)")
BBOX_RE = re.compile(r"<bbox>\s*\[([^]]+)\]\s*</bbox>", re.IGNORECASE)
TIME_RE = re.compile(r"<t>\s*\[([^]]+)\]\s*</t>", re.IGNORECASE)
TYPE_RE = re.compile(r"<type>\s*([^<]+)\s*</type>", re.IGNORECASE)
PROMPT_TS_RE = re.compile(r"\[T=([0-9.]+)s\]")

SYSTEM_PROMPT = (
    "You are a forensic video comparison analyst. The two ordered videos come "
    "from the same source clip and share global camera motion. Exactly one video "
    "contains a local AI-generated edit. Compare local spatiotemporal evidence; "
    "never treat camera motion itself as Real/Fake evidence. Output only the "
    "requested XML tags."
)


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


def normalize_path(value: Any) -> str:
    return str(value or "").replace("\\", "/")


def parse_path_maps(values: Sequence[str]) -> list[tuple[str, str]]:
    mappings = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--path-map must be OLD=NEW, got {value!r}")
        old, new = value.split("=", 1)
        mappings.append((normalize_path(old).rstrip("/"), normalize_path(new).rstrip("/")))
    return mappings


def map_path(value: Any, mappings: Sequence[tuple[str, str]]) -> str:
    path = normalize_path(value)
    for old, new in mappings:
        if path == old or path.startswith(old + "/"):
            return new + path[len(old) :]
    return path


def assistant_text(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                return str(message.get("content", ""))
    return ""


def user_text(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, Mapping) and message.get("role") == "user":
                return str(message.get("content", ""))
    return ""


def case_split(record: Mapping[str, Any]) -> tuple[str, str] | None:
    images = record.get("images")
    first = normalize_path(images[0]) if isinstance(images, list) and images else ""
    match = CASE_RE.search(first)
    return (match.group(1), match.group(2)) if match else None


def parse_csv_numbers(pattern: re.Pattern[str], text: str, expected: int) -> list[float] | None:
    match = pattern.search(text)
    if not match:
        return None
    try:
        values = [float(part.strip()) for part in match.group(1).split(",")]
    except ValueError:
        return None
    return values if len(values) == expected else None


def parse_sample(record: Mapping[str, Any], mappings: Sequence[tuple[str, str]]) -> dict[str, Any] | None:
    identity = case_split(record)
    if identity is None:
        return None
    case_id, split = identity
    images = [map_path(path, mappings) for path in record.get("images", [])]
    target = assistant_text(record)
    timestamps = [float(value) for value in PROMPT_TS_RE.findall(user_text(record))]
    if len(timestamps) != len(images):
        timestamps = [float(index) for index in range(len(images))]
    type_match = TYPE_RE.search(target)
    return {
        "case_id": case_id,
        "split": split,
        "images": images,
        "timestamps": timestamps,
        "evidence_bbox_1000": parse_csv_numbers(BBOX_RE, target, 4),
        "evidence_time": parse_csv_numbers(TIME_RE, target, 2),
        "artifact_type": type_match.group(1).strip() if type_match else "Unknown Artifact",
    }


def load_detection_pairs(path: str | Path, mappings: Sequence[tuple[str, str]]) -> dict[str, dict[str, dict[str, Any]]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for record in payload:
        if not isinstance(record, Mapping):
            continue
        sample = parse_sample(record, mappings)
        if sample:
            pairs.setdefault(sample["case_id"], {})[sample["split"]] = sample
    return {case_id: pair for case_id, pair in pairs.items() if set(pair) == {"real", "fake"}}


def load_camera(path: str | Path) -> dict[tuple[str, str], list[str]]:
    camera = {}
    for row in read_jsonl(path):
        match = CASE_RE.search(normalize_path(row.get("path")))
        if match:
            camera[(match.group(1), match.group(2))] = [str(label) for label in row.get("labels", [])]
    return camera


def clean_camera_labels(labels: Sequence[str]) -> list[str]:
    excluded = {"static"}
    return sorted({str(label).strip() for label in labels if str(label).strip().casefold() not in excluded})


def camera_bucket(labels: Sequence[str]) -> str:
    present = {label.casefold() for label in labels}
    if "complex-motion" in present:
        return "complex-motion"
    if "minor-motion" in present:
        return "minor-motion"
    if "no-motion" in present:
        return "no-motion"
    return "unknown"


def load_grounded_index(path: str | None, mappings: Sequence[tuple[str, str]]) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    output = {}
    for row in read_jsonl(path):
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue
        evidence = row.get("evidence_mask") if isinstance(row.get("evidence_mask"), Mapping) else {}
        shape = evidence.get("mask_shape") if isinstance(evidence.get("mask_shape"), Mapping) else {}
        output[case_id] = {
            "mask_npz": map_path(row.get("mask_npz") or evidence.get("mask_npz_path"), mappings),
            "mask_width": int(shape.get("width") or 0),
            "mask_height": int(shape.get("height") or 0),
            "edit_bbox_xyxy": row.get("edit_bbox_xyxy") or evidence.get("union_bbox_xyxy"),
            "edit_time_range_sec": row.get("edit_time_range_sec"),
            "operation": row.get("operation"),
            "generator_route": row.get("generator_route"),
            "vace_model": row.get("vace_model"),
            "target_text": row.get("target_text"),
            "case_manifest": map_path(row.get("case_manifest"), mappings),
        }
    return output


def normalized_bbox(box: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
        return None
    try:
        values = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = values
    return [
        round(max(0.0, min(1000.0, x1 * 1000.0 / width)), 3),
        round(max(0.0, min(1000.0, y1 * 1000.0 / height)), 3),
        round(max(0.0, min(1000.0, x2 * 1000.0 / width)), 3),
        round(max(0.0, min(1000.0, y2 * 1000.0 / height)), 3),
    ]


def negative_bbox(box: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    width = height = 150.0
    candidates = [
        [0.0, 0.0, width, height],
        [1000.0 - width, 0.0, 1000.0, height],
        [0.0, 1000.0 - height, width, 1000.0],
        [1000.0 - width, 1000.0 - height, 1000.0, 1000.0],
    ]

    def iou(candidate: Sequence[float]) -> float:
        ax1, ay1, ax2, ay2 = box
        bx1, by1, bx2, by2 = candidate
        inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    return [round(value, 3) for value in min(candidates, key=iou)]


def extract_case_ids(path: str | None) -> set[str] | None:
    if not path:
        return None
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected split JSON list: {path}")
    output = set()
    for record in payload:
        if isinstance(record, Mapping):
            identity = case_split(record)
            if identity:
                output.add(identity[0])
    return output


def split_cases(case_ids: Sequence[str], train_path: str | None, test_path: str | None, seed: int) -> tuple[set[str], set[str], str]:
    all_cases = set(case_ids)
    train = extract_case_ids(train_path)
    test = extract_case_ids(test_path)
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
    overlap = train & test
    if overlap:
        raise ValueError(f"train/test case leakage: {sorted(overlap)[:20]}")
    if not train or not test:
        raise ValueError("both train and test case sets must be non-empty")
    return train, test, source


def evenly_select(sample: Mapping[str, Any], max_frames: int) -> tuple[list[str], list[float]]:
    images, timestamps = list(sample["images"]), list(sample["timestamps"])
    if max_frames <= 0 or len(images) <= max_frames:
        return images, timestamps
    indices = sorted({round(index * (len(images) - 1) / (max_frames - 1)) for index in range(max_frames)})
    return [images[index] for index in indices], [timestamps[index] for index in indices]


def frame_lines(label: str, timestamps: Sequence[float]) -> list[str]:
    return [f"[Video {label} T={timestamp:.2f}s] <image>" for timestamp in timestamps]


def pair_user(a_times: Sequence[float], b_times: Sequence[float]) -> str:
    return "\n".join(
        [
            "Video A frames:",
            *frame_lines("A", a_times),
            "",
            "Video B frames:",
            *frame_lines("B", b_times),
            "",
            "The videos share source content and global camera motion.",
            "Identify the locally AI-edited video and localize the edit.",
            "Return <edited_video>A|B</edited_video> and <edit_bbox>[x1,y1,x2,y2]</edit_bbox>.",
        ]
    )


def answer(choice: str, bbox: Sequence[float], labels: Sequence[str], camera_aware: bool) -> str:
    parts = []
    if camera_aware:
        parts.append(f"<camera_motion>{json.dumps(list(labels), ensure_ascii=False)}</camera_motion>")
    parts.extend(
        [
            f"<edited_video>{choice}</edited_video>",
            f"<edit_bbox>{json.dumps(list(bbox), ensure_ascii=False)}</edit_bbox>",
        ]
    )
    return "\n".join(parts)


def build_pair_record(meta: Mapping[str, Any], real_first: bool, camera_aware: bool, max_frames: int) -> dict[str, Any]:
    real, fake = meta["real"], meta["fake"]
    real_images, real_times = evenly_select(real, max_frames)
    fake_images, fake_times = evenly_select(fake, max_frames)
    if real_first:
        a_images, a_times, a_split = real_images, real_times, "real"
        b_images, b_times, b_split = fake_images, fake_times, "fake"
        edited = "B"
        order = "real_first"
    else:
        a_images, a_times, a_split = fake_images, fake_times, "fake"
        b_images, b_times, b_split = real_images, real_times, "real"
        edited = "A"
        order = "fake_first"
    content = answer(edited, meta["bbox_1000"], meta["camera_labels"], camera_aware)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair_user(a_times, b_times)},
            {"role": "assistant", "content": content},
        ],
        "images": a_images + b_images,
        "case_id": meta["case_id"],
        "sample_id": f"{meta['case_id']}:{order}:{'camera' if camera_aware else 'local'}",
        "pair_order": order,
        "edited_video": edited,
        "video_a_source_split": a_split,
        "video_b_source_split": b_split,
        "edit_bbox_1000": meta["bbox_1000"],
        "negative_bbox_1000": meta["negative_bbox_1000"],
        "region_source": meta["region_source"],
        "mask_npz": meta["mask_npz"],
        "camera_labels": meta["camera_labels"],
        "motion_bucket": meta["motion_bucket"],
        "artifact_type": meta["artifact_type"],
    }


def preference_records(eval_record: Mapping[str, Any], camera_aware: bool) -> list[dict[str, Any]]:
    messages = [dict(message) for message in eval_record["messages"] if message.get("role") != "assistant"]
    edited = str(eval_record["edited_video"])
    wrong = "B" if edited == "A" else "A"
    common = {key: value for key, value in eval_record.items() if key not in {"messages"}}
    chosen = answer(edited, eval_record["edit_bbox_1000"], eval_record["camera_labels"], camera_aware)
    wrong_choice = answer(wrong, eval_record["edit_bbox_1000"], eval_record["camera_labels"], camera_aware)
    wrong_region = answer(edited, eval_record["negative_bbox_1000"], eval_record["camera_labels"], camera_aware)
    return [
        {
            **common,
            "messages": messages,
            "chosen": {"role": "assistant", "content": chosen},
            "rejected": {"role": "assistant", "content": wrong_choice},
            "preference_kind": "video_choice",
        },
        {
            **common,
            "messages": messages,
            "chosen": {"role": "assistant", "content": chosen},
            "rejected": {"role": "assistant", "content": wrong_region},
            "preference_kind": "localization",
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-json", required=True)
    parser.add_argument("--camera-jsonl", required=True)
    parser.add_argument("--grounded-index-jsonl")
    parser.add_argument("--dataa-train-json")
    parser.add_argument("--dataa-test-json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--max-train-cases", type=int, default=0)
    parser.add_argument("--max-test-cases", type=int, default=0)
    parser.add_argument("--require-true-mask", action="store_true")
    parser.add_argument("--check-mask-files", action="store_true")
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def take_cases(case_ids: set[str], limit: int, seed: int) -> set[str]:
    ordered = sorted(case_ids)
    random.Random(seed).shuffle(ordered)
    return set(ordered[:limit]) if limit > 0 else set(ordered)


def main() -> None:
    args = parse_args()
    mappings = parse_path_maps(args.path_map)
    pairs = load_detection_pairs(args.detection_json, mappings)
    camera = load_camera(args.camera_jsonl)
    grounded = load_grounded_index(args.grounded_index_jsonl, mappings)
    train_cases, test_cases, split_source = split_cases(
        sorted(pairs), args.dataa_train_json, args.dataa_test_json, args.seed
    )
    train_cases = take_cases(train_cases, args.max_train_cases, args.seed + 1)
    test_cases = take_cases(test_cases, args.max_test_cases, args.seed + 2)

    manifest = []
    camera_mismatch = []
    for case_id in sorted(train_cases | test_cases):
        pair = pairs[case_id]
        real_labels = clean_camera_labels(camera.get((case_id, "real"), []))
        fake_labels = clean_camera_labels(camera.get((case_id, "fake"), []))
        if real_labels and fake_labels and real_labels != fake_labels:
            camera_mismatch.append(case_id)
        labels = real_labels or fake_labels
        index = grounded.get(case_id, {})
        true_bbox = normalized_bbox(
            index.get("edit_bbox_xyxy"), int(index.get("mask_width") or 0), int(index.get("mask_height") or 0)
        )
        if true_bbox:
            bbox, region_source = true_bbox, "vace_M_gen_union_bbox"
        else:
            bbox, region_source = pair["fake"].get("evidence_bbox_1000"), "auto_evidence_bbox_fallback"
        if not bbox:
            raise ValueError(f"no usable region for case {case_id}")
        mask_npz = str(index.get("mask_npz") or "")
        if args.check_mask_files and mask_npz and not Path(mask_npz).is_file():
            raise FileNotFoundError(f"mask not found for {case_id}: {mask_npz}")
        manifest.append(
            {
                "case_id": case_id,
                "dataset_split": "train" if case_id in train_cases else "test",
                "real": pair["real"],
                "fake": pair["fake"],
                "camera_labels": labels,
                "motion_bucket": camera_bucket(labels),
                "camera_pair_consistent": bool(real_labels and fake_labels and real_labels == fake_labels),
                "artifact_type": pair["fake"]["artifact_type"],
                "bbox_1000": [round(float(value), 3) for value in bbox],
                "negative_bbox_1000": negative_bbox(bbox),
                "region_source": region_source,
                "mask_npz": mask_npz,
                "mask_width": int(index.get("mask_width") or 0),
                "mask_height": int(index.get("mask_height") or 0),
                "operation": index.get("operation"),
                "generator_route": index.get("generator_route"),
                "vace_model": index.get("vace_model"),
                "case_manifest": index.get("case_manifest"),
            }
        )

    true_mask_cases = sum(row["region_source"] == "vace_M_gen_union_bbox" and bool(row["mask_npz"]) for row in manifest)
    if args.require_true_mask and true_mask_cases != len(manifest):
        raise ValueError(f"true masks required, available for {true_mask_cases}/{len(manifest)} cases")

    train_meta = [row for row in manifest if row["dataset_split"] == "train"]
    test_meta = [row for row in manifest if row["dataset_split"] == "test"]
    eval_local, eval_camera = [], []
    dpo_local, dpo_camera = [], []
    for row in train_meta:
        for real_first in (True, False):
            local_record = build_pair_record(row, real_first, False, args.frames_per_video)
            camera_record = build_pair_record(row, real_first, True, args.frames_per_video)
            dpo_local.extend(preference_records(local_record, False))
            dpo_camera.extend(preference_records(camera_record, True))
    for row in test_meta:
        for real_first in (True, False):
            eval_local.append(build_pair_record(row, real_first, False, args.frames_per_video))
            eval_camera.append(build_pair_record(row, real_first, True, args.frames_per_video))

    out_dir = Path(args.out_dir)
    outputs = {
        "pair_manifest": out_dir / "dataa_counterfactual_pair_manifest.jsonl",
        "dpo_local_only": out_dir / "dataa_counterfactual_dpo_local_only.json",
        "dpo_camera_aware": out_dir / "dataa_counterfactual_dpo_camera_aware.json",
        "eval_local_only": out_dir / "dataa_counterfactual_eval_local_only.json",
        "eval_camera_aware": out_dir / "dataa_counterfactual_eval_camera_aware.json",
    }
    write_jsonl(outputs["pair_manifest"], manifest)
    write_json(outputs["dpo_local_only"], dpo_local)
    write_json(outputs["dpo_camera_aware"], dpo_camera)
    write_json(outputs["eval_local_only"], eval_local)
    write_json(outputs["eval_camera_aware"], eval_camera)

    summary = {
        "schema_version": "dataa_counterfactual_gate_sets_v1",
        "seed": args.seed,
        "split_source": split_source,
        "formal_gate_eligible": true_mask_cases == len(manifest),
        "counts": {
            "complete_pairs": len(pairs),
            "selected_train_cases": len(train_meta),
            "selected_test_cases": len(test_meta),
            "true_mask_cases": true_mask_cases,
            "fallback_bbox_cases": len(manifest) - true_mask_cases,
            "camera_pair_mismatches": len(camera_mismatch),
            "dpo_records_per_variant": len(dpo_local),
            "eval_records_per_variant": len(eval_local),
        },
        "leakage_audit": {
            "train_test_case_overlap": sorted(train_cases & test_cases),
            "all_eval_cases_held_out": not bool(train_cases & test_cases),
            "user_prompts_contain_no_gt_bbox": all(
                json.dumps(record["edit_bbox_1000"], ensure_ascii=False) not in user_text(record)
                and str(record["edit_bbox_1000"]) not in user_text(record)
                for record in eval_local
            ),
        },
        "train_motion_buckets": dict(Counter(row["motion_bucket"] for row in train_meta)),
        "test_motion_buckets": dict(Counter(row["motion_bucket"] for row in test_meta)),
        "artifact_types": dict(Counter(row["artifact_type"] for row in manifest)),
        "camera_mismatch_examples": camera_mismatch[:20],
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in outputs.items()
        },
    }
    summary_path = out_dir / "dataa_counterfactual_gate_sets_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
