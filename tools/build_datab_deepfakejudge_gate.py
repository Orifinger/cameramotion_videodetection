from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


ANSWER_RE = re.compile(r"<answer>\s*(Real|Fake)\s*</answer>", re.IGNORECASE)
TYPE_RE = re.compile(r"<type>\s*([^<]+?)\s*</type>", re.IGNORECASE)
BBOX_RE = re.compile(
    r"<bbox>\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*</bbox>",
    re.IGNORECASE,
)
TIME_RE = re.compile(
    r"<t>\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*</t>",
    re.IGNORECASE,
)
FRAME_TIME_RE = re.compile(r"\[T\s*=\s*(-?\d+(?:\.\d+)?)s\]", re.IGNORECASE)

ARTIFACT_CATEGORIES = (
    "Hand Anatomy Error",
    "Limb Structure Error",
    "Body Proportion Error",
    "Face Identity Drift",
    "Facial Landmark Distortion",
    "Face Boundary Fusion",
    "Malformed Text",
    "Inconsistent Text Across Frames",
    "Logo / Symbol Distortion",
    "Object Deformation",
    "Object Identity Drift",
    "Object Part Inconsistency",
    "Boundary Fusion",
    "Contact Region Artifact",
    "Occlusion Error",
    "Texture Flicker",
    "Material Inconsistency",
    "Lighting / Shadow Inconsistency",
    "Entity Reappearance Change",
    "Cross-frame Identity Drift",
    "Object Category Shift",
    "Implausible Contact",
    "Motion Discontinuity",
    "Physical Interaction Error",
    "Known-person Factual Implausibility",
    "Non-realistic Event Premise",
    "Role / Context Contradiction",
    "Synthetic Rendering Cue",
    "Over-smoothed Generated Texture",
    "Stylized / CGI Rendering Inconsistency",
)
ARTIFACT_CATEGORY_SET = set(ARTIFACT_CATEGORIES)
SUSPICIOUS_TEXT_MARKERS = ("\ufffd", "鈥", "淐", "锛", "銆")


def read_json(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def assistant_response(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, Mapping) and str(message.get("role", "")).lower() == "assistant":
            return str(message.get("content", ""))
    return ""


def user_prompt(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, Mapping) and str(message.get("role", "")).lower() == "user":
            return str(message.get("content", ""))
    return ""


def image_paths(row: Mapping[str, Any]) -> list[str]:
    images = row.get("images")
    if not isinstance(images, list):
        return []
    return [str(path) for path in images]


def derive_ground_truth(images: Sequence[str]) -> str:
    if not images:
        raise ValueError("record has no images")
    parts = [part.lower() for part in PurePosixPath(images[0].replace("\\", "/")).parts]
    labels = {label for label in ("real", "fake") if label in parts}
    if len(labels) != 1:
        raise ValueError(f"cannot derive one real/fake label from path: {images[0]}")
    return next(iter(labels))


def source_bucket(images: Sequence[str], label: str) -> str:
    if not images:
        return "unknown"
    parts = list(PurePosixPath(images[0].replace("\\", "/")).parts)
    lowered = [part.lower() for part in parts]
    try:
        index = lowered.index(label.lower())
    except ValueError:
        return "unknown"
    return parts[index + 1].casefold() if index + 1 < len(parts) else "unknown"


def sample_id(index: int, images: Sequence[str]) -> str:
    anchor = images[0] if images else f"row:{index}"
    digest = hashlib.sha1(anchor.encode("utf-8")).hexdigest()[:16]
    return f"datab_{index:06d}_{digest}"


def frame_times(row: Mapping[str, Any], count: int) -> list[float]:
    values = [float(value) for value in FRAME_TIME_RE.findall(user_prompt(row))]
    if len(values) == count:
        return values
    return [float(index) for index in range(count)]


def static_audit(response: str, gt_label: str, times: Sequence[float]) -> dict[str, Any]:
    answer_match = ANSWER_RE.search(response)
    answer = answer_match.group(1).lower() if answer_match else None
    types = [value.strip() for value in TYPE_RE.findall(response)]
    bboxes = [tuple(float(value) for value in match) for match in BBOX_RE.findall(response)]
    intervals = [tuple(float(value) for value in match) for match in TIME_RE.findall(response)]
    bbox_valid = all(0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000 for x1, y1, x2, y2 in bboxes)
    type_valid = all(value in ARTIFACT_CATEGORY_SET for value in types)
    if times:
        low, high = min(times), max(times)
        time_valid = all(low <= start <= end <= high + 1e-6 for start, end in intervals)
    else:
        time_valid = False if intervals else True
    hard_fail_reasons: list[str] = []
    if answer is None:
        hard_fail_reasons.append("missing_answer")
    elif answer != gt_label:
        hard_fail_reasons.append("answer_gt_mismatch")
    if not bbox_valid:
        hard_fail_reasons.append("invalid_bbox")
    if not type_valid:
        hard_fail_reasons.append("invalid_artifact_type")
    if not time_valid:
        hard_fail_reasons.append("invalid_time_interval")
    return {
        "candidate_answer": answer,
        "answer_matches_gt": answer == gt_label,
        "artifact_types": types,
        "artifact_type_count": len(types),
        "artifact_types_valid": type_valid,
        "bbox_count": len(bboxes),
        "bboxes_valid": bbox_valid,
        "time_interval_count": len(intervals),
        "time_intervals_valid": time_valid,
        "suspicious_text_encoding": any(marker in response for marker in SUSPICIOUS_TEXT_MARKERS),
        "hard_fail_reasons": hard_fail_reasons,
    }


def stratified_sample(records: Sequence[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if size <= 0 or size >= len(records):
        return list(records)
    rng = random.Random(seed)
    labels = sorted({str(row["gt_label"]) for row in records})
    base = size // len(labels)
    targets = {label: base for label in labels}
    for label in labels[: size % len(labels)]:
        targets[label] += 1
    selected: list[dict[str, Any]] = []
    for label in labels:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            if row["gt_label"] == label:
                groups[str(row["source_bucket"])].append(row)
        for values in groups.values():
            rng.shuffle(values)
        keys = sorted(groups)
        rng.shuffle(keys)
        cursor = 0
        while len([row for row in selected if row["gt_label"] == label]) < targets[label]:
            made_progress = False
            for key in keys:
                if cursor < len(groups[key]):
                    selected.append(groups[key][cursor])
                    made_progress = True
                    if len([row for row in selected if row["gt_label"] == label]) >= targets[label]:
                        break
            if not made_progress:
                break
            cursor += 1
    rng.shuffle(selected)
    return selected


def deranged_frame_sources(records: Sequence[dict[str, Any]], seed: int) -> dict[str, dict[str, Any]]:
    rng = random.Random(seed)
    output: dict[str, dict[str, Any]] = {}
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_label[str(row["gt_label"])].append(row)
    for values in by_label.values():
        ordered = list(values)
        rng.shuffle(ordered)
        if len(ordered) < 2:
            continue
        shift = rng.randrange(1, len(ordered))
        for index, row in enumerate(ordered):
            output[str(row["sample_id"])] = ordered[(index + shift) % len(ordered)]
    return output


def _format_number(value: float) -> str:
    rounded = round(value, 2)
    if abs(rounded - round(rounded)) < 1e-8:
        return str(int(round(rounded)))
    return f"{rounded:.2f}"


def shift_bboxes(response: str) -> tuple[str, bool]:
    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        x1, y1, x2, y2 = (float(value) for value in match.groups())
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            return match.group(0)
        if width >= 950 and height >= 950:
            replacement = (0.0, 0.0, 500.0, 500.0)
        else:
            candidates = (
                (0.0, 0.0, width, height),
                (1000.0 - width, 0.0, 1000.0, height),
                (0.0, 1000.0 - height, width, 1000.0),
                (1000.0 - width, 1000.0 - height, 1000.0, 1000.0),
            )
            old_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            replacement = max(
                candidates,
                key=lambda box: ((box[0] + box[2]) / 2.0 - old_center[0]) ** 2
                + ((box[1] + box[3]) / 2.0 - old_center[1]) ** 2,
            )
        changed = changed or any(abs(a - b) > 1e-6 for a, b in zip((x1, y1, x2, y2), replacement))
        values = ", ".join(_format_number(value) for value in replacement)
        return f"<bbox>[{values}]</bbox>"

    return BBOX_RE.sub(replace, response), changed


def shift_time_intervals(response: str, times: Sequence[float]) -> tuple[str, bool]:
    if not times:
        return response, False
    low, high = min(times), max(times)
    span = high - low
    if span <= 0:
        return response, False
    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        start, end = (float(value) for value in match.groups())
        duration = max(0.0, end - start)
        if duration >= 0.8 * span:
            new_duration = max(span / 3.0, 0.01)
            midpoint = (start + end) / 2.0
            if midpoint <= (low + high) / 2.0:
                new_start, new_end = high - new_duration, high
            else:
                new_start, new_end = low, low + new_duration
        else:
            candidates = ((low, low + duration), (high - duration, high))
            old_midpoint = (start + end) / 2.0
            new_start, new_end = max(
                candidates,
                key=lambda interval: abs((interval[0] + interval[1]) / 2.0 - old_midpoint),
            )
        changed = changed or abs(start - new_start) > 1e-6 or abs(end - new_end) > 1e-6
        return f"<t>[{_format_number(new_start)}, {_format_number(new_end)}]</t>"

    return TIME_RE.sub(replace, response), changed


def swap_artifact_types(response: str) -> tuple[str, bool]:
    changed = False
    offset = len(ARTIFACT_CATEGORIES) // 2
    replacement_map = {
        value: ARTIFACT_CATEGORIES[(index + offset) % len(ARTIFACT_CATEGORIES)]
        for index, value in enumerate(ARTIFACT_CATEGORIES)
    }

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        value = match.group(1).strip()
        replacement = replacement_map.get(value)
        if replacement is None:
            return match.group(0)
        changed = True
        return f"<type>{replacement}</type>"

    return TYPE_RE.sub(replace, response), changed


def judge_prompt(gt_label: str, candidate_response: str) -> str:
    return f"""You are provided with ordered video frames, a trusted ground-truth authenticity class (real or fake), and a candidate training annotation generated by another model. The annotation contains a final prediction and a rationale.

Evaluate the annotation itself. Do not re-label the sample. Judge whether its explanation is accurate, relevant, complete, and grounded in the ordered visual evidence.

For every claimed artifact, verify all of the following when present:
1. The visual phenomenon is actually visible in the frames.
2. The V4+ artifact category is semantically appropriate.
3. The <t>[start, end]</t> interval matches the frames where the phenomenon is visible.
4. The <bbox>[x1, y1, x2, y2]</bbox> region covers the claimed evidence. Coordinates are normalized independently within each frame to 0-1000.
5. The prose does not hallucinate entities, events, text, or visual defects.

For a real sample, check whether the stated stable or cleared regions are genuinely supported and whether the rationale avoids inventing artifacts. For a fake sample, check whether the claimed artifacts genuinely support the trusted fake label.

Provide a brief rationale inside <reasoning></reasoning>, followed by one holistic integer score inside <score></score>.

Rating guidelines:
- 5: Fully accurate, complete, and visually grounded; category, time, and region claims are precise.
- 4: Mostly accurate and grounded, with only minor imprecision or omission.
- 3: Partially correct, but contains noticeable generic wording, weak grounding, or incomplete category/time/region support.
- 2: Poor alignment; serious visual, category, temporal, or spatial errors, or substantial hallucination.
- 1: Unrelated, contradicted by the frames, or fundamentally incorrect.

Ground Truth Label: {gt_label}
Candidate response:
{candidate_response}

Output exactly:
<reasoning>{{brief evaluation}}</reasoning>
<score>{{integer from 1 to 5}}</score>"""


def input_messages(times: Sequence[float], prompt: str) -> list[dict[str, str]]:
    frame_lines = [f"[T={time:.2f}s] <image>" for time in times]
    content = "Ordered video frames:\n" + "\n".join(frame_lines) + "\n\n" + prompt
    return [{"role": "user", "content": content}]


def make_judge_row(
    base: Mapping[str, Any], variant: str, images: Sequence[str], candidate_response: str, control: Mapping[str, Any]
) -> dict[str, Any]:
    times = list(base["frame_times"])
    gt_label = str(base["gt_label"])
    return {
        "judge_id": f"{base['sample_id']}::{variant}",
        "sample_id": base["sample_id"],
        "variant": variant,
        "messages": input_messages(times, judge_prompt(gt_label, candidate_response)),
        "images": list(images),
        "metadata": {
            "gt_label": gt_label,
            "source_bucket": base["source_bucket"],
            "primary_artifact_type": base["primary_artifact_type"],
            "source_row_index": base["source_row_index"],
            "static_audit": base["static_audit"],
            "control": dict(control),
        },
    }


def build_records(
    source_rows: Sequence[dict[str, Any]], mode: str, sample_size: int, seed: int, check_images: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    invalid_reasons: Counter[str] = Counter()
    for index, row in enumerate(source_rows):
        images = image_paths(row)
        response = assistant_response(row)
        try:
            gt_label = derive_ground_truth(images)
        except ValueError:
            invalid_reasons["missing_or_ambiguous_gt_path"] += 1
            continue
        if not response:
            invalid_reasons["missing_assistant_response"] += 1
            continue
        if not images:
            invalid_reasons["no_images"] += 1
            continue
        if check_images and any(not Path(path).is_file() for path in images):
            invalid_reasons["missing_image_file"] += 1
            continue
        times = frame_times(row, len(images))
        audit = static_audit(response, gt_label, times)
        types = list(audit["artifact_types"])
        prepared.append(
            {
                "sample_id": sample_id(index, images),
                "source_row_index": index,
                "gt_label": gt_label,
                "source_bucket": source_bucket(images, gt_label),
                "primary_artifact_type": types[0] if types else "none",
                "images": images,
                "frame_times": times,
                "candidate_response": response,
                "static_audit": audit,
            }
        )

    selected = stratified_sample(prepared, sample_size, seed) if mode == "gate" else list(prepared)
    frame_controls = deranged_frame_sources(selected, seed + 1) if mode == "gate" else {}
    output: list[dict[str, Any]] = []
    variant_counts: Counter[str] = Counter()
    for base in selected:
        response = str(base["candidate_response"])
        images = list(base["images"])
        output.append(make_judge_row(base, "original", images, response, {}))
        variant_counts["original"] += 1
        if mode != "gate":
            continue

        frame_source = frame_controls.get(str(base["sample_id"]))
        if frame_source is not None:
            output.append(
                make_judge_row(
                    base,
                    "shuffled_frames",
                    frame_source["images"],
                    response,
                    {"frame_source_sample_id": frame_source["sample_id"]},
                )
            )
            variant_counts["shuffled_frames"] += 1

        shifted_bbox, bbox_changed = shift_bboxes(response)
        if bbox_changed:
            output.append(make_judge_row(base, "shifted_bbox", images, shifted_bbox, {}))
            variant_counts["shifted_bbox"] += 1

        shifted_time, time_changed = shift_time_intervals(response, base["frame_times"])
        if time_changed:
            output.append(make_judge_row(base, "shifted_time", images, shifted_time, {}))
            variant_counts["shifted_time"] += 1

        swapped_type, type_changed = swap_artifact_types(response)
        if type_changed:
            output.append(make_judge_row(base, "swapped_type", images, swapped_type, {}))
            variant_counts["swapped_type"] += 1

    summary = {
        "task": "DataB DeepfakeJudge pointwise input build",
        "mode": mode,
        "seed": seed,
        "source_records": len(source_rows),
        "eligible_records": len(prepared),
        "selected_original_records": len(selected),
        "judge_records": len(output),
        "variant_counts": dict(sorted(variant_counts.items())),
        "selected_label_counts": dict(sorted(Counter(row["gt_label"] for row in selected).items())),
        "selected_source_counts": dict(sorted(Counter(row["source_bucket"] for row in selected).items())),
        "eligible_image_count_distribution": dict(
            sorted(Counter(len(row["images"]) for row in prepared).items())
        ),
        "selected_primary_type_counts": dict(
            sorted(Counter(row["primary_artifact_type"] for row in selected).items())
        ),
        "static_hard_fail_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in selected
                    for reason in row["static_audit"]["hard_fail_reasons"]
                ).items()
            )
        ),
        "suspicious_text_encoding_count": sum(
            bool(row["static_audit"]["suspicious_text_encoding"]) for row in selected
        ),
        "invalid_source_record_reasons": dict(sorted(invalid_reasons.items())),
        "check_images": check_images,
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DataB inputs for a DeepfakeJudge reliability gate.")
    parser.add_argument("--datab-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--mode", choices=("gate", "full"), default="gate")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--check-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = read_json(args.datab_json)
    records, summary = build_records(source_rows, args.mode, args.sample_size, args.seed, args.check_images)
    summary["datab_json"] = str(Path(args.datab_json))
    summary["output_jsonl"] = str(Path(args.output_jsonl))
    write_jsonl(args.output_jsonl, records)
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
