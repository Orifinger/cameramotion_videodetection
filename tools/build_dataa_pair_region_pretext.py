#!/usr/bin/env python3
"""Build DataA paired-region pretext data without long camera templates.

The generated tasks use DataA's strongest existing supervision:

  same case + same camera motion + same <t>/<bbox>
    real: normal target region
    fake: local edited/artifact target region

Outputs are compact XML tags rather than generated CoT. This is meant for a
Stage-2 pretext smoke test after the residual probe shows that camera
compensated local evidence is useful.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


CASE_RE = re.compile(r"(dataA_v1_(?:dataset_v2_|textedit_reserve_)?\d+)/(real|fake)")
ANSWER_RE = re.compile(r"<answer>\s*(Fake|Real)\s*</answer>", re.IGNORECASE)
BBOX_RE = re.compile(r"<bbox>\s*\[([^\]]+)\]\s*</bbox>", re.IGNORECASE)
TIME_RE = re.compile(r"<t>\s*\[([^\]]+)\]\s*</t>", re.IGNORECASE)
TYPE_RE = re.compile(r"<type>\s*([^<]+)\s*</type>", re.IGNORECASE)
PROMPT_TS_RE = re.compile(r"\[T=([0-9.]+)s\]")


SYSTEM_PROMPT = """You are a video region analyst. Infer camera motion from the frames and judge the specified target region. Output only the requested XML tags."""


def parse_case_from_path(path: str) -> tuple[str | None, str | None]:
    match = CASE_RE.search(path.replace("\\", "/"))
    if not match:
        return None, None
    return match.group(1), match.group(2)


def map_path(path: str, old_prefix: str | None, new_prefix: str | None) -> str:
    if old_prefix and new_prefix and path.startswith(old_prefix):
        return new_prefix + path[len(old_prefix) :]
    return path


def parse_values(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def first_message(messages: list[dict[str, Any]], role: str) -> str:
    for msg in messages:
        if msg.get("role") == role:
            return str(msg.get("content", ""))
    return ""


def last_assistant(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def load_case_filter(path: str | None) -> set[str] | None:
    if not path:
        return None
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return set()
    if raw[0] in "[{":
        data = json.loads(raw)
        if isinstance(data, dict):
            values = data.get("case_ids") or data.get("ids") or data.get("cases") or []
        else:
            values = data
        return {str(x) for x in values}
    return {line.strip() for line in raw.splitlines() if line.strip()}


def load_camera(path: str | None) -> dict[tuple[str, str], dict[str, Any]]:
    if not path:
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            case_id, split = parse_case_from_path(str(item.get("path", "")))
            if case_id and split:
                out[(case_id, split)] = item
    return out


def load_pairs(
    detection_json: str,
    camera_jsonl: str | None,
    old_prefix: str | None,
    new_prefix: str | None,
    allowed_cases: set[str] | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    camera = load_camera(camera_jsonl)
    data = json.loads(Path(detection_json).read_text(encoding="utf-8"))
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for item in data:
        images = [map_path(str(p), old_prefix, new_prefix) for p in item.get("images", [])]
        if not images:
            continue
        case_id, split = parse_case_from_path(images[0])
        if not case_id or not split:
            continue
        if allowed_cases is not None and case_id not in allowed_cases:
            continue

        messages = item.get("messages", [])
        assistant = last_assistant(messages)
        user = first_message(messages, "user")
        bbox = parse_values(BBOX_RE, assistant)
        time_window = parse_values(TIME_RE, assistant)
        answer = parse_values(ANSWER_RE, assistant)
        if not bbox or not time_window or not answer:
            continue

        timestamps = [float(x) for x in PROMPT_TS_RE.findall(user)]
        if len(timestamps) != len(images):
            timestamps = [float(i) for i in range(len(images))]
        cam = camera.get((case_id, split), {})
        pairs.setdefault(case_id, {})[split] = {
            "case_id": case_id,
            "split": split,
            "images": images,
            "timestamps": timestamps,
            "bbox": bbox,
            "time_window": time_window,
            "answer": answer.title(),
            "artifact_type": parse_values(TYPE_RE, assistant) or "none",
            "camera_labels": list(cam.get("labels", [])),
            "camera_caption": str(cam.get("caption", "")),
        }
    return {k: v for k, v in pairs.items() if "real" in v and "fake" in v}


def camera_block(sample: dict[str, Any]) -> str:
    labels = sample.get("camera_labels") or []
    label_text = "; ".join(str(x) for x in labels) if labels else "unknown"
    caption = str(sample.get("camera_caption") or "").strip()
    if caption:
        return f"<camera_motion>\n<labels>{label_text}</labels>\n<caption>{caption}</caption>\n</camera_motion>"
    return f"<camera_motion>\n<labels>{label_text}</labels>\n</camera_motion>"


def target_block(sample: dict[str, Any]) -> str:
    return f"<target_region>\n<t>[{sample['time_window']}]</t>\n<bbox>[{sample['bbox']}]</bbox>\n</target_region>"


def frame_lines(prefix: str, sample: dict[str, Any]) -> list[str]:
    lines = []
    for timestamp in sample["timestamps"]:
        lines.append(f"[{prefix} T={timestamp:.2f}s] <image>")
    return lines


def parse_time_window(raw: str) -> tuple[float, float] | None:
    try:
        start, end = [float(x.strip()) for x in raw.split(",")]
    except Exception:
        return None
    return start, end


def evenly_pick(indices: list[int], max_count: int) -> list[int]:
    if len(indices) <= max_count:
        return indices
    if max_count <= 1:
        return [indices[len(indices) // 2]]
    picked = []
    for i in range(max_count):
        pos = round(i * (len(indices) - 1) / (max_count - 1))
        picked.append(indices[pos])
    return sorted(set(picked))


def limit_frames(sample: dict[str, Any], max_frames: int) -> dict[str, Any]:
    if max_frames <= 0 or len(sample["images"]) <= max_frames:
        return sample
    timestamps = list(sample["timestamps"])
    window = parse_time_window(str(sample["time_window"]))
    if window:
        start, end = window
        preferred = [i for i, ts in enumerate(timestamps) if start <= ts <= end]
    else:
        preferred = []
    selected = evenly_pick(preferred, max_frames) if preferred else []
    if len(selected) < max_frames:
        remaining = [i for i in range(len(timestamps)) if i not in selected]
        selected = sorted(selected + evenly_pick(remaining, max_frames - len(selected)))
    selected = selected[:max_frames]
    new_sample = dict(sample)
    new_sample["images"] = [sample["images"][i] for i in selected]
    new_sample["timestamps"] = [timestamps[i] for i in selected]
    return new_sample


def make_region_record(sample: dict[str, Any], max_frames: int) -> dict[str, Any]:
    sample = limit_frames(sample, max_frames)
    status = "artifact" if sample["split"] == "fake" else "normal"
    artifact_type = sample["artifact_type"] if status == "artifact" else "none"
    user = "\n".join(
        [
            "Video frames:",
            *frame_lines("Video", sample),
            "",
            "Target region:",
            f"<t>[{sample['time_window']}]</t>",
            f"<bbox>[{sample['bbox']}]</bbox>",
            "",
            "Infer camera motion and judge whether the target region is normal or artifact-like.",
        ]
    )
    assistant = "\n".join(
        [
            camera_block(sample),
            target_block(sample),
            f"<region_status>{status}</region_status>",
            f"<artifact_type>{artifact_type}</artifact_type>",
        ]
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "images": sample["images"],
        "case_id": sample["case_id"],
        "pretext_task": "region_status",
        "source_split": sample["split"],
    }


def make_pair_record(
    case_id: str,
    real: dict[str, Any],
    fake: dict[str, Any],
    rng: random.Random,
    max_frames_per_video: int,
) -> dict[str, Any]:
    real = limit_frames(real, max_frames_per_video)
    fake = limit_frames(fake, max_frames_per_video)
    fake_is_a = rng.random() < 0.5
    video_a = fake if fake_is_a else real
    video_b = real if fake_is_a else fake
    edited_video = "A" if fake_is_a else "B"
    user = "\n".join(
        [
            "Video A frames:",
            *frame_lines("Video A", video_a),
            "",
            "Video B frames:",
            *frame_lines("Video B", video_b),
            "",
            "The two videos share the same scene, camera motion, and target region.",
            "Target region:",
            f"<t>[{fake['time_window']}]</t>",
            f"<bbox>[{fake['bbox']}]</bbox>",
            "",
            "Infer camera motion and select which video contains the local synthetic edit.",
        ]
    )
    assistant = "\n".join(
        [
            camera_block(fake),
            target_block(fake),
            f"<edited_video>{edited_video}</edited_video>",
            f"<artifact_type>{fake['artifact_type']}</artifact_type>",
        ]
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "images": video_a["images"] + video_b["images"],
        "case_id": case_id,
        "pretext_task": "pair_selection",
        "edited_video": edited_video,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-json", required=True)
    parser.add_argument("--camera-jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--task", choices=["region", "pair", "both"], default="both")
    parser.add_argument("--case-id-file")
    parser.add_argument("--old-prefix")
    parser.add_argument("--new-prefix")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--region-max-frames", type=int, default=0, help="0 keeps all frames for region records.")
    parser.add_argument("--pair-max-frames-per-video", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    allowed_cases = load_case_filter(args.case_id_file)
    pairs = load_pairs(args.detection_json, args.camera_jsonl, args.old_prefix, args.new_prefix, allowed_cases)
    case_items = sorted(pairs.items())
    if args.max_pairs and args.max_pairs > 0:
        case_items = case_items[: args.max_pairs]

    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    for case_id, pair in case_items:
        real = pair["real"]
        fake = pair["fake"]
        if args.task in {"region", "both"}:
            records.append(make_region_record(real, args.region_max_frames))
            records.append(make_region_record(fake, args.region_max_frames))
        if args.task in {"pair", "both"}:
            records.append(make_pair_record(case_id, real, fake, rng, args.pair_max_frames_per_video))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")

    summary = {
        "pairs_loaded": len(pairs),
        "pairs_exported": len(case_items),
        "records_exported": len(records),
        "task": args.task,
        "out": args.out,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
