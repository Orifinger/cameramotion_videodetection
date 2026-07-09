#!/usr/bin/env python3
"""Probe whether camera-compensated local residuals separate DataA real/fake pairs.

This is a training-free gate for the camera-conditioned AIGC detection route.
It uses the existing DataA detection JSON where each case has a real/fake pair
with the same <t> and <bbox>. For each video, it estimates a global homography
between adjacent frames, warps the previous frame to the current frame, and
scores the residual inside the target bbox. If fake residuals are consistently
higher than real residuals, camera-compensated local evidence is a plausible
learning signal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CASE_RE = re.compile(r"(dataA_v1_(?:dataset_v2_|textedit_reserve_)?\d+)/(real|fake)")
ANSWER_RE = re.compile(r"<answer>\s*(Fake|Real)\s*</answer>", re.IGNORECASE)
BBOX_RE = re.compile(r"<bbox>\s*\[([^\]]+)\]\s*</bbox>", re.IGNORECASE)
TIME_RE = re.compile(r"<t>\s*\[([^\]]+)\]\s*</t>", re.IGNORECASE)
PROMPT_TS_RE = re.compile(r"\[T=([0-9.]+)s\]")


@dataclass
class Sample:
    case_id: str
    split: str
    images: list[str]
    timestamps: list[float]
    answer: str
    bbox_1000: tuple[float, float, float, float]
    time_window: tuple[float, float]
    camera_labels: list[str]
    camera_caption: str


@dataclass
class VideoScore:
    ok: bool
    score: float | None
    roi_mean: float | None
    global_mean: float | None
    frames_used: int
    reason: str = ""


def parse_float_list(raw: str, expected: int) -> tuple[float, ...]:
    vals = [float(x.strip()) for x in raw.split(",")]
    if len(vals) != expected:
        raise ValueError(f"expected {expected} values, got {len(vals)} from {raw!r}")
    return tuple(vals)


def parse_case_from_path(path: str) -> tuple[str | None, str | None]:
    m = CASE_RE.search(path.replace("\\", "/"))
    if not m:
        return None, None
    return m.group(1), m.group(2)


def map_path(path: str, old_prefix: str | None, new_prefix: str | None) -> str:
    if old_prefix and new_prefix and path.startswith(old_prefix):
        return new_prefix + path[len(old_prefix) :]
    return path


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


def load_camera_labels(path: str | None) -> dict[tuple[str, str], dict[str, Any]]:
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


def load_samples(
    detection_json: str,
    camera_jsonl: str | None,
    old_prefix: str | None,
    new_prefix: str | None,
) -> dict[str, dict[str, Sample]]:
    camera = load_camera_labels(camera_jsonl)
    with open(detection_json, encoding="utf-8") as f:
        data = json.load(f)

    pairs: dict[str, dict[str, Sample]] = {}
    for item in data:
        images = [map_path(str(p), old_prefix, new_prefix) for p in item.get("images", [])]
        if not images:
            continue
        case_id, split = parse_case_from_path(images[0])
        if not case_id or not split:
            continue

        messages = item.get("messages", [])
        assistant = last_assistant(messages)
        user = first_message(messages, "user")

        answer_m = ANSWER_RE.search(assistant)
        bbox_m = BBOX_RE.search(assistant)
        time_m = TIME_RE.search(assistant)
        if not answer_m or not bbox_m or not time_m:
            continue

        try:
            bbox = parse_float_list(bbox_m.group(1), 4)
            t0, t1 = parse_float_list(time_m.group(1), 2)
        except ValueError:
            continue

        timestamps = [float(x) for x in PROMPT_TS_RE.findall(user)]
        if len(timestamps) != len(images):
            timestamps = [float(i) for i in range(len(images))]

        cam = camera.get((case_id, split), {})
        sample = Sample(
            case_id=case_id,
            split=split,
            images=images,
            timestamps=timestamps,
            answer=answer_m.group(1).title(),
            bbox_1000=(bbox[0], bbox[1], bbox[2], bbox[3]),
            time_window=(float(t0), float(t1)),
            camera_labels=list(cam.get("labels", [])),
            camera_caption=str(cam.get("caption", "")),
        )
        pairs.setdefault(case_id, {})[split] = sample
    return pairs


def selected_frame_indices(timestamps: list[float], window: tuple[float, float]) -> list[int]:
    start, end = window
    idx = [i for i, t in enumerate(timestamps) if start <= t <= end]
    if len(idx) >= 2:
        return idx
    # If the GT interval is too narrow for the sampled timestamps, use the
    # nearest two frames so the probe still tests the annotated region.
    center = (start + end) / 2.0
    order = sorted(range(len(timestamps)), key=lambda i: abs(timestamps[i] - center))
    return sorted(order[: min(2, len(order))])


def bbox_to_pixels(
    bbox_1000: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_1000
    x1 = max(0, min(width - 1, round(x1 / 1000.0 * width)))
    x2 = max(1, min(width, round(x2 / 1000.0 * width)))
    y1 = max(0, min(height - 1, round(y1 / 1000.0 * height)))
    y2 = max(1, min(height, round(y2 / 1000.0 * height)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def make_feature_mask(
    height: int,
    width: int,
    bbox_1000: tuple[float, float, float, float],
    scale: float,
    pad_ratio: float,
):
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    small_h = round(height * scale)
    small_w = round(width * scale)
    mask = np.full((small_h, small_w), 255, dtype=np.uint8)
    x1, y1, x2, y2 = bbox_to_pixels(bbox_1000, width, height)
    pad_x = round((x2 - x1) * pad_ratio)
    pad_y = round((y2 - y1) * pad_ratio)
    x1 = max(0, round((x1 - pad_x) * scale))
    x2 = min(small_w, round((x2 + pad_x) * scale))
    y1 = max(0, round((y1 - pad_y) * scale))
    y2 = min(small_h, round((y2 + pad_y) * scale))
    if x2 > x1 and y2 > y1:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, thickness=-1)
    return mask


def score_video(
    sample: Sample,
    min_matches: int,
    max_dim: int,
    mask_bbox_for_homography: bool,
    mask_pad_ratio: float,
) -> VideoScore:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        return VideoScore(False, None, None, None, 0, f"missing dependency: {exc}")

    idx = selected_frame_indices(sample.timestamps, sample.time_window)
    if len(idx) < 2:
        return VideoScore(False, None, None, None, 0, "not enough selected frames")

    orb = cv2.ORB_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    roi_scores: list[float] = []
    global_scores: list[float] = []
    ratio_scores: list[float] = []

    for prev_i, cur_i in zip(idx, idx[1:]):
        prev = cv2.imread(sample.images[prev_i], cv2.IMREAD_COLOR)
        cur = cv2.imread(sample.images[cur_i], cv2.IMREAD_COLOR)
        if prev is None or cur is None:
            return VideoScore(False, None, None, None, 0, "missing frame")
        if prev.shape[:2] != cur.shape[:2]:
            cur = cv2.resize(cur, (prev.shape[1], prev.shape[0]))

        h, w = cur.shape[:2]
        scale = min(1.0, max_dim / float(max(h, w)))
        if scale < 1.0:
            prev_small = cv2.resize(prev, (round(w * scale), round(h * scale)))
            cur_small = cv2.resize(cur, (round(w * scale), round(h * scale)))
        else:
            prev_small = prev
            cur_small = cur

        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(cur_small, cv2.COLOR_BGR2GRAY)
        feature_mask = (
            make_feature_mask(h, w, sample.bbox_1000, scale, mask_pad_ratio)
            if mask_bbox_for_homography
            else None
        )
        kp1, des1 = orb.detectAndCompute(prev_gray, feature_mask)
        kp2, des2 = orb.detectAndCompute(cur_gray, feature_mask)
        if des1 is None or des2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
            continue

        matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)
        matches = matches[: min(len(matches), 400)]
        if len(matches) < min_matches:
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
        if H is None or inliers is None or int(inliers.sum()) < min_matches:
            continue

        if scale < 1.0:
            S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
            H = np.linalg.inv(S) @ H @ S

        warped = cv2.warpPerspective(prev, H, (w, h))
        diff = cv2.absdiff(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY))
        x1, y1, x2, y2 = bbox_to_pixels(sample.bbox_1000, w, h)
        roi = diff[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        roi_mean = float(roi.mean())
        global_mean = float(diff.mean())
        roi_scores.append(roi_mean)
        global_scores.append(global_mean)
        ratio_scores.append(roi_mean / (global_mean + 1e-6))

    if not ratio_scores:
        return VideoScore(False, None, None, None, 0, "no valid homography frame pairs")
    return VideoScore(
        True,
        float(statistics.mean(ratio_scores)),
        float(statistics.mean(roi_scores)),
        float(statistics.mean(global_scores)),
        len(ratio_scores) + 1,
    )


def auc_from_scores(pos: Iterable[float], neg: Iterable[float]) -> float | None:
    values = [(x, 1) for x in pos] + [(x, 0) for x in neg]
    if not values or not any(y for _, y in values) or all(y for _, y in values):
        return None
    values.sort(key=lambda p: p[0])
    rank_sum = 0.0
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[j][0] == values[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(y for _, y in values[i:j])
        i = j
    n_pos = sum(y for _, y in values)
    n_neg = len(values) - n_pos
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def bucket(labels: list[str]) -> str:
    label_set = set(labels)
    if "complex-motion" in label_set:
        return "complex-motion"
    if "minor-motion" in label_set:
        return "minor-motion"
    if "no-motion" in label_set or "static" in label_set:
        return "static"
    return "unknown"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in rows if r["ok"]]
    failed_rows = [r for r in rows if not r["ok"]]
    fake_scores = [float(r["fake_score"]) for r in ok_rows]
    real_scores = [float(r["real_score"]) for r in ok_rows]
    diffs = [f - r for f, r in zip(fake_scores, real_scores)]
    summary: dict[str, Any] = {
        "total_pairs": len(rows),
        "ok_pairs": len(ok_rows),
        "failed_pairs": len(failed_rows),
    }
    if failed_rows:
        reasons: dict[str, int] = {}
        for row in failed_rows:
            reason = str(row.get("reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
        summary["failure_reasons"] = dict(sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])))
    if ok_rows:
        summary.update(
            {
                "pair_accuracy_fake_gt_real": sum(d > 0 for d in diffs) / len(diffs),
                "auc_fake_vs_real": auc_from_scores(fake_scores, real_scores),
                "mean_fake_score": statistics.mean(fake_scores),
                "mean_real_score": statistics.mean(real_scores),
                "mean_diff": statistics.mean(diffs),
                "median_diff": statistics.median(diffs),
            }
        )
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in ok_rows:
        by_bucket.setdefault(str(row["motion_bucket"]), []).append(row)
    bucket_summary = {}
    for name, items in by_bucket.items():
        fs = [float(x["fake_score"]) for x in items]
        rs = [float(x["real_score"]) for x in items]
        ds = [f - r for f, r in zip(fs, rs)]
        bucket_summary[name] = {
            "n": len(items),
            "pair_accuracy_fake_gt_real": sum(d > 0 for d in ds) / len(ds),
            "auc_fake_vs_real": auc_from_scores(fs, rs),
            "mean_diff": statistics.mean(ds),
        }
    summary["by_motion_bucket"] = bucket_summary
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-json", required=True)
    parser.add_argument("--camera-jsonl")
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--old-prefix")
    parser.add_argument("--new-prefix")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--min-matches", type=int, default=12)
    parser.add_argument("--max-dim", type=int, default=640)
    parser.add_argument("--mask-pad-ratio", type=float, default=0.10)
    parser.add_argument(
        "--include-bbox-in-homography",
        action="store_true",
        help="Use target bbox features when estimating global homography. Default masks them out.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only parse pair structure; do not read images.")
    args = parser.parse_args()

    pairs = load_samples(args.detection_json, args.camera_jsonl, args.old_prefix, args.new_prefix)
    complete = [(k, v["real"], v["fake"]) for k, v in sorted(pairs.items()) if "real" in v and "fake" in v]
    if args.max_pairs and args.max_pairs > 0:
        complete = complete[: args.max_pairs]

    rows: list[dict[str, Any]] = []
    if args.dry_run:
        for case_id, real, fake in complete:
            rows.append(
                {
                    "case_id": case_id,
                    "ok": True,
                    "motion_bucket": bucket(real.camera_labels or fake.camera_labels),
                    "camera_labels": ";".join(real.camera_labels or fake.camera_labels),
                    "real_answer": real.answer,
                    "fake_answer": fake.answer,
                    "bbox": list(real.bbox_1000),
                    "time_window": list(real.time_window),
                    "real_score": math.nan,
                    "fake_score": math.nan,
                    "diff": math.nan,
                    "reason": "dry-run",
                }
            )
    else:
        for case_id, real, fake in complete:
            real_score = score_video(
                real,
                args.min_matches,
                args.max_dim,
                not args.include_bbox_in_homography,
                args.mask_pad_ratio,
            )
            fake_score = score_video(
                fake,
                args.min_matches,
                args.max_dim,
                not args.include_bbox_in_homography,
                args.mask_pad_ratio,
            )
            ok = real_score.ok and fake_score.ok
            row = {
                "case_id": case_id,
                "ok": ok,
                "motion_bucket": bucket(real.camera_labels or fake.camera_labels),
                "camera_labels": ";".join(real.camera_labels or fake.camera_labels),
                "real_answer": real.answer,
                "fake_answer": fake.answer,
                "bbox": list(real.bbox_1000),
                "time_window": list(real.time_window),
                "real_score": real_score.score,
                "fake_score": fake_score.score,
                "real_roi_mean": real_score.roi_mean,
                "fake_roi_mean": fake_score.roi_mean,
                "real_global_mean": real_score.global_mean,
                "fake_global_mean": fake_score.global_mean,
                "real_frames_used": real_score.frames_used,
                "fake_frames_used": fake_score.frames_used,
                "diff": (fake_score.score - real_score.score) if ok else None,
                "reason": "" if ok else f"real={real_score.reason}; fake={fake_score.reason}",
            }
            rows.append(row)

    summary = {
        "detection_json": args.detection_json,
        "camera_jsonl": args.camera_jsonl,
        "dry_run": args.dry_run,
        "mask_bbox_for_homography": not args.include_bbox_in_homography,
        "mask_pad_ratio": args.mask_pad_ratio,
        "complete_pairs_loaded": len(complete),
        "metrics": summarize(rows) if not args.dry_run else {"total_pairs": len(rows)},
    }

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    fieldnames = sorted({key for row in rows for key in row})
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
