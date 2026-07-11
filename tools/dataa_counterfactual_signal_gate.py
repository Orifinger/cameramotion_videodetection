#!/usr/bin/env python3
"""Gate 0: verify that DataA pairs contain localized edit signal.

For each source-matched Real/Fake pair, this script compares RGB differences
inside and outside the real VACE generation mask.  A bbox fallback is available
for diagnostics, but a run using fallback regions is never formal-gate eligible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


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


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def finite(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def percentile(values: Sequence[float], q: float) -> float:
    clean = sorted(finite(values))
    if not clean:
        return 0.0
    position = max(0.0, min(1.0, q)) * (len(clean) - 1)
    low, high = int(math.floor(position)), int(math.ceil(position))
    if low == high:
        return clean[low]
    weight = position - low
    return clean[low] * (1.0 - weight) + clean[high] * weight


def load_masks(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "masks" not in archive:
            raise ValueError("mask NPZ has no masks array")
        masks = archive["masks"]
    if masks.ndim != 3 or masks.shape[0] == 0:
        raise ValueError(f"mask array must be non-empty [N,H,W], got {masks.shape}")
    return (masks > 0).astype(np.uint8)


def bbox_mask(box: Sequence[float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in box]
    left = max(0, min(width, round(x1 * width / 1000.0)))
    top = max(0, min(height, round(y1 * height / 1000.0)))
    right = max(left + 1, min(width, round(x2 * width / 1000.0)))
    bottom = max(top + 1, min(height, round(y2 * height / 1000.0)))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = 1
    return mask


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    return (np.asarray(image.resize((width, height), Image.Resampling.NEAREST)) > 0).astype(np.uint8)


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def select_mask(masks: np.ndarray, frame_index: int, frame_count: int) -> np.ndarray:
    if frame_count <= 1 or masks.shape[0] <= 1:
        return masks[0]
    position = round(frame_index * (masks.shape[0] - 1) / (frame_count - 1))
    return masks[int(position)]


def evaluate_pair(row: Mapping[str, Any], allow_bbox_fallback: bool) -> dict[str, Any]:
    case_id = str(row.get("case_id", ""))
    real_images = list((row.get("real") or {}).get("images") or [])
    fake_images = list((row.get("fake") or {}).get("images") or [])
    if not real_images or len(real_images) != len(fake_images):
        return {"case_id": case_id, "ok": False, "failure": "real_fake_frame_count_mismatch"}

    mask_path = str(row.get("mask_npz") or "")
    use_true_mask = bool(mask_path and Path(mask_path).is_file())
    if use_true_mask:
        try:
            masks = load_masks(mask_path)
        except Exception as exc:  # noqa: BLE001
            return {"case_id": case_id, "ok": False, "failure": f"invalid_mask:{type(exc).__name__}:{exc}"}
        region_source = "vace_mask"
    elif allow_bbox_fallback:
        masks = None
        region_source = "bbox_fallback"
    else:
        return {"case_id": case_id, "ok": False, "failure": "missing_true_mask"}

    inside_values, outside_values, outside_identity, mask_areas = [], [], [], []
    valid_frames = 0
    for frame_index, (real_path, fake_path) in enumerate(zip(real_images, fake_images)):
        try:
            real = load_rgb(real_path)
            fake = load_rgb(fake_path)
        except Exception as exc:  # noqa: BLE001
            return {"case_id": case_id, "ok": False, "failure": f"image_read:{type(exc).__name__}:{exc}"}
        if real.shape != fake.shape:
            return {"case_id": case_id, "ok": False, "failure": f"image_shape_mismatch:{real.shape}:{fake.shape}"}
        height, width = real.shape[:2]
        if masks is not None:
            mask = select_mask(masks, frame_index, len(real_images))
            if mask.shape != (height, width):
                mask = resize_mask(mask, width, height)
        else:
            mask = bbox_mask(row.get("bbox_1000") or [], width, height)
        inside = mask > 0
        outside = ~inside
        if not inside.any() or not outside.any():
            continue
        diff = np.abs(fake - real).mean(axis=2)
        inside_values.append(float(diff[inside].mean()))
        outside_values.append(float(diff[outside].mean()))
        outside_identity.append(float((diff[outside] <= (2.0 / 255.0)).mean()))
        mask_areas.append(float(inside.mean()))
        valid_frames += 1

    if not valid_frames:
        return {"case_id": case_id, "ok": False, "failure": "no_valid_masked_frames"}
    inside_mean = float(np.mean(inside_values))
    outside_mean = float(np.mean(outside_values))
    return {
        "case_id": case_id,
        "dataset_split": row.get("dataset_split", ""),
        "motion_bucket": row.get("motion_bucket", "unknown"),
        "artifact_type": row.get("artifact_type", ""),
        "region_source": region_source,
        "camera_pair_consistent": bool(row.get("camera_pair_consistent")),
        "ok": True,
        "failure": "",
        "num_valid_frames": valid_frames,
        "inside_mean_abs_diff": inside_mean,
        "outside_mean_abs_diff": outside_mean,
        "inside_outside_ratio": safe_div(inside_mean, max(outside_mean, 1e-8)),
        "outside_near_identical_rate": float(np.mean(outside_identity)),
        "mean_mask_area_ratio": float(np.mean(mask_areas)),
    }


def aggregate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [item for item in items if item.get("ok")]
    ratios = [float(item["inside_outside_ratio"]) for item in valid]
    inside = [float(item["inside_mean_abs_diff"]) for item in valid]
    outside = [float(item["outside_mean_abs_diff"]) for item in valid]
    identity = [float(item["outside_near_identical_rate"]) for item in valid]
    return {
        "num_pairs": len(items),
        "num_valid_pairs": len(valid),
        "valid_pair_rate": safe_div(len(valid), len(items)),
        "true_mask_pair_rate": safe_div(
            sum(item.get("region_source") == "vace_mask" for item in valid), len(items)
        ),
        "camera_pair_consistency_rate": safe_div(
            sum(bool(item.get("camera_pair_consistent")) for item in valid), len(valid)
        ),
        "median_inside_mean_abs_diff": median(inside) if inside else 0.0,
        "median_outside_mean_abs_diff": median(outside) if outside else 0.0,
        "median_inside_outside_ratio": median(ratios) if ratios else 0.0,
        "p10_inside_outside_ratio": percentile(ratios, 0.10),
        "pair_rate_inside_gt_outside": safe_div(
            sum(float(item["inside_mean_abs_diff"]) > float(item["outside_mean_abs_diff"]) for item in valid),
            len(valid),
        ),
        "median_outside_near_identical_rate": median(identity) if identity else 0.0,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "case_id", "dataset_split", "motion_bucket", "artifact_type", "region_source",
        "camera_pair_consistent", "ok", "failure", "num_valid_frames",
        "inside_mean_abs_diff", "outside_mean_abs_diff", "inside_outside_ratio",
        "outside_near_identical_rate", "mean_mask_area_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--allow-bbox-fallback", action="store_true")
    parser.add_argument("--min-valid-pair-rate", type=float, default=0.90)
    parser.add_argument("--min-true-mask-rate", type=float, default=0.90)
    parser.add_argument("--min-camera-consistency", type=float, default=0.98)
    parser.add_argument("--min-median-ratio", type=float, default=2.0)
    parser.add_argument("--max-median-outside-diff", type=float, default=0.03)
    parser.add_argument("--min-inside-gt-outside-rate", type=float, default=0.70)
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.pair_manifest_jsonl)
    if args.split != "all":
        rows = [row for row in rows if row.get("dataset_split") == args.split]
    if args.max_pairs > 0:
        rows = rows[: args.max_pairs]
    if not rows:
        raise ValueError("no pair records selected")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(evaluate_pair, row, args.allow_bbox_fallback): row for row in rows}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: str(item.get("case_id", "")))
    overall = aggregate(results)
    failures = Counter(str(item.get("failure", "")) for item in results if not item.get("ok"))
    checks = {
        "valid_pair_rate": overall["valid_pair_rate"] >= args.min_valid_pair_rate,
        "true_mask_pair_rate": overall["true_mask_pair_rate"] >= args.min_true_mask_rate,
        "camera_pair_consistency_rate": overall["camera_pair_consistency_rate"] >= args.min_camera_consistency,
        "median_inside_outside_ratio": overall["median_inside_outside_ratio"] >= args.min_median_ratio,
        "median_outside_mean_abs_diff": overall["median_outside_mean_abs_diff"] <= args.max_median_outside_diff,
        "pair_rate_inside_gt_outside": overall["pair_rate_inside_gt_outside"] >= args.min_inside_gt_outside_rate,
    }
    formal_eligible = overall["true_mask_pair_rate"] >= args.min_true_mask_rate
    passed = formal_eligible and all(checks.values())
    summary = {
        "gate": "Gate 0 - localized counterfactual signal",
        "pair_manifest_jsonl": args.pair_manifest_jsonl,
        "formal_gate_eligible": formal_eligible,
        "status": "passed" if passed else "failed" if formal_eligible else "diagnostic_only",
        "thresholds": {
            "min_valid_pair_rate": args.min_valid_pair_rate,
            "min_true_mask_rate": args.min_true_mask_rate,
            "min_camera_consistency": args.min_camera_consistency,
            "min_median_ratio": args.min_median_ratio,
            "max_median_outside_diff": args.max_median_outside_diff,
            "min_inside_gt_outside_rate": args.min_inside_gt_outside_rate,
        },
        "checks": checks,
        "overall": overall,
        "failure_reasons": dict(failures),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "dataa_counterfactual_signal_gate_summary.json"
    items_path = out_dir / "dataa_counterfactual_signal_gate_items.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(items_path, results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved items: {items_path}")
    if args.fail_on_gate and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
