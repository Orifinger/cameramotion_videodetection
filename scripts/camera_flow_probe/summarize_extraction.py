#!/usr/bin/env python3
"""Audit dense-flow feature extraction before fitting any detector probe."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_flow_probe.contracts import read_jsonl


def _finite_quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.1)),
        "p90": float(np.quantile(array, 0.9)),
    }


def summarize(
    manifest_jsonl: Path,
    feature_dir: Path,
    *,
    min_coverage: float,
    min_positive_mask_rate: float,
    min_aligned_valid_case_rate: float,
    min_camera_inlier_rate: float,
    max_pair_camera_error_normalized: float,
) -> dict[str, Any]:
    rows = read_jsonl(manifest_jsonl)
    metadata: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: list[dict[str, str]] = []
    non_positive_mask_cases: list[dict[str, Any]] = []
    positive_mask_cases = 0
    aligned_valid_cases = 0
    aligned_valid_fractions: list[float] = []
    valid_by_split: Counter[str] = Counter()
    positive_by_split: Counter[str] = Counter()
    inlier_rates: list[float] = []
    pair_errors: list[float] = []
    by_bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    source_counts: Counter[str] = Counter()
    for row in rows:
        case_id = str(row["case_id"])
        npz_path = feature_dir / f"{case_id}.npz"
        json_path = feature_dir / f"{case_id}.json"
        if not npz_path.is_file() or not json_path.is_file():
            missing.append(case_id)
            continue
        try:
            item = json.loads(json_path.read_text(encoding="utf-8"))
            with np.load(npz_path, allow_pickle=False) as archive:
                aligned = archive["fake_local_aligned"]
                unaligned = archive["fake_local_unaligned"]
                aligned_mask = archive["fake_mask_aligned"]
                unaligned_mask = archive["fake_mask_unaligned"]
                aligned_label = archive["fake_label_aligned"].astype(bool)
                unaligned_label = archive["fake_label_unaligned"].astype(bool)
                aligned_valid = archive["fake_valid_aligned"].astype(bool)
                unaligned_valid = archive["fake_valid_unaligned"].astype(bool)
                raw_positive_count = int(aligned_label.sum())
                raw_positive_unaligned_count = int(unaligned_label.sum())
                positive_count = int((aligned_label & aligned_valid).sum())
                positive_unaligned_count = int((unaligned_label & unaligned_valid).sum())
                max_aligned_mask_fraction = float(aligned_mask.max()) if aligned_mask.size else 0.0
                max_unaligned_mask_fraction = float(unaligned_mask.max()) if unaligned_mask.size else 0.0
                max_valid_aligned_mask_fraction = (
                    float(aligned_mask[aligned_valid].max()) if aligned_valid.any() else 0.0
                )
                max_valid_unaligned_mask_fraction = (
                    float(unaligned_mask[unaligned_valid].max()) if unaligned_valid.any() else 0.0
                )
                if not np.isfinite(aligned).all() or not np.isfinite(unaligned).all():
                    raise ValueError("local feature arrays contain non-finite values")
        except Exception as exc:  # noqa: BLE001
            invalid.append({"case_id": case_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        metadata.append(item)
        split = str(row.get("dataset_split", "unknown"))
        valid_by_split[split] += 1
        aligned_valid_cases += int(aligned_valid.any())
        aligned_valid_fractions.append(float(aligned_valid.mean()) if aligned_valid.size else 0.0)
        if positive_count > 0:
            positive_mask_cases += 1
            positive_by_split[split] += 1
        else:
            non_positive_mask_cases.append(
                {
                    "case_id": case_id,
                    "dataset_split": split,
                    "source_name": str(row.get("source_name", "unknown")),
                    "motion_bucket": str(row.get("motion_bucket", "unknown")),
                    "aligned_positive_patch_count": positive_count,
                    "unaligned_positive_patch_count": positive_unaligned_count,
                    "aligned_raw_positive_patch_count": raw_positive_count,
                    "unaligned_raw_positive_patch_count": raw_positive_unaligned_count,
                    "aligned_valid_patch_count": int(aligned_valid.sum()),
                    "unaligned_valid_patch_count": int(unaligned_valid.sum()),
                    "max_aligned_mask_fraction": max_aligned_mask_fraction,
                    "max_unaligned_mask_fraction": max_unaligned_mask_fraction,
                    "max_valid_aligned_mask_fraction": max_valid_aligned_mask_fraction,
                    "max_valid_unaligned_mask_fraction": max_valid_unaligned_mask_fraction,
                }
            )
        source_counts[str(row.get("source_name", ""))] += 1
        bucket = str(row.get("motion_bucket", "unknown"))
        real_inlier = float((item.get("real_quality") or {}).get("median_camera_inlier_rate", float("nan")))
        fake_inlier = float((item.get("fake_quality") or {}).get("median_camera_inlier_rate", float("nan")))
        pair_error = float(
            (item.get("paired_camera_consistency") or {}).get("median_corner_error_normalized", float("nan"))
        )
        inlier_rates.extend([real_inlier, fake_inlier])
        pair_errors.append(pair_error)
        by_bucket[bucket]["inlier"].extend([real_inlier, fake_inlier])
        by_bucket[bucket]["pair_error"].append(pair_error)

    valid_count = len(metadata)
    coverage = valid_count / len(rows) if rows else 0.0
    positive_rate = positive_mask_cases / valid_count if valid_count else 0.0
    aligned_valid_case_rate = aligned_valid_cases / valid_count if valid_count else 0.0
    inlier_summary = _finite_quantiles(inlier_rates)
    pair_summary = _finite_quantiles(pair_errors)
    checks = {
        "feature_coverage": coverage >= min_coverage,
        "positive_mask_case_rate": positive_rate >= min_positive_mask_rate,
        "aligned_valid_case_rate": aligned_valid_case_rate >= min_aligned_valid_case_rate,
        "median_camera_inlier_rate": inlier_summary["median"] >= min_camera_inlier_rate,
        "median_real_fake_camera_error": pair_summary["median"] <= max_pair_camera_error_normalized,
        "no_invalid_feature_files": not invalid,
    }
    return {
        "gate": "dense camera-flow extraction audit",
        "status": "passed" if all(checks.values()) else "failed",
        "manifest_jsonl": str(manifest_jsonl),
        "feature_dir": str(feature_dir),
        "thresholds": {
            "min_coverage": min_coverage,
            "min_positive_mask_rate": min_positive_mask_rate,
            "min_aligned_valid_case_rate": min_aligned_valid_case_rate,
            "min_camera_inlier_rate": min_camera_inlier_rate,
            "max_pair_camera_error_normalized": max_pair_camera_error_normalized,
        },
        "checks": checks,
        "overall": {
            "manifest_cases": len(rows),
            "valid_feature_cases": valid_count,
            "coverage": coverage,
            "positive_mask_cases": positive_mask_cases,
            "positive_mask_case_rate": positive_rate,
            "aligned_valid_cases": aligned_valid_cases,
            "aligned_valid_case_rate": aligned_valid_case_rate,
            "aligned_valid_patch_fraction": _finite_quantiles(aligned_valid_fractions),
            "camera_inlier_rate": inlier_summary,
            "paired_camera_corner_error_normalized": pair_summary,
            "source_counts": dict(source_counts),
        },
        "positive_mask_by_split": {
            split: {
                "valid_feature_cases": int(valid_by_split[split]),
                "positive_mask_cases": int(positive_by_split[split]),
                "positive_mask_case_rate": (
                    float(positive_by_split[split] / valid_by_split[split]) if valid_by_split[split] else 0.0
                ),
            }
            for split in sorted(valid_by_split)
        },
        "by_motion_bucket": {
            bucket: {
                "camera_inlier_rate": _finite_quantiles(values["inlier"]),
                "paired_camera_corner_error_normalized": _finite_quantiles(values["pair_error"]),
            }
            for bucket, values in sorted(by_bucket.items())
        },
        "missing_cases": missing,
        "invalid_cases": invalid,
        "non_positive_mask_cases": non_positive_mask_cases,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--min-positive-mask-rate", type=float, default=0.90)
    parser.add_argument("--min-aligned-valid-case-rate", type=float, default=0.95)
    parser.add_argument("--min-camera-inlier-rate", type=float, default=0.50)
    parser.add_argument("--max-pair-camera-error-normalized", type=float, default=0.02)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = summarize(
        args.manifest_jsonl,
        args.feature_dir,
        min_coverage=args.min_coverage,
        min_positive_mask_rate=args.min_positive_mask_rate,
        min_aligned_valid_case_rate=args.min_aligned_valid_case_rate,
        min_camera_inlier_rate=args.min_camera_inlier_rate,
        max_pair_camera_error_normalized=args.max_pair_camera_error_normalized,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
