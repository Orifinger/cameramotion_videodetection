#!/usr/bin/env python3
"""Audit CTNE transition archives and build a reusable feature index."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_ctne_gate1.contracts import (
    FEATURE_SCHEMA_VERSION,
    dataset_slug,
    feature_filename,
    normalize_path,
    read_jsonl,
    validate_feature_archive,
    write_json,
    write_jsonl,
)


def _paths(feature_root: Path, row: dict[str, Any]) -> tuple[Path, Path]:
    slug = dataset_slug(row.get("dataset_slug") or row.get("dataset_name"))
    stem = Path(feature_filename(str(row["sample_id"]))).stem
    directory = feature_root / "features" / slug
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def continuous_motion_bucket(camera: np.ndarray) -> str:
    """Fixed, dataset-independent bucket used only for controls and reporting."""

    camera = np.asarray(camera, dtype=np.float32)
    translation = np.linalg.norm(camera[:, :2], axis=1)
    rotation = np.abs(camera[:, 2])
    scale = np.max(np.abs(camera[:, 3:5]), axis=1)
    median_translation = float(np.median(translation))
    median_rotation = float(np.median(rotation))
    median_scale = float(np.median(scale))
    if median_translation < 0.0015 and median_rotation < 0.002 and median_scale < 0.002:
        return "static/no-motion"
    if median_translation < 0.01 and median_rotation < 0.02 and median_scale < 0.02:
        return "minor-motion"
    return "complex-motion"


def audit(
    *,
    manifest_jsonl: Path,
    feature_root: Path,
    output_index_jsonl: Path,
    output_summary_json: Path,
    min_coverage: float,
) -> dict[str, Any]:
    manifest = read_jsonl(manifest_jsonl)
    eligible = [row for row in manifest if bool(row.get("ctne_available", int(row.get("frame_count", 0)) >= 3))]
    unavailable = [row for row in manifest if not bool(row.get("ctne_available", int(row.get("frame_count", 0)) >= 3))]
    valid: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    fit_models: Counter[str] = Counter()
    fallback_count = 0
    transition_count = 0
    dimensions: Counter[tuple[int, int]] = Counter()
    all_frame_contracts_exact = True
    for row in eligible:
        feature_path, metadata_path = _paths(feature_root, row)
        if not feature_path.is_file() or not metadata_path.is_file():
            missing.append(
                {
                    "sample_id": row["sample_id"],
                    "feature_exists": feature_path.is_file(),
                    "metadata_exists": metadata_path.is_file(),
                }
            )
            continue
        try:
            archive = validate_feature_archive(feature_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if archive["sample_id"] != str(row["sample_id"]) or archive["label"] != int(row["label"]):
                raise ValueError("archive identity or label does not match manifest")
            frame_contract = metadata.get("frame_contract") or {}
            selected_count = int(frame_contract.get("selected_frame_count", 0))
            if archive["transition_count"] != selected_count - 1:
                raise ValueError("transition count is not selected_frame_count - 1")
            max_frames = int(frame_contract.get("max_frames", -1))
            if max_frames == 0 and selected_count != int(row["frame_count"]):
                all_frame_contracts_exact = False
                raise ValueError("max_frames=0 archive did not use every listed frame")
            with np.load(feature_path, allow_pickle=False) as values:
                numeric_motion_bucket = continuous_motion_bucket(values["camera_context"])
            quality = metadata.get("quality") or {}
            fit_models.update({str(key): int(value) for key, value in (quality.get("camera_fit_model_counts") or {}).items()})
            fallback_count += int(quality.get("fit_exception_fallback_count", 0))
            transition_count += int(archive["transition_count"])
            dimensions[(archive["camera_dim"], archive["evidence_dim"])] += 1
            valid.append(
                {
                    **row,
                    "camera_label_motion_bucket": row.get("motion_bucket", "unknown"),
                    "motion_bucket": numeric_motion_bucket,
                    "motion_bucket_source": "fixed_thresholds_on_continuous_camera_context",
                    "feature_path": normalize_path(feature_path),
                    "feature_metadata_path": normalize_path(metadata_path),
                    "selected_frame_count": selected_count,
                    "transition_count": int(archive["transition_count"]),
                    "camera_dim": int(archive["camera_dim"]),
                    "evidence_dim": int(archive["evidence_dim"]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            invalid.append({"sample_id": row["sample_id"], "type": type(exc).__name__, "error": str(exc)})
    coverage = len(valid) / len(eligible) if eligible else 0.0
    checks = {
        "eligible_feature_coverage": coverage >= min_coverage,
        "single_feature_dimension_contract": len(dimensions) == 1,
        "formal_full_frame_contract": all_frame_contracts_exact,
        "no_invalid_archives": not invalid,
    }
    status = "passed" if all(checks.values()) else "failed"
    write_jsonl(output_index_jsonl, valid)
    summary = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "gate": "CTNE variable-length transition feature audit",
        "status": status,
        "manifest_jsonl": normalize_path(manifest_jsonl),
        "feature_root": normalize_path(feature_root),
        "output_index_jsonl": normalize_path(output_index_jsonl),
        "thresholds": {"min_eligible_coverage": min_coverage},
        "checks": checks,
        "overall": {
            "manifest_records": len(manifest),
            "eligible_records": len(eligible),
            "ctne_unavailable_records": len(unavailable),
            "valid_feature_records": len(valid),
            "coverage": coverage,
            "total_transitions": transition_count,
            "feature_dimensions": {f"camera={key[0]},evidence={key[1]}": value for key, value in dimensions.items()},
            "camera_fit_model_counts": dict(fit_models),
            "fit_exception_fallback_count": fallback_count,
        },
        "missing_count": len(missing),
        "invalid_count": len(invalid),
        "first_missing": missing[:50],
        "first_invalid": invalid[:50],
        "ctne_unavailable": [
            {"sample_id": row["sample_id"], "frame_count": int(row.get("frame_count", 0))}
            for row in unavailable[:100]
        ],
    }
    write_json(output_summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-index-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary-json", type=Path, required=True)
    parser.add_argument("--min-coverage", type=float, default=0.98)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = audit(
        manifest_jsonl=args.manifest_jsonl,
        feature_root=args.feature_root,
        output_index_jsonl=args.output_index_jsonl,
        output_summary_json=args.output_summary_json,
        min_coverage=args.min_coverage,
    )
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
