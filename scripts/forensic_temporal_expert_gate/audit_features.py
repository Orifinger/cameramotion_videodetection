#!/usr/bin/env python3
"""Validate extracted DINO archives and create a compact feature index."""

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

from scripts.forensic_temporal_expert_gate import FEATURE_SCHEMA_VERSION
from scripts.forensic_temporal_expert_gate.contracts import (
    feature_filename,
    normalize_path,
    read_json_or_jsonl,
    write_json,
    write_jsonl,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-index-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary-json", type=Path, required=True)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_json_or_jsonl(args.manifest_jsonl)
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    hidden_sizes: Counter[int] = Counter()
    grid_counts: Counter[int] = Counter()
    frame_counts: Counter[int] = Counter()
    for row in rows:
        path = args.feature_root / "features" / feature_filename(str(row["sample_id"]))
        try:
            with np.load(path, allow_pickle=False) as archive:
                cls = archive["cls_tokens"]
                patches = archive["patch_tokens"]
                checks = {
                    "schema": str(archive["schema_version"].item()) == FEATURE_SCHEMA_VERSION,
                    "cls_shape": cls.ndim == 2,
                    "patch_shape": patches.ndim == 3,
                    "frame_count": cls.shape[0] == int(row["frame_count"]),
                    "matching_dimensions": cls.shape[0] == patches.shape[0]
                    and cls.shape[-1] == patches.shape[-1],
                    "finite": bool(np.isfinite(cls).all() and np.isfinite(patches).all()),
                }
            if not all(checks.values()):
                raise ValueError(json.dumps(checks, sort_keys=True))
            item = dict(row)
            item["feature_path"] = normalize_path(path)
            item["feature_hidden_size"] = int(cls.shape[-1])
            item["feature_grid_tokens"] = int(patches.shape[-2])
            valid.append(item)
            hidden_sizes[int(cls.shape[-1])] += 1
            grid_counts[int(patches.shape[-2])] += 1
            frame_counts[int(cls.shape[0])] += 1
        except Exception as exc:  # noqa: BLE001
            invalid.append(
                {
                    "sample_id": row.get("sample_id"),
                    "feature_path": normalize_path(path),
                    "error": str(exc),
                }
            )
    coverage = len(valid) / len(rows) if rows else 0.0
    status = "passed" if coverage >= args.min_coverage and len(hidden_sizes) == 1 and len(grid_counts) == 1 else "failed"
    write_jsonl(args.output_index_jsonl, valid)
    summary = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "status": status,
        "manifest_jsonl": normalize_path(args.manifest_jsonl),
        "feature_root": normalize_path(args.feature_root),
        "feature_index_jsonl": normalize_path(args.output_index_jsonl),
        "expected_records": len(rows),
        "valid_records": len(valid),
        "coverage": coverage,
        "min_coverage": args.min_coverage,
        "hidden_size_counts": dict(sorted(hidden_sizes.items())),
        "grid_token_counts": dict(sorted(grid_counts.items())),
        "frame_count_distribution": dict(sorted(frame_counts.items())),
        "invalid_count": len(invalid),
        "first_invalid": invalid[:50],
    }
    write_json(args.output_summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
