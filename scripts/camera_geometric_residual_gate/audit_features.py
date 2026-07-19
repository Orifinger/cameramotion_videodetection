#!/usr/bin/env python3
"""Audit compact feature coverage and shape contracts."""

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

from scripts.camera_flow_probe.contracts import read_jsonl
from scripts.camera_geometric_residual_gate.contracts import FEATURE_SCHEMA_VERSION, feature_filename, write_json
from scripts.camera_geometric_residual_gate.features import VARIANT_DIMS


def _directory(dataset_name: str) -> str:
    return dataset_name.casefold().replace("-", "_")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_jsonl(args.manifest_jsonl)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    failures: list[dict[str, Any]] = []
    quality_counts: Counter[str] = Counter()
    valid = 0
    for row in rows:
        sample_id = str(row["sample_id"])
        path = args.feature_root / _directory(str(row["dataset_name"])) / feature_filename(sample_id)
        if not path.is_file():
            failures.append({"sample_id": sample_id, "reason": "missing", "path": str(path)})
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["schema_version"].item()) != FEATURE_SCHEMA_VERSION:
                    raise ValueError("schema mismatch")
                if str(archive["sample_id"].item()) != sample_id:
                    raise ValueError("sample_id mismatch")
                for variant, dimension in VARIANT_DIMS.items():
                    value = np.asarray(archive[variant])
                    if value.shape != (dimension,):
                        raise ValueError(f"{variant} shape={value.shape}, expected={(dimension,)}")
                    if not np.isfinite(value).all():
                        raise ValueError(f"{variant} contains non-finite values")
            valid += 1
            quality_counts[str(row.get("motion_bucket", "unknown"))] += 1
        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_id": sample_id, "reason": str(exc), "path": str(path)})
    coverage = valid / max(1, len(rows))
    summary = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "manifest_jsonl": str(args.manifest_jsonl),
        "requested_samples": len(rows),
        "valid_samples": valid,
        "coverage": coverage,
        "variant_dims": VARIANT_DIMS,
        "valid_motion_bucket_counts": dict(sorted(quality_counts.items())),
        "failure_count": len(failures),
        "first_failures": failures[:50],
        "status": "passed" if coverage >= args.min_coverage else "failed",
    }
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
