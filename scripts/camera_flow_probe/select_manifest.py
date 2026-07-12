#!/usr/bin/env python3
"""Select a deterministic source-and-motion-balanced probe subset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_flow_probe.contracts import read_jsonl, write_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="train")
    parser.add_argument("--per-source-motion", type=int, default=1)
    return parser.parse_args(argv)


def select_rows(
    rows: Sequence[dict[str, Any]],
    *,
    split: str,
    per_source_motion: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filtered = sorted(rows, key=lambda row: str(row["case_id"]))
    if split != "all":
        filtered = [row for row in filtered if row.get("dataset_split") == split]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, int] = defaultdict(int)
    by_motion: dict[str, int] = defaultdict(int)
    limit = max(1, per_source_motion)
    for row in filtered:
        key = (
            str(row.get("source_name", "unknown-source")),
            str(row.get("motion_bucket", "unknown")),
        )
        groups[key].append(row)
    selected: list[dict[str, Any]] = []
    by_source_motion: dict[str, int] = {}
    for (source, motion), values in sorted(groups.items()):
        chosen = values[:limit]
        selected.extend(chosen)
        by_source[source] += len(chosen)
        by_motion[motion] += len(chosen)
        by_source_motion[f"{source}|{motion}"] = len(chosen)
    return selected, {
        "case_count": len(selected),
        "by_source": dict(sorted(by_source.items())),
        "by_motion_bucket": dict(sorted(by_motion.items())),
        "by_source_motion": by_source_motion,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selected, summary = select_rows(
        read_jsonl(args.manifest_jsonl),
        split=args.split,
        per_source_motion=args.per_source_motion,
    )
    write_jsonl(args.out_jsonl, selected)
    print(
        json.dumps(
            {
                "out_jsonl": str(args.out_jsonl),
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
