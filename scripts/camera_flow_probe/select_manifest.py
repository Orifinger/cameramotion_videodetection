#!/usr/bin/env python3
"""Select a deterministic motion-balanced subset of a probe manifest."""

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
    parser.add_argument("--per-motion-bucket", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = sorted(read_jsonl(args.manifest_jsonl), key=lambda row: str(row["case_id"]))
    if args.split != "all":
        rows = [row for row in rows if row.get("dataset_split") == args.split]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("motion_bucket", "unknown"))].append(row)
    selected = [
        row
        for bucket in sorted(groups)
        for row in groups[bucket][: max(1, args.per_motion_bucket)]
    ]
    write_jsonl(args.out_jsonl, selected)
    print(
        json.dumps(
            {
                "out_jsonl": str(args.out_jsonl),
                "case_count": len(selected),
                "by_motion_bucket": {
                    bucket: min(len(values), max(1, args.per_motion_bucket))
                    for bucket, values in sorted(groups.items())
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
