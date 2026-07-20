#!/usr/bin/env python3
"""Check that an external CTNE benchmark does not overlap DataB identities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import PurePosixPath, Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_ctne_gate1.contracts import normalize_path, path_key, read_jsonl, write_json


def _tail_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    parts = PurePosixPath(normalize_path(row.get("frame_dir_path"))).parts
    tail = "/".join(parts[-2:]).casefold() if len(parts) >= 2 else "/".join(parts).casefold()
    return str(row.get("label_name", "unknown")).casefold(), str(row.get("generator_name", "unknown")).casefold(), tail


def audit(train_rows: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    train_paths = {path_key(row.get("frame_dir_path")): row for row in train_rows}
    train_tails: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in train_rows:
        train_tails.setdefault(_tail_key(row), []).append(row)
    exact: list[dict[str, Any]] = []
    identity: list[dict[str, Any]] = []
    for row in test_rows:
        normalized = path_key(row.get("frame_dir_path"))
        if normalized in train_paths:
            exact.append({"train": train_paths[normalized]["sample_id"], "test": row["sample_id"]})
        candidates = train_tails.get(_tail_key(row), [])
        for candidate in candidates:
            identity.append({"train": candidate["sample_id"], "test": row["sample_id"], "identity_tail": _tail_key(row)[2]})
    return {
        "gate": "CTNE train/external identity-overlap audit",
        "status": "passed" if not exact and not identity else "failed",
        "train_records": len(train_rows),
        "external_records": len(test_rows),
        "exact_path_overlap_count": len(exact),
        "source_label_tail_overlap_count": len(identity),
        "first_exact_overlaps": exact[:50],
        "first_identity_overlaps": identity[:50],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--external-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit(read_jsonl(args.train_manifest_jsonl), read_jsonl(args.external_manifest_jsonl))
    write_json(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
