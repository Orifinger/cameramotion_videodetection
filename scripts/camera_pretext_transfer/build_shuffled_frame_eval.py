#!/usr/bin/env python3
"""Build a dev control that keeps gold labels but permutes the input video frames."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from scripts.caspr_gate1.runtime import read_jsonl, write_json


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dev-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.canonical_dev_jsonl)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("motion_bucket", "unknown"))].append(row)
    ordered: list[dict[str, Any]] = []
    for name in sorted(groups, key=lambda key: (-len(groups[key]), key)):
        ordered.extend(sorted(groups[name], key=lambda row: str(row.get("case_id"))))
    if len(ordered) < 2:
        raise ValueError("shuffled-frame control needs at least two records")
    largest_group = max(len(values) for values in groups.values())
    shift = largest_group % len(ordered)
    if shift == 0:
        shift = 1
    donors = ordered[shift:] + ordered[:shift]
    donor_by_case = {str(row["case_id"]): donor for row, donor in zip(ordered, donors)}
    output: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row["case_id"])
        donor = donor_by_case[case_id]
        if str(donor.get("case_id")) == case_id:
            raise AssertionError(f"self donor survived frame permutation: {case_id}")
        item = dict(row)
        item["images"] = list(donor["images"])
        item["sample_id"] = f"{case_id}:real:canonical:shuffled_frames"
        item["input_case_id"] = donor.get("case_id")
        item["input_motion_bucket"] = donor.get("motion_bucket")
        item["input_camera_labels"] = donor.get("camera_labels", [])
        item["input_bucket_matches_gold"] = donor.get("motion_bucket") == row.get("motion_bucket")
        item["input_label_set_matches_gold"] = donor.get("camera_labels") == row.get("camera_labels")
        output.append(item)
    donor_ids = [str(row["input_case_id"]) for row in output]
    summary = {
        "gate": "Stage 1 shuffled-frame visual-dependence control",
        "canonical_dev_jsonl": args.canonical_dev_jsonl,
        "output_jsonl": args.output_jsonl,
        "num_records": len(output),
        "num_unique_donors": len(set(donor_ids)),
        "self_donor_count": sum(str(row["case_id"]) == str(row["input_case_id"]) for row in output),
        "gold_motion_bucket_counts": dict(Counter(str(row.get("motion_bucket")) for row in output)),
        "donor_motion_bucket_counts": dict(Counter(str(row.get("input_motion_bucket")) for row in output)),
        "bucket_mismatch_count": sum(not row["input_bucket_matches_gold"] for row in output),
        "bucket_mismatch_rate": sum(not row["input_bucket_matches_gold"] for row in output) / len(output),
        "label_set_mismatch_count": sum(not row["input_label_set_matches_gold"] for row in output),
        "label_set_mismatch_rate": sum(not row["input_label_set_matches_gold"] for row in output) / len(output),
        "permutation_shift": shift,
    }
    write_jsonl(Path(args.output_jsonl), output)
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
