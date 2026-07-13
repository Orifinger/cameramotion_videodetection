#!/usr/bin/env python3
"""Build the fixed 40step_v3 DataA detection-development split."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


CASE_RE = re.compile(r"(dataA_v1_(?:dataset_v2_|textedit_reserve_)?\d+)")


def read_json(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def first_image(record: Mapping[str, Any]) -> str:
    images = record.get("images")
    if not isinstance(images, list) or not images:
        return ""
    return str(images[0]).replace("\\", "/")


def case_id(record: Mapping[str, Any]) -> str:
    match = CASE_RE.search(first_image(record))
    return match.group(1) if match else ""


def side(record: Mapping[str, Any]) -> str:
    path = first_image(record)
    if "/real/" in path:
        return "real"
    if "/fake/" in path:
        return "fake"
    return "unknown"


def source_family(value: str) -> str:
    if "textedit_reserve" in value:
        return "vace13b_textedit_40step_v3"
    if "dataset_v2" in value:
        return "vace13b_dataset_40step_v3"
    return "vace14b_reused"


def split_case_ids(records: Sequence[Mapping[str, Any]]) -> set[str]:
    values = {case_id(row) for row in records}
    values.discard("")
    if not values:
        raise ValueError("the fixed development split contains no DataA case ids")
    return values


def build(
    detection_records: Sequence[Mapping[str, Any]],
    development_ids: set[str],
    *,
    check_images: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    by_case: dict[str, set[str]] = defaultdict(set)
    missing_images: list[str] = []
    camera_context_records = 0

    for raw in detection_records:
        identity = case_id(raw)
        if identity not in development_ids:
            continue
        record_side = side(raw)
        if record_side not in {"real", "fake"}:
            raise ValueError(f"cannot infer real/fake side for {identity}: {first_image(raw)}")
        key = (identity, record_side)
        if key in seen:
            raise ValueError(f"duplicate detection record: {identity}/{record_side}")
        seen.add(key)
        by_case[identity].add(record_side)
        record = dict(raw)
        images = record.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError(f"record has no images: {identity}/{record_side}")
        if check_images:
            for image in images:
                if not Path(str(image)).is_file():
                    missing_images.append(str(image))
                    if len(missing_images) >= 20:
                        break
        messages = record.get("messages")
        rendered = json.dumps(messages, ensure_ascii=False) if isinstance(messages, list) else ""
        camera_context_records += int("Camera Motion Context" in rendered)
        selected.append(record)

    missing_cases = sorted(development_ids - set(by_case))
    incomplete_cases = sorted(
        identity for identity, sides in by_case.items() if sides != {"real", "fake"}
    )
    if missing_cases:
        raise ValueError(f"missing development cases in detection JSON: {missing_cases[:20]}")
    if incomplete_cases:
        raise ValueError(f"incomplete real/fake development pairs: {incomplete_cases[:20]}")
    if missing_images:
        raise FileNotFoundError(f"missing frame images, first paths: {missing_images}")
    if camera_context_records:
        raise ValueError(
            f"camera context unexpectedly appears in {camera_context_records} detection records"
        )

    selected.sort(key=lambda row: (case_id(row), side(row)))
    side_counts = Counter(side(row) for row in selected)
    source_counts = Counter(source_family(identity) for identity in by_case)
    frame_counts = Counter(len(row.get("images") or []) for row in selected)
    summary = {
        "gate": "binary camera VQA adapter detection-retention data audit",
        "num_input_detection_records": len(detection_records),
        "num_fixed_development_cases": len(development_ids),
        "num_output_records": len(selected),
        "num_complete_pairs": len(by_case),
        "side_counts": dict(side_counts),
        "source_case_counts": dict(source_counts),
        "frame_count_distribution": {str(key): value for key, value in sorted(frame_counts.items())},
        "camera_context_records": camera_context_records,
        "missing_cases": missing_cases,
        "incomplete_cases": incomplete_cases,
        "check_images": check_images,
        "status": "passed",
    }
    return selected, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-json", required=True)
    parser.add_argument("--fixed-dev-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--check-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detection = read_json(args.detection_json)
    fixed_dev = read_json(args.fixed_dev_json)
    records, summary = build(
        detection,
        split_case_ids(fixed_dev),
        check_images=args.check_images,
    )
    summary.update(
        {
            "detection_json": args.detection_json,
            "fixed_dev_json": args.fixed_dev_json,
            "output_json": args.output_json,
        }
    )
    write_json(args.output_json, records)
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

