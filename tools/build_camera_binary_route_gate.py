#!/usr/bin/env python3
"""Build equal-data detection datasets for the frozen binary camera router."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.build_camera_joint_sft_gate import camera_text_in_detection_prompt


SOURCE_FILES = {
    "shared": "hard_route_shared.json",
    "no-motion": "hard_route_no_motion.json",
    "minor-motion": "hard_route_minor_motion.json",
    "complex-motion": "hard_route_complex_motion.json",
}
OUTPUT_FILES = {
    "shared": "binary_route_shared.json",
    "no-motion": "binary_route_no_motion.json",
    "motion": "binary_route_motion.json",
}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_id(row: Mapping[str, Any]) -> str:
    return str(row.get("route_record_id") or row.get("sample_id") or "")


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"expected a non-empty JSON list: {path}")
    records = [dict(row) for row in payload if isinstance(row, Mapping)]
    if len(records) != len(payload):
        raise ValueError(f"non-object records found: {path}")
    identifiers = [record_id(row) for row in records]
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError(f"missing or duplicate route record ids: {path}")
    return records


def validate_source_partition(sources: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    shared_ids = {record_id(row) for row in sources["shared"]}
    expert_id_lists = {
        bucket: [record_id(row) for row in sources[bucket]]
        for bucket in ("no-motion", "minor-motion", "complex-motion")
    }
    expert_ids = [identifier for values in expert_id_lists.values() for identifier in values]
    if len(expert_ids) != len(set(expert_ids)):
        raise AssertionError("three-class source experts are not disjoint")
    if shared_ids != set(expert_ids):
        raise AssertionError("three-class shared data is not the exact expert union")
    for bucket, rows in sources.items():
        if bucket == "shared":
            continue
        if any(str(row.get("route_bucket")) != bucket for row in rows):
            raise AssertionError(f"source expert {bucket} contains another route bucket")


def attach_binary_bucket(row: Mapping[str, Any], bucket: str) -> dict[str, Any]:
    output = copy.deepcopy(dict(row))
    output["binary_route_bucket"] = bucket
    output["binary_route_mapping"] = "no-motion_vs_minor-plus-complex-motion"
    return output


def validate_binary_partition(
    shared: Sequence[Mapping[str, Any]],
    no_motion: Sequence[Mapping[str, Any]],
    motion: Sequence[Mapping[str, Any]],
) -> None:
    no_ids = [record_id(row) for row in no_motion]
    motion_ids = [record_id(row) for row in motion]
    shared_ids = [record_id(row) for row in shared]
    expert_ids = no_ids + motion_ids
    if len(expert_ids) != len(set(expert_ids)):
        raise AssertionError("a detection record appears in both binary experts")
    if Counter(expert_ids) != Counter(shared_ids):
        raise AssertionError("binary shared data is not the exact disjoint expert union")
    for name, rows in (("no-motion", no_motion), ("motion", motion), ("shared", shared)):
        labels = Counter(str(row.get("detection_label")) for row in rows)
        if set(labels) != {"Real", "Fake"}:
            raise ValueError(f"binary branch {name} is not a Real/Fake dataset: {dict(labels)}")
        if labels["Real"] != labels["Fake"]:
            raise ValueError(f"binary branch {name} is not Real/Fake balanced: {dict(labels)}")
        if any(camera_text_in_detection_prompt(row) for row in rows):
            raise AssertionError(f"camera text leaked into binary detection branch {name}")
    if any(row.get("binary_route_bucket") != "no-motion" for row in no_motion):
        raise AssertionError("no-motion binary expert contains another binary route")
    if any(row.get("binary_route_bucket") != "motion" for row in motion):
        raise AssertionError("motion binary expert contains another binary route")


def branch_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "detection_labels": dict(Counter(str(row.get("detection_label")) for row in rows)),
        "three_class_buckets": dict(Counter(str(row.get("route_bucket")) for row in rows)),
        "binary_buckets": dict(Counter(str(row.get("binary_route_bucket")) for row in rows)),
        "domains": dict(Counter(str(row.get("route_domain")) for row in rows)),
        "sources": dict(Counter(str(row.get("gate_source")) for row in rows)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-route-data-dir", type=Path, required=True)
    parser.add_argument("--binary-audit-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--check-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = read_json(args.binary_audit_summary)
    if not isinstance(audit, Mapping) or audit.get("status") != "passed":
        raise ValueError(f"binary route audit did not pass: {args.binary_audit_summary}")
    if not all(bool(value) for value in dict(audit.get("checks", {})).values()):
        raise ValueError("binary route audit has a failed check")

    sources = {
        name: load_records(args.hard_route_data_dir / file_name)
        for name, file_name in SOURCE_FILES.items()
    }
    validate_source_partition(sources)

    no_motion = [attach_binary_bucket(row, "no-motion") for row in sources["no-motion"]]
    motion = [
        attach_binary_bucket(row, "motion")
        for bucket in ("minor-motion", "complex-motion")
        for row in sources[bucket]
    ]
    random.Random(args.seed + 1).shuffle(no_motion)
    random.Random(args.seed + 2).shuffle(motion)
    shared = [copy.deepcopy(row) for row in no_motion + motion]
    random.Random(args.seed + 3).shuffle(shared)
    validate_binary_partition(shared, no_motion, motion)

    if args.check_images:
        images = sorted({str(path) for row in shared for path in row.get("images", [])})
        missing = [path for path in images if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"missing {len(missing)}/{len(images)} binary detection images; first={missing[0]}"
            )

    payloads = {"shared": shared, "no-motion": no_motion, "motion": motion}
    outputs: dict[str, Any] = {}
    for name, rows in payloads.items():
        path = args.out_dir / OUTPUT_FILES[name]
        write_json(path, rows)
        outputs[name] = {
            "path": str(path),
            "sha256": sha256(path),
            **branch_summary(rows),
        }

    summary = {
        "schema_version": "camera_binary_route_detection_data_v1",
        "question": (
            "Does a frozen no-motion versus motion camera route select detection experts that improve "
            "Real/Fake classification over an equal-data shared model and a swapped-route control?"
        ),
        "binary_audit_summary": str(args.binary_audit_summary),
        "binary_audit_status": audit.get("status"),
        "hard_route_data_dir": str(args.hard_route_data_dir),
        "out_dir": str(args.out_dir),
        "seed": args.seed,
        "mapping": {
            "no-motion": ["no-motion"],
            "motion": ["minor-motion", "complex-motion"],
        },
        "shared_is_exact_disjoint_binary_expert_union": True,
        "combined_expert_records_equal_shared_records": len(no_motion) + len(motion) == len(shared),
        "camera_text_enters_detection_prompt": False,
        "images_checked": bool(args.check_images),
        "outputs": outputs,
    }
    summary_path = args.out_dir / "camera_binary_route_data_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
