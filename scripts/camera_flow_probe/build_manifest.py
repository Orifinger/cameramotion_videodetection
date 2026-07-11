#!/usr/bin/env python3
"""Build the strict 40step_v3 manifest for the camera-flow probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_flow_probe.contracts import build_probe_manifest
from scripts.dataa_v1.common import DataAError


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records-jsonl",
        type=Path,
        default=Path("res/dataA_v1/autolabel/dataa_vace_grounded_cot_v4_records_40step_v3.jsonl"),
    )
    parser.add_argument(
        "--camera-jsonl",
        type=Path,
        default=Path("camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl"),
    )
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=1080)
    parser.add_argument("--expected-test-cases", type=int, default=321)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--allow-nonfinal-contract", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_probe_manifest(
            records_jsonl=args.records_jsonl,
            camera_jsonl=args.camera_jsonl,
            test_split=args.test_split,
            out_jsonl=args.out_jsonl,
            out_summary=args.out_summary,
            expected_cases=args.expected_cases,
            expected_test_cases=args.expected_test_cases,
            check_files=bool(args.check_files),
            strict_final_contract=not bool(args.allow_nonfinal_contract),
        )
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
