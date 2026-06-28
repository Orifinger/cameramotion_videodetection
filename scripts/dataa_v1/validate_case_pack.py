#!/usr/bin/env python3
"""Validate a Data A v1 Stage P case manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.common import read_json
from scripts.dataa_v1.manifest import validate_manifest_payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = read_json(args.manifest)
    errors = validate_manifest_payload(payload)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("case pack manifest valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

