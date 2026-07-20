#!/usr/bin/env python3
"""Prepare supervised train/validation sequences from existing CTNE archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_discriminative_gate.data import prepare_datab


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-index-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--fit-transitions-per-video", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--clip-value", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = prepare_datab(
        feature_index_jsonl=args.feature_index_jsonl,
        output_dir=args.output_dir,
        pca_dim=args.pca_dim,
        fit_transitions_per_video=args.fit_transitions_per_video,
        seed=args.seed,
        clip_value=args.clip_value,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
