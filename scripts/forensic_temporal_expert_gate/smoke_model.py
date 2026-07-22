#!/usr/bin/env python3
"""One-second variable-length forward smoke for all expert heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from scripts.forensic_temporal_expert_gate.contracts import write_json
from scripts.forensic_temporal_expert_gate.model import ForensicTemporalExpert, ModelConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    cls = torch.randn(3, 17, 32, device=device)
    patches = torch.randn(3, 17, 16, 32, device=device)
    lengths = torch.tensor([11, 16, 17], device=device)
    checks = {}
    for mode in ("static", "ordered", "shuffled"):
        model = ForensicTemporalExpert(
            ModelConfig(input_dim=32, hidden_dim=24, dropout=0.0, mode=mode)
        ).to(device).eval()
        with torch.no_grad():
            output = model(cls, patches, lengths)
        checks[mode] = output.shape == (3,) and bool(torch.isfinite(output).all())
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "variable_frame_lengths": [11, 16, 17],
        "device": str(device),
    }
    write_json(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
