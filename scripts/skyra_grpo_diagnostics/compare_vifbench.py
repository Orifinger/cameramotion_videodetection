#!/usr/bin/env python3
"""Compare VIF-Bench summaries for the GRPO base and saved checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


METRIC_KEYS = ("balanced_accuracy", "fake_recall", "fake_f1")


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compact(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    average = result.get("average_across_fake_models")
    if not isinstance(average, Mapping):
        raise ValueError(f"{name} is missing average_across_fake_models")
    per_model = result.get("per_fake_model")
    if not isinstance(per_model, Mapping) or not per_model:
        raise ValueError(f"{name} has no per_fake_model metrics")
    return {
        "name": name,
        "coverage": float(result.get("coverage", 0.0)),
        "format_valid_rate": float(result.get("format_valid_rate", 0.0)),
        "num_predictions": int(result.get("num_predictions", 0)),
        "num_matched_predictions": int(result.get("num_matched_predictions", 0)),
        "fake_model_names": sorted(str(key) for key in per_model),
        **{key: float(average[key]) for key in METRIC_KEYS},
    }


def delta(current: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, float]:
    return {key: float(current[key]) - float(base[key]) for key in METRIC_KEYS}


def eligible_improvement(
    current: Mapping[str, Any],
    base: Mapping[str, Any],
    max_regression: float,
    min_gain: float,
) -> bool:
    changes = delta(current, base)
    primary = (changes["balanced_accuracy"], changes["fake_f1"])
    return min(primary) >= -max_regression and max(primary) >= min_gain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-json", required=True)
    parser.add_argument("--step50-json", required=True)
    parser.add_argument("--step100-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--min-format-valid", type=float, default=0.99)
    parser.add_argument("--max-primary-regression", type=float, default=0.001)
    parser.add_argument("--min-primary-gain", type=float, default=0.005)
    args = parser.parse_args()

    results = {
        "base": compact("base", read_json(args.base_json)),
        "step50": compact("step50", read_json(args.step50_json)),
        "step100": compact("step100", read_json(args.step100_json)),
    }
    base = results["base"]
    same_fake_models = all(
        result["fake_model_names"] == base["fake_model_names"]
        for result in results.values()
    )
    complete = all(
        result["coverage"] >= args.min_coverage
        and result["format_valid_rate"] >= args.min_format_valid
        for result in results.values()
    ) and same_fake_models
    improvements = {
        name: eligible_improvement(
            result,
            base,
            args.max_primary_regression,
            args.min_primary_gain,
        )
        for name, result in results.items()
        if name != "base"
    }
    status = "passed" if complete and any(improvements.values()) else "failed"

    output = {
        "gate": "GRPO checkpoints on full VIF-Bench",
        "status": status,
        "what_was_tested": (
            "The original DataB detection checkpoint, GRPO step 50, and GRPO step 100 use the same "
            "full VIF-Bench index, original no-camera detection prompt, generation settings, and parser."
        ),
        "what_was_not_tested": (
            "VIF-Bench was not used in this GRPO run, but it has been inspected repeatedly in this "
            "project, so this is not a pristine final held-out test. This experiment does not use camera data."
        ),
        "thresholds": {
            "min_coverage": args.min_coverage,
            "min_format_valid": args.min_format_valid,
            "max_primary_regression": args.max_primary_regression,
            "min_primary_gain": args.min_primary_gain,
        },
        "checks": {
            "all_outputs_complete": complete,
            "same_fake_model_set": same_fake_models,
            "step50_improves_without_material_regression": improvements["step50"],
            "step100_improves_without_material_regression": improvements["step100"],
        },
        "models": results,
        "step50_minus_base": delta(results["step50"], base),
        "step100_minus_base": delta(results["step100"], base),
        "selection_rule": (
            "A checkpoint passes when Balanced ACC and Fake F1 each drop by at most 0.1 percentage "
            "points and at least one improves by 0.5 percentage points. If both pass, select the one "
            "with the higher mean of Balanced ACC and Fake F1."
        ),
    }
    write_json(args.output_json, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
