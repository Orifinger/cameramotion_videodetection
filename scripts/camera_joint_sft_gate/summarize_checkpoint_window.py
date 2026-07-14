#!/usr/bin/env python3
"""Summarize early joint-SFT checkpoints for a camera/detection Pareto window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.camera_joint_sft_gate.summarize_dataa import compact as compact_dataa
from scripts.camera_joint_sft_gate.summarize_dataa import metric_deltas
from scripts.caspr_gate1.runtime import write_json


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(payload)


def compact_camera(payload: Mapping[str, Any]) -> dict[str, float | int]:
    conditions = payload.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("camera evaluation has no conditions object")
    matched = conditions.get("matched_frames")
    if not isinstance(matched, Mapping):
        raise ValueError("camera evaluation has no matched_frames condition")
    overall = matched.get("overall")
    macro = matched.get("macro")
    if not isinstance(overall, Mapping) or not isinstance(macro, Mapping):
        raise ValueError("camera evaluation lacks overall or macro metrics")
    return {
        "coverage": float(matched.get("coverage", 0.0)),
        "num_supported_labels": int(matched.get("num_supported_labels", 0)),
        "balanced_accuracy": float(overall.get("balanced_accuracy", 0.0)),
        "macro_average_precision": float(macro.get("average_precision", 0.0)),
        "paired_question_accuracy": float(matched.get("paired_question_accuracy", 0.0)),
    }


def camera_delta(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, float]:
    return {
        "balanced_accuracy": float(candidate["balanced_accuracy"])
        - float(control["balanced_accuracy"]),
        "macro_average_precision": float(candidate["macro_average_precision"])
        - float(control["macro_average_precision"]),
        "paired_question_accuracy": float(candidate["paired_question_accuracy"])
        - float(control["paired_question_accuracy"]),
    }


def build_summary(
    root: str | Path,
    steps: list[int],
    *,
    min_camera_macro_ap_gain: float,
    min_camera_balanced_gain: float,
    min_detection_primary_gain: float,
    max_detection_other_drop: float,
    min_coverage: float,
    min_format_valid: float,
) -> dict[str, Any]:
    root = Path(root)
    checkpoints: dict[str, Any] = {}
    candidate_steps: list[int] = []
    for step in steps:
        camera_detection = compact_camera(
            read_json(root / "camera_eval" / f"step_{step}_detection_only.json")
        )
        camera_correct = compact_camera(
            read_json(root / "camera_eval" / f"step_{step}_correct_camera.json")
        )
        dataa_detection = compact_dataa(
            read_json(
                root
                / "dataa"
                / f"step_{step}"
                / "detection_only"
                / "eval"
                / "camera_adapter"
                / "dataa_detection_camera_adapter_summary.json"
            )
        )
        dataa_correct = compact_dataa(
            read_json(
                root
                / "dataa"
                / f"step_{step}"
                / "correct_camera"
                / "eval"
                / "camera_adapter"
                / "dataa_detection_camera_adapter_summary.json"
            )
        )
        camera_gain = camera_delta(camera_correct, camera_detection)
        detection_gain = metric_deltas(dataa_correct, dataa_detection)
        checks = {
            "camera_coverage": (
                camera_correct["coverage"] >= min_coverage
                and camera_detection["coverage"] >= min_coverage
            ),
            "detection_coverage": (
                dataa_correct["coverage"] >= min_coverage
                and dataa_detection["coverage"] >= min_coverage
            ),
            "detection_format": (
                dataa_correct["format_valid_rate"] >= min_format_valid
                and dataa_detection["format_valid_rate"] >= min_format_valid
            ),
            "camera_ability_added": (
                camera_gain["macro_average_precision"] >= min_camera_macro_ap_gain
                or camera_gain["balanced_accuracy"] >= min_camera_balanced_gain
            ),
            "detection_gain_without_regression": (
                max(detection_gain["balanced_accuracy"], detection_gain["pair_accuracy"])
                >= min_detection_primary_gain
                and detection_gain["balanced_accuracy"] >= -max_detection_other_drop
                and detection_gain["pair_accuracy"] >= -max_detection_other_drop
                and detection_gain["fake_f1"] >= -max_detection_other_drop
            ),
        }
        passed = all(checks.values())
        if passed:
            candidate_steps.append(step)
        checkpoints[str(step)] = {
            "status": "candidate" if passed else "not_candidate",
            "checks": checks,
            "camera": {
                "detection_only": camera_detection,
                "correct_camera": camera_correct,
                "correct_minus_detection_only": camera_gain,
            },
            "dataa_detection": {
                "detection_only": dataa_detection,
                "correct_camera": dataa_correct,
                "correct_minus_detection_only": detection_gain,
            },
        }
    status = "candidate_found" if candidate_steps else "no_candidate_in_tested_steps"
    return {
        "gate": "联合训练早期 checkpoint 的相机-检测 Pareto 窗口审计",
        "status": status,
        "tested_steps": steps,
        "candidate_steps": candidate_steps,
        "thresholds": {
            "min_camera_macro_ap_gain": min_camera_macro_ap_gain,
            "min_camera_balanced_accuracy_gain": min_camera_balanced_gain,
            "min_detection_balanced_or_pair_gain": min_detection_primary_gain,
            "max_other_detection_metric_drop": max_detection_other_drop,
            "min_coverage": min_coverage,
            "min_format_valid": min_format_valid,
        },
        "checkpoints": checkpoints,
        "does_not_establish": (
            "This development audit does not establish final-test gains. A candidate must still "
            "beat a same-step flipped-camera control before VIF-Bench or RL."
        ),
        "next_action": (
            "Evaluate the earliest candidate against the same-step flipped-camera checkpoint."
            if candidate_steps
            else "Stop this independent-task SFT recipe unless the two tested steps show a clear improving tradeoff trend."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[698, 1396])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-camera-macro-ap-gain", type=float, default=0.03)
    parser.add_argument("--min-camera-balanced-gain", type=float, default=0.05)
    parser.add_argument("--min-detection-primary-gain", type=float, default=0.02)
    parser.add_argument("--max-detection-other-drop", type=float, default=0.01)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--min-format-valid", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        args.root,
        args.steps,
        min_camera_macro_ap_gain=args.min_camera_macro_ap_gain,
        min_camera_balanced_gain=args.min_camera_balanced_gain,
        min_detection_primary_gain=args.min_detection_primary_gain,
        max_detection_other_drop=args.max_detection_other_drop,
        min_coverage=args.min_coverage,
        min_format_valid=args.min_format_valid,
    )
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
