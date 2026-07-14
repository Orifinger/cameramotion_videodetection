#!/usr/bin/env python3
"""Summarize the four-model no-camera DataA detection gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.caspr_gate1.runtime import write_json


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(payload)


def compact(payload: Mapping[str, Any]) -> dict[str, Any]:
    basic = payload.get("basic")
    pair = payload.get("pair")
    if not isinstance(basic, Mapping) or not isinstance(pair, Mapping):
        raise ValueError("DataA evaluation JSON must contain basic and pair objects")
    gt = int(payload.get("num_gt_records", 0))
    matched = int(payload.get("num_matched_records", 0))
    result: dict[str, Any] = {
        "num_gt_records": gt,
        "num_matched_records": matched,
        "coverage": matched / gt if gt else 0.0,
        "format_valid_rate": float(basic.get("format_valid_rate", 0.0)),
        "accuracy": float(basic.get("accuracy", 0.0)),
        "balanced_accuracy": float(basic.get("balanced_accuracy", 0.0)),
        "fake_recall": float(basic.get("fake_recall", 0.0)),
        "real_recall": float(basic.get("real_recall", 0.0)),
        "fake_f1": float(basic.get("fake_f1", 0.0)),
        "pair_accuracy": float(pair.get("pair_accuracy", 0.0)),
        "num_pairs": int(pair.get("num_pairs", 0)),
    }
    iou = payload.get("iou")
    if isinstance(iou, Mapping):
        result["evidence"] = {
            "pred_evidence_sample_rate": float(iou.get("pred_evidence_sample_rate", 0.0)),
            "mean_best_temporal_iou": float(iou.get("mean_best_temporal_iou", 0.0)),
            "mean_best_bbox_iou": float(iou.get("mean_best_bbox_iou", 0.0)),
            "evidence_hit_t03_b03": float(iou.get("evidence_hit_t03_b03", 0.0)),
            "sample_any_evidence_hit_t03_b03": float(
                iou.get("sample_any_evidence_hit_t03_b03", 0.0)
            ),
        }
    return result


def metric_deltas(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(control[key])
        for key in ("balanced_accuracy", "fake_f1", "pair_accuracy", "format_valid_rate")
    }


def beats_control(
    delta: Mapping[str, float], min_primary_gain: float, max_other_drop: float
) -> bool:
    primary_gain = max(delta["balanced_accuracy"], delta["pair_accuracy"])
    return (
        primary_gain >= min_primary_gain
        and delta["balanced_accuracy"] >= -max_other_drop
        and delta["pair_accuracy"] >= -max_other_drop
        and delta["fake_f1"] >= -max_other_drop
    )


def build_summary(
    base_payload: Mapping[str, Any],
    detection_only_payload: Mapping[str, Any],
    correct_payload: Mapping[str, Any],
    flipped_payload: Mapping[str, Any],
    *,
    min_coverage: float,
    min_format_valid: float,
    min_primary_gain: float,
    max_other_drop: float,
) -> dict[str, Any]:
    models = {
        "base_detection_checkpoint": compact(base_payload),
        "detection_only_control": compact(detection_only_payload),
        "correct_camera_auxiliary": compact(correct_payload),
        "flipped_camera_auxiliary": compact(flipped_payload),
    }
    correct = models["correct_camera_auxiliary"]
    detection_only = models["detection_only_control"]
    flipped = models["flipped_camera_auxiliary"]
    deltas = {
        "correct_minus_detection_only": metric_deltas(correct, detection_only),
        "correct_minus_flipped": metric_deltas(correct, flipped),
        "correct_minus_base": metric_deltas(correct, models["base_detection_checkpoint"]),
    }
    checks = {
        "all_models_have_coverage": all(
            model["coverage"] >= min_coverage for model in models.values()
        ),
        "all_models_preserve_detection_format": all(
            model["format_valid_rate"] >= min_format_valid for model in models.values()
        ),
        "correct_camera_beats_detection_only": beats_control(
            deltas["correct_minus_detection_only"], min_primary_gain, max_other_drop
        ),
        "correct_camera_beats_flipped_camera": beats_control(
            deltas["correct_minus_flipped"], min_primary_gain, max_other_drop
        ),
    }
    passed = all(checks.values())
    return {
        "gate": "DataA 无相机文本四模型检测迁移门",
        "status": "passed" if passed else "failed",
        "what_was_tested": (
            "The base checkpoint and three equal-compute joint-SFT branches use the same held-out "
            "DataA ordered frames and original detection prompt. No camera text is supplied."
        ),
        "thresholds": {
            "min_coverage": min_coverage,
            "min_format_valid": min_format_valid,
            "min_balanced_or_pair_accuracy_gain": min_primary_gain,
            "max_drop_in_other_primary_or_fake_f1": max_other_drop,
        },
        "checks": checks,
        "models": models,
        "deltas": deltas,
        "does_not_establish": (
            "This development gate does not establish VIF-Bench transfer, a fresh final-test gain, "
            "or statistical significance."
        ),
        "next_action": (
            "Run paired uncertainty analysis, then VIF-Bench without camera text."
            if passed
            else "Stop before VIF-Bench and inspect per-case errors and task-transfer failure."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-eval", required=True)
    parser.add_argument("--detection-only-eval", required=True)
    parser.add_argument("--correct-camera-eval", required=True)
    parser.add_argument("--flipped-camera-eval", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--min-format-valid", type=float, default=0.95)
    parser.add_argument("--min-primary-gain", type=float, default=0.02)
    parser.add_argument("--max-other-drop", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        read_json(args.base_eval),
        read_json(args.detection_only_eval),
        read_json(args.correct_camera_eval),
        read_json(args.flipped_camera_eval),
        min_coverage=args.min_coverage,
        min_format_valid=args.min_format_valid,
        min_primary_gain=args.min_primary_gain,
        max_other_drop=args.max_other_drop,
    )
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
