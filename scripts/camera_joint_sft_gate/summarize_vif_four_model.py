#!/usr/bin/env python3
"""Summarize the fixed four-model ViF-Bench development comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.caspr_gate1.runtime import write_json


PRIMARY_METRICS = ("balanced_accuracy", "fake_f1")


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(payload)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compact(payload: Mapping[str, Any]) -> dict[str, Any]:
    average = payload.get("average_across_fake_models")
    per_model = payload.get("per_fake_model")
    if not isinstance(average, Mapping) or not isinstance(per_model, Mapping):
        raise ValueError("ViF evaluation lacks average_across_fake_models or per_fake_model")

    real_recalls: list[float] = []
    fake_precisions: list[float] = []
    predicted_fake_rates: list[float] = []
    for metrics in per_model.values():
        if not isinstance(metrics, Mapping):
            continue
        if metrics.get("real_recall") is not None:
            real_recalls.append(float(metrics["real_recall"]))
        if metrics.get("fake_precision") is not None:
            fake_precisions.append(float(metrics["fake_precision"]))
        confusion = metrics.get("confusion")
        num_pairs = int(metrics.get("num_pairs", 0))
        if isinstance(confusion, Mapping) and num_pairs > 0:
            predicted_fake = int(confusion.get("real_as_fake", 0)) + int(
                confusion.get("fake_as_fake", 0)
            )
            predicted_fake_rates.append(predicted_fake / (2.0 * num_pairs))

    return {
        "num_expected_predictions": int(payload.get("num_expected_predictions", 0)),
        "num_matched_predictions": int(payload.get("num_matched_predictions", 0)),
        "coverage": float(payload.get("coverage", 0.0)),
        "format_valid_rate": float(payload.get("format_valid_rate", 0.0)),
        "num_fake_models": int(average.get("num_models", 0)),
        "balanced_accuracy": float(average.get("balanced_accuracy", 0.0)),
        "real_recall": mean(real_recalls),
        "fake_recall": float(average.get("fake_recall", 0.0)),
        "fake_precision": mean(fake_precisions),
        "fake_f1": float(average.get("fake_f1", 0.0)),
        "predicted_fake_rate": mean(predicted_fake_rates),
    }


def metric_deltas(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, float]:
    keys = (
        "balanced_accuracy",
        "real_recall",
        "fake_recall",
        "fake_precision",
        "fake_f1",
        "predicted_fake_rate",
        "format_valid_rate",
    )
    return {
        key: float(candidate[key]) - float(control[key])
        for key in keys
        if candidate.get(key) is not None and control.get(key) is not None
    }


def per_source_deltas(
    candidate: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    candidate_sources = candidate.get("per_fake_model")
    control_sources = control.get("per_fake_model")
    if not isinstance(candidate_sources, Mapping) or not isinstance(control_sources, Mapping):
        raise ValueError("ViF evaluation lacks per_fake_model")
    output: dict[str, dict[str, float]] = {}
    for source in sorted(set(candidate_sources) & set(control_sources)):
        left = candidate_sources[source]
        right = control_sources[source]
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            continue
        output[str(source)] = {
            key: float(left[key]) - float(right[key])
            for key in ("balanced_accuracy", "real_recall", "fake_recall", "fake_f1")
            if left.get(key) is not None and right.get(key) is not None
        }
    return output


def comparison(
    candidate_compact: Mapping[str, Any],
    control_compact: Mapping[str, Any],
    source_deltas: Mapping[str, Mapping[str, float]],
    *,
    min_primary_gain: float,
    max_other_primary_drop: float,
    min_source_win_rate: float,
) -> dict[str, Any]:
    deltas = metric_deltas(candidate_compact, control_compact)
    source_values = [
        float(metrics["balanced_accuracy"])
        for metrics in source_deltas.values()
        if metrics.get("balanced_accuracy") is not None
    ]
    source_win_rate = (
        sum(value > 0.0 for value in source_values) / len(source_values)
        if source_values
        else 0.0
    )
    primary_gain = max(deltas[key] for key in PRIMARY_METRICS) >= min_primary_gain
    no_primary_regression = all(
        deltas[key] >= -max_other_primary_drop for key in PRIMARY_METRICS
    )
    checks = {
        "primary_gain": primary_gain,
        "no_other_primary_regression": no_primary_regression,
        "balanced_accuracy_source_win_rate": source_win_rate >= min_source_win_rate,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "deltas": deltas,
        "balanced_accuracy_source_win_rate": source_win_rate,
        "num_shared_fake_models": len(source_values),
        "per_fake_model_deltas": dict(source_deltas),
    }


def build_summary(
    base: Mapping[str, Any],
    detection_only: Mapping[str, Any],
    correct_camera: Mapping[str, Any],
    flipped_camera: Mapping[str, Any],
    *,
    min_coverage: float,
    min_format_valid: float,
    min_primary_gain: float,
    max_other_primary_drop: float,
    min_source_win_rate: float,
    min_class_recall: float,
) -> dict[str, Any]:
    raw_models = {
        "base_detection_checkpoint": base,
        "detection_only_control": detection_only,
        "correct_camera_auxiliary": correct_camera,
        "flipped_camera_auxiliary": flipped_camera,
    }
    models = {name: compact(payload) for name, payload in raw_models.items()}
    correct_vs_detection_sources = per_source_deltas(correct_camera, detection_only)
    correct_vs_flipped_sources = per_source_deltas(correct_camera, flipped_camera)
    correct_vs_detection = comparison(
        models["correct_camera_auxiliary"],
        models["detection_only_control"],
        correct_vs_detection_sources,
        min_primary_gain=min_primary_gain,
        max_other_primary_drop=max_other_primary_drop,
        min_source_win_rate=min_source_win_rate,
    )
    correct_vs_flipped = comparison(
        models["correct_camera_auxiliary"],
        models["flipped_camera_auxiliary"],
        correct_vs_flipped_sources,
        min_primary_gain=min_primary_gain,
        max_other_primary_drop=max_other_primary_drop,
        min_source_win_rate=min_source_win_rate,
    )
    all_models_valid = all(
        metrics["coverage"] >= min_coverage
        and metrics["format_valid_rate"] >= min_format_valid
        for metrics in models.values()
    )
    correct = models["correct_camera_auxiliary"]
    class_recall_not_collapsed = (
        correct["real_recall"] is not None
        and correct["fake_recall"] is not None
        and float(correct["real_recall"]) >= min_class_recall
        and float(correct["fake_recall"]) >= min_class_recall
    )
    checks = {
        "all_models_have_coverage_and_format": all_models_valid,
        "correct_camera_beats_detection_only": correct_vs_detection["status"] == "passed",
        "correct_camera_beats_flipped_camera": correct_vs_flipped["status"] == "passed",
        "correct_camera_has_no_class_recall_collapse": class_recall_not_collapsed,
    }
    passed = all(checks.values())
    return {
        "gate": "ViF-Bench 无相机文本四模型开发诊断",
        "status": "camera_candidate" if passed else "no_camera_gain",
        "what_was_tested": (
            "The original DataB detection checkpoint and three equal-step joint-SFT branches are "
            "evaluated on the same ViF-Bench frames with the original detection prompt. No camera "
            "caption or label is supplied at inference."
        ),
        "thresholds": {
            "min_coverage": min_coverage,
            "min_format_valid": min_format_valid,
            "min_balanced_accuracy_or_fake_f1_gain": min_primary_gain,
            "max_other_primary_metric_drop": max_other_primary_drop,
            "min_balanced_accuracy_source_win_rate": min_source_win_rate,
            "min_real_and_fake_recall": min_class_recall,
        },
        "checks": checks,
        "models": models,
        "comparisons": {
            "correct_minus_detection_only": correct_vs_detection,
            "correct_minus_flipped_camera": correct_vs_flipped,
            "correct_minus_base": {
                "deltas": metric_deltas(
                    models["correct_camera_auxiliary"],
                    models["base_detection_checkpoint"],
                )
            },
        },
        "does_not_establish": (
            "ViF-Bench has been repeatedly used during development. This result is a development "
            "diagnostic, not a pristine final-test claim and not a substitute for GenBuster-Bench "
            "or MintVid after the method is frozen."
        ),
        "next_action": (
            "Keep camera-conditioned detection as a candidate and run paired uncertainty analysis."
            if passed
            else "Stop the independent camera-VQA/detection interleaving recipe and redesign the task coupling."
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
    parser.add_argument("--min-format-valid", type=float, default=0.99)
    parser.add_argument("--min-primary-gain", type=float, default=0.01)
    parser.add_argument("--max-other-primary-drop", type=float, default=0.005)
    parser.add_argument("--min-source-win-rate", type=float, default=0.60)
    parser.add_argument("--min-class-recall", type=float, default=0.45)
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
        max_other_primary_drop=args.max_other_primary_drop,
        min_source_win_rate=args.min_source_win_rate,
        min_class_recall=args.min_class_recall,
    )
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
