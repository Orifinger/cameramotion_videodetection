#!/usr/bin/env python3
"""Compare the paired no-camera and predicted-camera ViF-Bench evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


PRIMARY_METRICS = ("balanced_accuracy", "real_recall", "fake_recall", "fake_f1")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation must be a JSON object: {path}")
    return payload


def mean_metric(evaluation: Mapping[str, Any], metric: str) -> float | None:
    values = []
    for item in evaluation.get("per_fake_model", {}).values():
        value = item.get(metric)
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def compact(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    average = evaluation.get("average_across_fake_models", {})
    metrics = {
        "balanced_accuracy": average.get("balanced_accuracy"),
        "real_recall": mean_metric(evaluation, "real_recall"),
        "fake_recall": average.get("fake_recall"),
        "fake_f1": average.get("fake_f1"),
    }
    return {
        "coverage": evaluation.get("coverage"),
        "format_valid_rate": evaluation.get("format_valid_rate"),
        "num_predictions": evaluation.get("num_predictions"),
        **metrics,
    }


def delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-camera-eval", type=Path, required=True)
    parser.add_argument("--with-camera-eval", type=Path, required=True)
    parser.add_argument("--camera-context-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--min-format-valid", type=float, default=0.99)
    args = parser.parse_args()

    no_camera_eval = read_json(args.no_camera_eval)
    with_camera_eval = read_json(args.with_camera_eval)
    context = read_json(args.camera_context_summary)
    no_camera = compact(no_camera_eval)
    with_camera = compact(with_camera_eval)
    deltas = {
        metric: delta(with_camera.get(metric), no_camera.get(metric))
        for metric in PRIMARY_METRICS
    }

    no_models = no_camera_eval.get("per_fake_model", {})
    camera_models = with_camera_eval.get("per_fake_model", {})
    common_models = sorted(set(no_models) & set(camera_models))
    per_model = {}
    balanced_wins = 0
    f1_wins = 0
    for model in common_models:
        model_deltas = {
            metric: delta(camera_models[model].get(metric), no_models[model].get(metric))
            for metric in PRIMARY_METRICS
        }
        per_model[model] = model_deltas
        balanced_wins += bool((model_deltas["balanced_accuracy"] or 0.0) > 0.0)
        f1_wins += bool((model_deltas["fake_f1"] or 0.0) > 0.0)

    engineering_checks = {
        "camera_context_full_coverage": float(context.get("coverage", 0.0)) >= 1.0,
        "no_camera_prediction_coverage": float(no_camera.get("coverage") or 0.0)
        >= args.min_coverage,
        "with_camera_prediction_coverage": float(with_camera.get("coverage") or 0.0)
        >= args.min_coverage,
        "no_camera_format_valid": float(no_camera.get("format_valid_rate") or 0.0)
        >= args.min_format_valid,
        "with_camera_format_valid": float(with_camera.get("format_valid_rate") or 0.0)
        >= args.min_format_valid,
        "same_fake_generator_set": set(no_models) == set(camera_models) and bool(no_models),
    }
    method_checks = {
        "balanced_accuracy_improves": deltas["balanced_accuracy"] is not None
        and deltas["balanced_accuracy"] > 0.0,
        "fake_f1_improves": deltas["fake_f1"] is not None and deltas["fake_f1"] > 0.0,
        "balanced_accuracy_wins_majority_of_generators": bool(common_models)
        and balanced_wins > len(common_models) / 2,
    }
    if not all(engineering_checks.values()):
        status = "invalid"
    elif all(method_checks.values()):
        status = "passed"
    else:
        status = "failed"

    output = {
        "gate": "DataB 显式 Camera labels+caption 的 ViF-Bench 配对检测比较",
        "status": status,
        "what_was_tested": (
            "两个模型从同一 Qwen3-VL-8B-Instruct 出发，在相同 5739 条 DataB 上使用相同训练配置；"
            "无 Camera 模型使用原检测提示词，Camera 模型仅追加冻结 CameraBench 模型预测的 labels+caption。"
        ),
        "camera_context_is_predicted_not_gold": True,
        "engineering_checks": engineering_checks,
        "method_checks": method_checks,
        "no_camera": no_camera,
        "with_predicted_camera": with_camera,
        "with_camera_minus_no_camera": deltas,
        "generator_comparison": {
            "num_common_generators": len(common_models),
            "balanced_accuracy_wins": balanced_wins,
            "fake_f1_wins": f1_wins,
            "per_generator_deltas": per_model,
        },
        "camera_context_summary": str(args.camera_context_summary),
        "does_not_establish": (
            "ViF-Bench 已用于项目开发；即使通过，也需要冻结方法后在独立 GenBuster benchmark 上确认。"
            "Camera caption 还包含场景语义，因此本轮不能把收益单独归因于纯相机运动标签。"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
