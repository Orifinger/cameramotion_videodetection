#!/usr/bin/env python3
"""Summarize Real/Fake transfer for the camera-intermediate GRPO controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.camera_joint_sft_gate.summarize_dataa import compact as compact_dataa
from scripts.camera_joint_sft_gate.summarize_vif_four_model import compact as compact_vif
from scripts.caspr_gate1.runtime import write_json


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(payload)


def deltas(candidate: Mapping[str, Any], control: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(control[key])
        for key in keys
        if candidate.get(key) is not None and control.get(key) is not None
    }


def dataa_beats(
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    min_gain: float,
    max_drop: float,
) -> tuple[bool, dict[str, float]]:
    result = deltas(
        candidate,
        control,
        ("balanced_accuracy", "fake_f1", "pair_accuracy", "format_valid_rate"),
    )
    passed = (
        max(result["balanced_accuracy"], result["pair_accuracy"]) >= min_gain
        and result["balanced_accuracy"] >= -max_drop
        and result["pair_accuracy"] >= -max_drop
        and result["fake_f1"] >= -max_drop
    )
    return passed, result


def dataa_retains(
    candidate: Mapping[str, Any], control: Mapping[str, Any], max_drop: float
) -> tuple[bool, dict[str, float]]:
    result = deltas(
        candidate,
        control,
        ("balanced_accuracy", "fake_f1", "pair_accuracy", "format_valid_rate"),
    )
    passed = all(
        result[key] >= -max_drop for key in ("balanced_accuracy", "fake_f1", "pair_accuracy")
    )
    return passed, result


def vif_beats(
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    min_gain: float,
    max_drop: float,
) -> tuple[bool, dict[str, float]]:
    result = deltas(
        candidate,
        control,
        ("balanced_accuracy", "fake_f1", "real_recall", "fake_recall", "format_valid_rate"),
    )
    passed = (
        max(result["balanced_accuracy"], result["fake_f1"]) >= min_gain
        and result["balanced_accuracy"] >= -max_drop
        and result["fake_f1"] >= -max_drop
    )
    return passed, result


def vif_retains(
    candidate: Mapping[str, Any], control: Mapping[str, Any], max_drop: float
) -> tuple[bool, dict[str, float]]:
    result = deltas(
        candidate,
        control,
        ("balanced_accuracy", "fake_f1", "real_recall", "fake_recall", "format_valid_rate"),
    )
    passed = all(result[key] >= -max_drop for key in ("balanced_accuracy", "fake_f1"))
    return passed, result


def build_dataa_summary(
    warm_payload: Mapping[str, Any],
    correct_payload: Mapping[str, Any],
    detection_only_payload: Mapping[str, Any],
    shuffled_payload: Mapping[str, Any] | None,
    *,
    min_coverage: float,
    min_format: float,
    min_gain: float,
    max_drop: float,
) -> dict[str, Any]:
    models = {
        "共同联合输出起点": compact_dataa(warm_payload),
        "正确相机联合奖励": compact_dataa(correct_payload),
        "仅检测奖励对照": compact_dataa(detection_only_payload),
    }
    if shuffled_payload is not None:
        models["打乱相机联合奖励对照"] = compact_dataa(shuffled_payload)
    correct = models["正确相机联合奖励"]
    pass_detection, versus_detection = dataa_beats(
        correct, models["仅检测奖励对照"], min_gain, max_drop
    )
    pass_warm, versus_warm = dataa_retains(correct, models["共同联合输出起点"], max_drop)
    comparisons: dict[str, Any] = {
        "正确相机减仅检测奖励": versus_detection,
        "正确相机减共同起点": versus_warm,
    }
    pass_shuffled = shuffled_payload is not None
    if shuffled_payload is not None:
        pass_shuffled, versus_shuffled = dataa_beats(
            correct, models["打乱相机联合奖励对照"], min_gain, max_drop
        )
        comparisons["正确相机减打乱相机"] = versus_shuffled
    checks = {
        "全部模型覆盖完整": all(model["coverage"] >= min_coverage for model in models.values()),
        "全部模型输出格式有效": all(
            model["format_valid_rate"] >= min_format for model in models.values()
        ),
        "正确相机优于仅检测奖励": pass_detection,
        "正确相机不劣于共同起点": pass_warm,
        "正确相机优于打乱相机": pass_shuffled,
    }
    return {
        "gate": "DataA 留出集上的检测主导相机中间变量联合 GRPO 门",
        "status": "passed" if all(checks.values()) else "failed",
        "what_was_tested": (
            "所有模型在同一 case-level 留出 DataA Real/Fake 对上使用相同联合提示词，"
            "推理不提供相机标签或 caption；只比较最终 Real/Fake 与成对检测指标。"
        ),
        "thresholds": {
            "min_coverage": min_coverage,
            "min_format_valid": min_format,
            "min_balanced_or_pair_gain": min_gain,
            "max_other_primary_drop": max_drop,
        },
        "checks": checks,
        "models": models,
        "comparisons": comparisons,
        "does_not_establish": (
            "DataA 是局部编辑开发门；通过后仍必须在 ViF-Bench 复核全生成视频迁移，"
            "冻结方法后再用 GenBuster benchmark 或 MintVid 做最终报告。"
        ),
        "next_action": (
            "记录局部编辑诊断结果，并继续运行三个分支的 ViF-Bench Real/Fake 主门。"
        ),
    }


def build_vif_summary(
    correct_warm_payload: Mapping[str, Any],
    correct_payload: Mapping[str, Any],
    detection_only_payload: Mapping[str, Any],
    shuffled_payload: Mapping[str, Any] | None,
    *,
    min_coverage: float,
    min_format: float,
    min_gain: float,
    max_drop: float,
) -> dict[str, Any]:
    models = {
        "共同联合输出起点": compact_vif(correct_warm_payload),
        "正确相机联合奖励": compact_vif(correct_payload),
        "仅检测奖励对照": compact_vif(detection_only_payload),
    }
    if shuffled_payload is not None:
        models["打乱相机联合奖励对照"] = compact_vif(shuffled_payload)
    correct = models["正确相机联合奖励"]
    pass_detection, versus_detection = vif_beats(
        correct, models["仅检测奖励对照"], min_gain, max_drop
    )
    pass_warm, versus_warm = vif_retains(correct, models["共同联合输出起点"], max_drop)
    comparisons: dict[str, Any] = {
        "正确相机减仅检测奖励": versus_detection,
        "正确相机减共同起点": versus_warm,
    }
    pass_shuffled = shuffled_payload is not None
    if shuffled_payload is not None:
        pass_shuffled, versus_shuffled = vif_beats(
            correct, models["打乱相机联合奖励对照"], min_gain, max_drop
        )
        comparisons["正确相机减打乱相机"] = versus_shuffled
    checks = {
        "全部模型覆盖完整": all(model["coverage"] >= min_coverage for model in models.values()),
        "全部模型输出格式有效": all(
            model["format_valid_rate"] >= min_format for model in models.values()
        ),
        "正确相机优于仅检测奖励": pass_detection,
        "正确相机不劣于共同起点": pass_warm,
        "正确相机优于打乱相机": pass_shuffled,
    }
    return {
        "gate": "ViF-Bench 上的检测主导相机中间变量联合 GRPO 门",
        "status": "camera_candidate" if all(checks.values()) else "no_camera_gain",
        "what_was_tested": (
            "共同起点和三个等数据、等步数 GRPO 分支在同一 ViF-Bench 上使用相同联合提示词；"
            "不输入相机标签或 caption，主指标是 Real/Fake balanced accuracy 与 fake F1。"
        ),
        "thresholds": {
            "min_coverage": min_coverage,
            "min_format_valid": min_format,
            "min_balanced_accuracy_or_fake_f1_gain": min_gain,
            "max_other_primary_drop": max_drop,
        },
        "checks": checks,
        "models": models,
        "comparisons": comparisons,
        "does_not_establish": (
            "ViF-Bench 已用于开发选择，不是最终未见测试；只有这里通过才冻结方法并转到"
            " GenBuster benchmark 或 MintVid。"
        ),
        "next_action": (
            "冻结训练配方，转到未用于选模的通用 benchmark。"
            if all(checks.values())
            else "停止当前联合奖励配方，不用相机 VQA 分数替代检测失败。"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("dataa", "vif"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--warm-eval", required=True)
        sub.add_argument("--correct-eval", required=True)
        sub.add_argument("--detection-only-eval", required=True)
        sub.add_argument("--shuffled-eval")
        sub.add_argument("--output-json", required=True)
        sub.add_argument("--min-coverage", type=float, default=0.99)
        sub.add_argument("--min-format-valid", type=float, default=0.95 if name == "dataa" else 0.99)
        sub.add_argument("--min-gain", type=float, default=0.02 if name == "dataa" else 0.01)
        sub.add_argument("--max-drop", type=float, default=0.01 if name == "dataa" else 0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = {
        "warm_payload": read_json(args.warm_eval),
        "correct_payload": read_json(args.correct_eval),
        "detection_only_payload": read_json(args.detection_only_eval),
        "shuffled_payload": read_json(args.shuffled_eval) if args.shuffled_eval else None,
        "min_coverage": args.min_coverage,
        "min_format": args.min_format_valid,
        "min_gain": args.min_gain,
        "max_drop": args.max_drop,
    }
    if args.command == "dataa":
        summary = build_dataa_summary(**payloads)
    else:
        summary = build_vif_summary(
            correct_warm_payload=payloads.pop("warm_payload"),
            **payloads,
        )
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
