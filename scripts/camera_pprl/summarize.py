#!/usr/bin/env python3
"""Summarize camera retention and ViF transfer for the phase-level Camera-PPRL run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.camera_joint_sft_gate.summarize_vif_four_model import compact as compact_vif
from scripts.caspr_gate1.runtime import write_json


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(payload)


def compact_camera(payload: Mapping[str, Any]) -> dict[str, Any]:
    conditions = payload.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("camera evaluation lacks conditions")

    def condition(name: str) -> Mapping[str, Any]:
        value = conditions.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"camera evaluation lacks {name}")
        return value

    matched = condition("matched_frames")
    opposite = condition("opposite_frames")
    no_frames = condition("no_frames")
    matched_overall = matched.get("overall")
    matched_macro = matched.get("macro")
    if not isinstance(matched_overall, Mapping) or not isinstance(matched_macro, Mapping):
        raise ValueError("matched camera condition lacks overall or macro metrics")
    return {
        "coverage": float(matched.get("coverage", 0.0)),
        "num_supported_labels": int(matched.get("num_supported_labels", 0)),
        "balanced_accuracy": float(matched_overall.get("balanced_accuracy", 0.0)),
        "macro_average_precision": float(matched_macro.get("average_precision", 0.0)),
        "macro_roc_auc": float(matched_macro.get("roc_auc", 0.0)),
        "paired_question_accuracy": float(matched.get("paired_question_accuracy", 0.0)),
        "opposite_frames_balanced_accuracy": float(
            opposite.get("overall", {}).get("balanced_accuracy", 0.0)
        ),
        "no_frames_balanced_accuracy": float(
            no_frames.get("overall", {}).get("balanced_accuracy", 0.0)
        ),
    }


def deltas(candidate: Mapping[str, Any], base: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, float]:
    return {key: float(candidate[key]) - float(base[key]) for key in keys}


def vif_checks(candidate: Mapping[str, Any], base: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, float]]:
    delta = deltas(
        candidate,
        base,
        (
            "balanced_accuracy",
            "fake_f1",
            "real_recall",
            "fake_recall",
            "format_valid_rate",
        ),
    )
    checks = {
        "coverage_and_format": candidate["coverage"] >= 0.99
        and candidate["format_valid_rate"] >= 0.99,
        "primary_gain": max(delta["balanced_accuracy"], delta["fake_f1"]) >= 0.01,
        "no_other_primary_regression": min(delta["balanced_accuracy"], delta["fake_f1"]) >= -0.005,
        "no_class_recall_collapse": candidate["real_recall"] >= 0.45
        and candidate["fake_recall"] >= 0.45,
    }
    return checks, delta


def build_summary(
    warm_camera_raw: Mapping[str, Any],
    pprl_camera_raw: Mapping[str, Any],
    recovery_camera_raw: Mapping[str, Any],
    direct_base_raw: Mapping[str, Any],
    direct_pprl_raw: Mapping[str, Any],
    recovery_base_raw: Mapping[str, Any],
    recovery_model_raw: Mapping[str, Any],
) -> dict[str, Any]:
    camera = {
        "correct_camera_joint_sft": compact_camera(warm_camera_raw),
        "camera_pprl": compact_camera(pprl_camera_raw),
        "camera_pprl_then_detection_recovery": compact_camera(recovery_camera_raw),
    }
    vif = {
        "correct_camera_joint_sft": compact_vif(direct_base_raw),
        "camera_pprl": compact_vif(direct_pprl_raw),
        "camera_pprl_repeated_base": compact_vif(recovery_base_raw),
        "camera_pprl_then_detection_recovery": compact_vif(recovery_model_raw),
    }
    pprl_camera_delta = deltas(
        camera["camera_pprl"],
        camera["correct_camera_joint_sft"],
        ("balanced_accuracy", "macro_average_precision", "paired_question_accuracy"),
    )
    recovery_camera_delta = deltas(
        camera["camera_pprl_then_detection_recovery"],
        camera["camera_pprl"],
        ("balanced_accuracy", "macro_average_precision", "paired_question_accuracy"),
    )
    direct_checks, direct_delta = vif_checks(
        vif["camera_pprl"], vif["correct_camera_joint_sft"]
    )
    recovery_checks, recovery_delta = vif_checks(
        vif["camera_pprl_then_detection_recovery"], vif["camera_pprl_repeated_base"]
    )
    camera_checks = {
        "pprl_camera_coverage": camera["camera_pprl"]["coverage"] >= 0.99,
        "pprl_retains_camera_macro_ap": pprl_camera_delta["macro_average_precision"] >= -0.02,
        "pprl_depends_on_matched_frames": (
            camera["camera_pprl"]["balanced_accuracy"]
            - camera["camera_pprl"]["opposite_frames_balanced_accuracy"]
            >= 0.10
        ),
        "recovery_retains_camera_macro_ap": recovery_camera_delta["macro_average_precision"] >= -0.05,
    }
    direct_pass = all(direct_checks.values()) and all(
        camera_checks[key]
        for key in (
            "pprl_camera_coverage",
            "pprl_retains_camera_macro_ap",
            "pprl_depends_on_matched_frames",
        )
    )
    recovery_pass = all(recovery_checks.values()) and camera_checks[
        "recovery_retains_camera_macro_ap"
    ]
    if direct_pass:
        status = "direct_pprl_candidate"
        next_action = "补做正确标签与翻转标签 PPRL 消融，并在冻结方法后评测保留 benchmark。"
    elif recovery_pass:
        status = "recovery_candidate_needs_control"
        next_action = "补做从同一联合 SFT 起点直接进行等量 detection recovery 的控制分支。"
    else:
        status = "no_detection_transfer"
        next_action = "停止扩大当前二元相机 PPRL；检查逐生成器结果后再决定是否改为显式联合检测奖励。"
    return {
        "gate": "正确相机二元前置强化学习与检测恢复分阶段验证",
        "status": status,
        "what_was_tested": (
            "Starting from the validated correct-camera joint-SFT model, a balanced binary-camera "
            "GRPO phase is evaluated before any detection recovery. A separate low-strength DataB "
            "detection-replay branch is then evaluated without overwriting the direct PPRL result. "
            "All ViF detection inference uses the original no-camera detection prompt."
        ),
        "thresholds": {
            "camera_macro_ap_max_direct_drop": 0.02,
            "camera_macro_ap_max_recovery_drop": 0.05,
            "min_matched_minus_opposite_camera_balanced_accuracy": 0.10,
            "min_vif_balanced_accuracy_or_fake_f1_gain": 0.01,
            "max_other_vif_primary_drop": 0.005,
            "min_vif_real_and_fake_recall": 0.45,
        },
        "checks": {
            "camera": camera_checks,
            "direct_pprl_vif": direct_checks,
            "detection_recovery_vif": recovery_checks,
        },
        "camera_eval": camera,
        "camera_deltas": {
            "pprl_minus_joint_sft": pprl_camera_delta,
            "recovery_minus_pprl": recovery_camera_delta,
        },
        "vif_eval": vif,
        "vif_deltas": {
            "pprl_minus_joint_sft": direct_delta,
            "recovery_minus_pprl": recovery_delta,
        },
        "does_not_establish": (
            "ViF-Bench is a repeatedly inspected development benchmark. The recovery branch has extra "
            "detection SFT compute and cannot establish a camera-specific gain until an equal-compute "
            "recovery-only control is run. GenBuster-Bench and MintVid remain untouched final evaluations."
        ),
        "next_action": next_action,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-camera-eval", required=True)
    parser.add_argument("--pprl-camera-eval", required=True)
    parser.add_argument("--recovery-camera-eval", required=True)
    parser.add_argument("--direct-vif-base-eval", required=True)
    parser.add_argument("--direct-vif-pprl-eval", required=True)
    parser.add_argument("--recovery-vif-base-eval", required=True)
    parser.add_argument("--recovery-vif-model-eval", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        read_json(args.warm_camera_eval),
        read_json(args.pprl_camera_eval),
        read_json(args.recovery_camera_eval),
        read_json(args.direct_vif_base_eval),
        read_json(args.direct_vif_pprl_eval),
        read_json(args.recovery_vif_base_eval),
        read_json(args.recovery_vif_model_eval),
    )
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
