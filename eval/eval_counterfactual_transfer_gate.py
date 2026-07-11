#!/usr/bin/env python3
"""Gate 2: judge DataA transfer, VIF retention, and camera contribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_metric(value: Any) -> float | None:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    return metric / 100.0 if metric > 1.0 else metric


def recursive_find(payload: Any, aliases: Sequence[str]) -> float | None:
    normalized_aliases = {alias.casefold().replace("-", "_").replace(" ", "_") for alias in aliases}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).casefold().replace("-", "_").replace(" ", "_")
            if normalized in normalized_aliases:
                metric = normalize_metric(value)
                if metric is not None:
                    return metric
        preferred = ["basic", "overall", "metrics", "average", "avg"]
        for key in preferred:
            if key in payload:
                found = recursive_find(payload[key], aliases)
                if found is not None:
                    return found
        for value in payload.values():
            if isinstance(value, (Mapping, list)):
                found = recursive_find(value, aliases)
                if found is not None:
                    return found
    elif isinstance(payload, list):
        for item in payload:
            found = recursive_find(item, aliases)
            if found is not None:
                return found
    return None


def detection_metrics(path: str | Path) -> dict[str, float]:
    payload = read_json(path)
    metrics = {
        "accuracy": recursive_find(payload, ("accuracy", "acc", "average_accuracy", "avg_acc")),
        "balanced_accuracy": recursive_find(payload, ("balanced_accuracy", "balanced_acc", "bacc")),
        "fake_f1": recursive_find(payload, ("fake_f1", "f1_fake", "f1")),
        "fake_recall": recursive_find(payload, ("fake_recall", "recall_fake", "recall")),
    }
    missing = [name for name, value in metrics.items() if value is None and name in {"balanced_accuracy", "fake_f1"}]
    if missing:
        raise ValueError(f"missing DataA metrics {missing} in {path}")
    return {name: float(value) for name, value in metrics.items() if value is not None}


def vif_metrics(path: str | None, acc: float | None, f1: float | None) -> dict[str, float]:
    payload = read_json(path) if path else {}
    resolved_acc = normalize_metric(acc) if acc is not None else recursive_find(
        payload, ("average_accuracy", "avg_acc", "accuracy", "acc")
    )
    resolved_f1 = normalize_metric(f1) if f1 is not None else recursive_find(
        payload, ("average_f1", "avg_f1", "fake_f1", "f1")
    )
    if resolved_acc is None or resolved_f1 is None:
        raise ValueError("VIF accuracy/F1 missing; provide summary JSON or direct --*-vif-acc/--*-vif-f1")
    return {"accuracy": resolved_acc, "f1": resolved_f1}


def motion_metrics(path: str | Path) -> dict[str, dict[str, float]]:
    payload = read_json(path)
    groups = payload.get("by_motion_bucket") if isinstance(payload, Mapping) else None
    if not isinstance(groups, Mapping):
        raise ValueError(f"missing by_motion_bucket in {path}")
    output = {}
    for bucket, values in groups.items():
        if not isinstance(values, Mapping):
            continue
        output[str(bucket)] = {
            "balanced_accuracy": float(normalize_metric(values.get("balanced_accuracy")) or 0.0),
            "fake_f1": float(normalize_metric(values.get("fake_f1")) or 0.0),
            "num_samples": float(values.get("num_samples") or 0.0),
        }
    return output


def delta(candidate: Mapping[str, float], control: Mapping[str, float], key: str) -> float:
    return float(candidate.get(key, 0.0)) - float(control.get(key, 0.0))


def detection_gain(candidate: Mapping[str, float], control: Mapping[str, float]) -> dict[str, float]:
    return {
        "balanced_accuracy": delta(candidate, control, "balanced_accuracy"),
        "fake_f1": delta(candidate, control, "fake_f1"),
        "accuracy": delta(candidate, control, "accuracy"),
    }


def vif_change(candidate: Mapping[str, float], control: Mapping[str, float]) -> dict[str, float]:
    return {
        "accuracy": delta(candidate, control, "accuracy"),
        "f1": delta(candidate, control, "f1"),
    }


def motion_gain(candidate, control) -> dict[str, dict[str, float]]:
    output = {}
    for bucket in sorted(set(candidate) | set(control)):
        output[bucket] = {
            "balanced_accuracy": float(candidate.get(bucket, {}).get("balanced_accuracy", 0.0))
            - float(control.get(bucket, {}).get("balanced_accuracy", 0.0)),
            "fake_f1": float(candidate.get(bucket, {}).get("fake_f1", 0.0))
            - float(control.get(bucket, {}).get("fake_f1", 0.0)),
        }
    return output


def branch_gate(
    name: str,
    candidate_dataa,
    control_dataa,
    candidate_vif,
    control_vif,
    candidate_motion,
    control_motion,
    min_dataa_gain: float,
    max_vif_drop: float,
    min_motion_gain: float,
) -> dict[str, Any]:
    dataa = detection_gain(candidate_dataa, control_dataa)
    vif = vif_change(candidate_vif, control_vif)
    motion = motion_gain(candidate_motion, control_motion)
    moving_gains = [
        value
        for bucket in ("minor-motion", "complex-motion")
        for value in (
            motion.get(bucket, {}).get("balanced_accuracy", -1.0),
            motion.get(bucket, {}).get("fake_f1", -1.0),
        )
    ]
    checks = {
        "dataa_gain": max(dataa["balanced_accuracy"], dataa["fake_f1"]) >= min_dataa_gain,
        "vif_accuracy_retention": vif["accuracy"] >= -max_vif_drop,
        "vif_f1_retention": vif["f1"] >= -max_vif_drop,
        "moving_bucket_gain": max(moving_gains, default=-1.0) >= min_motion_gain,
    }
    return {
        "name": name,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "dataa_gain": dataa,
        "vif_change": vif,
        "motion_bucket_gain": motion,
    }


def add_branch_args(parser: argparse.ArgumentParser, prefix: str, required: bool) -> None:
    parser.add_argument(f"--{prefix}-dataa-summary", required=required)
    parser.add_argument(f"--{prefix}-motion-summary", required=required)
    parser.add_argument(f"--{prefix}-vif-summary")
    parser.add_argument(f"--{prefix}-vif-acc", type=float)
    parser.add_argument(f"--{prefix}-vif-f1", type=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_branch_args(parser, "control", True)
    add_branch_args(parser, "pair", True)
    add_branch_args(parser, "camera", False)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-pair-dataa-gain", type=float, default=0.03)
    parser.add_argument("--min-camera-extra-gain", type=float, default=0.01)
    parser.add_argument("--max-vif-drop", type=float, default=0.01)
    parser.add_argument("--min-moving-bucket-gain", type=float, default=0.01)
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser.parse_args()


def load_branch(args: argparse.Namespace, prefix: str) -> dict[str, Any] | None:
    dataa_path = getattr(args, f"{prefix}_dataa_summary")
    motion_path = getattr(args, f"{prefix}_motion_summary")
    if not dataa_path and not motion_path:
        return None
    if not dataa_path or not motion_path:
        raise ValueError(f"{prefix} requires both DataA and motion summaries")
    return {
        "dataa": detection_metrics(dataa_path),
        "motion": motion_metrics(motion_path),
        "vif": vif_metrics(
            getattr(args, f"{prefix}_vif_summary"),
            getattr(args, f"{prefix}_vif_acc"),
            getattr(args, f"{prefix}_vif_f1"),
        ),
    }


def main() -> None:
    args = parse_args()
    control = load_branch(args, "control")
    pair = load_branch(args, "pair")
    camera = load_branch(args, "camera")
    assert control is not None and pair is not None
    pair_gate = branch_gate(
        "pair-only vs same-step detection replay control",
        pair["dataa"], control["dataa"], pair["vif"], control["vif"],
        pair["motion"], control["motion"], args.min_pair_dataa_gain,
        args.max_vif_drop, args.min_moving_bucket_gain,
    )
    camera_gate = None
    if camera is not None:
        camera_gate = branch_gate(
            "camera+pair vs pair-only",
            camera["dataa"], pair["dataa"], camera["vif"], pair["vif"],
            camera["motion"], pair["motion"], args.min_camera_extra_gain,
            args.max_vif_drop, args.min_moving_bucket_gain,
        )
    passed = pair_gate["status"] == "passed" and (
        camera_gate is None or camera_gate["status"] == "passed"
    )
    summary = {
        "gate": "Gate 2 - detection transfer and retention",
        "status": "passed" if passed else "failed",
        "interpretation": (
            "camera is supported as a core contribution"
            if camera_gate and camera_gate["status"] == "passed"
            else "pair learning passed but camera is not yet established"
            if pair_gate["status"] == "passed"
            else "counterfactual pretext has not transferred to detection"
        ),
        "thresholds": {
            "min_pair_dataa_gain": args.min_pair_dataa_gain,
            "min_camera_extra_gain": args.min_camera_extra_gain,
            "max_vif_drop": args.max_vif_drop,
            "min_moving_bucket_gain": args.min_moving_bucket_gain,
        },
        "metrics": {"control": control, "pair": pair, "camera": camera},
        "pair_transfer_gate": pair_gate,
        "camera_contribution_gate": camera_gate,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_gate and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
