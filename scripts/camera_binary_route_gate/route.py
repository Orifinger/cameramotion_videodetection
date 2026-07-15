#!/usr/bin/env python3
"""Map, compose, and evaluate frozen binary camera routes on VIF-Bench."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.camera_detection_retention.vifbench_retention import (
    evaluate_predictions,
    load_index,
    load_predictions,
)
from scripts.camera_hard_route_gate.route_manifest import (
    BINARY_ROUTE_BUCKETS,
    BINARY_WRONG_ROUTE,
    ROUTE_BUCKETS,
    binary_bucket,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


def vif_authenticity_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups = {"real": [], "fake": []}
    real_by_base: dict[str, str] = {}
    fake_rows: list[tuple[str, str]] = []
    for row in rows:
        source = str(row.get("aigc_model_name", ""))
        key = "real" if source.casefold() == "real" else "fake"
        route = str(row["binary_predicted_bucket"])
        groups[key].append(route)
        video_id = str(row.get("video_id", ""))
        if "/" not in video_id:
            continue
        _, base_id = video_id.split("/", 1)
        if key == "real":
            real_by_base[base_id] = route
        else:
            fake_rows.append((base_id, route))

    counts: dict[str, dict[str, int]] = {}
    distributions: dict[str, dict[str, float]] = {}
    for key, values in groups.items():
        counter = Counter(values)
        counts[key] = {bucket: counter[bucket] for bucket in BINARY_ROUTE_BUCKETS}
        distributions[key] = {
            bucket: counter[bucket] / len(values) if values else 0.0
            for bucket in BINARY_ROUTE_BUCKETS
        }
    total_variation = 0.5 * sum(
        abs(distributions["real"][bucket] - distributions["fake"][bucket])
        for bucket in BINARY_ROUTE_BUCKETS
    )
    pairs = [(real_by_base[base_id], route) for base_id, route in fake_rows if base_id in real_by_base]
    return {
        "route_counts_by_authenticity": counts,
        "route_distributions_by_authenticity": distributions,
        "real_fake_route_distribution_total_variation": total_variation,
        "num_real_fake_pairs": len(pairs),
        "paired_real_fake_same_route_rate": (
            sum(real == fake for real, fake in pairs) / len(pairs) if pairs else None
        ),
        "shortcut_warning": (
            "A large Real/Fake route-distribution gap means route alone correlates with authenticity. "
            "It must not be described as camera-conditioned artifact reasoning."
        ),
    }


def map_manifest(args: argparse.Namespace) -> None:
    source_rows = read_jsonl(args.input_manifest)
    if not source_rows:
        raise ValueError("binary VIF route mapping requires a non-empty three-class manifest")
    video_ids = [str(row.get("video_id", "")) for row in source_rows]
    if any(not value for value in video_ids) or len(video_ids) != len(set(video_ids)):
        raise ValueError("three-class VIF route video_id values must be unique and non-empty")

    rows: list[dict[str, Any]] = []
    for source in source_rows:
        predicted = str(source.get("predicted_bucket", ""))
        if predicted not in ROUTE_BUCKETS:
            raise ValueError(f"invalid frozen three-class route for {source.get('video_id')}: {predicted}")
        row = copy.deepcopy(source)
        binary = binary_bucket(predicted)
        row["binary_predicted_bucket"] = binary
        row["binary_route_bucket"] = binary
        row["binary_wrong_route_bucket"] = BINARY_WRONG_ROUTE[binary]
        row["binary_mapping"] = "no-motion_vs_minor-plus-complex-motion"
        rows.append(row)

    route_counts = Counter(str(row["binary_route_bucket"]) for row in rows)
    if set(route_counts) != set(BINARY_ROUTE_BUCKETS):
        raise ValueError(f"VIF binary route collapsed to fewer than two experts: {dict(route_counts)}")
    summary = {
        "schema_version": "vifbench_binary_camera_route_manifest_v1",
        "status": "completed",
        "input_manifest": str(args.input_manifest),
        "output_manifest": str(args.output_manifest),
        "num_videos": len(rows),
        "mapping": {
            "no-motion": ["no-motion"],
            "motion": ["minor-motion", "complex-motion"],
            "wrong_route_control": "swap no-motion and motion",
        },
        "route_counts": dict(route_counts),
        "route_distribution": {
            bucket: route_counts[bucket] / len(rows) for bucket in BINARY_ROUTE_BUCKETS
        },
        "authenticity_route_audit": vif_authenticity_audit(rows),
        "camera_text_enters_detection_prompt": False,
        "mapping_frozen_before_vif_detection_results": True,
    }
    write_jsonl(args.output_manifest, rows)
    write_json(args.output_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def prediction_map(path: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows, audit = load_predictions(path)
    return {str(row["video_id"]): row for row in rows}, audit


def parse_named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, path


def compose_predictions(args: argparse.Namespace) -> None:
    index = load_index(args.index_dir, args.expected_ranks, False)
    expected = index["expected"]
    manifest_rows = read_jsonl(args.route_manifest)
    manifest = {str(row.get("video_id")): row for row in manifest_rows}
    if len(manifest) != len(manifest_rows) or "" in manifest:
        raise ValueError("binary route manifest video_id values must be unique and non-empty")
    if set(manifest) != set(expected):
        raise ValueError(
            f"binary route/index mismatch: missing={len(set(expected) - set(manifest))}, "
            f"unexpected={len(set(manifest) - set(expected))}"
        )

    expert_specs = dict(args.expert)
    if set(expert_specs) != set(BINARY_ROUTE_BUCKETS):
        raise ValueError(f"binary expert paths must be exactly {sorted(BINARY_ROUTE_BUCKETS)}")
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    audits: dict[str, Any] = {}
    for name, path in sorted(expert_specs.items()):
        predictions[name], audits[name] = prediction_map(path)
    if args.shared_prediction_dir:
        predictions["shared"], audits["shared"] = prediction_map(args.shared_prediction_dir)

    selected_rows: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for video_id in sorted(expected):
        route = manifest[video_id]
        if args.route_mode == "predicted":
            expert = str(route["binary_route_bucket"])
        elif args.route_mode == "wrong":
            expert = str(route["binary_wrong_route_bucket"])
        elif args.route_mode == "shared":
            expert = "shared"
        else:
            raise ValueError(f"unsupported binary route mode: {args.route_mode}")
        if expert not in predictions:
            raise ValueError(f"selected binary expert {expert!r} has no prediction directory")
        source_row = predictions[expert].get(video_id)
        if source_row is None:
            raise ValueError(f"missing {expert} prediction for {video_id}")
        row = copy.deepcopy(source_row)
        row["binary_route_condition"] = args.route_mode
        row["binary_route_selected_expert"] = expert
        row["binary_route_predicted_bucket"] = route["binary_predicted_bucket"]
        row["binary_route_three_class_top1"] = route["predicted_bucket"]
        selected_rows.append(row)
        selected_counts[expert] += 1

    evaluation = evaluate_predictions(selected_rows, expected)
    write_json(args.output_predictions, selected_rows)
    summary = {
        "schema_version": "vifbench_binary_camera_route_composition_v1",
        "route_mode": args.route_mode,
        "route_manifest": str(args.route_manifest),
        "selected_expert_counts": dict(selected_counts),
        "prediction_audits": audits,
        "evaluation": evaluation,
        "output_predictions": str(args.output_predictions),
        "camera_text_enters_detection_prompt": False,
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def evaluation_payload(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    evaluation = summary.get("evaluation")
    return evaluation if isinstance(evaluation, Mapping) else summary


def average_metrics(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    return evaluation_payload(summary)["average_across_fake_models"]


def metric_delta(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> float:
    return float(average_metrics(left)[key]) - float(average_metrics(right)[key])


def per_model_win_rate(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> float | None:
    left_models = evaluation_payload(left)["per_fake_model"]
    right_models = evaluation_payload(right)["per_fake_model"]
    common = sorted(set(left_models) & set(right_models))
    if not common:
        return None
    return sum(float(left_models[name][key]) > float(right_models[name][key]) for name in common) / len(common)


def summarize_gate(args: argparse.Namespace) -> None:
    base = read_json(args.base_eval)
    shared = read_json(args.shared_summary)
    predicted = read_json(args.predicted_summary)
    wrong = read_json(args.wrong_summary)
    route = read_json(args.route_summary)
    deltas = {
        "predicted_minus_base": {
            key: metric_delta(predicted, base, key)
            for key in ("balanced_accuracy", "fake_recall", "fake_f1")
        },
        "predicted_minus_shared": {
            key: metric_delta(predicted, shared, key)
            for key in ("balanced_accuracy", "fake_recall", "fake_f1")
        },
        "predicted_minus_wrong": {
            key: metric_delta(predicted, wrong, key)
            for key in ("balanced_accuracy", "fake_recall", "fake_f1")
        },
    }
    base_gain = deltas["predicted_minus_base"]
    shared_gain = deltas["predicted_minus_shared"]
    wrong_gain = deltas["predicted_minus_wrong"]
    checks = {
        "all_coverage_at_least_99pct": all(
            float(evaluation_payload(value)["coverage"]) >= 0.99
            for value in (base, shared, predicted, wrong)
        ),
        "all_format_valid_at_least_99pct": all(
            float(evaluation_payload(value)["format_valid_rate"]) >= 0.99
            for value in (base, shared, predicted, wrong)
        ),
        "predicted_route_beats_original_base": (
            max(base_gain["balanced_accuracy"], base_gain["fake_f1"]) >= args.min_base_gain
            and min(base_gain["balanced_accuracy"], base_gain["fake_f1"]) >= -args.max_other_drop
        ),
        "predicted_route_beats_shared": (
            max(shared_gain["balanced_accuracy"], shared_gain["fake_f1"])
            >= args.min_shared_gain
            and min(shared_gain["balanced_accuracy"], shared_gain["fake_f1"])
            >= -args.max_other_drop
        ),
        "predicted_route_beats_swapped_wrong_route": (
            max(wrong_gain["balanced_accuracy"], wrong_gain["fake_f1"]) >= args.min_wrong_gain
            and min(wrong_gain["balanced_accuracy"], wrong_gain["fake_f1"])
            >= -args.max_other_drop
        ),
    }
    passed = all(checks.values())
    output = {
        "gate": "ViF-Bench 静止/有运动二路硬路由检测专家验证",
        "status": "passed" if passed else "failed",
        "what_was_tested": (
            "The original detection checkpoint, an equal-data shared LoRA, a frozen frame-predicted "
            "binary camera route over two detection experts, and an expert-swapped wrong route use "
            "the same VIF-Bench frames and no-camera detection prompt."
        ),
        "thresholds": {
            "min_predicted_minus_original_base_primary_gain": args.min_base_gain,
            "min_predicted_minus_shared_primary_gain": args.min_shared_gain,
            "min_predicted_minus_wrong_route_primary_gain": args.min_wrong_gain,
            "max_drop_in_other_primary_metric": args.max_other_drop,
        },
        "checks": checks,
        "models": {
            "original_detection_base": base,
            "shared_control": shared,
            "predicted_binary_camera_route": predicted,
            "swapped_wrong_route": wrong,
        },
        "deltas": deltas,
        "generator_level_win_rates": {
            "predicted_vs_base_balanced_accuracy": per_model_win_rate(
                predicted, base, "balanced_accuracy"
            ),
            "predicted_vs_shared_balanced_accuracy": per_model_win_rate(
                predicted, shared, "balanced_accuracy"
            ),
            "predicted_vs_wrong_balanced_accuracy": per_model_win_rate(
                predicted, wrong, "balanced_accuracy"
            ),
        },
        "vif_route_audit": route,
        "does_not_establish": (
            "Passing is a VIF-Bench development gate, not final paper evidence. The frozen method still "
            "requires a zero-overlap GenBuster benchmark evaluation and other held-out tests."
        ),
        "next_action": (
            "If passed, freeze the method and evaluate external benchmarks. If failed, stop hard routing "
            "instead of tuning the route on VIF labels."
        ),
    }
    write_json(args.output_json, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    mapping = subparsers.add_parser("map-manifest")
    mapping.add_argument("--input-manifest", required=True)
    mapping.add_argument("--output-manifest", required=True)
    mapping.add_argument("--output-summary", required=True)
    mapping.set_defaults(handler=map_manifest)

    compose = subparsers.add_parser("compose")
    compose.add_argument("--index-dir", required=True)
    compose.add_argument("--route-manifest", required=True)
    compose.add_argument("--expert", action="append", type=parse_named_path, required=True)
    compose.add_argument("--shared-prediction-dir")
    compose.add_argument("--route-mode", choices=("predicted", "wrong", "shared"), required=True)
    compose.add_argument("--output-predictions", required=True)
    compose.add_argument("--output-summary", required=True)
    compose.add_argument("--expected-ranks", type=int, default=16)
    compose.set_defaults(handler=compose_predictions)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--base-eval", required=True)
    summarize.add_argument("--shared-summary", required=True)
    summarize.add_argument("--predicted-summary", required=True)
    summarize.add_argument("--wrong-summary", required=True)
    summarize.add_argument("--route-summary", required=True)
    summarize.add_argument("--output-json", required=True)
    summarize.add_argument("--min-base-gain", type=float, default=0.005)
    summarize.add_argument("--min-shared-gain", type=float, default=0.005)
    summarize.add_argument("--min-wrong-gain", type=float, default=0.01)
    summarize.add_argument("--max-other-drop", type=float, default=0.005)
    summarize.set_defaults(handler=summarize_gate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
