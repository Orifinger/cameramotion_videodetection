#!/usr/bin/env python3
"""Diagnose whether binary camera experts exhibit the intended route crossover."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.camera_detection_retention.vifbench_retention import (
    VALID_ANSWERS,
    answer_label,
    load_index,
    load_predictions,
)
from scripts.camera_hard_route_gate.route_manifest import (
    BINARY_ROUTE_BUCKETS,
    read_jsonl,
    write_json,
)


EXPERTS = ("no-motion", "motion")


def prediction_map(path: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows, audit = load_predictions(path)
    return {str(row["video_id"]): row for row in rows}, audit


def gold_label(source: str) -> str:
    return "real" if source.casefold() == "real" else "fake"


def video_metrics(
    video_ids: Sequence[str],
    predictions: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    counts = {
        "real": Counter(),
        "fake": Counter(),
    }
    valid = 0
    predicted_fake = 0
    correct = 0
    for video_id in video_ids:
        gold = gold_label(str(expected[video_id]["aigc_model_name"]))
        predicted = answer_label(predictions[video_id])
        if predicted in VALID_ANSWERS:
            valid += 1
        if predicted == "fake":
            predicted_fake += 1
        counts[gold][predicted] += 1
        correct += predicted == gold

    real_total = sum(counts["real"].values())
    fake_total = sum(counts["fake"].values())
    real_correct = counts["real"]["real"]
    fake_correct = counts["fake"]["fake"]
    real_recall = real_correct / real_total if real_total else None
    fake_recall = fake_correct / fake_total if fake_total else None
    balanced_accuracy = (
        (real_recall + fake_recall) / 2.0
        if real_recall is not None and fake_recall is not None
        else None
    )
    false_positive = counts["real"]["fake"]
    fake_precision_denominator = fake_correct + false_positive
    fake_precision = (
        fake_correct / fake_precision_denominator if fake_precision_denominator else 0.0
    )
    fake_f1_denominator = fake_precision + (fake_recall or 0.0)
    fake_f1 = (
        2.0 * fake_precision * fake_recall / fake_f1_denominator
        if fake_recall is not None and fake_f1_denominator
        else 0.0
    )
    total = len(video_ids)
    return {
        "num_videos": total,
        "num_real": real_total,
        "num_fake": fake_total,
        "coverage": len(predictions) / len(expected) if expected else 0.0,
        "format_valid_rate": valid / total if total else None,
        "accuracy": correct / total if total else None,
        "balanced_accuracy": balanced_accuracy,
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "fake_precision": fake_precision,
        "fake_f1": fake_f1,
        "predicted_fake_rate": predicted_fake / total if total else None,
        "confusion": {
            "real_as_real": real_correct,
            "real_as_fake": counts["real"]["fake"],
            "real_invalid": real_total - real_correct - counts["real"]["fake"],
            "fake_as_fake": fake_correct,
            "fake_as_real": counts["fake"]["real"],
            "fake_invalid": fake_total - fake_correct - counts["fake"]["real"],
        },
    }


def exact_binomial_two_sided(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if discordant == 0:
        return None
    lower = min(left_only, right_only)
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(index + 1)
        - math.lgamma(discordant - index + 1)
        - discordant * math.log(2.0)
        for index in range(lower + 1)
    ]
    maximum = max(log_terms)
    log_cdf = maximum + math.log(sum(math.exp(value - maximum) for value in log_terms))
    return min(1.0, 2.0 * math.exp(log_cdf))


def paired_correctness(
    video_ids: Sequence[str],
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    counts = Counter()
    for video_id in video_ids:
        gold = gold_label(str(expected[video_id]["aigc_model_name"]))
        left_correct = answer_label(left[video_id]) == gold
        right_correct = answer_label(right[video_id]) == gold
        if left_correct and right_correct:
            counts["both_correct"] += 1
        elif left_correct:
            counts["left_only_correct"] += 1
        elif right_correct:
            counts["right_only_correct"] += 1
        else:
            counts["both_wrong"] += 1
    total = len(video_ids)
    left_accuracy = (counts["both_correct"] + counts["left_only_correct"]) / total
    right_accuracy = (counts["both_correct"] + counts["right_only_correct"]) / total
    return {
        **dict(counts),
        "left_accuracy": left_accuracy,
        "right_accuracy": right_accuracy,
        "left_minus_right_accuracy": left_accuracy - right_accuracy,
        "mcnemar_exact_two_sided_p": exact_binomial_two_sided(
            counts["left_only_correct"], counts["right_only_correct"]
        ),
    }


def per_generator_balanced_accuracy(
    route_video_ids: Sequence[str],
    predictions: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    real_ids = [
        video_id
        for video_id in route_video_ids
        if str(expected[video_id]["aigc_model_name"]).casefold() == "real"
    ]
    real_metrics = video_metrics(real_ids, predictions, expected)
    real_recall = real_metrics["real_recall"]
    fake_ids_by_source: dict[str, list[str]] = defaultdict(list)
    for video_id in route_video_ids:
        source = str(expected[video_id]["aigc_model_name"])
        if source.casefold() != "real":
            fake_ids_by_source[source].append(video_id)

    output: dict[str, dict[str, Any]] = {}
    for source, fake_ids in sorted(fake_ids_by_source.items()):
        fake_correct = sum(answer_label(predictions[video_id]) == "fake" for video_id in fake_ids)
        fake_recall = fake_correct / len(fake_ids)
        output[source] = {
            "num_fake": len(fake_ids),
            "num_real": len(real_ids),
            "real_recall": real_recall,
            "fake_recall": fake_recall,
            "balanced_accuracy": (
                (real_recall + fake_recall) / 2.0 if real_recall is not None else None
            ),
        }
    return output


def winner_counts(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    counts = Counter()
    for source in sorted(set(left) & set(right)):
        left_value = left[source].get("balanced_accuracy")
        right_value = right[source].get("balanced_accuracy")
        if left_value is None or right_value is None:
            counts["unsupported"] += 1
        elif left_value > right_value:
            counts["no_motion_expert_wins"] += 1
        elif right_value > left_value:
            counts["motion_expert_wins"] += 1
        else:
            counts["ties"] += 1
    return dict(counts)


def classify_pattern(no_route_advantage: float, motion_route_advantage: float) -> str:
    tolerance = 1e-12
    if no_route_advantage > tolerance and motion_route_advantage > tolerance:
        return "semantic_crossover"
    if no_route_advantage <= tolerance and motion_route_advantage >= -tolerance:
        return "motion_expert_dominates_or_no_motion_expert_is_weaker"
    if no_route_advantage >= -tolerance and motion_route_advantage <= tolerance:
        return "no_motion_expert_dominates_or_motion_expert_is_weaker"
    return "experts_are_semantically_reversed"


def diagnose(args: argparse.Namespace) -> None:
    index = load_index(args.index_dir, args.expected_ranks, False)
    expected = index["expected"]
    route_rows = read_jsonl(args.route_manifest)
    route_by_id = {str(row.get("video_id", "")): row for row in route_rows}
    if len(route_by_id) != len(route_rows) or "" in route_by_id:
        raise ValueError("binary route manifest video_id values must be unique and non-empty")
    if set(route_by_id) != set(expected):
        raise ValueError(
            f"route/index mismatch: missing={len(set(expected) - set(route_by_id))}, "
            f"unexpected={len(set(route_by_id) - set(expected))}"
        )

    prediction_dirs = {
        "no-motion": args.no_motion_prediction_dir,
        "motion": args.motion_prediction_dir,
    }
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    prediction_audits: dict[str, Any] = {}
    for expert, path in prediction_dirs.items():
        predictions[expert], prediction_audits[expert] = prediction_map(path)
        if set(predictions[expert]) != set(expected):
            raise ValueError(
                f"{expert} prediction coverage mismatch: "
                f"missing={len(set(expected) - set(predictions[expert]))}, "
                f"unexpected={len(set(predictions[expert]) - set(expected))}"
            )

    subsets = {
        "all": sorted(expected),
        "route_no_motion": sorted(
            video_id
            for video_id, row in route_by_id.items()
            if row.get("binary_route_bucket") == "no-motion"
        ),
        "route_motion": sorted(
            video_id
            for video_id, row in route_by_id.items()
            if row.get("binary_route_bucket") == "motion"
        ),
    }
    if not subsets["route_no_motion"] or not subsets["route_motion"]:
        raise ValueError("binary route diagnostic requires both route subsets")

    metrics: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    generator_metrics: dict[str, Any] = {}
    for subset_name, video_ids in subsets.items():
        metrics[subset_name] = {
            expert: video_metrics(video_ids, predictions[expert], expected) for expert in EXPERTS
        }
        comparisons[subset_name] = paired_correctness(
            video_ids,
            predictions["no-motion"],
            predictions["motion"],
            expected,
        )
        per_generator = {
            expert: per_generator_balanced_accuracy(
                video_ids, predictions[expert], expected
            )
            for expert in EXPERTS
        }
        generator_metrics[subset_name] = {
            "experts": per_generator,
            "winner_counts": winner_counts(
                per_generator["no-motion"], per_generator["motion"]
            ),
        }

    no_route_advantage = (
        metrics["route_no_motion"]["no-motion"]["balanced_accuracy"]
        - metrics["route_no_motion"]["motion"]["balanced_accuracy"]
    )
    motion_route_advantage = (
        metrics["route_motion"]["motion"]["balanced_accuracy"]
        - metrics["route_motion"]["no-motion"]["balanced_accuracy"]
    )
    pattern = classify_pattern(no_route_advantage, motion_route_advantage)
    compact = {
        "diagnostic": "ViF binary camera expert crossover",
        "status": "completed",
        "pattern": pattern,
        "route_support": {
            name: {
                "num_videos": len(video_ids),
                "num_real": sum(
                    str(expected[video_id]["aigc_model_name"]).casefold() == "real"
                    for video_id in video_ids
                ),
                "num_fake": sum(
                    str(expected[video_id]["aigc_model_name"]).casefold() != "real"
                    for video_id in video_ids
                ),
            }
            for name, video_ids in subsets.items()
        },
        "balanced_accuracy": {
            subset: {
                expert: metrics[subset][expert]["balanced_accuracy"] for expert in EXPERTS
            }
            for subset in subsets
        },
        "real_recall": {
            subset: {
                expert: metrics[subset][expert]["real_recall"] for expert in EXPERTS
            }
            for subset in subsets
        },
        "fake_recall": {
            subset: {
                expert: metrics[subset][expert]["fake_recall"] for expert in EXPERTS
            }
            for subset in subsets
        },
        "predicted_fake_rate": {
            subset: {
                expert: metrics[subset][expert]["predicted_fake_rate"] for expert in EXPERTS
            }
            for subset in subsets
        },
        "intended_route_advantages": {
            "no_motion_expert_minus_motion_expert_on_no_motion_route": no_route_advantage,
            "motion_expert_minus_no_motion_expert_on_motion_route": motion_route_advantage,
        },
        "paired_correctness": comparisons,
        "generator_winner_counts": {
            subset: generator_metrics[subset]["winner_counts"] for subset in subsets
        },
        "interpretation": {
            "semantic_crossover": (
                "Each expert is stronger on its intended route subset; investigate router transfer "
                "and composition rather than global expert quality."
            ),
            "motion_expert_dominates_or_no_motion_expert_is_weaker": (
                "The motion expert is at least as strong on both route subsets; data volume or "
                "optimization imbalance likely dominates camera semantics."
            ),
            "no_motion_expert_dominates_or_motion_expert_is_weaker": (
                "The no-motion expert is at least as strong on both route subsets; data volume or "
                "optimization imbalance likely dominates camera semantics."
            ),
            "experts_are_semantically_reversed": (
                "Each expert is stronger on the opposite route subset; inspect label contracts, "
                "domain shift, and whether the chosen camera partition is anti-correlated with "
                "useful detection specialization."
            ),
        }[pattern],
        "does_not_establish": (
            "This is an offline failure diagnosis using already observed VIF predictions. It cannot "
            "rescue the failed hard-route method or justify relabeling the swapped control."
        ),
    }
    full = {
        **compact,
        "index_dir": str(args.index_dir),
        "route_manifest": str(args.route_manifest),
        "prediction_audits": prediction_audits,
        "metrics": metrics,
        "generator_metrics": generator_metrics,
    }
    write_json(args.output_json, full)
    write_json(args.output_compact_json, compact)
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"\nFull result: {args.output_json}")
    print(f"Compact result: {args.output_compact_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--route-manifest", required=True)
    parser.add_argument("--no-motion-prediction-dir", required=True)
    parser.add_argument("--motion-prediction-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-compact-json", required=True)
    parser.add_argument("--expected-ranks", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    diagnose(parse_args())


if __name__ == "__main__":
    main()
