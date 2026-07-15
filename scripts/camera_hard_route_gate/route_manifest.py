#!/usr/bin/env python3
"""Build, aggregate, and apply three-class camera routes on VIF-Bench."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.camera_detection_retention.vifbench_retention import (
    evaluate_predictions,
    load_index,
    load_predictions,
)
from tools.build_camera_hard_route_gate import (
    CAMERA_SYSTEM_PROMPT,
    QUESTION_BY_LABEL,
    ROUTE_BUCKETS,
    camera_prompt,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CYCLIC_ROUTE = {
    "no-motion": "minor-motion",
    "minor-motion": "complex-motion",
    "complex-motion": "no-motion",
}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(dict(row))
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def natural_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name.casefold())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def list_frames(frame_dir: str | Path) -> list[str]:
    directory = Path(frame_dir)
    return [
        str(path)
        for path in sorted(
            (path for path in directory.iterdir() if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES),
            key=natural_key,
        )
    ]


def list_vif_protocol_frames(frame_dir: str | Path, require_timestamps: bool) -> list[str]:
    directory = Path(frame_dir)
    timestamps_path = directory / "timestamps.txt"
    if not timestamps_path.is_file():
        if require_timestamps:
            raise FileNotFoundError(f"missing VIF-Bench timestamps file: {timestamps_path}")
        return list_frames(directory)
    timestamps = [
        line.strip()
        for line in timestamps_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    frames = [directory / f"{index}.png" for index in range(1, len(timestamps) + 1)]
    missing = [str(path) for path in frames if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"VIF-Bench timestamp/frame contract is incomplete under {directory}; first={missing[0]}"
        )
    return [str(path) for path in frames]


def route_question_record(
    video_id: str,
    source: str,
    frame_dir: str,
    images: Sequence[str],
    bucket: str,
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": CAMERA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": camera_prompt(len(images), QUESTION_BY_LABEL[bucket]),
            },
        ],
        "images": list(images),
        "assistant_prefix": "",
        "sample_id": f"vif-route:{video_id}:{bucket}",
        "video_id": video_id,
        "case_id": video_id,
        "visual_source_case_id": video_id,
        "camera_primitive": bucket,
        "aigc_model_name": source,
        "frame_dir": frame_dir,
        "visual_condition": "matched_frames",
    }


def build_vif_inputs(args: argparse.Namespace) -> None:
    index = load_index(args.index_dir, args.expected_ranks, args.check_frame_dirs)
    rows: list[dict[str, Any]] = []
    frame_counts: Counter[int] = Counter()
    bad_frame_counts: list[dict[str, Any]] = []
    for video_id, metadata in sorted(index["expected"].items()):
        frame_dir = metadata["frame_dir"]
        images = list_vif_protocol_frames(frame_dir, args.require_timestamps)
        frame_counts[len(images)] += 1
        if args.expected_frames > 0 and len(images) != args.expected_frames:
            bad_frame_counts.append(
                {"video_id": video_id, "frame_dir": frame_dir, "frames": len(images)}
            )
            if not args.allow_frame_count_mismatch:
                continue
        for bucket in ROUTE_BUCKETS:
            rows.append(
                route_question_record(
                    video_id,
                    metadata["aigc_model_name"],
                    frame_dir,
                    images,
                    bucket,
                )
            )
    if bad_frame_counts and not args.allow_frame_count_mismatch:
        raise ValueError(
            f"{len(bad_frame_counts)} VIF-Bench videos do not have exactly "
            f"{args.expected_frames} frames; first={bad_frame_counts[0]}"
        )
    expected_records = index["num_expected_videos"] * len(ROUTE_BUCKETS)
    if len(rows) != expected_records:
        raise AssertionError(f"route question count mismatch: {len(rows)} != {expected_records}")
    write_jsonl(args.output_jsonl, rows)
    summary = {
        "schema_version": "vifbench_three_class_route_input_v1",
        "index_dir": str(args.index_dir),
        "num_videos": index["num_expected_videos"],
        "num_route_questions": len(rows),
        "route_buckets": list(ROUTE_BUCKETS),
        "frame_count_distribution": {str(key): value for key, value in sorted(frame_counts.items())},
        "expected_frames": args.expected_frames,
        "frame_count_mismatches": bad_frame_counts[:100],
        "same_frames_as_detection_index": True,
        "mirrors_vifbench_timestamp_frame_protocol": args.require_timestamps,
        "camera_text_enters_detection_prompt": False,
        "output_jsonl": str(args.output_jsonl),
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_score_rows(prediction_dir: str | Path) -> list[dict[str, Any]]:
    paths = sorted(Path(prediction_dir).glob("rank_*.jsonl"))
    if not paths:
        paths = sorted(Path(prediction_dir).rglob("rank_*.jsonl"))
    if not paths:
        raise ValueError(f"no rank_*.jsonl score files found under {prediction_dir}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"route score without sample_id: {path}")
            if sample_id in seen:
                raise ValueError(f"duplicate route score: {sample_id}")
            seen.add(sample_id)
            rows.append(row)
    return rows


def softmax(scores: Mapping[str, float]) -> dict[str, float]:
    maximum = max(scores.values())
    values = {key: math.exp(value - maximum) for key, value in scores.items()}
    denominator = sum(values.values())
    return {key: value / denominator for key, value in values.items()}


def quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def route_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if row.get("route_gold_bucket") in ROUTE_BUCKETS]
    if not eligible:
        return None
    confusion = {bucket: Counter() for bucket in ROUTE_BUCKETS}
    correct = 0
    accepted = 0
    accepted_correct = 0
    for row in eligible:
        gold = str(row["route_gold_bucket"])
        predicted = str(row["predicted_bucket"])
        confusion[gold][predicted] += 1
        correct += predicted == gold
        if row["route_bucket"] != "shared":
            accepted += 1
            accepted_correct += predicted == gold
    per_bucket = {}
    recalls = []
    for bucket in ROUTE_BUCKETS:
        total = sum(confusion[bucket].values())
        recall = confusion[bucket][bucket] / total if total else None
        if recall is not None:
            recalls.append(recall)
        per_bucket[bucket] = {
            "support": total,
            "recall": recall,
            "predictions": dict(confusion[bucket]),
        }
    pair_predictions: dict[str, dict[str, str]] = defaultdict(dict)
    for row in eligible:
        case_id = str(row.get("case_id", ""))
        kind = str(row.get("visual_kind", ""))
        if case_id and kind in {"real", "fake"}:
            pair_predictions[case_id][kind] = str(row["predicted_bucket"])
    complete_pairs = [value for value in pair_predictions.values() if set(value) == {"real", "fake"}]
    pair_consistency = (
        sum(value["real"] == value["fake"] for value in complete_pairs) / len(complete_pairs)
        if complete_pairs
        else None
    )
    return {
        "num_gold_videos": len(eligible),
        "accuracy": correct / len(eligible),
        "macro_recall": sum(recalls) / len(recalls) if recalls else None,
        "accepted_coverage": accepted / len(eligible),
        "accepted_accuracy": accepted_correct / accepted if accepted else None,
        "per_bucket": per_bucket,
        "real_fake_pair_route_consistency": pair_consistency,
        "num_complete_real_fake_pairs": len(complete_pairs),
    }


def authenticity_route_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    labeled = [row for row in rows if row.get("aigc_model_name")]
    if not labeled:
        return None
    groups = {"real": [], "fake": []}
    for row in labeled:
        key = "real" if str(row["aigc_model_name"]).casefold() == "real" else "fake"
        groups[key].append(str(row["predicted_bucket"]))
    distributions: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for key, values in groups.items():
        counter = Counter(values)
        counts[key] = {bucket: counter[bucket] for bucket in ROUTE_BUCKETS}
        distributions[key] = {
            bucket: counter[bucket] / len(values) if values else 0.0 for bucket in ROUTE_BUCKETS
        }
    total_variation = 0.5 * sum(
        abs(distributions["real"][bucket] - distributions["fake"][bucket])
        for bucket in ROUTE_BUCKETS
    )

    real_by_base: dict[str, str] = {}
    fake_rows: list[tuple[str, str]] = []
    for row in labeled:
        video_id = str(row.get("video_id", ""))
        if "/" not in video_id:
            continue
        source, base_id = video_id.split("/", 1)
        bucket = str(row["predicted_bucket"])
        if source.casefold() == "real":
            real_by_base[base_id] = bucket
        else:
            fake_rows.append((base_id, bucket))
    paired = [(real_by_base[base_id], bucket) for base_id, bucket in fake_rows if base_id in real_by_base]
    return {
        "route_counts_by_authenticity": counts,
        "route_distributions_by_authenticity": distributions,
        "real_fake_route_distribution_total_variation": total_variation,
        "num_real_fake_pairs": len(paired),
        "paired_real_fake_same_predicted_route_rate": (
            sum(real == fake for real, fake in paired) / len(paired) if paired else None
        ),
        "interpretation": (
            "A large distribution gap means route alone correlates with Real/Fake on this benchmark. "
            "It must be reported as a possible benchmark shortcut and does not by itself establish "
            "camera-conditioned artifact reasoning."
        ),
    }


def aggregate_routes(args: argparse.Namespace) -> None:
    inputs = read_jsonl(args.input_jsonl)
    by_sample = {str(row.get("sample_id")): row for row in inputs}
    if len(by_sample) != len(inputs) or "" in by_sample:
        raise ValueError("route input sample_id values must be non-empty and unique")
    scores = load_score_rows(args.prediction_dir)
    score_by_sample = {str(row["sample_id"]): row for row in scores}
    missing = sorted(set(by_sample) - set(score_by_sample))
    unexpected = sorted(set(score_by_sample) - set(by_sample))
    if missing or unexpected:
        raise ValueError(
            f"route score coverage mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )

    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    metadata: dict[str, dict[str, Any]] = {}
    for sample_id, input_row in by_sample.items():
        video_id = str(input_row.get("video_id") or input_row.get("case_id") or "")
        bucket = str(input_row.get("camera_primitive") or "")
        if not video_id or bucket not in ROUTE_BUCKETS:
            raise ValueError(f"invalid route input record: {sample_id}")
        score = float(score_by_sample[sample_id]["yes_minus_no_score"])
        if not math.isfinite(score):
            raise ValueError(f"non-finite route score: {sample_id}")
        if bucket in grouped[video_id]:
            raise ValueError(f"duplicate route bucket for {video_id}:{bucket}")
        grouped[video_id][bucket] = score
        metadata.setdefault(video_id, input_row)

    manifest: list[dict[str, Any]] = []
    margins: list[float] = []
    top_probabilities: list[float] = []
    for video_id in sorted(grouped):
        bucket_scores = grouped[video_id]
        if set(bucket_scores) != set(ROUTE_BUCKETS):
            raise ValueError(f"incomplete three-class route scores for {video_id}: {bucket_scores}")
        probabilities = softmax(bucket_scores)
        order = sorted(ROUTE_BUCKETS, key=lambda bucket: (-probabilities[bucket], bucket))
        predicted = order[0]
        top_probability = probabilities[order[0]]
        margin = top_probability - probabilities[order[1]]
        fallback_reasons = []
        if top_probability < args.min_top_probability:
            fallback_reasons.append("top_probability")
        if margin <= args.min_margin:
            fallback_reasons.append("margin")
        selected = "shared" if fallback_reasons else predicted
        input_row = metadata[video_id]
        manifest.append(
            {
                "video_id": video_id,
                "aigc_model_name": input_row.get("aigc_model_name"),
                "frame_dir": input_row.get("frame_dir"),
                "case_id": input_row.get("case_id"),
                "visual_kind": input_row.get("visual_kind"),
                "source_family": input_row.get("source_family"),
                "route_gold_bucket": input_row.get("route_gold_bucket"),
                "yes_minus_no_scores": {bucket: bucket_scores[bucket] for bucket in ROUTE_BUCKETS},
                "relative_probabilities": {bucket: probabilities[bucket] for bucket in ROUTE_BUCKETS},
                "predicted_bucket": predicted,
                "route_bucket": selected,
                "cyclic_route_bucket": "shared" if selected == "shared" else CYCLIC_ROUTE[predicted],
                "top_relative_probability": top_probability,
                "relative_probability_margin": margin,
                "fallback_to_shared": bool(fallback_reasons),
                "fallback_reasons": fallback_reasons,
            }
        )
        margins.append(margin)
        top_probabilities.append(top_probability)

    write_jsonl(args.output_manifest, manifest)
    metrics = route_metrics(manifest)
    summary = {
        "schema_version": "three_class_camera_route_manifest_v1",
        "input_jsonl": str(args.input_jsonl),
        "prediction_dir": str(args.prediction_dir),
        "output_manifest": str(args.output_manifest),
        "num_question_records": len(inputs),
        "num_route_videos": len(manifest),
        "coverage": len(score_by_sample) / len(by_sample) if by_sample else 0.0,
        "thresholds": {
            "min_top_relative_probability": args.min_top_probability,
            "min_relative_probability_margin": args.min_margin,
            "note": "These are relative three-way scores, not calibrated class probabilities.",
        },
        "route_counts": dict(Counter(str(row["route_bucket"]) for row in manifest)),
        "predicted_bucket_counts": dict(Counter(str(row["predicted_bucket"]) for row in manifest)),
        "fallback_count": sum(bool(row["fallback_to_shared"]) for row in manifest),
        "confidence_distribution": {
            "top_probability_p10": quantile(top_probabilities, 0.10),
            "top_probability_median": quantile(top_probabilities, 0.50),
            "top_probability_p90": quantile(top_probabilities, 0.90),
            "margin_p10": quantile(margins, 0.10),
            "margin_median": quantile(margins, 0.50),
            "margin_p90": quantile(margins, 0.90),
        },
        "heldout_route_metrics": metrics,
        "authenticity_route_audit": authenticity_route_audit(manifest),
        "camera_text_enters_detection_prompt": False,
    }
    write_json(args.summary_json, summary)
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
        raise ValueError("route manifest video_id values must be non-empty and unique")
    if set(manifest) != set(expected):
        raise ValueError(
            f"route manifest/index mismatch: missing={len(set(expected) - set(manifest))}, "
            f"unexpected={len(set(manifest) - set(expected))}"
        )

    expert_specs = dict(args.expert)
    required_experts = set(ROUTE_BUCKETS)
    if set(expert_specs) != required_experts:
        raise ValueError(f"expert paths must be exactly {sorted(required_experts)}")
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
            expert = str(route["route_bucket"])
        elif args.route_mode == "cyclic":
            expert = str(route["cyclic_route_bucket"])
        elif args.route_mode == "shared":
            expert = "shared"
        else:
            raise ValueError(f"unsupported route mode: {args.route_mode}")
        if expert not in predictions:
            raise ValueError(f"selected expert {expert!r} has no prediction directory")
        source_row = predictions[expert].get(video_id)
        if source_row is None:
            raise ValueError(f"missing {expert} prediction for {video_id}")
        row = copy.deepcopy(source_row)
        row["hard_route_condition"] = args.route_mode
        row["hard_route_selected_expert"] = expert
        row["hard_route_predicted_bucket"] = route["predicted_bucket"]
        row["hard_route_fallback_to_shared"] = route["fallback_to_shared"]
        row["hard_route_top_relative_probability"] = route["top_relative_probability"]
        row["hard_route_relative_probability_margin"] = route["relative_probability_margin"]
        selected_rows.append(row)
        selected_counts[expert] += 1

    evaluation = evaluate_predictions(selected_rows, expected)
    write_json(args.output_predictions, selected_rows)
    summary = {
        "schema_version": "vifbench_hard_route_composition_v1",
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
    cyclic = read_json(args.cyclic_summary)
    deltas = {
        "predicted_minus_base": {
            key: metric_delta(predicted, base, key)
            for key in ("balanced_accuracy", "fake_recall", "fake_f1")
        },
        "predicted_minus_shared": {
            key: metric_delta(predicted, shared, key)
            for key in ("balanced_accuracy", "fake_recall", "fake_f1")
        },
        "predicted_minus_cyclic": {
            key: metric_delta(predicted, cyclic, key)
            for key in ("balanced_accuracy", "fake_recall", "fake_f1")
        },
    }
    base_gain = deltas["predicted_minus_base"]
    route_gain = deltas["predicted_minus_shared"]
    causal_gain = deltas["predicted_minus_cyclic"]
    checks = {
        "all_coverage_at_least_99pct": all(
            float(evaluation_payload(value)["coverage"]) >= 0.99
            for value in (base, shared, predicted, cyclic)
        ),
        "all_format_valid_at_least_99pct": all(
            float(evaluation_payload(value)["format_valid_rate"]) >= 0.99
            for value in (base, shared, predicted, cyclic)
        ),
        "predicted_route_beats_original_base": (
            max(base_gain["balanced_accuracy"], base_gain["fake_f1"])
            >= args.min_base_gain
            and min(base_gain["balanced_accuracy"], base_gain["fake_f1"])
            >= -args.max_other_drop
        ),
        "predicted_route_beats_shared": (
            max(route_gain["balanced_accuracy"], route_gain["fake_f1"])
            >= args.min_shared_gain
            and min(route_gain["balanced_accuracy"], route_gain["fake_f1"])
            >= -args.max_other_drop
        ),
        "predicted_route_beats_cyclic": (
            max(causal_gain["balanced_accuracy"], causal_gain["fake_f1"])
            >= args.min_cyclic_gain
            and min(causal_gain["balanced_accuracy"], causal_gain["fake_f1"])
            >= -args.max_other_drop
        ),
    }
    passed = all(checks.values())
    output = {
        "gate": "ViF-Bench 三分类相机运动硬路由检测专家验证",
        "status": "passed" if passed else "failed",
        "what_was_tested": (
            "The original detection checkpoint, a shared detection LoRA, frame-predicted camera routing "
            "over three detection experts, and a cyclically wrong route are evaluated with the same "
            "no-camera detection prompt and VIF-Bench frames. Expert predictions are generated once; "
            "only route selection changes."
        ),
        "thresholds": {
            "min_predicted_minus_original_base_primary_gain": args.min_base_gain,
            "min_predicted_minus_shared_primary_gain": args.min_shared_gain,
            "min_predicted_minus_cyclic_primary_gain": args.min_cyclic_gain,
            "max_drop_in_other_primary_metric": args.max_other_drop,
        },
        "checks": checks,
        "models": {
            "original_detection_base": base,
            "shared_control": shared,
            "predicted_camera_route": predicted,
            "cyclic_wrong_route": cyclic,
        },
        "deltas": deltas,
        "generator_level_win_rates": {
            "predicted_vs_shared_balanced_accuracy": per_model_win_rate(
                predicted, shared, "balanced_accuracy"
            ),
            "predicted_vs_base_balanced_accuracy": per_model_win_rate(
                predicted, base, "balanced_accuracy"
            ),
            "predicted_vs_cyclic_balanced_accuracy": per_model_win_rate(
                predicted, cyclic, "balanced_accuracy"
            ),
        },
        "does_not_establish": (
            "Passing is a low-cost mechanism gate for camera-conditioned specialization, not the final "
            "paper result. VIF-Bench has been used for development; a frozen method still requires the "
            "GenBuster benchmark and other held-out evaluation."
        ),
        "next_action": (
            "If passed, replace hard expert selection with an internal soft camera-conditioned residual "
            "adapter and freeze the method before final benchmarks. If failed, do not spend compute on "
            "the same routing family."
        ),
    }
    write_json(args.output_json, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-vif-inputs")
    build.add_argument("--index-dir", required=True)
    build.add_argument("--output-jsonl", required=True)
    build.add_argument("--summary-json", required=True)
    build.add_argument("--expected-ranks", type=int, default=16)
    build.add_argument("--expected-frames", type=int, default=16)
    build.add_argument("--check-frame-dirs", action="store_true")
    build.add_argument("--require-timestamps", action="store_true")
    build.add_argument("--allow-frame-count-mismatch", action="store_true")
    build.set_defaults(handler=build_vif_inputs)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--input-jsonl", required=True)
    aggregate.add_argument("--prediction-dir", required=True)
    aggregate.add_argument("--output-manifest", required=True)
    aggregate.add_argument("--summary-json", required=True)
    aggregate.add_argument("--min-top-probability", type=float, default=0.0)
    aggregate.add_argument("--min-margin", type=float, default=0.0)
    aggregate.set_defaults(handler=aggregate_routes)

    compose = subparsers.add_parser("compose")
    compose.add_argument("--index-dir", required=True)
    compose.add_argument("--route-manifest", required=True)
    compose.add_argument("--expert", action="append", type=parse_named_path, required=True)
    compose.add_argument("--shared-prediction-dir")
    compose.add_argument("--route-mode", choices=("predicted", "cyclic", "shared"), required=True)
    compose.add_argument("--output-predictions", required=True)
    compose.add_argument("--output-summary", required=True)
    compose.add_argument("--expected-ranks", type=int, default=16)
    compose.set_defaults(handler=compose_predictions)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--base-eval", required=True)
    summarize.add_argument("--shared-summary", required=True)
    summarize.add_argument("--predicted-summary", required=True)
    summarize.add_argument("--cyclic-summary", required=True)
    summarize.add_argument("--output-json", required=True)
    summarize.add_argument("--min-base-gain", type=float, default=0.005)
    summarize.add_argument("--min-shared-gain", type=float, default=0.005)
    summarize.add_argument("--min-cyclic-gain", type=float, default=0.01)
    summarize.add_argument("--max-other-drop", type=float, default=0.005)
    summarize.set_defaults(handler=summarize_gate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
