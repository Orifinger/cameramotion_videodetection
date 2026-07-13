#!/usr/bin/env python3
"""Audit, merge, and compare VIF-Bench retention predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


VALID_ANSWERS = {"real", "fake"}


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_id_from_frame_dir(frame_dir: str) -> str:
    parts = Path(frame_dir).parts
    try:
        base_index = parts.index("test_normalized")
        return "/".join(parts[base_index + 1 :])
    except ValueError:
        path = Path(frame_dir)
        return f"{path.parent.name}/{path.name}"


def load_index(
    index_dir: str | Path,
    expected_ranks: int | None = None,
    check_frame_dirs: bool = False,
) -> dict[str, Any]:
    index_dir = Path(index_dir)
    files = sorted(index_dir.glob("test_index.rank*.json"))
    if expected_ranks is not None and len(files) != expected_ranks:
        raise ValueError(
            f"expected {expected_ranks} VIF-Bench index shards under {index_dir}, found {len(files)}"
        )
    if not files:
        raise ValueError(f"no VIF-Bench index shards found under {index_dir}")

    expected: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    missing_frame_dirs: list[str] = []
    skipped_full_videos = 0
    source_counts: defaultdict[str, int] = defaultdict(int)
    for index_file in files:
        payload = read_json(index_file)
        if not isinstance(payload, Mapping):
            raise ValueError(f"index shard must contain an object: {index_file}")
        for source, frame_dirs in payload.items():
            if not isinstance(frame_dirs, list):
                raise ValueError(f"index source {source!r} must contain a list: {index_file}")
            for raw_path in frame_dirs:
                frame_dir = str(raw_path)
                if "full-videos" in frame_dir:
                    skipped_full_videos += 1
                    continue
                video_id = video_id_from_frame_dir(frame_dir)
                if video_id in expected:
                    duplicates.append(video_id)
                    continue
                if check_frame_dirs and not Path(frame_dir).is_dir():
                    missing_frame_dirs.append(frame_dir)
                expected[video_id] = {
                    "aigc_model_name": str(source),
                    "frame_dir": frame_dir,
                }
                source_counts[str(source)] += 1

    if duplicates:
        raise ValueError(f"duplicate video ids across index shards: {duplicates[:10]}")
    return {
        "index_dir": str(index_dir),
        "num_index_shards": len(files),
        "index_files": [str(path) for path in files],
        "num_expected_videos": len(expected),
        "num_skipped_full_videos": skipped_full_videos,
        "source_counts": dict(sorted(source_counts.items())),
        "expected": expected,
        "missing_frame_dirs": missing_frame_dirs,
    }


def load_predictions(prediction_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prediction_dir = Path(prediction_dir)
    files = sorted(prediction_dir.glob("rank_*/*.json"))
    if not files:
        files = sorted(prediction_dir.rglob("*.json"))
    rows: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    seen: set[str] = set()
    for path in files:
        payload = read_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"prediction file must contain a list: {path}")
        for item in payload:
            if not isinstance(item, Mapping):
                raise ValueError(f"prediction item must contain an object: {path}")
            row = dict(item)
            video_id = str(row.get("video_id", "")).strip()
            if not video_id:
                raise ValueError(f"prediction is missing video_id: {path}")
            if video_id in seen:
                duplicate_ids.append(video_id)
                continue
            seen.add(video_id)
            rows.append(row)
    if duplicate_ids:
        raise ValueError(f"duplicate prediction video ids: {duplicate_ids[:10]}")
    return rows, {
        "prediction_dir": str(prediction_dir),
        "num_prediction_files": len(files),
        "prediction_files": [str(path) for path in files],
        "num_predictions": len(rows),
    }


def answer_label(row: Mapping[str, Any]) -> str:
    return str(row.get("answer", "")).strip().lower()


def binary_metrics(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    num_pairs = len(pairs)
    if not num_pairs:
        return {
            "num_pairs": 0,
            "balanced_accuracy": None,
            "fake_recall": None,
            "fake_f1": None,
        }

    real_correct = sum(real_answer == "real" for real_answer, _ in pairs)
    fake_correct = sum(fake_answer != "real" for _, fake_answer in pairs)
    false_positive = num_pairs - real_correct
    false_negative = num_pairs - fake_correct
    real_recall = real_correct / num_pairs
    fake_recall = fake_correct / num_pairs
    balanced_accuracy = (real_recall + fake_recall) / 2.0
    precision_denominator = fake_correct + false_positive
    precision = fake_correct / precision_denominator if precision_denominator else 0.0
    f1_denominator = precision + fake_recall
    fake_f1 = 2.0 * precision * fake_recall / f1_denominator if f1_denominator else 0.0
    return {
        "num_pairs": num_pairs,
        "balanced_accuracy": balanced_accuracy,
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "fake_precision": precision,
        "fake_f1": fake_f1,
        "confusion": {
            "real_as_real": real_correct,
            "real_as_fake": false_positive,
            "fake_as_fake": fake_correct,
            "fake_as_real": false_negative,
        },
    }


def mean_supported(per_model: Mapping[str, Mapping[str, Any]], key: str) -> float | None:
    values = [float(metrics[key]) for metrics in per_model.values() if metrics.get(key) is not None]
    return sum(values) / len(values) if values else None


def evaluate_predictions(
    rows: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    row_by_id = {str(row["video_id"]): row for row in rows}
    expected_ids = set(expected)
    predicted_ids = set(row_by_id)
    matched_ids = expected_ids & predicted_ids
    valid_predictions = sum(answer_label(row_by_id[video_id]) in VALID_ANSWERS for video_id in matched_ids)

    by_base: defaultdict[str, dict[str, Any]] = defaultdict(dict)
    for video_id in sorted(matched_ids):
        row = row_by_id[video_id]
        source = str(row.get("aigc_model_name") or expected[video_id]["aigc_model_name"])
        parts = video_id.split("/", 1)
        if len(parts) != 2:
            continue
        base_id = parts[1]
        answer = answer_label(row)
        if source.lower() == "real":
            by_base[base_id]["real"] = answer
        else:
            by_base[base_id][source] = answer

    official_pairs: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    strict_pairs: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    invalid_pairs: defaultdict[str, int] = defaultdict(int)
    unpaired_fake: defaultdict[str, int] = defaultdict(int)
    for values in by_base.values():
        real_answer = values.get("real")
        for source, fake_answer in values.items():
            if source == "real":
                continue
            if real_answer is None:
                unpaired_fake[source] += 1
                continue
            official_pairs[source].append((str(real_answer), str(fake_answer)))
            if real_answer in VALID_ANSWERS and fake_answer in VALID_ANSWERS:
                strict_pairs[source].append((str(real_answer), str(fake_answer)))
            else:
                invalid_pairs[source] += 1

    per_model: dict[str, dict[str, Any]] = {}
    for source in sorted(official_pairs):
        metrics = binary_metrics(official_pairs[source])
        metrics["strict_valid_pair_metrics"] = binary_metrics(strict_pairs[source])
        metrics["invalid_pairs"] = invalid_pairs[source]
        metrics["unpaired_fake_predictions"] = unpaired_fake[source]
        per_model[source] = metrics

    average = {
        "num_models": len(per_model),
        "balanced_accuracy": mean_supported(per_model, "balanced_accuracy"),
        "fake_recall": mean_supported(per_model, "fake_recall"),
        "fake_f1": mean_supported(per_model, "fake_f1"),
    }
    return {
        "num_expected_predictions": len(expected_ids),
        "num_predictions": len(rows),
        "num_matched_predictions": len(matched_ids),
        "coverage": len(matched_ids) / len(expected_ids) if expected_ids else 0.0,
        "format_valid_rate": valid_predictions / len(matched_ids) if matched_ids else 0.0,
        "num_missing_predictions": len(expected_ids - predicted_ids),
        "missing_prediction_ids": sorted(expected_ids - predicted_ids)[:100],
        "num_unexpected_predictions": len(predicted_ids - expected_ids),
        "unexpected_prediction_ids": sorted(predicted_ids - expected_ids)[:100],
        "average_across_fake_models": average,
        "per_fake_model": per_model,
    }


def metric_delta(camera: Mapping[str, Any], base: Mapping[str, Any], key: str) -> float | None:
    camera_value = camera.get(key)
    base_value = base.get(key)
    if camera_value is None or base_value is None:
        return None
    return float(camera_value) - float(base_value)


def build_retention_summary(
    base_eval: Mapping[str, Any],
    camera_eval: Mapping[str, Any],
    max_accuracy_drop: float = 0.03,
    max_f1_drop: float = 0.03,
    min_coverage: float = 0.99,
    min_format_valid: float = 0.99,
) -> dict[str, Any]:
    base_average = base_eval["average_across_fake_models"]
    camera_average = camera_eval["average_across_fake_models"]
    deltas = {
        "balanced_accuracy": metric_delta(camera_average, base_average, "balanced_accuracy"),
        "fake_recall": metric_delta(camera_average, base_average, "fake_recall"),
        "fake_f1": metric_delta(camera_average, base_average, "fake_f1"),
        "format_valid_rate": float(camera_eval["format_valid_rate"])
        - float(base_eval["format_valid_rate"]),
    }
    base_models = set(base_eval["per_fake_model"])
    camera_models = set(camera_eval["per_fake_model"])
    per_model_deltas = {}
    for model in sorted(base_models & camera_models):
        per_model_deltas[model] = {
            key: metric_delta(
                camera_eval["per_fake_model"][model],
                base_eval["per_fake_model"][model],
                key,
            )
            for key in ("balanced_accuracy", "fake_recall", "fake_f1")
        }
    checks = {
        "base_coverage": float(base_eval["coverage"]) >= min_coverage,
        "camera_coverage": float(camera_eval["coverage"]) >= min_coverage,
        "base_format_valid": float(base_eval["format_valid_rate"]) >= min_format_valid,
        "camera_format_valid": float(camera_eval["format_valid_rate"]) >= min_format_valid,
        "same_fake_model_set": base_models == camera_models and bool(base_models),
        "average_balanced_accuracy_retained": deltas["balanced_accuracy"] is not None
        and deltas["balanced_accuracy"] >= -max_accuracy_drop,
        "average_fake_f1_retained": deltas["fake_f1"] is not None
        and deltas["fake_f1"] >= -max_f1_drop,
    }
    passed = all(checks.values())
    return {
        "gate": "camera VQA adapter detection retention on VIF-Bench",
        "status": "passed" if passed else "failed",
        "what_was_tested": (
            "The original DataB detection checkpoint and the same checkpoint with the final binary-camera "
            "LoRA use identical VIF-Bench inputs, deterministic decoding, and the same original no-camera "
            "detection prompt. No camera label or caption is provided at inference."
        ),
        "what_was_not_tested": (
            "This is a camera-only adapter retention diagnostic, not a jointly trained camera-assisted "
            "detector and not evidence of a detection improvement caused by camera motion. VIF-Bench has "
            "also been used repeatedly during project development, so it is external to adapter training "
            "but not a pristine final model-selection test."
        ),
        "thresholds": {
            "min_coverage": min_coverage,
            "min_format_valid": min_format_valid,
            "max_average_balanced_accuracy_drop": max_accuracy_drop,
            "max_average_fake_f1_drop": max_f1_drop,
        },
        "checks": checks,
        "base_detection_checkpoint": dict(base_eval),
        "camera_vqa_adapter": dict(camera_eval),
        "camera_minus_base": deltas,
        "per_fake_model_camera_minus_base": per_model_deltas,
        "next_action": (
            "Combine this result with the fixed-DataA retention gate before selecting the detection replay mix."
        ),
    }


def run_audit(args: argparse.Namespace) -> None:
    index = load_index(args.index_dir, args.expected_ranks, args.check_frame_dirs)
    system_prompt = Path(args.system_prompt_file)
    user_suffix = Path(args.user_prompt_suffix_file)
    suffix_text = user_suffix.read_text(encoding="utf-8")
    camera_placeholders = [
        token for token in ("{camera_labels}", "{camera_caption}") if token in suffix_text
    ]
    checks = {
        "index_shard_count": index["num_index_shards"] == args.expected_ranks,
        "nonempty_index": index["num_expected_videos"] > 0,
        "all_frame_dirs_exist": not index["missing_frame_dirs"],
        "system_prompt_exists": system_prompt.is_file(),
        "user_suffix_exists": user_suffix.is_file(),
        "no_camera_placeholders": not camera_placeholders,
    }
    output = {
        "gate": "VIF-Bench camera-adapter retention preflight",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "prompt_mode": "no_camera",
        "camera_context_provided": False,
        "system_prompt_file": str(system_prompt),
        "system_prompt_sha256": sha256(system_prompt) if system_prompt.is_file() else None,
        "user_prompt_suffix_file": str(user_suffix),
        "user_prompt_suffix_sha256": sha256(user_suffix) if user_suffix.is_file() else None,
        "camera_placeholders": camera_placeholders,
        "index": {key: value for key, value in index.items() if key != "expected"},
    }
    write_json(args.output_json, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if output["status"] != "passed":
        raise SystemExit(2)


def run_evaluate(args: argparse.Namespace) -> None:
    index = load_index(args.index_dir, args.expected_ranks, check_frame_dirs=False)
    expected = index["expected"]
    base_rows, base_files = load_predictions(args.base_prediction_dir)
    camera_rows, camera_files = load_predictions(args.camera_prediction_dir)
    write_json(args.base_merged_json, base_rows)
    write_json(args.camera_merged_json, camera_rows)
    base_eval = evaluate_predictions(base_rows, expected)
    camera_eval = evaluate_predictions(camera_rows, expected)
    base_eval["prediction_files"] = base_files
    camera_eval["prediction_files"] = camera_files
    write_json(args.base_eval_json, base_eval)
    write_json(args.camera_eval_json, camera_eval)
    summary = build_retention_summary(
        base_eval,
        camera_eval,
        max_accuracy_drop=args.max_accuracy_drop,
        max_f1_drop=args.max_f1_drop,
        min_coverage=args.min_coverage,
        min_format_valid=args.min_format_valid,
    )
    summary["index_dir"] = str(args.index_dir)
    summary["base_merged_json"] = str(args.base_merged_json)
    summary["camera_merged_json"] = str(args.camera_merged_json)
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--index-dir", required=True)
    audit.add_argument("--system-prompt-file", required=True)
    audit.add_argument("--user-prompt-suffix-file", required=True)
    audit.add_argument("--output-json", required=True)
    audit.add_argument("--expected-ranks", type=int, default=16)
    audit.add_argument("--check-frame-dirs", action="store_true")
    audit.set_defaults(func=run_audit)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--index-dir", required=True)
    evaluate.add_argument("--base-prediction-dir", required=True)
    evaluate.add_argument("--camera-prediction-dir", required=True)
    evaluate.add_argument("--base-merged-json", required=True)
    evaluate.add_argument("--camera-merged-json", required=True)
    evaluate.add_argument("--base-eval-json", required=True)
    evaluate.add_argument("--camera-eval-json", required=True)
    evaluate.add_argument("--output-json", required=True)
    evaluate.add_argument("--expected-ranks", type=int, default=16)
    evaluate.add_argument("--min-coverage", type=float, default=0.99)
    evaluate.add_argument("--min-format-valid", type=float, default=0.99)
    evaluate.add_argument("--max-accuracy-drop", type=float, default=0.03)
    evaluate.add_argument("--max-f1-drop", type=float, default=0.03)
    evaluate.set_defaults(func=run_evaluate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
