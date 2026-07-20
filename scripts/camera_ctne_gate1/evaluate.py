#!/usr/bin/env python3
"""Evaluate matched, unconditional, shuffled, and camera-only CTNE controls."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_ctne_gate1.contracts import MODEL_SCHEMA_VERSION, normalize_path, read_jsonl, write_json
from scripts.camera_ctne_gate1.controls import (
    best_balanced_threshold as _best_balanced_threshold,
    binary_metrics as _binary_metrics,
    shuffled_donor_indices,
)
from scripts.camera_ctne_gate1.flow_model import load_flow
from scripts.camera_ctne_gate1.preprocessing import (
    CTNEPreprocessor,
    camera_video_summary,
    load_feature_arrays,
    resample_sequence,
)


def _generator_macro_balanced(
    rows: Sequence[Mapping[str, Any]], labels: np.ndarray, scores: np.ndarray, threshold: float
) -> tuple[float, dict[str, float]]:
    predictions = (np.asarray(scores) >= threshold).astype(np.int64)
    real = labels == 0
    real_recall = float((predictions[real] == 0).mean()) if real.any() else float("nan")
    by_generator: dict[str, float] = {}
    for generator in sorted({str(row.get("generator_name", "unknown")) for row, label in zip(rows, labels) if label == 1}):
        mask = np.asarray(
            [label == 1 and str(row.get("generator_name", "unknown")) == generator for row, label in zip(rows, labels)],
            dtype=bool,
        )
        fake_recall = float((predictions[mask] == 1).mean())
        by_generator[generator] = float((real_recall + fake_recall) / 2.0)
    macro = float(np.mean(list(by_generator.values()))) if by_generator else float("nan")
    return macro, by_generator


def _aggregate_transition_scores(values: np.ndarray, lengths: Sequence[int], mean_weight: float) -> np.ndarray:
    output: list[float] = []
    offset = 0
    for length in lengths:
        current = values[offset : offset + length]
        output.append(float(mean_weight * current.mean() + (1.0 - mean_weight) * np.quantile(current, 0.90)))
        offset += length
    if offset != values.size:
        raise AssertionError("transition-score lengths do not consume the flattened array")
    return np.asarray(output, dtype=np.float64)


def _load_dataset(
    rows: Sequence[Mapping[str, Any]],
    preprocessor: CTNEPreprocessor,
    *,
    shuffle_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int], dict[str, int], np.ndarray]:
    camera_raw: list[np.ndarray] = []
    camera_scaled: list[np.ndarray] = []
    evidence_projected: list[np.ndarray] = []
    lengths: list[int] = []
    camera_summaries: list[np.ndarray] = []
    for row in rows:
        raw_camera, raw_evidence = load_feature_arrays(row)
        scaled, projected = preprocessor.transform(raw_camera, raw_evidence)
        camera_raw.append(raw_camera)
        camera_scaled.append(scaled)
        evidence_projected.append(projected)
        lengths.append(raw_camera.shape[0])
        camera_summaries.append(camera_video_summary(raw_camera))
    donors, relaxation_counts = shuffled_donor_indices(rows, shuffle_seed)
    shuffled_scaled = [
        (resample_sequence(camera_raw[donor], lengths[index]) - preprocessor.camera_mean) / preprocessor.camera_scale
        for index, donor in enumerate(donors)
    ]
    return (
        np.concatenate(camera_scaled).astype(np.float32),
        np.concatenate(shuffled_scaled).astype(np.float32),
        np.concatenate(evidence_projected).astype(np.float32),
        lengths,
        donors,
        relaxation_counts,
        np.stack(camera_summaries).astype(np.float32),
    )


@torch.inference_mode()
def _transition_nll(
    model: torch.nn.Module,
    evidence: np.ndarray,
    context: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, evidence.shape[0], batch_size):
        end = min(evidence.shape[0], start + batch_size)
        y = torch.from_numpy(evidence[start:end]).to(device)
        c = torch.from_numpy(context[start:end]).to(device)
        parts.append((-model.log_prob(y, context=c)).float().cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def _discover_models(model_root: Path) -> dict[int, dict[str, Path]]:
    output: dict[int, dict[str, Path]] = {}
    for seed_dir in sorted((model_root / "models").glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        modes = {
            mode: seed_dir / mode
            for mode in ("matched", "unconditional")
            if (seed_dir / mode / "model.pt").is_file() and (seed_dir / mode / "config.json").is_file()
        }
        if set(modes) == {"matched", "unconditional"}:
            output[seed] = modes
    if not output:
        raise FileNotFoundError(f"no complete matched/unconditional seed pairs under {model_root}")
    return output


def _score_flow_controls(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_root: Path,
    preprocessor: CTNEPreprocessor,
    device: torch.device,
    batch_size: int,
    mean_weight: float,
    shuffle_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[int, np.ndarray]], list[int], dict[str, int], np.ndarray]:
    matched_context, shuffled_context, evidence, lengths, donors, relaxation, camera_summaries = _load_dataset(
        rows, preprocessor, shuffle_seed=shuffle_seed
    )
    zero_context = np.zeros_like(matched_context)
    models = _discover_models(model_root)
    by_seed: dict[str, dict[int, np.ndarray]] = {"matched": {}, "unconditional": {}, "shuffled": {}}
    for seed, paths in models.items():
        matched_model, _ = load_flow(paths["matched"], device)
        matched_nll = _transition_nll(matched_model, evidence, matched_context, device=device, batch_size=batch_size)
        shuffled_nll = _transition_nll(matched_model, evidence, shuffled_context, device=device, batch_size=batch_size)
        by_seed["matched"][seed] = _aggregate_transition_scores(matched_nll, lengths, mean_weight)
        by_seed["shuffled"][seed] = _aggregate_transition_scores(shuffled_nll, lengths, mean_weight)
        del matched_model
        unconditional_model, _ = load_flow(paths["unconditional"], device)
        unconditional_nll = _transition_nll(
            unconditional_model, evidence, zero_context, device=device, batch_size=batch_size
        )
        by_seed["unconditional"][seed] = _aggregate_transition_scores(unconditional_nll, lengths, mean_weight)
        del unconditional_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ensemble = {
        method: np.mean(np.stack([scores for _, scores in sorted(values.items())]), axis=0)
        for method, values in by_seed.items()
    }
    return ensemble, by_seed, donors, relaxation, camera_summaries


def _fit_camera_only(train_rows: Sequence[Mapping[str, Any]], summaries: np.ndarray) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=17))
    model.fit(summaries, labels)
    return model


def _bootstrap_delta(
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    first_threshold: float,
    second_threshold: float,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    real = np.flatnonzero(labels == 0)
    fake = np.flatnonzero(labels == 1)
    auc_deltas: list[float] = []
    macro_deltas: list[float] = []
    for _ in range(iterations):
        indices = np.concatenate(
            [rng.choice(real, size=real.size, replace=True), rng.choice(fake, size=fake.size, replace=True)]
        )
        sampled_rows = [rows[int(index)] for index in indices]
        sampled_labels = labels[indices]
        first_values = first[indices]
        second_values = second[indices]
        auc_deltas.append(float(roc_auc_score(sampled_labels, first_values) - roc_auc_score(sampled_labels, second_values)))
        first_macro, _ = _generator_macro_balanced(sampled_rows, sampled_labels, first_values, first_threshold)
        second_macro, _ = _generator_macro_balanced(sampled_rows, sampled_labels, second_values, second_threshold)
        macro_deltas.append(first_macro - second_macro)
    def summarize(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "ci95_lower": float(np.quantile(array, 0.025)),
            "ci95_upper": float(np.quantile(array, 0.975)),
        }
    return {"iterations": iterations, "roc_auc_delta": summarize(auc_deltas), "generator_macro_balanced_delta": summarize(macro_deltas)}


def _method_report(
    rows: Sequence[Mapping[str, Any]], labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    metrics = _binary_metrics(labels, scores, threshold)
    macro, by_generator = _generator_macro_balanced(rows, labels, scores, threshold)
    metrics["threshold_from_datab_validation"] = float(threshold)
    metrics["generator_macro_balanced_accuracy"] = macro
    metrics["by_generator_balanced_accuracy"] = by_generator
    return metrics


def _direction_consistency(
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    matched: np.ndarray,
    unconditional: np.ndarray,
    matched_threshold: float,
    unconditional_threshold: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in ("motion_bucket", "generator_name"):
        values: dict[str, float] = {}
        groups = sorted({str(row.get(field, "unknown")) for row in rows})
        if field == "generator_name":
            groups = [group for group in groups if group != "real"]
        for group in groups:
            if field == "generator_name":
                mask = np.asarray(
                    [
                        label == 0 or (label == 1 and str(row.get(field, "unknown")) == group)
                        for row, label in zip(rows, labels)
                    ],
                    dtype=bool,
                )
            else:
                mask = np.asarray([str(row.get(field, "unknown")) == group for row in rows], dtype=bool)
            if mask.sum() < 20 or np.unique(labels[mask]).size < 2:
                continue
            first = _binary_metrics(labels[mask], matched[mask], matched_threshold)["balanced_accuracy"]
            second = _binary_metrics(labels[mask], unconditional[mask], unconditional_threshold)["balanced_accuracy"]
            values[group] = float(first - second)
        positive = sum(delta > 0 for delta in values.values())
        output[field] = {
            "deltas": values,
            "positive_groups": positive,
            "supported_groups": len(values),
            "positive_rate": positive / len(values) if values else 0.0,
        }
    return output


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 <= args.mean_weight <= 1.0:
        raise ValueError("mean_weight must be in [0,1]")
    all_validation = read_jsonl(args.validation_index_jsonl)
    validation = [row for row in all_validation if str(row.get("dataset_split")) == "val"]
    camera_train = [row for row in all_validation if str(row.get("dataset_split")) == "train"]
    test_all = read_jsonl(args.test_index_jsonl)
    test = [row for row in test_all if args.test_split == "all" or str(row.get("dataset_split")) == args.test_split]
    if not validation or not test:
        raise ValueError(f"empty validation/test rows: {len(validation)} {len(test)}")
    device = torch.device(args.device)
    preprocessor = CTNEPreprocessor.load(args.model_root / "preprocessor.npz")
    val_scores, val_by_seed, _, val_relaxation, val_camera_summary = _score_flow_controls(
        validation,
        model_root=args.model_root,
        preprocessor=preprocessor,
        device=device,
        batch_size=args.batch_size,
        mean_weight=args.mean_weight,
        shuffle_seed=args.shuffle_seed,
    )
    test_scores, test_by_seed, donors, test_relaxation, test_camera_summary = _score_flow_controls(
        test,
        model_root=args.model_root,
        preprocessor=preprocessor,
        device=device,
        batch_size=args.batch_size,
        mean_weight=args.mean_weight,
        shuffle_seed=args.shuffle_seed,
    )
    val_labels = np.asarray([int(row["label"]) for row in validation], dtype=np.int64)
    test_labels = np.asarray([int(row["label"]) for row in test], dtype=np.int64)
    thresholds: dict[str, float] = {}
    validation_metrics: dict[str, Any] = {}
    for method in ("matched", "unconditional", "shuffled"):
        threshold, metrics = _best_balanced_threshold(val_labels, val_scores[method])
        thresholds[method] = threshold
        validation_metrics[method] = metrics

    train_camera_summary = np.stack(
        [camera_video_summary(load_feature_arrays(row)[0]) for row in camera_train]
    ).astype(np.float32)
    camera_only = _fit_camera_only(camera_train, train_camera_summary)
    val_camera_scores = camera_only.predict_proba(val_camera_summary)[:, 1]
    test_camera_scores = camera_only.predict_proba(test_camera_summary)[:, 1]
    camera_threshold, camera_val_metrics = _best_balanced_threshold(val_labels, val_camera_scores)
    thresholds["camera_only"] = camera_threshold
    validation_metrics["camera_only"] = camera_val_metrics
    test_scores["camera_only"] = test_camera_scores

    reports = {
        method: _method_report(test, test_labels, scores, thresholds[method])
        for method, scores in test_scores.items()
    }
    seed_reports: dict[str, dict[str, Any]] = {}
    for method, values in test_by_seed.items():
        seed_reports[method] = {
            str(seed): _method_report(test, test_labels, scores, thresholds[method])
            for seed, scores in sorted(values.items())
        }
    matched_vs_unconditional = _bootstrap_delta(
        test,
        test_labels,
        test_scores["matched"],
        test_scores["unconditional"],
        thresholds["matched"],
        thresholds["unconditional"],
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    matched_vs_shuffled = _bootstrap_delta(
        test,
        test_labels,
        test_scores["matched"],
        test_scores["shuffled"],
        thresholds["matched"],
        thresholds["shuffled"],
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed + 1,
    )
    auc_delta = float(reports["matched"]["roc_auc"] - reports["unconditional"]["roc_auc"])
    macro_delta = float(
        reports["matched"]["generator_macro_balanced_accuracy"]
        - reports["unconditional"]["generator_macro_balanced_accuracy"]
    )
    directions = _direction_consistency(
        test,
        test_labels,
        test_scores["matched"],
        test_scores["unconditional"],
        thresholds["matched"],
        thresholds["unconditional"],
    )
    motion_consistent = directions["motion_bucket"]["positive_groups"] >= 2
    generator_consistent = directions["generator_name"]["positive_rate"] > 0.5
    checks = {
        "matched_has_increment_over_unconditional": (
            (auc_delta >= args.min_primary_gain and macro_delta >= -args.max_other_drop)
            or (macro_delta >= args.min_primary_gain and auc_delta >= -args.max_other_drop)
        ),
        "matched_beats_shuffled_with_paired_ci": (
            matched_vs_shuffled["roc_auc_delta"]["ci95_lower"] > 0.0
            or matched_vs_shuffled["generator_macro_balanced_delta"]["ci95_lower"] > 0.0
        ),
        "camera_only_not_suspiciously_high": reports["camera_only"]["roc_auc"] <= args.max_camera_only_auc,
        "motion_bucket_direction_consistency": motion_consistent,
        "generator_direction_consistency": generator_consistent,
        "all_seed_pairs_present": len(_discover_models(args.model_root)) >= args.min_seeds,
    }
    status = "passed" if all(checks.values()) else "failed"
    output = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "gate": "Gate 1 - continuous camera-conditioned temporal normality expert",
        "status": status,
        "what_was_tested": (
            "Matched continuous RAFT camera context, an equal-capacity zero-context flow, and an evaluation-only "
            "sample-shuffled context use identical variable-length transitions, evidence, preprocessing, and frame indices."
        ),
        "inputs": {
            "model_root": normalize_path(args.model_root),
            "validation_index_jsonl": normalize_path(args.validation_index_jsonl),
            "test_index_jsonl": normalize_path(args.test_index_jsonl),
            "test_dataset_name": args.dataset_name,
        },
        "thresholds": {
            "min_primary_gain": args.min_primary_gain,
            "max_other_drop": args.max_other_drop,
            "max_camera_only_auc": args.max_camera_only_auc,
            "min_seeds": args.min_seeds,
            "decision_thresholds_selected_on_datab_validation": thresholds,
        },
        "checks": checks,
        "settings": {
            "video_score_mean_weight": args.mean_weight,
            "video_score_tail_quantile": 0.90,
            "shuffle_seed": args.shuffle_seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "inference_uses_camera_text": False,
            "qwen_used_in_gate1": False,
        },
        "validation": {"num_samples": len(validation), "metrics": validation_metrics, "shuffle_relaxation_counts": val_relaxation},
        "test": {"num_samples": len(test), "reports": reports, "shuffle_relaxation_counts": test_relaxation},
        "deltas": {
            "matched_minus_unconditional_roc_auc": auc_delta,
            "matched_minus_unconditional_generator_macro_balanced_accuracy": macro_delta,
            "matched_minus_unconditional_bootstrap": matched_vs_unconditional,
            "matched_minus_shuffled_bootstrap": matched_vs_shuffled,
        },
        "direction_consistency": directions,
        "per_seed_reports": seed_reports,
        "does_not_establish": (
            "Gate 1 does not establish a Qwen Real/Fake gain. Gate 2 is permitted only after camera increment "
            "replicates across the required external benchmarks."
        ),
        "next_action": "Run the second external benchmark if this is the first; only then decide whether to start Gate 2.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "ctne_gate1_summary.json", output)
    with (args.output_dir / "ctne_gate1_items.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "sample_id",
            "label",
            "source_name",
            "generator_name",
            "motion_bucket",
            "frame_count",
            "transition_count",
            "matched_score",
            "unconditional_score",
            "shuffled_score",
            "camera_only_score",
            "shuffled_camera_donor_sample_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(test):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "label": row["label"],
                    "source_name": row.get("source_name"),
                    "generator_name": row.get("generator_name"),
                    "motion_bucket": row.get("motion_bucket"),
                    "frame_count": row.get("frame_count"),
                    "transition_count": row.get("transition_count"),
                    "matched_score": test_scores["matched"][index],
                    "unconditional_score": test_scores["unconditional"][index],
                    "shuffled_score": test_scores["shuffled"][index],
                    "camera_only_score": test_scores["camera_only"][index],
                    "shuffled_camera_donor_sample_id": test[donors[index]]["sample_id"],
                }
            )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--validation-index-jsonl", type=Path, required=True)
    parser.add_argument("--test-index-jsonl", type=Path, required=True)
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mean-weight", type=float, default=0.5)
    parser.add_argument("--shuffle-seed", type=int, default=20260721)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument("--min-primary-gain", type=float, default=0.01)
    parser.add_argument("--max-other-drop", type=float, default=0.005)
    parser.add_argument("--max-camera-only-auc", type=float, default=0.65)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
