#!/usr/bin/env python3
"""Calibrate on DataB validation and evaluate the supervised camera interaction gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_ctne_gate1.contracts import normalize_path, read_jsonl, write_json
from scripts.camera_ctne_gate1.controls import best_balanced_threshold, shuffled_donor_indices
from scripts.camera_ctne_gate1.evaluate import (
    _bootstrap_delta,
    _direction_consistency,
    _method_report,
)
from scripts.camera_discriminative_gate import SCHEMA_VERSION
from scripts.camera_discriminative_gate.data import (
    PackedSequences,
    SupervisedPreprocessor,
    build_packed_sequences,
)
from scripts.camera_discriminative_gate.model import load_model, score_model

METHODS = ("matched", "evidence_only", "shuffled_camera", "camera_only")
MODEL_MODES = ("matched", "zero_camera", "camera_only")


def _discover_models(model_root: Path) -> tuple[dict[int, dict[str, Path]], dict[str, Any]]:
    models: dict[int, dict[str, Path]] = {}
    configs: dict[int, dict[str, dict[str, Any]]] = {}
    for seed_dir in sorted((model_root / "models").glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        modes: dict[str, Path] = {}
        mode_configs: dict[str, dict[str, Any]] = {}
        for mode in MODEL_MODES:
            path = seed_dir / mode
            if (path / "model.pt").is_file() and (path / "config.json").is_file():
                modes[mode] = path
                mode_configs[mode] = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if set(modes) == set(MODEL_MODES):
            models[seed] = modes
            configs[seed] = mode_configs
    if not models:
        raise FileNotFoundError(f"no complete three-mode seed groups under {model_root}")
    equal_initialization = all(
        len({configs[seed][mode]["initial_state_fingerprint"] for mode in MODEL_MODES}) == 1
        for seed in configs
    )
    equal_parameters = all(
        len({int(configs[seed][mode]["parameter_count"]) for mode in MODEL_MODES}) == 1
        for seed in configs
    )
    return models, {
        "seeds": sorted(models),
        "complete_seed_groups": len(models),
        "same_initialization_within_seed": equal_initialization,
        "same_parameter_count_within_seed": equal_parameters,
    }


def _score_conditions(
    packed: PackedSequences,
    *,
    model_root: Path,
    device: torch.device,
    batch_size: int,
    shuffle_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[int, np.ndarray]], list[int], dict[str, int], dict[str, Any]]:
    models, model_audit = _discover_models(model_root)
    donors, relaxation = shuffled_donor_indices(packed.rows, shuffle_seed)
    shuffled_camera = [packed.sequence(donor)[0] for donor in donors]
    by_seed: dict[str, dict[int, np.ndarray]] = {method: {} for method in METHODS}
    for seed, paths in models.items():
        matched, _ = load_model(paths["matched"], device)
        by_seed["matched"][seed] = score_model(
            matched,
            packed,
            mode="matched",
            device=device,
            batch_size=batch_size,
        )
        by_seed["shuffled_camera"][seed] = score_model(
            matched,
            packed,
            mode="matched",
            device=device,
            batch_size=batch_size,
            camera_overrides=shuffled_camera,
        )
        del matched
        evidence_only, _ = load_model(paths["zero_camera"], device)
        by_seed["evidence_only"][seed] = score_model(
            evidence_only,
            packed,
            mode="zero_camera",
            device=device,
            batch_size=batch_size,
        )
        del evidence_only
        camera_only, _ = load_model(paths["camera_only"], device)
        by_seed["camera_only"][seed] = score_model(
            camera_only,
            packed,
            mode="camera_only",
            device=device,
            batch_size=batch_size,
        )
        del camera_only
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ensemble = {
        method: np.mean(np.stack([scores for _, scores in sorted(seed_scores.items())]), axis=0)
        for method, seed_scores in by_seed.items()
    }
    return ensemble, by_seed, donors, relaxation, model_audit


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    packed = PackedSequences.load(args.packed_npz, args.rows_jsonl)
    val_indices = [
        index for index, row in enumerate(packed.rows) if str(row.get("dataset_split")) == args.validation_split
    ]
    if not val_indices:
        raise ValueError(f"no rows for validation split {args.validation_split}")
    validation = packed.subset(val_indices)
    scores, by_seed, _, relaxation, model_audit = _score_conditions(
        validation,
        model_root=args.model_root,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        shuffle_seed=args.shuffle_seed,
    )
    thresholds: dict[str, float] = {}
    metrics: dict[str, Any] = {}
    for method in METHODS:
        thresholds[method], metrics[method] = best_balanced_threshold(validation.labels, scores[method])
    per_seed_auc = {
        method: {
            str(seed): _method_report(
                validation.rows,
                validation.labels,
                values,
                thresholds[method],
            )["roc_auc"]
            for seed, values in sorted(seed_scores.items())
        }
        for method, seed_scores in by_seed.items()
    }
    checks = {
        "both_labels_present": np.unique(validation.labels).size == 2,
        "same_initialization_within_seed": model_audit["same_initialization_within_seed"],
        "same_parameter_count_within_seed": model_audit["same_parameter_count_within_seed"],
        "at_least_three_complete_seeds": model_audit["complete_seed_groups"] >= args.min_seeds,
        "finite_thresholds": all(np.isfinite(value) for value in thresholds.values()),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "gate": "DataB validation calibration for supervised camera interaction",
        "status": "passed" if all(checks.values()) else "failed",
        "calibration_source": "DataB held-out validation only",
        "model_root": normalize_path(args.model_root),
        "packed_npz": normalize_path(args.packed_npz),
        "num_validation_samples": len(validation),
        "thresholds": thresholds,
        "validation_metrics": metrics,
        "per_seed_roc_auc": per_seed_auc,
        "shuffle_relaxation_counts": relaxation,
        "model_audit": model_audit,
        "checks": checks,
        "external_benchmark_tuning_permitted": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "calibration.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _load_or_build_external_pack(
    *,
    feature_index_jsonl: Path,
    preprocessor: SupervisedPreprocessor,
    packed_npz: Path,
    rows_jsonl: Path,
    overwrite: bool,
) -> PackedSequences:
    if not overwrite and packed_npz.is_file() and rows_jsonl.is_file():
        packed = PackedSequences.load(packed_npz, rows_jsonl)
        if packed.preprocessor_fingerprint == preprocessor.fingerprint():
            return packed
    rows = read_jsonl(feature_index_jsonl)
    packed = build_packed_sequences(rows, preprocessor)
    packed.save(packed_npz, rows_jsonl)
    return packed


def _write_items(
    output_csv: Path,
    packed: PackedSequences,
    scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    donors: Sequence[int],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "sample_id",
            "label",
            "generator_name",
            "motion_bucket",
            "shuffled_camera_donor",
        ]
        for method in METHODS:
            fields.extend([f"{method}_score", f"{method}_prediction"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(packed.rows):
            item: dict[str, Any] = {
                "sample_id": row["sample_id"],
                "label": int(packed.labels[index]),
                "generator_name": row.get("generator_name", "unknown"),
                "motion_bucket": row.get("motion_bucket", "unknown"),
                "shuffled_camera_donor": packed.rows[int(donors[index])]["sample_id"],
            }
            for method in METHODS:
                item[f"{method}_score"] = float(scores[method][index])
                item[f"{method}_prediction"] = int(scores[method][index] >= thresholds[method])
            writer.writerow(item)


def evaluate_external(args: argparse.Namespace) -> dict[str, Any]:
    calibration = json.loads((args.calibration_dir / "calibration.json").read_text(encoding="utf-8"))
    if calibration.get("status") != "passed":
        raise ValueError("DataB calibration did not pass")
    thresholds = {key: float(value) for key, value in calibration["thresholds"].items()}
    preprocessor = SupervisedPreprocessor.load(args.model_root / "preprocessor.npz")
    packed = _load_or_build_external_pack(
        feature_index_jsonl=args.feature_index_jsonl,
        preprocessor=preprocessor,
        packed_npz=args.packed_npz,
        rows_jsonl=args.rows_jsonl,
        overwrite=args.overwrite_pack,
    )
    if np.unique(packed.labels).size != 2:
        raise ValueError("external benchmark requires both Real and Fake samples")
    scores, by_seed, donors, relaxation, model_audit = _score_conditions(
        packed,
        model_root=args.model_root,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        shuffle_seed=args.shuffle_seed,
    )
    reports = {
        method: _method_report(packed.rows, packed.labels, values, thresholds[method])
        for method, values in scores.items()
    }
    per_seed = {
        method: {
            str(seed): _method_report(packed.rows, packed.labels, values, thresholds[method])
            for seed, values in sorted(seed_values.items())
        }
        for method, seed_values in by_seed.items()
    }
    matched_vs_evidence = _bootstrap_delta(
        packed.rows,
        packed.labels,
        scores["matched"],
        scores["evidence_only"],
        thresholds["matched"],
        thresholds["evidence_only"],
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    matched_vs_shuffled = _bootstrap_delta(
        packed.rows,
        packed.labels,
        scores["matched"],
        scores["shuffled_camera"],
        thresholds["matched"],
        thresholds["shuffled_camera"],
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed + 1,
    )
    auc_delta = float(reports["matched"]["roc_auc"] - reports["evidence_only"]["roc_auc"])
    macro_delta = float(
        reports["matched"]["generator_macro_balanced_accuracy"]
        - reports["evidence_only"]["generator_macro_balanced_accuracy"]
    )
    directions = _direction_consistency(
        packed.rows,
        packed.labels,
        scores["matched"],
        scores["evidence_only"],
        thresholds["matched"],
        thresholds["evidence_only"],
    )
    seed_deltas = {
        str(seed): float(
            per_seed["matched"][str(seed)]["roc_auc"]
            - per_seed["evidence_only"][str(seed)]["roc_auc"]
        )
        for seed in sorted(by_seed["matched"])
    }
    primary_gain = (
        auc_delta >= args.min_primary_gain and macro_delta >= -args.max_other_drop
    ) or (
        macro_delta >= args.min_primary_gain and auc_delta >= -args.max_other_drop
    )
    matched_beats_shuffled = (
        matched_vs_shuffled["roc_auc_delta"]["ci95_lower"] > 0.0
        or matched_vs_shuffled["generator_macro_balanced_delta"]["ci95_lower"] > 0.0
    )
    camera_only_not_primary = (
        reports["camera_only"]["roc_auc"] <= args.max_camera_only_auc
        and reports["matched"]["roc_auc"] >= reports["camera_only"]["roc_auc"]
    )
    checks = {
        "matched_has_external_increment_over_evidence_only": primary_gain,
        "matched_beats_shuffled_camera_with_paired_ci": matched_beats_shuffled,
        "camera_only_is_not_the_primary_detector": camera_only_not_primary,
        "at_least_two_motion_buckets_improve": directions["motion_bucket"]["positive_groups"] >= 2,
        "majority_of_supported_generators_improve": directions["generator_name"]["positive_rate"] > 0.5,
        "at_least_two_of_three_seeds_improve_auc": sum(value > 0.0 for value in seed_deltas.values()) >= 2,
        "three_complete_equal_capacity_seed_groups": (
            model_audit["complete_seed_groups"] >= args.min_seeds
            and model_audit["same_initialization_within_seed"]
            and model_audit["same_parameter_count_within_seed"]
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "gate": f"Supervised camera-evidence interaction gate - {args.dataset_name}",
        "status": "passed" if all(checks.values()) else "failed",
        "what_was_tested": (
            "A variable-length FiLM classifier was trained directly on DataB Real/Fake labels. "
            "The only controlled input change is matched, zero, shuffled, or camera-only continuous camera context."
        ),
        "inputs": {
            "dataset_name": args.dataset_name,
            "feature_index_jsonl": normalize_path(args.feature_index_jsonl),
            "model_root": normalize_path(args.model_root),
            "calibration_dir": normalize_path(args.calibration_dir),
        },
        "settings": {
            "num_samples": len(packed),
            "threshold_source": "DataB held-out validation only",
            "bootstrap_iterations": args.bootstrap_iterations,
            "shuffle_seed": args.shuffle_seed,
            "qwen_used": False,
            "camera_text_used": False,
            "real_fake_supervision_used": True,
        },
        "thresholds": thresholds,
        "reports": reports,
        "deltas": {
            "matched_minus_evidence_only_roc_auc": auc_delta,
            "matched_minus_evidence_only_generator_macro_balanced_accuracy": macro_delta,
            "matched_minus_evidence_only_bootstrap": matched_vs_evidence,
            "matched_minus_shuffled_camera_bootstrap": matched_vs_shuffled,
            "per_seed_matched_minus_evidence_only_roc_auc": seed_deltas,
        },
        "direction_consistency": directions,
        "shuffle_relaxation_counts": relaxation,
        "model_audit": model_audit,
        "checks": checks,
        "does_not_establish": (
            "Passing this gate establishes incremental value for the frozen camera features, not yet a Qwen Real/Fake gain."
        ),
        "next_action": (
            "Proceed to frozen-Qwen score fusion and GenBuster benchmark."
            if all(checks.values())
            else "Stop camera as the primary detection contribution; do not start Qwen fusion or camera RL."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "camera_discriminative_gate_summary.json", result)
    _write_items(
        args.output_dir / "camera_discriminative_gate_items.csv",
        packed,
        scores,
        thresholds,
        donors,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--packed-npz", type=Path, required=True)
    calibrate_parser.add_argument("--rows-jsonl", type=Path, required=True)
    calibrate_parser.add_argument("--model-root", type=Path, required=True)
    calibrate_parser.add_argument("--output-dir", type=Path, required=True)
    calibrate_parser.add_argument("--validation-split", default="val")
    external = subparsers.add_parser("external")
    external.add_argument("--feature-index-jsonl", type=Path, required=True)
    external.add_argument("--model-root", type=Path, required=True)
    external.add_argument("--calibration-dir", type=Path, required=True)
    external.add_argument("--packed-npz", type=Path, required=True)
    external.add_argument("--rows-jsonl", type=Path, required=True)
    external.add_argument("--dataset-name", required=True)
    external.add_argument("--output-dir", type=Path, required=True)
    external.add_argument("--overwrite-pack", action="store_true")
    for target in (calibrate_parser, external):
        target.add_argument("--batch-size", type=int, default=256)
        target.add_argument("--shuffle-seed", type=int, default=20260723)
        target.add_argument("--min-seeds", type=int, default=3)
        target.add_argument("--device", default="cuda")
    external.add_argument("--bootstrap-iterations", type=int, default=2000)
    external.add_argument("--bootstrap-seed", type=int, default=20260724)
    external.add_argument("--min-primary-gain", type=float, default=0.01)
    external.add_argument("--max-other-drop", type=float, default=0.005)
    external.add_argument("--max-camera-only-auc", type=float, default=0.65)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "calibrate":
        result = calibrate(args)
    else:
        result = evaluate_external(args)
    return 0 if result["status"] == "passed" else (0 if args.command == "external" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
