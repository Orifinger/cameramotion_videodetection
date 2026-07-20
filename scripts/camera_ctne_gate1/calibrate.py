#!/usr/bin/env python3
"""Freeze DataB-validation thresholds and the camera-only shortcut probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_ctne_gate1.contracts import MODEL_SCHEMA_VERSION, normalize_path, read_jsonl, write_json
from scripts.camera_ctne_gate1.evaluate import _best_balanced_threshold, _fit_camera_only, _score_flow_controls
from scripts.camera_ctne_gate1.preprocessing import CTNEPreprocessor, camera_video_summary, load_feature_arrays


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.feature_index_jsonl)
    train = [row for row in rows if str(row.get("dataset_split")) == "train"]
    validation = [row for row in rows if str(row.get("dataset_split")) == "val"]
    if not train or not validation:
        raise ValueError(f"need DataB train and val rows, found {len(train)} and {len(validation)}")
    preprocessor = CTNEPreprocessor.load(args.model_root / "preprocessor.npz")
    scores, _, _, relaxation, val_camera_summaries = _score_flow_controls(
        validation,
        model_root=args.model_root,
        preprocessor=preprocessor,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        mean_weight=args.mean_weight,
        shuffle_seed=args.shuffle_seed,
    )
    labels = np.asarray([int(row["label"]) for row in validation], dtype=np.int64)
    thresholds: dict[str, float] = {}
    metrics: dict[str, Any] = {}
    for method in ("matched", "unconditional", "shuffled"):
        thresholds[method], metrics[method] = _best_balanced_threshold(labels, scores[method])

    train_summaries = np.stack(
        [camera_video_summary(load_feature_arrays(row)[0]) for row in train]
    ).astype(np.float32)
    camera_only = _fit_camera_only(train, train_summaries)
    camera_scores = camera_only.predict_proba(val_camera_summaries)[:, 1]
    thresholds["camera_only"], metrics["camera_only"] = _best_balanced_threshold(labels, camera_scores)
    scaler = camera_only.steps[0][1]
    classifier = camera_only.steps[1][1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / "camera_only.tmp.npz"
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(MODEL_SCHEMA_VERSION),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float32),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float32),
        coefficient=np.asarray(classifier.coef_[0], dtype=np.float32),
        intercept=np.asarray(classifier.intercept_[0], dtype=np.float32),
    )
    temporary.replace(args.output_dir / "camera_only.npz")
    summary = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "status": "passed",
        "calibration_source": "DataB held-out validation only",
        "feature_index_jsonl": normalize_path(args.feature_index_jsonl),
        "model_root": normalize_path(args.model_root),
        "train_samples_for_camera_only": len(train),
        "validation_samples": len(validation),
        "video_score_mean_weight": args.mean_weight,
        "video_score_tail_quantile": 0.90,
        "shuffle_seed": args.shuffle_seed,
        "thresholds": thresholds,
        "validation_metrics": metrics,
        "shuffle_relaxation_counts": relaxation,
        "external_benchmark_tuning_permitted": False,
    }
    write_json(args.output_dir / "calibration.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--feature-index-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mean-weight", type=float, default=0.5)
    parser.add_argument("--shuffle-seed", type=int, default=20260721)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    calibrate(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
