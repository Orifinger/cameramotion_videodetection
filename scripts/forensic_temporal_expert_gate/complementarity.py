#!/usr/bin/env python3
"""Fixed, label-free fusion of Qwen confidence and the temporal expert on ViF."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.forensic_temporal_expert_gate.contracts import normalize_path, read_json_or_jsonl, write_json
from scripts.forensic_temporal_expert_gate.metrics import classification_metrics, logit, sigmoid


def canonical_video_id(value: Any) -> str:
    text = re.sub(r"/+", "/", str(value or "").strip().replace("\\", "/"))
    if text.casefold().startswith("vifbench:"):
        text = text.split(":", 1)[1]
    lowered = text.casefold()
    extracted = False
    for marker in ("/parsed_frames/parsed_frames/", "/test_normalized/"):
        if marker in lowered:
            text = text[lowered.index(marker) + len(marker) :]
            extracted = True
            break
    parts = list(PurePosixPath(text.lstrip("/")).parts)
    if extracted and len(parts) >= 3 and parts[0].casefold() in {"real", "fake"}:
        parts = parts[1:]
    if len(parts) < 2:
        raise ValueError(f"cannot canonicalize video id: {value!r}")
    if parts[0].casefold() == "real":
        parts[0] = "real"
    return "/".join(parts)


def load_historical(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_json_or_jsonl(path)
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = canonical_video_id(row.get("video_id", row.get("sample_id", "")))
        if key in output:
            raise ValueError(f"duplicate historical Qwen id: {key}")
        output[key] = row
    return output


def load_confidence(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    files = [path] if path.is_file() else sorted(path.glob("rank_*.jsonl"))
    output: dict[str, dict[str, Any]] = {}
    for file_path in files:
        for row in read_json_or_jsonl(file_path):
            if "fake_pair_probability" not in row:
                continue
            key = canonical_video_id(row.get("video_id", ""))
            if key in output:
                raise ValueError(f"duplicate Qwen confidence id: {key}")
            output[key] = row
    return output


def load_expert(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "video_id", "label", "generator_name",
            "static_score", "ordered_score", "ordered_shuffled_input_score", "shuffled_trained_score",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"expert items missing columns: {sorted(missing)}")
        for raw in reader:
            key = canonical_video_id(raw["video_id"])
            if key in output:
                raise ValueError(f"duplicate expert id: {key}")
            output[key] = dict(raw)
    return output


def generator_macro(
    labels: np.ndarray,
    predictions: np.ndarray,
    generators: np.ndarray,
) -> float:
    real = labels == 0
    real_recall = float((predictions[real] == 0).mean())
    values: list[float] = []
    for generator in sorted(set(generators[labels == 1])):
        mask = (labels == 1) & (generators == generator)
        values.append(0.5 * (real_recall + float((predictions[mask] == 1).mean())))
    return float(np.mean(values))


def bootstrap(
    rows: Sequence[Mapping[str, Any]],
    baseline: np.ndarray,
    fused: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        video_id = str(row["video_id"])
        base = video_id.split("/", 1)[-1]
        groups[base].append(index)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    bacc: list[float] = []
    macro: list[float] = []
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    generators = np.asarray([str(row["generator_name"]) for row in rows], dtype=object)
    for _ in range(iterations):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        indices = np.asarray([index for key in sampled for index in groups[str(key)]], dtype=np.int64)
        y = labels[indices]
        if np.unique(y).size < 2:
            continue
        base_pred = (baseline[indices] >= 0.5).astype(np.int64)
        fused_pred = (fused[indices] >= 0.5).astype(np.int64)
        real = y == 0
        fake = y == 1
        base_bacc = 0.5 * (float((base_pred[real] == 0).mean()) + float((base_pred[fake] == 1).mean()))
        fused_bacc = 0.5 * (float((fused_pred[real] == 0).mean()) + float((fused_pred[fake] == 1).mean()))
        bacc.append(fused_bacc - base_bacc)
        macro.append(
            generator_macro(y, fused_pred, generators[indices])
            - generator_macro(y, base_pred, generators[indices])
        )

    def summarize(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "ci95_lower": float(np.quantile(array, 0.025)),
            "ci95_upper": float(np.quantile(array, 0.975)),
        }

    return {
        "iterations": len(bacc),
        "balanced_accuracy_delta": summarize(bacc),
        "generator_macro_balanced_accuracy_delta": summarize(macro),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate1-summary", type=Path, required=True)
    parser.add_argument("--expert-items-csv", type=Path, required=True)
    parser.add_argument("--historical-qwen-predictions", type=Path, required=True)
    parser.add_argument("--qwen-confidence", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expert-weight", type=float, default=0.25)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--min-primary-gain", type=float, default=0.015)
    parser.add_argument("--min-control-margin", type=float, default=0.01)
    parser.add_argument("--max-real-recall-drop", type=float, default=0.01)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    gate1 = json.loads(args.gate1_summary.read_text(encoding="utf-8"))
    historical = load_historical(args.historical_qwen_predictions)
    confidence = load_confidence(args.qwen_confidence)
    expert = load_expert(args.expert_items_csv)
    common = sorted(set(historical) & set(expert))
    rows: list[dict[str, Any]] = []
    missing_confidence = 0
    invalid_historical = 0
    for video_id in common:
        erow = expert[video_id]
        hrow = historical[video_id]
        answer = str(hrow.get("answer", "")).strip().casefold()
        if answer not in {"real", "fake"}:
            invalid_historical += 1
            continue
        crow = confidence.get(video_id)
        if crow is None:
            missing_confidence += 1
            continue
        qwen_score = float(crow["fake_pair_probability"])
        rows.append(
            {
                "video_id": video_id,
                "label": int(erow["label"]),
                "generator_name": str(erow["generator_name"]),
                "qwen_score": qwen_score,
                "qwen_answer": answer,
                **{key: float(erow[f"{key}_score"]) for key in ("static", "ordered", "ordered_shuffled_input", "shuffled_trained")},
            }
        )
    coverage = len(rows) / len(expert) if expert else 0.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if coverage < args.min_coverage:
        summary = {
            "gate": "Qwen 与原生尺度时序专家的固定融合互补性门（Gate 2）",
            "status": "conclusion_insufficient",
            "reason": "Qwen confidence coverage is below the pre-registered threshold",
            "coverage": coverage,
            "min_coverage": args.min_coverage,
            "expert_rows": len(expert),
            "historical_rows": len(historical),
            "confidence_rows": len(confidence),
            "joined_rows": len(rows),
            "missing_confidence": missing_confidence,
            "invalid_historical_answers": invalid_historical,
        }
        write_json(args.output_dir / "forensic_temporal_expert_gate2_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 4

    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    generators = [row["generator_name"] for row in rows]
    qwen = np.asarray([row["qwen_score"] for row in rows], dtype=np.float64)
    qwen_hard = (qwen >= 0.5).astype(np.int64)
    archived = np.asarray([int(row["qwen_answer"] == "fake") for row in rows], dtype=np.int64)
    reproduction = float((qwen_hard == archived).mean())
    baseline = classification_metrics(labels, qwen, 0.5, generators)
    fused: dict[str, np.ndarray] = {}
    reports: dict[str, dict[str, Any]] = {}
    for condition in ("static", "ordered", "ordered_shuffled_input", "shuffled_trained"):
        expert_scores = np.asarray([row[condition] for row in rows], dtype=np.float64)
        values = sigmoid(logit(qwen) + args.expert_weight * logit(expert_scores))
        fused[condition] = values
        reports[condition] = classification_metrics(labels, values, 0.5, generators)

    primary = ("balanced_accuracy", "generator_macro_balanced_accuracy")
    gains = {key: float(reports["ordered"][key] - baseline[key]) for key in primary}
    control_margins = {
        control: {key: float(reports["ordered"][key] - reports[control][key]) for key in primary}
        for control in ("static", "ordered_shuffled_input", "shuffled_trained")
    }
    bootstrap_result = bootstrap(
        rows, qwen, fused["ordered"], iterations=args.bootstrap_iterations, seed=29
    )
    bootstrap_positive = max(
        bootstrap_result["balanced_accuracy_delta"]["ci95_lower"],
        bootstrap_result["generator_macro_balanced_accuracy_delta"]["ci95_lower"],
    ) > 0
    checks = {
        "gate1_passed": gate1.get("status") == "passed",
        "confidence_coverage": coverage >= args.min_coverage,
        "confidence_reproduces_archived_answer": reproduction >= 0.99,
        "fixed_ordered_fusion_improves_primary_metric": max(gains.values()) >= args.min_primary_gain,
        "group_bootstrap_lower_bound_positive": bootstrap_positive,
        "ordered_fusion_beats_all_controls": all(
            max(values.values()) >= args.min_control_margin for values in control_margins.values()
        ),
        "real_recall_preserved": float(reports["ordered"]["real_recall"] - baseline["real_recall"]) >= -args.max_real_recall_drop,
    }
    status = "passed" if all(checks.values()) else "failed"
    quadrants = Counter()
    ordered_prediction = (fused["ordered"] >= 0.5).astype(np.int64)
    expert_prediction = np.asarray([int(row["ordered"] >= 0.5) for row in rows], dtype=np.int64)
    for label, qpred, epred in zip(labels, qwen_hard, expert_prediction):
        quadrants[
            "both_correct" if qpred == label and epred == label
            else "expert_only_correct" if epred == label
            else "qwen_only_correct" if qpred == label
            else "both_wrong"
        ] += 1
    items_path = args.output_dir / "forensic_temporal_expert_gate2_items.csv"
    with items_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "video_id", "label", "generator_name", "qwen_score", "ordered_expert_score", "ordered_fused_score", "ordered_fused_prediction"
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "video_id": row["video_id"],
                    "label": row["label"],
                    "generator_name": row["generator_name"],
                    "qwen_score": row["qwen_score"],
                    "ordered_expert_score": row["ordered"],
                    "ordered_fused_score": float(fused["ordered"][index]),
                    "ordered_fused_prediction": int(ordered_prediction[index]),
                }
            )
    summary = {
        "gate": "Qwen 与原生尺度时序专家的固定融合互补性门（Gate 2）",
        "status": status,
        "what_was_tested": "A fixed beta=0.25 logit fusion registered before reading ViF labels; no router or weight is fitted on ViF.",
        "expert_weight": args.expert_weight,
        "development_dataset": "ViF-Bench",
        "genbuster_closed_benchmark_touched": False,
        "coverage": coverage,
        "checks": checks,
        "qwen_confidence_archived_answer_reproduction": reproduction,
        "baseline_qwen": baseline,
        "fixed_fusion_metrics": reports,
        "ordered_fusion_gains": gains,
        "ordered_control_margins": control_margins,
        "bootstrap_ordered_fusion_minus_qwen": bootstrap_result,
        "hard_error_quadrants": dict(quadrants),
        "items_csv": normalize_path(items_path),
        "does_not_establish": "Only a future untouched GenBuster Closed Benchmark run can establish final generalization.",
        "next_action": "If both gates pass, freeze the recipe, retrain the expert on all 6766 DataB rows, then run GenBuster Closed Benchmark once.",
    }
    write_json(args.output_dir / "forensic_temporal_expert_gate2_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
