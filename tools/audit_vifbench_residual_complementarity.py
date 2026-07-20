#!/usr/bin/env python3
"""Audit whether an existing temporal expert complements Qwen on ViF-Bench.

The script does not train either detector. It joins per-video predictions, reports
the paired error quadrants, estimates an oracle upper bound, and runs a grouped
out-of-fold logistic fusion diagnostic. The fusion diagnostic is development
analysis only because ViF-Bench labels participate in cross-validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


VALID_ANSWERS = {"real": 0, "fake": 1}


def _normalize(value: Any) -> str:
    return re.sub(r"/+", "/", str(value).strip().replace("\\", "/"))


def canonical_video_id(value: Any) -> str:
    """Map Qwen video IDs and expert sample IDs to generator/base-id."""
    text = _normalize(value)
    if re.match(r"^[^/:]+:/", text):
        text = text.split(":", 1)[1]
    lowered = text.casefold()
    extracted_from_frame_path = False
    for marker in ("/parsed_frames/parsed_frames/", "/test_normalized/"):
        if marker in lowered:
            text = text[lowered.index(marker) + len(marker) :]
            extracted_from_frame_path = True
            break
    text = text.lstrip("/")
    parts = list(PurePosixPath(text).parts)
    if (
        extracted_from_frame_path
        and len(parts) >= 3
        and parts[0].casefold() in {"real", "fake"}
    ):
        parts = parts[1:]
    if len(parts) < 2:
        raise ValueError(f"cannot derive generator/base video id from {value!r}")
    if parts[0].casefold() == "real":
        parts[0] = "real"
    return "/".join(parts)


def base_identity(video_id: str) -> str:
    parts = canonical_video_id(video_id).split("/", 1)
    return parts[1] if len(parts) == 2 else parts[0]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_qwen_predictions(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    files: list[Path]
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("rank_*/*.json")) or sorted(path.rglob("*.json"))
    else:
        raise FileNotFoundError(f"Qwen prediction path does not exist: {path}")
    rows: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    invalid_payloads: list[str] = []
    for file_path in files:
        payload = _read_json(file_path)
        if not isinstance(payload, list):
            invalid_payloads.append(str(file_path))
            continue
        for raw in payload:
            if not isinstance(raw, Mapping):
                continue
            video_id = canonical_video_id(raw.get("video_id", ""))
            if video_id in rows:
                duplicates.append(video_id)
                continue
            answer = str(raw.get("answer", "")).strip().casefold()
            rows[video_id] = {
                "video_id": video_id,
                "answer": answer,
                "prediction": VALID_ANSWERS.get(answer),
                "raw": dict(raw),
            }
    if duplicates:
        raise ValueError(f"duplicate Qwen video IDs: {duplicates[:10]}")
    if not rows:
        raise ValueError(f"no Qwen prediction rows loaded from {path}")
    return rows, {
        "path": str(path),
        "files": len(files),
        "rows": len(rows),
        "invalid_payload_files": invalid_payloads,
    }


def load_expert_items(
    path: Path,
    *,
    score_column: str,
    prediction_column: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "label", score_column, prediction_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"expert CSV is missing columns: {sorted(missing)}")
        for raw in reader:
            video_id = canonical_video_id(raw["sample_id"])
            if video_id in rows:
                duplicates.append(video_id)
                continue
            score = float(raw[score_column])
            prediction = int(raw[prediction_column])
            label = int(raw["label"])
            if prediction not in (0, 1) or label not in (0, 1) or not np.isfinite(score):
                raise ValueError(f"invalid expert row for {video_id}")
            rows[video_id] = {
                "video_id": video_id,
                "label": label,
                "score": score,
                "prediction": prediction,
                "generator_name": str(raw.get("generator_name") or video_id.split("/", 1)[0]),
                "motion_bucket": str(raw.get("motion_bucket") or "unknown"),
                "raw": dict(raw),
            }
    if duplicates:
        raise ValueError(f"duplicate expert video IDs: {duplicates[:10]}")
    return rows, {"path": str(path), "rows": len(rows)}


def join_rows(
    qwen: Mapping[str, Mapping[str, Any]], expert: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qwen_ids = set(qwen)
    expert_ids = set(expert)
    joined: list[dict[str, Any]] = []
    label_mismatches: list[str] = []
    invalid_qwen: list[str] = []
    for video_id in sorted(qwen_ids & expert_ids):
        qrow = qwen[video_id]
        erow = expert[video_id]
        qpred = qrow["prediction"]
        if qpred is None:
            invalid_qwen.append(video_id)
            continue
        expected_label = 0 if video_id.split("/", 1)[0].casefold() == "real" else 1
        if int(erow["label"]) != expected_label:
            label_mismatches.append(video_id)
            continue
        joined.append(
            {
                "video_id": video_id,
                "base_id": base_identity(video_id),
                "label": expected_label,
                "generator_name": erow["generator_name"],
                "motion_bucket": erow["motion_bucket"],
                "qwen_prediction": int(qpred),
                "expert_score": float(erow["score"]),
                "expert_prediction": int(erow["prediction"]),
            }
        )
    return joined, {
        "qwen_rows": len(qwen),
        "expert_rows": len(expert),
        "id_intersection": len(qwen_ids & expert_ids),
        "joined_valid_rows": len(joined),
        "coverage_over_qwen": len(joined) / len(qwen_ids) if qwen_ids else 0.0,
        "coverage_over_expert": len(joined) / len(expert_ids) if expert_ids else 0.0,
        "qwen_format_valid_rate": (
            (len(qwen_ids & expert_ids) - len(invalid_qwen)) / len(qwen_ids & expert_ids)
            if qwen_ids & expert_ids
            else 0.0
        ),
        "qwen_only_count": len(qwen_ids - expert_ids),
        "expert_only_count": len(expert_ids - qwen_ids),
        "invalid_qwen_count": len(invalid_qwen),
        "label_mismatch_count": len(label_mismatches),
        "first_qwen_only": sorted(qwen_ids - expert_ids)[:20],
        "first_expert_only": sorted(expert_ids - qwen_ids)[:20],
        "first_invalid_qwen": invalid_qwen[:20],
        "first_label_mismatch": label_mismatches[:20],
    }


def _safe_div(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    scores: np.ndarray | None = None,
    generators: Sequence[str] | None = None,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    real = labels == 0
    fake = labels == 1
    real_recall = float((predictions[real] == 0).mean()) if real.any() else None
    fake_recall = float((predictions[fake] == 1).mean()) if fake.any() else None
    true_fake = int(((predictions == 1) & fake).sum())
    predicted_fake = int((predictions == 1).sum())
    fake_precision = _safe_div(true_fake, predicted_fake)
    fake_f1 = (
        2.0 * fake_precision * fake_recall / (fake_precision + fake_recall)
        if fake_precision is not None and fake_recall is not None and fake_precision + fake_recall
        else 0.0
    )
    result: dict[str, Any] = {
        "num_samples": int(labels.size),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": (real_recall + fake_recall) / 2.0
        if real_recall is not None and fake_recall is not None
        else None,
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "fake_precision": fake_precision,
        "fake_f1": float(fake_f1),
        "predicted_fake_rate": float((predictions == 1).mean()),
        "confusion": {
            "real_as_real": int(((predictions == 0) & real).sum()),
            "real_as_fake": int(((predictions == 1) & real).sum()),
            "fake_as_fake": true_fake,
            "fake_as_real": int(((predictions == 0) & fake).sum()),
        },
    }
    if scores is not None and np.unique(labels).size == 2:
        scores = np.asarray(scores, dtype=np.float64)
        result["roc_auc"] = float(roc_auc_score(labels, scores))
        result["average_precision"] = float(average_precision_score(labels, scores))
    if generators is not None and real_recall is not None:
        per_generator: dict[str, float] = {}
        values = np.asarray([str(value) for value in generators], dtype=object)
        for generator in sorted(set(values[fake])):
            mask = fake & (values == generator)
            generator_fake_recall = float((predictions[mask] == 1).mean())
            per_generator[generator] = (real_recall + generator_fake_recall) / 2.0
        result["generator_macro_balanced_accuracy"] = (
            float(np.mean(list(per_generator.values()))) if per_generator else None
        )
        result["per_fake_generator_balanced_accuracy"] = per_generator
    return result


def residual_quadrants(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        q_correct = int(row["qwen_prediction"]) == int(row["label"])
        e_correct = int(row["expert_prediction"]) == int(row["label"])
        key = (
            "both_correct"
            if q_correct and e_correct
            else "expert_only_correct"
            if e_correct
            else "qwen_only_correct"
            if q_correct
            else "both_wrong"
        )
        counts[key] += 1
    total = len(rows)
    qwen_wrong = counts["expert_only_correct"] + counts["both_wrong"]
    qwen_correct = counts["both_correct"] + counts["qwen_only_correct"]
    oracle_correct = total - counts["both_wrong"]
    return {
        "counts": dict(counts),
        "rates": {key: value / total for key, value in counts.items()},
        "expert_rescue_rate_among_qwen_errors": _safe_div(counts["expert_only_correct"], qwen_wrong),
        "expert_harm_pool_among_qwen_correct": _safe_div(counts["qwen_only_correct"], qwen_correct),
        "oracle_accuracy": _safe_div(oracle_correct, total),
        "oracle_gain_over_qwen_accuracy": _safe_div(counts["expert_only_correct"], total),
        "discordant_expert_only_correct": counts["expert_only_correct"],
        "discordant_qwen_only_correct": counts["qwen_only_correct"],
    }


def _sample_weights(labels: np.ndarray, generators: np.ndarray) -> np.ndarray:
    weights = np.zeros(labels.size, dtype=np.float64)
    real = labels == 0
    fake = labels == 1
    if real.any():
        weights[real] = 0.5 / real.sum()
    fake_generators = sorted(set(generators[fake]))
    for generator in fake_generators:
        mask = fake & (generators == generator)
        weights[mask] = 0.5 / len(fake_generators) / mask.sum()
    return weights * labels.size


def grouped_oof_fusion(
    rows: Sequence[Mapping[str, Any]],
    expert_scores: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    qwen = np.asarray([int(row["qwen_prediction"]) for row in rows], dtype=np.float64)
    groups = np.asarray([str(row["base_id"]) for row in rows], dtype=object)
    generators = np.asarray([str(row["generator_name"]) for row in rows], dtype=object)
    features = np.column_stack([qwen, np.asarray(expert_scores, dtype=np.float64)])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = np.full(labels.size, np.nan, dtype=np.float64)
    audit: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(splitter.split(features, labels, groups)):
        if np.unique(labels[train]).size != 2 or np.unique(labels[test]).size != 2:
            raise ValueError(f"fold {fold} does not contain both labels")
        model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
        model.fit(
            features[train],
            labels[train],
            sample_weight=_sample_weights(labels[train], generators[train]),
        )
        probabilities[test] = model.predict_proba(features[test])[:, 1]
        audit.append(
            {
                "fold": fold,
                "train_samples": int(train.size),
                "test_samples": int(test.size),
                "train_groups": int(np.unique(groups[train]).size),
                "test_groups": int(np.unique(groups[test]).size),
                "group_overlap": int(len(set(groups[train]) & set(groups[test]))),
                "coefficients": [float(value) for value in model.coef_[0]],
                "intercept": float(model.intercept_[0]),
            }
        )
    if not np.isfinite(probabilities).all():
        raise AssertionError("out-of-fold fusion left unscored rows")
    return probabilities, audit


def grouped_bootstrap_delta(
    rows: Sequence[Mapping[str, Any]],
    left_predictions: np.ndarray,
    right_predictions: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    group_rows: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        group_rows[str(row["base_id"])].append(index)
    groups = sorted(group_rows)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    labels_all = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    generators_all = np.asarray([str(row["generator_name"]) for row in rows], dtype=object)
    for _ in range(iterations):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.asarray([index for group in sampled for index in group_rows[str(group)]], dtype=np.int64)
        labels = labels_all[indices]
        generators = generators_all[indices]
        left = classification_metrics(labels, left_predictions[indices], generators=generators)
        right = classification_metrics(labels, right_predictions[indices], generators=generators)
        deltas.append(
            float(left["generator_macro_balanced_accuracy"])
            - float(right["generator_macro_balanced_accuracy"])
        )
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "metric": "generator_macro_balanced_accuracy",
        "iterations": iterations,
        "mean_delta": float(values.mean()),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
        "positive_rate": float((values > 0.0).mean()),
    }


def _permutation_control(
    rows: Sequence[Mapping[str, Any]],
    expert_scores: np.ndarray,
    *,
    folds: int,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    generators = [str(row["generator_name"]) for row in rows]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for repeat in range(repeats):
        shuffled = np.asarray(expert_scores)[rng.permutation(len(rows))]
        probabilities, _ = grouped_oof_fusion(rows, shuffled, folds=folds, seed=seed)
        report = classification_metrics(
            labels,
            (probabilities >= 0.5).astype(np.int64),
            scores=probabilities,
            generators=generators,
        )
        values.append(float(report["generator_macro_balanced_accuracy"]))
    array = np.asarray(values, dtype=np.float64)
    return {
        "repeats": repeats,
        "mean_generator_macro_balanced_accuracy": float(array.mean()),
        "median_generator_macro_balanced_accuracy": float(np.median(array)),
        "p95_generator_macro_balanced_accuracy": float(np.quantile(array, 0.95)),
        "maximum_generator_macro_balanced_accuracy": float(array.max()),
        "all_values": values,
    }


def _subgroup_quadrants(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: residual_quadrants(value) for key, value in sorted(groups.items())}


def write_items(path: Path, rows: Sequence[Mapping[str, Any]], fusion_scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video_id",
        "base_id",
        "label",
        "generator_name",
        "motion_bucket",
        "qwen_prediction",
        "expert_score",
        "expert_prediction",
        "error_quadrant",
        "fusion_oof_score",
        "fusion_oof_prediction",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, score in zip(rows, fusion_scores):
            q_correct = int(row["qwen_prediction"]) == int(row["label"])
            e_correct = int(row["expert_prediction"]) == int(row["label"])
            quadrant = (
                "both_correct"
                if q_correct and e_correct
                else "expert_only_correct"
                if e_correct
                else "qwen_only_correct"
                if q_correct
                else "both_wrong"
            )
            writer.writerow(
                {
                    **{field: row[field] for field in fields[:8]},
                    "error_quadrant": quadrant,
                    "fusion_oof_score": float(score),
                    "fusion_oof_prediction": int(score >= 0.5),
                }
            )


def write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    qwen = summary["models"]["qwen"]
    expert = summary["models"]["temporal_expert"]
    fusion = summary["models"]["grouped_oof_fusion"]
    quadrants = summary["residual_complementarity"]["overall"]
    delta = summary["deltas"]
    lines = [
        "# ViF-Bench 残余错误互补性审计",
        "",
        f"- 状态：**{summary['status']}**",
        f"- 有效配对样本：{summary['join_audit']['joined_valid_rows']}",
        f"- 专家分数：`{summary['settings']['expert_score_column']}`",
        "- 限制：融合器使用 ViF-Bench 标签做分组交叉验证，只是开发诊断，不是正式测试集提升。",
        "",
        "## 主要指标",
        "",
        "| 方法 | Balanced ACC | Generator-macro Balanced ACC | Fake F1 | AUROC |",
        "|---|---:|---:|---:|---:|",
        f"| Qwen 原判定 | {qwen['balanced_accuracy']:.4f} | {qwen['generator_macro_balanced_accuracy']:.4f} | {qwen['fake_f1']:.4f} | - |",
        f"| 时序专家 | {expert['balanced_accuracy']:.4f} | {expert['generator_macro_balanced_accuracy']:.4f} | {expert['fake_f1']:.4f} | {expert['roc_auc']:.4f} |",
        f"| 分组 OOF 融合 | {fusion['balanced_accuracy']:.4f} | {fusion['generator_macro_balanced_accuracy']:.4f} | {fusion['fake_f1']:.4f} | {fusion['roc_auc']:.4f} |",
        "",
        "## 互补错误",
        "",
        f"- 两者都对：{quadrants['counts'].get('both_correct', 0)}",
        f"- 仅专家正确（潜在 rescue）：{quadrants['counts'].get('expert_only_correct', 0)}",
        f"- 仅 Qwen 正确（潜在 harm）：{quadrants['counts'].get('qwen_only_correct', 0)}",
        f"- 两者都错：{quadrants['counts'].get('both_wrong', 0)}",
        f"- Oracle accuracy：{quadrants['oracle_accuracy']:.4f}",
        "",
        "## 验收",
        "",
        f"- 融合相对 Qwen 的 generator-macro Balanced ACC：{delta['fusion_minus_qwen_generator_macro_balanced_accuracy']:+.4f}",
        f"- 分组 bootstrap 95% CI：[{delta['fusion_minus_qwen_group_bootstrap']['ci95_lower']:+.4f}, {delta['fusion_minus_qwen_group_bootstrap']['ci95_upper']:+.4f}]",
        f"- 融合相对打乱专家分数 95 分位：{delta['fusion_minus_permuted_p95_generator_macro_balanced_accuracy']:+.4f}",
        "",
        summary["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    qwen, qwen_audit = load_qwen_predictions(args.qwen_predictions)
    expert, expert_audit = load_expert_items(
        args.expert_items_csv,
        score_column=args.expert_score_column,
        prediction_column=args.expert_prediction_column,
    )
    rows, join_audit = join_rows(qwen, expert)
    if len(rows) < args.min_samples:
        raise ValueError(f"only {len(rows)} valid joined rows; require at least {args.min_samples}")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    qwen_predictions = np.asarray([int(row["qwen_prediction"]) for row in rows], dtype=np.int64)
    expert_scores = np.asarray([float(row["expert_score"]) for row in rows], dtype=np.float64)
    expert_predictions = np.asarray([int(row["expert_prediction"]) for row in rows], dtype=np.int64)
    generators = [str(row["generator_name"]) for row in rows]

    qwen_report = classification_metrics(labels, qwen_predictions, generators=generators)
    expert_report = classification_metrics(
        labels, expert_predictions, scores=expert_scores, generators=generators
    )
    fusion_scores, fold_audit = grouped_oof_fusion(
        rows, expert_scores, folds=args.folds, seed=args.seed
    )
    fusion_predictions = (fusion_scores >= 0.5).astype(np.int64)
    fusion_report = classification_metrics(
        labels, fusion_predictions, scores=fusion_scores, generators=generators
    )
    bootstrap = grouped_bootstrap_delta(
        rows,
        fusion_predictions,
        qwen_predictions,
        iterations=args.bootstrap_iterations,
        seed=args.seed + 1000,
    )
    permutation = _permutation_control(
        rows,
        expert_scores,
        folds=args.folds,
        seed=args.seed + 2000,
        repeats=args.permutation_repeats,
    )
    qwen_macro = float(qwen_report["generator_macro_balanced_accuracy"])
    fusion_macro = float(fusion_report["generator_macro_balanced_accuracy"])
    fusion_gain = fusion_macro - qwen_macro
    permuted_margin = fusion_macro - float(permutation["p95_generator_macro_balanced_accuracy"])
    overall_quadrants = residual_quadrants(rows)
    checks = {
        "paired_coverage": min(join_audit["coverage_over_qwen"], join_audit["coverage_over_expert"])
        >= args.min_coverage,
        "qwen_format_valid": join_audit["qwen_format_valid_rate"] >= args.min_coverage,
        "no_label_mismatch": join_audit["label_mismatch_count"] == 0,
        "expert_has_rescue_candidates": (
            overall_quadrants["expert_rescue_rate_among_qwen_errors"] is not None
            and overall_quadrants["expert_rescue_rate_among_qwen_errors"] >= args.min_rescue_rate
        ),
        "oof_fusion_has_minimum_gain": fusion_gain >= args.min_fusion_gain,
        "group_bootstrap_ci_is_positive": bootstrap["ci95_lower"] > 0.0,
        "matched_expert_beats_permuted_p95": permuted_margin > 0.0,
        "all_folds_are_group_disjoint": all(item["group_overlap"] == 0 for item in fold_audit),
    }
    passed = all(checks.values())
    summary: dict[str, Any] = {
        "schema_version": "vifbench_residual_complementarity_audit_v1",
        "gate": "ViF-Bench 上 Qwen 与时序专家的残余错误互补性审计",
        "status": "passed_for_model_fusion" if passed else "failed",
        "what_was_tested": (
            "在相同 ViF-Bench 样本上，检查已训练 Qwen 检测器与相机无关时序证据专家是否犯不同错误，"
            "以及专家分数能否在按原视频身份隔离的 out-of-fold 诊断中提高最终 Real/Fake 判定。"
        ),
        "inputs": {"qwen": qwen_audit, "expert": expert_audit},
        "settings": {
            "expert_score_column": args.expert_score_column,
            "expert_prediction_column": args.expert_prediction_column,
            "folds": args.folds,
            "group_key": "base video identity shared by Real/Fake variants",
            "bootstrap_iterations": args.bootstrap_iterations,
            "permutation_repeats": args.permutation_repeats,
            "qwen_score_available": False,
            "fusion_features": ["Qwen hard Real/Fake prediction", "temporal expert continuous score"],
        },
        "join_audit": join_audit,
        "models": {
            "qwen": qwen_report,
            "temporal_expert": expert_report,
            "grouped_oof_fusion": fusion_report,
        },
        "residual_complementarity": {
            "overall": overall_quadrants,
            "by_generator": _subgroup_quadrants(rows, "generator_name"),
            "by_motion_bucket": _subgroup_quadrants(rows, "motion_bucket"),
        },
        "deltas": {
            "fusion_minus_qwen_balanced_accuracy": float(fusion_report["balanced_accuracy"])
            - float(qwen_report["balanced_accuracy"]),
            "fusion_minus_qwen_generator_macro_balanced_accuracy": fusion_gain,
            "fusion_minus_qwen_fake_f1": float(fusion_report["fake_f1"])
            - float(qwen_report["fake_f1"]),
            "fusion_minus_qwen_group_bootstrap": bootstrap,
            "fusion_minus_permuted_p95_generator_macro_balanced_accuracy": permuted_margin,
        },
        "permuted_expert_negative_control": permutation,
        "fold_audit": fold_audit,
        "thresholds": {
            "min_samples": args.min_samples,
            "min_coverage": args.min_coverage,
            "min_expert_rescue_rate_among_qwen_errors": args.min_rescue_rate,
            "min_oof_fusion_generator_macro_balanced_accuracy_gain": args.min_fusion_gain,
            "require_positive_group_bootstrap_ci_lower": True,
            "require_fusion_above_permuted_expert_p95": True,
        },
        "checks": checks,
        "does_not_establish": (
            "该审计没有证明正式 benchmark 提升：ViF-Bench 标签参与了交叉拟合；Qwen 只有硬标签而没有校准 logits；"
            "通过后仍需在 DataB/独立校准集训练路由器，并在未用于选择方案的 GenBuster benchmark 上测试。"
        ),
        "interpretation": (
            "通过：时序专家对 Qwen 残余错误具有样本对齐的可学习互补性，可以进入独立校准路由。"
            if passed
            else "未通过：即使允许分组交叉拟合，时序专家也没有形成稳健且超过打乱负对照的 Qwen 检测增量；停止该专家融合配方。"
        ),
        "next_action": (
            "在 DataB 留出集训练冻结专家路由器，并只在 GenBuster benchmark 做一次外部验收。"
            if passed
            else "不要训练 Qwen 或 RL；先依据逐样本 CSV 判断专家缺少信号还是错误不互补，再决定是否更换切入点。"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "vifbench_residual_complementarity_summary.json"
    items_path = args.output_dir / "vifbench_residual_complementarity_items.csv"
    report_path = args.output_dir / "vifbench_residual_complementarity_report.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_items(items_path, rows, fusion_scores)
    write_markdown(report_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-predictions", type=Path, required=True)
    parser.add_argument("--expert-items-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expert-score-column", default="evidence_only_score")
    parser.add_argument("--expert-prediction-column", default="evidence_only_prediction")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--permutation-repeats", type=int, default=100)
    parser.add_argument("--min-samples", type=int, default=3000)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--min-rescue-rate", type=float, default=0.10)
    parser.add_argument("--min-fusion-gain", type=float, default=0.005)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
