#!/usr/bin/env python3
"""Grouped ViF-Bench audit for Qwen confidence and temporal/camera experts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from tools.audit_vifbench_residual_complementarity import (
    base_identity,
    canonical_video_id,
    classification_metrics,
)


ANSWER_TO_LABEL = {"real": 0, "fake": 1}


def read_jsonl_files(path: Path) -> list[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.glob("rank_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no confidence JSONL files found under {path}")
    rows: list[dict[str, Any]] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8-sig") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def load_confidence_rows(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    raw_rows = read_jsonl_files(path)
    rows: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    statuses = Counter()
    prompt_hashes = Counter()
    token_contract_valid = 0
    score_answer_matches = 0
    for raw in raw_rows:
        video_id = canonical_video_id(raw.get("video_id"))
        if video_id in rows:
            duplicate_ids.append(video_id)
            continue
        status = str(raw.get("status", "unknown"))
        statuses[status] += 1
        prompt_hashes[str(raw.get("prompt_contract_sha256", ""))] += 1
        if status != "ok":
            continue
        answer = str(raw.get("archived_answer", "")).strip().casefold()
        prediction = ANSWER_TO_LABEL.get(answer)
        margin = float(raw["fake_minus_real_logit_margin"])
        if prediction is None or not np.isfinite(margin):
            continue
        token_contract_valid += 1
        score_answer_matches += int(bool(raw.get("score_matches_archived_answer")))
        rows[video_id] = {
            "video_id": video_id,
            "prediction": prediction,
            "margin": margin,
            "fake_probability": float(raw["fake_pair_probability"]),
            "raw": raw,
        }
    if duplicate_ids:
        raise ValueError(f"duplicate confidence video IDs: {duplicate_ids[:10]}")
    return rows, {
        "path": str(path),
        "raw_rows": len(raw_rows),
        "valid_rows": len(rows),
        "status_counts": dict(statuses),
        "prompt_contract_hashes": dict(prompt_hashes),
        "token_contract_valid_rate": token_contract_valid / len(raw_rows) if raw_rows else 0.0,
        "score_answer_agreement_rate": (
            score_answer_matches / token_contract_valid if token_contract_valid else 0.0
        ),
    }


def load_expert_rows(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    required = {
        "sample_id",
        "label",
        "generator_name",
        "motion_bucket",
        "matched_score",
        "evidence_only_score",
        "shuffled_camera_score",
        "camera_only_score",
    }
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"expert CSV is missing columns: {sorted(missing)}")
        for raw in reader:
            video_id = canonical_video_id(raw["sample_id"])
            if video_id in rows:
                raise ValueError(f"duplicate expert video ID: {video_id}")
            parsed = {
                "video_id": video_id,
                "label": int(raw["label"]),
                "generator_name": str(raw["generator_name"]),
                "motion_bucket": str(raw["motion_bucket"]),
            }
            for name in (
                "matched_score",
                "evidence_only_score",
                "shuffled_camera_score",
                "camera_only_score",
            ):
                parsed[name] = float(raw[name])
                if not np.isfinite(parsed[name]):
                    raise ValueError(f"non-finite {name} for {video_id}")
            rows[video_id] = parsed
    return rows, {"path": str(path), "rows": len(rows)}


def join_rows(
    confidence: Mapping[str, Mapping[str, Any]],
    expert: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    label_mismatches: list[str] = []
    for video_id in sorted(set(confidence) & set(expert)):
        qwen = confidence[video_id]
        auxiliary = expert[video_id]
        expected = 0 if video_id.split("/", 1)[0].casefold() == "real" else 1
        if int(auxiliary["label"]) != expected:
            label_mismatches.append(video_id)
            continue
        joined.append(
            {
                "video_id": video_id,
                "base_id": base_identity(video_id),
                "label": expected,
                "generator_name": auxiliary["generator_name"],
                "motion_bucket": auxiliary["motion_bucket"],
                "qwen_prediction": int(qwen["prediction"]),
                "qwen_margin": float(qwen["margin"]),
                "qwen_fake_probability": float(qwen["fake_probability"]),
                "matched_score": float(auxiliary["matched_score"]),
                "evidence_only_score": float(auxiliary["evidence_only_score"]),
                "shuffled_camera_score": float(auxiliary["shuffled_camera_score"]),
                "camera_only_score": float(auxiliary["camera_only_score"]),
            }
        )
    return joined, {
        "confidence_rows": len(confidence),
        "expert_rows": len(expert),
        "id_intersection": len(set(confidence) & set(expert)),
        "joined_valid_rows": len(joined),
        "coverage_over_confidence": len(joined) / len(confidence) if confidence else 0.0,
        "coverage_over_expert": len(joined) / len(expert) if expert else 0.0,
        "label_mismatch_count": len(label_mismatches),
        "first_label_mismatches": label_mismatches[:20],
    }


def sample_weights(labels: np.ndarray, generators: np.ndarray) -> np.ndarray:
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


def grouped_oof_scores(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    *,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    groups = np.asarray([str(row["base_id"]) for row in rows], dtype=object)
    generators = np.asarray([str(row["generator_name"]) for row in rows], dtype=object)
    features = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64
    )
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    output = np.full(labels.size, np.nan, dtype=np.float64)
    fold_audit: list[dict[str, Any]] = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(features, labels, groups), start=1
    ):
        scaler = StandardScaler()
        train_features = scaler.fit_transform(features[train_index])
        test_features = scaler.transform(features[test_index])
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed + fold)
        weights = sample_weights(labels[train_index], generators[train_index])
        model.fit(train_features, labels[train_index], sample_weight=weights)
        output[test_index] = model.predict_proba(test_features)[:, 1]
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        fold_audit.append(
            {
                "fold": fold,
                "train_samples": int(train_index.size),
                "test_samples": int(test_index.size),
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                "group_overlap": len(train_groups & test_groups),
            }
        )
    if np.isnan(output).any():
        raise RuntimeError(f"OOF scores missing for features {feature_names}")
    return output, fold_audit


def generator_macro_bacc(
    labels: np.ndarray, predictions: np.ndarray, generators: np.ndarray
) -> float:
    value = classification_metrics(
        labels, predictions, generators=generators
    ).get("generator_macro_balanced_accuracy")
    if value is None:
        raise RuntimeError("generator-macro balanced accuracy is undefined")
    return float(value)


def grouped_bootstrap_delta(
    rows: Sequence[Mapping[str, Any]],
    left_predictions: np.ndarray,
    right_predictions: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    generators = np.asarray([str(row["generator_name"]) for row in rows], dtype=object)
    groups = np.asarray([str(row["base_id"]) for row in rows], dtype=object)
    group_to_indices: dict[str, np.ndarray] = {
        group: np.flatnonzero(groups == group) for group in sorted(set(groups))
    }
    group_names = np.asarray(list(group_to_indices), dtype=object)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        sampled_groups = rng.choice(group_names, size=len(group_names), replace=True)
        sampled = np.concatenate([group_to_indices[str(group)] for group in sampled_groups])
        if np.unique(labels[sampled]).size < 2:
            continue
        left = generator_macro_bacc(
            labels[sampled], left_predictions[sampled], generators[sampled]
        )
        right = generator_macro_bacc(
            labels[sampled], right_predictions[sampled], generators[sampled]
        )
        deltas.append(left - right)
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "iterations_requested": iterations,
        "iterations_valid": int(values.size),
        "mean_delta": float(values.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "probability_delta_gt_zero": float((values > 0).mean()),
    }


def metric_bundle(
    rows: Sequence[Mapping[str, Any]], scores: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    generators = [str(row["generator_name"]) for row in rows]
    return classification_metrics(
        labels, predictions, scores=scores, generators=generators
    )


def write_items(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
) -> None:
    base_fields = list(rows[0].keys())
    score_fields = [f"{name}_oof_score" for name in scores]
    prediction_fields = [f"{name}_prediction" for name in predictions]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_fields + score_fields + prediction_fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            output = dict(row)
            output.update({f"{name}_oof_score": float(value[index]) for name, value in scores.items()})
            output.update(
                {f"{name}_prediction": int(value[index]) for name, value in predictions.items()}
            )
            writer.writerow(output)


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    models = summary["models"]
    lines = [
        "# ViF-Bench 强检测置信度与时序/相机专家融合诊断",
        "",
        "本实验沿用历史强检测模型的最终答案，仅补算 `<answer>` 位置的 Real/Fake token logit，再用原视频分组的五折 OOF 检查专家是否能在低置信度样本上可靠纠错。",
        "",
        "| 条件 | Balanced ACC | 生成器宏平均 Balanced ACC | AUROC |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "qwen_hard": "历史 Qwen 硬预测",
        "qwen_confidence": "仅 Qwen logit 置信度校准",
        "qwen_plus_evidence": "Qwen + 无相机时序证据",
        "qwen_plus_matched_camera": "Qwen + 正确相机交互专家",
        "qwen_plus_shuffled_camera": "Qwen + 打乱相机交互专家",
        "qwen_plus_camera_only": "Qwen + 仅相机专家",
    }
    for name, label in labels.items():
        metric = models[name]
        lines.append(
            f"| {label} | {metric['balanced_accuracy']:.4f} | "
            f"{metric['generator_macro_balanced_accuracy']:.4f} | "
            f"{metric.get('roc_auc', float('nan')):.4f} |"
        )
    lines.extend(
        [
            "",
            f"- 总体状态：`{summary['status']}`",
            f"- 时序专家增量是否通过：`{summary['checks']['temporal_expert_adds_value']}`",
            f"- 正确相机交互是否通过：`{summary['checks']['matched_camera_adds_value']}`",
            f"- 下一步：{summary['next_action']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    confidence, confidence_audit = load_confidence_rows(args.confidence_scores)
    expert, expert_audit = load_expert_rows(args.expert_items_csv)
    rows, join_audit = join_rows(confidence, expert)
    if len(rows) < args.min_samples:
        raise RuntimeError(f"too few joined rows: {len(rows)} < {args.min_samples}")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    generators = np.asarray([str(row["generator_name"]) for row in rows], dtype=object)
    hard_qwen = np.asarray([int(row["qwen_prediction"]) for row in rows], dtype=np.int64)

    feature_contracts = {
        "qwen_confidence": ["qwen_margin"],
        "qwen_plus_evidence": ["qwen_margin", "evidence_only_score"],
        "qwen_plus_matched_camera": ["qwen_margin", "matched_score"],
        "qwen_plus_shuffled_camera": ["qwen_margin", "shuffled_camera_score"],
        "qwen_plus_camera_only": ["qwen_margin", "camera_only_score"],
    }
    scores: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {"qwen_hard": hard_qwen}
    fold_audits: dict[str, Any] = {}
    for name, feature_names in feature_contracts.items():
        model_scores, fold_audit = grouped_oof_scores(
            rows, feature_names, folds=args.folds, seed=args.seed
        )
        scores[name] = model_scores
        predictions[name] = (model_scores >= 0.5).astype(np.int64)
        fold_audits[name] = fold_audit
    models = {
        "qwen_hard": classification_metrics(labels, hard_qwen, generators=generators),
        **{
            name: metric_bundle(rows, scores[name], predictions[name])
            for name in feature_contracts
        },
    }

    comparisons = {
        "evidence_minus_qwen_confidence": grouped_bootstrap_delta(
            rows,
            predictions["qwen_plus_evidence"],
            predictions["qwen_confidence"],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 101,
        ),
        "matched_minus_qwen_confidence": grouped_bootstrap_delta(
            rows,
            predictions["qwen_plus_matched_camera"],
            predictions["qwen_confidence"],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 102,
        ),
        "matched_minus_evidence": grouped_bootstrap_delta(
            rows,
            predictions["qwen_plus_matched_camera"],
            predictions["qwen_plus_evidence"],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 103,
        ),
        "matched_minus_shuffled": grouped_bootstrap_delta(
            rows,
            predictions["qwen_plus_matched_camera"],
            predictions["qwen_plus_shuffled_camera"],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 104,
        ),
    }
    point_pairs = {
        "evidence_minus_qwen_confidence": ("qwen_plus_evidence", "qwen_confidence"),
        "matched_minus_qwen_confidence": ("qwen_plus_matched_camera", "qwen_confidence"),
        "matched_minus_evidence": ("qwen_plus_matched_camera", "qwen_plus_evidence"),
        "matched_minus_shuffled": ("qwen_plus_matched_camera", "qwen_plus_shuffled_camera"),
    }
    for name, (left, right) in point_pairs.items():
        comparisons[name]["point_delta"] = (
            models[left]["generator_macro_balanced_accuracy"]
            - models[right]["generator_macro_balanced_accuracy"]
        )
    temporal = comparisons["evidence_minus_qwen_confidence"]
    matched_vs_evidence = comparisons["matched_minus_evidence"]
    matched_vs_shuffled = comparisons["matched_minus_shuffled"]
    coverage_ok = (
        join_audit["coverage_over_expert"] >= args.min_coverage
        and confidence_audit["token_contract_valid_rate"] >= args.min_coverage
    )
    score_contract_ok = confidence_audit["score_answer_agreement_rate"] >= args.min_score_answer_agreement
    temporal_pass = (
        temporal["point_delta"] >= args.min_gain
        and temporal["ci95"][0] > 0.0
    )
    camera_pass = (
        matched_vs_evidence["point_delta"] >= args.min_gain
        and matched_vs_evidence["ci95"][0] > 0.0
        and matched_vs_shuffled["mean_delta"] >= args.min_gain
        and matched_vs_shuffled["ci95"][0] > 0.0
    )
    if coverage_ok and score_contract_ok and camera_pass:
        status = "camera_supported_for_external_validation"
        next_action = "在独立 DataB 校准集训练冻结路由器，并只在未参与方案选择的 GenBuster benchmark 做一次正式验证。"
    elif coverage_ok and score_contract_ok and temporal_pass:
        status = "temporal_supported_but_camera_failed"
        next_action = "相机主张停止；若继续，只将无相机时序专家带到独立校准集和 GenBuster benchmark。"
    else:
        status = "failed"
        next_action = "停止当前 camera/RAFT-DINO 专家融合配方，按强检测模型的残余错误类型重新选择专项专家。"

    summary: dict[str, Any] = {
        "gate": "ViF-Bench 强检测置信度条件下的时序与相机专家融合门",
        "status": status,
        "what_was_tested": (
            "历史强 Qwen 硬预测保持不变；补算 Real/Fake answer-token logit，"
            "在原视频分组五折 OOF 中比较仅置信度、无相机时序证据、正确相机交互、"
            "打乱相机交互和仅相机专家。"
        ),
        "development_only": (
            "ViF-Bench 标签参与 OOF 融合器拟合，因此本门只能选择方向，不能作为论文最终测试增益。"
        ),
        "thresholds": {
            "min_samples": args.min_samples,
            "min_coverage": args.min_coverage,
            "min_score_answer_agreement": args.min_score_answer_agreement,
            "min_generator_macro_balanced_accuracy_gain": args.min_gain,
            "bootstrap_ci_lower_must_exceed_zero": True,
        },
        "checks": {
            "coverage": coverage_ok,
            "answer_token_score_reproduces_archived_answer": score_contract_ok,
            "all_grouped_folds_have_zero_group_overlap": all(
                fold["group_overlap"] == 0
                for audits in fold_audits.values()
                for fold in audits
            ),
            "temporal_expert_adds_value": temporal_pass,
            "matched_camera_adds_value": camera_pass,
        },
        "inputs": {
            "confidence": confidence_audit,
            "expert": expert_audit,
            "join": join_audit,
        },
        "feature_contracts": feature_contracts,
        "models": models,
        "comparisons": comparisons,
        "fold_audits": fold_audits,
        "next_action": next_action,
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "vifbench_confidence_fusion_summary.json"
    items_path = output_dir / "vifbench_confidence_fusion_items.csv"
    report_path = output_dir / "vifbench_confidence_fusion_report.md"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_items(items_path, rows, scores, predictions)
    write_report(report_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary: {summary_path}")
    print(f"Saved items: {items_path}")
    print(f"Saved report: {report_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence-scores", type=Path, required=True)
    parser.add_argument("--expert-items-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--min-samples", type=int, default=3000)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--min-score-answer-agreement", type=float, default=0.99)
    parser.add_argument("--min-gain", type=float, default=0.005)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
