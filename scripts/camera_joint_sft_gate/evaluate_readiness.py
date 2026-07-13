#!/usr/bin/env python3
"""Audit binary-camera rollout exploration before spending compute on GRPO."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.caspr_gate1.runtime import read_jsonl, write_json


def load_rollouts(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.glob("rank_*.jsonl"))
    rows: list[dict[str, Any]] = []
    for file_path in files:
        rows.extend(read_jsonl(file_path))
    return rows


def parse_binary_response(response: str) -> str | None:
    stripped = response.strip()
    if stripped.casefold() == "yes":
        return "Yes"
    if stripped.casefold() == "no":
        return "No"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-jsonl", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--expected-k", type=int, default=8)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--min-format-pass-at-k", type=float, default=0.90)
    parser.add_argument("--min-correct-pass-at-k", type=float, default=0.50)
    parser.add_argument("--min-both-answers-rate", type=float, default=0.10)
    parser.add_argument("--min-reward-variance-rate", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_rows = read_jsonl(args.gold_jsonl)
    gold = {str(row["sample_id"]): row for row in gold_rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_rollouts(args.rollouts):
        grouped[str(row.get("sample_id"))].append(row)

    item_metrics: list[dict[str, Any]] = []
    for sample_id, expected_row in gold.items():
        rows = sorted(grouped.get(sample_id, []), key=lambda row: int(row.get("rollout_index", -1)))
        expected = str(expected_row["answer"])
        rewards: list[float] = []
        valid_answers: list[str] = []
        responses: list[str] = []
        for row in rows:
            response = str(row.get("response", ""))
            parsed = parse_binary_response(response)
            format_valid = parsed is not None
            correct = parsed == expected
            reward = 0.10 * float(format_valid) + 0.90 * float(correct)
            rewards.append(reward)
            if parsed is not None:
                valid_answers.append(parsed)
            responses.append(response)
        variance = statistics.pvariance(rewards) if len(rewards) > 1 else 0.0
        item_metrics.append(
            {
                "sample_id": sample_id,
                "case_id": expected_row.get("case_id"),
                "camera_primitive": expected_row.get("camera_primitive"),
                "expected_answer": expected,
                "num_rollouts": len(rows),
                "format_pass": bool(valid_answers),
                "correct_pass": expected in valid_answers,
                "both_answers_sampled": set(valid_answers) == {"Yes", "No"},
                "reward_variance": variance,
                "reward_has_variance": variance > 1e-8,
                "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
                "num_unique_responses": len(set(responses)),
            }
        )

    total = len(item_metrics)
    matched = sum(bool(item["num_rollouts"]) for item in item_metrics)

    def rate(key: str) -> float:
        return sum(bool(item[key]) for item in item_metrics) / total if total else 0.0

    metrics = {
        "num_gold_samples": total,
        "num_samples_with_rollouts": matched,
        "coverage": matched / total if total else 0.0,
        "expected_k": args.expected_k,
        "rollout_count_distribution": dict(Counter(item["num_rollouts"] for item in item_metrics)),
        "all_samples_have_expected_k": all(
            item["num_rollouts"] == args.expected_k for item in item_metrics
        ),
        "format_pass_at_k": rate("format_pass"),
        "correct_answer_pass_at_k": rate("correct_pass"),
        "both_answers_sampled_rate": rate("both_answers_sampled"),
        "reward_variance_rate": rate("reward_has_variance"),
        "mean_group_reward": (
            statistics.fmean(item["mean_reward"] for item in item_metrics) if item_metrics else 0.0
        ),
        "mean_unique_responses": (
            statistics.fmean(item["num_unique_responses"] for item in item_metrics)
            if item_metrics else 0.0
        ),
    }
    checks = {
        "coverage": metrics["coverage"] >= args.min_coverage,
        "expected_rollout_count": metrics["all_samples_have_expected_k"],
        "format_exploration": metrics["format_pass_at_k"] >= args.min_format_pass_at_k,
        "correct_answer_exploration": (
            metrics["correct_answer_pass_at_k"] >= args.min_correct_pass_at_k
        ),
        "both_binary_actions_explored": (
            metrics["both_answers_sampled_rate"] >= args.min_both_answers_rate
        ),
        "nonconstant_group_rewards": (
            metrics["reward_variance_rate"] >= args.min_reward_variance_rate
        ),
    }
    if all(checks.values()):
        status = "rl_ready"
        next_action = "可以进入短程二元相机 VQA GRPO 验证，暂不启动完整 RL。"
    elif checks["coverage"] and metrics["reward_variance_rate"] > 0:
        status = "borderline"
        next_action = "仅允许短程 GRPO；先检查答案探索率和逐题奖励，不能直接跑完整 RL。"
    else:
        status = "not_ready"
        next_action = "不要启动 GRPO；先修复输出格式、采样探索或奖励饱和问题。"
    summary = {
        "gate": "二元相机 VQA 的 GRPO 前置可训练性检查",
        "status": status,
        "what_was_tested": (
            "在留出 DataA 的同一条二元相机问题上采样 K 个 Yes/No 回答，以完全可验证的"
            "格式奖励和答案正确奖励检查探索与组内奖励方差；这里不检验下游检测是否提升。"
        ),
        "reward": {
            "format": 0.10,
            "correct_answer": 0.90,
            "maximum": 1.0,
        },
        "thresholds": {
            "min_coverage": args.min_coverage,
            "min_format_pass_at_k": args.min_format_pass_at_k,
            "min_correct_pass_at_k": args.min_correct_pass_at_k,
            "min_both_answers_rate": args.min_both_answers_rate,
            "min_reward_variance_rate": args.min_reward_variance_rate,
        },
        "checks": checks,
        "metrics": metrics,
        "next_action": next_action,
        "items": item_metrics,
    }
    write_json(args.output_json, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "items"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
