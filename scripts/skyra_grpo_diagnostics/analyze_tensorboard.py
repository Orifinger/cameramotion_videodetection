#!/usr/bin/env python3
"""Export TensorBoard scalars and diagnostic plots for a GRPO run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--group-count", type=int, default=16)
    return parser.parse_args()


def load_scalars(event_dir: Path) -> pd.DataFrame:
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    frames = []
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        frames.append(
            pd.DataFrame(
                {
                    "step": [event.step for event in events],
                    tag: [event.value for event in events],
                }
            ).set_index("step")
        )
    if not frames:
        raise RuntimeError(f"no scalar events found under {event_dir}")
    return pd.concat(frames, axis=1).sort_index()


def rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1, center=True).mean()


def plot_lines(
    ax: plt.Axes,
    frame: pd.DataFrame,
    tags: list[str],
    labels: list[str],
    window: int,
    *,
    ylim: tuple[float, float] | None = None,
) -> None:
    for tag, label in zip(tags, labels, strict=True):
        if tag not in frame:
            continue
        line = ax.plot(frame.index, frame[tag], alpha=0.18, linewidth=0.8)[0]
        ax.plot(
            frame.index,
            rolling(frame[tag], window),
            label=label,
            color=line.get_color(),
            linewidth=2.0,
        )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("GRPO step")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def window_stats(frame: pd.DataFrame, tag: str, width: int = 10) -> dict[str, float]:
    values = frame[tag].dropna()
    slope = float(np.polyfit(values.index.to_numpy(), values.to_numpy(), 1)[0])
    return {
        "mean": float(values.mean()),
        "std_across_steps": float(values.std(ddof=0)),
        "first": float(values.iloc[0]),
        "last": float(values.iloc[-1]),
        "first_window_mean": float(values.iloc[:width].mean()),
        "last_window_mean": float(values.iloc[-width:].mean()),
        "last_minus_first_window": float(values.iloc[-width:].mean() - values.iloc[:width].mean()),
        "min": float(values.min()),
        "min_step": int(values.idxmin()),
        "max": float(values.max()),
        "max_step": int(values.idxmax()),
        "linear_slope_per_step": slope,
    }


def correlation(frame: pd.DataFrame, left: str, right: str) -> float | None:
    values = frame[[left, right]].dropna()
    if len(values) < 2 or values[left].std() == 0 or values[right].std() == 0:
        return None
    return float(values[left].corr(values[right]))


def build_summary(frame: pd.DataFrame, group_count: int) -> dict[str, object]:
    selected = [
        "reward_extra/score/mean",
        "reward_extra/accuracy_reward/mean",
        "reward_extra/inspection_reward/mean",
        "reward_extra/correct/mean",
        "reward_extra/format_valid/mean",
        "reward_extra/pred_fake/mean",
        "reward_extra/gt_fake/mean",
        "reward_extra/false_positive/mean",
        "reward_extra/false_negative/mean",
        "reward_extra/raw_check_count/mean",
        "reward_extra/strict_check_count/mean",
        "reward_extra/duplicate_check_count/mean",
        "reward_extra/invalid_check_count/mean",
        "reward_extra/wrong_positive_reward/mean",
        "reward_extra/response_chars/mean",
        "reward_extra/grpo_zero_std_group_rate",
        "reward_extra/grpo_group_reward_std_mean",
        "actor/entropy",
        "actor/kl_loss",
        "actor/pg_loss",
        "actor/grad_norm",
        "actor/pg_clipfrac",
        "response_length/mean",
        "response_length/clip_ratio",
        "perf/time_per_step",
        "perf/throughput",
        "perf/max_memory_allocated_gb",
        "perf/max_memory_reserved_gb",
    ]
    metrics = {tag: window_stats(frame, tag) for tag in selected if tag in frame}
    zero_std = frame["reward_extra/grpo_zero_std_group_rate"]
    summary: dict[str, object] = {
        "num_steps": int(len(frame)),
        "first_step": int(frame.index.min()),
        "last_step": int(frame.index.max()),
        "metrics": metrics,
        "derived": {
            "mean_informative_group_rate": float(1.0 - zero_std.mean()),
            "mean_informative_groups_per_step": float((1.0 - zero_std.mean()) * group_count),
            "last_window_informative_groups_per_step": float(
                (1.0 - zero_std.iloc[-10:].mean()) * group_count
            ),
            "steps_with_zero_std_rate_ge_0_75": int((zero_std >= 0.75).sum()),
            "steps_with_perfect_rollout_accuracy": int(
                (frame["reward_extra/correct/mean"] == 1.0).sum()
            ),
            "steps_with_perfect_format": int(
                (frame["reward_extra/format_valid/mean"] == 1.0).sum()
            ),
            "estimated_prompt_fraction_of_epoch": float(len(frame) * group_count / 6252),
        },
        "correlations": {
            "score_vs_correct": correlation(
                frame, "reward_extra/score/mean", "reward_extra/correct/mean"
            ),
            "score_vs_inspection_reward": correlation(
                frame,
                "reward_extra/score/mean",
                "reward_extra/inspection_reward/mean",
            ),
            "score_vs_raw_check_count": correlation(
                frame,
                "reward_extra/score/mean",
                "reward_extra/raw_check_count/mean",
            ),
            "raw_check_count_vs_response_length": correlation(
                frame,
                "reward_extra/raw_check_count/mean",
                "response_length/mean",
            ),
            "score_vs_group_reward_std": correlation(
                frame,
                "reward_extra/score/mean",
                "reward_extra/grpo_group_reward_std_mean",
            ),
            "zero_std_rate_vs_group_reward_std": correlation(
                frame,
                "reward_extra/grpo_zero_std_group_rate",
                "reward_extra/grpo_group_reward_std_mean",
            ),
        },
    }
    return summary


def make_plots(frame: pd.DataFrame, output_dir: Path, window: int) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    plot_lines(
        axes[0],
        frame,
        [
            "reward_extra/score/mean",
            "reward_extra/accuracy_reward/mean",
            "reward_extra/inspection_reward/mean",
        ],
        ["total score", "accuracy component", "inspection component"],
        window,
    )
    axes[0].set_title("Reward and reward components")
    plot_lines(
        axes[1],
        frame,
        [
            "reward_extra/correct/mean",
            "reward_extra/format_valid/mean",
            "reward_extra/wrong_positive_reward/mean",
        ],
        ["rollout accuracy", "format valid", "wrong but positive reward"],
        window,
        ylim=(-0.02, 1.04),
    )
    axes[1].set_title("Task correctness and reward leakage")
    save_figure(fig, output_dir, "01_reward_and_accuracy.png")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    plot_lines(
        axes[0],
        frame,
        ["reward_extra/gt_fake/mean", "reward_extra/pred_fake/mean"],
        ["ground-truth Fake rate", "predicted Fake rate"],
        window,
        ylim=(-0.02, 1.02),
    )
    axes[0].set_title("Class balance and prediction bias")
    plot_lines(
        axes[1],
        frame,
        ["reward_extra/false_positive/mean", "reward_extra/false_negative/mean"],
        ["Real predicted Fake", "Fake predicted Real"],
        window,
        ylim=(-0.005, 0.13),
    )
    axes[1].set_title("Error directions")
    save_figure(fig, output_dir, "02_classification_bias.png")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    plot_lines(
        axes[0],
        frame,
        [
            "reward_extra/raw_check_count/mean",
            "reward_extra/strict_check_count/mean",
        ],
        ["raw evidence blocks", "structurally valid unique blocks"],
        window,
    )
    axes[0].set_title("Evidence-count behavior")
    plot_lines(
        axes[1],
        frame,
        [
            "reward_extra/duplicate_check_count/mean",
            "reward_extra/invalid_check_count/mean",
            "reward_extra/wrong_positive_reward/mean",
        ],
        ["duplicate", "invalid", "wrong but positively rewarded"],
        window,
        ylim=(-0.005, 0.13),
    )
    axes[1].set_title("Structural reward-hacking indicators")
    save_figure(fig, output_dir, "03_evidence_and_hacking.png")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    plot_lines(
        axes[0],
        frame,
        ["reward_extra/grpo_zero_std_group_rate"],
        ["zero-reward-variance group rate"],
        window,
        ylim=(-0.02, 1.02),
    )
    axes[0].set_title("Fraction of prompt groups with zero GRPO signal")
    plot_lines(
        axes[1],
        frame,
        ["reward_extra/grpo_group_reward_std_mean"],
        ["mean within-group reward std"],
        window,
    )
    axes[1].set_title("Within-group reward variation")
    save_figure(fig, output_dir, "04_grpo_learning_signal.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    plot_lines(axes[0, 0], frame, ["actor/entropy"], ["entropy"], window)
    axes[0, 0].set_title("Policy entropy")
    plot_lines(axes[0, 1], frame, ["actor/kl_loss"], ["KL to reference"], window)
    axes[0, 1].set_title("Reference-policy KL loss")
    plot_lines(axes[1, 0], frame, ["actor/pg_loss"], ["policy-gradient loss"], window)
    axes[1, 0].set_title("Policy-gradient loss")
    plot_lines(axes[1, 1], frame, ["actor/grad_norm"], ["gradient norm"], window)
    axes[1, 1].set_title("Gradient norm")
    save_figure(fig, output_dir, "05_optimizer_stability.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    plot_lines(
        axes[0, 0],
        frame,
        ["response_length/mean"],
        ["mean response tokens"],
        window,
    )
    axes[0, 0].set_title("Response length")
    plot_lines(
        axes[0, 1],
        frame,
        ["perf/time_per_step"],
        ["seconds per step"],
        window,
    )
    axes[0, 1].set_title("Step time")
    plot_lines(
        axes[1, 0],
        frame,
        ["perf/throughput"],
        ["throughput"],
        window,
    )
    axes[1, 0].set_title("Training throughput")
    plot_lines(
        axes[1, 1],
        frame,
        ["perf/max_memory_allocated_gb", "perf/max_memory_reserved_gb"],
        ["allocated GB", "reserved GB"],
        window,
    )
    axes[1, 1].set_title("Peak GPU memory")
    save_figure(fig, output_dir, "06_length_and_system.png")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_scalars(args.event_dir)
    frame.to_csv(args.output_dir / "all_scalars.csv", index_label="step")
    summary = build_summary(frame, args.group_count)
    (args.output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_plots(frame, args.output_dir, args.smooth_window)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
