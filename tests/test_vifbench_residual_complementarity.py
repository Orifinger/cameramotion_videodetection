from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

from tools.audit_vifbench_residual_complementarity import canonical_video_id, run


def test_canonical_video_id_matches_qwen_and_expert_contracts() -> None:
    assert canonical_video_id("HunyuanVideo-I2V/abc_001") == "HunyuanVideo-I2V/abc_001"
    assert (
        canonical_video_id(
            "vif-bench:/tmp/1vif-bench/parsed_frames/parsed_frames/Fake/"
            "HunyuanVideo-I2V/abc_001"
        )
        == "HunyuanVideo-I2V/abc_001"
    )
    assert (
        canonical_video_id(
            "/tmp/1vif-bench/parsed_frames/parsed_frames/Real/Real/abc_001"
        )
        == "real/abc_001"
    )


def test_residual_audit_detects_complementary_expert(tmp_path: Path) -> None:
    qwen_rows = []
    expert_rows = []
    for index in range(30):
        base_id = f"clip_{index:03d}"
        for generator, label in (("real", 0), ("gen_a", 1), ("gen_b", 1)):
            video_id = f"{generator}/{base_id}"
            qwen_prediction = label
            if index % 4 == 0:
                qwen_prediction = 1 - label
            qwen_rows.append(
                {
                    "video_id": video_id,
                    "answer": "Fake" if qwen_prediction else "Real",
                    "aigc_model_name": "Real" if label == 0 else generator,
                }
            )
            score = (
                0.05 + 0.01 * (index % 3)
                if label == 0
                else 0.93 - 0.01 * (index % 3)
            )
            prefix = "Real/real" if label == 0 else f"Fake/{generator}"
            expert_rows.append(
                {
                    "sample_id": (
                        "/tmp/1vif-bench/parsed_frames/parsed_frames/"
                        f"{prefix}/{base_id}"
                    ),
                    "label": label,
                    "generator_name": "real" if label == 0 else generator,
                    "motion_bucket": "complex-motion" if index % 2 else "minor-motion",
                    "evidence_only_score": score,
                    "evidence_only_prediction": label,
                }
            )

    qwen_path = tmp_path / "qwen.json"
    qwen_path.write_text(json.dumps(qwen_rows), encoding="utf-8")
    expert_path = tmp_path / "expert.csv"
    with expert_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(expert_rows[0]))
        writer.writeheader()
        writer.writerows(expert_rows)

    output_dir = tmp_path / "output"
    summary = run(
        Namespace(
            qwen_predictions=qwen_path,
            expert_items_csv=expert_path,
            output_dir=output_dir,
            expert_score_column="evidence_only_score",
            expert_prediction_column="evidence_only_prediction",
            folds=5,
            seed=20260720,
            bootstrap_iterations=100,
            permutation_repeats=10,
            min_samples=50,
            min_coverage=0.99,
            min_rescue_rate=0.10,
            min_fusion_gain=0.005,
        )
    )

    assert summary["join_audit"]["joined_valid_rows"] == 90
    assert (
        summary["residual_complementarity"]["overall"]["counts"]["expert_only_correct"]
        > 0
    )
    assert summary["models"]["grouped_oof_fusion"]["balanced_accuracy"] == 1.0
    assert all(item["group_overlap"] == 0 for item in summary["fold_audit"])
    assert (output_dir / "vifbench_residual_complementarity_summary.json").is_file()
    assert (output_dir / "vifbench_residual_complementarity_items.csv").is_file()
    assert (output_dir / "vifbench_residual_complementarity_report.md").is_file()