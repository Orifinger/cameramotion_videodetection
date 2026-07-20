from __future__ import annotations

import csv
import json
import math
import re
from argparse import Namespace
from pathlib import Path

from scripts.vifbench_qwen_confidence.score_historical_answers import (
    answer_token_contract,
)
from tools.audit_vifbench_confidence_fusion import run


class DummyTokenizer:
    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        tokens = re.findall(r"Real|Fake|.", text, flags=re.DOTALL)
        ids: list[int] = []
        for token in tokens:
            if token not in self.token_to_id:
                token_id = len(self.token_to_id) + 1
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
            ids.append(self.token_to_id[token])
        return ids

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.id_to_token[token_id] for token_id in token_ids)


def test_answer_token_contract_finds_single_real_fake_substitution() -> None:
    tokenizer = DummyTokenizer()
    result = answer_token_contract(
        tokenizer, "<think>evidence</think>\n<answer>Fake</answer>"
    )
    assert result["valid"] is True
    assert result["archived_answer"] == "Fake"
    assert result["real_token_text"] == "Real"
    assert result["fake_token_text"] == "Fake"


class ContextSplitTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if "<answer>Real</answer>" in text:
            return [1, 2, 10, 20, 30]
        if "<answer>Fake</answer>" in text:
            return [1, 2, 11, 21, 30]
        raise ValueError(text)

    def decode(self, token_ids: list[int]) -> str:
        return {10: "Real", 11: "Fake"}.get(token_ids[0], "suffix")


def test_answer_token_contract_accepts_context_dependent_suffix_split() -> None:
    result = answer_token_contract(
        ContextSplitTokenizer(), "<answer>Real</answer>"
    )
    assert result["valid"] is True
    assert result["scoring_scope"] == "first_divergent_answer_token"
    assert result["real_token_id"] == 10
    assert result["fake_token_id"] == 11
    assert result["real_candidate_token_count"] == 2
    assert result["fake_candidate_token_count"] == 2

def test_confidence_fusion_audit_uses_grouped_oof(tmp_path: Path) -> None:
    confidence_dir = tmp_path / "confidence"
    confidence_dir.mkdir()
    confidence_rows = []
    expert_rows = []
    for index in range(60):
        for generator, label in (("real", 0), ("gen_a", 1), ("gen_b", 1)):
            video_id = f"{generator}/clip_{index:03d}"
            qwen_prediction = label if index % 4 else 1 - label
            margin = 2.0 if qwen_prediction else -2.0
            fake_probability = 1.0 / (1.0 + math.exp(-margin))
            confidence_rows.append(
                {
                    "video_id": video_id,
                    "status": "ok",
                    "archived_answer": "Fake" if qwen_prediction else "Real",
                    "fake_minus_real_logit_margin": margin,
                    "fake_pair_probability": fake_probability,
                    "score_matches_archived_answer": True,
                    "prompt_contract_sha256": "test-contract",
                }
            )
            evidence = 3.0 if label else -3.0
            expert_rows.append(
                {
                    "sample_id": (
                        f"/tmp/parsed_frames/parsed_frames/"
                        f"{'Fake' if label else 'Real'}/{generator}/clip_{index:03d}"
                    ),
                    "label": label,
                    "generator_name": generator,
                    "motion_bucket": "complex-motion" if index % 2 else "minor-motion",
                    "matched_score": evidence,
                    "evidence_only_score": evidence,
                    "shuffled_camera_score": float((index % 5) - 2),
                    "camera_only_score": float((index % 3) - 1),
                }
            )
    with (confidence_dir / "rank_00.jsonl").open("w", encoding="utf-8") as handle:
        for row in confidence_rows:
            handle.write(json.dumps(row) + "\n")
    expert_path = tmp_path / "expert.csv"
    with expert_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(expert_rows[0]))
        writer.writeheader()
        writer.writerows(expert_rows)

    output_dir = tmp_path / "output"
    summary = run(
        Namespace(
            confidence_scores=confidence_dir,
            expert_items_csv=expert_path,
            output_dir=output_dir,
            folds=5,
            seed=20260720,
            bootstrap_iterations=100,
            min_samples=100,
            min_coverage=0.99,
            min_score_answer_agreement=0.99,
            min_gain=0.005,
        )
    )

    assert summary["inputs"]["join"]["joined_valid_rows"] == 180
    assert (
        summary["models"]["qwen_plus_evidence"]["generator_macro_balanced_accuracy"]
        > summary["models"]["qwen_confidence"]["generator_macro_balanced_accuracy"]
    )
    assert summary["checks"]["all_grouped_folds_have_zero_group_overlap"] is True
    assert (output_dir / "vifbench_confidence_fusion_summary.json").is_file()
    assert (output_dir / "vifbench_confidence_fusion_items.csv").is_file()
    assert (output_dir / "vifbench_confidence_fusion_report.md").is_file()
