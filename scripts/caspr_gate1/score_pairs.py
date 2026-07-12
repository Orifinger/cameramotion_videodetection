#!/usr/bin/env python3
"""Score DataA Real/Fake videos independently with the CASPR verdict prompt."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from scripts.caspr_gate1.runtime import (
    attach_adapter,
    candidate_token_ids,
    cleanup_distributed,
    init_distributed,
    load_model,
    load_processor,
    prepare_batch,
    read_jsonl,
    set_seed,
    verdict_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--pairs-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = init_distributed()
    set_seed(args.seed, rank)
    device = torch.device("cuda", local_rank)
    pairs = read_jsonl(args.pairs_jsonl)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    processor = load_processor(args.model_path)
    real_token_id, fake_token_id = candidate_token_ids(processor.tokenizer)
    model = load_model(args.model_path, args.attn_implementation, torch.bfloat16)
    if args.adapter_path:
        model = attach_adapter(model, args.adapter_path)
    model.to(device)
    model.eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rank_{rank:02d}.jsonl"
    written = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle, torch.inference_mode():
        for index, pair in enumerate(pairs):
            if index % world_size != rank:
                continue
            batch = prepare_batch([pair["real"], pair["fake"]], processor, device, args.max_pixels)
            outputs = model(**batch, use_cache=False)
            scores = verdict_scores(outputs.logits, real_token_id, fake_token_id)
            real_score, fake_score = float(scores[0]), float(scores[1])
            if not math.isfinite(real_score) or not math.isfinite(fake_score):
                raise FloatingPointError(f"non-finite score for pair {pair.get('pair_id')}")
            row = {
                "data_index": index,
                "pair_id": pair.get("pair_id"),
                "case_id": pair.get("case_id"),
                "source_family": pair.get("source_family"),
                "motion_bucket": pair.get("motion_bucket"),
                "camera_labels": pair.get("camera_labels", []),
                "camera_pair_consistent": pair.get("camera_pair_consistent"),
                "real_score": real_score,
                "fake_score": fake_score,
                "score_margin_fake_minus_real": fake_score - real_score,
                "real_prediction": "Fake" if real_score > 0 else "Real",
                "fake_prediction": "Fake" if fake_score > 0 else "Real",
                "model_name": args.model_name,
                "base_model": args.model_path,
                "adapter_path": args.adapter_path,
                "rank": rank,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written % 10 == 0:
                print(f"rank={rank} scored={written}", flush=True)
    print(f"rank={rank} saved={output_path} records={written}", flush=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
