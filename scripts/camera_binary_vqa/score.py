#!/usr/bin/env python3
"""Score Yes versus No for one or more binary camera VQA conditions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from scripts.camera_binary_vqa.runtime import (
    answer_token_ids,
    attach_adapter,
    cleanup_distributed,
    init_distributed,
    load_model,
    load_processor,
    prepare_prompt_batch,
    read_jsonl,
    set_seed,
    yes_minus_no_score,
)


def condition_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("condition must be NAME=JSONL")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("condition must be NAME=JSONL")
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--condition", action="append", type=condition_spec, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-stage", required=True)
    parser.add_argument("--max-samples-per-condition", type=int, default=0)
    parser.add_argument("--video-max-pixels", type=int, default=16384)
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--cpu-threads-per-rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = init_distributed()
    torch.set_num_threads(max(1, args.cpu_threads_per_rank))
    set_seed(args.seed, rank)
    device = torch.device("cuda", local_rank)
    processor = load_processor(args.model_path)
    no_token_id, yes_token_id = answer_token_ids(processor.tokenizer)
    model = load_model(args.model_path, args.attn_implementation, torch.bfloat16)
    if args.adapter_path:
        model = attach_adapter(model, args.adapter_path)
    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rank_{rank:02d}.jsonl"
    total_written = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle, torch.inference_mode():
        for condition_name, jsonl_path in args.condition:
            rows = read_jsonl(jsonl_path)
            if args.max_samples_per_condition > 0:
                rows = rows[: args.max_samples_per_condition]
            condition_written = 0
            for index, sample in enumerate(rows):
                if index % world_size != rank:
                    continue
                batch = prepare_prompt_batch(
                    sample,
                    processor,
                    device,
                    args.video_max_pixels,
                    args.video_fps,
                )
                outputs = model(**batch, use_cache=False)
                score = float(yes_minus_no_score(outputs.logits, no_token_id, yes_token_id)[0])
                if not math.isfinite(score):
                    raise FloatingPointError(
                        f"non-finite score for condition={condition_name} sample={sample.get('sample_id')}"
                    )
                prediction = "Yes" if score >= 0.0 else "No"
                result = {
                    "condition": condition_name,
                    "data_index": index,
                    "sample_id": sample.get("sample_id"),
                    "pair_id": sample.get("pair_id"),
                    "case_id": sample.get("case_id"),
                    "visual_source_case_id": sample.get("visual_source_case_id"),
                    "camera_primitive": sample.get("camera_primitive"),
                    "answer": sample.get("answer"),
                    "answer_id": sample.get("answer_id"),
                    "yes_minus_no_score": score,
                    "yes_probability": 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, score)))),
                    "prediction": prediction,
                    "model_stage": args.model_stage,
                    "base_model": args.model_path,
                    "adapter_path": args.adapter_path,
                    "rank": rank,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                condition_written += 1
                total_written += 1
                if condition_written % 20 == 0:
                    print(
                        f"rank={rank} condition={condition_name} scored={condition_written}",
                        flush=True,
                    )
    print(f"rank={rank} saved={output_path} records={total_written}", flush=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
