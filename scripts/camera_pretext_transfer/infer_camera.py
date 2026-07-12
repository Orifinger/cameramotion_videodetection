#!/usr/bin/env python3
"""Generate camera-motion labels for the fixed DataA development records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.caspr_gate1.runtime import (
    attach_adapter, cleanup_distributed, init_distributed, load_model, load_processor,
    prepare_batch, read_jsonl, set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = init_distributed()
    set_seed(args.seed, rank)
    device = torch.device("cuda", local_rank)
    rows = read_jsonl(args.eval_jsonl)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    processor = load_processor(args.model_path)
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
        for index, sample in enumerate(rows):
            if index % world_size != rank:
                continue
            batch = prepare_batch([sample], processor, device, args.max_pixels)
            input_length = int(batch["input_ids"].shape[1])
            generated = model.generate(
                **batch, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
            )
            response = processor.batch_decode(
                generated[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0].strip()
            result = {
                "data_index": index,
                "case_id": sample.get("case_id"),
                "camera_labels": sample.get("camera_labels", []),
                "motion_bucket": sample.get("motion_bucket"),
                "source_family": sample.get("source_family"),
                "prompt_variant": sample.get("prompt_variant"),
                "response": response,
                "model_name": args.model_name,
                "base_model": args.model_path,
                "adapter_path": args.adapter_path,
                "rank": rank,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
            if written % 10 == 0:
                print(f"rank={rank} generated={written}", flush=True)
    print(f"rank={rank} saved={output_path} records={written}", flush=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
