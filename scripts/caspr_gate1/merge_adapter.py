#!/usr/bin/env python3
"""Merge a CASPR PEFT adapter for the existing VIF-Bench inference scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.caspr_gate1.runtime import attach_adapter, load_model, load_processor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attn-implementation", default="")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    processor = load_processor(args.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model_path, args.attn_implementation, torch.bfloat16)
    model = attach_adapter(model, args.adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")


if __name__ == "__main__":
    main()
