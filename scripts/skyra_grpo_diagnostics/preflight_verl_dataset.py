#!/usr/bin/env python3
"""Load two DataB samples through verl's real Qwen3-VL dataset path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-prompt-length", type=int, default=12288)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.verl_root).resolve()))

    from omegaconf import OmegaConf
    from transformers import AutoProcessor, AutoTokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    config = OmegaConf.create(
        {
            "cache_dir": "/tmp/verl_datab_preflight_cache",
            "prompt_key": "prompt",
            "image_key": "images",
            "video_key": "videos",
            "max_prompt_length": args.max_prompt_length,
            "return_raw_chat": False,
            "return_full_prompt": True,
            "truncation": "error",
            "filter_overlong_prompts": False,
            "filter_overlong_prompts_workers": 1,
            "return_multi_modal_inputs": True,
            "use_shm": False,
        }
    )
    dataset = RLHFDataset(
        data_files=args.train_parquet,
        tokenizer=tokenizer,
        config=config,
        processor=processor,
    )

    selected: dict[str, int] = {}
    for index in range(len(dataset)):
        label = dataset.dataframe[index]["reward_model"]["ground_truth"]
        selected.setdefault(label, index)
        if len(selected) == 2:
            break

    samples = []
    checks = {
        "processor_is_qwen3_vl": "Qwen3VLProcessor" in processor.__class__.__name__,
        "image_processor_is_qwen3_vl": "Qwen3VLImageProcessor"
        in processor.image_processor.__class__.__name__,
        "both_classes_loaded": set(selected) == {"Fake", "Real"},
        "all_samples_have_16_images": True,
        "all_position_ids_are_4d": True,
        "all_prompts_within_limit": True,
    }
    for label, index in sorted(selected.items()):
        item = dataset[index]
        input_length = int(item["input_ids"].shape[-1])
        position_shape = list(item["position_ids"].shape)
        image_count = len(item["multi_modal_data"].get("image", []))
        checks["all_samples_have_16_images"] &= image_count == 16
        checks["all_position_ids_are_4d"] &= len(position_shape) == 2 and position_shape[0] == 4
        checks["all_prompts_within_limit"] &= input_length <= args.max_prompt_length
        samples.append(
            {
                "label": label,
                "dataset_index": index,
                "input_length": input_length,
                "position_ids_shape": position_shape,
                "image_count": image_count,
                "multi_modal_input_keys": sorted(item.get("multi_modal_inputs", {}).keys()),
                "prompt_image_tokens": item.get("full_prompts", "").count("<|image_pad|>"),
            }
        )

    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "verl_root": str(Path(args.verl_root).resolve()),
        "model_path": args.model_path,
        "train_parquet": args.train_parquet,
        "dataset_records": len(dataset),
        "processor_class": processor.__class__.__name__,
        "image_processor_class": processor.image_processor.__class__.__name__,
        "checks": checks,
        "samples": samples,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
