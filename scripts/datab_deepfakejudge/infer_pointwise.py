from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


IMAGE_TOKEN_RE = re.compile(r"<image>")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(dict(row))
    return rows


def structured_messages(sample: Mapping[str, Any], max_pixels: int) -> list[dict[str, Any]]:
    raw_messages = sample.get("messages")
    raw_images = sample.get("images")
    if not isinstance(raw_messages, list) or not isinstance(raw_images, list):
        raise ValueError("sample must contain messages and images lists")
    image_iter = iter(str(path) for path in raw_images)
    used = 0
    output: list[dict[str, Any]] = []
    for message in raw_messages:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        parts: list[dict[str, Any]] = []
        cursor = 0
        for match in IMAGE_TOKEN_RE.finditer(content):
            if match.start() > cursor:
                parts.append({"type": "text", "text": content[cursor : match.start()]})
            try:
                image_path = next(image_iter)
            except StopIteration as exc:
                raise ValueError("more <image> tokens than image paths") from exc
            parts.append({"type": "image", "image": image_path, "max_pixels": max_pixels})
            used += 1
            cursor = match.end()
        if cursor < len(content):
            parts.append({"type": "text", "text": content[cursor:]})
        output.append({"role": role, "content": parts if parts else content})
    if used != len(raw_images):
        raise ValueError(f"image token/path mismatch: used={used}, paths={len(raw_images)}")
    return output


def distributed_setup() -> tuple[int, int, int, Any]:
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def distributed_cleanup() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def load_model(model_path: str, device: Any, attn_implementation: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        model_class = Qwen2_5_VLForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForImageTextToText

        model_class = AutoModelForImageTextToText
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = model_class.from_pretrained(model_path, **kwargs).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    return model, processor


def process_vision(messages: Sequence[Mapping[str, Any]]) -> tuple[list[Any], list[Any]]:
    from qwen_vl_utils import process_vision_info

    result = process_vision_info(messages)
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("unexpected qwen_vl_utils.process_vision_info result")
    return list(result[0] or []), list(result[1] or [])


def infer_one(
    sample: Mapping[str, Any],
    model: Any,
    processor: Any,
    device: Any,
    max_pixels: int,
    max_new_tokens: int,
) -> str:
    import torch

    messages = structured_messages(sample, max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision(messages)
    kwargs: dict[str, Any] = {
        "text": [text],
        "padding": True,
        "return_tensors": "pt",
    }
    if images:
        kwargs["images"] = images
    if videos:
        kwargs["videos"] = videos
    inputs = processor(**kwargs)
    inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    prompt_length = inputs["input_ids"].shape[1]
    output_ids = generated[:, prompt_length:]
    return processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def merge_shards(output_dir: Path, expected: int, world_size: int) -> None:
    rows: list[dict[str, Any]] = []
    for rank in range(world_size):
        shard = output_dir / "shards" / f"rank_{rank:02d}.jsonl"
        if not shard.is_file():
            raise FileNotFoundError(f"missing prediction shard: {shard}")
        rows.extend(read_jsonl(shard))
    rows.sort(key=lambda row: str(row.get("judge_id", "")))
    if len(rows) != expected:
        raise RuntimeError(f"prediction count mismatch: expected={expected}, actual={len(rows)}")
    judge_ids = [str(row.get("judge_id", "")) for row in rows]
    if len(judge_ids) != len(set(judge_ids)):
        raise RuntimeError("duplicate judge_id values found while merging predictions")
    output = output_dir / "predictions.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "task": "DataB DeepfakeJudge pointwise inference",
        "expected_records": expected,
        "prediction_records": len(rows),
        "world_size": world_size,
        "predictions_jsonl": str(output),
    }
    (output_dir / "inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepfakeJudge pointwise inference on DataB records.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main() -> None:
    import torch
    import torch.distributed as dist

    args = parse_args()
    rank, _local_rank, world_size, device = distributed_setup()
    torch.manual_seed(args.seed + rank)
    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.input_jsonl)
    local_records = records[rank::world_size]
    model, processor = load_model(args.model_path, device, args.attn_implementation)
    shard_path = shard_dir / f"rank_{rank:02d}.jsonl"
    started = time.time()
    with shard_path.open("w", encoding="utf-8") as handle:
        for local_index, sample in enumerate(local_records, 1):
            sample_started = time.time()
            try:
                prediction = infer_one(
                    sample,
                    model,
                    processor,
                    device,
                    args.max_pixels,
                    args.max_new_tokens,
                )
                error = None
            except Exception as exc:
                prediction = ""
                error = f"{type(exc).__name__}: {exc}"
            row = {
                "judge_id": sample.get("judge_id"),
                "sample_id": sample.get("sample_id"),
                "variant": sample.get("variant"),
                "prediction": prediction,
                "error": error,
                "metadata": sample.get("metadata", {}),
                "elapsed_seconds": time.time() - sample_started,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if local_index % 10 == 0:
                print(
                    f"rank={rank} completed={local_index}/{len(local_records)} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        merge_shards(output_dir, len(records), world_size)
    distributed_cleanup()


if __name__ == "__main__":
    main()
