from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

IMAGE_TOKEN_RE = re.compile(r"<image>")
REAL_CANDIDATE = " Real"
FAKE_CANDIDATE = " Fake"


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
            if isinstance(row, Mapping):
                rows.append(dict(row))
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def distributed_info() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def init_distributed() -> tuple[int, int, int]:
    import torch
    import torch.distributed as dist

    rank, local_rank, world_size = distributed_info()
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0) -> None:
    import torch

    value = seed + rank
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def candidate_token_ids(tokenizer: Any) -> tuple[int, int]:
    ids: dict[str, int] = {}
    for label, candidate in (("Real", REAL_CANDIDATE), ("Fake", FAKE_CANDIDATE)):
        encoded = tokenizer.encode(candidate, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"CASPR Gate 1 requires a one-token candidate for {candidate!r}; "
                f"tokenizer produced {encoded}. Do not continue with a silently changed score."
            )
        ids[label] = int(encoded[0])
    if ids["Real"] == ids["Fake"]:
        raise ValueError("Real and Fake candidate token ids are identical")
    return ids["Real"], ids["Fake"]


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
        if "<image>" not in content:
            output.append({"role": role, "content": content})
            continue
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
        output.append({"role": role, "content": parts})
    if used != len(raw_images):
        raise ValueError(f"image token/path mismatch: used={used}, paths={len(raw_images)}")
    return output


def _vision_inputs(
    conversation: Sequence[Mapping[str, Any]], image_patch_size: int
) -> tuple[list[Any], list[Any]]:
    from qwen_vl_utils import process_vision_info

    result = process_vision_info(conversation, image_patch_size=image_patch_size)
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("unexpected qwen_vl_utils.process_vision_info result")
    return list(result[0] or []), list(result[1] or [])


def prepare_batch(
    samples: Sequence[Mapping[str, Any]], processor: Any, device: Any, max_pixels: int
) -> dict[str, Any]:
    conversations = [structured_messages(sample, max_pixels) for sample in samples]
    texts = [
        processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        + str(sample.get("assistant_prefix", "<verdict>"))
        for conversation, sample in zip(conversations, samples)
    ]
    images: list[Any] = []
    videos: list[Any] = []
    image_processor = getattr(processor, "image_processor", None)
    image_patch_size = int(getattr(image_processor, "patch_size", 16))
    for conversation in conversations:
        sample_images, sample_videos = _vision_inputs(conversation, image_patch_size)
        images.extend(sample_images)
        videos.extend(sample_videos)
    kwargs: dict[str, Any] = {
        "text": texts,
        "padding": True,
        "return_tensors": "pt",
        "do_resize": False,
    }
    if images:
        kwargs["images"] = images
    if videos:
        kwargs["videos"] = videos
    batch = processor(**kwargs)
    return {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in batch.items()}


def load_processor(model_path: str) -> Any:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    return processor


def load_model(model_path: str, attn_implementation: str, dtype: Any) -> Any:
    from transformers import AutoModelForCausalLM

    try:
        from transformers import AutoModelForImageTextToText

        auto_class = AutoModelForImageTextToText
    except ImportError:
        auto_class = AutoModelForCausalLM
    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        return auto_class.from_pretrained(model_path, **kwargs)
    except (ValueError, TypeError):
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def attach_new_lora(
    model: Any, rank: int, alpha: int, dropout: float, target_modules: Sequence[str]
) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, config)


def attach_adapter(model: Any, adapter_path: str, is_trainable: bool = False) -> Any:
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)


def next_record(records: Sequence[dict[str, Any]], draw_index: int, rank: int, world_size: int, seed: int) -> dict[str, Any]:
    if not records:
        raise ValueError("empty record set")
    global_index = draw_index * world_size + rank
    epoch, offset = divmod(global_index, len(records))
    order = list(range(len(records)))
    random.Random(seed + epoch).shuffle(order)
    return records[order[offset]]


def verdict_scores(logits: Any, real_token_id: int, fake_token_id: int) -> Any:
    import torch

    last_logits = logits[:, -1, :].float()
    selected = last_logits[:, [real_token_id, fake_token_id]]
    log_probs = torch.log_softmax(selected, dim=-1)
    return log_probs[:, 1] - log_probs[:, 0]


def binary_verdict_loss(scores: Any, labels: Sequence[str]) -> Any:
    import torch
    import torch.nn.functional as functional

    targets = torch.tensor([1.0 if label == "Fake" else 0.0 for label in labels], device=scores.device)
    return functional.binary_cross_entropy_with_logits(scores, targets)
