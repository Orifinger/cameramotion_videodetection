"""Runtime helpers for one-video binary camera VQA training and scoring."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from scripts.camera_pretext_transfer.runtime import target_supervision_span
from scripts.caspr_gate1.runtime import (
    attach_adapter,
    attach_new_lora,
    cleanup_distributed,
    init_distributed,
    load_model,
    load_processor,
    next_record,
    read_jsonl,
    set_seed,
    write_json,
)


VIDEO_TOKEN_RE = re.compile(r"<video>")
YES_CANDIDATE = "Yes"
NO_CANDIDATE = "No"


def answer_token_ids(tokenizer: Any) -> tuple[int, int]:
    output: dict[str, int] = {}
    for answer in ("No", "Yes"):
        encoded = tokenizer.encode(answer, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"binary camera VQA requires a one-token {answer!r} answer; got token ids {encoded}"
            )
        output[answer] = int(encoded[0])
    if output["No"] == output["Yes"]:
        raise ValueError("Yes and No candidate token ids are identical")
    return output["No"], output["Yes"]


def structured_messages(
    sample: Mapping[str, Any], video_max_pixels: int, video_fps: float
) -> list[dict[str, Any]]:
    raw_messages = sample.get("messages")
    raw_videos = sample.get("videos", [])
    if not isinstance(raw_messages, list) or not isinstance(raw_videos, list):
        raise ValueError("sample must contain messages and videos lists")
    videos = iter(str(path) for path in raw_videos)
    used = 0
    output: list[dict[str, Any]] = []
    for message in raw_messages:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", ""))
        content = message.get("content", "")
        if not isinstance(content, str) or "<video>" not in content:
            output.append({"role": role, "content": content})
            continue
        parts: list[dict[str, Any]] = []
        cursor = 0
        for match in VIDEO_TOKEN_RE.finditer(content):
            if match.start() > cursor:
                parts.append({"type": "text", "text": content[cursor : match.start()]})
            try:
                video_path = next(videos)
            except StopIteration as exc:
                raise ValueError("more <video> tokens than video paths") from exc
            parts.append(
                {
                    "type": "video",
                    "video": video_path,
                    "fps": float(video_fps),
                    "max_pixels": int(video_max_pixels),
                }
            )
            used += 1
            cursor = match.end()
        if cursor < len(content):
            parts.append({"type": "text", "text": content[cursor:]})
        output.append({"role": role, "content": parts})
    if used != len(raw_videos):
        raise ValueError(f"video token/path mismatch: used={used}, paths={len(raw_videos)}")
    return output


def process_vision(conversation: Sequence[Mapping[str, Any]], patch_size: int) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    try:
        result = process_vision_info(
            conversation,
            image_patch_size=patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        metadata_enabled = True
    except TypeError:
        result = process_vision_info(
            conversation,
            image_patch_size=patch_size,
            return_video_kwargs=True,
        )
        metadata_enabled = False
    if not isinstance(result, tuple) or len(result) != 3:
        raise RuntimeError("qwen_vl_utils must return images, videos, and video kwargs")
    images, videos, video_kwargs = result
    video_metadata = None
    if metadata_enabled and videos:
        if all(isinstance(item, tuple) and len(item) == 2 for item in videos):
            videos, video_metadata = zip(*videos)
            videos, video_metadata = list(videos), list(video_metadata)
    return {
        "images": images,
        "videos": videos,
        "video_metadata": video_metadata,
        "video_kwargs": dict(video_kwargs or {}),
    }


def processor_batch(
    conversation: Sequence[Mapping[str, Any]], text: str, processor: Any, device: Any
) -> dict[str, Any]:
    image_processor = getattr(processor, "image_processor", None)
    patch_size = int(getattr(image_processor, "patch_size", 16))
    vision = process_vision(conversation, patch_size)
    kwargs: dict[str, Any] = {
        "text": [text],
        "padding": True,
        "return_tensors": "pt",
        "do_resize": False,
    }
    if vision["images"]:
        kwargs["images"] = vision["images"]
    if vision["videos"]:
        kwargs["videos"] = vision["videos"]
        kwargs.update(vision["video_kwargs"])
    if vision["video_metadata"] is not None:
        kwargs["video_metadata"] = vision["video_metadata"]
    batch = processor(**kwargs)
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def prepare_prompt_batch(
    sample: Mapping[str, Any],
    processor: Any,
    device: Any,
    video_max_pixels: int,
    video_fps: float,
) -> dict[str, Any]:
    conversation = structured_messages(sample, video_max_pixels, video_fps)
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    text += str(sample.get("assistant_prefix", ""))
    return processor_batch(conversation, text, processor, device)


def prepare_sft_batch(
    sample: Mapping[str, Any],
    processor: Any,
    device: Any,
    video_max_pixels: int,
    video_fps: float,
) -> dict[str, Any]:
    conversation = structured_messages(sample, video_max_pixels, video_fps)
    target = str(sample["target_text"])
    full_conversation = list(conversation) + [{"role": "assistant", "content": target}]
    text = processor.apply_chat_template(
        full_conversation, tokenize=False, add_generation_prompt=False
    )
    batch = processor_batch(full_conversation, text, processor, device)
    labels = batch["input_ids"].clone()
    labels.fill_(-100)
    target_ids = processor.tokenizer.encode(target, add_special_tokens=False)
    start, end = target_supervision_span(
        batch["input_ids"][0].tolist(), batch["attention_mask"][0].tolist(), target_ids
    )
    if start < 0:
        raise ValueError(f"target tokens were not found in rendered conversation: {target!r}")
    labels[0, start:end] = batch["input_ids"][0, start:end]
    batch["labels"] = labels
    return batch


def yes_minus_no_score(logits: Any, no_token_id: int, yes_token_id: int) -> Any:
    import torch

    selected = logits[:, -1, [no_token_id, yes_token_id]].float()
    log_probs = torch.log_softmax(selected, dim=-1)
    return log_probs[:, 1] - log_probs[:, 0]


__all__ = [
    "answer_token_ids",
    "attach_adapter",
    "attach_new_lora",
    "cleanup_distributed",
    "init_distributed",
    "load_model",
    "load_processor",
    "next_record",
    "prepare_prompt_batch",
    "prepare_sft_batch",
    "read_jsonl",
    "set_seed",
    "write_json",
    "yes_minus_no_score",
]
