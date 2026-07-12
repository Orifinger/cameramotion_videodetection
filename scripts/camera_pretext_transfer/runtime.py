from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Mapping, Sequence

from scripts.caspr_gate1.runtime import _vision_inputs, structured_messages

CAMERA_TAG_RE = re.compile(r"\s*<camera_motion>(.*?)</camera_motion>\s*", re.DOTALL)


def find_last_subsequence(values: Sequence[int], needle: Sequence[int]) -> int:
    if not needle or len(needle) > len(values):
        return -1
    for start in range(len(values) - len(needle), -1, -1):
        if list(values[start : start + len(needle)]) == list(needle):
            return start
    return -1


def target_supervision_span(
    input_ids: Sequence[int], attention_mask: Sequence[int], target_ids: Sequence[int]
) -> tuple[int, int]:
    start = find_last_subsequence(input_ids, target_ids)
    if start < 0:
        return -1, -1
    valid_positions = [index for index, value in enumerate(attention_mask) if int(value) != 0]
    if not valid_positions:
        return -1, -1
    end = valid_positions[-1] + 1
    if end < start + len(target_ids):
        return -1, -1
    return start, end


def prepare_sft_batch(
    samples: Sequence[Mapping[str, Any]], processor: Any, device: Any, max_pixels: int
) -> dict[str, Any]:
    conversations = [structured_messages(sample, max_pixels) for sample in samples]
    target_texts = [str(sample["target_text"]) for sample in samples]
    full_conversations = [
        conversation + [{"role": "assistant", "content": target}]
        for conversation, target in zip(conversations, target_texts)
    ]
    texts = [
        processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        for conversation in full_conversations
    ]
    images: list[Any] = []
    videos: list[Any] = []
    image_processor = getattr(processor, "image_processor", None)
    patch_size = int(getattr(image_processor, "patch_size", 16))
    for conversation in full_conversations:
        sample_images, sample_videos = _vision_inputs(conversation, patch_size)
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
    labels = batch["input_ids"].clone()
    labels.fill_(-100)
    for row_index, target in enumerate(target_texts):
        target_ids = processor.tokenizer.encode(target, add_special_tokens=False)
        input_ids = batch["input_ids"][row_index].tolist()
        attention_mask = batch["attention_mask"][row_index].tolist()
        start, end = target_supervision_span(input_ids, attention_mask, target_ids)
        if start < 0:
            raise ValueError(f"camera target tokens were not found in rendered conversation: {target}")
        # Include the chat template's assistant terminator so generation learns to stop cleanly.
        labels[row_index, start:end] = batch["input_ids"][row_index, start:end]
    batch["labels"] = labels
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def parse_camera_response(response: str, allowed_labels: Sequence[str]) -> dict[str, Any]:
    match = CAMERA_TAG_RE.fullmatch(response)
    if not match:
        return {"format_valid": False, "labels": [], "reason": "tag_or_extra_text"}
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {"format_valid": False, "labels": [], "reason": "invalid_json"}
    if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
        return {"format_valid": False, "labels": [], "reason": "not_string_list"}
    if len(payload) != len(set(payload)):
        return {"format_valid": False, "labels": [], "reason": "duplicate_labels"}
    unknown = [label for label in payload if label not in allowed_labels]
    if unknown:
        return {"format_valid": False, "labels": [], "reason": "unknown_labels", "unknown": unknown}
    return {"format_valid": True, "labels": payload, "reason": ""}


def multilabel_metrics(
    gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    allowed_labels: Sequence[str],
) -> dict[str, Any]:
    predictions = {str(row.get("case_id")): row for row in prediction_rows}
    label_stats = {label: Counter(tp=0, fp=0, fn=0) for label in allowed_labels}
    format_valid = exact = bucket_correct = 0
    matched = 0
    parse_reasons: Counter[str] = Counter()
    motion_labels = ("complex-motion", "minor-motion", "no-motion")

    def bucket(labels: Sequence[str]) -> str:
        for label in motion_labels:
            if label in labels:
                return label
        return "unknown"

    for gold in gold_rows:
        case_id = str(gold.get("case_id"))
        prediction = predictions.get(case_id)
        if prediction is None:
            continue
        matched += 1
        parsed = parse_camera_response(str(prediction.get("response", "")), allowed_labels)
        parse_reasons[parsed["reason"] or "valid"] += 1
        predicted = set(parsed["labels"] if parsed["format_valid"] else [])
        expected = set(gold.get("camera_labels", []))
        format_valid += int(parsed["format_valid"])
        exact += int(predicted == expected and parsed["format_valid"])
        bucket_correct += int(bucket(predicted) == bucket(expected) and parsed["format_valid"])
        for label in allowed_labels:
            present_gold = label in expected
            present_pred = label in predicted
            label_stats[label]["tp"] += int(present_gold and present_pred)
            label_stats[label]["fp"] += int(not present_gold and present_pred)
            label_stats[label]["fn"] += int(present_gold and not present_pred)
    per_label: dict[str, Any] = {}
    total_tp = total_fp = total_fn = 0
    supported_f1: list[float] = []
    for label, stats in label_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = tp + fn
        per_label[label] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }
        if support:
            supported_f1.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall else 0.0
    )
    total = len(gold_rows)
    return {
        "num_gold": total,
        "num_predictions": len(prediction_rows),
        "num_matched": matched,
        "coverage": matched / total if total else 0.0,
        "format_valid_rate": format_valid / total if total else 0.0,
        "exact_set_accuracy": exact / total if total else 0.0,
        "coarse_motion_bucket_accuracy": bucket_correct / total if total else 0.0,
        "micro_f1": micro_f1,
        "macro_f1_supported_labels": sum(supported_f1) / len(supported_f1) if supported_f1 else 0.0,
        "num_supported_labels": len(supported_f1),
        "parse_reasons": dict(parse_reasons),
        "per_label": per_label,
    }
