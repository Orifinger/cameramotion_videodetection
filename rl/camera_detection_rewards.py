#!/usr/bin/env python3
"""Deterministic ms-swift rewards for camera pretext and video detection GRPO."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

try:
    from swift.plugin import ORM, orms
except ImportError:  # Keep pure reward logic locally testable without ms-swift.
    class ORM:  # type: ignore[no-redef]
        pass

    orms: dict[str, Any] = {}


CAMERA_LABEL_ORDER = [
    "very-unsteady",
    "unsteady",
    "minimal-shaking",
    "no-shaking",
    "complex-motion",
    "minor-motion",
    "no-motion",
    "fast-speed",
    "regular-speed",
    "slow-speed",
    "dolly-in",
    "dolly-out",
    "truck-left",
    "truck-right",
    "pedestal-up",
    "pedestal-down",
    "pan-left",
    "pan-right",
    "tilt-up",
    "tilt-down",
    "roll-CW",
    "roll-CCW",
    "zoom-in",
    "zoom-out",
    "arc-CW",
    "arc-CCW",
    "side-tracking",
    "lead-tracking",
    "tail-tracking",
    "aerial-tracking",
    "arc-tracking",
    "pan-tracking",
    "tilt-tracking",
]
CAMERA_LOOKUP = {label.casefold().replace("_", "-"): label for label in CAMERA_LABEL_ORDER}
EXCLUDED_LABELS = {"static"}
CAMERA_TAG_RE = re.compile(r"<camera_motion>\s*(.*?)\s*</camera_motion>", re.DOTALL | re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
THINK_TAG_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ParsedCameraLabels:
    labels: tuple[str, ...]
    unknown: tuple[str, ...]
    valid: bool
    duplicate: bool


def completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("content", "response", "completion", "text"):
            if key in value:
                return str(value[key])
    return str(value or "")


def normalize_camera_label(value: Any) -> tuple[str | None, str | None]:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None, None
    folded = cleaned.casefold().replace("_", "-")
    if folded in EXCLUDED_LABELS:
        return None, None
    canonical = CAMERA_LOOKUP.get(folded)
    if canonical is not None:
        return canonical, None
    return None, folded


def normalize_truth_labels(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                value = [part.strip() for part in text.split(",")]
        else:
            value = [part.strip() for part in text.split(",")]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        value = [value]

    present: set[str] = set()
    for raw in value:
        canonical, _unknown = normalize_camera_label(raw)
        if canonical:
            present.add(canonical)
    return [label for label in CAMERA_LABEL_ORDER if label in present]


def parse_camera_completion(value: Any) -> ParsedCameraLabels:
    text = completion_text(value)
    matches = CAMERA_TAG_RE.findall(text)
    if len(matches) != 1:
        return ParsedCameraLabels((), (), False, False)
    try:
        payload = json.loads(matches[0])
    except (json.JSONDecodeError, TypeError):
        return ParsedCameraLabels((), (), False, False)
    if not isinstance(payload, list) or not payload:
        return ParsedCameraLabels((), (), False, False)
    if any(not isinstance(item, str) or not item.strip() for item in payload):
        return ParsedCameraLabels((), (), False, False)

    labels: list[str] = []
    unknown: list[str] = []
    seen_raw: set[str] = set()
    duplicate = False
    for item in payload:
        folded = item.strip().casefold().replace("_", "-")
        if folded in seen_raw:
            duplicate = True
        seen_raw.add(folded)
        canonical, unknown_value = normalize_camera_label(item)
        if canonical:
            labels.append(canonical)
        elif unknown_value:
            unknown.append(unknown_value)
    ordered = tuple(label for label in CAMERA_LABEL_ORDER if label in set(labels))
    return ParsedCameraLabels(ordered, tuple(sorted(set(unknown))), True, duplicate)


def camera_set_f1(prediction: Any, truth: Any) -> float:
    parsed = parse_camera_completion(prediction)
    if not parsed.valid:
        return 0.0
    predicted = set(parsed.labels)
    predicted.update(f"__unknown__:{item}" for item in parsed.unknown)
    target = set(normalize_truth_labels(truth))
    if not predicted and not target:
        return 1.0
    tp = len(predicted & target)
    fp = len(predicted - target)
    fn = len(target - predicted)
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def camera_exact_match(prediction: Any, truth: Any) -> float:
    parsed = parse_camera_completion(prediction)
    if not parsed.valid or parsed.unknown or parsed.duplicate:
        return 0.0
    return float(set(parsed.labels) == set(normalize_truth_labels(truth)))


def camera_format_valid(prediction: Any) -> float:
    parsed = parse_camera_completion(prediction)
    return float(parsed.valid and not parsed.unknown and not parsed.duplicate)


def parse_detection_answer(value: Any) -> str | None:
    matches = ANSWER_TAG_RE.findall(completion_text(value))
    if len(matches) != 1:
        return None
    answer = matches[0].strip().casefold()
    if answer == "fake":
        return "Fake"
    if answer == "real":
        return "Real"
    return None


def normalize_detection_truth(value: Any) -> str | None:
    text = str(value if value is not None else "").strip()
    tagged = parse_detection_answer(text)
    if tagged:
        return tagged
    folded = text.casefold()
    if folded in {"1", "fake", "synthetic", "generated"}:
        return "Fake"
    if folded in {"0", "real", "authentic", "genuine"}:
        return "Real"
    return None


def detection_binary_correct(prediction: Any, truth: Any) -> float:
    predicted = parse_detection_answer(prediction)
    target = normalize_detection_truth(truth)
    return float(predicted is not None and target is not None and predicted == target)


def detection_format_valid(prediction: Any) -> float:
    text = completion_text(prediction)
    think = THINK_TAG_RE.findall(text)
    answer = ANSWER_TAG_RE.findall(text)
    if len(think) != 1 or len(answer) != 1 or not think[0].strip():
        return 0.0
    return float(parse_detection_answer(text) is not None)


def _camera_truth_per_completion(value: Any, count: int) -> list[Any]:
    if count <= 0:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows = list(value)
        if len(rows) == count and any(
            isinstance(item, (list, tuple, set, dict)) for item in rows
        ):
            return rows
        if len(rows) == count and all(isinstance(item, str) and item.strip().startswith("[") for item in rows):
            return rows
        return [rows for _ in range(count)]
    return [value for _ in range(count)]


def _scalar_truth_per_completion(value: Any, count: int) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows = list(value)
        if len(rows) == count:
            return rows
    return [value for _ in range(count)]


class CameraSetF1Reward(ORM):
    def __call__(self, completions, **kwargs) -> list[float]:
        truths = _camera_truth_per_completion(kwargs.get("camera_labels"), len(completions))
        return [camera_set_f1(pred, truth) for pred, truth in zip(completions, truths)]


class CameraExactReward(ORM):
    def __call__(self, completions, **kwargs) -> list[float]:
        truths = _camera_truth_per_completion(kwargs.get("camera_labels"), len(completions))
        return [camera_exact_match(pred, truth) for pred, truth in zip(completions, truths)]


class CameraFormatReward(ORM):
    def __call__(self, completions, **kwargs) -> list[float]:
        return [camera_format_valid(pred) for pred in completions]


class DetectionBinaryReward(ORM):
    def __call__(self, completions, **kwargs) -> list[float]:
        truths = _scalar_truth_per_completion(kwargs.get("label"), len(completions))
        return [detection_binary_correct(pred, truth) for pred, truth in zip(completions, truths)]


class DetectionFormatReward(ORM):
    def __call__(self, completions, **kwargs) -> list[float]:
        return [detection_format_valid(pred) for pred in completions]


orms["camera_set_f1"] = CameraSetF1Reward
orms["camera_exact"] = CameraExactReward
orms["camera_format"] = CameraFormatReward
orms["detection_binary_acc"] = DetectionBinaryReward
orms["detection_format"] = DetectionFormatReward
