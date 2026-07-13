#!/usr/bin/env python3
"""Skyra-style and diagnostic reward variants for DataB GRPO."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(Fake|Real)\s*</answer>", re.IGNORECASE | re.DOTALL)
FULL_RESPONSE_RE = re.compile(
    r"\s*<think>.+?</think>\s*<answer>\s*(?:Fake|Real)\s*</answer>\s*",
    re.IGNORECASE | re.DOTALL,
)
FAKE_BLOCK_RE = re.compile(
    r"<type>\s*(.*?)\s*</type>\s*in\s*<t>\s*(.*?)\s*</t>\s*at\s*<bbox>\s*(.*?)\s*</bbox>",
    re.IGNORECASE | re.DOTALL,
)
REAL_BLOCK_RE = re.compile(
    r"<t>\s*(.*?)\s*</t>\s*at\s*<bbox>\s*(.*?)\s*</bbox>",
    re.IGNORECASE | re.DOTALL,
)
NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")

ARTIFACT_TYPES = {
    "Hand Anatomy Error",
    "Limb Structure Error",
    "Body Proportion Error",
    "Face Identity Drift",
    "Facial Landmark Distortion",
    "Face Boundary Fusion",
    "Malformed Text",
    "Inconsistent Text Across Frames",
    "Logo / Symbol Distortion",
    "Object Deformation",
    "Object Identity Drift",
    "Object Part Inconsistency",
    "Boundary Fusion",
    "Contact Region Artifact",
    "Occlusion Error",
    "Texture Flicker",
    "Material Inconsistency",
    "Lighting / Shadow Inconsistency",
    "Entity Reappearance Change",
    "Cross-frame Identity Drift",
    "Object Category Shift",
    "Implausible Contact",
    "Motion Discontinuity",
    "Physical Interaction Error",
    "Known-person Factual Implausibility",
    "Non-realistic Event Premise",
    "Role / Context Contradiction",
    "Synthetic Rendering Cue",
    "Over-smoothed Generated Texture",
    "Stylized / CGI Rendering Inconsistency",
}
ARTIFACT_TYPE_LOOKUP = {value.casefold(): value for value in ARTIFACT_TYPES}

REWARD_VARIANTS = {
    "paper_asymmetric_inspection",
    "symmetric_zero_inspection",
    "asymmetric_outer_format",
    "asymmetric_answer_only",
    "strict_unique_inspection",
    "inspection_only_hackable",
    "official_repository_bug",
}


@dataclass(frozen=True)
class EvidenceStats:
    raw_count: int
    strict_unique_count: int
    duplicate_count: int
    invalid_count: int
    invalid_bbox_count: int
    invalid_time_count: int
    invalid_type_count: int


def _normalize_truth(value: Any) -> str | None:
    text = str(value or "").strip()
    tagged = ANSWER_RE.findall(text)
    if len(tagged) == 1:
        return tagged[0].title()
    if text.casefold() in {"fake", "1", "generated", "synthetic"}:
        return "Fake"
    if text.casefold() in {"real", "0", "authentic", "genuine"}:
        return "Real"
    return None


def _parse_answer(text: str) -> str | None:
    matches = ANSWER_RE.findall(text)
    return matches[0].title() if len(matches) == 1 else None


def _parse_number_list(value: str, count: int) -> tuple[float, ...] | None:
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        return None
    pieces = [piece.strip() for piece in value[1:-1].split(",")]
    if len(pieces) != count or any(NUMBER_RE.fullmatch(piece) is None for piece in pieces):
        return None
    return tuple(float(piece) for piece in pieces)


def _duration(extra_info: Any) -> float | None:
    if not isinstance(extra_info, dict):
        return None
    value = extra_info.get("duration_seconds")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _evidence_stats(text: str, prediction: str | None, extra_info: Any) -> EvidenceStats:
    think_matches = THINK_RE.findall(text)
    if len(think_matches) != 1 or prediction not in {"Fake", "Real"}:
        return EvidenceStats(0, 0, 0, 0, 0, 0, 0)
    think = think_matches[0]
    duration = _duration(extra_info)

    if prediction == "Fake":
        blocks = [(kind, time, bbox) for kind, time, bbox in FAKE_BLOCK_RE.findall(think)]
    else:
        clean_think = FAKE_BLOCK_RE.sub("", think)
        blocks = [(None, time, bbox) for time, bbox in REAL_BLOCK_RE.findall(clean_think)]

    seen: set[tuple[Any, ...]] = set()
    strict_unique = 0
    duplicates = 0
    invalid_bbox = 0
    invalid_time = 0
    invalid_type = 0
    for kind, time_text, bbox_text in blocks:
        canonical_type = None
        if prediction == "Fake":
            canonical_type = ARTIFACT_TYPE_LOOKUP.get(str(kind).strip().casefold())
            if canonical_type is None:
                invalid_type += 1

        time_values = _parse_number_list(time_text, 2)
        time_valid = time_values is not None
        if time_values is not None:
            start, end = time_values
            time_valid = start >= 0 and end >= start and (duration is None or end <= duration + 0.05)
        if not time_valid:
            invalid_time += 1

        bbox_values = _parse_number_list(bbox_text, 4)
        bbox_valid = bbox_values is not None
        if bbox_values is not None:
            x1, y1, x2, y2 = bbox_values
            bbox_valid = all(0 <= value <= 1000 for value in bbox_values) and x2 > x1 and y2 > y1
        if not bbox_valid:
            invalid_bbox += 1

        valid = time_valid and bbox_valid and (prediction == "Real" or canonical_type is not None)
        if not valid:
            continue
        key = (canonical_type, time_values, bbox_values)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        strict_unique += 1

    invalid = len(blocks) - strict_unique - duplicates
    return EvidenceStats(
        raw_count=len(blocks),
        strict_unique_count=strict_unique,
        duplicate_count=duplicates,
        invalid_count=max(0, invalid),
        invalid_bbox_count=invalid_bbox,
        invalid_time_count=invalid_time,
        invalid_type_count=invalid_type,
    )


def _check_reward(count: int) -> float:
    return min(math.log1p(max(0, count)), math.log1p(3))


def _asymmetric_accuracy(prediction: str | None, truth: str | None) -> float:
    if prediction is not None and truth is not None and prediction == truth:
        return 1.0
    if truth == "Real" and prediction == "Fake":
        return -0.2
    return 0.0


def _symmetric_zero_accuracy(prediction: str | None, truth: str | None) -> float:
    return float(prediction is not None and truth is not None and prediction == truth)


def _official_repository_bug_score(text: str, prediction: str | None, truth: str | None) -> float:
    if truth == "Real" and prediction == "Fake":
        return -0.2
    if truth == "Fake" and prediction == "Real":
        return 0.0
    if prediction is None or truth is None or prediction != truth:
        return 0.0
    think_matches = THINK_RE.findall(text)
    if len(think_matches) != 1:
        return 0.0
    think = think_matches[0]
    if truth == "Fake":
        evidence_count = len(REAL_BLOCK_RE.findall(think))
    else:
        # The published code aliases fake and real patterns, removes all matches,
        # and therefore gives correct Real responses zero evidence reward.
        evidence_count = len(REAL_BLOCK_RE.findall(REAL_BLOCK_RE.sub("", think)))
    return 0.8 * _check_reward(evidence_count)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,
    reward_variant: str = "paper_asymmetric_inspection",
    **_: Any,
) -> dict[str, float | str]:
    del data_source
    if reward_variant not in REWARD_VARIANTS:
        raise ValueError(f"unknown reward variant: {reward_variant}")

    text = str(solution_str or "")
    prediction = _parse_answer(text)
    truth = _normalize_truth(ground_truth)
    format_valid = bool(FULL_RESPONSE_RE.fullmatch(text))
    correct = prediction is not None and truth is not None and prediction == truth
    stats = _evidence_stats(text, prediction, extra_info)
    raw_inspection = _check_reward(stats.raw_count) if format_valid else 0.0
    strict_inspection = _check_reward(stats.strict_unique_count) if format_valid and correct else 0.0
    asymmetric = _asymmetric_accuracy(prediction, truth)
    symmetric_zero = _symmetric_zero_accuracy(prediction, truth)

    accuracy_component = asymmetric
    inspection_component = raw_inspection
    if reward_variant == "paper_asymmetric_inspection":
        score = 0.8 * asymmetric + 0.2 * raw_inspection
    elif reward_variant == "symmetric_zero_inspection":
        accuracy_component = symmetric_zero
        score = 0.8 * symmetric_zero + 0.2 * raw_inspection
    elif reward_variant == "asymmetric_outer_format":
        inspection_component = float(format_valid)
        score = 0.8 * asymmetric + 0.2 * inspection_component
    elif reward_variant == "asymmetric_answer_only":
        inspection_component = 0.0
        score = asymmetric
    elif reward_variant == "strict_unique_inspection":
        inspection_component = strict_inspection
        score = 0.8 * asymmetric + 0.2 * strict_inspection
    elif reward_variant == "inspection_only_hackable":
        accuracy_component = 0.0
        score = raw_inspection
    else:
        accuracy_component = asymmetric
        inspection_component = 0.0
        score = _official_repository_bug_score(text, prediction, truth)

    false_positive = truth == "Real" and prediction == "Fake"
    false_negative = truth == "Fake" and prediction == "Real"
    result: dict[str, float | str] = {
        "score": float(score),
        "accuracy_reward": float(accuracy_component),
        "inspection_reward": float(inspection_component),
        "correct": float(correct),
        "format_valid": float(format_valid),
        "pred_fake": float(prediction == "Fake"),
        "gt_fake": float(truth == "Fake"),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "answer_invalid": float(prediction is None),
        "raw_check_count": float(stats.raw_count),
        "strict_check_count": float(stats.strict_unique_count),
        "duplicate_check_count": float(stats.duplicate_count),
        "invalid_check_count": float(stats.invalid_count),
        "invalid_bbox_count": float(stats.invalid_bbox_count),
        "invalid_time_count": float(stats.invalid_time_count),
        "invalid_type_count": float(stats.invalid_type_count),
        "wrong_positive_reward": float(not correct and score > 0),
        "response_chars": float(len(text)),
    }
    if isinstance(extra_info, dict) and extra_info.get("sample_id") is not None:
        result["diagnostic_sample_id"] = str(extra_info["sample_id"])
    return result


if __name__ == "__main__":
    print("\n".join(sorted(REWARD_VARIANTS)))
