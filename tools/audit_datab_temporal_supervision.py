#!/usr/bin/env python3
"""Audit how much temporal supervision exists in a DataB SFT JSON file."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TEMPORAL_ARTIFACT_TYPES = {
    "Face Identity Drift",
    "Inconsistent Text Across Frames",
    "Object Identity Drift",
    "Texture Flicker",
    "Entity Reappearance Change",
    "Cross-frame Identity Drift",
    "Object Category Shift",
    "Motion Discontinuity",
}

STRONG_TEMPORAL_PATTERNS = {
    "across_frames": re.compile(r"\bacross (?:all |the )?frames?\b", re.I),
    "between_frames": re.compile(r"\bbetween (?:the )?frames?\b", re.I),
    "frame_to_frame": re.compile(r"\bframe[- ]to[- ]frame\b", re.I),
    "over_time": re.compile(r"\bover time\b", re.I),
    "sequence_progression": re.compile(
        r"\b(?:as|while) (?:the )?(?:video|sequence|frames?) "
        r"(?:progress(?:es)?|unfolds?|continues?)\b",
        re.I,
    ),
    "ordered_frames": re.compile(
        r"\b(?:earlier|later|subsequent|successive|consecutive|adjacent) frames?\b",
        re.I,
    ),
    "throughout_sequence": re.compile(
        r"\bthroughout (?:all |the )?(?:video|sequence|frames?)\b", re.I
    ),
    "temporal_term": re.compile(r"\btempor(?:al|ally)\b", re.I),
    "reappearance": re.compile(r"\breappear(?:s|ed|ing|ance)?\b", re.I),
    "flicker": re.compile(r"\bflicker(?:s|ed|ing)?\b", re.I),
    "drift": re.compile(r"\bdrift(?:s|ed|ing)?\b", re.I),
    "discontinuity": re.compile(r"\bdiscontinu(?:ity|ous|ously)\b", re.I),
}

WEAK_MOTION_PATTERN = re.compile(
    r"\b(?:motion|movement|moving|moves|moved|camera|pan(?:s|ned|ning)?|"
    r"zoom(?:s|ed|ing)?|tilt(?:s|ed|ing)?|track(?:s|ed|ing)?|shake|shaking)\b",
    re.I,
)

TYPE_PATTERN = re.compile(r"<type>\s*(.*?)\s*</type>", re.I | re.S)
TIME_PATTERN = re.compile(
    r"<t>\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*</t>",
    re.I,
)
BBOX_PATTERN = re.compile(r"<bbox>\s*\[(.*?)\]\s*</bbox>", re.I | re.S)
ANSWER_PATTERN = re.compile(r"<answer>\s*(Real|Fake)\s*</answer>", re.I)
USER_TIMESTAMP_PATTERN = re.compile(r"\[T\s*=\s*(-?\d+(?:\.\d+)?)s?\]", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--examples-per-group", type=int, default=3)
    return parser.parse_args()


def read_message(item: dict[str, Any], role: str) -> str:
    for message in item.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content", ""))
    return ""


def rate(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return numerator / denominator


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_example(
    examples: dict[str, list[dict[str, Any]]],
    group: str,
    index: int,
    answer: str,
    assistant: str,
    limit: int,
) -> None:
    bucket = examples.setdefault(group, [])
    if len(bucket) >= limit:
        return
    compact = re.sub(r"\s+", " ", assistant).strip()
    bucket.append(
        {
            "record_index": index,
            "answer": answer,
            "assistant_excerpt": compact[:500],
        }
    )


def audit(data: list[dict[str, Any]], source_path: Path, examples_per_group: int) -> dict[str, Any]:
    answers: Counter[str] = Counter()
    frame_counts: Counter[int] = Counter()
    artifact_occurrences: Counter[str] = Counter()
    artifact_record_counts: Counter[str] = Counter()
    temporal_phrase_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}

    valid_think_answer = 0
    records_with_time_tags = 0
    records_with_bbox_tags = 0
    records_with_artifact_types = 0
    records_with_temporal_artifact_type = 0
    fake_records_with_temporal_artifact_type = 0
    fake_records_with_only_non_temporal_types = 0
    records_with_strong_temporal_language = 0
    fake_records_with_strong_temporal_language = 0
    real_records_with_strong_temporal_language = 0
    records_with_weak_motion_language = 0
    user_prompts_with_timestamps = 0
    total_time_tags = 0
    broad_time_tags = 0
    full_clip_time_tags = 0
    valid_time_tags_for_ratio = 0

    for index, item in enumerate(data):
        assistant = read_message(item, "assistant")
        user = read_message(item, "user")
        answer_match = ANSWER_PATTERN.search(assistant)
        answer = answer_match.group(1).title() if answer_match else "Unknown"
        answers[answer] += 1
        frame_counts[len(item.get("images", []))] += 1

        if "<think>" in assistant and "</think>" in assistant and answer_match:
            valid_think_answer += 1

        user_timestamps = [float(value) for value in USER_TIMESTAMP_PATTERN.findall(user)]
        if user_timestamps:
            user_prompts_with_timestamps += 1
        clip_start = min(user_timestamps) if user_timestamps else None
        clip_end = max(user_timestamps) if user_timestamps else None
        clip_duration = (
            clip_end - clip_start
            if clip_start is not None and clip_end is not None and clip_end > clip_start
            else None
        )

        types = [re.sub(r"\s+", " ", value).strip() for value in TYPE_PATTERN.findall(assistant)]
        unique_types = set(types)
        artifact_occurrences.update(types)
        artifact_record_counts.update(unique_types)
        if types:
            records_with_artifact_types += 1

        temporal_types = sorted(unique_types & TEMPORAL_ARTIFACT_TYPES)
        if temporal_types:
            records_with_temporal_artifact_type += 1
            if answer == "Fake":
                fake_records_with_temporal_artifact_type += 1
            add_example(
                examples,
                "temporal_artifact_type",
                index,
                answer,
                assistant,
                examples_per_group,
            )
        elif answer == "Fake" and types:
            fake_records_with_only_non_temporal_types += 1

        matched_temporal_phrases = []
        for name, pattern in STRONG_TEMPORAL_PATTERNS.items():
            if pattern.search(assistant):
                temporal_phrase_counts[name] += 1
                matched_temporal_phrases.append(name)
        if matched_temporal_phrases:
            records_with_strong_temporal_language += 1
            if answer == "Fake":
                fake_records_with_strong_temporal_language += 1
            elif answer == "Real":
                real_records_with_strong_temporal_language += 1
            add_example(
                examples,
                f"strong_temporal_language_{answer.lower()}",
                index,
                answer,
                assistant,
                examples_per_group,
            )

        if WEAK_MOTION_PATTERN.search(assistant):
            records_with_weak_motion_language += 1

        time_tags = [(float(start), float(end)) for start, end in TIME_PATTERN.findall(assistant)]
        bbox_tags = BBOX_PATTERN.findall(assistant)
        total_time_tags += len(time_tags)
        records_with_time_tags += int(bool(time_tags))
        records_with_bbox_tags += int(bool(bbox_tags))
        if clip_duration:
            for start, end in time_tags:
                span_ratio = max(0.0, end - start) / clip_duration
                valid_time_tags_for_ratio += 1
                broad_time_tags += int(span_ratio >= 0.8)
                full_clip_time_tags += int(
                    abs(start - clip_start) <= 0.05 and abs(end - clip_end) <= 0.05
                )

    total = len(data)
    fake_total = answers["Fake"]
    real_total = answers["Real"]
    temporal_occurrences = sum(
        count for name, count in artifact_occurrences.items() if name in TEMPORAL_ARTIFACT_TYPES
    )
    artifact_total = sum(artifact_occurrences.values())

    return {
        "schema_version": "datab_temporal_supervision_audit_v1",
        "source": {
            "path": str(source_path.resolve()),
            "sha256": sha256_file(source_path),
        },
        "definition": {
            "important_caveat": (
                "A <t> interval is required by the prompt and is not counted as temporal reasoning. "
                "Temporal supervision is reported separately using conservative artifact types and "
                "explicit cross-frame language."
            ),
            "conservative_temporal_artifact_types": sorted(TEMPORAL_ARTIFACT_TYPES),
            "strong_temporal_language_patterns": sorted(STRONG_TEMPORAL_PATTERNS),
        },
        "overall": {
            "records": total,
            "answers": dict(answers),
            "frame_count_distribution": {str(k): v for k, v in sorted(frame_counts.items())},
            "valid_think_answer_records": valid_think_answer,
            "valid_think_answer_rate": rate(valid_think_answer, total),
            "user_prompts_with_timestamps": user_prompts_with_timestamps,
            "user_prompt_timestamp_rate": rate(user_prompts_with_timestamps, total),
        },
        "structured_evidence": {
            "records_with_artifact_types": records_with_artifact_types,
            "artifact_type_record_rate": rate(records_with_artifact_types, total),
            "artifact_type_occurrences": artifact_total,
            "artifact_occurrence_counts": dict(artifact_occurrences.most_common()),
            "artifact_record_counts": dict(artifact_record_counts.most_common()),
            "records_with_time_tags": records_with_time_tags,
            "time_tag_record_rate": rate(records_with_time_tags, total),
            "total_time_tags": total_time_tags,
            "records_with_bbox_tags": records_with_bbox_tags,
            "bbox_tag_record_rate": rate(records_with_bbox_tags, total),
            "valid_time_tags_for_span_ratio": valid_time_tags_for_ratio,
            "broad_time_tags_ge_80pct": broad_time_tags,
            "broad_time_tag_rate": rate(broad_time_tags, valid_time_tags_for_ratio),
            "full_clip_time_tags": full_clip_time_tags,
            "full_clip_time_tag_rate": rate(full_clip_time_tags, valid_time_tags_for_ratio),
        },
        "temporal_supervision": {
            "temporal_artifact_occurrences": temporal_occurrences,
            "temporal_artifact_occurrence_rate_among_artifacts": rate(
                temporal_occurrences, artifact_total
            ),
            "records_with_temporal_artifact_type": records_with_temporal_artifact_type,
            "record_rate_with_temporal_artifact_type": rate(
                records_with_temporal_artifact_type, total
            ),
            "fake_records_with_temporal_artifact_type": fake_records_with_temporal_artifact_type,
            "fake_record_rate_with_temporal_artifact_type": rate(
                fake_records_with_temporal_artifact_type, fake_total
            ),
            "fake_records_with_only_non_temporal_types": fake_records_with_only_non_temporal_types,
            "fake_record_rate_with_only_non_temporal_types": rate(
                fake_records_with_only_non_temporal_types, fake_total
            ),
            "records_with_strong_temporal_language": records_with_strong_temporal_language,
            "strong_temporal_language_rate": rate(records_with_strong_temporal_language, total),
            "fake_records_with_strong_temporal_language": fake_records_with_strong_temporal_language,
            "fake_strong_temporal_language_rate": rate(
                fake_records_with_strong_temporal_language, fake_total
            ),
            "real_records_with_strong_temporal_language": real_records_with_strong_temporal_language,
            "real_strong_temporal_language_rate": rate(
                real_records_with_strong_temporal_language, real_total
            ),
            "strong_temporal_phrase_record_counts": dict(temporal_phrase_counts.most_common()),
            "records_with_weak_motion_language": records_with_weak_motion_language,
            "weak_motion_language_rate": rate(records_with_weak_motion_language, total),
        },
        "examples": examples,
    }


def markdown_report(report: dict[str, Any]) -> str:
    overall = report["overall"]
    structured = report["structured_evidence"]
    temporal = report["temporal_supervision"]
    top_types = list(structured["artifact_occurrence_counts"].items())[:12]

    lines = [
        "# DataB 时序监督内容审计",
        "",
        f"- 输入：`{report['source']['path']}`",
        f"- SHA256：`{report['source']['sha256']}`",
        f"- 样本数：{overall['records']}（Real {overall['answers'].get('Real', 0)} / "
        f"Fake {overall['answers'].get('Fake', 0)}）",
        f"- 帧数分布：{overall['frame_count_distribution']}",
        "",
        "## 口径",
        "",
        "`<t>[start, end]</t>` 是 system prompt 强制要求的输出格式，不能单独证明模型进行了跨帧推理。",
        "本审计分别统计保守的时序伪影类别、显式跨帧语言，以及时间标签覆盖范围。",
        "",
        "## 核心结果",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 具有保守时序伪影类别的 Fake 样本 | "
        f"{temporal['fake_records_with_temporal_artifact_type']} / "
        f"{overall['answers'].get('Fake', 0)} "
        f"({pct(temporal['fake_record_rate_with_temporal_artifact_type'])}) |",
        f"| 时序类别在全部伪影标签中的占比 | "
        f"{temporal['temporal_artifact_occurrences']} / "
        f"{structured['artifact_type_occurrences']} "
        f"({pct(temporal['temporal_artifact_occurrence_rate_among_artifacts'])}) |",
        f"| 仅含非时序类别的 Fake 样本 | "
        f"{temporal['fake_records_with_only_non_temporal_types']} "
        f"({pct(temporal['fake_record_rate_with_only_non_temporal_types'])}) |",
        f"| 含显式跨帧语言的全部样本 | "
        f"{temporal['records_with_strong_temporal_language']} "
        f"({pct(temporal['strong_temporal_language_rate'])}) |",
        f"| 含显式跨帧语言的 Fake 样本 | "
        f"{temporal['fake_records_with_strong_temporal_language']} "
        f"({pct(temporal['fake_strong_temporal_language_rate'])}) |",
        f"| 含显式跨帧语言的 Real 样本 | "
        f"{temporal['real_records_with_strong_temporal_language']} "
        f"({pct(temporal['real_strong_temporal_language_rate'])}) |",
        f"| 覆盖至少 80% 视频长度的时间标签 | "
        f"{structured['broad_time_tags_ge_80pct']} / "
        f"{structured['valid_time_tags_for_span_ratio']} "
        f"({pct(structured['broad_time_tag_rate'])}) |",
        f"| 精确覆盖整段视频的时间标签 | "
        f"{structured['full_clip_time_tags']} / "
        f"{structured['valid_time_tags_for_span_ratio']} "
        f"({pct(structured['full_clip_time_tag_rate'])}) |",
        "",
        "## 伪影类别分布",
        "",
        "| 类别 | 出现次数 |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in top_types)
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "该审计衡量训练文本中是否存在显式时序监督，不衡量 Qwen3-VL 是否真正利用帧顺序，",
            "也不衡量这些时序描述是否能提高 Real/Fake 检测。后两者需要输入干预和逐样本残余错误分析。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError("Expected the input JSON root to be a list")

    report = audit(data, args.input, args.examples_per_group)
    rendered_json = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    rendered_markdown = markdown_report(report)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered_json, encoding="utf-8")
    else:
        print(rendered_json)

    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(rendered_markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
