from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence


SCORE_RE = re.compile(r"<score>\s*([1-5])\s*</score>", re.IGNORECASE)
REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.IGNORECASE | re.DOTALL)
CONTROL_VARIANTS = ("shuffled_frames", "shifted_bbox", "shifted_time", "swapped_type")


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


def parse_prediction(row: Mapping[str, Any]) -> dict[str, Any]:
    text = str(row.get("prediction", ""))
    score_match = SCORE_RE.search(text)
    reasoning_match = REASONING_RE.search(text)
    error = row.get("error")
    return {
        "score": int(score_match.group(1)) if score_match else None,
        "reasoning": reasoning_match.group(1).strip() if reasoning_match else "",
        "format_valid": score_match is not None and reasoning_match is not None and not error,
        "error": error,
    }


def score_stats(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    distribution = Counter(values)
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "score_distribution": {str(score): distribution.get(score, 0) for score in range(1, 6)},
        "score_ge_4_rate": sum(value >= 4 for value in values) / len(values),
        "score_le_2_rate": sum(value <= 2 for value in values) / len(values),
    }


def pair_stats(pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
    if not pairs:
        return {"count": 0}
    deltas = [original - control for original, control in pairs]
    return {
        "count": len(pairs),
        "original_mean": mean(original for original, _ in pairs),
        "control_mean": mean(control for _, control in pairs),
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        "original_gt_control_rate": sum(delta > 0 for delta in deltas) / len(deltas),
        "original_eq_control_rate": sum(delta == 0 for delta in deltas) / len(deltas),
        "original_lt_control_rate": sum(delta < 0 for delta in deltas) / len(deltas),
    }


def finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate(rows: Sequence[dict[str, Any]], expected: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parsed_rows: list[dict[str, Any]] = []
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        parsed = parse_prediction(row)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        result = {
            "judge_id": str(row.get("judge_id", "")),
            "sample_id": str(row.get("sample_id", "")),
            "variant": str(row.get("variant", "")),
            "score": parsed["score"],
            "format_valid": parsed["format_valid"],
            "reasoning": parsed["reasoning"],
            "error": parsed["error"],
            "gt_label": str(metadata.get("gt_label", "unknown")),
            "source_bucket": str(metadata.get("source_bucket", "unknown")),
            "primary_artifact_type": str(metadata.get("primary_artifact_type", "none")),
            "source_row_index": metadata.get("source_row_index"),
            "static_hard_fail_reasons": ";".join(
                metadata.get("static_audit", {}).get("hard_fail_reasons", [])
                if isinstance(metadata.get("static_audit"), Mapping)
                else []
            ),
        }
        parsed_rows.append(result)
        by_sample[result["sample_id"]][result["variant"]] = result

    actual = len(rows)
    denominator = expected if expected is not None else actual
    valid_count = sum(bool(row["format_valid"]) for row in parsed_rows)
    coverage = actual / denominator if denominator else 0.0
    format_valid_rate = valid_count / actual if actual else 0.0
    variant_scores: dict[str, list[int]] = defaultdict(list)
    for row in parsed_rows:
        if row["score"] is not None:
            variant_scores[row["variant"]].append(int(row["score"]))

    comparisons: dict[str, dict[str, Any]] = {}
    for variant in CONTROL_VARIANTS:
        pairs: list[tuple[int, int]] = []
        for variants in by_sample.values():
            original = variants.get("original")
            control = variants.get(variant)
            if original and control and original["score"] is not None and control["score"] is not None:
                pairs.append((int(original["score"]), int(control["score"])))
        comparisons[variant] = pair_stats(pairs)

    original_rows = [row for row in parsed_rows if row["variant"] == "original" and row["score"] is not None]
    by_label = {
        label: score_stats([int(row["score"]) for row in original_rows if row["gt_label"] == label])
        for label in sorted({row["gt_label"] for row in original_rows})
    }
    hard_fail_count = sum(bool(row["static_hard_fail_reasons"]) for row in original_rows)

    thresholds = {
        "min_prediction_coverage": 0.98,
        "min_format_valid_rate": 0.95,
        "min_original_gt_shuffled_frames_rate": 0.70,
        "min_original_gt_local_control_rate": 0.65,
        "min_local_control_pairs": 30,
    }
    checks: dict[str, bool | None] = {
        "prediction_coverage": coverage >= thresholds["min_prediction_coverage"],
        "format_valid_rate": format_valid_rate >= thresholds["min_format_valid_rate"],
    }
    shuffled_rate = comparisons["shuffled_frames"].get("original_gt_control_rate")
    checks["visual_frame_dependence"] = bool(
        comparisons["shuffled_frames"].get("count", 0) >= 30
        and shuffled_rate is not None
        and shuffled_rate >= thresholds["min_original_gt_shuffled_frames_rate"]
    )
    supported_local = 0
    passed_local = 0
    for variant in ("shifted_bbox", "shifted_time", "swapped_type"):
        stats = comparisons[variant]
        count = int(stats.get("count", 0))
        rate = stats.get("original_gt_control_rate")
        key = f"sensitive_to_{variant}"
        if count < thresholds["min_local_control_pairs"]:
            checks[key] = None
            continue
        supported_local += 1
        passed = rate is not None and rate >= thresholds["min_original_gt_local_control_rate"]
        checks[key] = passed
        passed_local += int(passed)
    checks["at_least_one_local_claim_control"] = supported_local > 0 and passed_local > 0
    required = (
        checks["prediction_coverage"],
        checks["format_valid_rate"],
        checks["visual_frame_dependence"],
        checks["at_least_one_local_claim_control"],
    )
    status = "passed" if all(value is True for value in required) else "failed"
    summary = {
        "gate": "DataB DeepfakeJudge-7B reliability gate",
        "status": status,
        "what_was_tested": (
            "Whether the pointwise judge scores the original DataB annotation above same-label shuffled frames "
            "and deterministic bbox, time, or artifact-type corruptions before it is used for dataset filtering."
        ),
        "expected_prediction_records": denominator,
        "prediction_records": actual,
        "prediction_coverage": coverage,
        "format_valid_count": valid_count,
        "format_valid_rate": format_valid_rate,
        "thresholds": thresholds,
        "checks": checks,
        "by_variant": {variant: score_stats(values) for variant, values in sorted(variant_scores.items())},
        "paired_controls": comparisons,
        "original_by_gt_label": by_label,
        "original_static_hard_fail_count": hard_fail_count,
        "original_static_hard_fail_rate": hard_fail_count / len(original_rows) if original_rows else 0.0,
        "next_action": (
            "Sample 100 original records for blinded human calibration; do not filter the full DataB yet."
            if status == "passed"
            else "Do not use this judge for full DataB filtering; inspect failed controls and model outputs."
        ),
    }
    return summary, parsed_rows


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "judge_id",
        "sample_id",
        "variant",
        "score",
        "format_valid",
        "gt_label",
        "source_bucket",
        "primary_artifact_type",
        "source_row_index",
        "static_hard_fail_reasons",
        "reasoning",
        "error",
    )
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the DataB DeepfakeJudge reliability gate.")
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--input-summary-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions_jsonl)
    expected = None
    if args.input_summary_json:
        with Path(args.input_summary_json).open("r", encoding="utf-8-sig") as handle:
            input_summary = json.load(handle)
        expected = int(input_summary.get("judge_records", 0)) or None
    summary, items = evaluate(rows, expected)
    summary["predictions_jsonl"] = str(Path(args.predictions_jsonl))
    summary["items_csv"] = str(Path(args.output_csv))
    write_json(args.output_json, summary)
    write_csv(args.output_csv, items)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=finite_or_none))


if __name__ == "__main__":
    main()
