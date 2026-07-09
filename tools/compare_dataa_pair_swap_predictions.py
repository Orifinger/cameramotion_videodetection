#!/usr/bin/env python3
"""Compare original and A/B-swapped DataA pair-selection predictions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


EDITED_TAG_RE = re.compile(r"<edited_video>\s*([AB])\s*</edited_video>", re.IGNORECASE | re.DOTALL)
ANSWER_TAG_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE | re.DOTALL)
CASE_RE = re.compile(r"(dataA_v1(?:_[A-Za-z][A-Za-z0-9]*)*_\d+)")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_prediction_files(path: str | Path) -> Iterable[Path]:
    path = Path(path)
    if path.is_file():
        yield path
        return
    for candidate in sorted(path.rglob("*.json")):
        name = candidate.name.lower()
        if "summary" in name or "metrics" in name:
            continue
        yield candidate


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for file_path in iter_prediction_files(path):
        payload = load_json(file_path)
        rows = payload.get("predictions") if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping):
                item = dict(row)
                item["_source_file"] = str(file_path)
                predictions.append(item)
    return predictions


def get_assistant(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def parse_choice(text: Any) -> str:
    text = str(text or "")
    for pattern in (
        EDITED_TAG_RE,
        ANSWER_TAG_RE,
        re.compile(r"\bedited\s+video\s*(?:is|:)?\s*([AB])\b", re.IGNORECASE),
        re.compile(r"\bvideo\s*([AB])\b", re.IGNORECASE),
    ):
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    if re.fullmatch(r"\s*[ABab]\s*", text):
        return text.strip().upper()
    return "UNKNOWN"


def invert(choice: str) -> str:
    return "B" if choice == "A" else "A" if choice == "B" else "UNKNOWN"


def case_id(record: Mapping[str, Any], index: int) -> str:
    if record.get("case_id"):
        return str(record["case_id"])
    images = record.get("images")
    first = str(images[0]).replace("\\", "/") if isinstance(images, list) and images else ""
    match = CASE_RE.search(first)
    return match.group(1) if match else f"sample_{index:06d}"


def gt_by_index(path: str | Path) -> dict[int, dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"expected list in {path}")
    out: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(data):
        if not isinstance(record, Mapping):
            continue
        out[index] = {
            "case_id": case_id(record, index),
            "gt": parse_choice(record.get("edited_video") or get_assistant(record)),
        }
    return out


def pred_by_index(path: str | Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for row in load_predictions(path):
        if not isinstance(row.get("data_index"), int):
            continue
        response = row.get("response", row.get("prediction", row.get("raw_response", "")))
        out[int(row["data_index"])] = parse_choice(response)
    return out


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-gt", required=True)
    parser.add_argument("--base-pred", required=True)
    parser.add_argument("--swap-gt", required=True)
    parser.add_argument("--swap-pred", required=True)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_gt = gt_by_index(args.base_gt)
    swap_gt = gt_by_index(args.swap_gt)
    base_pred = pred_by_index(args.base_pred)
    swap_pred = pred_by_index(args.swap_pred)

    indices = sorted(set(base_gt) & set(swap_gt) & set(base_pred) & set(swap_pred))
    valid = [i for i in indices if base_pred[i] in {"A", "B"} and swap_pred[i] in {"A", "B"}]
    base_correct = [i for i in valid if base_pred[i] == base_gt[i]["gt"]]
    swap_correct = [i for i in valid if swap_pred[i] == swap_gt[i]["gt"]]
    flipped = [i for i in valid if swap_pred[i] == invert(base_pred[i])]
    same_position = [i for i in valid if swap_pred[i] == base_pred[i]]
    both_correct = [i for i in valid if base_pred[i] == base_gt[i]["gt"] and swap_pred[i] == swap_gt[i]["gt"]]
    both_correct_and_flipped = [i for i in both_correct if swap_pred[i] == invert(base_pred[i])]

    summary = {
        "base_gt": args.base_gt,
        "base_pred": args.base_pred,
        "swap_gt": args.swap_gt,
        "swap_pred": args.swap_pred,
        "num_base_gt": len(base_gt),
        "num_swap_gt": len(swap_gt),
        "num_base_pred": len(base_pred),
        "num_swap_pred": len(swap_pred),
        "num_matched_indices": len(indices),
        "num_valid_pairs": len(valid),
        "base_accuracy_on_matched": safe_div(len(base_correct), len(valid)),
        "swap_accuracy_on_matched": safe_div(len(swap_correct), len(valid)),
        "base_pred_A_rate": safe_div(sum(1 for i in valid if base_pred[i] == "A"), len(valid)),
        "swap_pred_A_rate": safe_div(sum(1 for i in valid if swap_pred[i] == "A"), len(valid)),
        "prediction_flip_rate": safe_div(len(flipped), len(valid)),
        "same_position_prediction_rate": safe_div(len(same_position), len(valid)),
        "both_correct_rate": safe_div(len(both_correct), len(valid)),
        "both_correct_and_flipped_rate": safe_div(len(both_correct_and_flipped), len(valid)),
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
