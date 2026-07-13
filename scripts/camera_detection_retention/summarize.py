#!/usr/bin/env python3
"""Compare original-prompt DataA detection before and after camera VQA LoRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(payload)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compact(payload: Mapping[str, Any]) -> dict[str, Any]:
    basic = payload.get("basic")
    pair = payload.get("pair")
    if not isinstance(basic, Mapping) or not isinstance(pair, Mapping):
        raise ValueError("DataA evaluation JSON must contain basic and pair objects")
    gt = int(payload.get("num_gt_records", 0))
    matched = int(payload.get("num_matched_records", 0))
    output = {
        "num_gt_records": gt,
        "num_matched_records": matched,
        "coverage": matched / gt if gt else 0.0,
        "format_valid_rate": float(basic.get("format_valid_rate", 0.0)),
        "accuracy": float(basic.get("accuracy", 0.0)),
        "balanced_accuracy": float(basic.get("balanced_accuracy", 0.0)),
        "fake_recall": float(basic.get("fake_recall", 0.0)),
        "real_recall": float(basic.get("real_recall", 0.0)),
        "fake_f1": float(basic.get("fake_f1", 0.0)),
        "pair_accuracy": float(pair.get("pair_accuracy", 0.0)),
        "num_pairs": int(pair.get("num_pairs", 0)),
    }
    iou = payload.get("iou")
    if isinstance(iou, Mapping):
        output["evidence"] = {
            "pred_evidence_sample_rate": float(iou.get("pred_evidence_sample_rate", 0.0)),
            "mean_best_temporal_iou": float(iou.get("mean_best_temporal_iou", 0.0)),
            "mean_best_bbox_iou": float(iou.get("mean_best_bbox_iou", 0.0)),
            "evidence_hit_t03_b03": float(iou.get("evidence_hit_t03_b03", 0.0)),
            "sample_any_evidence_hit_t03_b03": float(
                iou.get("sample_any_evidence_hit_t03_b03", 0.0)
            ),
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-eval", required=True)
    parser.add_argument("--camera-eval", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--min-format-valid", type=float, default=0.95)
    parser.add_argument("--max-balanced-accuracy-drop", type=float, default=0.03)
    parser.add_argument("--max-fake-f1-drop", type=float, default=0.03)
    parser.add_argument("--max-pair-accuracy-drop", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = compact(read_json(args.base_eval))
    camera = compact(read_json(args.camera_eval))
    deltas = {
        "balanced_accuracy": camera["balanced_accuracy"] - base["balanced_accuracy"],
        "fake_f1": camera["fake_f1"] - base["fake_f1"],
        "pair_accuracy": camera["pair_accuracy"] - base["pair_accuracy"],
        "format_valid_rate": camera["format_valid_rate"] - base["format_valid_rate"],
    }
    checks = {
        "base_coverage": base["coverage"] >= args.min_coverage,
        "camera_coverage": camera["coverage"] >= args.min_coverage,
        "camera_format_valid": camera["format_valid_rate"] >= args.min_format_valid,
        "balanced_accuracy_retained": deltas["balanced_accuracy"]
        >= -args.max_balanced_accuracy_drop,
        "fake_f1_retained": deltas["fake_f1"] >= -args.max_fake_f1_drop,
        "pair_accuracy_retained": deltas["pair_accuracy"] >= -args.max_pair_accuracy_drop,
    }
    output = {
        "gate": "camera VQA adapter detection retention on fixed DataA development cases",
        "status": "passed" if all(checks.values()) else "failed",
        "what_was_tested": (
            "The original detection checkpoint and the same checkpoint with the final binary-camera "
            "LoRA use identical 40step_v3 frames and the original detection prompt. No camera text is "
            "provided at detection inference."
        ),
        "what_was_not_tested": (
            "This diagnostic does not test a jointly trained detector, VIF-Bench retention, or a camera-driven "
            "detection improvement. The fixed DataA development identities are not a fresh final test set."
        ),
        "thresholds": {
            "min_coverage": args.min_coverage,
            "min_format_valid": args.min_format_valid,
            "max_balanced_accuracy_drop": args.max_balanced_accuracy_drop,
            "max_fake_f1_drop": args.max_fake_f1_drop,
            "max_pair_accuracy_drop": args.max_pair_accuracy_drop,
        },
        "checks": checks,
        "base_detection_checkpoint": base,
        "camera_vqa_adapter": camera,
        "camera_minus_base": deltas,
        "next_action": (
            "Proceed to an equal-step detection-only versus detection-plus-camera-auxiliary training gate."
            if all(checks.values())
            else "Use explicit detection replay in the joint gate; do not treat the camera-only adapter as a detector."
        ),
    }
    write_json(args.output_json, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

