#!/usr/bin/env python3
"""Build a grounded-CoT input index from Data A VACE attempt directories.

This script is intentionally read-only with respect to VACE artifacts. It scans
one or more run roots, keeps only complete full-video Real/Fake pairs, and emits
a compact JSONL contract for downstream V4 grounded-CoT authoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.common import DataAError, read_json, utc_now_iso, write_json
from scripts.dataa_v1.media_io import VideoMeta, assert_video_compatible, ffprobe_video


SCHEMA_VERSION = "dataA_v1_vace_grounded_cot_input_index_v1"


def _clean_text(value: Any, *, max_len: int = 500) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())[:max_len]


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def _shape(meta: VideoMeta) -> dict[str, Any]:
    return {
        "fps": float(meta.fps),
        "frame_count": int(meta.frame_count),
        "height": int(meta.height),
        "width": int(meta.width),
        "duration": float(meta.duration),
        "codec_name": meta.codec_name,
    }


def _load_masks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise DataAError(f"mask npz missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "masks" not in archive:
            raise DataAError(f"mask npz missing masks array: {path}")
        masks = archive["masks"]
        frame_indices = archive["frame_indices"] if "frame_indices" in archive else np.arange(masks.shape[0], dtype=np.int32)
    if masks.ndim != 3:
        raise DataAError(f"mask array must be [N,H,W], got {masks.shape}: {path}")
    if frame_indices.ndim != 1 or frame_indices.shape[0] != masks.shape[0]:
        raise DataAError(f"frame_indices do not match masks: {path}")
    return frame_indices.astype(np.int32), (masks > 0).astype(np.uint8)


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _union_bbox(boxes: Sequence[list[int]]) -> list[int] | None:
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _canonical_mapping(manifest: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    source_clip = manifest.get("source_clip") or {}
    canonical = source_clip.get("canonical") or {}
    mapping = canonical.get("frame_mapping") or []
    output: dict[int, Mapping[str, Any]] = {}
    if isinstance(mapping, list):
        for item in mapping:
            if isinstance(item, Mapping) and "canonical_frame" in item:
                output[int(item["canonical_frame"])] = item
    return output


def _frame_time_sec(
    *,
    canonical_frame: int,
    manifest: Mapping[str, Any],
    frame_mapping: Mapping[int, Mapping[str, Any]],
) -> float | None:
    source_clip = manifest.get("source_clip") or {}
    native = source_clip.get("native") or {}
    canonical = source_clip.get("canonical") or {}
    source_fps = float(native.get("source_fps") or 0.0)
    mapped = frame_mapping.get(int(canonical_frame))
    if mapped and source_fps > 0 and mapped.get("source_frame_float") is not None:
        return float(mapped["source_frame_float"]) / source_fps
    start_time = native.get("start_time_sec")
    generation_fps = float(canonical.get("generation_fps") or canonical.get("fps") or 0.0)
    if start_time is not None and generation_fps > 0:
        return float(start_time) + float(canonical_frame) / generation_fps
    return None


def _mask_evidence(mask_npz: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    frame_indices, masks = _load_masks(mask_npz)
    source_clip = manifest.get("source_clip") or {}
    canonical = source_clip.get("canonical") or {}
    valid_frame_count = int(canonical.get("valid_frame_count") or masks.shape[0])
    valid_frame_count = max(0, min(valid_frame_count, masks.shape[0]))
    frame_mapping = _canonical_mapping(manifest)
    tube = []
    boxes: list[list[int]] = []
    areas: list[int] = []
    for frame_pos in range(valid_frame_count):
        mask = masks[frame_pos]
        area = int(mask.sum())
        if area <= 0:
            continue
        bbox = _bbox_from_mask(mask)
        if bbox is None:
            continue
        canonical_frame = int(frame_indices[frame_pos]) if frame_pos < frame_indices.shape[0] else int(frame_pos)
        boxes.append(bbox)
        areas.append(area)
        tube.append(
            {
                "canonical_frame": canonical_frame,
                "source_time_sec": _frame_time_sec(
                    canonical_frame=canonical_frame,
                    manifest=manifest,
                    frame_mapping=frame_mapping,
                ),
                "bbox_xyxy": bbox,
                "area_px": area,
            }
        )
    key_index = int(np.argmax(np.array(areas))) if areas else None
    key_frame = tube[key_index] if key_index is not None else None
    return {
        "mask_npz_path": str(mask_npz),
        "mask_shape": {"frame_count": int(masks.shape[0]), "height": int(masks.shape[1]), "width": int(masks.shape[2])},
        "valid_frame_count": valid_frame_count,
        "nonempty_frame_count": int(len(tube)),
        "empty_valid_frame_count": int(valid_frame_count - len(tube)),
        "union_bbox_xyxy": _union_bbox(boxes),
        "key_frame": key_frame,
        "bbox_tube": tube,
        "area_stats": {
            "min_px": int(min(areas)) if areas else 0,
            "max_px": int(max(areas)) if areas else 0,
            "mean_px": float(sum(areas) / len(areas)) if areas else 0.0,
        },
    }


def _first_existing(*paths: Any) -> str:
    for value in paths:
        if value:
            path = Path(str(value))
            if path.is_file():
                return str(path)
    return str(paths[0]) if paths else ""


def _target_text(target: Mapping[str, Any]) -> str:
    for key in ("display_phrase", "canonical_concept", "taxonomy_label", "candidate_class", "region_family"):
        text = _clean_text(target.get(key), max_len=120)
        if text:
            return text
    entity = target.get("inventory_entity")
    if isinstance(entity, Mapping):
        for key in ("display_phrase", "sam3_prompt_phrase", "fine_type_raw", "coarse_type"):
            text = _clean_text(entity.get(key), max_len=120)
            if text:
                return text
    return ""


def _attempt_record(
    *,
    attempt_dir: Path,
    run_root: Path,
    run_label: str,
    vace_model: str,
    ffprobe_bin: str,
    strict_video_check: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    manifest_path = attempt_dir / "case_manifest.json"
    generation_path = attempt_dir / "generation_result.json"
    if not manifest_path.is_file():
        return None, "missing_case_manifest"
    manifest = _read_json_or_empty(manifest_path)
    generation = _read_json_or_empty(generation_path)
    case_id = _clean_text(manifest.get("case_id") or attempt_dir.name, max_len=160)
    full_video = generation.get("full_video") or manifest.get("full_video") or {}
    real_path = Path(_first_existing(full_video.get("full_real_path"), attempt_dir / "full_real.mp4"))
    fake_path = Path(_first_existing(full_video.get("full_fake_path"), attempt_dir / "full_fake.mp4"))
    if not real_path.is_file():
        return None, "missing_full_real"
    if not fake_path.is_file():
        return None, "missing_full_fake"

    video_meta = None
    if strict_video_check:
        real_meta = ffprobe_video(real_path, ffprobe_bin=ffprobe_bin)
        fake_meta = ffprobe_video(fake_path, ffprobe_bin=ffprobe_bin)
        assert_video_compatible(real_meta, fake_meta)
        video_meta = {"real": _shape(real_meta), "fake": _shape(fake_meta)}

    masks = manifest.get("mask_layers") or {}
    mask_npz = Path(str(masks.get("M_gen") or attempt_dir / "target_mask_gen.npz"))
    alpha_npz = Path(str(masks.get("M_alpha") or attempt_dir / "target_mask_alpha.npz"))
    if not mask_npz.is_file():
        return None, "missing_target_mask_gen_npz"
    evidence = _mask_evidence(mask_npz, manifest)
    if evidence["union_bbox_xyxy"] is None:
        return None, "empty_target_mask_gen"

    source_clip = manifest.get("source_clip") or {}
    native = source_clip.get("native") or {}
    canonical = source_clip.get("canonical") or {}
    target = manifest.get("target") or {}
    donor = manifest.get("donor") or None
    prompt = manifest.get("prompt") or {}
    sampling_meta = manifest.get("sampling_meta") or {}
    edit_time_range = (
        full_video.get("source_time_range_sec")
        or [
            native.get("start_time_sec"),
            native.get("end_time_sec"),
        ]
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "run_id": run_root.name,
        "run_label": run_label,
        "vace_model": vace_model,
        "attempt_dir": str(attempt_dir),
        "worker_id": attempt_dir.parent.parent.name if attempt_dir.parent.name == "attempts" else "",
        "real_video": str(real_path),
        "fake_video": str(fake_path),
        "case_manifest": str(manifest_path),
        "generation_result": str(generation_path) if generation_path.is_file() else "",
        "artifact_warnings": [] if generation_path.is_file() else ["missing_generation_result"],
        "mask_npz": str(mask_npz),
        "alpha_npz": str(alpha_npz) if alpha_npz.is_file() else "",
        "operation": _clean_text(manifest.get("operation"), max_len=120),
        "generator_route": _clean_text(manifest.get("generator_route"), max_len=160),
        "target": {
            "track_id": target.get("track_id"),
            "video_id": target.get("video_id"),
            "display_phrase": target.get("display_phrase"),
            "canonical_concept": target.get("canonical_concept"),
            "candidate_class": target.get("candidate_class"),
            "region_family": target.get("region_family"),
            "taxonomy_label": target.get("taxonomy_label"),
            "compatibility_group": target.get("compatibility_group"),
            "inventory_entity": target.get("inventory_entity") or {},
        },
        "donor": None
        if not isinstance(donor, Mapping)
        else {
            "track_id": donor.get("track_id"),
            "video_id": donor.get("video_id"),
            "display_phrase": donor.get("display_phrase"),
            "canonical_concept": donor.get("canonical_concept"),
            "taxonomy_label": donor.get("taxonomy_label"),
            "compatibility_group": donor.get("compatibility_group"),
        },
        "target_text": _target_text(target if isinstance(target, Mapping) else {}),
        "target_taxonomy": target.get("taxonomy_label"),
        "edit_time_range_sec": edit_time_range,
        "edit_bbox_xyxy": evidence["union_bbox_xyxy"],
        "evidence_mask": evidence,
        "source_clip": {
            "source_video_path": source_clip.get("source_video_path"),
            "native": native,
            "canonical": {
                "fps": canonical.get("fps"),
                "generation_fps": canonical.get("generation_fps"),
                "frame_count": canonical.get("frame_count"),
                "valid_frame_count": canonical.get("valid_frame_count"),
                "pad_frame_count": canonical.get("pad_frame_count"),
                "height": canonical.get("height"),
                "width": canonical.get("width"),
            },
        },
        "full_video": full_video,
        "video_meta": video_meta,
        "mask_policy": sampling_meta.get("mask_policy") or {},
        "artifact_policy": sampling_meta.get("artifact_policy") or {},
        "sampling_meta": {
            "schema_version": sampling_meta.get("schema_version"),
            "subject_first_source": sampling_meta.get("subject_first_source"),
            "taxonomy": sampling_meta.get("taxonomy") or {},
            "vace_model_plan": sampling_meta.get("vace_model_plan") or {},
        },
        "prompt": {
            "model_prompt": prompt.get("model_prompt"),
            "control_prompt": prompt.get("control_prompt"),
        },
    }
    return record, None


def build_index(
    *,
    run_roots: Sequence[Path],
    out_jsonl: Path,
    out_summary: Path,
    ffprobe_bin: str,
    strict_video_check: bool,
    max_cases: int | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    skipped = Counter()
    run_counts = Counter()
    op_counts = Counter()
    model_counts = Counter()
    for run_root in run_roots:
        if not run_root.is_dir():
            skipped["missing_run_root"] += 1
            continue
        vace_model = "vace14b" if "vace14b" in str(run_root).lower() else "vace13b" if "vace13b" in str(run_root).lower() else ""
        for manifest_path in sorted(run_root.rglob("case_manifest.json")):
            attempt_dir = manifest_path.parent
            try:
                record, reason = _attempt_record(
                    attempt_dir=attempt_dir,
                    run_root=run_root,
                    run_label=run_root.name,
                    vace_model=vace_model,
                    ffprobe_bin=ffprobe_bin,
                    strict_video_check=strict_video_check,
                )
            except DataAError as exc:
                skipped[f"attempt_error:{str(exc).split(':', 1)[0]}"] += 1
                continue
            if record is None:
                skipped[str(reason or "unknown_skip")] += 1
                continue
            records.append(record)
            run_counts[record["run_id"]] += 1
            op_counts[record["operation"]] += 1
            model_counts[record["vace_model"]] += 1
            if max_cases is not None and len(records) >= max_cases:
                break
        if max_cases is not None and len(records) >= max_cases:
            break

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now_iso(),
        "out_jsonl": str(out_jsonl),
        "run_roots": [str(path) for path in run_roots],
        "strict_video_check": bool(strict_video_check),
        "record_count": len(records),
        "run_counts": dict(run_counts),
        "vace_model_counts": dict(model_counts),
        "operation_counts": dict(op_counts),
        "skipped_counts": dict(skipped),
    }
    write_json(out_summary, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", type=Path, required=True, help="VACE run root. Can be repeated.")
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("res/dataA_v1/autolabel/dataa_vace_grounded_cot_input_index.jsonl"),
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("res/dataA_v1/autolabel/dataa_vace_grounded_cot_input_summary.json"),
    )
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--no-video-check", action="store_true", help="Skip ffprobe full_real/full_fake compatibility checks.")
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_index(
            run_roots=args.run_root,
            out_jsonl=args.out_jsonl,
            out_summary=args.out_summary,
            ffprobe_bin=str(args.ffprobe_bin),
            strict_video_check=not bool(args.no_video_check),
            max_cases=args.max_cases,
        )
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        "grounded_cot_index "
        f"records={summary['record_count']} "
        f"runs={summary['run_counts']} "
        f"skipped={summary['skipped_counts']} "
        f"out={args.out_jsonl}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
