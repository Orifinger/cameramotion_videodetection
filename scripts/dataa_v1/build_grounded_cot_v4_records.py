#!/usr/bin/env python3
"""Prepare Data A VACE outputs for v4 grounded-CoT authoring.

The downstream v4 evidence-rescue script expects frame directories plus
grounded evidence fields. Data A already knows the edited time range and mask
bbox, so this adapter extracts aligned Real/Fake frames and emits records with
``edit_time_range`` and normalized ``edit_bbox``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.common import DataAError, utc_now_iso, write_json


SCHEMA_VERSION = "dataA_v1_vace_grounded_cot_v4_record_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataAError(f"invalid jsonl at {path}:{line_no}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _clean(value: Any, *, max_len: int = 240) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())[:max_len]


def _time_range(record: Mapping[str, Any]) -> list[float] | None:
    for key in ("edit_time_range_sec", "edit_time_range", "time_range"):
        value = record.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
            try:
                start = float(value[0])
                end = float(value[1])
            except (TypeError, ValueError):
                continue
            if end < start:
                start, end = end, start
            return [round(max(0.0, start), 4), round(max(0.0, end), 4)]
    return None


def _mask_shape(record: Mapping[str, Any]) -> tuple[int, int] | None:
    evidence = record.get("evidence_mask") if isinstance(record.get("evidence_mask"), Mapping) else {}
    shape = evidence.get("mask_shape") if isinstance(evidence.get("mask_shape"), Mapping) else {}
    width = shape.get("width")
    height = shape.get("height")
    if width and height:
        return int(width), int(height)
    video_meta = record.get("video_meta") if isinstance(record.get("video_meta"), Mapping) else {}
    fake_meta = video_meta.get("fake") if isinstance(video_meta.get("fake"), Mapping) else {}
    width = fake_meta.get("width")
    height = fake_meta.get("height")
    if width and height:
        return int(width), int(height)
    source_clip = record.get("source_clip") if isinstance(record.get("source_clip"), Mapping) else {}
    canonical = source_clip.get("canonical") if isinstance(source_clip.get("canonical"), Mapping) else {}
    width = canonical.get("width")
    height = canonical.get("height")
    if width and height:
        return int(width), int(height)
    return None


def _bbox(record: Mapping[str, Any]) -> tuple[list[int], list[int]] | None:
    value = record.get("edit_bbox_xyxy") or record.get("edit_bbox") or record.get("bbox_xyxy") or record.get("bbox")
    if not (isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 4):
        return None
    try:
        raw = [int(round(float(value[i]))) for i in range(4)]
    except (TypeError, ValueError):
        return None
    shape = _mask_shape(record)
    if not shape:
        return raw, raw
    width, height = shape
    if width <= 0 or height <= 0:
        return raw, raw
    x1, y1, x2, y2 = raw
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    norm = [
        int(round(x1 / width * 1000.0)),
        int(round(y1 / height * 1000.0)),
        int(round(x2 / width * 1000.0)),
        int(round(y2 / height * 1000.0)),
    ]
    norm = [max(0, min(1000, v)) for v in norm]
    return raw, norm


def _linspace(start: float, end: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [round((start + end) / 2.0, 4)]
    return [round(start + (end - start) * i / (count - 1), 4) for i in range(count)]


def _existing_frames(frame_dir: Path) -> list[Path]:
    return sorted(frame_dir.glob("*.png"))


def _write_timestamps(frame_dir: Path, timestamps: Iterable[float]) -> None:
    (frame_dir / "timestamps.txt").write_text(
        "\n".join(f"{float(t):.4f}" for t in timestamps) + "\n",
        encoding="utf-8",
    )


def _extract_frames(
    *,
    video_path: Path,
    frame_dir: Path,
    start: float,
    end: float,
    max_frames: int,
    ffmpeg_bin: str,
    overwrite: bool,
) -> tuple[list[str], list[float]]:
    if not video_path.is_file():
        raise DataAError(f"missing video: {video_path}")
    frame_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = _existing_frames(frame_dir)
        ts_path = frame_dir / "timestamps.txt"
        if existing and ts_path.is_file():
            timestamps = [float(line.strip()) for line in ts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return [str(path) for path in existing], timestamps[: len(existing)]
    for path in _existing_frames(frame_dir):
        path.unlink()

    duration = max(0.04, float(end) - float(start))
    fps = max(1.0, float(max_frames) / duration)
    output_pattern = str(frame_dir / "%04d.png")
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, float(start)):.6f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.6f}",
        "-vf",
        f"fps={fps:.8f}",
        "-frames:v",
        str(max_frames),
        output_pattern,
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise DataAError(f"ffmpeg_extract_failed:{video_path}:{result.stderr.strip()[:500]}")
    frames = _existing_frames(frame_dir)
    if not frames:
        raise DataAError(f"ffmpeg_extract_empty:{video_path}")
    timestamps = _linspace(float(start), float(end), len(frames))
    _write_timestamps(frame_dir, timestamps)
    return [str(path) for path in frames], timestamps


def _target_record(record: Mapping[str, Any]) -> dict[str, Any]:
    target = record.get("target") if isinstance(record.get("target"), Mapping) else {}
    target_text = _clean(record.get("target_text") or target.get("display_phrase") or target.get("taxonomy_label") or "the marked subject")
    out = dict(target)
    out["display_phrase"] = target_text
    return out


def _build_record(
    *,
    record: Mapping[str, Any],
    frame_root: Path,
    ffmpeg_bin: str,
    max_frames: int,
    overwrite_frames: bool,
    extract_frames: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    case_id = _clean(record.get("case_id"), max_len=160)
    if not case_id:
        return None, "missing_case_id"
    time_range = _time_range(record)
    if not time_range:
        return None, "missing_edit_time_range"
    bbox_pair = _bbox(record)
    if not bbox_pair:
        return None, "missing_edit_bbox"
    raw_bbox, norm_bbox = bbox_pair
    real_video = Path(str(record.get("real_video") or ""))
    fake_video = Path(str(record.get("fake_video") or ""))
    if not real_video.is_file():
        return None, "missing_real_video"
    if not fake_video.is_file():
        return None, "missing_fake_video"

    case_frame_root = frame_root / case_id
    real_frame_dir = case_frame_root / "real"
    fake_frame_dir = case_frame_root / "fake"
    if extract_frames:
        _extract_frames(
            video_path=real_video,
            frame_dir=real_frame_dir,
            start=time_range[0],
            end=time_range[1],
            max_frames=max_frames,
            ffmpeg_bin=ffmpeg_bin,
            overwrite=overwrite_frames,
        )
        _extract_frames(
            video_path=fake_video,
            frame_dir=fake_frame_dir,
            start=time_range[0],
            end=time_range[1],
            max_frames=max_frames,
            ffmpeg_bin=ffmpeg_bin,
            overwrite=overwrite_frames,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "run_id": record.get("run_id"),
        "vace_model": record.get("vace_model"),
        "operation": record.get("operation"),
        "target": _target_record(record),
        "target_phrase": _clean(record.get("target_text") or "the marked subject"),
        "real_video": str(real_video),
        "fake_video": str(fake_video),
        "real_frame_dir": str(real_frame_dir),
        "fake_frame_dir": str(fake_frame_dir),
        "edit_time_range": [round(float(time_range[0]), 2), round(float(time_range[1]), 2)],
        "time_range": [round(float(time_range[0]), 2), round(float(time_range[1]), 2)],
        "edit_bbox": norm_bbox,
        "evidence_bbox": norm_bbox,
        "edit_bbox_source_xyxy": raw_bbox,
        "case_manifest_path": record.get("case_manifest"),
        "generation_result_path": record.get("generation_result"),
        "mask_npz": record.get("mask_npz"),
        "artifact_warnings": record.get("artifact_warnings") or [],
        "dataa_index_schema_version": record.get("schema_version"),
    }, None


def build_records(
    *,
    input_index: Path,
    out_jsonl: Path,
    out_summary: Path,
    frame_root: Path,
    ffmpeg_bin: str,
    max_frames: int,
    limit: int | None,
    overwrite_frames: bool,
    dry_run: bool,
) -> dict[str, Any]:
    source_rows = _read_jsonl(input_index)
    if limit is not None and limit > 0:
        source_rows = source_rows[:limit]
    records: list[dict[str, Any]] = []
    skipped = Counter()
    run_counts = Counter()
    op_counts = Counter()
    model_counts = Counter()
    for row in source_rows:
        try:
            record, reason = _build_record(
                record=row,
                frame_root=frame_root,
                ffmpeg_bin=ffmpeg_bin,
                max_frames=max_frames,
                overwrite_frames=overwrite_frames,
                extract_frames=not dry_run,
            )
        except DataAError as exc:
            skipped[str(exc).split(":", 1)[0]] += 1
            continue
        if record is None:
            skipped[str(reason or "unknown_skip")] += 1
            continue
        records.append(record)
        run_counts[str(record.get("run_id") or "")] += 1
        op_counts[str(record.get("operation") or "")] += 1
        model_counts[str(record.get("vace_model") or "")] += 1

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        with out_jsonl.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now_iso(),
        "dry_run": bool(dry_run),
        "input_index": str(input_index),
        "out_jsonl": str(out_jsonl),
        "frame_root": str(frame_root),
        "source_record_count": len(source_rows),
        "record_count": len(records),
        "max_frames": int(max_frames),
        "run_counts": dict(run_counts),
        "vace_model_counts": dict(model_counts),
        "operation_counts": dict(op_counts),
        "skipped_counts": dict(skipped),
    }
    if not dry_run:
        write_json(out_summary, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-index",
        type=Path,
        default=Path("res/dataA_v1/autolabel/dataa_vace_grounded_cot_input_index.jsonl"),
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("res/dataA_v1/autolabel/dataa_vace_grounded_cot_v4_records.jsonl"),
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("res/dataA_v1/autolabel/dataa_vace_grounded_cot_v4_records_summary.json"),
    )
    parser.add_argument(
        "--frame-root",
        type=Path,
        default=Path("/tmp/cameramotion_det/dataA_v1/autolabel/dataa_vace_grounded_cot_frames"),
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite-frames", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_records(
            input_index=args.input_index,
            out_jsonl=args.out_jsonl,
            out_summary=args.out_summary,
            frame_root=args.frame_root,
            ffmpeg_bin=str(args.ffmpeg_bin),
            max_frames=args.max_frames,
            limit=args.limit,
            overwrite_frames=bool(args.overwrite_frames),
            dry_run=bool(args.dry_run),
        )
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        "grounded_cot_v4_records "
        f"records={summary['record_count']} "
        f"runs={summary['run_counts']} "
        f"skipped={summary['skipped_counts']} "
        f"out={args.out_jsonl}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
