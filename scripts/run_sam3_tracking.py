#!/usr/bin/env python3
"""Build a SAM3 multi-instance track bank from Qwen v4 candidates.

Run after Qwen v4 candidate discovery:
    CUDA_VISIBLE_DEVICES=<physical_gpu_id> python scripts/run_sam3_tracking.py

The script keeps all SAM3 instance tracks for every Qwen concept. It does not
select a final focus region and does not plan an edit operation.
"""

from __future__ import annotations

import json
import math
import argparse
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sam3_tracking_config import (
    QWEN_INPUT_SCHEMA_VERSION,
    QWEN_SAM3_CANDIDATES_PATH,
    SAM3_CHECKPOINT_PATH,
    SAM3_CLEAR_CACHE_THRESHOLD,
    SAM3_CLOSE_SESSION_RUN_GC,
    SAM3_FAILURES_PATH,
    SAM3_MAX_BORDER_TOUCH_RATIO,
    SAM3_MAX_CANDIDATES_PER_VIDEO,
    SAM3_MAX_MEDIAN_AREA_RATIO,
    SAM3_MAX_VIDEOS,
    SAM3_MIN_LONGEST_VISIBLE_RUN,
    SAM3_MIN_MEDIAN_AREA_RATIO,
    SAM3_MIN_VISIBLE_FRAME_RATIO,
    SAM3_OUTPUT_PROB_THRESH,
    SAM3_PROPAGATION_DIRECTION,
    SAM3_QUALITY_TRACKS_PATH,
    SAM3_RUN_SUMMARY_PATH,
    SAM3_SAVE_MASK_TUBES,
    SAM3_SCHEMA_VERSION,
    SAM3_SOURCE_ROOT,
    SAM3_TRACK_MASK_ROOT,
    SAM3_TRACKS_ALL_PATH,
)

build_sam3_video_predictor = None


REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id", "region_family", "candidate_class", "target_scope",
    "canonical_concept", "display_phrase", "sam_prompt", "instance_count_hint",
    "visual_disambiguators", "screen_region", "temporal_presence",
}
ALLOWED_FAMILIES = {"physical_instance", "editable_surface"}
ALLOWED_CLASSES = {
    "human", "animal", "vehicle", "handheld_object", "bounded_object",
    "display_screen", "sign_or_poster", "paper_book_map", "framed_art",
    "apparel_panel", "vehicle_panel", "package_front",
}
ALLOWED_HINTS = {"unique_in_video", "possibly_multiple", "unknown"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_scalar(value: Any) -> float | None:
    values = to_numpy(value)
    if values.size != 1:
        return None
    try:
        number = float(values.reshape(-1)[0])
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def longest_run(indices: list[int]) -> int:
    if not indices:
        return 0
    values = sorted(set(indices))
    best = current = 1
    for previous, current_index in zip(values, values[1:]):
        current = current + 1 if current_index == previous + 1 else 1
        best = max(best, current)
    return best


def touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def gpu_memory_snapshot() -> dict[str, int] | None:
    if torch is None or not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    return {
        "device": int(device),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def video_frame_count(video_path: str) -> int | None:
    if cv2 is None:
        return None
    capture = cv2.VideoCapture(video_path)
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return count if count > 0 else None


def prompt_anchor_frame(candidate: dict[str, Any], frame_count: int | None) -> int:
    if not frame_count or frame_count <= 1:
        return 0
    last = frame_count - 1
    presence = candidate["temporal_presence"]
    if presence in {"middle", "brief"}:
        return last // 2
    if presence == "late":
        return int(round(last * 0.75))
    return 0


def validate_candidate(raw: Any, index: int) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw, dict):
        return None, f"candidate[{index}] must be an object"
    missing = sorted(REQUIRED_CANDIDATE_FIELDS - set(raw))
    if missing:
        return None, f"candidate[{index}] missing fields: {missing}"
    family = raw["region_family"]
    candidate_class = raw["candidate_class"]
    target_scope = raw["target_scope"]
    if family not in ALLOWED_FAMILIES or candidate_class not in ALLOWED_CLASSES:
        return None, f"candidate[{index}] invalid family or candidate_class"
    required_scope = "whole_instance" if family == "physical_instance" else "whole_surface"
    if target_scope != required_scope:
        return None, f"candidate[{index}] family and target_scope mismatch"
    if raw["instance_count_hint"] not in ALLOWED_HINTS:
        return None, f"candidate[{index}] invalid instance_count_hint"
    if not isinstance(raw["visual_disambiguators"], list):
        return None, f"candidate[{index}] visual_disambiguators must be a list"
    if not isinstance(raw["sam_prompt"], str) or not raw["sam_prompt"].strip():
        return None, f"candidate[{index}] missing sam_prompt"
    return dict(raw), None


def load_input() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not QWEN_SAM3_CANDIDATES_PATH.is_file():
        raise FileNotFoundError(f"Missing Qwen v4 candidate view: {QWEN_SAM3_CANDIDATES_PATH}")
    dataset = json.loads(QWEN_SAM3_CANDIDATES_PATH.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or dataset.get("schema_version") != QWEN_INPUT_SCHEMA_VERSION:
        raise ValueError(f"Expected Qwen schema {QWEN_INPUT_SCHEMA_VERSION!r}")
    videos = dataset.get("videos")
    if not isinstance(videos, list):
        raise ValueError("Qwen candidate input must contain a top-level videos list")
    records = [
        item for item in videos
        if isinstance(item, dict)
        and item.get("video_id")
        and item.get("video_path")
        and isinstance(item.get("sam3_candidates"), list)
    ]
    return dataset, records


class Sam3Runner:
    def __init__(self) -> None:
        global build_sam3_video_predictor
        if not SAM3_SOURCE_ROOT.is_dir() or not SAM3_CHECKPOINT_PATH.is_file():
            raise FileNotFoundError("SAM3 source directory or checkpoint is unavailable")
        if torch is None:
            raise RuntimeError("PyTorch is required for SAM3")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SAM3")
        if build_sam3_video_predictor is None:
            if str(SAM3_SOURCE_ROOT) not in sys.path:
                sys.path.insert(0, str(SAM3_SOURCE_ROOT))
            from sam3.model_builder import build_sam3_video_predictor as builder
            build_sam3_video_predictor = builder
        self.predictor = build_sam3_video_predictor(
            checkpoint_path=str(SAM3_CHECKPOINT_PATH),
            gpus_to_use=[0],
        )
        print(f"[sam3] loaded: {gpu_memory_snapshot()}")

    def start_session(self, video_path: str) -> str:
        response = self.predictor.handle_request({"type": "start_session", "resource_path": video_path})
        return str(response["session_id"])

    def reset_session(self, session_id: str) -> None:
        self.predictor.handle_request({"type": "reset_session", "session_id": session_id})

    def add_text_prompt(self, session_id: str, prompt: str, frame_index: int) -> dict[str, Any]:
        return self.predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_index,
            "text": prompt,
            "output_prob_thresh": SAM3_OUTPUT_PROB_THRESH,
        })

    def propagate(self, session_id: str):
        yield from self.predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": SAM3_PROPAGATION_DIRECTION,
            "output_prob_thresh": SAM3_OUTPUT_PROB_THRESH,
        })

    def close_session(self, session_id: str) -> Any:
        return self.predictor.handle_request({
            "type": "close_session",
            "session_id": session_id,
            "run_gc_collect": SAM3_CLOSE_SESSION_RUN_GC,
            "clear_cache_threshold": SAM3_CLEAR_CACHE_THRESHOLD,
        })

    def shutdown(self) -> None:
        self.predictor.shutdown()


def save_mask_tube(video_id: str, candidate_id: str, object_id: int, frames: list[int], masks: list[np.ndarray]) -> str | None:
    if not SAM3_SAVE_MASK_TUBES or not masks:
        return None
    directory = SAM3_TRACK_MASK_ROOT / video_id
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"{candidate_id}__obj_{object_id}.npz"
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            frame_indices=np.asarray(frames, dtype=np.int32),
            masks=np.stack(masks, axis=0).astype(np.uint8, copy=False),
        )
    temporary.replace(output_path)
    return str(output_path)


def evaluate_track(track: dict[str, Any]) -> tuple[str, list[str], float]:
    reasons: list[str] = []
    if track["visible_frame_ratio"] < SAM3_MIN_VISIBLE_FRAME_RATIO:
        reasons.append("visible_frame_ratio")
    if track["longest_visible_run"] < SAM3_MIN_LONGEST_VISIBLE_RUN:
        reasons.append("longest_visible_run")
    if track["median_area_ratio"] < SAM3_MIN_MEDIAN_AREA_RATIO:
        reasons.append("median_area_ratio_too_small")
    if track["median_area_ratio"] > SAM3_MAX_MEDIAN_AREA_RATIO:
        reasons.append("median_area_ratio_too_large")
    if track["border_touch_ratio"] > SAM3_MAX_BORDER_TOUCH_RATIO:
        reasons.append("border_touch_ratio")
    stable = min(track["longest_visible_run"] / max(track["propagation_frame_count"], 1), 1.0)
    area_good = float(SAM3_MIN_MEDIAN_AREA_RATIO <= track["median_area_ratio"] <= SAM3_MAX_MEDIAN_AREA_RATIO)
    score = (
        0.40 * track["visible_frame_ratio"]
        + 0.25 * stable
        + 0.15 * area_good
        + 0.10 * (1.0 - track["border_touch_ratio"])
        + 0.10 * (track["mean_detection_score"] or 0.0)
    )
    return ("pass" if not reasons else "fail"), reasons, round(score, 6)


def track_candidate(runner: Sam3Runner, session_id: str, video_id: str, candidate: dict[str, Any], source_frames: int | None) -> dict[str, Any]:
    runner.reset_session(session_id)
    anchor_frame = prompt_anchor_frame(candidate, source_frames)
    prompt_response = runner.add_text_prompt(session_id, candidate["sam_prompt"], anchor_frame)

    tracks_by_object: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    propagation_frames: set[int] = set()
    output_keys: set[str] = set()
    mask_shape: tuple[int, int] | None = None

    for packet in runner.propagate(session_id):
        if not isinstance(packet, dict) or "frame_index" not in packet:
            continue
        frame_index = int(packet["frame_index"])
        outputs = packet.get("outputs")
        if not isinstance(outputs, dict):
            continue
        propagation_frames.add(frame_index)
        output_keys.update(outputs)
        object_ids = to_numpy(outputs.get("out_obj_ids", []))
        masks = to_numpy(outputs.get("out_binary_masks", []))
        boxes = to_numpy(outputs.get("out_boxes_xywh", []))
        scores = to_numpy(outputs.get("out_probs", []))
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None]
        if masks.ndim == 3 and len(masks):
            mask_shape = (int(masks.shape[1]), int(masks.shape[2]))
        if object_ids.ndim == 0:
            object_ids = object_ids.reshape(1)
        for item_index, raw_id in enumerate(object_ids.reshape(-1).tolist()):
            if item_index >= len(masks):
                continue
            mask = np.asarray(masks[item_index], dtype=bool)
            if mask.ndim != 2 or not mask.any():
                continue
            tracks_by_object[int(raw_id)][frame_index] = {
                "mask": mask,
                "box": boxes[item_index].tolist() if item_index < len(boxes) else None,
                "score": to_scalar(scores[item_index]) if item_index < len(scores) else None,
            }

    propagation_frame_count = len(propagation_frames)
    tracks: list[dict[str, Any]] = []
    for object_id, observations in tracks_by_object.items():
        ordered = sorted(observations.items())
        frames = [frame for frame, _ in ordered]
        masks = [data["mask"] for _, data in ordered]
        area_ratios = [float(mask.mean()) for mask in masks]
        confidence_scores = [data["score"] for _, data in ordered if data["score"] is not None]
        border_flags = [touches_border(mask) for mask in masks]
        track = {
            "track_id": f"{video_id}__{candidate['candidate_id']}__obj_{object_id}",
            "sam_object_id": int(object_id),
            "first_frame_index": min(frames),
            "last_frame_index": max(frames),
            "visible_frame_count": len(frames),
            "propagation_frame_count": propagation_frame_count,
            "visible_frame_ratio": round(len(frames) / max(propagation_frame_count, 1), 6),
            "longest_visible_run": longest_run(frames),
            "median_area_ratio": round(float(np.median(area_ratios)), 8),
            "mean_area_ratio": round(float(np.mean(area_ratios)), 8),
            "std_area_ratio": round(float(np.std(area_ratios)), 8),
            "mean_detection_score": round(float(np.mean(confidence_scores)), 6) if confidence_scores else None,
            "border_touch_ratio": round(float(np.mean(border_flags)), 6),
            "bbox_coordinate_convention": "sam3_output_xywh",
            "bbox_tube_xywh": [
                {"frame_index": frame, "bbox_xywh": data["box"]}
                for frame, data in ordered
            ],
            "mask_tube_path": save_mask_tube(video_id, candidate["candidate_id"], int(object_id), frames, masks),
        }
        quality_status, quality_reasons, quality_score = evaluate_track(track)
        track["quality_status"] = quality_status
        track["quality_reasons"] = quality_reasons
        track["track_quality_score"] = quality_score
        tracks.append(track)

    tracks.sort(key=lambda item: item["track_quality_score"], reverse=True)
    return {
        "candidate_id": candidate["candidate_id"],
        "region_family": candidate["region_family"],
        "candidate_class": candidate["candidate_class"],
        "canonical_concept": candidate["canonical_concept"],
        "display_phrase": candidate["display_phrase"],
        "sam_prompt": candidate["sam_prompt"],
        "instance_count_hint": candidate["instance_count_hint"],
        "visual_disambiguators": candidate["visual_disambiguators"],
        "prompt_frame_index": anchor_frame,
        "source_video_frame_count": source_frames,
        "anchor_policy": "qwen_temporal_presence",
        "add_prompt_output_keys": sorted(prompt_response.get("outputs", {})) if isinstance(prompt_response.get("outputs"), dict) else None,
        "propagation_frame_count": propagation_frame_count,
        "mask_frame_height_width": list(mask_shape) if mask_shape else None,
        "propagation_output_keys": sorted(output_keys),
        "tracks": tracks,
        "status": "success" if tracks else "no_track",
    }


def process_video(runner: Sam3Runner, video: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    video_id = str(video["video_id"])
    video_path = str(video["video_path"])
    source_frames = video_frame_count(video_path)
    result: dict[str, Any] = {
        "video_id": video_id,
        "relative_path": video.get("relative_path"),
        "video_path": video_path,
        "source_video_frame_count": source_frames,
        "status": "failure",
        "candidate_results": [],
        "created_at_utc": utc_now(),
        "gpu_memory_before": gpu_memory_snapshot(),
    }
    candidates: list[dict[str, Any]] = []
    input_errors: list[str] = []
    for index, raw_candidate in enumerate(video["sam3_candidates"][:SAM3_MAX_CANDIDATES_PER_VIDEO]):
        candidate, error = validate_candidate(raw_candidate, index)
        if error:
            input_errors.append(error)
        elif candidate:
            candidates.append(candidate)
    result["input_candidate_count"] = len(video["sam3_candidates"])
    result["processed_candidate_count"] = len(candidates)
    if input_errors:
        result["input_errors"] = input_errors
    if not candidates:
        result["status"] = "no_valid_candidate"
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        result["gpu_memory_after"] = gpu_memory_snapshot()
        return result

    session_id: str | None = None
    try:
        session_id = runner.start_session(video_path)
        for candidate in candidates:
            try:
                candidate_result = track_candidate(runner, session_id, video_id, candidate, source_frames)
            except Exception as exc:
                candidate_result = {
                    "candidate_id": candidate["candidate_id"],
                    "sam_prompt": candidate["sam_prompt"],
                    "display_phrase": candidate["display_phrase"],
                    "status": "failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc(limit=5),
                }
            result["candidate_results"].append(candidate_result)
        statuses = [item["status"] for item in result["candidate_results"]]
        result["status"] = "success" if "success" in statuses else "no_track" if "no_track" in statuses else "failure"
    except Exception as exc:
        result.update({
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc(limit=8),
        })
    finally:
        if session_id:
            try:
                result["close_session"] = runner.close_session(session_id)
            except Exception as exc:
                result["close_session_error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        result["gpu_memory_after"] = gpu_memory_snapshot()
    return result


def flatten_quality_tracks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for video_result in results:
        for candidate_result in video_result.get("candidate_results", []):
            if not isinstance(candidate_result, dict):
                continue
            for track in candidate_result.get("tracks", []):
                if isinstance(track, dict) and track.get("quality_status") == "pass":
                    selected.append({
                        "video_id": video_result["video_id"],
                        "relative_path": video_result.get("relative_path"),
                        "video_path": video_result["video_path"],
                        "candidate_id": candidate_result["candidate_id"],
                        "candidate_class": candidate_result.get("candidate_class"),
                        "canonical_concept": candidate_result.get("canonical_concept"),
                        "display_phrase": candidate_result.get("display_phrase"),
                        "sam_prompt": candidate_result.get("sam_prompt"),
                        **track,
                    })
    return selected


def write_outputs(dataset: dict[str, Any], input_videos: list[dict[str, Any]], results: list[dict[str, Any]], started: float) -> None:
    quality_tracks = flatten_quality_tracks(results)
    atomic_write_json(SAM3_TRACKS_ALL_PATH, {
        "schema_version": SAM3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "input_file": str(QWEN_SAM3_CANDIDATES_PATH),
        "videos": results,
    })
    atomic_write_json(SAM3_QUALITY_TRACKS_PATH, {
        "schema_version": SAM3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "tracks": quality_tracks,
    })
    atomic_write_json(SAM3_FAILURES_PATH, {
        "schema_version": SAM3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "failures": [result for result in results if result.get("status") == "failure"],
    })
    status_counts: dict[str, int] = defaultdict(int)
    all_track_count = 0
    for result in results:
        status_counts[result["status"]] += 1
        for candidate_result in result.get("candidate_results", []):
            if isinstance(candidate_result, dict):
                all_track_count += len(candidate_result.get("tracks", []))
    atomic_write_json(SAM3_RUN_SUMMARY_PATH, {
        "schema_version": SAM3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "input_file": str(QWEN_SAM3_CANDIDATES_PATH),
        "input_schema_version": dataset.get("schema_version"),
        "input_video_records": len(input_videos),
        "processed_videos": len(results),
        "video_status_totals": dict(status_counts),
        "track_totals": {
            "all_tracks": all_track_count,
            "quality_pass_tracks": len(quality_tracks),
            "mask_tubes_saved": SAM3_SAVE_MASK_TUBES,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "throughput_videos_per_min": round(len(results) / max(time.perf_counter() - started, 1e-9) * 60.0, 3),
        "sam3_source_root": str(SAM3_SOURCE_ROOT),
        "sam3_checkpoint_path": str(SAM3_CHECKPOINT_PATH),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3 tracking from Qwen candidate JSON.")
    parser.add_argument("--qwen-candidates", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--mask-root", type=Path, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--all-videos", action="store_true")
    parser.add_argument("--max-candidates-per-video", type=int, default=None)
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    global QWEN_SAM3_CANDIDATES_PATH, SAM3_TRACKS_ALL_PATH, SAM3_QUALITY_TRACKS_PATH
    global SAM3_FAILURES_PATH, SAM3_RUN_SUMMARY_PATH, SAM3_TRACK_MASK_ROOT
    global SAM3_MAX_VIDEOS, SAM3_MAX_CANDIDATES_PER_VIDEO

    if args.qwen_candidates is not None:
        QWEN_SAM3_CANDIDATES_PATH = Path(args.qwen_candidates)
    if args.out_root is not None:
        out_root = Path(args.out_root)
        SAM3_TRACKS_ALL_PATH = out_root / "sam3_tracks_all.json"
        SAM3_QUALITY_TRACKS_PATH = out_root / "sam3_quality_tracks.json"
        SAM3_FAILURES_PATH = out_root / "sam3_failures.json"
        SAM3_RUN_SUMMARY_PATH = out_root / "sam3_run_summary.json"
    if args.mask_root is not None:
        SAM3_TRACK_MASK_ROOT = Path(args.mask_root)
    if args.all_videos:
        SAM3_MAX_VIDEOS = None
    elif args.max_videos is not None:
        if args.max_videos < 0:
            raise ValueError("--max-videos must be non-negative")
        SAM3_MAX_VIDEOS = int(args.max_videos)
    if args.max_candidates_per_video is not None:
        if args.max_candidates_per_video <= 0:
            raise ValueError("--max-candidates-per-video must be positive")
        SAM3_MAX_CANDIDATES_PER_VIDEO = int(args.max_candidates_per_video)


def main() -> None:
    configure_runtime(parse_args())
    started = time.perf_counter()
    dataset, videos = load_input()
    if SAM3_MAX_VIDEOS is not None:
        videos = videos[:SAM3_MAX_VIDEOS]
    if not videos:
        raise RuntimeError("No valid Qwen v4 candidates available")
    print(f"[input] selected {len(videos)} video(s) from {QWEN_SAM3_CANDIDATES_PATH}")
    runner = Sam3Runner()
    results: list[dict[str, Any]] = []
    try:
        for index, video in enumerate(videos, start=1):
            print(f"[video {index}/{len(videos)}] {video['video_id']}")
            result = process_video(runner, video)
            results.append(result)
            write_outputs(dataset, videos, results, started)
            print(f"[video {index}/{len(videos)}] status={result['status']} elapsed={result['elapsed_seconds']}s")
    finally:
        runner.shutdown()
    print(f"[done] track bank: {SAM3_TRACKS_ALL_PATH}")
    print(f"[done] quality tracks: {SAM3_QUALITY_TRACKS_PATH}")


if __name__ == "__main__":
    main()
