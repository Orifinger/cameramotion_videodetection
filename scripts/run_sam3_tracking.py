#!/usr/bin/env python3
"""Build a SAM3 multi-instance track bank from Qwen v4 concept candidates.

Run from the project root after Qwen v4 has produced its candidate view:
    CUDA_VISIBLE_DEVICES=<physical_gpu_id> python scripts/run_sam3_tracking.py

One SAM3 predictor stays resident. For every video and every Qwen candidate,
SAM3 may return multiple matching object identities; all of them are retained.
This stage does not choose a final focus region or an editing instruction.
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
    SAM3_MAX_CANDIDATES_PER_VIDEO,
    SAM3_MAX_MEDIAN_AREA_RATIO,
    SAM3_MAX_VIDEOS,
    SAM3_MIN_LONGEST_VISIBLE_RUN,
    SAM3_MIN_MEDIAN_AREA_RATIO,
    SAM3_MIN_VISIBLE_FRAME_RATIO,
    SAM3_MAX_BORDER_TOUCH_RATIO,
    SAM3_OUTPUT_PROB_THRESH,
    SAM3_PROGRESS_EVERY,
    SAM3_PROMPT_FRAME_INDEX,
    SAM3_PROPAGATION_DIRECTION,
    SAM3_QUALITY_TRACKS_PATH,
    SAM3_RESULT_ROOT,
    SAM3_RUN_SUMMARY_PATH,
    SAM3_SAVE_EVERY,
    SAM3_SAVE_MASK_TUBES,
    SAM3_SCHEMA_VERSION,
    SAM3_SOURCE_ROOT,
    SAM3_TRACK_MASK_ROOT,
    SAM3_TRACKS_ALL_PATH,
)

if str(SAM3_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_SOURCE_ROOT))
from sam3.model_builder import build_sam3_video_predictor


REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id",
    "region_family",
    "candidate_class",
    "target_scope",
    "canonical_concept",
    "display_phrase",
    "sam_prompt",
    "instance_count_hint",
    "visual_disambiguators",
    "screen_region",
    "temporal_presence",
}
ALLOWED_FAMILIES = {"physical_instance", "editable_surface"}
ALLOWED_SCOPES = {"whole_instance", "whole_surface"}
ALLOWED_CLASSES = {
    "human",
    "animal",
    "vehicle",
    "handheld_object",
    "bounded_object",
    "display_screen",
    "sign_or_poster",
    "paper_book_map",
    "framed_art",
    "apparel_panel",
    "vehicle_panel",
    "package_front",
}
ALLOWED_INSTANCE_HINTS = {"unique_in_video", "possibly_multiple", "unknown"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp_path.replace(path)


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"Qwen v4 SAM3 candidate file is missing: {path}. "
            "Run scripts/run_qwen_object_proposals.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def as_numpy(value: Any) -> np.ndarray:
    """Safely normalize SAM3 CPU/GPU tensor-like outputs to NumPy."""
    if value is None:
        return np.asarray([])
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").numpy()
    return np.asarray(value)


def safe_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        value = value.detach().to("cpu")
    try:
        array = np.asarray(value)
        if array.size != 1:
            return None
        result = float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def longest_consecutive_run(frame_indices: list[int]) -> int:
    if not frame_indices:
        return 0
    ordered = sorted(set(frame_indices))
    longest = current = 1
    for previous, current_index in zip(ordered, ordered[1:]):
        if current_index == previous + 1:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
    return longest


def border_touched(mask: np.ndarray) -> bool:
    if mask.ndim != 2 or not mask.any():
        return False
    return bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def gpu_memory_snapshot() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    return {
        "device": int(device),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def validate_candidate(raw: Any, index: int) -> tuple[dict[str, Any] | None, str | None]:
    """Strictly accept only the structured Qwen v4 candidate contract."""
    if not isinstance(raw, dict):
        return None, f"candidate[{index}] must be an object, got {type(raw).__name__}"
    missing = sorted(REQUIRED_CANDIDATE_FIELDS - set(raw))
    if missing:
        return None, f"candidate[{index}] missing required fields: {missing}"

    candidate = dict(raw)
    family = candidate.get("region_family")
    candidate_class = candidate.get("candidate_class")
    scope = candidate.get("target_scope")
    prompt = candidate.get("sam_prompt")
    if family not in ALLOWED_FAMILIES:
        return None, f"candidate[{index}] invalid region_family={family!r}"
    if candidate_class not in ALLOWED_CLASSES:
        return None, f"candidate[{index}] invalid candidate_class={candidate_class!r}"
    if scope not in ALLOWED_SCOPES:
        return None, f"candidate[{index}] invalid target_scope={scope!r}"
    expected_scope = "whole_instance" if family == "physical_instance" else "whole_surface"
    if scope != expected_scope:
        return None, (
            f"candidate[{index}] family/scope mismatch: {family!r} requires "
            f"{expected_scope!r}, got {scope!r}"
        )
    if not isinstance(prompt, str) or not prompt.strip():
        return None, f"candidate[{index}] missing non-empty sam_prompt"
    if candidate.get("instance_count_hint") not in ALLOWED_INSTANCE_HINTS:
        return None, f"candidate[{index}] invalid instance_count_hint"
    if not isinstance(candidate.get("visual_disambiguators"), list):
        return None, f"candidate[{index}] visual_disambiguators must be a list"

    candidate["candidate_id"] = str(candidate["candidate_id"])
    candidate["sam_prompt"] = prompt.strip()
    candidate["display_phrase"] = str(candidate["display_phrase"]).strip()
    candidate["canonical_concept"] = str(candidate["canonical_concept"]).strip()
    return candidate, None


def extract_video_records(dataset: Any) -> list[dict[str, Any]]:
    """Validate the exact v4 unified Qwen candidate view."""
    if not isinstance(dataset, dict):
        raise ValueError(
            "Invalid Qwen input: top-level value must be a dict with schema_version and videos."
        )
    schema_version = dataset.get("schema_version")
    if schema_version != QWEN_INPUT_SCHEMA_VERSION:
        raise ValueError(
            f"Invalid Qwen input schema_version={schema_version!r}; "
            f"expected {QWEN_INPUT_SCHEMA_VERSION!r}."
        )
    videos = dataset.get("videos")
    if not isinstance(videos, list):
        raise ValueError("Invalid Qwen input: top-level 'videos' must be a list.")

    valid: list[dict[str, Any]] = []
    for index, record in enumerate(videos):
        if not isinstance(record, dict):
            print(f"[input] skip non-dict video record index={index}")
            continue
        if not record.get("video_id") or not record.get("video_path"):
            print(f"[input] skip incomplete record index={index}")
            continue
        candidates = record.get("sam3_candidates")
        if not isinstance(candidates, list):
            print(f"[input] skip record with non-list sam3_candidates: {record['video_id']}")
            continue
        valid.append(record)
    return valid


class Sam3Runner:
    """One resident official SAM3 predictor with one active video session."""

    def __init__(self) -> None:
        if not SAM3_SOURCE_ROOT.is_dir():
            raise FileNotFoundError(f"SAM3_SOURCE_ROOT does not exist: {SAM3_SOURCE_ROOT}")
        if not SAM3_CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(
                f"SAM3_CHECKPOINT_PATH does not exist: {SAM3_CHECKPOINT_PATH}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SAM3 tracking.")
        self.predictor = build_sam3_video_predictor(
            checkpoint_path=str(SAM3_CHECKPOINT_PATH),
            gpus_to_use=[0],
        )
        print(f"[sam3] loaded on local CUDA device 0; {gpu_memory_snapshot()}")

    def start_session(self, video_path: str) -> str:
        response = self.predictor.handle_request(
            {"type": "start_session", "resource_path": str(video_path)}
        )
        return str(response["session_id"])

    def reset_session(self, session_id: str) -> None:
        self.predictor.handle_request({"type": "reset_session", "session_id": session_id})

    def add_text_prompt(self, session_id: str, text: str) -> dict[str, Any]:
        return self.predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": SAM3_PROMPT_FRAME_INDEX,
                "text": text,
                "output_prob_thresh": SAM3_OUTPUT_PROB_THRESH,
            }
        )

    def propagate(self, session_id: str):
        yield from self.predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": SAM3_PROPAGATION_DIRECTION,
                "output_prob_thresh": SAM3_OUTPUT_PROB_THRESH,
            }
        )

    def close_session(self, session_id: str) -> dict[str, Any]:
        return self.predictor.handle_request(
            {
                "type": "close_session",
                "session_id": session_id,
                "run_gc_collect": SAM3_CLOSE_SESSION_RUN_GC,
                "clear_cache_threshold": SAM3_CLEAR_CACHE_THRESHOLD,
            }
        )

    def shutdown(self) -> None:
        self.predictor.shutdown()


def save_sparse_mask_tube(
    video_id: str,
    candidate_id: str,
    sam_object_id: int,
    frame_indices: list[int],
    masks: list[np.ndarray],
) -> str | None:
    if not SAM3_SAVE_MASK_TUBES or not masks:
        return None
    target_dir = SAM3_TRACK_MASK_ROOT / video_id
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{candidate_id}__obj_{sam_object_id}.npz"
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "wb") as handle:
        np.savez_compressed(
            handle,
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
            masks=np.stack(masks, axis=0).astype(np.uint8, copy=False),
        )
    temp_path.replace(path)
    return str(path)


def track_quality(track: dict[str, Any]) -> tuple[str, list[str], float]:
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

    area = track["median_area_ratio"]
    area_quality = 1.0 if SAM3_MIN_MEDIAN_AREA_RATIO <= area <= SAM3_MAX_MEDIAN_AREA_RATIO else 0.0
    stability = min(track["longest_visible_run"] / max(track["propagation_frame_count"], 1), 1.0)
    interior = 1.0 - track["border_touch_ratio"]
    score = (
        0.40 * track["visible_frame_ratio"]
        + 0.25 * stability
        + 0.15 * area_quality
        + 0.10 * interior
        + 0.10 * (track["mean_detection_score"] or 0.0)
    )
    return ("pass" if not reasons else "fail"), reasons, round(score, 6)


def collect_candidate_tracks(
    runner: Sam3Runner,
    session_id: str,
    video_id: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Run one semantic concept and retain every returned SAM3 object identity."""
    runner.reset_session(session_id)
    add_response = runner.add_text_prompt(session_id, candidate["sam_prompt"])

    objects: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "frames": [],
            "masks": [],
            "area_ratios": [],
            "scores": [],
            "boxes_xywh": [],
            "border_touches": [],
        }
    )
    propagation_frame_count = 0
    frame_shape: tuple[int, int] | None = None
    output_keys_seen: set[str] = set()

    for item in runner.propagate(session_id):
        if not isinstance(item, dict) or "frame_index" not in item:
            continue
        frame_index = int(item["frame_index"])
        outputs = item.get("outputs")
        if not isinstance(outputs, dict):
            continue
        output_keys_seen.update(outputs.keys())
        propagation_frame_count += 1

        object_ids = as_numpy(outputs.get("out_obj_ids", []))
        masks = as_numpy(outputs.get("out_binary_masks", []))
        boxes = as_numpy(outputs.get("out_boxes_xywh", []))
        scores = as_numpy(outputs.get("out_probs", []))

        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.ndim == 3 and masks.shape[0] > 0:
            frame_shape = (int(masks.shape[1]), int(masks.shape[2]))
        if object_ids.ndim == 0:
            object_ids = object_ids.reshape(1)

        for item_index, raw_object_id in enumerate(object_ids.reshape(-1).tolist()):
            if item_index >= len(masks):
                continue
            mask = np.asarray(masks[item_index], dtype=bool)
            if mask.ndim != 2 or not mask.any():
                continue
            area_ratio = float(mask.mean())
            box = boxes[item_index].tolist() if item_index < len(boxes) else None
            score = safe_float(scores[item_index]) if item_index < len(scores) else None
            state = objects[int(raw_object_id)]
            state["frames"].append(frame_index)
            state["masks"].append(mask)
            state["area_ratios"].append(area_ratio)
            state["border_touches"].append(border_touched(mask))
            state["boxes_xywh"].append({"frame_index": frame_index, "bbox_xywh_norm": box})
            if score is not None:
                state["scores"].append(score)

    tracks: list[dict[str, Any]] = []
    for sam_object_id, state in objects.items():
        frames = state["frames"]
        area_ratios = state["area_ratios"]
        median_area_ratio = float(np.median(area_ratios)) if area_ratios else 0.0
        mean_area_ratio = float(np.mean(area_ratios)) if area_ratios else 0.0
        std_area_ratio = float(np.std(area_ratios)) if area_ratios else 0.0
        track: dict[str, Any] = {
            "track_id": f"{video_id}__{candidate['candidate_id']}__obj_{sam_object_id}",
            "sam_object_id": int(sam_object_id),
            "first_frame_index": min(frames),
            "last_frame_index": max(frames),
            "visible_frame_count": len(frames),
            "propagation_frame_count": propagation_frame_count,
            "visible_frame_ratio": round(len(frames) / max(propagation_frame_count, 1), 6),
            "longest_visible_run": longest_consecutive_run(frames),
            "median_area_ratio": round(median_area_ratio, 8),
            "mean_area_ratio": round(mean_area_ratio, 8),
            "std_area_ratio": round(std_area_ratio, 8),
            "mean_detection_score": round(float(np.mean(state["scores"])), 6) if state["scores"] else None,
            "border_touch_ratio": round(float(np.mean(state["border_touches"])) if state["border_touches"] else 0.0, 6),
            "bbox_tube_xywh_norm": state["boxes_xywh"],
            "mask_tube_path": save_sparse_mask_tube(
                video_id,
                candidate["candidate_id"],
                int(sam_object_id),
                frames,
                state["masks"],
            ),
        }
        status, reasons, score = track_quality(track)
        track["quality_status"] = status
        track["quality_reasons"] = reasons
        track["track_quality_score"] = score
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
        "add_prompt_output_keys": sorted(add_response.get("outputs", {}).keys()) if isinstance(add_response.get("outputs"), dict) else None,
        "propagation_frame_count": propagation_frame_count,
        "mask_frame_height_width": list(frame_shape) if frame_shape else None,
        "propagation_output_keys": sorted(output_keys_seen),
        "tracks": tracks,
        "status": "success" if tracks else "no_track",
    }


def process_video(runner: Sam3Runner, video: dict[str, Any]) -> dict[str, Any]:
    video_id = str(video["video_id"])
    video_path = str(video["video_path"])
    started = time.perf_counter()
    result: dict[str, Any] = {
        "video_id": video_id,
        "relative_path": video.get("relative_path"),
        "video_path": video_path,
        "status": "failure",
        "candidate_results": [],
        "created_at_utc": utc_now(),
        "gpu_memory_before": gpu_memory_snapshot(),
    }

    raw_candidates = video.get("sam3_candidates")
    if not isinstance(raw_candidates, list):
        result["error"] = "sam3_candidates is not a list"
        return result

    candidates: list[dict[str, Any]] = []
    input_errors: list[str] = []
    for index, raw_candidate in enumerate(raw_candidates[:SAM3_MAX_CANDIDATES_PER_VIDEO]):
        candidate, error = validate_candidate(raw_candidate, index)
        if error is not None:
            input_errors.append(error)
        elif candidate is not None:
            candidates.append(candidate)

    result["input_candidate_count"] = len(raw_candidates)
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
                candidate_result = collect_candidate_tracks(runner, session_id, video_id, candidate)
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

        candidate_statuses = [item.get("status") for item in result["candidate_results"]]
        if "success" in candidate_statuses:
            result["status"] = "success"
        elif "no_track" in candidate_statuses:
            result["status"] = "no_track"
        else:
            result["status"] = "failure"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback_tail"] = traceback.format_exc(limit=8)
    finally:
        if session_id is not None:
            try:
                result["close_session"] = runner.close_session(session_id)
            except Exception as exc:
                result["close_session_error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        result["gpu_memory_after"] = gpu_memory_snapshot()
    return result


def flatten_quality_tracks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    for video_result in results:
        for candidate_result in video_result.get("candidate_results", []):
            if not isinstance(candidate_result, dict):
                continue
            for track in candidate_result.get("tracks", []):
                if not isinstance(track, dict) or track.get("quality_status") != "pass":
                    continue
                tracks.append({
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
    return tracks


def build_summary(dataset: dict[str, Any], videos: list[dict[str, Any]], results: list[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    statuses: dict[str, int] = defaultdict(int)
    candidate_statuses: dict[str, int] = defaultdict(int)
    all_tracks = 0
    quality_tracks = 0
    for result in results:
        statuses[result["status"]] += 1
        for candidate_result in result.get("candidate_results", []):
            if not isinstance(candidate_result, dict):
                continue
            candidate_statuses[candidate_result.get("status", "unknown")] += 1
            candidate_tracks = candidate_result.get("tracks", [])
            if isinstance(candidate_tracks, list):
                all_tracks += len(candidate_tracks)
                quality_tracks += sum(track.get("quality_status") == "pass" for track in candidate_tracks if isinstance(track, dict))
    return {
        "schema_version": SAM3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "input_file": str(QWEN_SAM3_CANDIDATES_PATH),
        "input_schema_version": dataset.get("schema_version"),
        "input_video_records": len(videos),
        "processed_videos": len(results),
        "video_status_totals": dict(statuses),
        "candidate_status_totals": dict(candidate_statuses),
        "track_totals": {
            "all_tracks": all_tracks,
            "quality_pass_tracks": quality_tracks,
            "mask_tubes_saved": SAM3_SAVE_MASK_TUBES,
        },
        "elapsed_seconds": round(elapsed_seconds, 3),
        "throughput_videos_per_min": round(len(results) / max(elapsed_seconds, 1e-9) * 60.0, 3),
        "sam3_source_root": str(SAM3_SOURCE_ROOT),
        "sam3_checkpoint_path": str(SAM3_CHECKPOINT_PATH),
    }


def write_outputs(dataset: dict[str, Any], videos: list[dict[str, Any]], results: list[dict[str, Any]], started: float) -> None:
    aggregate = {
        "schema_version": SAM3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "input_file": str(QWEN_SAM3_CANDIDATES_PATH),
        "videos": results,
    }
    failures = [result for result in results if result.get("status") == "failure"]
    quality_tracks = flatten_quality_tracks(results)
    summary = build_summary(dataset, videos, results, time.perf_counter() - started)
    atomic_write_json(SAM3_TRACKS_ALL_PATH, aggregate)
    atomic_write_json(SAM3_QUALITY_TRACKS_PATH, {
        "schema_version": SAM3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "tracks": quality_tracks,
    })
    atomic_write_json(SAM3_FAILURES_PATH, {
        "schema_version": SAM3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "failures": failures,
    })
    atomic_write_json(SAM3_RUN_SUMMARY_PATH, summary)


def main() -> None:
    started = time.perf_counter()
    dataset = load_json(QWEN_SAM3_CANDIDATES_PATH)
    videos = extract_video_records(dataset)
    if SAM3_MAX_VIDEOS is not None:
        videos = videos[:SAM3_MAX_VIDEOS]
    if not videos:
        raise RuntimeError("No valid videos were found in Qwen v4 sam3_candidates.json.")

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
    print(f"[done] summary: {SAM3_RUN_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
