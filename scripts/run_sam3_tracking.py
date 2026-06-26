"""Run the first real SAM3 text-guided video-tracking smoke test.

Run from the project root, binding one physical GPU before Python imports torch:
    CUDA_VISIBLE_DEVICES=<physical_gpu_id> python scripts/run_sam3_tracking.py

This version deliberately has one visible GPU and one resident SAM3 predictor.
It verifies the real SAM3 request/response contract and produces compact track
metadata only. Dense mask storage and multi-worker scheduling are the next
stage after this smoke result is visually checked.
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

from configs.sam3_tracking_config import (
    QWEN_SAM3_CANDIDATES_PATH,
    SAM3_CHECKPOINT_PATH,
    SAM3_CLEAR_CACHE_THRESHOLD,
    SAM3_CLOSE_SESSION_RUN_GC,
    SAM3_FOCUS_REGIONS_PATH,
    SAM3_FAILURES_PATH,
    SAM3_MAX_CANDIDATES_PER_VIDEO,
    SAM3_MAX_VIDEOS,
    SAM3_OUTPUT_PROB_THRESH,
    SAM3_PROMPT_FRAME_INDEX,
    SAM3_PROPAGATION_DIRECTION,
    SAM3_RESULT_ROOT,
    SAM3_RUN_SUMMARY_PATH,
    SAM3_SCHEMA_VERSION,
    SAM3_SOURCE_ROOT,
    SAM3_TRACKS_ALL_PATH,
)

# Import the official source without editing it. This must happen before SAM3
# imports so the local cloned source is used, not a possibly unrelated package.
sys.path.insert(0, str(SAM3_SOURCE_ROOT))
from sam3.model_builder import build_sam3_video_predictor


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"Qwen SAM3 candidate file is missing: {path}. "
            "Run scripts/run_qwen_object_proposals.py first."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_video_records(dataset: Any) -> list[dict[str, Any]]:
    """Read the unified Qwen view, whose top-level schema is { ..., videos: [...] }."""
    if isinstance(dataset, list):
        # Temporary compatibility for an early hand-written / legacy export.
        videos = dataset
    elif isinstance(dataset, dict):
        videos = dataset.get("videos")
    else:
        videos = None

    if not isinstance(videos, list):
        keys = sorted(dataset.keys()) if isinstance(dataset, dict) else None
        raise ValueError(
            "Invalid qwen_sam3_candidates.json shape. Expected a top-level "
            f"'videos' list, got type={type(dataset).__name__}, keys={keys}."
        )

    valid: list[dict[str, Any]] = []
    for index, record in enumerate(videos):
        if not isinstance(record, dict):
            print(f"[input] skip non-dict video record at index={index}")
            continue
        if not record.get("video_id") or not record.get("video_path"):
            print(f"[input] skip incomplete record index={index}: {record!r}")
            continue
        valid.append(record)
    return valid


def normalize_candidate(raw: Any, index: int) -> tuple[dict[str, Any] | None, str | None]:
    """Return a structured candidate and an optional compatibility warning.

    v3 candidates are dictionaries. A legacy bare string is accepted only for
    smoke-test continuity; it should be replaced by rerunning Qwen v3 before a
    dataset-scale run.
    """
    if isinstance(raw, str):
        phrase = raw.strip()
        if not phrase:
            return None, "empty legacy string candidate"
        return {
            "candidate_id": f"legacy_{index:02d}",
            "sam_prompt": phrase,
            "display_phrase": phrase,
            "region_family": "legacy_unknown",
            "editable_priority": None,
            "legacy_candidate": True,
        }, "legacy bare-string candidate; rerun Qwen v3 before full processing"

    if not isinstance(raw, dict):
        return None, f"candidate has unsupported type={type(raw).__name__}"

    prompt = raw.get("sam_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, "missing non-empty sam_prompt"

    candidate = dict(raw)
    candidate["candidate_id"] = str(candidate.get("candidate_id") or f"candidate_{index:02d}")
    candidate["sam_prompt"] = prompt.strip()
    candidate["display_phrase"] = str(candidate.get("display_phrase") or candidate["sam_prompt"])
    return candidate, None


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


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


class Sam3Runner:
    """One resident official SAM3 predictor, with one active video session."""

    def __init__(self) -> None:
        if not SAM3_SOURCE_ROOT.is_dir():
            raise FileNotFoundError(f"SAM3_SOURCE_ROOT does not exist: {SAM3_SOURCE_ROOT}")
        if not SAM3_CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"SAM3_CHECKPOINT_PATH does not exist: {SAM3_CHECKPOINT_PATH}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this SAM3 smoke-test script.")

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
        request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": SAM3_PROPAGATION_DIRECTION,
            "output_prob_thresh": SAM3_OUTPUT_PROB_THRESH,
        }
        yield from self.predictor.handle_stream_request(request)

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
        # Important for the later multi-GPU implementation. Safe on one GPU too.
        self.predictor.shutdown()


def collect_candidate_tracks(
    runner: Sam3Runner,
    session_id: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Track exactly one text concept in a reset video session.

    Resetting before each candidate prevents a later text prompt from inheriting
    the detector/tracker state of a previous semantic concept.
    """
    runner.reset_session(session_id)
    add_response = runner.add_text_prompt(session_id, candidate["sam_prompt"])

    objects: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"frames": [], "areas": [], "area_ratios": [], "scores": [], "boxes_xywh": []}
    )
    frame_count_seen = 0
    frame_shape: tuple[int, int] | None = None
    output_keys_seen: set[str] = set()

    for item in runner.propagate(session_id):
        frame_index = int(item["frame_index"])
        outputs = item.get("outputs")
        if not isinstance(outputs, dict):
            continue
        output_keys_seen.update(outputs.keys())
        frame_count_seen += 1

        object_ids = np.asarray(outputs.get("out_obj_ids", []))
        masks = np.asarray(outputs.get("out_binary_masks", []))
        boxes = np.asarray(outputs.get("out_boxes_xywh", []))
        scores = np.asarray(outputs.get("out_probs", []))

        if masks.ndim == 3 and masks.shape[0] > 0:
            frame_shape = (int(masks.shape[1]), int(masks.shape[2]))

        for item_index, raw_object_id in enumerate(object_ids.tolist()):
            if item_index >= len(masks):
                continue
            mask = np.asarray(masks[item_index], dtype=bool)
            if mask.ndim != 2 or not mask.any():
                continue

            object_id = int(raw_object_id)
            area = int(mask.sum())
            total_pixels = int(mask.shape[0] * mask.shape[1])
            area_ratio = area / max(total_pixels, 1)
            box = boxes[item_index].tolist() if item_index < len(boxes) else None
            score = safe_float(scores[item_index]) if item_index < len(scores) else None

            state = objects[object_id]
            state["frames"].append(frame_index)
            state["areas"].append(area)
            state["area_ratios"].append(area_ratio)
            if score is not None:
                state["scores"].append(score)
            state["boxes_xywh"].append(
                {"frame_index": frame_index, "bbox_xywh_norm": box}
            )

    track_summaries: list[dict[str, Any]] = []
    for object_id, state in objects.items():
        frames = state["frames"]
        area_ratios = state["area_ratios"]
        longest_run = longest_consecutive_run(frames)
        visible_frame_ratio = len(frames) / max(frame_count_seen, 1)
        median_area_ratio = float(np.median(area_ratios)) if area_ratios else 0.0
        mean_area_ratio = float(np.mean(area_ratios)) if area_ratios else 0.0
        mean_score = float(np.mean(state["scores"])) if state["scores"] else None
        track_summaries.append(
            {
                "sam_object_id": object_id,
                "first_frame_index": min(frames),
                "last_frame_index": max(frames),
                "visible_frame_count": len(frames),
                "visible_frame_ratio": round(visible_frame_ratio, 6),
                "longest_visible_run": longest_run,
                "median_area_ratio": round(median_area_ratio, 8),
                "mean_area_ratio": round(mean_area_ratio, 8),
                "mean_detection_score": None if mean_score is None else round(mean_score, 6),
                # This compact tube is intentional for the smoke test. Dense masks
                # are not yet persisted until the response contract is inspected.
                "bbox_tube_xywh_norm": state["boxes_xywh"],
            }
        )

    def candidate_track_score(track: dict[str, Any]) -> float:
        # A deliberately simple smoke-test score, not the final focus-region ranker.
        stability = min(track["longest_visible_run"] / max(frame_count_seen, 1), 1.0)
        area = track["median_area_ratio"]
        area_quality = 1.0 if 0.002 <= area <= 0.90 else 0.0
        det_score = track["mean_detection_score"] or 0.0
        return 0.55 * track["visible_frame_ratio"] + 0.25 * stability + 0.12 * area_quality + 0.08 * det_score

    for track in track_summaries:
        track["candidate_track_score"] = round(candidate_track_score(track), 6)
    track_summaries.sort(key=lambda item: item["candidate_track_score"], reverse=True)

    return {
        "candidate_id": candidate["candidate_id"],
        "sam_prompt": candidate["sam_prompt"],
        "display_phrase": candidate["display_phrase"],
        "region_family": candidate.get("region_family"),
        "editable_priority": candidate.get("editable_priority"),
        "add_prompt_output_keys": sorted(add_response.get("outputs", {}).keys())
        if isinstance(add_response.get("outputs"), dict)
        else None,
        "propagation_frame_count": frame_count_seen,
        "mask_frame_height_width": list(frame_shape) if frame_shape else None,
        "propagation_output_keys": sorted(output_keys_seen),
        "tracks": track_summaries,
        "best_track": track_summaries[0] if track_summaries else None,
        "status": "success" if track_summaries else "no_track",
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
        "focus_track": None,
        "created_at_utc": utc_now(),
        "gpu_memory_before": gpu_memory_snapshot(),
    }

    raw_candidates = video.get("sam3_candidates", [])
    if not isinstance(raw_candidates, list):
        result["error"] = "sam3_candidates is not a list"
        return result

    normalized: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, raw_candidate in enumerate(raw_candidates[:SAM3_MAX_CANDIDATES_PER_VIDEO]):
        candidate, warning = normalize_candidate(raw_candidate, index)
        if warning:
            warnings.append(f"candidate[{index}]: {warning}")
        if candidate is not None:
            normalized.append(candidate)

    result["input_candidate_count"] = len(raw_candidates)
    result["processed_candidate_count"] = len(normalized)
    if warnings:
        result["input_warnings"] = warnings
    if not normalized:
        result["status"] = "no_valid_candidate"
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        result["gpu_memory_after"] = gpu_memory_snapshot()
        return result

    session_id: str | None = None
    try:
        session_id = runner.start_session(video_path)
        for candidate in normalized:
            try:
                candidate_result = collect_candidate_tracks(runner, session_id, candidate)
            except Exception as exc:  # preserve other candidates when one prompt fails
                candidate_result = {
                    "candidate_id": candidate["candidate_id"],
                    "sam_prompt": candidate["sam_prompt"],
                    "display_phrase": candidate["display_phrase"],
                    "status": "failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc(limit=4),
                }
            result["candidate_results"].append(candidate_result)

        viable = [
            item for item in result["candidate_results"]
            if item.get("status") == "success" and isinstance(item.get("best_track"), dict)
        ]
        if viable:
            def focus_score(item: dict[str, Any]) -> float:
                best = item["best_track"]
                priority = safe_float(item.get("editable_priority")) or 0.0
                return float(best["candidate_track_score"]) + 0.01 * priority

            result["focus_track"] = max(viable, key=focus_score)
            result["status"] = "success"
        elif any(item.get("status") == "no_track" for item in result["candidate_results"]):
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


def build_summary(dataset: Any, videos: list[dict[str, Any]], results: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    statuses = defaultdict(int)
    for result in results:
        statuses[result["status"]] += 1
    return {
        "schema_version": SAM3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "input_file": str(QWEN_SAM3_CANDIDATES_PATH),
        "input_schema_version": dataset.get("schema_version") if isinstance(dataset, dict) else None,
        "input_video_records": len(videos),
        "processed_videos": len(results),
        "status_totals": dict(statuses),
        "elapsed_seconds": round(elapsed, 3),
        "throughput_videos_per_min": round(len(results) / max(elapsed, 1e-9) * 60.0, 3),
        "sam3_source_root": str(SAM3_SOURCE_ROOT),
        "sam3_checkpoint_path": str(SAM3_CHECKPOINT_PATH),
    }


def main() -> None:
    started = time.perf_counter()
    dataset = load_json(QWEN_SAM3_CANDIDATES_PATH)
    videos = extract_video_records(dataset)
    if SAM3_MAX_VIDEOS is not None:
        videos = videos[:SAM3_MAX_VIDEOS]
    if not videos:
        raise RuntimeError("No valid videos were found in qwen_sam3_candidates.json.")

    print(f"[input] selected {len(videos)} video(s) from {QWEN_SAM3_CANDIDATES_PATH}")
    runner = Sam3Runner()
    results: list[dict[str, Any]] = []
    try:
        for index, video in enumerate(videos, start=1):
            print(f"[video {index}/{len(videos)}] {video['video_id']}")
            result = process_video(runner, video)
            results.append(result)
            print(
                f"[video {index}/{len(videos)}] status={result['status']} "
                f"elapsed={result['elapsed_seconds']}s"
            )
    finally:
        runner.shutdown()

    aggregate = {
        "schema_version": SAM3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "input_file": str(QWEN_SAM3_CANDIDATES_PATH),
        "videos": results,
    }
    failures = [result for result in results if result["status"] == "failure"]
    focus_regions = [
        {
            "video_id": result["video_id"],
            "relative_path": result.get("relative_path"),
            "video_path": result["video_path"],
            "focus_track": result["focus_track"],
        }
        for result in results
        if result["status"] == "success" and result.get("focus_track") is not None
    ]
    summary = build_summary(dataset, videos, results, time.perf_counter() - started)

    atomic_write_json(SAM3_TRACKS_ALL_PATH, aggregate)
    atomic_write_json(SAM3_FOCUS_REGIONS_PATH, {
        "schema_version": SAM3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "focus_regions": focus_regions,
    })
    atomic_write_json(SAM3_FAILURES_PATH, {
        "schema_version": SAM3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "failures": failures,
    })
    atomic_write_json(SAM3_RUN_SUMMARY_PATH, summary)

    print(f"[done] tracks: {SAM3_TRACKS_ALL_PATH}")
    print(f"[done] focus regions: {SAM3_FOCUS_REGIONS_PATH}")
    print(f"[done] summary: {SAM3_RUN_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
