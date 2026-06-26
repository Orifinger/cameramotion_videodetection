#!/usr/bin/env python3
"""SAM3 v4 multi-instance track-bank runner."""

from __future__ import annotations
import json, math, sys, time, traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
try:
    import cv2
except ImportError:
    cv2 = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.sam3_tracking_config import (
    QWEN_INPUT_SCHEMA_VERSION, QWEN_SAM3_CANDIDATES_PATH,
    SAM3_CHECKPOINT_PATH, SAM3_CLEAR_CACHE_THRESHOLD, SAM3_CLOSE_SESSION_RUN_GC,
    SAM3_FAILURES_PATH, SAM3_MAX_BORDER_TOUCH_RATIO, SAM3_MAX_CANDIDATES_PER_VIDEO,
    SAM3_MAX_MEDIAN_AREA_RATIO, SAM3_MAX_VIDEOS, SAM3_MIN_LONGEST_VISIBLE_RUN,
    SAM3_MIN_MEDIAN_AREA_RATIO, SAM3_MIN_VISIBLE_FRAME_RATIO,
    SAM3_OUTPUT_PROB_THRESH, SAM3_PROPAGATION_DIRECTION, SAM3_QUALITY_TRACKS_PATH,
    SAM3_RUN_SUMMARY_PATH, SAM3_SAVE_MASK_TUBES, SAM3_SCHEMA_VERSION,
    SAM3_SOURCE_ROOT, SAM3_TRACK_MASK_ROOT, SAM3_TRACKS_ALL_PATH,
)
if str(SAM3_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_SOURCE_ROOT))
from sam3.model_builder import build_sam3_video_predictor

REQUIRED = {
    "candidate_id", "region_family", "candidate_class", "target_scope",
    "canonical_concept", "display_phrase", "sam_prompt", "instance_count_hint",
    "visual_disambiguators", "screen_region", "temporal_presence",
}
FAMILIES = {"physical_instance", "editable_surface"}
CLASSES = {
    "human", "animal", "vehicle", "handheld_object", "bounded_object",
    "display_screen", "sign_or_poster", "paper_book_map", "framed_art",
    "apparel_panel", "vehicle_panel", "package_front",
}
HINTS = {"unique_in_video", "possibly_multiple", "unknown"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def npv(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def scalar(value: Any) -> float | None:
    array = npv(value)
    if array.size != 1:
        return None
    try:
        answer = float(array.reshape(-1)[0])
    except (ValueError, TypeError):
        return None
    return answer if math.isfinite(answer) else None


def run_length(frames: list[int]) -> int:
    if not frames:
        return 0
    values = sorted(set(frames))
    best = current = 1
    for a, b in zip(values, values[1:]):
        current = current + 1 if b == a + 1 else 1
        best = max(best, current)
    return best


def hits_border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def gpu() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    d = torch.cuda.current_device()
    return {
        "device": int(d),
        "allocated_bytes": int(torch.cuda.memory_allocated(d)),
        "reserved_bytes": int(torch.cuda.memory_reserved(d)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(d)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(d)),
    }


def frame_count(path: str) -> int | None:
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return n if n > 0 else None


def anchor(candidate: dict[str, Any], nframes: int | None) -> int:
    if not nframes or nframes <= 1:
        return 0
    last = nframes - 1
    hint = candidate["temporal_presence"]
    if hint in {"middle", "brief"}:
        return last // 2
    if hint == "late":
        return int(round(last * 0.75))
    return 0


def validate(raw: Any, index: int) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw, dict):
        return None, f"candidate[{index}] is {type(raw).__name__}, not object"
    missing = sorted(REQUIRED - set(raw))
    if missing:
        return None, f"candidate[{index}] missing {missing}"
    family = raw["region_family"]
    scope = raw["target_scope"]
    if family not in FAMILIES or raw["candidate_class"] not in CLASSES:
        return None, f"candidate[{index}] has invalid family/class"
    needed_scope = "whole_instance" if family == "physical_instance" else "whole_surface"
    if scope != needed_scope:
        return None, f"candidate[{index}] family/scope mismatch"
    if raw["instance_count_hint"] not in HINTS:
        return None, f"candidate[{index}] invalid instance_count_hint"
    if not isinstance(raw["visual_disambiguators"], list):
        return None, f"candidate[{index}] disambiguators must be list"
    if not isinstance(raw["sam_prompt"], str) or not raw["sam_prompt"].strip():
        return None, f"candidate[{index}] invalid sam_prompt"
    return dict(raw), None


def read_candidates() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not QWEN_SAM3_CANDIDATES_PATH.is_file():
        raise FileNotFoundError(f"Missing Qwen v4 candidate file: {QWEN_SAM3_CANDIDATES_PATH}")
    dataset = json.loads(QWEN_SAM3_CANDIDATES_PATH.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or dataset.get("schema_version") != QWEN_INPUT_SCHEMA_VERSION:
        raise ValueError(f"Expected schema {QWEN_INPUT_SCHEMA_VERSION!r}")
    videos = dataset.get("videos")
    if not isinstance(videos, list):
        raise ValueError("Qwen candidate input lacks top-level videos list")
    checked = [
        video for video in videos
        if isinstance(video, dict)
        and video.get("video_id")
        and video.get("video_path")
        and isinstance(video.get("sam3_candidates"), list)
    ]
    return dataset, checked


class Runner:
    def __init__(self) -> None:
        if not SAM3_SOURCE_ROOT.is_dir() or not SAM3_CHECKPOINT_PATH.is_file():
            raise FileNotFoundError("SAM3 source or checkpoint is unavailable")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        self.model = build_sam3_video_predictor(
            checkpoint_path=str(SAM3_CHECKPOINT_PATH), gpus_to_use=[0]
        )
        print("[sam3] loaded", gpu())

    def start(self, path: str) -> str:
        response = self.model.handle_request({"type": "start_session", "resource_path": path})
        return str(response["session_id"])

    def reset(self, session_id: str) -> None:
        self.model.handle_request({"type": "reset_session", "session_id": session_id})

    def prompt(self, session_id: str, text: str, frame_index: int) -> dict[str, Any]:
        return self.model.handle_request({
            "type": "add_prompt", "session_id": session_id, "frame_index": frame_index,
            "text": text, "output_prob_thresh": SAM3_OUTPUT_PROB_THRESH,
        })

    def propagate(self, session_id: str):
        yield from self.model.handle_stream_request({
            "type": "propagate_in_video", "session_id": session_id,
            "propagation_direction": SAM3_PROPAGATION_DIRECTION,
            "output_prob_thresh": SAM3_OUTPUT_PROB_THRESH,
        })

    def close(self, session_id: str) -> Any:
        return self.model.handle_request({
            "type": "close_session", "session_id": session_id,
            "run_gc_collect": SAM3_CLOSE_SESSION_RUN_GC,
            "clear_cache_threshold": SAM3_CLEAR_CACHE_THRESHOLD,
        })

    def shutdown(self) -> None:
        self.model.shutdown()


def save_masks(video_id: str, candidate_id: str, object_id: int, frames: list[int], masks: list[np.ndarray]) -> str | None:
    if not SAM3_SAVE_MASK_TUBES or not masks:
        return None
    folder = SAM3_TRACK_MASK_ROOT / video_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{candidate_id}__obj_{object_id}.npz"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(
            handle,
            frame_indices=np.asarray(frames, dtype=np.int32),
            masks=np.stack(masks).astype(np.uint8, copy=False),
        )
    tmp.replace(path)
    return str(path)


def quality(track: dict[str, Any]) -> tuple[str, list[str], float]:
    bad: list[str] = []
    if track["visible_frame_ratio"] < SAM3_MIN_VISIBLE_FRAME_RATIO:
        bad.append("visible_frame_ratio")
    if track["longest_visible_run"] < SAM3_MIN_LONGEST_VISIBLE_RUN:
        bad.append("longest_visible_run")
    if track["median_area_ratio"] < SAM3_MIN_MEDIAN_AREA_RATIO:
        bad.append("median_area_ratio_too_small")
    if track["median_area_ratio"] > SAM3_MAX_MEDIAN_AREA_RATIO:
        bad.append("median_area_ratio_too_large")
    if track["border_touch_ratio"] > SAM3_MAX_BORDER_TOUCH_RATIO:
        bad.append("border_touch_ratio")
    stable = min(track["longest_visible_run"] / max(track["propagation_frame_count"], 1), 1.0)
    area_ok = float(SAM3_MIN_MEDIAN_AREA_RATIO <= track["median_area_ratio"] <= SAM3_MAX_MEDIAN_AREA_RATIO)
    score = (
        .40 * track["visible_frame_ratio"]
        + .25 * stable
        + .15 * area_ok
        + .10 * (1 - track["border_touch_ratio"])
        + .10 * (track["mean_detection_score"] or 0.0)
    )
    return ("pass" if not bad else "fail"), bad, round(score, 6)


def track_candidate(runner: Runner, session_id: str, video_id: str, candidate: dict[str, Any], nframes: int | None) -> dict[str, Any]:
    runner.reset(session_id)
    prompt_frame = anchor(candidate, nframes)
    response = runner.prompt(session_id, candidate["sam_prompt"], prompt_frame)
    by_object: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    output_keys: set[str] = set()
    seen_frames: set[int] = set()
    shape: tuple[int, int] | None = None

    for packet in runner.propagate(session_id):
        if not isinstance(packet, dict) or "frame_index" not in packet:
            continue
        frame = int(packet["frame_index"])
        output = packet.get("outputs")
        if not isinstance(output, dict):
            continue
        seen_frames.add(frame)
        output_keys.update(output)
        object_ids = npv(output.get("out_obj_ids", []))
        masks = npv(output.get("out_binary_masks", []))
        boxes = npv(output.get("out_boxes_xywh", []))
        scores = npv(output.get("out_probs", []))
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None]
        if masks.ndim == 3 and len(masks):
            shape = (int(masks.shape[1]), int(masks.shape[2]))
        if object_ids.ndim == 0:
            object_ids = object_ids.reshape(1)
        for pos, raw_id in enumerate(object_ids.reshape(-1).tolist()):
            if pos >= len(masks):
                continue
            mask = np.asarray(masks[pos], dtype=bool)
            if mask.ndim != 2 or not mask.any():
                continue
            box = boxes[pos].tolist() if pos < len(boxes) else None
            score = scalar(scores[pos]) if pos < len(scores) else None
            by_object[int(raw_id)][frame] = {"mask": mask, "box": box, "score": score}

    nprop = len(seen_frames)
    tracks: list[dict[str, Any]] = []
    for object_id, items in by_object.items():
        ordered = sorted(items.items())
        frames = [frame for frame, _ in ordered]
        masks = [entry["mask"] for _, entry in ordered]
        areas = [float(mask.mean()) for mask in masks]
        scores = [entry["score"] for _, entry in ordered if entry["score"] is not None]
        borders = [hits_border(mask) for mask in masks]
        track = {
            "track_id": f"{video_id}__{candidate['candidate_id']}__obj_{object_id}",
            "sam_object_id": object_id,
            "first_frame_index": min(frames),
            "last_frame_index": max(frames),
            "visible_frame_count": len(frames),
            "propagation_frame_count": nprop,
            "visible_frame_ratio": round(len(frames) / max(nprop, 1), 6),
            "longest_visible_run": run_length(frames),
            "median_area_ratio": round(float(np.median(areas)), 8),
            "mean_area_ratio": round(float(np.mean(areas)), 8),
            "std_area_ratio": round(float(np.std(areas)), 8),
            "mean_detection_score": round(float(np.mean(scores)), 6) if scores else None,
            "border_touch_ratio": round(float(np.mean(borders)), 6),
            "bbox_tube_xywh_norm": [
                {"frame_index": frame, "bbox_xywh_norm": entry["box"]}
                for frame, entry in ordered
            ],
            "mask_tube_path": save_masks(video_id, candidate["candidate_id"], object_id, frames, masks),
        }
        status, reasons, score = quality(track)
        track["quality_status"] = status
        track["quality_reasons"] = reasons
        track["track_quality_score"] = score
        tracks.append(track)
    tracks.sort(key=lambda x: x["track_quality_score"], reverse=True)
    return {
        "candidate_id": candidate["candidate_id"],
        "region_family": candidate["region_family"],
        "candidate_class": candidate["candidate_class"],
        "canonical_concept": candidate["canonical_concept"],
        "display_phrase": candidate["display_phrase"],
        "sam_prompt": candidate["sam_prompt"],
        "instance_count_hint": candidate["instance_count_hint"],
        "visual_disambiguators": candidate["visual_disambiguators"],
        "prompt_frame_index": prompt_frame,
        "source_video_frame_count": nframes,
        "anchor_policy": "qwen_temporal_presence",
        "add_prompt_output_keys": sorted(response.get("outputs", {})) if isinstance(response.get("outputs"), dict) else None,
        "propagation_frame_count": nprop,
        "mask_frame_height_width": list(shape) if shape else None,
        "propagation_output_keys": sorted(output_keys),
        "tracks": tracks,
        "status": "success" if tracks else "no_track",
    }


def process_video(runner: Runner, video: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    video_id, path = str(video["video_id"]), str(video["video_path"])
    nframes = frame_count(path)
    result: dict[str, Any] = {
        "video_id": video_id, "relative_path": video.get("relative_path"),
        "video_path": path, "source_video_frame_count": nframes, "status": "failure",
        "candidate_results": [], "created_at_utc": now(), "gpu_memory_before": gpu(),
    }
    checked: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, raw in enumerate(video["sam3_candidates"][:SAM3_MAX_CANDIDATES_PER_VIDEO]):
        candidate, err = validate(raw, index)
        if err:
            errors.append(err)
        else:
            checked.append(candidate)
    result["input_candidate_count"] = len(video["sam3_candidates"])
    result["processed_candidate_count"] = len(checked)
    if errors:
        result["input_errors"] = errors
    if not checked:
        result["status"] = "no_valid_candidate"
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        result["gpu_memory_after"] = gpu()
        return result
    session_id: str | None = None
    try:
        session_id = runner.start(path)
        for candidate in checked:
            try:
                result["candidate_results"].append(track_candidate(runner, session_id, video_id, candidate, nframes))
            except Exception as exc:
                result["candidate_results"].append({
                    "candidate_id": candidate["candidate_id"], "sam_prompt": candidate["sam_prompt"],
                    "display_phrase": candidate["display_phrase"], "status": "failure",
                    "error_type": type(exc).__name__, "error": str(exc),
                    "traceback_tail": traceback.format_exc(limit=5),
                })
        statuses = [item["status"] for item in result["candidate_results"]]
        result["status"] = "success" if "success" in statuses else "no_track" if "no_track" in statuses else "failure"
    except Exception as exc:
        result.update({
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback_tail": traceback.format_exc(limit=8),
        })
    finally:
        if session_id:
            try:
                result["close_session"] = runner.close(session_id)
            except Exception as exc:
                result["close_session_error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        result["gpu_memory_after"] = gpu()
    return result


def quality_index(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video in results:
        for candidate in video.get("candidate_results", []):
            if not isinstance(candidate, dict):
                continue
            for track in candidate.get("tracks", []):
                if isinstance(track, dict) and track.get("quality_status") == "pass":
                    rows.append({
                        "video_id": video["video_id"], "relative_path": video.get("relative_path"),
                        "video_path": video["video_path"], "candidate_id": candidate["candidate_id"],
                        "candidate_class": candidate.get("candidate_class"),
                        "canonical_concept": candidate.get("canonical_concept"),
                        "display_phrase": candidate.get("display_phrase"),
                        "sam_prompt": candidate.get("sam_prompt"), **track,
                    })
    return rows


def write_all(dataset: dict[str, Any], input_videos: list[dict[str, Any]], results: list[dict[str, Any]], started: float) -> None:
    rows = quality_index(results)
    dump(SAM3_TRACKS_ALL_PATH, {
        "schema_version": SAM3_SCHEMA_VERSION, "created_at_utc": now(),
        "input_file": str(QWEN_SAM3_CANDIDATES_PATH), "videos": results,
    })
    dump(SAM3_QUALITY_TRACKS_PATH, {
        "schema_version": SAM3_SCHEMA_VERSION, "created_at_utc": now(), "tracks": rows,
    })
    dump(SAM3_FAILURES_PATH, {
        "schema_version": SAM3_SCHEMA_VERSION, "created_at_utc": now(),
        "failures": [x for x in results if x.get("status") == "failure"],
    })
    status: dict[str, int] = defaultdict(int)
    all_tracks = 0
    for result in results:
        status[result["status"]] += 1
        for candidate in result.get("candidate_results", []):
            if isinstance(candidate, dict):
                all_tracks += len(candidate.get("tracks", []))
    dump(SAM3_RUN_SUMMARY_PATH, {
        "schema_version": SAM3_SCHEMA_VERSION, "created_at_utc": now(),
        "input_file": str(QWEN_SAM3_CANDIDATES_PATH),
        "input_schema_version": dataset.get("schema_version"),
        "input_video_records": len(input_videos), "processed_videos": len(results),
        "video_status_totals": dict(status),
        "track_totals": {
            "all_tracks": all_tracks, "quality_pass_tracks": len(rows),
            "mask_tubes_saved": SAM3_SAVE_MASK_TUBES,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "throughput_videos_per_min": round(len(results) / max(time.perf_counter() - started, 1e-9) * 60, 3),
        "sam3_source_root": str(SAM3_SOURCE_ROOT),
        "sam3_checkpoint_path": str(SAM3_CHECKPOINT_PATH),
    })


def main() -> None:
    started = time.perf_counter()
    dataset, videos = read_candidates()
    if SAM3_MAX_VIDEOS is not None:
        videos = videos[:SAM3_MAX_VIDEOS]
    if not videos:
        raise RuntimeError("No valid Qwen v4 candidates available")
    print(f"[input] selected {len(videos)} video(s)")
    runner = Runner()
    results: list[dict[str, Any]] = []
    try:
        for index, video in enumerate(videos, 1):
            print(f"[video {index}/{len(videos)}] {video['video_id']}")
            results.append(process_video(runner, video))
            write_all(dataset, videos, results, started)
            print(f"[video {index}/{len(videos)}] {results[-1]['status']} {results[-1]['elapsed_seconds']}s")
    finally:
        runner.shutdown()
    print(f"[done] track bank: {SAM3_TRACKS_ALL_PATH}")
    print(f"[done] quality tracks: {SAM3_QUALITY_TRACKS_PATH}")


if __name__ == "__main__":
    main()
