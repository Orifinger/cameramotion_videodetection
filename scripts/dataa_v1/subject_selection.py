"""Subject-first target selection for Data A v1.

This module only reads existing track-bank records and mask tubes. It does not
run object discovery, tracking, pairing, media packaging, or VACE inference.
"""

from __future__ import annotations

import hashlib
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from .common import DataAError, read_json
from .mask_io import MaskTube, load_mask_tube
from .media_io import ffprobe_video
from .path_resolver import PathResolver
from .schema import MASK_PATH_KEYS, TRACK_LIST_KEYS, as_records, first_value


DEFAULT_SUBJECT_SELECTION_CONFIG: Dict[str, Any] = {
    "random_seed": 20260629,
    "primary_probability": 0.90,
    "universal_gate": {
        "min_contiguous_visible_seconds": 1.0,
        "min_median_mask_area_ratio": 0.012,
        "min_p20_mask_area_ratio": 0.006,
        "min_median_bbox_short_side_720": 80,
    },
    "secondary_gate": {
        "min_median_mask_area_ratio": 0.020,
        "min_p20_mask_area_ratio": 0.010,
        "min_median_bbox_short_side_720": 112,
    },
    "operation_gates": {
        "object_swap": {
            "min_median_mask_area_ratio": 0.025,
            "min_p20_mask_area_ratio": 0.012,
            "min_median_bbox_short_side_720": 112,
        },
        "person_appearance_swap": {
            "min_median_mask_area_ratio": 0.025,
            "min_p20_mask_area_ratio": 0.012,
            "min_median_bbox_short_side_720": 112,
        },
        "object_attribute_edit": {
            "min_median_mask_area_ratio": 0.020,
            "min_p20_mask_area_ratio": 0.010,
            "min_median_bbox_short_side_720": 96,
        },
        "surface_content_edit": {
            "min_median_mask_area_ratio": 0.012,
            "min_p20_mask_area_ratio": 0.006,
            "min_median_bbox_short_side_720": 80,
        },
        "surface_attribute_edit": {
            "min_median_mask_area_ratio": 0.012,
            "min_p20_mask_area_ratio": 0.006,
            "min_median_bbox_short_side_720": 80,
        },
    },
    "score_weights": {
        "robust_area": 0.40,
        "frame_centrality": 0.20,
        "temporal_visibility": 0.20,
        "track_quality": 0.15,
        "semantic_compatibility": 0.05,
    },
    "semantic_bonus": {
        "human": 0.05,
        "vehicle": 0.03,
        "bounded_object": 0.02,
        "display_screen": 0.01,
        "sign_or_poster": 0.01,
        "framed_art": 0.01,
        "paper_book_map": 0.01,
    },
    "secondary_dedupe": {
        "median_bbox_iou_threshold": 0.70,
        "median_bbox_containment_threshold": 0.80,
    },
}

SUBJECT_MASK_PATH_KEYS = tuple(
    dict.fromkeys(
        (
            *MASK_PATH_KEYS,
            "mask_npz_path",
            "mask_tube_npz_path",
            "mask_tube_npz",
            "sam3_mask_path",
            "sam3_mask_npz_path",
            "track_mask_path",
            "track_mask_npz_path",
            "track_mask_npz",
            "npz_path",
        )
    )
)
NESTED_MASK_CONTAINERS = ("mask", "mask_tube", "sam3", "sam3_mask", "track_mask")
NESTED_MASK_PATH_KEYS = ("path", "local_path", "npz_path", "file", *SUBJECT_MASK_PATH_KEYS)


def _merge_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {key: value for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_selection_config(path: Path | None = None, *, overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    config = dict(DEFAULT_SUBJECT_SELECTION_CONFIG)
    if path is not None:
        config = _merge_config(config, read_json(path))
    if overrides:
        config = _merge_config(config, overrides)
    return config


@dataclass
class SubjectMetrics:
    median_mask_area_ratio: float
    p20_mask_area_ratio: float
    median_bbox_short_side_720: float
    frame_centrality: float
    temporal_visibility_seconds: float
    track_quality: float
    semantic_compatibility: float
    semantic_label: str
    subject_score: float
    robust_area: float
    bbox_by_frame: Dict[int, tuple[float, float, float, float]] = field(repr=False)


@dataclass
class EvaluatedTrack:
    record: Dict[str, Any]
    video_id: str
    track_id: str
    candidate_class: Optional[str]
    canonical_concept: Optional[str]
    metrics: Optional[SubjectMetrics]
    eligible_universal: bool
    eligible_secondary_gate: bool
    rejection_tags: list[str]
    rejection_reasons: list[str] = field(default_factory=list)
    selection_status: str = "not_selected"
    selection_role: Optional[str] = None
    selection_mode: Optional[str] = None
    secondary_pool_size: int = 0
    selection_random_value: Optional[float] = None

    @property
    def subject_score(self) -> float:
        return 0.0 if self.metrics is None else float(self.metrics.subject_score)


@dataclass
class VideoSelection:
    video_id: str
    selected: Optional[EvaluatedTrack]
    primary: Optional[EvaluatedTrack]
    secondary_pool: list[EvaluatedTrack]
    candidates: list[EvaluatedTrack]
    random_value: Optional[float]


def _optional_str(value: Any) -> Optional[str]:
    return None if value in (None, "") else str(value)


def _optional_path_str(value: Any) -> Optional[str]:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and value:
        return value
    return None


def _required_str(record: Mapping[str, Any], key: str, *, fallback: str | None = None) -> str:
    value = _optional_str(record.get(key))
    if value:
        return value
    if fallback is not None:
        return fallback
    raise DataAError(f"track record missing required field: {key}")


def _mask_path_candidates(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(label: str, value: Any) -> None:
        path = _optional_path_str(value)
        if not path or path in seen:
            return
        seen.add(path)
        candidates.append((label, path))

    for key in SUBJECT_MASK_PATH_KEYS:
        add(key, record.get(key))

    for container_key in NESTED_MASK_CONTAINERS:
        nested = record.get(container_key)
        if not isinstance(nested, Mapping):
            continue
        for key in NESTED_MASK_PATH_KEYS:
            add(f"{container_key}.{key}", nested.get(key))
    return candidates


def _mask_load_error_tag(exc: Exception) -> str:
    message = str(exc)
    if "does not exist" in message:
        return "invalid_mask_tube:mask_npz_does_not_exist"
    if "must contain frame_indices and masks" in message:
        return "invalid_mask_tube:npz_missing_frame_indices_or_masks"
    if "frame_indices must be int32" in message:
        return "invalid_mask_tube:frame_indices_not_int32"
    if "frame_indices must be 1D" in message:
        return "invalid_mask_tube:frame_indices_not_1d"
    if "masks must have [N,H,W]" in message:
        return "invalid_mask_tube:masks_not_nhw"
    if "masks must be uint8" in message:
        return "invalid_mask_tube:masks_not_uint8"
    if "different N" in message:
        return "invalid_mask_tube:frame_mask_count_mismatch"
    if "empty mask tube" in message:
        return "invalid_mask_tube:empty_mask_tube"
    if "not strictly increasing" in message:
        return "invalid_mask_tube:frame_indices_not_strictly_increasing"
    return f"invalid_mask_tube:{type(exc).__name__}"


def _blocked_track(
    raw: Dict[str, Any],
    video_id: str,
    track_id: str,
    candidate_class: Optional[str],
    canonical_concept: Optional[str],
    tags: list[str],
    reasons: list[str] | None = None,
) -> EvaluatedTrack:
    return EvaluatedTrack(
        record=raw,
        video_id=video_id,
        track_id=track_id,
        candidate_class=candidate_class,
        canonical_concept=canonical_concept,
        metrics=None,
        eligible_universal=False,
        eligible_secondary_gate=False,
        rejection_tags=tags,
        rejection_reasons=reasons or [],
        selection_status="ineligible_small_or_weak",
    )


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _longest_contiguous_run(frame_indices: np.ndarray) -> int:
    if frame_indices.size == 0:
        return 0
    longest = current = 1
    previous = int(frame_indices[0])
    for raw in frame_indices[1:]:
        value = int(raw)
        if value - previous == 1:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
        previous = value
    return max(longest, current)


def _mask_bboxes_xywh(tube: MaskTube) -> Dict[int, tuple[float, float, float, float]]:
    bboxes: Dict[int, tuple[float, float, float, float]] = {}
    for frame_index, mask in zip(tube.frame_indices, tube.masks):
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        bboxes[int(frame_index)] = (float(x0), float(y0), float(x1 - x0), float(y1 - y0))
    return bboxes


def _canonical_bboxes(
    bboxes: Mapping[int, tuple[float, float, float, float]],
    *,
    source_height: int,
    source_width: int,
    canonical_height: int = 720,
    canonical_width: int = 1280,
) -> Dict[int, tuple[float, float, float, float]]:
    sx = canonical_width / float(source_width)
    sy = canonical_height / float(source_height)
    return {
        frame: (x * sx, y * sy, w * sx, h * sy)
        for frame, (x, y, w, h) in bboxes.items()
    }


def _median_bbox_short_side(bboxes: Mapping[int, tuple[float, float, float, float]]) -> float:
    if not bboxes:
        return 0.0
    return float(np.median([min(w, h) for _x, _y, w, h in bboxes.values()]))


def _frame_centrality(bboxes: Mapping[int, tuple[float, float, float, float]], *, height: int = 720, width: int = 1280) -> float:
    if not bboxes:
        return 0.0
    distances = []
    max_distance = float(np.sqrt(0.5))
    for x, y, w, h in bboxes.values():
        cx = (x + w / 2.0) / width
        cy = (y + h / 2.0) / height
        distances.append(np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) / max_distance)
    return _clamp(1.0 - float(np.median(distances)))


def _track_quality(record: Mapping[str, Any]) -> float:
    for key in ("track_quality_score", "quality_score", "score"):
        value = record.get(key)
        if value not in (None, ""):
            try:
                return _clamp(float(value))
            except (TypeError, ValueError):
                return 0.0
    return 0.5


def _semantic_label(record: Mapping[str, Any], bonus: Mapping[str, Any]) -> str:
    values = [
        record.get("region_family"),
        record.get("candidate_class"),
        record.get("canonical_concept"),
        record.get("content_domain"),
    ]
    text = " ".join(str(v).lower() for v in values if v not in (None, ""))
    for label in bonus:
        if label.lower() in text:
            return str(label)
    return "other"


def _semantic_score(record: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[float, str]:
    bonus = config.get("semantic_bonus") or {}
    label = _semantic_label(record, bonus)
    raw_bonus = float(bonus.get(label, 0.0))
    max_bonus = max([float(v) for v in bonus.values()] + [0.05])
    return _clamp(raw_bonus / max_bonus), label


def _robust_area(median_ratio: float, p20_ratio: float, config: Mapping[str, Any]) -> float:
    universal = config["universal_gate"]
    secondary = config["secondary_gate"]
    median_ref = max(float(secondary["min_median_mask_area_ratio"]) * 4.0, float(universal["min_median_mask_area_ratio"]))
    p20_ref = max(float(secondary["min_p20_mask_area_ratio"]) * 4.0, float(universal["min_p20_mask_area_ratio"]))
    return 0.5 * _clamp(median_ratio / median_ref) + 0.5 * _clamp(p20_ratio / p20_ref)


class VideoFpsResolver:
    def __init__(self, *, ffprobe_bin: str = "ffprobe") -> None:
        self.ffprobe_bin = ffprobe_bin
        self._cache: Dict[str, float] = {}
        self._errors: Dict[str, str] = {}

    def resolve(self, record: Mapping[str, Any]) -> tuple[Optional[float], Optional[str]]:
        video_id = _optional_str(record.get("video_id") or record.get("source_video_id")) or "<missing_video_id>"
        if video_id in self._cache:
            return self._cache[video_id], None
        for key in ("source_fps", "fps"):
            if record.get(key) not in (None, ""):
                try:
                    fps = float(record[key])
                except (TypeError, ValueError):
                    break
                if fps > 0:
                    self._cache[video_id] = fps
                    return fps, None
        video_path = _optional_str(record.get("video_path") or record.get("source_video_path") or record.get("path"))
        if not video_path:
            return None, "source_fps_unavailable"
        try:
            fps = ffprobe_video(Path(video_path), ffprobe_bin=self.ffprobe_bin).fps
        except Exception as exc:  # noqa: BLE001 - audit exact reason, do not hide data problems
            self._errors[video_id] = f"{type(exc).__name__}: {exc}"
            return None, "source_fps_unavailable"
        self._cache[video_id] = fps
        return fps, None


def evaluate_track(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    fps_resolver: VideoFpsResolver,
    path_resolver: PathResolver | None = None,
) -> EvaluatedTrack:
    raw = dict(record)
    video_id = _required_str(raw, "video_id", fallback=_optional_str(raw.get("source_video_id")))
    track_id = _required_str(raw, "track_id", fallback=_optional_str(raw.get("candidate_id")))
    candidate_class = _optional_str(raw.get("candidate_class"))
    canonical_concept = _optional_str(raw.get("canonical_concept"))
    rejection_tags: list[str] = []

    resolver = path_resolver or PathResolver({})
    mask_candidates = _mask_path_candidates(raw)
    if not mask_candidates:
        return _blocked_track(
            raw,
            video_id,
            track_id,
            candidate_class,
            canonical_concept,
            ["missing_mask_tube_path"],
            [f"no string mask path field found; checked={','.join(SUBJECT_MASK_PATH_KEYS)}"],
        )
    mask_path: str | None = None
    mask_path_reasons: list[str] = []
    for key, candidate_path in mask_candidates:
        resolved = resolver.resolve(candidate_path)
        if resolved.state in {"readable_persistent", "readable_volatile"} and resolved.resolved_path:
            mask_path = resolved.resolved_path
            raw["mask_tube_path"] = candidate_path
            raw["subject_selection_mask_path_key"] = key
            raw["subject_selection_resolved_mask_path"] = resolved.resolved_path
            raw["subject_selection_mask_path_state"] = resolved.state
            break
        mask_path_reasons.append(f"{key}={candidate_path} -> {resolved.state}: {resolved.note}")
    if not mask_path:
        return _blocked_track(
            raw,
            video_id,
            track_id,
            candidate_class,
            canonical_concept,
            ["mask_tube_path_unreadable"],
            mask_path_reasons,
        )
    fps, fps_error = fps_resolver.resolve(raw)
    if fps_error or fps is None or fps <= 0:
        return _blocked_track(raw, video_id, track_id, candidate_class, canonical_concept, [fps_error or "source_fps_unavailable"])
    try:
        tube = load_mask_tube(Path(mask_path))
    except Exception as exc:  # noqa: BLE001
        return _blocked_track(
            raw,
            video_id,
            track_id,
            candidate_class,
            canonical_concept,
            [_mask_load_error_tag(exc)],
            [f"{type(exc).__name__}: {exc}; path={mask_path}"],
        )

    areas = tube.masks.reshape(tube.masks.shape[0], -1).mean(axis=1)
    median_area = float(np.median(areas))
    p20_area = float(np.quantile(areas, 0.20))
    source_bboxes = _mask_bboxes_xywh(tube)
    canonical_bboxes = _canonical_bboxes(source_bboxes, source_height=tube.height, source_width=tube.width)
    median_short = _median_bbox_short_side(canonical_bboxes)
    temporal_seconds = float(_longest_contiguous_run(tube.frame_indices) / fps)
    centrality = _frame_centrality(canonical_bboxes)
    quality = _track_quality(raw)
    semantic, semantic_label = _semantic_score(raw, config)
    robust = _robust_area(median_area, p20_area, config)
    temporal_score = _clamp(temporal_seconds / 5.0)
    weights = config["score_weights"]
    subject_score = (
        float(weights["robust_area"]) * robust
        + float(weights["frame_centrality"]) * centrality
        + float(weights["temporal_visibility"]) * temporal_score
        + float(weights["track_quality"]) * quality
        + float(weights["semantic_compatibility"]) * semantic
    )

    universal = config["universal_gate"]
    if temporal_seconds < float(universal["min_contiguous_visible_seconds"]):
        rejection_tags.append("visible_duration_too_short")
    if median_area < float(universal["min_median_mask_area_ratio"]):
        rejection_tags.append("median_mask_area_ratio_below_universal_threshold")
    if p20_area < float(universal["min_p20_mask_area_ratio"]):
        rejection_tags.append("p20_mask_area_ratio_below_universal_threshold")
    if median_short < float(universal["min_median_bbox_short_side_720"]):
        rejection_tags.append("bbox_short_side_too_small_after_vace_resize")

    secondary = config["secondary_gate"]
    secondary_ok = (
        median_area >= float(secondary["min_median_mask_area_ratio"])
        and p20_area >= float(secondary["min_p20_mask_area_ratio"])
        and median_short >= float(secondary["min_median_bbox_short_side_720"])
    )
    metrics = SubjectMetrics(
        median_mask_area_ratio=median_area,
        p20_mask_area_ratio=p20_area,
        median_bbox_short_side_720=median_short,
        frame_centrality=centrality,
        temporal_visibility_seconds=temporal_seconds,
        track_quality=quality,
        semantic_compatibility=semantic,
        semantic_label=semantic_label,
        subject_score=float(subject_score),
        robust_area=robust,
        bbox_by_frame=canonical_bboxes,
    )
    return EvaluatedTrack(
        record=raw,
        video_id=video_id,
        track_id=track_id,
        candidate_class=candidate_class,
        canonical_concept=canonical_concept,
        metrics=metrics,
        eligible_universal=not rejection_tags,
        eligible_secondary_gate=secondary_ok and not rejection_tags,
        rejection_tags=rejection_tags,
        selection_status="ineligible_small_or_weak" if rejection_tags else "eligible_not_selected",
    )


def evaluate_tracks(
    records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    ffprobe_bin: str = "ffprobe",
    path_resolver: PathResolver | None = None,
    progress_every: int = 0,
    num_workers: int = 1,
) -> list[EvaluatedTrack]:
    fps_resolver = VideoFpsResolver(ffprobe_bin=ffprobe_bin)
    resolver = path_resolver or PathResolver({})
    items = list(records)
    total = len(items)
    workers = max(1, min(int(num_workers), total or 1))
    if progress_every > 0:
        print(f"subject_selection workers: {workers}", file=sys.stderr, flush=True)
    if workers > 1 and total > 1:
        return _evaluate_tracks_parallel(
            items,
            config,
            ffprobe_bin=ffprobe_bin,
            path_resolver=resolver,
            progress_every=progress_every,
            num_workers=workers,
        )

    evaluated: list[EvaluatedTrack] = []
    for index, record in enumerate(items, start=1):
        evaluated.append(evaluate_track(record, config, fps_resolver=fps_resolver, path_resolver=resolver))
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == total):
            print(f"subject_selection progress: evaluated {index}/{total} tracks", file=sys.stderr, flush=True)
    return evaluated


def _evaluate_track_worker(payload: tuple[Mapping[str, Any], Mapping[str, Any], str, PathResolver]) -> EvaluatedTrack:
    record, config, ffprobe_bin, resolver = payload
    return evaluate_track(
        record,
        config,
        fps_resolver=VideoFpsResolver(ffprobe_bin=ffprobe_bin),
        path_resolver=resolver,
    )


def _evaluate_tracks_parallel(
    items: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    ffprobe_bin: str,
    path_resolver: PathResolver,
    progress_every: int,
    num_workers: int,
) -> list[EvaluatedTrack]:
    total = len(items)
    evaluated: list[EvaluatedTrack | None] = [None] * total
    done = 0
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_evaluate_track_worker, (record, config, ffprobe_bin, path_resolver)): index
            for index, record in enumerate(items)
        }
        for future in as_completed(futures):
            evaluated[futures[future]] = future.result()
            done += 1
            if progress_every > 0 and (done == 1 or done % progress_every == 0 or done == total):
                print(f"subject_selection progress: evaluated {done}/{total} tracks", file=sys.stderr, flush=True)
    return [item for item in evaluated if item is not None]


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter
    iou = 0.0 if union <= 0 else inter / union
    containment = 0.0 if min(area_a, area_b) <= 0 else inter / min(area_a, area_b)
    return float(iou), float(containment)


def _overlaps_primary(candidate: EvaluatedTrack, primary: EvaluatedTrack, config: Mapping[str, Any]) -> bool:
    if candidate.metrics is None or primary.metrics is None:
        return False
    common = sorted(set(candidate.metrics.bbox_by_frame).intersection(primary.metrics.bbox_by_frame))
    if not common:
        return False
    ious = []
    containments = []
    for frame in common:
        iou, containment = _bbox_iou(candidate.metrics.bbox_by_frame[frame], primary.metrics.bbox_by_frame[frame])
        ious.append(iou)
        containments.append(containment)
    dedupe = config.get("secondary_dedupe") or {}
    return (
        float(np.median(ious)) >= float(dedupe.get("median_bbox_iou_threshold", 0.70))
        or float(np.median(containments)) >= float(dedupe.get("median_bbox_containment_threshold", 0.80))
    )


def _stable_rng(seed: int, video_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{video_id}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _weighted_secondary(rng: random.Random, secondary_pool: list[EvaluatedTrack]) -> EvaluatedTrack:
    total = sum(max(item.subject_score, 0.0) for item in secondary_pool)
    if total <= 0:
        return sorted(secondary_pool, key=lambda item: item.track_id)[0]
    pick = rng.random() * total
    running = 0.0
    for item in sorted(secondary_pool, key=lambda item: item.track_id):
        running += max(item.subject_score, 0.0)
        if pick <= running:
            return item
    return sorted(secondary_pool, key=lambda item: item.track_id)[-1]


def select_subjects_by_video(evaluated: Iterable[EvaluatedTrack], config: Mapping[str, Any]) -> Dict[str, VideoSelection]:
    by_video: Dict[str, list[EvaluatedTrack]] = {}
    for track in evaluated:
        by_video.setdefault(track.video_id, []).append(track)

    selections: Dict[str, VideoSelection] = {}
    seed = int(config.get("random_seed", 20260629))
    primary_probability = float(config.get("primary_probability", 0.85))
    for video_id, candidates in sorted(by_video.items()):
        for candidate in candidates:
            candidate.selection_status = "ineligible_small_or_weak" if candidate.rejection_tags else "eligible_not_selected"
            candidate.selection_role = None
            candidate.selection_mode = None
            candidate.secondary_pool_size = 0
            candidate.selection_random_value = None
        eligible = [item for item in candidates if item.eligible_universal]
        if not eligible:
            selections[video_id] = VideoSelection(video_id, None, None, [], candidates, None)
            continue
        primary = max(eligible, key=lambda item: (item.subject_score, item.track_id))
        primary.selection_status = "primary_subject"
        primary.selection_role = "primary_subject"
        secondary_pool = []
        for item in eligible:
            if item is primary:
                continue
            if not item.eligible_secondary_gate:
                continue
            if _overlaps_primary(item, primary, config):
                item.selection_status = "duplicate_secondary_high_iou"
                item.rejection_tags = sorted(set(item.rejection_tags + ["duplicate_with_primary_subject"]))
                continue
            item.selection_status = "eligible_secondary"
            item.selection_role = "eligible_secondary"
            secondary_pool.append(item)

        rng = _stable_rng(seed, video_id)
        random_value = rng.random() if secondary_pool else 0.0
        if not secondary_pool:
            selected = primary
            selected.selection_role = "fallback_primary"
            selected.selection_mode = "fallback"
        elif random_value < primary_probability:
            selected = primary
            selected.selection_role = "primary_subject"
            selected.selection_mode = "primary_weighted"
        else:
            selected = _weighted_secondary(rng, secondary_pool)
            selected.selection_role = "eligible_secondary"
            selected.selection_mode = "secondary_weighted"
        selected.selection_status = "selected"
        selected.secondary_pool_size = len(secondary_pool)
        selected.selection_random_value = random_value
        selections[video_id] = VideoSelection(video_id, selected, primary, secondary_pool, candidates, random_value)
    return selections


def metric_payload(track: EvaluatedTrack) -> Dict[str, Any]:
    if track.metrics is None:
        return {
            "subject_score": 0.0,
            "median_mask_area_ratio": 0.0,
            "p20_mask_area_ratio": 0.0,
            "median_bbox_short_side_720": 0.0,
            "temporal_visibility_seconds": 0.0,
            "frame_centrality": 0.0,
            "track_quality": 0.0,
            "semantic_compatibility": 0.0,
        }
    metrics = track.metrics
    return {
        "subject_score": metrics.subject_score,
        "median_mask_area_ratio": metrics.median_mask_area_ratio,
        "p20_mask_area_ratio": metrics.p20_mask_area_ratio,
        "median_bbox_short_side_720": metrics.median_bbox_short_side_720,
        "temporal_visibility_seconds": metrics.temporal_visibility_seconds,
        "frame_centrality": metrics.frame_centrality,
        "track_quality": metrics.track_quality,
        "semantic_compatibility": metrics.semantic_compatibility,
        "semantic_label": metrics.semantic_label,
        "robust_area": metrics.robust_area,
    }


def audit_record(track: EvaluatedTrack) -> Dict[str, Any]:
    payload = {
        "video_id": track.video_id,
        "track_id": track.track_id,
        "candidate_class": track.candidate_class,
        "canonical_concept": track.canonical_concept,
        "selection_status": track.selection_status,
        "selection_role": track.selection_role,
        "secondary_pool_size": track.secondary_pool_size,
        "rejection_tags": track.rejection_tags,
        "rejection_reasons": track.rejection_reasons,
    }
    payload.update(metric_payload(track))
    return payload


def load_track_bank_records(path: Path) -> list[Dict[str, Any]]:
    return as_records(read_json(path), TRACK_LIST_KEYS, "track-bank")
