#!/usr/bin/env python3
"""Render visual mask videos for a Data A v1 generation plan.

The original .npz mask tube remains the canonical, lossless supervision. This tool
creates MP4s only for visual QA/debugging:
  - *_mask_raw.mp4: black background + white binary mask
  - *_mask_overlay.mp4: source frames + translucent mask + rectangular bbox

By default it renders the full visible span from min(frame_indices) to
max(frame_indices), keeping black mask frames for tracking gaps so they are visible.

Example:
  python scripts/render_dataa_v1_mask_videos.py \
    --plan res/dataA_v1/plans/vace14b_stage1_quota_plan.json \
    --out-dir res/dataA_v1/review/vace14b_stage1/mask_videos
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


TARGET_COLOR = (0, 70, 255)  # BGR orange-red
DONOR_COLOR = (0, 210, 80)   # BGR green


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_tube(path_value: str) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Mask tube not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        frame_indices = data["frame_indices"].astype(np.int32)
        masks = data["masks"].astype(np.uint8)
    if masks.ndim != 3 or len(frame_indices) != len(masks):
        raise ValueError(f"Invalid mask tube: {path}")
    order = np.argsort(frame_indices)
    return frame_indices[order], masks[order]


def open_video(path_value: str) -> Tuple[cv2.VideoCapture, float, int, int, int]:
    cap = cv2.VideoCapture(path_value)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path_value}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0 or nframes <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: {path_value}")
    return cap, fps, width, height, nframes


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask.astype(np.uint8)
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)


def bbox_xywh(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return x1, y1, x2 - x1, y2 - y1


def longest_visible_run(frame_indices: np.ndarray) -> Tuple[int, int]:
    if len(frame_indices) == 0:
        raise ValueError("Empty frame_indices")
    best_start = cur_start = int(frame_indices[0])
    best_end = cur_end = int(frame_indices[0])
    for idx in frame_indices[1:]:
        idx = int(idx)
        if idx == cur_end + 1:
            cur_end = idx
        else:
            if cur_end - cur_start > best_end - best_start:
                best_start, best_end = cur_start, cur_end
            cur_start = cur_end = idx
    if cur_end - cur_start > best_end - best_start:
        best_start, best_end = cur_start, cur_end
    return best_start, best_end


def make_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video: {path}")
    return writer


def annotate(frame: np.ndarray, label: str, frame_idx: int, mask_present: bool) -> np.ndarray:
    out = frame.copy()
    text = f"{label} | frame={frame_idx} | mask={'on' if mask_present else 'off'}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = frame.copy()
    active = mask > 0
    if active.any():
        color_img = np.empty_like(out)
        color_img[:, :] = color
        out[active] = cv2.addWeighted(out[active], 0.56, color_img[active], 0.44, 0)
        box = bbox_xywh(mask)
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), color, 2, cv2.LINE_AA)
    return out


def resolve_range(frame_indices: np.ndarray, mode: str) -> Tuple[int, int]:
    if mode == "full_visible_span":
        return int(frame_indices.min()), int(frame_indices.max())
    if mode == "longest_visible_run":
        return longest_visible_run(frame_indices)
    raise ValueError(f"Unknown range mode: {mode}")


def render_track(
    *,
    video_path: str,
    mask_path: str,
    out_mask_path: Path,
    out_overlay_path: Path,
    label: str,
    color: Tuple[int, int, int],
    range_mode: str,
) -> Dict[str, Any]:
    frame_indices, masks = load_tube(mask_path)
    cap, fps, width, height, nframes = open_video(video_path)
    start, end = resolve_range(frame_indices, range_mode)
    start = max(0, start)
    end = min(nframes - 1, end)

    mask_lookup = {int(idx): masks[pos] for pos, idx in enumerate(frame_indices)}
    raw_writer = make_writer(out_mask_path, fps, width, height)
    overlay_writer = make_writer(out_overlay_path, fps, width, height)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    rendered = 0
    try:
        for frame_idx in range(start, end + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Decode failed at frame={frame_idx}: {video_path}")
            source_mask = mask_lookup.get(frame_idx)
            present = source_mask is not None
            if source_mask is None:
                mask = np.zeros((height, width), dtype=np.uint8)
            else:
                mask = resize_mask(source_mask, height, width)
                mask = (mask > 0).astype(np.uint8)

            # Binary visualizer uses 0/255 and is intentionally visual-only;
            # compressed MP4 must not replace the original lossless NPZ.
            raw = np.zeros((height, width, 3), dtype=np.uint8)
            raw[mask > 0] = 255
            raw = annotate(raw, f"{label} binary mask", frame_idx, present)
            ov = annotate(overlay_mask(frame, mask, color), f"{label} overlay", frame_idx, present)
            raw_writer.write(raw)
            overlay_writer.write(ov)
            rendered += 1
    finally:
        cap.release()
        raw_writer.release()
        overlay_writer.release()

    return {
        "source_video_path": video_path,
        "source_mask_path": mask_path,
        "range_mode": range_mode,
        "start_frame": start,
        "end_frame": end,
        "rendered_frames": rendered,
        "fps": fps,
        "resolution": [width, height],
        "mask_raw_video": str(out_mask_path),
        "mask_overlay_video": str(out_overlay_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--range-mode",
        choices=["full_visible_span", "longest_visible_run"],
        default="full_visible_span",
        help="full_visible_span keeps tracking gaps visible; longest_visible_run makes shorter continuous previews.",
    )
    parser.add_argument("--skip-errors", action="store_true", help="Record a failing case and continue.")
    args = parser.parse_args()

    plan = read_json(args.plan)
    case_results: List[Dict[str, Any]] = []

    for case in plan.get("cases", []):
        case_id = str(case["case_id"])
        case_dir = args.out_dir / case_id
        item: Dict[str, Any] = {
            "case_id": case_id,
            "operation": case.get("operation"),
            "generator_route": case.get("generator_route"),
            "status": "rendered",
            "target": None,
            "donor": None,
            "errors": [],
        }
        try:
            target = case["target"]
            item["target"] = render_track(
                video_path=target["video_path"],
                mask_path=target["mask_tube_path"],
                out_mask_path=case_dir / "target_mask_raw.mp4",
                out_overlay_path=case_dir / "target_mask_overlay.mp4",
                label=f"TARGET {case_id}",
                color=TARGET_COLOR,
                range_mode=args.range_mode,
            )
            donor = case.get("donor")
            if donor:
                item["donor"] = render_track(
                    video_path=donor["video_path"],
                    mask_path=donor["mask_tube_path"],
                    out_mask_path=case_dir / "donor_mask_raw.mp4",
                    out_overlay_path=case_dir / "donor_mask_overlay.mp4",
                    label=f"DONOR {case_id}",
                    color=DONOR_COLOR,
                    range_mode=args.range_mode,
                )
        except Exception as exc:  # noqa: BLE001
            item["status"] = "render_error"
            item["errors"].append(f"{type(exc).__name__}: {exc}")
            if not args.skip_errors:
                raise
        case_results.append(item)
        print(f"{case_id}: {item['status']}")

    manifest = {
        "schema_version": "dataA_v1_mask_video_visualization",
        "source_plan": str(args.plan),
        "range_mode": args.range_mode,
        "case_count": len(case_results),
        "cases": case_results,
    }
    write_json(args.out_dir / "mask_video_manifest.json", manifest)
    print(f"wrote: {args.out_dir / 'mask_video_manifest.json'}")


if __name__ == "__main__":
    main()
