#!/usr/bin/env python3
"""Render a visual review pack for a frozen/draft Data A generation plan.

This script does NOT generate videos. It helps finish the current planning stage by:
- rendering target and donor contact sheets with SAM3 masks and rectangular bboxes;
- selecting a donor reference-frame candidate by mask area/border score;
- exporting an editable case-review scaffold with pending clip/edit/prompt decisions.

Example:
  python scripts/prepare_dataa_v1_review_pack.py \
    --plan res/dataA_v1/plans/vace14b_stage1_quota_plan.json \
    --out-dir res/dataA_v1/review/vace14b_stage1
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


TARGET_COLOR = (0, 60, 255)  # BGR red-orange
DONOR_COLOR = (0, 210, 80)  # BGR green


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_tube(path_value: str) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        frame_indices = payload["frame_indices"].astype(np.int32)
        masks = payload["masks"].astype(np.uint8)
    if masks.ndim != 3 or len(frame_indices) != len(masks):
        raise ValueError(f"Invalid mask tube schema: {path}")
    return frame_indices, masks


def open_video(path_value: str) -> Tuple[cv2.VideoCapture, float, int]:
    cap = cv2.VideoCapture(path_value)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path_value}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        fps = 25.0
    return cap, fps, nframes


def decode_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to decode frame {frame_idx}")
    return frame


def resize_mask(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape == (h, w):
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def mask_bbox_xywh(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return x1, y1, x2 - x1, y2 - y1


def mask_score(mask: np.ndarray) -> float:
    h, w = mask.shape
    area = float((mask > 0).sum()) / max(1.0, h * w)
    border = np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1])).astype(bool).mean()
    bbox = mask_bbox_xywh(mask)
    if bbox is None:
        return -1.0
    _, _, bw, bh = bbox
    compact = min(1.0, (bw * bh) / max(1.0, h * w) * 8.0)
    return area * 1.0 + compact * 0.15 - border * 0.25


def evenly_spaced_positions(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def overlay(frame: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], label: str) -> np.ndarray:
    out = frame.copy()
    m = resize_mask(mask, out.shape[:2]) > 0
    tint = np.zeros_like(out)
    tint[:, :] = color
    out[m] = cv2.addWeighted(out[m], 0.55, tint[m], 0.45, 0)
    bbox = mask_bbox_xywh(m.astype(np.uint8))
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), color, 2, lineType=cv2.LINE_AA)
        cv2.putText(out, label, (max(4, x), max(22, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def annotate(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 31), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def fit(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def make_grid(images: Sequence[np.ndarray], cols: int = 2) -> np.ndarray:
    if not images:
        raise ValueError("No images for grid")
    rows = math.ceil(len(images) / cols)
    cell_h = max(x.shape[0] for x in images)
    cell_w = max(x.shape[1] for x in images)
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 30, dtype=np.uint8)
    for i, image in enumerate(images):
        r, c = divmod(i, cols)
        y, x = r * cell_h, c * cell_w
        canvas[y:y + image.shape[0], x:x + image.shape[1]] = image
    return canvas


def longest_visible_run(frame_indices: np.ndarray) -> Tuple[int, int]:
    if len(frame_indices) == 0:
        return 0, 0
    best_start = cur_start = int(frame_indices[0])
    best_end = cur_end = int(frame_indices[0])
    for f in frame_indices[1:]:
        f = int(f)
        if f <= cur_end + 1:
            cur_end = f
        else:
            if cur_end - cur_start > best_end - best_start:
                best_start, best_end = cur_start, cur_end
            cur_start = cur_end = f
    if cur_end - cur_start > best_end - best_start:
        best_start, best_end = cur_start, cur_end
    return best_start, best_end


def make_reference_preview(frame: np.ndarray, mask: np.ndarray, path: Path) -> Dict[str, Any]:
    mask = resize_mask(mask, frame.shape[:2]) > 0
    bbox = mask_bbox_xywh(mask.astype(np.uint8))
    if bbox is None:
        raise ValueError("Cannot create donor preview from empty mask")
    x, y, w, h = bbox
    pad = max(8, round(max(w, h) * 0.12))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
    crop = frame[y1:y2, x1:x2].copy()
    alpha = (mask[y1:y2, x1:x2].astype(np.uint8) * 255)
    white = np.full_like(crop, 255)
    white[alpha > 0] = crop[alpha > 0]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), white)
    cv2.imwrite(str(path.with_name(path.stem + "_alpha.png")), alpha)
    return {
        "crop_xywh": [x1, y1, x2 - x1, y2 - y1],
        "image_path": str(path),
        "alpha_path": str(path.with_name(path.stem + "_alpha.png")),
    }


def render_track_preview(
    video_path: str,
    mask_path: str,
    out_path: Path,
    title: str,
    color: Tuple[int, int, int],
    frames_per_case: int,
    max_side: int,
) -> Dict[str, Any]:
    frame_indices, masks = load_tube(mask_path)
    cap, fps, _ = open_video(video_path)
    positions = evenly_spaced_positions(len(frame_indices), frames_per_case)
    tiles: List[np.ndarray] = []
    for pos in positions:
        frame_idx = int(frame_indices[pos])
        frame = decode_frame(cap, frame_idx)
        image = overlay(frame, masks[pos], color, f"frame={frame_idx}")
        image = annotate(fit(image, max_side), title)
        tiles.append(image)
    cap.release()
    grid = make_grid(tiles, cols=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    scores = np.asarray([mask_score(m) for m in masks], dtype=np.float32)
    best_pos = int(scores.argmax())
    run_start, run_end = longest_visible_run(frame_indices)
    return {
        "preview_path": str(out_path),
        "fps": fps,
        "visible_frame_range": [int(frame_indices.min()), int(frame_indices.max())],
        "longest_visible_run_frames": [run_start, run_end],
        "longest_visible_run_seconds": [round(run_start / fps, 3), round(run_end / fps, 3)],
        "best_reference_frame": int(frame_indices[best_pos]),
        "best_reference_position": best_pos,
        "best_reference_score": round(float(scores[best_pos]), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--frames-per-case", type=int, default=4)
    parser.add_argument("--max-side", type=int, default=720)
    parser.add_argument("--skip-missing", action="store_true", help="Record errors and continue instead of stopping.")
    args = parser.parse_args()

    plan = read_json(args.plan)
    cases = plan.get("cases", [])
    results: List[Dict[str, Any]] = []
    markdown: List[str] = ["# Data A v1 case visual review index", ""]

    for case in cases:
        case_id = case["case_id"]
        case_dir = args.out_dir / case_id
        result: Dict[str, Any] = {
            "case_id": case_id,
            "operation": case.get("operation"),
            "generator_route": case.get("generator_route"),
            "status": "needs_visual_review",
            "target": case.get("target"),
            "donor": case.get("donor"),
            "errors": [],
        }
        try:
            target = case["target"]
            target_preview = render_track_preview(
                target["video_path"], target["mask_tube_path"], case_dir / "target_contact_sheet.jpg",
                f"TARGET | {case_id} | {case['operation']} | {target.get('canonical_concept', '')}",
                TARGET_COLOR, args.frames_per_case, args.max_side,
            )
            result["target_review"] = target_preview
            result["suggested_target_clip"] = {
                "status": "pending_manual_confirm",
                "visible_run_seconds": target_preview["longest_visible_run_seconds"],
                "suggestion": "Choose a 3–5 second no-cut subclip within this visible run.",
            }

            donor = case.get("donor")
            if donor:
                donor_preview = render_track_preview(
                    donor["video_path"], donor["mask_tube_path"], case_dir / "donor_contact_sheet.jpg",
                    f"DONOR | {case_id} | {donor.get('canonical_concept', '')}",
                    DONOR_COLOR, args.frames_per_case, args.max_side,
                )
                result["donor_review"] = donor_preview
                frame_indices, masks = load_tube(donor["mask_tube_path"])
                best_pos = int(donor_preview["best_reference_position"])
                cap, _, _ = open_video(donor["video_path"])
                best_frame = decode_frame(cap, int(frame_indices[best_pos]))
                cap.release()
                result["donor_reference_candidate"] = make_reference_preview(
                    best_frame, masks[best_pos], case_dir / "donor_reference_candidate.png"
                )
                result["donor_reference_candidate"]["frame_index"] = int(frame_indices[best_pos])

            result["edit_spec"] = {
                "source_description": target.get("canonical_concept"),
                "target_description": None,
                "prompt": None,
                "review_decision": "pending",
                "review_notes": None,
            }
            result["evidence_bbox"] = {
                "source": "SAM3 mask-derived per-frame rectangular bbox",
                "target_bbox_tube_field": target.get("bbox_tube_xywh"),
                "final_output_policy": "Preserve per-frame bbox tube; derive union/normalized [0,1000] bbox after the final target clip is frozen.",
            }
        except Exception as exc:  # noqa: BLE001
            result["status"] = "review_pack_error"
            result["errors"].append(f"{type(exc).__name__}: {exc}")
            if not args.skip_missing:
                raise
        results.append(result)

        markdown.extend([
            f"## {case_id}",
            f"- operation: `{case.get('operation')}`",
            f"- route: `{case.get('generator_route')}`",
            f"- target: `{case.get('target', {}).get('video_id')}` / `{case.get('target', {}).get('canonical_concept')}`",
            f"- donor: `{case.get('donor', {}).get('video_id') if case.get('donor') else 'none'}` / `{case.get('donor', {}).get('canonical_concept') if case.get('donor') else 'none'}`",
            f"- review record: `{case_id}/`",
            "",
        ])

    scaffold = {
        "schema_version": "dataA_v1_case_review_scaffold",
        "source_plan": str(args.plan),
        "case_count": len(results),
        "cases": results,
    }
    write_json(args.out_dir / "case_review_scaffold.json", scaffold)
    (args.out_dir / "review_index.md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"wrote review pack: {args.out_dir}")
    print(f"cases: {len(results)}")


if __name__ == "__main__":
    main()
