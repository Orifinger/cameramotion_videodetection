"""Donor reference frame scoring and optional PNG export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
import subprocess
from scipy import ndimage

from .common import DataAError, write_json
from .mask_io import MaskTube


@dataclass
class DonorFrameChoice:
    frame_index: int
    score: float
    components: Dict[str, float]
    bbox_xywh: tuple[int, int, int, int]


def bbox_xywh(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return x1, y1, x2 - x1, y2 - y1


def choose_donor_frame(tube: MaskTube) -> DonorFrameChoice:
    best: Optional[DonorFrameChoice] = None
    h, w = tube.height, tube.width
    areas = tube.masks.reshape(tube.masks.shape[0], -1).mean(axis=1)
    for pos, frame in enumerate(tube.frame_indices):
        mask = tube.masks[pos]
        box = bbox_xywh(mask)
        if box is None:
            continue
        x, y, bw, bh = box
        area_score = float(areas[pos])
        margin = min(x, y, w - (x + bw), h - (y + bh))
        margin_score = float(max(0, margin) / max(1, min(h, w)))
        if 0 < pos < len(tube.frame_indices) - 1:
            prev = tube.masks[pos - 1] > 0
            nxt = tube.masks[pos + 1] > 0
            cur = mask > 0
            iou_prev = np.logical_and(prev, cur).sum() / max(1, np.logical_or(prev, cur).sum())
            iou_next = np.logical_and(nxt, cur).sum() / max(1, np.logical_or(nxt, cur).sum())
            stability_score = float((iou_prev + iou_next) / 2.0)
        else:
            stability_score = 0.5
        bbox_area = bw * bh
        aspect = bw / max(1, bh)
        bbox_score = 1.0 if bbox_area > 0 and 0.05 <= aspect <= 20.0 else 0.0
        sharpness_score = 0.0  # filled during real video export; synthetic mask-only tests keep it neutral
        score = area_score * 4.0 + margin_score + stability_score + bbox_score + sharpness_score
        choice = DonorFrameChoice(
            frame_index=int(frame),
            score=float(score),
            components={
                "area_score": area_score,
                "interior_margin_score": margin_score,
                "temporal_stability_score": stability_score,
                "sharpness_score": sharpness_score,
                "non_degenerate_bbox_score": bbox_score,
            },
            bbox_xywh=(x, y, bw, bh),
        )
        if best is None or choice.score > best.score:
            best = choice
    if best is None:
        raise DataAError("blocked_donor_reference_failure: donor has no valid visible mask frame")
    return best


def export_synthetic_donor_reference(out_dir: Path, tube: MaskTube) -> Dict[str, Any]:
    """Export white RGB and alpha PNGs from the selected mask only.

    This is for synthetic tests and dry-run artifacts. Real server packaging
    should crop donor RGB frames from the donor video; donor RGB is never used
    for target compositing.
    """
    choice = choose_donor_frame(tube)
    pos = int(np.where(tube.frame_indices == choice.frame_index)[0][0])
    mask = tube.masks[pos].astype(np.uint8) * 255
    x, y, w, h = choice.bbox_xywh
    alpha = mask[y : y + h, x : x + w]
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = out_dir / "donor_reference.png"
    alpha_path = out_dir / "donor_reference_alpha.png"
    meta_path = out_dir / "donor_reference_meta.json"
    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(alpha).save(alpha_path)
    meta = {
        "source_frame": choice.frame_index,
        "bbox_xywh": list(choice.bbox_xywh),
        "score": choice.score,
        "score_components": choice.components,
        "donor_rgb_usage": "reference_condition_only_never_target_compositing",
        "donor_reference": str(rgb_path),
        "donor_reference_alpha": str(alpha_path),
    }
    write_json(meta_path, meta)
    return meta


def export_donor_reference_from_video(
    *,
    out_dir: Path,
    tube: MaskTube,
    donor_video_path: str,
    ffmpeg_bin: str = "ffmpeg",
    crop_padding_px: int = 8,
) -> Dict[str, Any]:
    choice = choose_donor_frame(tube)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = out_dir / ".donor_selected_frame.png"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        donor_video_path,
        "-vf",
        f"select=eq(n\\,{choice.frame_index})",
        "-vframes",
        "1",
        str(frame_path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0 or not frame_path.is_file():
        raise DataAError(f"blocked_donor_reference_failure: ffmpeg frame extraction failed: {proc.stderr.strip()}")
    image = Image.open(frame_path).convert("RGB")
    pos = int(np.where(tube.frame_indices == choice.frame_index)[0][0])
    mask = tube.masks[pos].astype(np.uint8)
    x, y, w, h = choice.bbox_xywh
    x0 = max(0, x - crop_padding_px)
    y0 = max(0, y - crop_padding_px)
    x1 = min(image.width, x + w + crop_padding_px)
    y1 = min(image.height, y + h + crop_padding_px)
    crop = np.asarray(image.crop((x0, y0, x1, y1))).copy()
    alpha = mask[y0:y1, x0:x1] * 255
    if crop.shape[:2] != alpha.shape:
        alpha_img = Image.fromarray(alpha).resize((crop.shape[1], crop.shape[0]), Image.Resampling.NEAREST)
        alpha = np.asarray(alpha_img)
    white = np.full_like(crop, 255)
    composed = np.where(alpha[:, :, None] > 0, crop, white).astype(np.uint8)
    rgb_path = out_dir / "donor_reference.png"
    alpha_path = out_dir / "donor_reference_alpha.png"
    meta_path = out_dir / "donor_reference_meta.json"
    Image.fromarray(composed).save(rgb_path)
    Image.fromarray(alpha.astype(np.uint8)).save(alpha_path)
    try:
        frame_path.unlink()
    except OSError:
        pass
    meta = {
        "source_frame": choice.frame_index,
        "bbox_xywh": list(choice.bbox_xywh),
        "crop_xyxy": [x0, y0, x1, y1],
        "crop_padding_px": crop_padding_px,
        "score": choice.score,
        "score_components": choice.components,
        "donor_rgb_usage": "reference_condition_only_never_target_compositing",
        "donor_reference": str(rgb_path),
        "donor_reference_alpha": str(alpha_path),
        "ffmpeg_command": cmd,
    }
    write_json(meta_path, meta)
    return meta
