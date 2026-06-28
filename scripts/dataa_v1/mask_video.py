"""VACE mask-video writing and round-trip validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import imageio.v2 as imageio
import numpy as np
import subprocess
import tempfile
from PIL import Image

from .common import DataAError


def mask_frames_to_rgb(masks: np.ndarray) -> list[np.ndarray]:
    if masks.ndim != 3:
        raise DataAError(f"masks must be [N,H,W], got {masks.shape}")
    binary = (masks > 0).astype(np.uint8) * 255
    return [np.repeat(frame[:, :, None], 3, axis=2) for frame in binary]


def _fallback_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".synthetic_roundtrip.npz")


def write_mask_video(
    path: Path,
    masks: np.ndarray,
    *,
    fps: int,
    macro_block_size: int | None = 1,
    allow_synthetic_npz_fallback: bool = False,
) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = mask_frames_to_rgb(masks)
    try:
        imageio.mimsave(path, frames, fps=fps, macro_block_size=macro_block_size, format="FFMPEG")
        backend = "imageio_ffmpeg"
    except Exception as exc:  # noqa: BLE001
        if allow_synthetic_npz_fallback:
            np.savez_compressed(_fallback_path(path), masks=(masks > 0).astype(np.uint8), fps=np.array([fps], dtype=np.int32))
            return {
                "path": str(path),
                "sidecar_path": str(_fallback_path(path)),
                "backend": "synthetic_npz_sidecar_no_real_mp4",
                "fps": fps,
                "frame_count": int(masks.shape[0]),
                "height": int(masks.shape[1]),
                "width": int(masks.shape[2]),
                "warning": "no ffmpeg writer available; this fallback is only valid for synthetic tests",
            }
        raise DataAError(f"cannot write mask video with imageio ffmpeg writer: {type(exc).__name__}: {exc}") from exc
    return {"path": str(path), "backend": backend, "fps": fps, "frame_count": int(masks.shape[0]), "height": int(masks.shape[1]), "width": int(masks.shape[2])}


def write_mask_video_ffmpeg(path: Path, masks: np.ndarray, *, fps: int, ffmpeg_bin: str = "ffmpeg") -> Dict[str, Any]:
    """Write a formal VACE mask video with lossless RGB encoding.

    No fallback is allowed here. If ffmpeg is missing or the codec fails, the
    caller gets a DataAError and must not invoke VACE.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = mask_frames_to_rgb(masks)
    with tempfile.TemporaryDirectory(prefix=".mask_frames_", dir=str(path.parent)) as temp_dir:
        temp = Path(temp_dir)
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(temp / f"{index:06d}.png")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(temp / "%06d.png"),
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-pix_fmt",
            "rgb24",
            str(path),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise DataAError(f"ffmpeg mask video write failed: {proc.stderr.strip()}")
    return {
        "path": str(path),
        "backend": "ffmpeg_libx264rgb_lossless",
        "fps": fps,
        "frame_count": int(masks.shape[0]),
        "height": int(masks.shape[1]),
        "width": int(masks.shape[2]),
        "command": cmd,
    }


def read_mask_video(path: Path, *, allow_synthetic_npz_fallback: bool = False) -> np.ndarray:
    fallback = _fallback_path(path)
    if allow_synthetic_npz_fallback and fallback.is_file():
        with np.load(fallback, allow_pickle=False) as archive:
            return archive["masks"].astype(np.uint8)
    if not path.is_file():
        raise DataAError(f"mask video does not exist: {path}")
    frames = []
    reader = imageio.get_reader(path, format="FFMPEG")
    try:
        for frame in reader:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                channel = arr
            else:
                channel = arr[:, :, 0]
            frames.append((channel >= 128).astype(np.uint8))
    finally:
        reader.close()
    if not frames:
        raise DataAError(f"mask video decoded zero frames: {path}")
    return np.stack(frames, axis=0)


def read_mask_video_ffmpeg(path: Path, *, ffmpeg_bin: str = "ffmpeg") -> np.ndarray:
    if not path.is_file():
        raise DataAError(f"mask video does not exist: {path}")
    with tempfile.TemporaryDirectory(prefix=".mask_decode_", dir=str(path.parent)) as temp_dir:
        temp = Path(temp_dir)
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(path),
            str(temp / "%06d.png"),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise DataAError(f"ffmpeg mask video decode failed: {proc.stderr.strip()}")
        frames = []
        for frame_path in sorted(temp.glob("*.png")):
            arr = np.asarray(Image.open(frame_path).convert("RGB"))
            frames.append((arr[:, :, 0] >= 128).astype(np.uint8))
    if not frames:
        raise DataAError(f"ffmpeg decoded zero mask frames: {path}")
    return np.stack(frames, axis=0)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def validate_mask_video_roundtrip(
    path: Path,
    m_gen: np.ndarray,
    *,
    allow_synthetic_npz_fallback: bool = False,
    ffmpeg_bin: str | None = None,
) -> Dict[str, Any]:
    if ffmpeg_bin:
        decoded = read_mask_video_ffmpeg(path, ffmpeg_bin=ffmpeg_bin)
        backend = "ffmpeg_decoded_video"
    else:
        decoded = read_mask_video(path, allow_synthetic_npz_fallback=allow_synthetic_npz_fallback)
        backend = "synthetic_npz_sidecar" if allow_synthetic_npz_fallback and _fallback_path(path).is_file() else "decoded_video"
    frame_count_match = decoded.shape[0] == m_gen.shape[0]
    shape_match = decoded.shape == m_gen.shape
    if frame_count_match and shape_match:
        ious = [_iou(decoded[i], m_gen[i]) for i in range(m_gen.shape[0])]
        pixel_equal = bool(np.array_equal(decoded.astype(np.uint8), (m_gen > 0).astype(np.uint8)))
    else:
        ious = [0.0]
        pixel_equal = False
    report = {
        "frame_count_match": frame_count_match,
        "shape_match": shape_match,
        "thresholded_iou_mean": float(np.mean(ious)),
        "thresholded_iou_min": float(np.min(ious)),
        "pixel_equal_after_threshold": pixel_equal,
        "backend": backend,
    }
    report["status"] = "ok" if all(
        [
            report["frame_count_match"],
            report["shape_match"],
            report["thresholded_iou_mean"] == 1.0,
            report["thresholded_iou_min"] == 1.0,
            report["pixel_equal_after_threshold"],
        ]
    ) else "blocked_mask_video_mismatch"
    return report
