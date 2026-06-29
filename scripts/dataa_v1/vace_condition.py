"""Build the masked source-video condition required by VACE MV2V."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from .common import DataAError
from .media_io import VideoMeta, assert_video_compatible, ffprobe_video


VACE_MISSING_RGB = 127


def resize_masks_nearest(masks: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Resize [N,H,W] binary masks without anti-aliasing."""
    if masks.ndim != 3:
        raise DataAError(f"masks must be [N,H,W], got {masks.shape}")
    if height <= 0 or width <= 0:
        raise DataAError(f"invalid target mask shape: {width}x{height}")
    count, src_h, src_w = masks.shape
    if (src_h, src_w) == (height, width):
        return (masks > 0).astype(np.uint8)
    y = np.minimum(((np.arange(height) + 0.5) * src_h / height).astype(np.int64), src_h - 1)
    x = np.minimum(((np.arange(width) + 0.5) * src_w / width).astype(np.int64), src_w - 1)
    resized = masks[:, y, :][:, :, x]
    if resized.shape != (count, height, width):
        raise DataAError(f"mask resize shape mismatch: {resized.shape}")
    return (resized > 0).astype(np.uint8)


def _require_mask_matches_video(masks: np.ndarray, meta: VideoMeta) -> None:
    expected = (meta.frame_count, meta.height, meta.width)
    if masks.shape != expected:
        raise DataAError(f"VACE source/mask mismatch: masks={masks.shape}, video={expected}")


def export_vace_condition_video(
    *,
    source_clip: Path,
    masks: np.ndarray,
    out_path: Path,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> Dict[str, Any]:
    """Write VACE `src_video`: original RGB outside mask and exact gray 127 inside.

    The implementation streams raw RGB frames between two ffmpeg processes so
    the missing-area value remains exactly 127 after lossless RGB encoding.
    """
    source_meta = ffprobe_video(source_clip, ffprobe_bin=ffprobe_bin)
    _require_mask_matches_video(masks, source_meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_bytes = source_meta.width * source_meta.height * 3
    decode_cmd = [
        ffmpeg_bin, "-v", "error", "-i", str(source_clip), "-an",
        "-frames:v", str(source_meta.frame_count), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    encode_cmd = [
        ffmpeg_bin, "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{source_meta.width}x{source_meta.height}", "-framerate", f"{source_meta.fps:.8f}",
        "-i", "pipe:0", "-frames:v", str(source_meta.frame_count), "-r", f"{source_meta.fps:.8f}",
        "-fps_mode", "cfr", "-an", "-c:v", "libx264rgb", "-crf", "0", "-pix_fmt", "rgb24", str(out_path),
    ]

    import subprocess

    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert decoder.stdout is not None
        assert encoder.stdin is not None
        for frame_index in range(source_meta.frame_count):
            raw = decoder.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                raise DataAError(f"VACE condition decode ended at frame {frame_index}; expected {source_meta.frame_count} frames")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(source_meta.height, source_meta.width, 3).copy()
            frame[masks[frame_index] > 0] = VACE_MISSING_RGB
            encoder.stdin.write(frame.tobytes())
    except Exception:
        decoder.terminate()
        encoder.terminate()
        raise
    finally:
        if encoder.stdin is not None and not encoder.stdin.closed:
            encoder.stdin.close()

    decode_error = (decoder.stderr.read() if decoder.stderr is not None else b"").decode("utf-8", errors="replace")
    encode_error = (encoder.stderr.read() if encoder.stderr is not None else b"").decode("utf-8", errors="replace")
    if decoder.wait() != 0:
        raise DataAError(f"VACE condition source decode failed: {decode_error.strip()}")
    if encoder.wait() != 0:
        raise DataAError(f"VACE condition encode failed: {encode_error.strip()}")

    condition_meta = ffprobe_video(out_path, ffprobe_bin=ffprobe_bin)
    assert_video_compatible(source_meta, condition_meta)
    return {
        "path": str(out_path),
        "source_clip": str(source_clip),
        "mask_semantics": "white_generate_black_retain",
        "missing_rgb_value": VACE_MISSING_RGB,
        "encoding": "libx264rgb_crf0_rgb24",
        "source_meta": source_meta.__dict__,
        "condition_meta": condition_meta.__dict__,
        "mask_foreground_ratio": float((masks > 0).mean()),
        "decode_command": decode_cmd,
        "encode_command": encode_cmd,
    }
