"""M_alpha compositing helpers.

Donor RGB is intentionally not accepted by any function in this module.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

from .common import DataAError
from .mask_io import load_mask_tube


def composite_arrays(real: np.ndarray, generated: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if real.shape != generated.shape:
        raise DataAError(f"real/generated shape mismatch: {real.shape} vs {generated.shape}")
    if alpha.ndim != 3 or real.ndim != 4:
        raise DataAError(f"expected real/generated [N,H,W,C] and alpha [N,H,W], got {real.shape}, {alpha.shape}")
    if alpha.shape != real.shape[:3]:
        raise DataAError(f"alpha shape mismatch: {alpha.shape} vs {real.shape[:3]}")
    alpha_f = np.clip(alpha.astype(np.float32), 0.0, 1.0)[:, :, :, None]
    out = alpha_f * generated.astype(np.float32) + (1.0 - alpha_f) * real.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _decode_video_to_frames(path: Path, out_dir: Path, *, ffmpeg_bin: str) -> None:
    cmd = [ffmpeg_bin, "-y", "-i", str(path), str(out_dir / "%06d.png")]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise DataAError(f"ffmpeg decode failed for {path}: {proc.stderr.strip()}")


def _encode_frames_to_video(frame_dir: Path, out_path: Path, *, fps: int, ffmpeg_bin: str) -> None:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "%06d.png"),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise DataAError(f"ffmpeg encode failed for {out_path}: {proc.stderr.strip()}")


def composite_videos(
    *,
    real_video: Path,
    generated_video: Path,
    alpha_npz: Path,
    out_path: Path,
    fps: int,
    ffmpeg_bin: str = "ffmpeg",
) -> Dict[str, Any]:
    alpha_tube = load_mask_tube(alpha_npz)
    alpha = alpha_tube.masks.astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".composite_", dir=str(out_path.parent)) as temp_dir:
        temp = Path(temp_dir)
        real_dir = temp / "real"
        gen_dir = temp / "generated"
        out_dir = temp / "out"
        real_dir.mkdir()
        gen_dir.mkdir()
        out_dir.mkdir()
        _decode_video_to_frames(real_video, real_dir, ffmpeg_bin=ffmpeg_bin)
        _decode_video_to_frames(generated_video, gen_dir, ffmpeg_bin=ffmpeg_bin)
        real_frames = sorted(real_dir.glob("*.png"))
        gen_frames = sorted(gen_dir.glob("*.png"))
        if len(real_frames) != len(gen_frames) or len(real_frames) != alpha.shape[0]:
            raise DataAError(
                f"compositing frame count mismatch: real={len(real_frames)} generated={len(gen_frames)} alpha={alpha.shape[0]}"
            )
        for index, (real_frame, gen_frame) in enumerate(zip(real_frames, gen_frames)):
            real = np.asarray(Image.open(real_frame).convert("RGB"))
            gen = np.asarray(Image.open(gen_frame).convert("RGB"))
            frame_alpha = alpha[index]
            if frame_alpha.shape != real.shape[:2]:
                frame_alpha = np.asarray(Image.fromarray(frame_alpha).resize((real.shape[1], real.shape[0]), Image.Resampling.BILINEAR))
            out = composite_arrays(real[None, ...], gen[None, ...], frame_alpha[None, ...])[0]
            Image.fromarray(out).save(out_dir / f"{index:06d}.png")
        _encode_frames_to_video(out_dir, out_path, fps=fps, ffmpeg_bin=ffmpeg_bin)
    return {
        "fake_pair_render": str(out_path),
        "formula": "Fake = M_alpha * Generated + (1 - M_alpha) * Real",
        "donor_rgb_used": False,
    }
