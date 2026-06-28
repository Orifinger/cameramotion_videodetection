"""FFprobe/FFmpeg helpers for production media packaging."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .common import DataAError


@dataclass(frozen=True)
class VideoMeta:
    path: str
    fps: float
    frame_count: int
    duration: float
    height: int
    width: int
    has_audio: bool
    codec_name: Optional[str] = None


def _parse_rate(value: str) -> float:
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        if den_f == 0:
            return 0.0
        return float(num) / den_f
    return float(value)


def ffprobe_video(path: Path, *, ffprobe_bin: str = "ffprobe") -> VideoMeta:
    if not path.is_file():
        raise DataAError(f"source video path does not exist: {path}")
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-print_format",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise DataAError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise DataAError(f"ffprobe returned invalid JSON for {path}: {exc}") from exc
    video_stream = next((s for s in payload.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video_stream:
        raise DataAError(f"ffprobe found no video stream: {path}")
    audio_stream = any(s.get("codec_type") == "audio" for s in payload.get("streams", []))
    fps = _parse_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1")
    if fps <= 0:
        raise DataAError(f"ffprobe could not parse reliable fps for {path}")
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise DataAError(f"ffprobe returned invalid resolution for {path}: {width}x{height}")
    frame_count = int(video_stream.get("nb_frames") or 0)
    duration = float(video_stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    if frame_count <= 0 and duration > 0:
        frame_count = int(round(duration * fps))
    if frame_count <= 0 or duration <= 0:
        raise DataAError(f"ffprobe returned invalid frame_count/duration for {path}")
    return VideoMeta(
        path=str(path),
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        height=height,
        width=width,
        has_audio=audio_stream,
        codec_name=video_stream.get("codec_name"),
    )


def export_source_real_raw(
    *,
    source_video: Path,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    source_fps: float,
    ffmpeg_bin: str = "ffmpeg",
) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = start_frame / source_fps
    duration = (end_frame - start_frame + 1) / source_fps
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{start_time:.6f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration:.6f}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise DataAError(f"ffmpeg source_real_raw export failed: {proc.stderr.strip()}")
    return {"path": str(out_path), "command": cmd}


def export_canonical_source_clip(
    *,
    source_video: Path,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    source_fps: float,
    canonical_fps: int,
    frame_count: int,
    height: int,
    width: int,
    ffmpeg_bin: str = "ffmpeg",
    audio_policy: str = "drop",
) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = start_frame / source_fps
    duration = (end_frame - start_frame + 1) / source_fps
    scale = f"scale={width}:{height}:flags=lanczos,fps={canonical_fps},trim=end_frame={frame_count},setpts=PTS-STARTPTS"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{start_time:.6f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration:.6f}",
        "-vf",
        scale,
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise DataAError(f"ffmpeg canonical source_clip export failed: {proc.stderr.strip()}")
    return {"path": str(out_path), "command": cmd, "audio_policy": audio_policy}


def assert_video_compatible(a: VideoMeta, b: VideoMeta) -> None:
    if round(a.fps, 6) != round(b.fps, 6) or a.frame_count != b.frame_count or a.height != b.height or a.width != b.width:
        raise DataAError(
            f"video compatibility mismatch: {a.path} fps={a.fps} frames={a.frame_count} shape={a.width}x{a.height}; "
            f"{b.path} fps={b.fps} frames={b.frame_count} shape={b.width}x{b.height}"
        )
