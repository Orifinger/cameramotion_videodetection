"""Full-video Real/Fake reassembly for Data A v1."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

from .common import DataAError
from .compositing import composite_videos
from .media_io import VideoMeta, crop_video_frames, ffprobe_video, fps_arg


def _run(cmd: list[str], label: str) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise DataAError(f"{label} failed: {proc.stderr.strip()}")


def _encode_source_segment(
    *,
    source_video: Path,
    out_path: Path,
    start_time: float,
    duration: float,
    fps: float,
    height: int,
    width: int,
    ffmpeg_bin: str,
) -> Path | None:
    if duration <= 1e-6:
        return None
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{start_time:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(source_video),
        "-vf",
        f"scale={width}:{height}:flags=lanczos,fps={fps_arg(fps)},setpts=PTS-STARTPTS",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    _run(cmd, "full-video source segment encode")
    return out_path


def _normalize_video(
    *,
    source_video: Path,
    out_path: Path,
    fps: float,
    height: int,
    width: int,
    ffmpeg_bin: str,
    duration: float | None = None,
) -> Path:
    cmd = [ffmpeg_bin, "-y", "-i", str(source_video)]
    if duration is not None:
        cmd.extend(["-t", f"{duration:.6f}"])
    cmd.extend(
        [
            "-vf",
            f"scale={width}:{height}:flags=lanczos,fps={fps_arg(fps)},setpts=PTS-STARTPTS",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
    )
    _run(cmd, "full-video normalization")
    return out_path


def _concat_videos(*, parts: list[Path], out_path: Path, ffmpeg_bin: str) -> None:
    if not parts:
        raise DataAError("blocked_full_video_reassembly_failure: no parts to concatenate")
    list_path = out_path.parent / "full_fake_concat.txt"
    with list_path.open("w", encoding="utf-8") as handle:
        for part in parts:
            escaped = str(part.resolve()).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-fflags",
        "+genpts",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    _run(cmd, "full-video concat")


def _shape(meta: VideoMeta) -> Dict[str, Any]:
    return {"fps": meta.fps, "frame_count": meta.frame_count, "height": meta.height, "width": meta.width}


def _same_fps_shape(a: VideoMeta, b: VideoMeta) -> bool:
    return round(a.fps, 6) == round(b.fps, 6) and a.height == b.height and a.width == b.width


def repair_one_frame_full_pair_mismatch(
    *,
    full_real: Path,
    full_fake: Path,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    execute: bool = True,
) -> Dict[str, Any]:
    """Trim the longer full-video pair member when ffmpeg rounding leaves a small end skew."""

    real_meta = ffprobe_video(full_real, ffprobe_bin=ffprobe_bin)
    fake_meta = ffprobe_video(full_fake, ffprobe_bin=ffprobe_bin)
    if (round(real_meta.fps, 6), real_meta.frame_count, real_meta.height, real_meta.width) == (
        round(fake_meta.fps, 6),
        fake_meta.frame_count,
        fake_meta.height,
        fake_meta.width,
    ):
        return {
            "status": "already_aligned",
            "full_real": _shape(real_meta),
            "full_fake": _shape(fake_meta),
        }
    if not _same_fps_shape(real_meta, fake_meta):
        return {
            "status": "not_repairable",
            "reason": "fps_or_shape_mismatch",
            "full_real": _shape(real_meta),
            "full_fake": _shape(fake_meta),
        }
    diff = real_meta.frame_count - fake_meta.frame_count
    frame_skew = abs(diff)
    if frame_skew not in {1, 2}:
        return {
            "status": "not_repairable",
            "reason": f"frame_count_diff_exceeds_two:{diff}",
            "full_real": _shape(real_meta),
            "full_fake": _shape(fake_meta),
        }

    target_count = min(real_meta.frame_count, fake_meta.frame_count)
    trim_path = full_real if diff > 0 else full_fake
    trim_label = "full_real" if diff > 0 else "full_fake"
    if not execute:
        return {
            "status": "would_repair",
            "action": f"trim_{trim_label}_last_{frame_skew}_frames",
            "target_frame_count": target_count,
            "before": {"full_real": _shape(real_meta), "full_fake": _shape(fake_meta)},
        }
    tmp_path = trim_path.with_name(f".{trim_path.stem}.trimmed_end_skew.{trim_path.suffix.lstrip('.')}")
    try:
        crop_video_frames(
            source_video=trim_path,
            out_path=tmp_path,
            frame_count=target_count,
            fps=real_meta.fps,
            ffmpeg_bin=ffmpeg_bin,
        )
        tmp_path.replace(trim_path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    repaired_real = ffprobe_video(full_real, ffprobe_bin=ffprobe_bin)
    repaired_fake = ffprobe_video(full_fake, ffprobe_bin=ffprobe_bin)
    _assert_full_pair(repaired_real, repaired_fake)
    return {
        "status": "repaired",
        "action": f"trimmed_{trim_label}_last_{frame_skew}_frames",
        "target_frame_count": target_count,
        "before": {"full_real": _shape(real_meta), "full_fake": _shape(fake_meta)},
        "after": {"full_real": _shape(repaired_real), "full_fake": _shape(repaired_fake)},
    }


def _assert_full_pair(real: VideoMeta, fake: VideoMeta) -> None:
    real_sig = (round(real.fps, 6), real.frame_count, real.height, real.width)
    fake_sig = (round(fake.fps, 6), fake.frame_count, fake.height, fake.width)
    if real_sig != fake_sig:
        raise DataAError(f"blocked_full_video_reassembly_mismatch: full_real={_shape(real)} full_fake={_shape(fake)}")


def reassemble_full_video_pair(
    *,
    manifest: Mapping[str, Any],
    attempt_dir: Path,
    generated_raw_video: Path,
    final_generated_video: Path,
    ffmpeg_bin: str,
    ffprobe_bin: str,
) -> Dict[str, Any]:
    source_clip = manifest.get("source_clip") or {}
    native = source_clip.get("native") or {}
    canonical = source_clip.get("canonical") or {}
    source_video_path = source_clip.get("source_video_path")
    if not source_video_path:
        raise DataAError("blocked_full_video_reassembly_failure: source_video_path missing from manifest")
    source_video = Path(str(source_video_path))
    if not source_video.is_file():
        raise DataAError(f"blocked_full_video_reassembly_failure: source video missing: {source_video}")

    source_start_time = float(native.get("start_time_sec", float(native.get("start_frame", 0)) / float(native.get("source_fps", 1))))
    source_end_time = float(native.get("end_time_sec", source_start_time + float(canonical.get("source_duration_sec", 0))))
    generation_fps = float(canonical.get("generation_fps") or canonical.get("fps") or 16.0)
    full_meta = ffprobe_video(source_video, ffprobe_bin=ffprobe_bin)
    height = int(canonical.get("height") or full_meta.height)
    width = int(canonical.get("width") or full_meta.width)
    valid_frame_count = int(canonical.get("valid_frame_count") or canonical.get("frame_count") or 0)
    frame_count = int(canonical.get("frame_count") or valid_frame_count)

    full_real = attempt_dir / "full_real.mp4"
    full_fake = attempt_dir / "full_fake.mp4"
    full_real_norm = _normalize_video(
        source_video=source_video,
        out_path=full_real,
        fps=full_meta.fps,
        height=height,
        width=width,
        ffmpeg_bin=ffmpeg_bin,
    )

    alpha_npz = Path(str((manifest.get("mask_layers") or {}).get("M_alpha") or attempt_dir / "target_mask_alpha.npz"))
    source_clip_path = Path(str(source_clip.get("source_clip_path") or attempt_dir / "source_clip.mp4"))
    if not alpha_npz.is_file() or not source_clip_path.is_file() or not generated_raw_video.is_file():
        raise DataAError(
            "blocked_full_video_reassembly_failure: missing source_clip/generated_raw/M_alpha for compositing"
        )

    edited_padded = attempt_dir / "edited_segment_padded.mp4"
    composite = composite_videos(
        real_video=source_clip_path,
        generated_video=generated_raw_video,
        alpha_npz=alpha_npz,
        out_path=edited_padded,
        fps=generation_fps,
        ffmpeg_bin=ffmpeg_bin,
    )
    if valid_frame_count and frame_count and valid_frame_count < frame_count:
        edited_segment = attempt_dir / "edited_segment.mp4"
        crop = crop_video_frames(
            source_video=edited_padded,
            out_path=edited_segment,
            frame_count=valid_frame_count,
            fps=generation_fps,
            ffmpeg_bin=ffmpeg_bin,
        )
    else:
        edited_segment = edited_padded
        crop = {"status": "not_required", "path": str(edited_segment)}

    with tempfile.TemporaryDirectory(prefix=".full_video_", dir=str(attempt_dir)) as temp_dir:
        temp = Path(temp_dir)
        parts: list[Path] = []
        prefix = _encode_source_segment(
            source_video=source_video,
            out_path=temp / "prefix.mp4",
            start_time=0.0,
            duration=max(0.0, source_start_time),
            fps=full_meta.fps,
            height=height,
            width=width,
            ffmpeg_bin=ffmpeg_bin,
        )
        if prefix is not None:
            parts.append(prefix)
        edit_norm = _normalize_video(
            source_video=edited_segment,
            out_path=temp / "edited.mp4",
            fps=full_meta.fps,
            height=height,
            width=width,
            ffmpeg_bin=ffmpeg_bin,
            duration=max(0.0, source_end_time - source_start_time),
        )
        parts.append(edit_norm)
        suffix = _encode_source_segment(
            source_video=source_video,
            out_path=temp / "suffix.mp4",
            start_time=source_end_time,
            duration=max(0.0, full_meta.duration - source_end_time),
            fps=full_meta.fps,
            height=height,
            width=width,
            ffmpeg_bin=ffmpeg_bin,
        )
        if suffix is not None:
            parts.append(suffix)
        _concat_videos(parts=parts, out_path=full_fake, ffmpeg_bin=ffmpeg_bin)

    real_meta = ffprobe_video(full_real_norm, ffprobe_bin=ffprobe_bin)
    fake_meta = ffprobe_video(full_fake, ffprobe_bin=ffprobe_bin)
    alignment_repair = repair_one_frame_full_pair_mismatch(
        full_real=full_real_norm,
        full_fake=full_fake,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
    )
    real_meta = ffprobe_video(full_real_norm, ffprobe_bin=ffprobe_bin)
    fake_meta = ffprobe_video(full_fake, ffprobe_bin=ffprobe_bin)
    _assert_full_pair(real_meta, fake_meta)
    return {
        "status": "ok",
        "full_real_path": str(full_real_norm),
        "full_fake_path": str(full_fake),
        "edited_segment_path": str(edited_segment),
        "edited_segment_padded_path": str(edited_padded),
        "source_time_range_sec": [source_start_time, source_end_time],
        "generation_fps": generation_fps,
        "full_fps": full_meta.fps,
        "shape": _shape(real_meta),
        "alignment_repair": alignment_repair,
        "compositing": composite,
        "crop": crop,
        "donor_rgb_used": False,
    }
