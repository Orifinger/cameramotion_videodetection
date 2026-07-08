#!/usr/bin/env python3
"""Resize DataA frame images to a fixed resolution.

This is intended for DataA SFT frame folders such as:
  /tmp/cameramotion_det/dataA_v1/autolabel/dataa_vace_grounded_cot_frames

The script preserves the directory tree under a new root and optionally rewrites
SFT JSON image paths to point at the resized frame root.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_size(value: str) -> tuple[int, int]:
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("size must look like 672x384")
    left, right = value.lower().split("x", 1)
    try:
        width = int(left)
        height = int(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("size must contain integer width/height") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width/height must be positive")
    return width, height


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def resize_one(
    src: Path,
    dst: Path,
    size: tuple[int, int],
    overwrite: bool,
    *,
    png_compress_level: int,
    jpeg_quality: int,
) -> str:
    if dst.exists() and not overwrite:
        try:
            with Image.open(dst) as im:
                if im.size == size:
                    return "skipped"
        except Exception:
            pass

    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        # Exact whole-image affine resize: no crop, no padding. Normalized bbox
        # coordinates remain valid because the entire image coordinate system is
        # scaled in x and y together with the image.
        if im.mode not in {"RGB", "RGBA"}:
            im = im.convert("RGB")
        resized = im.resize(size, Image.Resampling.LANCZOS)
        save_kwargs: dict[str, Any] = {}
        suffix = dst.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            save_kwargs.update({"quality": jpeg_quality, "subsampling": 0})
        elif suffix == ".png":
            save_kwargs.update({"compress_level": png_compress_level})
        resized.save(dst, **save_kwargs)
    return "written"


def resize_image_job(job: tuple[str, str, tuple[int, int], bool, int, int]) -> tuple[str, str]:
    src_text, dst_text, size, overwrite, png_compress_level, jpeg_quality = job
    src = Path(src_text)
    dst = Path(dst_text)
    try:
        status = resize_one(
            src,
            dst,
            size,
            overwrite,
            png_compress_level=png_compress_level,
            jpeg_quality=jpeg_quality,
        )
        return status, ""
    except Exception as exc:
        return "failed", f"[FAILED] {src} -> {dst}: {exc}"


def resize_tree(
    src_root: Path,
    dst_root: Path,
    *,
    size: tuple[int, int],
    overwrite: bool,
    copy_non_images: bool,
    workers: int,
    progress_every: int,
    png_compress_level: int,
    jpeg_quality: int,
) -> dict[str, int]:
    if not src_root.exists():
        raise FileNotFoundError(f"source root does not exist: {src_root}")
    if not src_root.is_dir():
        raise NotADirectoryError(f"source root is not a directory: {src_root}")

    stats = {
        "images_written": 0,
        "images_skipped": 0,
        "images_failed": 0,
        "non_images_copied": 0,
        "non_images_skipped": 0,
    }

    image_jobs: list[tuple[str, str, tuple[int, int], bool, int, int]] = []
    non_image_pairs: list[tuple[Path, Path]] = []

    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        if is_image(src):
            image_jobs.append(
                (str(src), str(dst), size, overwrite, png_compress_level, jpeg_quality)
            )
        else:
            non_image_pairs.append((src, dst))

    print(f"Discovered {len(image_jobs)} images and {len(non_image_pairs)} non-image files.")
    if workers <= 0:
        workers = os.cpu_count() or 1
    workers = max(1, workers)
    print(f"Using workers: {workers}")

    processed = 0
    if workers == 1:
        for job in image_jobs:
            status, message = resize_image_job(job)
            if status == "written":
                stats["images_written"] += 1
            elif status == "skipped":
                stats["images_skipped"] += 1
            else:
                stats["images_failed"] += 1
                print(message)
            processed += 1
            if progress_every > 0 and processed % progress_every == 0:
                print(f"Processed {processed}/{len(image_jobs)} images...")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(resize_image_job, job) for job in image_jobs]
            for future in as_completed(futures):
                status, message = future.result()
                if status == "written":
                    stats["images_written"] += 1
                elif status == "skipped":
                    stats["images_skipped"] += 1
                else:
                    stats["images_failed"] += 1
                    print(message)
                processed += 1
                if progress_every > 0 and processed % progress_every == 0:
                    print(f"Processed {processed}/{len(image_jobs)} images...")

    for src, dst in non_image_pairs:
        if copy_non_images:
            if dst.exists() and not overwrite:
                stats["non_images_skipped"] += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            stats["non_images_copied"] += 1
        else:
            stats["non_images_skipped"] += 1

    return stats


def rewrite_sft_json(input_json: Path, output_json: Path, src_root: Path, dst_root: Path) -> int:
    src_norm = str(src_root).replace("\\", "/").rstrip("/")
    dst_norm = str(dst_root).replace("\\", "/").rstrip("/")
    data = json.load(input_json.open("r", encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected list SFT JSON: {input_json}")

    replaced = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        images = item.get("images")
        if not isinstance(images, list):
            continue
        new_images = []
        for image in images:
            image_text = str(image).replace("\\", "/")
            if image_text.startswith(src_norm):
                image_text = dst_norm + image_text[len(src_norm) :]
                replaced += 1
            new_images.append(image_text)
        item["images"] = new_images

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    return replaced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize DataA frame tree to a fixed size.")
    parser.add_argument(
        "--src-root",
        default="/tmp/cameramotion_det/dataA_v1/autolabel/dataa_vace_grounded_cot_frames",
        help="Original DataA frame root.",
    )
    parser.add_argument(
        "--dst-root",
        default="/tmp/cameramotion_det/dataA_v1/autolabel/dataa_vace_grounded_cot_frames_672x384",
        help="Output resized frame root.",
    )
    parser.add_argument("--size", type=parse_size, default=parse_size("672x384"))
    parser.add_argument("--overwrite", action="store_true", help="Recreate existing resized images.")
    parser.add_argument("--copy-non-images", action="store_true", help="Copy non-image files too.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel image resize workers. Use 0 for os.cpu_count().",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N processed images; 0 disables progress logs.",
    )
    parser.add_argument(
        "--png-compress-level",
        type=int,
        default=1,
        choices=range(0, 10),
        metavar="[0-9]",
        help="PNG compression level. Lower is faster and larger; default 1.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality when resizing jpg/jpeg files.",
    )
    parser.add_argument("--input-json", default=None, help="Optional SFT JSON to rewrite.")
    parser.add_argument("--output-json", default=None, help="Output path for rewritten SFT JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    stats = resize_tree(
        src_root,
        dst_root,
        size=args.size,
        overwrite=args.overwrite,
        copy_non_images=args.copy_non_images,
        workers=args.workers,
        progress_every=args.progress_every,
        png_compress_level=args.png_compress_level,
        jpeg_quality=args.jpeg_quality,
    )
    print("== resize stats ==")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"src_root: {src_root}")
    print(f"dst_root: {dst_root}")
    print(f"size: {args.size[0]}x{args.size[1]}")

    if args.input_json or args.output_json:
        if not args.input_json or not args.output_json:
            raise ValueError("--input-json and --output-json must be provided together")
        replaced = rewrite_sft_json(Path(args.input_json), Path(args.output_json), src_root, dst_root)
        print("== json rewrite ==")
        print(f"input_json: {args.input_json}")
        print(f"output_json: {args.output_json}")
        print(f"image_paths_replaced: {replaced}")


if __name__ == "__main__":
    main()
