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
import shutil
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


def resize_one(src: Path, dst: Path, size: tuple[int, int], overwrite: bool) -> str:
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
        # scaled uniformly in x and y.
        if im.mode not in {"RGB", "RGBA"}:
            im = im.convert("RGB")
        resized = im.resize(size, Image.Resampling.LANCZOS)
        save_kwargs: dict[str, Any] = {}
        if dst.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs.update({"quality": 95, "subsampling": 0})
        resized.save(dst, **save_kwargs)
    return "written"


def resize_tree(
    src_root: Path,
    dst_root: Path,
    *,
    size: tuple[int, int],
    overwrite: bool,
    copy_non_images: bool,
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

    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        if is_image(src):
            try:
                status = resize_one(src, dst, size, overwrite)
            except Exception as exc:
                stats["images_failed"] += 1
                print(f"[FAILED] {src} -> {dst}: {exc}")
                continue
            if status == "written":
                stats["images_written"] += 1
            else:
                stats["images_skipped"] += 1
        elif copy_non_images:
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
