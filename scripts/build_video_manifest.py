#!/usr/bin/env python3
"""Build one deterministic JSON manifest for CameraBench videos.

The manifest is the only input list for the Qwen object-proposal stage.
It deliberately contains filesystem metadata only; camera-motion annotations are
not injected into the Qwen prompt in this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def parse_extensions(raw: str) -> set[str]:
    result = set()
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        result.add(item if item.startswith(".") else f".{item}")
    if not result:
        raise ValueError("At least one video extension is required.")
    return result


def stable_video_id(relative_path: str) -> str:
    """Filesystem-independent ID; source relative_path remains separately stored."""
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:16]
    return f"vid_{digest}"


def iter_videos(video_root: Path, extensions: set[str]) -> Iterable[dict]:
    for path in sorted(video_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        relative_path = path.relative_to(video_root).as_posix()
        stat = path.stat()
        yield {
            "video_id": stable_video_id(relative_path),
            "relative_path": relative_path,
            "video_path": str(path.resolve()),
            "filename": path.name,
            "stem": path.stem,
            "suffix": path.suffix.lower(),
            "size_bytes": stat.st_size,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one JSON manifest from a video directory.")
    parser.add_argument("--video-root", required=True, help="CameraBench MP4 root directory.")
    parser.add_argument("--output", required=True, help="Output JSON manifest path.")
    parser.add_argument(
        "--extensions",
        default=".mp4",
        help="Comma-separated extensions. Default: .mp4",
    )
    args = parser.parse_args()

    video_root = Path(args.video_root).expanduser().resolve()
    if not video_root.is_dir():
        raise FileNotFoundError(f"Video root does not exist or is not a directory: {video_root}")

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    videos = list(iter_videos(video_root, parse_extensions(args.extensions)))
    manifest = {
        "schema_version": "cambench_video_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_root": str(video_root),
        "num_videos": len(videos),
        "videos": videos,
    }

    tmp_path = output.with_suffix(output.suffix + ".tmp")
    tmp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(output)

    print(f"[OK] found {len(videos)} matching videos")
    print(f"[OK] manifest: {output}")


if __name__ == "__main__":
    main()
