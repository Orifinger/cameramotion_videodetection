import os
import json
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}


def scan_videos(video_root: str):
    video_root = Path(video_root)
    items = []

    for p in video_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            rel_id = str(p.relative_to(video_root))
            items.append({
                "video_id": rel_id,
                "video_path": str(p),
                "ext": p.suffix.lower(),
                "size_bytes": p.stat().st_size
            })

    return items


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    videos = scan_videos(args.video_root)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for v in videos:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    print(f"[OK] found {len(videos)} videos -> {args.output}")


if __name__ == "__main__":
    main()
