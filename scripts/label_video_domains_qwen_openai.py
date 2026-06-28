#!/usr/bin/env python3
"""Label CameraBench video style/content domains with an OpenAI-compatible Qwen3-VL endpoint.

This is deliberately conservative: it only produces video-level pairing metadata,
not editing instructions. It samples a few frames and writes the same schema used
by build_dataa_v1_catalog_and_pairs.py.

Example:
  python label_video_domains_qwen_openai.py \
    --index res/dataA_v1/registries/video_domain_index_v1.json \
    --out res/dataA_v1/registries/video_domain_index_v1_labeled.json \
    --base-url http://127.0.0.1:8002/v1 --api-key EMPTY --model Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from openai import OpenAI

PROMPT = """You are labeling a video ONLY for donor-pair compatibility in a local-video-editing dataset.
Inspect the sampled frames and return one JSON object, with no markdown:
{
  "content_domain": "real_live_action | animation_cartoon | game_scene | cg_rendered | mixed | unknown",
  "style_domain": "real_live_action | animation_cartoon | game_scene | cg_rendered | mixed | unknown",
  "face_visibility": "none | small | clear",
  "human_count": 0,
  "domain_confidence": 0.0,
  "brief_reason": "short visual reason"
}
Rules:
- real_live_action: ordinary camera-recorded real-world or film footage.
- animation_cartoon: 2D/illustrated/anime/cartoon footage.
- game_scene: recognizable video-game rendered footage or UI/gameplay.
- cg_rendered: photorealistic or stylized 3D CGI not evidently gameplay.
- mixed: important mixture of domains.
- Do not infer whether a video is AIGC. This task is ONLY visual domain classification.
"""

ALLOWED = {"real_live_action", "animation_cartoon", "game_scene", "cg_rendered", "mixed", "unknown"}


def image_to_data_url(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def sample_frames(video_path: str, n: int = 4) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        cap.release()
        raise RuntimeError(f"Invalid frame count: {video_path}")
    ids = sorted({round((count - 1) * q) for q in (0.08, 0.35, 0.65, 0.92)})[:n]
    urls: List[str] = []
    for idx in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            scale = min(1.0, 640.0 / max(h, w))
            if scale < 1.0:
                frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            urls.append(image_to_data_url(frame))
    cap.release()
    if not urls:
        raise RuntimeError(f"No frames decoded: {video_path}")
    return urls


def parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"No JSON object in model output: {text[:200]}")
    obj = json.loads(m.group(0))
    domain = str(obj.get("content_domain", "unknown")).strip().lower().replace(" ", "_")
    style = str(obj.get("style_domain", domain)).strip().lower().replace(" ", "_")
    if domain not in ALLOWED:
        domain = "unknown"
    if style not in ALLOWED:
        style = domain
    face = str(obj.get("face_visibility", "none")).strip().lower()
    if face not in {"none", "small", "clear"}:
        face = "none"
    try:
        confidence = max(0.0, min(1.0, float(obj.get("domain_confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        human_count = max(0, int(obj.get("human_count", 0)))
    except (TypeError, ValueError):
        human_count = 0
    return {
        "content_domain": domain,
        "style_domain": style,
        "face_visibility": face,
        "human_count": human_count,
        "domain_confidence": confidence,
        "brief_reason": str(obj.get("brief_reason", ""))[:300],
    }


def label_one(row: Dict[str, Any], base_url: str, api_key: str, model: str, retries: int) -> Dict[str, Any]:
    client = OpenAI(base_url=base_url, api_key=api_key)
    last_error: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            content: List[Dict[str, Any]] = [{"type": "text", "text": PROMPT}]
            for url in sample_frames(str(row["video_path"])):
                content.append({"type": "image_url", "image_url": {"url": url}})
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=250,
            )
            parsed = parse_json(response.choices[0].message.content or "")
            return {**row, **parsed, "status": "labeled_by_qwen3", "error": None}
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(8, 1 + attempt * 2))
    return {**row, "status": "failed", "error": last_error}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model", required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--only-status", default="needs_qwen3_label")
    args = p.parse_args()

    payload = json.loads(args.index.read_text(encoding="utf-8"))
    rows = payload.get("videos", [])
    todo = [x for x in rows if x.get("status") == args.only_status]
    done = [x for x in rows if x.get("status") != args.only_status]

    print(f"labeling {len(todo)} videos; preserving {len(done)} existing rows")
    labeled: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(label_one, row, args.base_url, args.api_key, args.model, args.retries) for row in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            labeled.append(row)
            if i % 20 == 0 or i == len(todo):
                print(f"completed {i}/{len(todo)}")

    by_id = {str(x["video_id"]): x for x in done + labeled}
    payload["videos"] = [by_id[k] for k in sorted(by_id)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
