#!/usr/bin/env python3
"""Label video visual domains by directly calling an existing local vLLM Qwen3-VL server.

No external OpenAI service is used. The script POSTs to the local vLLM
OpenAI-compatible endpoint, normally http://<host>:<port>/v1/chat/completions.

Example:
  python scripts/label_video_domains_qwen_vllm.py \
    --index res/dataA_v1/registries/video_domain_index_v1.json \
    --out res/dataA_v1/registries/video_domain_index_v1_labeled.json \
    --vllm-base-url http://127.0.0.1:8002/v1 \
    --model Qwen3-VL-8B-Instruct --workers 4
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

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
ALLOWED_DOMAINS = {"real_live_action", "animation_cartoon", "game_scene", "cg_rendered", "mixed", "unknown"}


def endpoint_from_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/chat/completions"
    if base.endswith("/chat/completions"):
        return base
    return base + "/v1/chat/completions"


def image_to_data_url(frame: Any) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    data = base64.b64encode(buf.tobytes()).decode("ascii")
    return "data:image/jpeg;base64," + data


def sample_frames(video_path: str, n: int = 4) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Invalid frame count: {video_path}")
    frame_ids = sorted({round((frame_count - 1) * q) for q in (0.08, 0.35, 0.65, 0.92)})[:n]
    out: List[str] = []
    for idx in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = min(1.0, 640.0 / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        out.append(image_to_data_url(frame))
    cap.release()
    if not out:
        raise RuntimeError(f"No frames decoded: {video_path}")
    return out


def parse_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text.strip(), flags=re.S)
    if not match:
        raise ValueError(f"No JSON object in model output: {text[:300]}")
    obj = json.loads(match.group(0))
    domain = str(obj.get("content_domain", "unknown")).strip().lower().replace(" ", "_")
    style = str(obj.get("style_domain", domain)).strip().lower().replace(" ", "_")
    if domain not in ALLOWED_DOMAINS:
        domain = "unknown"
    if style not in ALLOWED_DOMAINS:
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


def call_vllm(endpoint: str, api_key: str, model: str, image_urls: List[str], timeout: int) -> str:
    content: List[Dict[str, Any]] = [{"type": "text", "text": PROMPT}]
    content.extend({"type": "image_url", "image_url": {"url": u}} for u in image_urls)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 250,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    try:
        return payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected vLLM response: {payload}") from exc


def label_one(row: Dict[str, Any], endpoint: str, api_key: str, model: str, timeout: int, retries: int) -> Dict[str, Any]:
    last_error: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            text = call_vllm(endpoint, api_key, model, sample_frames(str(row["video_path"])), timeout)
            return {**row, **parse_json(text), "status": "labeled_by_qwen3_vllm", "error": None}
        except (RuntimeError, ValueError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(8, 1 + 2 * attempt))
    return {**row, "status": "failed", "error": last_error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--vllm-base-url", required=True, help="For example http://127.0.0.1:8002/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY", help="vLLM commonly accepts EMPTY; this is not an external API key.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--only-status", default="needs_qwen3_label")
    args = parser.parse_args()

    payload = json.loads(args.index.read_text(encoding="utf-8"))
    rows = payload.get("videos", [])
    todo = [row for row in rows if row.get("status") == args.only_status]
    preserved = [row for row in rows if row.get("status") != args.only_status]
    endpoint = endpoint_from_base(args.vllm_base_url)
    print(f"Calling local vLLM endpoint: {endpoint}")
    print(f"Labeling {len(todo)} videos; preserving {len(preserved)} existing rows")

    labeled: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(label_one, row, endpoint, args.api_key, args.model, args.timeout, args.retries)
            for row in todo
        ]
        for index, future in enumerate(as_completed(futures), 1):
            labeled.append(future.result())
            if index % 20 == 0 or index == len(todo):
                print(f"completed {index}/{len(todo)}")

    by_id = {str(row["video_id"]): row for row in preserved + labeled}
    payload["videos"] = [by_id[k] for k in sorted(by_id)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
