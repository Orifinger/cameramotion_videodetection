#!/usr/bin/env python3
"""Concurrent Qwen3-VL object + region proposal dispatcher (UNIFIED JSON VERSION).

Key changes:
- NO per-video JSON files
- single unified dataset: qwen_region_candidates_all.json
- supports SAM3 candidates + deferred candidates
- async 40 concurrency
- resumable execution
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.object_proposal_config import (
    ALL_CANDIDATES_PATH,
    MAX_CONCURRENCY,
    MAX_RETRIES,
    MAX_VIDEOS,
    OVERWRITE_EXISTING,
    QWEN_API_BASE,
    QWEN_MODEL_NAME,
    REQUEST_TIMEOUT_SEC,
    SAVE_EVERY,
    SHUFFLE_MANIFEST,
    TEMPERATURE,
    MANIFEST_PATH,
    ENABLE_JSON_SCHEMA,
    PROGRESS_EVERY,
    RUN_SUMMARY_PATH,
)

SCHEMA_VERSION = "qwen_region_candidates_v2"


# ============================================================
# Prompt
# ============================================================
OBJECT_PROPOSAL_PROMPT = """
You are extracting editable regions for a video editing pipeline.

Return two lists:
1. sam3_candidates: regions that can be localized using SAM 3.1
2. deferred_candidates: regions requiring specialized text/UI/overlay tracking

sam3_candidates must include:
- physical objects
- display surfaces (screens, billboards, posters, signs)

For display surfaces, return the whole surface (e.g., "phone screen"), NOT internal text.

Return JSON only.
""".strip()


# ============================================================
# IO helpers
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ============================================================
# Manifest
# ============================================================

def load_manifest():
    m = json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8"))
    return m["videos"]


def load_existing():
    if ALL_CANDIDATES_PATH.exists():
        return json.loads(ALL_CANDIDATES_PATH.read_text(encoding="utf-8"))
    return {"schema_version": SCHEMA_VERSION, "videos": []}


# ============================================================
# vLLM call
# ============================================================

def payload(video_path):
    return {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": Path(video_path).resolve().as_uri()}},
                    {"type": "text", "text": OBJECT_PROPOSAL_PROMPT},
                ],
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": 1200,
    }


def extract_json(text: str):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    return json.loads(text[start:end+1])


# ============================================================
# Unified memory store
# ============================================================
class UnifiedStore:
    def __init__(self, data):
        self.data = data
        self.map = {v["video_id"]: v for v in data["videos"]}

    def update(self, video_id, record):
        self.map[video_id] = record

    def export(self):
        self.data["videos"] = list(self.map.values())
        return self.data


# ============================================================
# Worker
# ============================================================
async def worker(session, sem, video, store, stats):
    async with sem:
        for i in range(MAX_RETRIES):
            try:
                async with session.post(
                    QWEN_API_BASE + "/chat/completions",
                    json=payload(video["video_path"]),
                    timeout=REQUEST_TIMEOUT_SEC,
                ) as r:
                    res = await r.json()
                    content = res["choices"][0]["message"]["content"]
                    parsed = extract_json(content)

                    record = {
                        "video_id": video["video_id"],
                        "video_path": video["video_path"],
                        "status": "success",
                        "schema_version": SCHEMA_VERSION,
                        "data": parsed,
                    }

                    store.update(video["video_id"], record)
                    stats["success"] += 1
                    return

            except Exception as e:
                last_err = str(e)
                await asyncio.sleep(2 ** i)

        store.update(video["video_id"], {
            "video_id": video["video_id"],
            "video_path": video["video_path"],
            "status": "failure",
            "error": last_err,
        })
        stats["failure"] += 1


# ============================================================
# Main
# ============================================================
async def main():
    videos = load_manifest()
    if MAX_VIDEOS:
        videos = videos[:MAX_VIDEOS]

    base = load_existing()
    store = UnifiedStore(base)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    stats = {"success": 0, "failure": 0}

    async with aiohttp.ClientSession() as session:
        tasks = [worker(session, sem, v, store, stats) for v in videos]

        for i, fut in enumerate(asyncio.as_completed(tasks)):
            await fut

            if i % SAVE_EVERY == 0:
                atomic_write(ALL_CANDIDATES_PATH, store.export())
                print(f"[SAVE] {i}/{len(videos)}")

    atomic_write(ALL_CANDIDATES_PATH, store.export())

    atomic_write(RUN_SUMMARY_PATH, {
        "success": stats["success"],
        "failure": stats["failure"],
        "total": len(videos),
        "time": utc_now(),
    })


if __name__ == "__main__":
    asyncio.run(main())
