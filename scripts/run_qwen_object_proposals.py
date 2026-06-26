#!/usr/bin/env python3
"""Concurrent, resumable Qwen3-VL object proposal inference.

Normal use:
    python scripts/run_qwen_object_proposals.py

All paths and runtime parameters are defined in configs/object_proposal_config.py.
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
    ENABLE_JSON_SCHEMA,
    FAILURE_DIR,
    MANIFEST_PATH,
    MAX_CONCURRENCY,
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    MAX_VIDEOS,
    OVERWRITE_EXISTING,
    PROGRESS_EVERY,
    QWEN_API_BASE,
    QWEN_MODEL_NAME,
    REQUEST_TIMEOUT_SEC,
    RESULT_DIR,
    RUN_SUMMARY_PATH,
    SHUFFLE_MANIFEST,
    TEMPERATURE,
)

SCHEMA_VERSION = "focus_object_candidates_v1"
ALLOWED_CATEGORIES = {"person", "animal", "vehicle", "object"}
ALLOWED_SCREEN_REGIONS = {"left", "center", "right", "upper", "lower", "unknown"}
ALLOWED_PRESENCE = {"throughout", "mostly", "early", "middle", "late", "brief"}
ALLOWED_ROLES = {
    "primary_subject",
    "secondary_subject",
    "vehicle",
    "animal",
    "handheld_item",
    "foreground_object",
    "background_object",
}

OBJECT_PROPOSAL_PROMPT = """
You are proposing object candidates for downstream video segmentation and tracking.

Inspect the full video. Return up to 8 independent objects that are suitable for
later local video editing and cross-frame segmentation. Select only objects that
are visually salient, reasonably distinct, visible for a substantial part of the
video, and likely to have clear boundaries and stable tracks under camera motion.

Prefer people, animals, vehicles, bicycles, boats, airplanes, bags, clothing,
handheld items, and other salient foreground objects.

Do not include the whole scene, sky, ground, road, wall, room, landscape,
background texture, text, subtitles, UI, HUD, logos, maps, screens, dense crowds
as one object, tiny objects, transparent or strongly reflective objects, thin
structures, or objects that appear only momentarily.

`display_phrase` is for human review and may use color or location to identify an
instance. `sam_prompt` is for text-guided segmentation: it must be a short,
plain English noun phrase, with no pronouns, timestamps, full sentences, or long
spatial relations. It may be less specific than display_phrase.

Return JSON only. Do not output markdown fences, reasoning, or any text outside
of the JSON object.
""".strip()

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "objects", "no_candidate_reason"],
    "properties": {
        "schema_version": {"type": "string"},
        "objects": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "candidate_id",
                    "display_phrase",
                    "sam_prompt",
                    "category",
                    "attributes",
                    "screen_region",
                    "temporal_presence",
                    "role",
                    "editable_priority",
                    "selection_reason",
                ],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "display_phrase": {"type": "string"},
                    "sam_prompt": {"type": "string"},
                    "category": {"type": "string", "enum": sorted(ALLOWED_CATEGORIES)},
                    "attributes": {"type": "array", "items": {"type": "string"}},
                    "screen_region": {"type": "string", "enum": sorted(ALLOWED_SCREEN_REGIONS)},
                    "temporal_presence": {"type": "string", "enum": sorted(ALLOWED_PRESENCE)},
                    "role": {"type": "string", "enum": sorted(ALLOWED_ROLES)},
                    "editable_priority": {"type": "integer", "minimum": 1, "maximum": 8},
                    "selection_reason": {"type": "string"},
                },
            },
        },
        "no_candidate_reason": {"type": ["string", "null"]},
    },
}


class ProposalError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_manifest() -> list[dict[str, Any]]:
    manifest_path = Path(MANIFEST_PATH)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run python scripts/build_video_manifest.py first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    videos = manifest.get("videos")
    if not isinstance(videos, list):
        raise ValueError(f"Invalid manifest: missing list field 'videos' in {manifest_path}")

    required = {"video_id", "video_path", "relative_path"}
    seen_ids: set[str] = set()
    valid: list[dict[str, Any]] = []
    for item in videos:
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"Invalid manifest item: {item}")
        video_id = str(item["video_id"])
        if video_id in seen_ids:
            raise ValueError(f"Duplicate video_id in manifest: {video_id}")
        seen_ids.add(video_id)
        valid.append(item)
    return valid


def result_path(video_id: str) -> Path:
    return Path(RESULT_DIR) / f"{video_id}.json"


def failure_path(video_id: str) -> Path:
    return Path(FAILURE_DIR) / f"{video_id}.json"


def is_success_record(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("status") == "success" and data.get("schema_version") == SCHEMA_VERSION
    except (OSError, json.JSONDecodeError):
        return False


def selected_tasks(videos: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    missing_paths = 0
    candidates: list[dict[str, Any]] = []
    for video in videos:
        if not Path(video["video_path"]).is_file():
            missing_paths += 1
            atomic_write_json(
                failure_path(video["video_id"]),
                {
                    "status": "failure",
                    "schema_version": SCHEMA_VERSION,
                    "video": video,
                    "created_at_utc": utc_now(),
                    "error_type": "missing_video_file",
                    "errors": [{"message": f"File does not exist: {video['video_path']}"}],
                },
            )
            continue
        if not OVERWRITE_EXISTING and is_success_record(result_path(video["video_id"])):
            continue
        candidates.append(video)

    if SHUFFLE_MANIFEST:
        random.Random(20260626).shuffle(candidates)
    if MAX_VIDEOS is not None:
        candidates = candidates[:MAX_VIDEOS]
    skipped = len(videos) - missing_paths - len(candidates)
    return candidates, skipped, missing_paths


def vllm_payload(video_path: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": False,
    }
    if ENABLE_JSON_SCHEMA:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "focus_object_candidates",
                "schema": JSON_SCHEMA,
                "strict": True,
            },
        }
    return payload


def extract_balanced_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :] if first_newline >= 0 else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    starts = [index for index, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            char = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : end + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
    raise ProposalError("Model response did not contain a parseable JSON object")


def clean_text(value: Any, max_length: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())[:max_length]


def normalize_proposal(raw: dict[str, Any]) -> dict[str, Any]:
    raw_objects = raw.get("objects")
    if not isinstance(raw_objects, list):
        raise ProposalError("JSON field 'objects' must be a list")

    objects: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for item in raw_objects:
        if not isinstance(item, dict):
            continue
        display_phrase = clean_text(item.get("display_phrase"), 180)
        sam_prompt = clean_text(item.get("sam_prompt"), 80).lower()
        category = clean_text(item.get("category"), 32).lower()
        if not display_phrase or not sam_prompt or category not in ALLOWED_CATEGORIES:
            continue
        if sam_prompt in seen_prompts:
            continue
        seen_prompts.add(sam_prompt)

        attrs = item.get("attributes", [])
        if not isinstance(attrs, list):
            attrs = []
        attributes = [clean_text(a, 64) for a in attrs if clean_text(a, 64)]

        screen_region = clean_text(item.get("screen_region"), 32).lower()
        presence = clean_text(item.get("temporal_presence"), 32).lower()
        role = clean_text(item.get("role"), 32).lower()
        if screen_region not in ALLOWED_SCREEN_REGIONS:
            screen_region = "unknown"
        if presence not in ALLOWED_PRESENCE:
            presence = "mostly"
        if role not in ALLOWED_ROLES:
            role = "foreground_object"

        objects.append(
            {
                "candidate_id": f"obj_{len(objects) + 1:02d}",
                "display_phrase": display_phrase,
                "sam_prompt": sam_prompt,
                "category": category,
                "attributes": attributes,
                "screen_region": screen_region,
                "temporal_presence": presence,
                "role": role,
                "editable_priority": len(objects) + 1,
                "selection_reason": clean_text(item.get("selection_reason"), 240),
            }
        )
        if len(objects) == 8:
            break

    no_candidate_reason = raw.get("no_candidate_reason")
    no_candidate_reason = clean_text(no_candidate_reason, 240) if no_candidate_reason else None
    if not objects and not no_candidate_reason:
        no_candidate_reason = "No valid object candidate remained after output validation."

    return {
        "schema_version": SCHEMA_VERSION,
        "objects": objects,
        "no_candidate_reason": no_candidate_reason,
    }


def response_content(response_json: dict[str, Any]) -> str:
    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProposalError(f"Unexpected OpenAI response format: {response_json}") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    raise ProposalError(f"Unsupported message.content type: {type(content).__name__}")


async def check_server(session: aiohttp.ClientSession) -> None:
    url = f"{QWEN_API_BASE.rstrip('/')}/models"
    async with session.get(url) as response:
        body = await response.text()
        if response.status >= 400:
            raise RuntimeError(f"Qwen server health check failed ({response.status}): {body[:500]}")
        try:
            model_ids = {item["id"] for item in json.loads(body).get("data", [])}
        except (json.JSONDecodeError, KeyError, TypeError):
            model_ids = set()
        if model_ids and QWEN_MODEL_NAME not in model_ids:
            raise RuntimeError(
                f"QWEN_MODEL_NAME={QWEN_MODEL_NAME!r} is not served. Available models: {sorted(model_ids)}"
            )


async def infer_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    video: dict[str, Any],
) -> tuple[str, float, list[dict[str, Any]]]:
    endpoint = f"{QWEN_API_BASE.rstrip('/')}/chat/completions"
    errors: list[dict[str, Any]] = []
    started = time.perf_counter()

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.post(endpoint, json=vllm_payload(video["video_path"])) as response:
                    body = await response.text()
                    if response.status >= 400:
                        raise ProposalError(f"HTTP {response.status}: {body[:1000]}")
                    parsed = json.loads(body)
                    raw_content = response_content(parsed)
                    proposal = normalize_proposal(extract_balanced_json(raw_content))

                record = {
                    "status": "success",
                    "schema_version": SCHEMA_VERSION,
                    "created_at_utc": utc_now(),
                    "model": QWEN_MODEL_NAME,
                    "video": video,
                    "attempt_count": attempt,
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    "proposal": proposal,
                    "raw_model_content": raw_content,
                }
                atomic_write_json(result_path(video["video_id"]), record)
                return "success", time.perf_counter() - started, errors
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, ProposalError) as exc:
                errors.append({"attempt": attempt, "error_type": type(exc).__name__, "message": str(exc)[:2000]})
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(min(30.0, 2.0 ** (attempt - 1)))

    failure_record = {
        "status": "failure",
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "model": QWEN_MODEL_NAME,
        "video": video,
        "attempt_count": MAX_RETRIES,
        "errors": errors,
    }
    atomic_write_json(failure_path(video["video_id"]), failure_record)
    return "failure", time.perf_counter() - started, errors


async def run() -> None:
    videos = load_manifest()
    tasks, skipped, missing = selected_tasks(videos)
    for directory in (Path(RESULT_DIR), Path(FAILURE_DIR), Path(RUN_SUMMARY_PATH).parent):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    state = {
        "schema_version": "qwen_object_proposal_run_summary_v1",
        "started_at_utc": utc_now(),
        "config": {
            "api_base": QWEN_API_BASE,
            "model": QWEN_MODEL_NAME,
            "max_concurrency": MAX_CONCURRENCY,
            "max_retries": MAX_RETRIES,
            "json_schema_enabled": ENABLE_JSON_SCHEMA,
            "max_videos": MAX_VIDEOS,
        },
        "manifest_path": str(MANIFEST_PATH),
        "manifest_total": len(videos),
        "queued": len(tasks),
        "skipped_existing_success": skipped,
        "missing_files": missing,
        "success": 0,
        "failure": 0,
        "completed": 0,
    }
    atomic_write_json(Path(RUN_SUMMARY_PATH), state)

    if not tasks:
        state["finished_at_utc"] = utc_now()
        state["elapsed_seconds"] = 0.0
        atomic_write_json(Path(RUN_SUMMARY_PATH), state)
        print("[OK] no videos need processing")
        return

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC)
    connector = aiohttp.TCPConnector(limit=max(64, MAX_CONCURRENCY * 2), ttl_dns_cache=300)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        await check_server(session)
        futures = [asyncio.create_task(infer_one(session, semaphore, video)) for video in tasks]
        for future in asyncio.as_completed(futures):
            status, _, _ = await future
            state[status] += 1
            state["completed"] += 1
            completed = state["completed"]
            if completed % PROGRESS_EVERY == 0 or completed == state["queued"]:
                elapsed = max(time.perf_counter() - started, 1e-6)
                rate = state["completed"] / elapsed * 60.0
                state["elapsed_seconds"] = round(elapsed, 3)
                state["throughput_videos_per_min"] = round(rate, 3)
                atomic_write_json(Path(RUN_SUMMARY_PATH), state)
                print(
                    f"[PROGRESS] {completed}/{state['queued']} | "
                    f"success={state['success']} failure={state['failure']} | {rate:.2f} videos/min"
                )

    elapsed = time.perf_counter() - started
    state["finished_at_utc"] = utc_now()
    state["elapsed_seconds"] = round(elapsed, 3)
    state["throughput_videos_per_min"] = round(state["completed"] / max(elapsed, 1e-6) * 60.0, 3)
    atomic_write_json(Path(RUN_SUMMARY_PATH), state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[STOPPED] interrupted; successful per-video result files are preserved and will be skipped on rerun.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
