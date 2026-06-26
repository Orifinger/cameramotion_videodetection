#!/usr/bin/env python3
"""Concurrent Qwen video-region proposal dispatcher.

Normal use:
    python scripts/run_qwen_object_proposals.py

The script reads all paths and hyperparameters from
configs/object_proposal_config.py. It writes small JSON metadata only:
one complete source-of-truth JSON and four derived views. Original MP4 files
remain under /tmp and are never copied.
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
    ENABLE_JSON_SCHEMA,
    MANIFEST_PATH,
    MAX_CONCURRENCY,
    MAX_DEFERRED_CANDIDATES,
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    MAX_SAM3_CANDIDATES,
    MAX_VIDEOS,
    OVERWRITE_EXISTING,
    PERSISTENT_WATERMARK_PATH,
    PROGRESS_EVERY,
    QWEN_API_BASE,
    QWEN_MODEL_NAME,
    REQUEST_TIMEOUT_SEC,
    RETRY_FAILURE_RECORDS,
    RUN_SUMMARY_PATH,
    SAM3_CANDIDATES_PATH,
    SAVE_EVERY,
    SCENE_TEXT_GRAPHIC_PATH,
    SCREEN_OVERLAY_PATH,
    SHUFFLE_MANIFEST,
    TEMPERATURE,
)

SCHEMA_VERSION = "qwen_region_candidates_v3"

ALLOWED_SAM3_FAMILIES = {"physical_instance", "editable_surface"}
ALLOWED_TARGET_SCOPES = {"whole_instance", "whole_surface"}
ALLOWED_SCREEN_REGIONS = {"left", "center", "right", "upper", "lower", "unknown"}
ALLOWED_PRESENCE = {"throughout", "mostly", "early", "middle", "late", "brief"}
ALLOWED_ROLES = {
    "primary_subject",
    "secondary_subject",
    "vehicle",
    "animal",
    "handheld_item",
    "foreground_object",
    "editable_display",
    "editable_planar_surface",
}
ALLOWED_DEFERRED_FAMILIES = {
    "scene_text_graphic_detail",
    "screen_overlay",
    "persistent_watermark",
}
ALLOWED_DEFERRED_REASONS = {
    "requires_text_detection_and_tracking",
    "requires_screen_coordinate_tracking",
    "exclude_from_main_data",
}

# Backstop filter: detailed prompt is the primary policy; this only prevents
# obvious background and component candidates from reaching the SAM stage.
BANNED_EXACT_SAM_PROMPTS = {
    "background", "building", "building facade", "ceiling", "cityscape",
    "crowd", "entire view", "floor", "landscape", "road", "scene",
    "sidewalk", "sky", "wall", "windows",
}
BANNED_SAM_SUBSTRINGS = {
    "bolt", "car door handle", "door handle", "door hinge", "door latch",
    "door lock", "door seal", "hinge", "latch", "lock", "screw", "trim",
    "wheel rim",
}


OBJECT_PROPOSAL_PROMPT = """
You are selecting a SMALL set of high-quality editable region candidates from a
complete video for a downstream segmentation-and-tracking pipeline.

Your task is NOT to inventory every visible object. Select only regions that are
likely to be useful for local video editing and likely to remain coherent across
frames under camera motion.

Return exactly two separate lists:

1. `sam3_candidates`
   Main-pipeline candidates that can be described by a short English noun phrase,
   segmented as one coherent region, and tracked across the video.

2. `deferred_candidates`
   Potentially useful editable regions that should be recorded but NOT sent to
   the current segmentation-and-tracking pipeline because they need specialized
   text, overlay, or screen-coordinate localization later.

========================
A. sam3_candidates
========================

Only include one of these two region families:

1) `physical_instance`
A clearly visible, bounded, independent physical entity.

Typical examples:
- person, animal, car, bus, bicycle, motorcycle, skateboard, boat, airplane;
- backpack, handbag, suitcase, chair, instrument, sports ball, handheld object;
- a salient foreground object with a coherent outline.

2) `editable_surface`
A complete, visible, bounded physical carrier surface that can be edited as one
unit.

Typical examples:
- phone screen, tablet screen, television screen, computer monitor,
  dashboard display;
- storefront sign, billboard, poster, book cover, paper map;
- T-shirt, vehicle door panel, package front, product label surface.

For an `editable_surface`, select the WHOLE physical carrier surface rather than
a small internal detail.

Correct:
- "phone screen"
- "car dashboard display"
- "storefront sign"
- "billboard"
- "T-shirt"

Incorrect:
- "map labels on the phone screen"
- "text on the storefront sign"
- "logo on the T-shirt"
- "icon on the dashboard display"

Candidate quality requirements:
- visually salient and sufficiently large;
- distinct from nearby instances;
- visible for a substantial part of the video whenever possible;
- has a reasonably clear boundary;
- likely to remain trackable despite camera motion, moderate occlusion, or pose
  change;
- useful for a local operation such as replace, remove, modify, or inpaint.

Do not output redundant nested choices. For example, do not select both a phone
and the same phone screen unless they are independently useful and non-overlapping
editing targets. Do not repeat the same object using alternate descriptions.

========================
B. Strict exclusions from sam3_candidates
========================

Never include:
- the whole scene or an unbounded background region: sky, floor, ceiling, road,
  sidewalk, wall, building facade, landscape, cityscape, generic background;
- dense/unbounded collections such as a crowd, a set of trees, or a set of
  windows;
- tiny, brief, ambiguous, heavily occluded, transparent, strongly reflective, or
  very thin targets;
- object parts or mechanical subcomponents: door handle, hinge, lock, latch,
  seal, trim, screw, bolt, wheel rim, and similar pieces;
- text-level, logo-level, icon-level, subtitle-level, HUD-level, or UI-level
  details as standalone targets;
- vague targets such as "background", "scene", "everything", or "entire view".

Do not decompose a parent object into parts.

Bad:
- car door handle, car door lock, car door hinge, car door trim

Good:
- car

========================
C. deferred_candidates
========================

Record these useful regions separately, but do NOT include them in
`sam3_candidates`:

- `scene_text_graphic_detail`
  Exact printed text, small logo, graphic, map label, or internal content on a
  sign, poster, screen, clothing, vehicle, or other physical carrier.

- `screen_overlay`
  Subtitles, HUD, game UI, lower-thirds, chat overlays, screen-fixed interface,
  and other image-plane overlays.

- `persistent_watermark`
  Channel bugs, platform marks, copyright marks, or logos fixed at a screen
  corner. These are recorded for analysis and excluded from the main editing
  dataset by default.

When a deferred region lies inside a selected carrier surface, reference its
`parent_candidate_id`. Otherwise use null.

========================
D. Field rules and ranking
========================

For every `sam3_candidates` item:
- `display_phrase`: human-readable instance description. Add color, clothing,
  or relative position only when it disambiguates an instance.
- `sam_prompt`: a short, plain English noun phrase for segmentation. Do not use
  pronouns, timestamps, sentences, or long spatial relations.
- `region_family`: `physical_instance` or `editable_surface`.
- `target_scope`: `whole_instance` for physical_instance, `whole_surface` for
  editable_surface.
- `editable_priority`: rank only the sam3 candidates from 1 upward. Rank by
  expected editability and cross-frame trackability, not merely by category.

Return at most %d `sam3_candidates` and at most %d `deferred_candidates`.

Return JSON only. Do not output Markdown, explanations, chain-of-thought, or
any text outside the JSON object.

Required JSON structure:
{
  "schema_version": "qwen_region_candidates_v3",
  "sam3_candidates": [
    {
      "candidate_id": "sam3_01",
      "region_family": "physical_instance|editable_surface",
      "target_scope": "whole_instance|whole_surface",
      "display_phrase": "string",
      "sam_prompt": "string",
      "attributes": ["string"],
      "screen_region": "left|center|right|upper|lower|unknown",
      "temporal_presence": "throughout|mostly|early|middle|late|brief",
      "role": "primary_subject|secondary_subject|vehicle|animal|handheld_item|foreground_object|editable_display|editable_planar_surface",
      "editable_priority": 1,
      "selection_reason": "string"
    }
  ],
  "deferred_candidates": [
    {
      "candidate_id": "deferred_01",
      "region_family": "scene_text_graphic_detail|screen_overlay|persistent_watermark",
      "display_phrase": "string",
      "parent_candidate_id": "sam3_01 or null",
      "screen_region": "left|center|right|upper|lower|unknown",
      "temporal_presence": "throughout|mostly|early|middle|late|brief",
      "deferred_reason": "requires_text_detection_and_tracking|requires_screen_coordinate_tracking|exclude_from_main_data"
    }
  ],
  "no_sam3_candidate_reason": null
}
""" % (MAX_SAM3_CANDIDATES, MAX_DEFERRED_CANDIDATES)


def schema_object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


JSON_SCHEMA: dict[str, Any] = schema_object(
    {
        "schema_version": {"type": "string"},
        "sam3_candidates": {
            "type": "array",
            "maxItems": MAX_SAM3_CANDIDATES,
            "items": schema_object(
                {
                    "candidate_id": {"type": "string"},
                    "region_family": {"type": "string", "enum": sorted(ALLOWED_SAM3_FAMILIES)},
                    "target_scope": {"type": "string", "enum": sorted(ALLOWED_TARGET_SCOPES)},
                    "display_phrase": {"type": "string"},
                    "sam_prompt": {"type": "string"},
                    "attributes": {"type": "array", "items": {"type": "string"}},
                    "screen_region": {"type": "string", "enum": sorted(ALLOWED_SCREEN_REGIONS)},
                    "temporal_presence": {"type": "string", "enum": sorted(ALLOWED_PRESENCE)},
                    "role": {"type": "string", "enum": sorted(ALLOWED_ROLES)},
                    "editable_priority": {"type": "integer", "minimum": 1, "maximum": MAX_SAM3_CANDIDATES},
                    "selection_reason": {"type": "string"},
                },
                [
                    "candidate_id", "region_family", "target_scope", "display_phrase",
                    "sam_prompt", "attributes", "screen_region", "temporal_presence",
                    "role", "editable_priority", "selection_reason",
                ],
            ),
        },
        "deferred_candidates": {
            "type": "array",
            "maxItems": MAX_DEFERRED_CANDIDATES,
            "items": schema_object(
                {
                    "candidate_id": {"type": "string"},
                    "region_family": {"type": "string", "enum": sorted(ALLOWED_DEFERRED_FAMILIES)},
                    "display_phrase": {"type": "string"},
                    "parent_candidate_id": {"type": ["string", "null"]},
                    "screen_region": {"type": "string", "enum": sorted(ALLOWED_SCREEN_REGIONS)},
                    "temporal_presence": {"type": "string", "enum": sorted(ALLOWED_PRESENCE)},
                    "deferred_reason": {"type": "string", "enum": sorted(ALLOWED_DEFERRED_REASONS)},
                },
                [
                    "candidate_id", "region_family", "display_phrase", "parent_candidate_id",
                    "screen_region", "temporal_presence", "deferred_reason",
                ],
            ),
        },
        "no_sam3_candidate_reason": {"type": ["string", "null"]},
    },
    ["schema_version", "sam3_candidates", "deferred_candidates", "no_sam3_candidate_reason"],
)


class ProposalError(RuntimeError):
    """A model response cannot be used as a region-proposal record."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def clean_text(value: Any, max_length: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())[:max_length]


def enum_or_default(value: Any, allowed: set[str], default: str) -> str:
    normalized = clean_text(value, 64).lower()
    return normalized if normalized in allowed else default


def is_banned_sam_prompt(sam_prompt: str) -> bool:
    value = sam_prompt.lower().strip()
    if value in BANNED_EXACT_SAM_PROMPTS:
        return True
    return any(fragment in value for fragment in BANNED_SAM_SUBSTRINGS)


def extract_balanced_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :] if first_newline >= 0 else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            current = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    try:
                        candidate = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(candidate, dict):
                        return candidate
    raise ProposalError("Model response did not contain a parseable JSON object")


def response_content(response_data: dict[str, Any]) -> str:
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProposalError(f"Unexpected OpenAI response: {response_data}") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    raise ProposalError(f"Unsupported message.content type: {type(content).__name__}")


def normalize_proposal(raw: dict[str, Any]) -> dict[str, Any]:
    raw_sam = raw.get("sam3_candidates")
    raw_deferred = raw.get("deferred_candidates")
    if not isinstance(raw_sam, list) or not isinstance(raw_deferred, list):
        raise ProposalError(
            "Expected list fields 'sam3_candidates' and 'deferred_candidates'; "
            "bare-string candidate lists are invalid."
        )

    sam3_candidates: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    raw_to_canonical_id: dict[str, str] = {}

    for item in raw_sam:
        if not isinstance(item, dict):
            continue
        display_phrase = clean_text(item.get("display_phrase"), 180)
        sam_prompt = clean_text(item.get("sam_prompt"), 80).lower()
        family = enum_or_default(item.get("region_family"), ALLOWED_SAM3_FAMILIES, "")
        if not display_phrase or not sam_prompt or not family:
            continue
        if sam_prompt in seen_prompts or is_banned_sam_prompt(sam_prompt):
            continue

        target_scope = enum_or_default(item.get("target_scope"), ALLOWED_TARGET_SCOPES, "")
        expected_scope = "whole_instance" if family == "physical_instance" else "whole_surface"
        if target_scope != expected_scope:
            target_scope = expected_scope

        attributes = item.get("attributes")
        if not isinstance(attributes, list):
            attributes = []
        attributes = [clean_text(x, 64) for x in attributes]
        attributes = [x for x in attributes if x][:6]

        original_id = clean_text(item.get("candidate_id"), 64)
        canonical_id = f"sam3_{len(sam3_candidates) + 1:02d}"
        raw_to_canonical_id[original_id] = canonical_id
        seen_prompts.add(sam_prompt)

        default_role = "foreground_object"
        if family == "editable_surface":
            default_role = (
                "editable_display"
                if any(word in sam_prompt for word in ("screen", "display", "monitor"))
                else "editable_planar_surface"
            )

        sam3_candidates.append(
            {
                "candidate_id": canonical_id,
                "region_family": family,
                "target_scope": target_scope,
                "display_phrase": display_phrase,
                "sam_prompt": sam_prompt,
                "attributes": attributes,
                "screen_region": enum_or_default(item.get("screen_region"), ALLOWED_SCREEN_REGIONS, "unknown"),
                "temporal_presence": enum_or_default(item.get("temporal_presence"), ALLOWED_PRESENCE, "mostly"),
                "role": enum_or_default(item.get("role"), ALLOWED_ROLES, default_role),
                "editable_priority": len(sam3_candidates) + 1,
                "selection_reason": clean_text(item.get("selection_reason"), 280),
            }
        )
        if len(sam3_candidates) >= MAX_SAM3_CANDIDATES:
            break

    deferred_candidates: list[dict[str, Any]] = []
    seen_deferred: set[tuple[str, str]] = set()
    for item in raw_deferred:
        if not isinstance(item, dict):
            continue
        family = enum_or_default(item.get("region_family"), ALLOWED_DEFERRED_FAMILIES, "")
        display_phrase = clean_text(item.get("display_phrase"), 180)
        if not family or not display_phrase:
            continue
        signature = (family, display_phrase.lower())
        if signature in seen_deferred:
            continue
        seen_deferred.add(signature)

        raw_parent = clean_text(item.get("parent_candidate_id"), 64)
        parent_candidate_id = raw_to_canonical_id.get(raw_parent)
        default_reason = (
            "requires_text_detection_and_tracking"
            if family == "scene_text_graphic_detail"
            else "requires_screen_coordinate_tracking"
            if family == "screen_overlay"
            else "exclude_from_main_data"
        )
        deferred_candidates.append(
            {
                "candidate_id": f"deferred_{len(deferred_candidates) + 1:02d}",
                "region_family": family,
                "display_phrase": display_phrase,
                "parent_candidate_id": parent_candidate_id,
                "screen_region": enum_or_default(item.get("screen_region"), ALLOWED_SCREEN_REGIONS, "unknown"),
                "temporal_presence": enum_or_default(item.get("temporal_presence"), ALLOWED_PRESENCE, "mostly"),
                "deferred_reason": enum_or_default(item.get("deferred_reason"), ALLOWED_DEFERRED_REASONS, default_reason),
            }
        )
        if len(deferred_candidates) >= MAX_DEFERRED_CANDIDATES:
            break

    no_sam3_candidate_reason = clean_text(raw.get("no_sam3_candidate_reason"), 280)
    if not sam3_candidates and not no_sam3_candidate_reason:
        no_sam3_candidate_reason = "No valid candidate satisfied the current segmentation-and-tracking criteria."

    return {
        "schema_version": SCHEMA_VERSION,
        "sam3_candidates": sam3_candidates,
        "deferred_candidates": deferred_candidates,
        "no_sam3_candidate_reason": no_sam3_candidate_reason or None,
    }


def load_manifest() -> list[dict[str, Any]]:
    path = Path(MANIFEST_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}. Run python scripts/build_video_manifest.py first.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    videos = manifest.get("videos")
    if not isinstance(videos, list):
        raise ValueError(f"Invalid manifest, missing list field 'videos': {path}")

    required = {"video_id", "video_path", "relative_path"}
    seen_ids: set[str] = set()
    checked: list[dict[str, Any]] = []
    for video in videos:
        if not isinstance(video, dict) or not required.issubset(video):
            raise ValueError(f"Invalid manifest video record: {video}")
        video_id = str(video["video_id"])
        if video_id in seen_ids:
            raise ValueError(f"Duplicate video_id in manifest: {video_id}")
        seen_ids.add(video_id)
        checked.append(video)
    return checked


def default_dataset() -> dict[str, Any]:
    now = utc_now()
    return {"schema_version": SCHEMA_VERSION, "created_at_utc": now, "updated_at_utc": now, "videos": []}


class UnifiedStore:
    """Only the main coroutine updates this store and writes aggregate JSON."""

    def __init__(self, existing: dict[str, Any], manifest_videos: list[dict[str, Any]]) -> None:
        self.data = existing if isinstance(existing, dict) else default_dataset()
        records = self.data.get("videos")
        if not isinstance(records, list):
            records = []
        self.by_video_id = {
            str(record["video_id"]): record
            for record in records
            if isinstance(record, dict) and "video_id" in record
        }
        self.manifest_order = [str(video["video_id"]) for video in manifest_videos]
        self.data["schema_version"] = SCHEMA_VERSION
        self.data.setdefault("created_at_utc", utc_now())

    def get(self, video_id: str) -> dict[str, Any] | None:
        return self.by_video_id.get(str(video_id))

    def put(self, record: dict[str, Any]) -> None:
        self.by_video_id[str(record["video_id"])] = record

    def export_all(self) -> dict[str, Any]:
        ordered = [self.by_video_id[video_id] for video_id in self.manifest_order if video_id in self.by_video_id]
        manifest_ids = set(self.manifest_order)
        ordered.extend(record for video_id, record in self.by_video_id.items() if video_id not in manifest_ids)
        self.data["schema_version"] = SCHEMA_VERSION
        self.data["updated_at_utc"] = utc_now()
        self.data["videos"] = ordered
        return self.data


def is_current_terminal_record(record: dict[str, Any] | None) -> bool:
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        return False
    if record.get("status") not in {"success", "no_sam3_candidate"}:
        return False
    proposal = record.get("proposal")
    return isinstance(proposal, dict) and isinstance(proposal.get("sam3_candidates"), list) and isinstance(proposal.get("deferred_candidates"), list)


def make_payload(video_path: str) -> dict[str, Any]:
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
            "json_schema": {"name": "qwen_region_candidates_v3", "schema": JSON_SCHEMA, "strict": True},
        }
    return payload


async def check_server(session: aiohttp.ClientSession) -> None:
    endpoint = f"{QWEN_API_BASE.rstrip('/')}/models"
    async with session.get(endpoint) as response:
        body = await response.text()
        if response.status >= 400:
            raise RuntimeError(f"Qwen server health check failed with HTTP {response.status}: {body[:800]}")
        try:
            available = {item["id"] for item in json.loads(body).get("data", []) if isinstance(item, dict) and "id" in item}
        except (TypeError, ValueError, json.JSONDecodeError):
            available = set()
        if available and QWEN_MODEL_NAME not in available:
            raise RuntimeError(f"QWEN_MODEL_NAME={QWEN_MODEL_NAME!r} is not served. Available models: {sorted(available)}")


async def infer_one(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, video: dict[str, Any]) -> dict[str, Any]:
    endpoint = f"{QWEN_API_BASE.rstrip('/')}/chat/completions"
    errors: list[dict[str, Any]] = []
    started = time.perf_counter()

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.post(endpoint, json=make_payload(video["video_path"])) as response:
                    response_text = await response.text()
                    if response.status >= 400:
                        raise ProposalError(f"HTTP {response.status}: {response_text[:1500]}")
                    response_json = json.loads(response_text)
                    parsed = extract_balanced_json(response_content(response_json))
                    proposal = normalize_proposal(parsed)

                status = "success" if proposal["sam3_candidates"] else "no_sam3_candidate"
                return {
                    "video_id": video["video_id"],
                    "relative_path": video["relative_path"],
                    "video_path": video["video_path"],
                    "status": status,
                    "schema_version": SCHEMA_VERSION,
                    "created_at_utc": utc_now(),
                    "model": QWEN_MODEL_NAME,
                    "attempt_count": attempt,
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    "proposal": proposal,
                }
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, ProposalError) as exc:
                errors.append({"attempt": attempt, "error_type": type(exc).__name__, "message": str(exc)[:2000]})
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(min(30.0, 2.0 ** (attempt - 1)))

    return {
        "video_id": video["video_id"],
        "relative_path": video["relative_path"],
        "video_path": video["video_path"],
        "status": "failure",
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "model": QWEN_MODEL_NAME,
        "attempt_count": MAX_RETRIES,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "errors": errors,
    }


def make_family_view(all_records: list[dict[str, Any]], family: str, path: Path) -> dict[str, Any]:
    videos: list[dict[str, Any]] = []
    for record in all_records:
        proposal = record.get("proposal")
        candidates = proposal.get("deferred_candidates") if isinstance(proposal, dict) else None
        if not isinstance(candidates, list):
            continue
        selected = [item for item in candidates if isinstance(item, dict) and item.get("region_family") == family]
        if selected:
            videos.append({
                "video_id": record["video_id"],
                "relative_path": record.get("relative_path"),
                "video_path": record.get("video_path"),
                "status": record["status"],
                "deferred_candidates": selected,
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "source_file": str(ALL_CANDIDATES_PATH),
        "view_file": str(path),
        "region_family": family,
        "num_videos": len(videos),
        "num_candidates": sum(len(video["deferred_candidates"]) for video in videos),
        "videos": videos,
    }


def write_all_outputs(store: UnifiedStore) -> dict[str, Any]:
    all_dataset = store.export_all()
    all_records = all_dataset["videos"]
    atomic_write_json(Path(ALL_CANDIDATES_PATH), all_dataset)

    sam3_videos: list[dict[str, Any]] = []
    for record in all_records:
        proposal = record.get("proposal")
        candidates = proposal.get("sam3_candidates") if isinstance(proposal, dict) else None
        if isinstance(candidates, list) and candidates:
            sam3_videos.append({
                "video_id": record["video_id"],
                "relative_path": record.get("relative_path"),
                "video_path": record.get("video_path"),
                "status": record["status"],
                "sam3_candidates": candidates,
            })
    atomic_write_json(Path(SAM3_CANDIDATES_PATH), {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "source_file": str(ALL_CANDIDATES_PATH),
        "num_videos": len(sam3_videos),
        "num_candidates": sum(len(video["sam3_candidates"]) for video in sam3_videos),
        "videos": sam3_videos,
    })
    atomic_write_json(Path(SCENE_TEXT_GRAPHIC_PATH), make_family_view(all_records, "scene_text_graphic_detail", Path(SCENE_TEXT_GRAPHIC_PATH)))
    atomic_write_json(Path(SCREEN_OVERLAY_PATH), make_family_view(all_records, "screen_overlay", Path(SCREEN_OVERLAY_PATH)))
    atomic_write_json(Path(PERSISTENT_WATERMARK_PATH), make_family_view(all_records, "persistent_watermark", Path(PERSISTENT_WATERMARK_PATH)))
    return all_dataset


def build_summary(started_at: str, started_perf: float, manifest_total: int, queued: int, skipped: int, missing_files: int, counts: dict[str, int], dataset: dict[str, Any], finished: bool) -> dict[str, Any]:
    elapsed = time.perf_counter() - started_perf
    records = dataset.get("videos", [])
    return {
        "schema_version": "qwen_region_candidate_run_summary_v3",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now() if finished else None,
        "config": {
            "api_base": QWEN_API_BASE,
            "model": QWEN_MODEL_NAME,
            "max_concurrency": MAX_CONCURRENCY,
            "max_retries": MAX_RETRIES,
            "max_sam3_candidates": MAX_SAM3_CANDIDATES,
            "max_deferred_candidates": MAX_DEFERRED_CANDIDATES,
            "json_schema_enabled": ENABLE_JSON_SCHEMA,
            "max_videos": MAX_VIDEOS,
        },
        "manifest_path": str(MANIFEST_PATH),
        "manifest_total": manifest_total,
        "queued_this_run": queued,
        "skipped_existing_terminal": skipped,
        "missing_video_files": missing_files,
        "processed_this_run": counts,
        "dataset_status_totals": {
            "success": sum(record.get("status") == "success" for record in records),
            "no_sam3_candidate": sum(record.get("status") == "no_sam3_candidate" for record in records),
            "failure": sum(record.get("status") == "failure" for record in records),
            "records": len(records),
        },
        "elapsed_seconds": round(elapsed, 3),
        "throughput_videos_per_min": round(sum(counts.values()) / max(elapsed, 1e-6) * 60.0, 3),
    }


def select_tasks(manifest_videos: list[dict[str, Any]], store: UnifiedStore) -> tuple[list[dict[str, Any]], int, int]:
    to_process: list[dict[str, Any]] = []
    skipped = 0
    missing_files = 0
    for video in manifest_videos:
        video_id = str(video["video_id"])
        if not Path(video["video_path"]).is_file():
            missing_files += 1
            store.put({
                "video_id": video_id,
                "relative_path": video["relative_path"],
                "video_path": video["video_path"],
                "status": "failure",
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": utc_now(),
                "model": QWEN_MODEL_NAME,
                "attempt_count": 0,
                "errors": [{"attempt": 0, "error_type": "missing_video_file", "message": f"File does not exist: {video['video_path']}"}],
            })
            continue
        existing = store.get(video_id)
        terminal = is_current_terminal_record(existing)
        failed_and_disabled = isinstance(existing, dict) and existing.get("status") == "failure" and not RETRY_FAILURE_RECORDS
        if not OVERWRITE_EXISTING and (terminal or failed_and_disabled):
            skipped += 1
            continue
        to_process.append(video)

    if SHUFFLE_MANIFEST:
        random.Random(20260626).shuffle(to_process)
    if MAX_VIDEOS is not None:
        to_process = to_process[:MAX_VIDEOS]
    return to_process, skipped, missing_files


async def run() -> None:
    manifest_videos = load_manifest()
    existing = default_dataset()
    result_path = Path(ALL_CANDIDATES_PATH)
    if result_path.is_file():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"[WARN] unreadable existing result file; new v3 dataset will be created: {result_path}")

    store = UnifiedStore(existing, manifest_videos)
    tasks, skipped, missing_files = select_tasks(manifest_videos, store)
    started_at = utc_now()
    started_perf = time.perf_counter()
    counts = {"success": 0, "no_sam3_candidate": 0, "failure": 0}

    dataset = write_all_outputs(store)
    atomic_write_json(Path(RUN_SUMMARY_PATH), build_summary(started_at, started_perf, len(manifest_videos), len(tasks), skipped, missing_files, counts, dataset, False))

    if not tasks:
        dataset = write_all_outputs(store)
        summary = build_summary(started_at, started_perf, len(manifest_videos), 0, skipped, missing_files, counts, dataset, True)
        atomic_write_json(Path(RUN_SUMMARY_PATH), summary)
        print("[OK] no videos need processing")
        return

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC)
    connector = aiohttp.TCPConnector(limit=max(64, MAX_CONCURRENCY * 2), ttl_dns_cache=300)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        await check_server(session)
        futures = [asyncio.create_task(infer_one(session, semaphore, video)) for video in tasks]
        for completed, future in enumerate(asyncio.as_completed(futures), start=1):
            record = await future
            store.put(record)
            counts[record["status"]] += 1

            if completed %% SAVE_EVERY == 0 or completed == len(tasks):
                dataset = write_all_outputs(store)
                atomic_write_json(Path(RUN_SUMMARY_PATH), build_summary(started_at, started_perf, len(manifest_videos), len(tasks), skipped, missing_files, counts, dataset, False))
            if completed %% PROGRESS_EVERY == 0 or completed == len(tasks):
                rate = completed / max(time.perf_counter() - started_perf, 1e-6) * 60.0
                print(f"[PROGRESS] {completed}/{len(tasks)} | success={counts['success']} no_sam3={counts['no_sam3_candidate']} failure={counts['failure']} | {rate:.2f} videos/min")

    dataset = write_all_outputs(store)
    summary = build_summary(started_at, started_perf, len(manifest_videos), len(tasks), skipped, missing_files, counts, dataset, True)
    atomic_write_json(Path(RUN_SUMMARY_PATH), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[STOPPED] Completed records are preserved in the unified JSON and skipped on rerun.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
