#!/usr/bin/env python3
"""Qwen3-VL v4 candidate-concept discovery for CameraBench local editing.

This stage produces a compact high-recall bank of independently bounded visual
concepts for later SAM3 tracking. It does not choose a final focus region or an
editing operation. Source MP4s stay in /tmp; all outputs are unified JSON files.
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
    PARSER_REJECTIONS_PATH,
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

SCHEMA_VERSION = "qwen_region_candidates_v4"

ALLOWED_SAM3_FAMILIES = {"physical_instance", "editable_surface"}
ALLOWED_TARGET_SCOPES = {"whole_instance", "whole_surface"}
ALLOWED_SCREEN_REGIONS = {"left", "center", "right", "upper", "lower", "unknown"}
ALLOWED_PRESENCE = {"throughout", "mostly", "early", "middle", "late", "brief"}
ALLOWED_INSTANCE_COUNT_HINTS = {"unique_in_video", "possibly_multiple", "unknown"}

PHYSICAL_INSTANCE_CLASSES = {
    "human",
    "animal",
    "vehicle",
    "handheld_object",
    "bounded_object",
}
EDITABLE_SURFACE_CLASSES = {
    "display_screen",
    "sign_or_poster",
    "paper_book_map",
    "framed_art",
    "apparel_panel",
    "vehicle_panel",
    "package_front",
}
ALLOWED_CANDIDATE_CLASSES = PHYSICAL_INSTANCE_CLASSES | EDITABLE_SURFACE_CLASSES

ALLOWED_DEFERRED_FAMILIES = {
    "scene_text_graphic_detail",
    "screen_overlay",
    "persistent_watermark",
    "scene_structure_or_environment",
    "collection_or_group",
    "part_or_tiny_detail",
    "dynamic_material_region",
}
ALLOWED_DEFERRED_REASONS = {
    "requires_text_detection_and_tracking",
    "requires_screen_coordinate_tracking",
    "requires_specialized_segmentation",
    "not_independent_bounded_target",
    "exclude_from_main_data",
}
ALLOWED_NO_CANDIDATE_REASONS = {
    "no_independent_bounded_target",
    "only_deferred_regions",
    "target_too_small_or_occluded",
    "unknown",
}

BANNED_CANONICAL_EXACT = {
    "background", "building", "building facade", "ceiling", "cityscape",
    "cloud", "corridor", "crowd", "entire view", "face", "fire", "floor",
    "fog", "group", "hallway", "hand", "landscape", "ocean", "road",
    "scene", "sky", "smoke", "stack", "wall", "water surface", "object",
    "thing", "window", "windows",
}
BANNED_CANONICAL_SUBSTRINGS = {
    "group of ", "pile of ", "stack of ", "crowd of ", "door handle",
    "door hinge", "door latch", "door lock", "wheel rim", "license plate",
    "button", "subtitle", "hud", "logo",
}
GENERIC_SAM_PROMPTS = {
    "animal", "car", "human", "man", "object", "person", "thing",
    "vehicle", "woman",
}

OBJECT_PROPOSAL_PROMPT = f"""
You are performing candidate visual-concept discovery from a complete video.

This is NOT final edit planning. Do not decide what a future editor should
change. Your job is to propose a compact, high-recall set of visual concepts
that can later be segmented and tracked by SAM3.

Return at most {MAX_SAM3_CANDIDATES} `sam3_candidates` and at most
{MAX_DEFERRED_CANDIDATES} `deferred_candidates`. Do not invent candidates.
A video may have zero main candidates.

A main candidate must be one independently bounded target that can plausibly
receive an instance mask. Select whole objects or whole physical carrier
surfaces, never their tiny parts or internal text/UI details.

MAIN CANDIDATE FAMILIES

1) `physical_instance`
Use `target_scope = "whole_instance"` and one of:
- `human`: one person;
- `animal`: one animal;
- `vehicle`: one car, bus, bicycle, motorcycle, boat, aircraft, skateboard;
- `handheld_object`: one independently held item;
- `bounded_object`: one bounded object such as chair, bag, suitcase, tool,
  instrument, ball, appliance, furniture item, box, door, lamp, statue.

2) `editable_surface`
Use `target_scope = "whole_surface"` and one of:
- `display_screen`: phone, tablet, computer monitor, television, dashboard display;
- `sign_or_poster`: sign, poster, billboard, advertisement panel;
- `paper_book_map`: paper, book cover, map;
- `framed_art`: framed painting or photograph;
- `apparel_panel`: complete T-shirt/front clothing panel;
- `vehicle_panel`: complete vehicle door/body panel;
- `package_front`: complete visible package/product front.

Do not decide whether a candidate is finally editable by a particular model.
Favor recall across distinct, independently bounded targets, while keeping the
set compact. Do not produce alternate names for the same target or nested
duplicates such as both a phone and the exact same phone screen.

STRICT EXCLUSIONS FROM `sam3_candidates`
- whole scenes, environments, or unbounded structures: building, building facade,
  wall, floor, ceiling, road, sidewalk, hallway, corridor, bridge, stairs,
  train tracks, courtyard, landscape, forest, mountain, sky, ocean, river,
  water surface;
- collections: crowd, group of people, row, pile, stack, scattered items;
- body parts, tiny components, text/UI details: hand, face, feet, handle, hinge,
  lock, screw, logo, license plate, button, subtitle, HUD;
- amorphous dynamic material: smoke, fog, fire, cloud, waves, splash, ink;
- a person/object shown inside a display. Select the whole display instead when
  the display itself is a valid bounded carrier.

For candidates with multiple same-category instances, make `sam_prompt` a
short disambiguated noun phrase. Prefer stable visual attributes such as
clothing/color, vehicle appearance, carried object, or stable action:
- good: "woman in white coat", "white taxi", "man holding red umbrella"
- bad when ambiguous: "person", "woman", "car", "object"
Do not write long relational sentences or timestamps.

For each `sam3_candidates` item:
- `canonical_concept` is the core target noun, such as "person", "poster",
  "television screen", "white taxi"; it must name the target itself, not context.
- `display_phrase` is a human-readable description.
- `sam_prompt` is a short English noun phrase for SAM3.
- `instance_count_hint` is `unique_in_video`, `possibly_multiple`, or `unknown`.
- `visual_disambiguators` lists stable attributes used to distinguish the target.
  Use [] only when no stable disambiguation is needed.
- `screen_region` and `temporal_presence` are approximate weak hints.

Use `deferred_candidates` to record visible regions that may be useful later
but must not enter the SAM3 main route:
- `scene_text_graphic_detail`: text/logo/graphic inside a physical carrier;
- `screen_overlay`: subtitles, HUD, lower-thirds, game UI;
- `persistent_watermark`: corner-fixed watermark/platform mark;
- `scene_structure_or_environment`: wall, building facade, road, landscape;
- `collection_or_group`: crowd/group/row/pile/stack;
- `part_or_tiny_detail`: hand, component, button, license plate;
- `dynamic_material_region`: smoke, water, fire, fog, waves.

For `deferred_candidates`, set `parent_candidate_id` only if it lies inside a
main candidate; otherwise use null.

Return JSON only. Do not include Markdown or explanations outside JSON.
"""


def schema_object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


JSON_SCHEMA: dict[str, Any] = schema_object(
    {
        "schema_version": {"type": "string", "enum": [SCHEMA_VERSION]},
        "sam3_candidates": {
            "type": "array",
            "maxItems": MAX_SAM3_CANDIDATES,
            "items": schema_object(
                {
                    "candidate_id": {"type": "string"},
                    "region_family": {"type": "string", "enum": sorted(ALLOWED_SAM3_FAMILIES)},
                    "candidate_class": {"type": "string", "enum": sorted(ALLOWED_CANDIDATE_CLASSES)},
                    "target_scope": {"type": "string", "enum": sorted(ALLOWED_TARGET_SCOPES)},
                    "canonical_concept": {"type": "string"},
                    "display_phrase": {"type": "string"},
                    "sam_prompt": {"type": "string"},
                    "instance_count_hint": {"type": "string", "enum": sorted(ALLOWED_INSTANCE_COUNT_HINTS)},
                    "visual_disambiguators": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
                    "screen_region": {"type": "string", "enum": sorted(ALLOWED_SCREEN_REGIONS)},
                    "temporal_presence": {"type": "string", "enum": sorted(ALLOWED_PRESENCE)},
                },
                [
                    "candidate_id", "region_family", "candidate_class", "target_scope",
                    "canonical_concept", "display_phrase", "sam_prompt",
                    "instance_count_hint", "visual_disambiguators", "screen_region",
                    "temporal_presence",
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
                    "candidate_id", "region_family", "display_phrase",
                    "parent_candidate_id", "screen_region", "temporal_presence",
                    "deferred_reason",
                ],
            ),
        },
        "no_sam3_candidate_reason": {"type": ["string", "null"]},
    },
    ["schema_version", "sam3_candidates", "deferred_candidates", "no_sam3_candidate_reason"],
)


class ProposalError(RuntimeError):
    """A model response cannot be converted into a valid proposal record."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def clean_text(value: Any, max_length: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())[:max_length]


def enum_or_default(value: Any, allowed: set[str], default: str) -> str:
    normalized = clean_text(value, 96).lower()
    return normalized if normalized in allowed else default


def normalize_string_list(value: Any, max_items: int = 4) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    clean: list[str] = []
    for item in value:
        text = clean_text(item, 80)
        key = text.lower()
        if text and key not in seen:
            clean.append(text)
            seen.add(key)
        if len(clean) >= max_items:
            break
    return clean


def canonical_is_banned(canonical_concept: str) -> bool:
    concept = canonical_concept.lower().strip()
    return concept in BANNED_CANONICAL_EXACT or any(
        fragment in concept for fragment in BANNED_CANONICAL_SUBSTRINGS
    )


def expected_scope_for_family(family: str) -> str:
    return "whole_instance" if family == "physical_instance" else "whole_surface"


def class_matches_family(candidate_class: str, family: str) -> bool:
    return (
        family == "physical_instance" and candidate_class in PHYSICAL_INSTANCE_CLASSES
    ) or (
        family == "editable_surface" and candidate_class in EDITABLE_SURFACE_CLASSES
    )


def extract_balanced_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1:] if first_newline >= 0 else text
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
                        parsed = json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
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


def make_rejection(raw_item: Any, reason: str, detail: str = "") -> dict[str, Any]:
    item = raw_item if isinstance(raw_item, dict) else {}
    return {
        "raw_candidate_id": clean_text(item.get("candidate_id"), 64) or None,
        "canonical_concept": clean_text(item.get("canonical_concept"), 120) or None,
        "sam_prompt": clean_text(item.get("sam_prompt"), 120) or None,
        "reason": reason,
        "detail": detail or None,
    }


def normalize_proposal(raw: dict[str, Any]) -> dict[str, Any]:
    raw_sam = raw.get("sam3_candidates")
    raw_deferred = raw.get("deferred_candidates")
    if not isinstance(raw_sam, list) or not isinstance(raw_deferred, list):
        raise ProposalError("Expected list fields 'sam3_candidates' and 'deferred_candidates'.")

    candidates: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    raw_to_canonical_id: dict[str, str] = {}

    for raw_item in raw_sam:
        if not isinstance(raw_item, dict):
            rejections.append(make_rejection(raw_item, "unsupported_candidate_type"))
            continue

        family = enum_or_default(raw_item.get("region_family"), ALLOWED_SAM3_FAMILIES, "")
        candidate_class = enum_or_default(raw_item.get("candidate_class"), ALLOWED_CANDIDATE_CLASSES, "")
        target_scope = enum_or_default(raw_item.get("target_scope"), ALLOWED_TARGET_SCOPES, "")
        canonical_concept = clean_text(raw_item.get("canonical_concept"), 120).lower()
        display_phrase = clean_text(raw_item.get("display_phrase"), 180)
        sam_prompt = clean_text(raw_item.get("sam_prompt"), 120)
        prompt_key = sam_prompt.lower()
        instance_count_hint = enum_or_default(
            raw_item.get("instance_count_hint"), ALLOWED_INSTANCE_COUNT_HINTS, "unknown"
        )
        visual_disambiguators = normalize_string_list(raw_item.get("visual_disambiguators"), max_items=4)

        if not family or not candidate_class or not target_scope:
            rejections.append(make_rejection(raw_item, "invalid_family_class_or_scope"))
            continue
        if not class_matches_family(candidate_class, family):
            rejections.append(make_rejection(raw_item, "family_class_mismatch"))
            continue
        if target_scope != expected_scope_for_family(family):
            rejections.append(make_rejection(raw_item, "family_scope_mismatch"))
            continue
        if not canonical_concept or not display_phrase or not sam_prompt:
            rejections.append(make_rejection(raw_item, "missing_required_text"))
            continue
        if canonical_is_banned(canonical_concept):
            rejections.append(make_rejection(raw_item, "banned_canonical_concept"))
            continue
        if prompt_key in seen_prompts:
            rejections.append(make_rejection(raw_item, "duplicate_sam_prompt"))
            continue
        if instance_count_hint == "possibly_multiple" and prompt_key in GENERIC_SAM_PROMPTS:
            rejections.append(make_rejection(raw_item, "ambiguous_generic_sam_prompt"))
            continue

        raw_id = clean_text(raw_item.get("candidate_id"), 64)
        candidate_id = f"sam3_{len(candidates) + 1:02d}"
        if raw_id:
            raw_to_canonical_id[raw_id] = candidate_id
        seen_prompts.add(prompt_key)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "region_family": family,
                "candidate_class": candidate_class,
                "target_scope": target_scope,
                "canonical_concept": canonical_concept,
                "display_phrase": display_phrase,
                "sam_prompt": sam_prompt,
                "instance_count_hint": instance_count_hint,
                "visual_disambiguators": visual_disambiguators,
                "screen_region": enum_or_default(raw_item.get("screen_region"), ALLOWED_SCREEN_REGIONS, "unknown"),
                "temporal_presence": enum_or_default(raw_item.get("temporal_presence"), ALLOWED_PRESENCE, "mostly"),
            }
        )
        if len(candidates) >= MAX_SAM3_CANDIDATES:
            break

    deferred: list[dict[str, Any]] = []
    seen_deferred: set[tuple[str, str]] = set()
    for raw_item in raw_deferred:
        if not isinstance(raw_item, dict):
            continue
        family = enum_or_default(raw_item.get("region_family"), ALLOWED_DEFERRED_FAMILIES, "")
        display_phrase = clean_text(raw_item.get("display_phrase"), 180)
        if not family or not display_phrase:
            continue
        signature = (family, display_phrase.lower())
        if signature in seen_deferred:
            continue
        seen_deferred.add(signature)
        raw_parent_id = clean_text(raw_item.get("parent_candidate_id"), 64)
        default_reason = (
            "requires_text_detection_and_tracking"
            if family == "scene_text_graphic_detail"
            else "requires_screen_coordinate_tracking"
            if family == "screen_overlay"
            else "requires_specialized_segmentation"
            if family in {"scene_structure_or_environment", "dynamic_material_region"}
            else "not_independent_bounded_target"
            if family in {"collection_or_group", "part_or_tiny_detail"}
            else "exclude_from_main_data"
        )
        deferred.append(
            {
                "candidate_id": f"deferred_{len(deferred) + 1:02d}",
                "region_family": family,
                "display_phrase": display_phrase,
                "parent_candidate_id": raw_to_canonical_id.get(raw_parent_id),
                "screen_region": enum_or_default(raw_item.get("screen_region"), ALLOWED_SCREEN_REGIONS, "unknown"),
                "temporal_presence": enum_or_default(raw_item.get("temporal_presence"), ALLOWED_PRESENCE, "mostly"),
                "deferred_reason": enum_or_default(raw_item.get("deferred_reason"), ALLOWED_DEFERRED_REASONS, default_reason),
            }
        )
        if len(deferred) >= MAX_DEFERRED_CANDIDATES:
            break

    raw_reason = clean_text(raw.get("no_sam3_candidate_reason"), 100).lower()
    no_sam3_candidate_reason = raw_reason if raw_reason in ALLOWED_NO_CANDIDATE_REASONS else None
    if not candidates and no_sam3_candidate_reason is None:
        no_sam3_candidate_reason = "only_deferred_regions" if deferred else "no_independent_bounded_target"

    return {
        "schema_version": SCHEMA_VERSION,
        "sam3_candidates": candidates,
        "deferred_candidates": deferred,
        "no_sam3_candidate_reason": no_sam3_candidate_reason,
        "parser_rejections": rejections,
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
        manifest_ids = set(self.manifest_order)
        ordered = [self.by_video_id[video_id] for video_id in self.manifest_order if video_id in self.by_video_id]
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
    return (
        isinstance(proposal, dict)
        and isinstance(proposal.get("sam3_candidates"), list)
        and isinstance(proposal.get("deferred_candidates"), list)
        and isinstance(proposal.get("parser_rejections"), list)
    )


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
            "json_schema": {"name": SCHEMA_VERSION, "schema": JSON_SCHEMA, "strict": True},
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


def make_rejection_view(all_records: list[dict[str, Any]]) -> dict[str, Any]:
    videos: list[dict[str, Any]] = []
    for record in all_records:
        proposal = record.get("proposal")
        rejected = proposal.get("parser_rejections") if isinstance(proposal, dict) else None
        if isinstance(rejected, list) and rejected:
            videos.append({
                "video_id": record["video_id"],
                "relative_path": record.get("relative_path"),
                "video_path": record.get("video_path"),
                "parser_rejections": rejected,
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "source_file": str(ALL_CANDIDATES_PATH),
        "view_file": str(PARSER_REJECTIONS_PATH),
        "num_videos": len(videos),
        "num_rejections": sum(len(video["parser_rejections"]) for video in videos),
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
    atomic_write_json(Path(PARSER_REJECTIONS_PATH), make_rejection_view(all_records))
    return all_dataset


def build_summary(started_at: str, started_perf: float, manifest_total: int, queued: int, skipped: int, missing_files: int, counts: dict[str, int], dataset: dict[str, Any], finished: bool) -> dict[str, Any]:
    elapsed = time.perf_counter() - started_perf
    records = dataset.get("videos", [])
    proposals = [record.get("proposal") for record in records if isinstance(record, dict) and isinstance(record.get("proposal"), dict)]
    return {
        "schema_version": "qwen_region_candidate_run_summary_v4",
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
        "candidate_totals": {
            "sam3_candidates": sum(len(proposal.get("sam3_candidates", [])) for proposal in proposals),
            "deferred_candidates": sum(len(proposal.get("deferred_candidates", [])) for proposal in proposals),
            "parser_rejections": sum(len(proposal.get("parser_rejections", [])) for proposal in proposals),
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
        random.Random(20260627).shuffle(to_process)
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
            print(f"[WARN] unreadable existing v4 result file; new dataset will be created: {result_path}")

    store = UnifiedStore(existing, manifest_videos)
    tasks, skipped, missing_files = select_tasks(manifest_videos, store)
    started_at = utc_now()
    started_perf = time.perf_counter()
    counts = {"success": 0, "no_sam3_candidate": 0, "failure": 0}

    dataset = write_all_outputs(store)
    atomic_write_json(Path(RUN_SUMMARY_PATH), build_summary(started_at, started_perf, len(manifest_videos), len(tasks), skipped, missing_files, counts, dataset, False))
    if not tasks:
        dataset = write_all_outputs(store)
        atomic_write_json(Path(RUN_SUMMARY_PATH), build_summary(started_at, started_perf, len(manifest_videos), 0, skipped, missing_files, counts, dataset, True))
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
            if completed % SAVE_EVERY == 0 or completed == len(tasks):
                dataset = write_all_outputs(store)
                atomic_write_json(Path(RUN_SUMMARY_PATH), build_summary(started_at, started_perf, len(manifest_videos), len(tasks), skipped, missing_files, counts, dataset, False))
            if completed % PROGRESS_EVERY == 0 or completed == len(tasks):
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
        print("\n[STOPPED] Completed records are preserved in unified JSON outputs.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
