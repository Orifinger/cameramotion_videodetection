#!/usr/bin/env python3
"""Qwen3-VL v4 candidate-concept discovery for CameraBench local editing.

This stage produces a compact high-recall bank of independently bounded visual
concepts for later SAM3 tracking. It does not choose a final focus region or an
editing operation. Source MP4s stay in /tmp; all outputs are unified JSON files.
"""

from __future__ import annotations

import asyncio
import argparse
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
REGION_SCHEMA_VERSION = "qwen_region_candidates_v4"
INVENTORY_RAW_SCHEMA_VERSION = "qwen_video_inventory_v2_raw"
INVENTORY_NORMALIZED_SCHEMA_VERSION = "qwen_video_inventory_v2_normalized"
DEFAULT_MAX_SAM3_CANDIDATES = MAX_SAM3_CANDIDATES
DEFAULT_MAX_DEFERRED_CANDIDATES = MAX_DEFERRED_CANDIDATES
PROMPT_PROFILE = "default"
MAX_INVENTORY_ENTITIES = 32

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
ALLOWED_INVENTORY_COARSE_TYPES = {
    "person", "animal", "vehicle", "object", "surface", "text_region",
    "food", "plant", "other", "unknown",
}
ALLOWED_INVENTORY_VISUAL_DOMAINS = {
    "real", "cartoon", "3d_render", "toy", "statue", "mannequin", "unclear",
}
ALLOWED_INVENTORY_PERSON_VIEWS = {"front", "back", "side", "three_quarter", "unclear", "not_person"}
ALLOWED_INVENTORY_SALIENCE = {"primary", "secondary", "background", "unclear"}
ALLOWED_INVENTORY_FOREGROUND = {"foreground", "midground", "background", "unclear"}
ALLOWED_INVENTORY_SIZE = {"large", "medium", "small", "tiny", "unclear"}
ALLOWED_INVENTORY_VISIBILITY = {"complete", "partial", "occluded", "truncated", "unclear"}
ALLOWED_INVENTORY_SUITABILITY = {"good", "maybe", "bad", "unclear"}
ALLOWED_INVENTORY_OPERATIONS = {
    "person_appearance_swap", "object_swap", "surface_attribute_edit",
    "surface_content_edit", "object_attribute_edit", "reserve", "blocked",
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

BASE_OBJECT_PROPOSAL_PROMPT = OBJECT_PROPOSAL_PROMPT
SUBJECT_FIRST_RERUN_APPENDIX = """

SUBJECT-FIRST RECOVERY RERUN MODE

This run is only for videos whose previous candidates were not suitable for
local AIGC editing. Prefer candidates that can create visible full-video fake
evidence after mask-based editing.

Priority order:
1) Independently bounded visible people that are not tiny. Include a person even
   if the previous run focused on a nearby object.
2) Large foreground movable objects such as bags, vehicles, furniture, tools,
   balls, instruments, appliances, or packages.
3) Whole editable carrier surfaces such as screens, posters, signs, books,
   paper, maps, framed art, apparel panels, vehicle panels, or package fronts.

Avoid tiny details, body parts, text glyphs, logos, groups, unbounded background
regions, and ambiguous generic prompts. Use short disambiguated SAM prompts for
multiple similar instances.
"""

INVENTORY_V2_PROMPT = f"""
You are building a video object inventory for a downstream SAM3 + VACE local
editing dataset.

This is NOT final pairing and NOT final editing. Your job is to list visible
entities in the complete video with high recall, especially foreground people
and main objects. Do not invent entities. A video may have zero usable entities.

Return a JSON object with:
- `schema_version`: "{INVENTORY_RAW_SCHEMA_VERSION}";
- `scene_summary`: one short sentence;
- `entities`: at most {MAX_INVENTORY_ENTITIES} entity objects;
- `no_editable_entity_reason`: null or a short reason.

For every visible entity that could plausibly be segmented or used as a donor
reference, provide:
- `entity_id`: a stable local id such as "entity_001";
- `coarse_type`: one of person, animal, vehicle, object, surface, text_region,
  food, plant, other, unknown;
- `fine_type_raw`: a short natural class name such as "adult man", "white car",
  "street light", "poster", "display screen";
- `visual_domain`: real, cartoon, 3d_render, toy, statue, mannequin, or unclear;
- `person_view`: front, back, side, three_quarter, unclear, or not_person;
- `salience`: primary, secondary, background, or unclear;
- `foreground_status`: foreground, midground, background, or unclear;
- `size_level`: large, medium, small, tiny, or unclear;
- `visibility`: complete, partial, occluded, truncated, or unclear;
- `edit_suitability`: good, maybe, bad, or unclear;
- `donor_suitability`: good, maybe, bad, or unclear;
- `sam3_prompt_phrase`: a short noun phrase for SAM3, never a long sentence;
- `suggested_operation`: person_appearance_swap, object_swap,
  surface_attribute_edit, surface_content_edit, object_attribute_edit, reserve,
  or blocked;
- `notes`: short evidence, without speculation;
- `uncertain_reason`: short reason when classification or suitability is unclear.

Important policy:
- Prefer listing a non-tiny visible person over tiny or odd objects.
- Distinguish real people from cartoon characters, 3D characters, statues,
  mannequins, toys, and people printed on screens/posters.
- Distinguish road vehicles, aircraft, and boats. Do not merge them into a
  generic vehicle when the subtype is visible.
- Distinguish outdoor street lights from indoor lamps.
- For screen/poster/sign/book/map, list the whole carrier surface, not only the
  internal text.
- Mark thin, tiny, heavily occluded, or ambiguous objects as `maybe` or `bad`.

Return JSON only. Do not include Markdown or explanations outside JSON.
"""

NORMALIZED_INVENTORY_V2_PROMPT = f"""
You are normalizing a video object inventory into a fixed taxonomy for a SAM3 +
VACE local editing dataset.

Use the following target taxonomy labels when possible:
- person.real.front, person.real.side, person.real.back, person.real.unclear
- person.cartoon, person.3d_character, person.statue_or_mannequin
- animal.generic
- vehicle.road.car, vehicle.road.bus_truck_van, vehicle.road.motorcycle_bicycle
- vehicle.air.aircraft, vehicle.water.boat
- surface.screen, surface.poster_sign, surface.book_paper_map, surface.framed_art
- object.lighting.street_light, object.lighting.indoor_lamp
- object.furniture.chair, object.furniture.table
- object.bag_suitcase, object.handheld, object.food, object.plant,
  object.sports_ball, object.generic
- unknown

Return a JSON object with:
- `schema_version`: "{INVENTORY_NORMALIZED_SCHEMA_VERSION}";
- `scene_summary`: one short sentence;
- `entities`: at most {MAX_INVENTORY_ENTITIES} entity objects;
- `no_editable_entity_reason`: null or a short reason.

Each entity must include the same fields as inventory_v2 plus:
- `taxonomy_label`: one of the labels above;
- `compatibility_group_hint`: a short group hint, such as person.real,
  vehicle.road.car_like, vehicle.air, surface, object.bag_suitcase, or unknown.

Rules:
- If a visible non-tiny real person exists, label it as a person and mark it as
  high priority for person_appearance_swap.
- Real people must not be normalized as cartoon/3D/statue/mannequin.
- Aircraft must not be normalized as road vehicles; road vehicles must not be
  normalized as aircraft.
- Street lights must not be normalized as indoor lamps.
- If no taxonomy label fits, use unknown and explain in `uncertain_reason`.
- `sam3_prompt_phrase` must be a short noun phrase suitable for SAM3.

Return JSON only. Do not include Markdown or explanations outside JSON.
"""


def build_object_proposal_prompt(profile: str) -> str:
    if profile == "inventory_v2":
        return INVENTORY_V2_PROMPT
    if profile == "normalized_inventory_v2":
        return NORMALIZED_INVENTORY_V2_PROMPT
    prompt = BASE_OBJECT_PROPOSAL_PROMPT.replace(
        f"Return at most {DEFAULT_MAX_SAM3_CANDIDATES} `sam3_candidates` and at most\n"
        f"{DEFAULT_MAX_DEFERRED_CANDIDATES} `deferred_candidates`.",
        f"Return at most {MAX_SAM3_CANDIDATES} `sam3_candidates` and at most\n"
        f"{MAX_DEFERRED_CANDIDATES} `deferred_candidates`.",
    )
    if profile == "subject_first_rerun":
        prompt += SUBJECT_FIRST_RERUN_APPENDIX
    return prompt


def schema_object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def build_inventory_json_schema() -> dict[str, Any]:
    normalized = PROMPT_PROFILE == "normalized_inventory_v2"
    entity_properties: dict[str, Any] = {
        "entity_id": {"type": "string"},
        "coarse_type": {"type": "string", "enum": sorted(ALLOWED_INVENTORY_COARSE_TYPES)},
        "fine_type_raw": {"type": "string"},
        "visual_domain": {"type": "string", "enum": sorted(ALLOWED_INVENTORY_VISUAL_DOMAINS)},
        "person_view": {"type": "string", "enum": sorted(ALLOWED_INVENTORY_PERSON_VIEWS)},
        "salience": {"type": "string", "enum": sorted(ALLOWED_INVENTORY_SALIENCE)},
        "foreground_status": {"type": "string", "enum": sorted(ALLOWED_INVENTORY_FOREGROUND)},
        "size_level": {"type": "string", "enum": sorted(ALLOWED_INVENTORY_SIZE)},
        "visibility": {"type": "string", "enum": sorted(ALLOWED_INVENTORY_VISIBILITY)},
        "edit_suitability": {"type": "string", "enum": sorted(ALLOWED_INVENTORY_SUITABILITY)},
        "donor_suitability": {"type": "string", "enum": sorted(ALLOWED_INVENTORY_SUITABILITY)},
        "sam3_prompt_phrase": {"type": "string"},
        "suggested_operation": {"type": "string", "enum": sorted(ALLOWED_INVENTORY_OPERATIONS)},
        "notes": {"type": "string"},
        "uncertain_reason": {"type": "string"},
    }
    required = list(entity_properties)
    if normalized:
        entity_properties["taxonomy_label"] = {"type": "string"}
        entity_properties["compatibility_group_hint"] = {"type": "string"}
        required.extend(["taxonomy_label", "compatibility_group_hint"])
    return schema_object(
        {
            "schema_version": {"type": "string", "enum": [SCHEMA_VERSION]},
            "scene_summary": {"type": "string"},
            "entities": {
                "type": "array",
                "maxItems": MAX_INVENTORY_ENTITIES,
                "items": schema_object(entity_properties, required),
            },
            "no_editable_entity_reason": {"type": ["string", "null"]},
        },
        ["schema_version", "scene_summary", "entities", "no_editable_entity_reason"],
    )


def build_json_schema() -> dict[str, Any]:
    if PROMPT_PROFILE in {"inventory_v2", "normalized_inventory_v2"}:
        return build_inventory_json_schema()
    return schema_object(
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


JSON_SCHEMA: dict[str, Any] = build_json_schema()


class ProposalError(RuntimeError):
    """A model response cannot be converted into a valid proposal record."""


class ResponseFormatError(ProposalError):
    """The server answered, but the JSON shape does not match this profile."""


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


def first_clean_text(raw: dict[str, Any], keys: tuple[str, ...], limit: int) -> str:
    for key in keys:
        text = clean_text(raw.get(key), limit)
        if text:
            return text
    return ""


def coarse_type_from_text(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ("person", "man", "woman", "boy", "girl", "human")):
        return "person"
    if any(word in lowered for word in ("dog", "cat", "horse", "bird", "animal")):
        return "animal"
    if any(word in lowered for word in ("car", "bus", "truck", "van", "bicycle", "motorcycle", "boat", "aircraft", "plane", "vehicle")):
        return "vehicle"
    if any(word in lowered for word in ("screen", "poster", "sign", "book", "paper", "map", "painting", "panel")):
        return "surface"
    if any(word in lowered for word in ("food", "fruit", "meal", "cake")):
        return "food"
    if any(word in lowered for word in ("plant", "tree", "flower")):
        return "plant"
    return "object" if lowered else "unknown"


def inventory_operation_from_coarse_type(coarse_type: str) -> str:
    if coarse_type == "person":
        return "person_appearance_swap"
    if coarse_type == "surface":
        return "surface_content_edit"
    if coarse_type in {"animal", "vehicle", "object", "food", "plant"}:
        return "object_swap"
    return "reserve"


def inventory_entity_from_sam3_candidate(raw_item: dict[str, Any], index: int) -> dict[str, Any]:
    candidate_class = clean_text(raw_item.get("candidate_class"), 80).lower()
    fine_type = first_clean_text(
        raw_item,
        ("canonical_concept", "sam_prompt", "display_phrase", "candidate_class"),
        120,
    ).lower()
    if candidate_class == "human":
        coarse_type = "person"
    elif candidate_class in {"display_screen", "sign_or_poster", "paper_book_map", "framed_art", "apparel_panel", "vehicle_panel", "package_front"}:
        coarse_type = "surface"
    elif candidate_class == "animal":
        coarse_type = "animal"
    elif candidate_class == "vehicle":
        coarse_type = "vehicle"
    else:
        coarse_type = coarse_type_from_text(fine_type)
    return {
        "entity_id": clean_text(raw_item.get("candidate_id"), 64) or f"entity_{index:03d}",
        "coarse_type": coarse_type,
        "fine_type_raw": fine_type,
        "visual_domain": "unclear",
        "person_view": "unclear" if coarse_type == "person" else "not_person",
        "salience": "unclear",
        "foreground_status": "unclear",
        "size_level": "unclear",
        "visibility": "unclear",
        "edit_suitability": "maybe",
        "donor_suitability": "maybe",
        "sam3_prompt_phrase": first_clean_text(raw_item, ("sam_prompt", "canonical_concept", "display_phrase"), 120) or fine_type,
        "suggested_operation": inventory_operation_from_coarse_type(coarse_type),
        "notes": clean_text(raw_item.get("display_phrase"), 300),
        "uncertain_reason": "converted_from_sam3_candidates_fallback",
    }


def inventory_entities_from_raw(raw: dict[str, Any]) -> tuple[list[Any], str]:
    for key in ("entities", "visible_entities", "objects", "object_inventory", "inventory", "items"):
        value = raw.get(key)
        if isinstance(value, list):
            return value, key
    sam3_candidates = raw.get("sam3_candidates")
    if isinstance(sam3_candidates, list):
        return [
            inventory_entity_from_sam3_candidate(item, index)
            for index, item in enumerate(sam3_candidates, start=1)
            if isinstance(item, dict)
        ], "sam3_candidates"
    keys = ", ".join(sorted(str(key) for key in raw.keys())[:30])
    raise ResponseFormatError(
        "Expected list field 'entities' or alias "
        "'visible_entities'/'objects'/'object_inventory'/'inventory'/'items'/'sam3_candidates'; "
        f"top_level_keys=[{keys}]"
    )


def normalize_inventory_proposal(raw: dict[str, Any]) -> dict[str, Any]:
    raw_entities, entity_source_field = inventory_entities_from_raw(raw)
    normalized_profile = PROMPT_PROFILE == "normalized_inventory_v2"
    entities: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_prompts: set[tuple[str, str]] = set()
    for raw_item in raw_entities:
        if not isinstance(raw_item, dict):
            rejections.append({"reason": "unsupported_entity_type", "detail": type(raw_item).__name__})
            continue
        raw_id = clean_text(raw_item.get("entity_id"), 64)
        entity_id = raw_id or f"entity_{len(entities) + 1:03d}"
        if entity_id in seen_ids:
            entity_id = f"entity_{len(entities) + 1:03d}"
        fine_type_raw = first_clean_text(
            raw_item,
            ("fine_type_raw", "fine_type", "class", "category", "type", "name", "object_name", "canonical_concept", "description"),
            120,
        ).lower()
        prompt = first_clean_text(
            raw_item,
            ("sam3_prompt_phrase", "sam_prompt", "prompt", "canonical_concept", "name", "object_name", "fine_type_raw", "description"),
            120,
        )
        coarse_type = enum_or_default(
            raw_item.get("coarse_type") or raw_item.get("coarse_category") or raw_item.get("category_type"),
            ALLOWED_INVENTORY_COARSE_TYPES,
            coarse_type_from_text(fine_type_raw or prompt),
        )
        if not fine_type_raw and not prompt:
            rejections.append({"entity_id": entity_id, "reason": "missing_fine_type_and_sam3_prompt"})
            continue
        prompt_key = (coarse_type, prompt.lower())
        if prompt and prompt_key in seen_prompts:
            rejections.append({"entity_id": entity_id, "reason": "duplicate_sam3_prompt_phrase", "sam3_prompt_phrase": prompt})
            continue
        seen_ids.add(entity_id)
        if prompt:
            seen_prompts.add(prompt_key)
        entity = {
            "entity_id": entity_id,
            "coarse_type": coarse_type,
            "fine_type_raw": fine_type_raw,
            "visual_domain": enum_or_default(raw_item.get("visual_domain"), ALLOWED_INVENTORY_VISUAL_DOMAINS, "unclear"),
            "person_view": enum_or_default(raw_item.get("person_view"), ALLOWED_INVENTORY_PERSON_VIEWS, "unclear"),
            "salience": enum_or_default(raw_item.get("salience"), ALLOWED_INVENTORY_SALIENCE, "unclear"),
            "foreground_status": enum_or_default(raw_item.get("foreground_status"), ALLOWED_INVENTORY_FOREGROUND, "unclear"),
            "size_level": enum_or_default(raw_item.get("size_level"), ALLOWED_INVENTORY_SIZE, "unclear"),
            "visibility": enum_or_default(raw_item.get("visibility"), ALLOWED_INVENTORY_VISIBILITY, "unclear"),
            "edit_suitability": enum_or_default(raw_item.get("edit_suitability"), ALLOWED_INVENTORY_SUITABILITY, "unclear"),
            "donor_suitability": enum_or_default(raw_item.get("donor_suitability"), ALLOWED_INVENTORY_SUITABILITY, "unclear"),
            "sam3_prompt_phrase": prompt or fine_type_raw,
            "suggested_operation": enum_or_default(raw_item.get("suggested_operation"), ALLOWED_INVENTORY_OPERATIONS, "reserve"),
            "notes": clean_text(raw_item.get("notes"), 300),
            "uncertain_reason": clean_text(raw_item.get("uncertain_reason"), 220),
        }
        if normalized_profile:
            entity["taxonomy_label"] = clean_text(raw_item.get("taxonomy_label"), 120).lower() or "unknown"
            entity["compatibility_group_hint"] = clean_text(raw_item.get("compatibility_group_hint"), 120).lower() or "unknown"
        entities.append(entity)
        if len(entities) >= MAX_INVENTORY_ENTITIES:
            break
    return {
        "schema_version": SCHEMA_VERSION,
        "scene_summary": clean_text(raw.get("scene_summary"), 400),
        "entities": entities,
        "no_editable_entity_reason": clean_text(raw.get("no_editable_entity_reason"), 200) or None,
        "entity_source_field": entity_source_field,
        "parser_rejections": rejections,
    }


def normalize_model_response(raw: dict[str, Any]) -> dict[str, Any]:
    if PROMPT_PROFILE in {"inventory_v2", "normalized_inventory_v2"}:
        return normalize_inventory_proposal(raw)
    return normalize_proposal(raw)


def proposal_has_primary_output(proposal: dict[str, Any]) -> bool:
    if "entities" in proposal:
        return bool(proposal.get("entities"))
    return bool(proposal.get("sam3_candidates"))


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
    if not isinstance(proposal, dict) or not isinstance(proposal.get("parser_rejections"), list):
        return False
    if PROMPT_PROFILE in {"inventory_v2", "normalized_inventory_v2"}:
        return isinstance(proposal.get("entities"), list)
    return isinstance(proposal.get("sam3_candidates"), list) and isinstance(proposal.get("deferred_candidates"), list)


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
                    proposal = normalize_model_response(parsed)

                status = "success" if proposal_has_primary_output(proposal) else "no_sam3_candidate"
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
                if isinstance(exc, ResponseFormatError):
                    break
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

    if PROMPT_PROFILE in {"inventory_v2", "normalized_inventory_v2"}:
        inventory_videos: list[dict[str, Any]] = []
        flat_entities: list[dict[str, Any]] = []
        for record in all_records:
            proposal = record.get("proposal")
            entities = proposal.get("entities") if isinstance(proposal, dict) else None
            if not isinstance(entities, list):
                continue
            packed_entities = []
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                item = dict(entity)
                item["video_id"] = record["video_id"]
                item["video_path"] = record.get("video_path")
                item["relative_path"] = record.get("relative_path")
                flat_entities.append(item)
                packed_entities.append(entity)
            inventory_videos.append({
                "video_id": record["video_id"],
                "relative_path": record.get("relative_path"),
                "video_path": record.get("video_path"),
                "status": record["status"],
                "scene_summary": proposal.get("scene_summary") if isinstance(proposal, dict) else None,
                "entities": packed_entities,
            })
        atomic_write_json(Path(ALL_CANDIDATES_PATH).with_name("qwen_inventory_entities.json"), {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": utc_now(),
            "source_file": str(ALL_CANDIDATES_PATH),
            "num_videos": len(inventory_videos),
            "num_entities": len(flat_entities),
            "videos": inventory_videos,
            "entities": flat_entities,
        })

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
    entities = [
        entity
        for proposal in proposals
        for entity in (proposal.get("entities") or [])
        if isinstance(entity, dict)
    ]
    return {
        "schema_version": "qwen_region_candidate_run_summary_v4",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now() if finished else None,
        "config": {
            "api_base": QWEN_API_BASE,
            "model": QWEN_MODEL_NAME,
            "max_concurrency": MAX_CONCURRENCY,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "max_retries": MAX_RETRIES,
            "max_sam3_candidates": MAX_SAM3_CANDIDATES,
            "max_deferred_candidates": MAX_DEFERRED_CANDIDATES,
            "prompt_profile": PROMPT_PROFILE,
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
            "inventory_entities": len(entities),
            "inventory_unknown_taxonomy": sum(str(entity.get("taxonomy_label") or "") == "unknown" for entity in entities),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL object proposal discovery.")
    parser.add_argument("--manifest", type=Path, default=None, help="Override video manifest path.")
    parser.add_argument("--out-root", type=Path, default=None, help="Override Qwen output directory.")
    parser.add_argument("--api-base", default=None, help="Override OpenAI-compatible Qwen API base.")
    parser.add_argument("--model-name", default=None, help="Override served Qwen model name.")
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None, help="Limit videos for this invocation.")
    parser.add_argument("--all-videos", action="store_true", help="Process all videos from the manifest.")
    parser.add_argument("--max-sam3-candidates", type=int, default=None)
    parser.add_argument("--max-deferred-candidates", type=int, default=None)
    parser.add_argument(
        "--prompt-profile",
        choices=("default", "subject_first_rerun", "inventory_v2", "normalized_inventory_v2"),
        default="default",
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--no-retry-failures", action="store_true")
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    global MANIFEST_PATH, QWEN_API_BASE, QWEN_MODEL_NAME, MAX_CONCURRENCY
    global MAX_OUTPUT_TOKENS, MAX_VIDEOS, MAX_SAM3_CANDIDATES, MAX_DEFERRED_CANDIDATES
    global OVERWRITE_EXISTING, RETRY_FAILURE_RECORDS, PROMPT_PROFILE
    global ALL_CANDIDATES_PATH, SAM3_CANDIDATES_PATH, SCENE_TEXT_GRAPHIC_PATH
    global SCREEN_OVERLAY_PATH, PERSISTENT_WATERMARK_PATH, PARSER_REJECTIONS_PATH
    global RUN_SUMMARY_PATH, OBJECT_PROPOSAL_PROMPT, JSON_SCHEMA
    global SCHEMA_VERSION

    PROMPT_PROFILE = str(args.prompt_profile)
    if PROMPT_PROFILE == "inventory_v2":
        SCHEMA_VERSION = INVENTORY_RAW_SCHEMA_VERSION
    elif PROMPT_PROFILE == "normalized_inventory_v2":
        SCHEMA_VERSION = INVENTORY_NORMALIZED_SCHEMA_VERSION
    else:
        SCHEMA_VERSION = REGION_SCHEMA_VERSION
    if args.manifest is not None:
        MANIFEST_PATH = Path(args.manifest)
    if args.out_root is not None or PROMPT_PROFILE in {"inventory_v2", "normalized_inventory_v2"}:
        out_root = Path(args.out_root) if args.out_root is not None else PROJECT_ROOT / "res" / "qwen_inventory_v2"
        if PROMPT_PROFILE == "inventory_v2":
            ALL_CANDIDATES_PATH = out_root / "qwen_inventory_v2_raw.json"
            SAM3_CANDIDATES_PATH = out_root / "qwen_sam3_candidates_inventory_v2_raw.json"
        elif PROMPT_PROFILE == "normalized_inventory_v2":
            ALL_CANDIDATES_PATH = out_root / "qwen_inventory_v2_normalized.json"
            SAM3_CANDIDATES_PATH = out_root / "qwen_sam3_candidates_inventory_v2.json"
        else:
            ALL_CANDIDATES_PATH = out_root / "qwen_region_candidates_all.json"
            SAM3_CANDIDATES_PATH = out_root / "qwen_sam3_candidates.json"
        SCENE_TEXT_GRAPHIC_PATH = out_root / "qwen_deferred_scene_text_graphic.json"
        SCREEN_OVERLAY_PATH = out_root / "qwen_deferred_screen_overlay.json"
        PERSISTENT_WATERMARK_PATH = out_root / "qwen_deferred_persistent_watermark.json"
        PARSER_REJECTIONS_PATH = out_root / "qwen_parser_rejections.json"
        RUN_SUMMARY_PATH = out_root / "qwen_run_summary.json"
    if args.api_base:
        QWEN_API_BASE = str(args.api_base)
    if args.model_name:
        QWEN_MODEL_NAME = str(args.model_name)
    if args.max_concurrency is not None:
        if args.max_concurrency <= 0:
            raise ValueError("--max-concurrency must be positive")
        MAX_CONCURRENCY = int(args.max_concurrency)
    if args.max_output_tokens is not None:
        if args.max_output_tokens <= 0:
            raise ValueError("--max-output-tokens must be positive")
        MAX_OUTPUT_TOKENS = int(args.max_output_tokens)
    if args.all_videos:
        MAX_VIDEOS = None
    elif args.max_videos is not None:
        if args.max_videos < 0:
            raise ValueError("--max-videos must be non-negative")
        MAX_VIDEOS = int(args.max_videos)
    if args.max_sam3_candidates is not None:
        if args.max_sam3_candidates <= 0:
            raise ValueError("--max-sam3-candidates must be positive")
        MAX_SAM3_CANDIDATES = int(args.max_sam3_candidates)
    if args.max_deferred_candidates is not None:
        if args.max_deferred_candidates < 0:
            raise ValueError("--max-deferred-candidates must be non-negative")
        MAX_DEFERRED_CANDIDATES = int(args.max_deferred_candidates)
    if args.overwrite_existing:
        OVERWRITE_EXISTING = True
    if args.no_retry_failures:
        RETRY_FAILURE_RECORDS = False
    OBJECT_PROPOSAL_PROMPT = build_object_proposal_prompt(PROMPT_PROFILE)
    JSON_SCHEMA = build_json_schema()


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
    args = parse_args()
    configure_runtime(args)
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[STOPPED] Completed records are preserved in unified JSON outputs.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
