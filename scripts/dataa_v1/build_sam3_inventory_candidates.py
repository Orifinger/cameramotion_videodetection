#!/usr/bin/env python3
"""Convert Qwen inventory-v2 entities into SAM3 text-prompt candidates."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.common import DataAError, read_json, utc_now_iso, write_json


DEFAULT_INVENTORY = Path("res/qwen_inventory_v2/qwen_inventory_v2_normalized.json")
DEFAULT_OUT = Path("res/qwen_inventory_v2/qwen_sam3_candidates_inventory_v2.json")
SAM3_SCHEMA_VERSION = "qwen_region_candidates_v4"

SURFACE_LABEL_PREFIXES = ("surface.",)
PERSON_LABEL_PREFIXES = ("person.real.", "person.cartoon", "person.3d_character")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_project_path(path: Path | str | None) -> Path | None:
    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else _repo_root() / value


def _clean(value: Any, *, max_len: int = 160) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())[:max_len]


def _iter_inventory_videos(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise DataAError("inventory JSON must be an object")
    videos = payload.get("videos")
    if isinstance(videos, list):
        return [dict(item) for item in videos if isinstance(item, Mapping)]
    entities = payload.get("entities")
    if isinstance(entities, list):
        by_video: dict[str, dict[str, Any]] = {}
        for entity in entities:
            if not isinstance(entity, Mapping) or not entity.get("video_id"):
                continue
            video_id = str(entity["video_id"])
            video = by_video.setdefault(
                video_id,
                {
                    "video_id": video_id,
                    "video_path": entity.get("video_path"),
                    "relative_path": entity.get("relative_path"),
                    "status": "success",
                    "entities": [],
                },
            )
            video["entities"].append(dict(entity))
        return list(by_video.values())
    raise DataAError("inventory JSON must contain videos[] or flat entities[]")


def _region_class_for_entity(entity: Mapping[str, Any]) -> tuple[str, str, str]:
    label = _clean(entity.get("taxonomy_label"), max_len=120).lower()
    coarse = _clean(entity.get("coarse_type"), max_len=80).lower()
    if label.startswith(SURFACE_LABEL_PREFIXES) or coarse in {"surface", "text_region"}:
        if label.startswith("surface.screen"):
            return "editable_surface", "display_screen", "whole_surface"
        if label.startswith("surface.poster_sign"):
            return "editable_surface", "sign_or_poster", "whole_surface"
        if label.startswith("surface.book_paper_map"):
            return "editable_surface", "paper_book_map", "whole_surface"
        if label.startswith("surface.framed_art"):
            return "editable_surface", "framed_art", "whole_surface"
        return "editable_surface", "sign_or_poster", "whole_surface"
    if label.startswith(PERSON_LABEL_PREFIXES) or coarse == "person":
        return "physical_instance", "human", "whole_instance"
    if label.startswith("animal.") or coarse == "animal":
        return "physical_instance", "animal", "whole_instance"
    if label.startswith("vehicle.") or coarse == "vehicle":
        return "physical_instance", "vehicle", "whole_instance"
    if coarse in {"object", "food", "plant", "other"} or label.startswith(("object.", "person.statue_or_mannequin")):
        if "handheld" in label:
            return "physical_instance", "handheld_object", "whole_instance"
        return "physical_instance", "bounded_object", "whole_instance"
    return "physical_instance", "bounded_object", "whole_instance"


def _keep_entity(entity: Mapping[str, Any], *, include_bad: bool) -> tuple[bool, str]:
    prompt = _clean(entity.get("sam3_prompt_phrase"), max_len=120)
    if not prompt:
        return False, "missing_sam3_prompt_phrase"
    label = _clean(entity.get("taxonomy_label"), max_len=120).lower()
    if label == "unknown":
        return False, "unknown_taxonomy"
    if not include_bad:
        if _clean(entity.get("edit_suitability")).lower() == "bad" and _clean(entity.get("donor_suitability")).lower() == "bad":
            return False, "edit_and_donor_bad"
        if _clean(entity.get("size_level")).lower() == "tiny":
            return False, "tiny_entity"
    return True, "kept"


def _candidate_from_entity(entity: Mapping[str, Any], index: int) -> dict[str, Any]:
    family, candidate_class, target_scope = _region_class_for_entity(entity)
    entity_id = _clean(entity.get("entity_id"), max_len=64) or f"entity_{index:03d}"
    prompt = _clean(entity.get("sam3_prompt_phrase"), max_len=120)
    fine = _clean(entity.get("fine_type_raw"), max_len=120).lower()
    display = _clean(entity.get("notes"), max_len=180) or fine or prompt
    temporal_presence = "mostly"
    if _clean(entity.get("visibility")).lower() in {"partial", "occluded", "truncated"}:
        temporal_presence = "middle"
    return {
        "candidate_id": entity_id,
        "region_family": family,
        "candidate_class": candidate_class,
        "target_scope": target_scope,
        "canonical_concept": fine or prompt.lower(),
        "display_phrase": display,
        "sam_prompt": prompt,
        "instance_count_hint": "possibly_multiple",
        "visual_disambiguators": [
            value
            for value in [
                _clean(entity.get("visual_domain"), max_len=40),
                _clean(entity.get("person_view"), max_len=40),
                _clean(entity.get("foreground_status"), max_len=40),
            ]
            if value and value not in {"unclear", "not_person"}
        ][:4],
        "screen_region": "unknown",
        "temporal_presence": temporal_presence,
        "inventory_entity": dict(entity),
    }


def build_candidates(
    *,
    inventory: Path,
    out_path: Path,
    include_bad: bool = False,
    max_candidates_per_video: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    payload = read_json(inventory)
    videos = _iter_inventory_videos(payload)
    out_videos: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    kept_counts: Counter[str] = Counter()
    for video in videos:
        entities = video.get("entities")
        if not isinstance(entities, list):
            continue
        candidates: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        for entity in entities:
            if not isinstance(entity, Mapping):
                continue
            keep, reason = _keep_entity(entity, include_bad=include_bad)
            if not keep:
                rejection_counts[reason] += 1
                rejections.append({"entity_id": entity.get("entity_id"), "reason": reason})
                continue
            candidates.append(_candidate_from_entity(entity, len(candidates) + 1))
            label = _clean(entity.get("taxonomy_label"), max_len=120).lower() or "<missing>"
            kept_counts[label] += 1
            if max_candidates_per_video is not None and len(candidates) >= max_candidates_per_video:
                break
        if candidates:
            out_videos.append(
                {
                    "video_id": video.get("video_id"),
                    "relative_path": video.get("relative_path"),
                    "video_path": video.get("video_path"),
                    "status": "success",
                    "sam3_candidates": candidates,
                    "inventory_rejections": rejections,
                }
            )
    result = {
        "schema_version": SAM3_SCHEMA_VERSION,
        "generated_at_utc": utc_now_iso(),
        "source_inventory": str(inventory),
        "num_videos": len(out_videos),
        "num_candidates": sum(len(video["sam3_candidates"]) for video in out_videos),
        "rejection_counts": dict(rejection_counts),
        "kept_taxonomy_counts": dict(kept_counts.most_common()),
        "videos": out_videos,
    }
    if not dry_run:
        write_json(out_path, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--include-bad", action="store_true")
    parser.add_argument("--max-candidates-per-video", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_candidates(
            inventory=_resolve_project_path(args.inventory) or args.inventory,
            out_path=_resolve_project_path(args.out) or args.out,
            include_bad=bool(args.include_bad),
            max_candidates_per_video=args.max_candidates_per_video,
            dry_run=bool(args.dry_run),
        )
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        "sam3_inventory_candidates "
        f"dry_run={args.dry_run} videos={result['num_videos']} "
        f"candidates={result['num_candidates']} out={args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
