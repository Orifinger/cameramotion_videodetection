#!/usr/bin/env python3
"""Build Data A v1 editability catalog and donor-pair pool.

Inputs
------
1) sam3_quality_tracks_enriched.json
2) video_domain_index_v1.json (video-level labels, created by a short Qwen3-VL pass)

Outputs
-------
- generator_registry_v1.json
- operation_registry_v1.json
- candidate_operation_rules_v1.json
- track_editability_catalog_v1.json
- donor_pair_pool_v1.json
- pairing_stats_v1.json

This script does NOT render donor crops and does NOT generate videos. It only builds
reproducible candidate metadata. It intentionally leaves face swapping and insertion
as deferred routes until a face-track / placement-site pipeline exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

GENERATOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    "wan2.1_vace_14b": {
        "display_name": "Wan2.1-VACE-14B",
        "enabled": True,
        "smoke_status": "pending",
        "operations": [
            "person_appearance_swap", "object_swap", "object_attribute_edit",
            "surface_content_edit", "surface_attribute_edit",
        ],
        "requires_sam3_mask": True,
        "requires_donor_for_reference_routes": True,
        "notes": "Primary mask-native local video editor for v1.",
    },
    "videopainter": {
        "display_name": "VideoPainter",
        "enabled": False,
        "enabled_after_smoke_test": True,
        "smoke_status": "pending",
        "operations": ["object_removal"],
        "requires_sam3_mask": True,
        "requires_donor_for_reference_routes": False,
        "notes": "Removal-only route; no donor asset is required.",
    },
    "pisco_14b": {
        "display_name": "PISCO-14B",
        "enabled": False,
        "enabled_after_smoke_test": True,
        "smoke_status": "pending",
        "operations": ["object_insertion"],
        "requires_sam3_mask": False,
        "requires_placement_site": True,
        "requires_donor_for_reference_routes": True,
        "notes": "Insertion route; requires a separate placement-site tube.",
    },
    "facefusion": {
        "display_name": "FaceFusion",
        "enabled": False,
        "enabled_after_smoke_test": True,
        "smoke_status": "pending",
        "operations": ["face_identity_swap"],
        "requires_sam3_mask": False,
        "requires_face_track": True,
        "requires_donor_for_reference_routes": True,
        "notes": "Face route; only use consented or synthetic donor identities.",
    },
}

OPERATION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "face_identity_swap": {
        "global_weight": 0.16,
        "routes": [{"route_id": "facefusion_face_swap", "weight": 1.0}],
        "requires_donor": True,
        "requires_face_track": True,
        "requires_sam3_track": False,
        "status": "deferred_until_face_pipeline",
    },
    "person_appearance_swap": {
        "global_weight": 0.14,
        "routes": [
            {"route_id": "vace14b_person_reference_swap", "weight": 0.75},
            {"route_id": "vace14b_person_text_appearance", "weight": 0.25},
        ],
        "requires_donor": False,
        "requires_sam3_track": True,
        "status": "enabled_after_vace_smoke",
    },
    "object_swap": {
        "global_weight": 0.25,
        "routes": [
            {"route_id": "vace14b_object_reference_swap", "weight": 0.70},
            {"route_id": "vace14b_object_text_swap", "weight": 0.30},
        ],
        "requires_donor": False,
        "requires_sam3_track": True,
        "status": "enabled_after_vace_smoke",
    },
    "object_attribute_edit": {
        "global_weight": 0.10,
        "routes": [
            {"route_id": "vace14b_object_attribute_text", "weight": 0.60},
            {"route_id": "vace14b_object_attribute_reference", "weight": 0.40},
        ],
        "requires_donor": False,
        "requires_sam3_track": True,
        "status": "enabled_after_vace_smoke",
    },
    "surface_content_edit": {
        "global_weight": 0.15,
        "routes": [
            {"route_id": "vace14b_surface_reference_content", "weight": 0.80},
            {"route_id": "vace14b_surface_text_content", "weight": 0.20},
        ],
        "requires_donor": False,
        "requires_sam3_track": True,
        "status": "enabled_after_vace_smoke",
    },
    "surface_attribute_edit": {
        "global_weight": 0.07,
        "routes": [
            {"route_id": "vace14b_surface_attribute_text", "weight": 0.60},
            {"route_id": "vace14b_surface_attribute_reference", "weight": 0.40},
        ],
        "requires_donor": False,
        "requires_sam3_track": True,
        "status": "enabled_after_vace_smoke",
    },
    "object_insertion": {
        "global_weight": 0.07,
        "routes": [{"route_id": "pisco_reference_insertion", "weight": 1.0}],
        "requires_donor": True,
        "requires_placement_site": True,
        "requires_sam3_track": False,
        "status": "deferred_until_placement_pipeline",
    },
    "object_removal": {
        "global_weight": 0.06,
        "routes": [{"route_id": "videopainter_removal", "weight": 1.0}],
        "requires_donor": False,
        "requires_sam3_track": True,
        "status": "enabled_after_videopainter_smoke",
    },
}

CANDIDATE_OPERATION_RULES: Dict[str, Dict[str, Any]] = {
    "human": {
        "coarse_group": "human",
        "operations": {"person_appearance_swap": 1.00},
        "pairing_key": "human",
        "notes": "Face swap is not attached here; it needs a dedicated face track.",
    },
    "bounded_object": {
        "coarse_group": "generic_object",
        "operations": {"object_swap": 1.00, "object_attribute_edit": 0.65, "object_removal": 0.25},
        "pairing_key": "semantic_bucket",
    },
    "handheld_object": {
        "coarse_group": "handheld_object",
        "operations": {"object_swap": 0.80, "object_attribute_edit": 0.60},
        "pairing_key": "semantic_bucket",
        "notes": "Later reject/penalize if hand contact is severe.",
    },
    "vehicle": {
        "coarse_group": "vehicle",
        "operations": {"object_swap": 0.55, "object_attribute_edit": 1.00},
        "pairing_key": "vehicle",
    },
    "display_screen": {
        "coarse_group": "screen_or_display",
        "operations": {"surface_content_edit": 1.00, "surface_attribute_edit": 0.35},
        "pairing_key": "surface_subtype",
    },
    "sign_or_poster": {
        "coarse_group": "sign_or_poster",
        "operations": {"surface_content_edit": 1.00, "surface_attribute_edit": 0.45},
        "pairing_key": "surface_subtype",
    },
    "framed_art": {
        "coarse_group": "framed_visual",
        "operations": {"surface_content_edit": 1.00},
        "pairing_key": "surface_subtype",
    },
    "paper_book_map": {
        "coarse_group": "paper_or_map",
        "operations": {"surface_content_edit": 0.85},
        "pairing_key": "surface_subtype",
    },
    "apparel_panel": {
        "coarse_group": "apparel_surface",
        "operations": {"surface_attribute_edit": 1.00},
        "pairing_key": "surface_subtype",
    },
}

OBJECT_BUCKET_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("cup_mug", ("cup", "mug", "tumbler", "goblet")),
    ("bottle_container", ("bottle", "jar", "can", "container", "thermos")),
    ("book_notebook", ("book", "notebook", "magazine", "journal")),
    ("backpack_bag", ("backpack", "bag", "purse", "handbag", "suitcase", "luggage")),
    ("ball", ("ball", "football", "soccer", "basketball", "baseball")),
    ("toy_plush", ("toy", "stuffed", "plush", "doll", "figurine")),
    ("box_package", ("box", "package", "carton", "case")),
    ("phone", ("phone", "smartphone", "mobile phone")),
    ("chair_stool", ("chair", "stool", "seat", "lounge chair", "pool chair")),
    ("lamp", ("lamp", "light", "street lamp")),
    ("small_appliance", ("kettle", "toaster", "camera", "speaker", "headphone", "computer")),
    ("table", ("table", "desk", "counter")),
    ("decor_object", ("statue", "vase", "plant pot", "clock", "frame")),
]

SURFACE_SUBTYPE_RULES: Dict[str, str] = {
    "display_screen": "screen",
    "sign_or_poster": "sign_or_poster",
    "framed_art": "framed_visual",
    "paper_book_map": "paper_or_map",
    "apparel_panel": "apparel",
}

ALLOWED_DOMAINS = {"real_live_action", "animation_cartoon", "game_scene", "cg_rendered", "mixed", "unknown"}


def stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def normalise_text(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").split())


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def pick_quality(track: Mapping[str, Any]) -> float:
    for key in ("track_quality_score", "quality_score", "score"):
        if key in track and track.get(key) is not None:
            return clamp(safe_float(track.get(key), 0.5))
    visible = safe_float(track.get("visible_frame_ratio"), 0.5)
    border = safe_float(track.get("border_touch_ratio"), 0.0)
    return clamp(0.80 * visible + 0.20 * (1.0 - border))


def infer_semantic_bucket(candidate_class: str, concept: str) -> str:
    c = normalise_text(concept)
    if candidate_class == "human":
        return "human"
    if candidate_class == "vehicle":
        return "vehicle"
    if candidate_class == "animal":
        return "animal"
    if candidate_class in SURFACE_SUBTYPE_RULES:
        return SURFACE_SUBTYPE_RULES[candidate_class]
    if candidate_class in {"bounded_object", "handheld_object"}:
        for bucket, words in OBJECT_BUCKET_RULES:
            if any(word in c for word in words):
                return bucket
        return "generic_unknown"
    return "unknown"


def load_domains(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("videos"), list):
        rows = payload["videos"]
    elif isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = [{"video_id": k, **(v if isinstance(v, dict) else {})} for k, v in payload.items()]
    else:
        raise ValueError(f"Unsupported domain index schema in {path}")
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("video_id"):
            continue
        domain = str(row.get("content_domain") or row.get("style_domain") or "unknown").strip().lower().replace(" ", "_")
        if domain not in ALLOWED_DOMAINS:
            domain = "unknown"
        style_domain = str(row.get("style_domain") or domain).strip().lower().replace(" ", "_")
        out[str(row["video_id"])] = {
            "content_domain": domain,
            "style_domain": style_domain,
            "domain_confidence": clamp(safe_float(row.get("domain_confidence"), 0.0)),
            "domain_status": row.get("status", "unreviewed"),
        }
    return out


def build_track_record(track: Mapping[str, Any], domain_meta: Mapping[str, Any]) -> Dict[str, Any]:
    candidate_class = str(track.get("candidate_class") or "unknown")
    rule = CANDIDATE_OPERATION_RULES.get(candidate_class)
    concept = str(track.get("canonical_concept") or track.get("display_phrase") or "unknown")
    quality = pick_quality(track)
    domain = str(domain_meta.get("content_domain", "unknown"))
    semantic_bucket = infer_semantic_bucket(candidate_class, concept)
    surface_subtype = SURFACE_SUBTYPE_RULES.get(candidate_class)
    base: Dict[str, Any] = {
        "video_id": str(track.get("video_id")),
        "video_path": track.get("video_path"),
        "relative_path": track.get("relative_path"),
        "track_id": str(track.get("track_id")),
        "candidate_id": track.get("candidate_id"),
        "region_family": track.get("region_family"),
        "candidate_class": candidate_class,
        "canonical_concept": concept,
        "display_phrase": track.get("display_phrase"),
        "sam_prompt": track.get("sam_prompt"),
        "mask_tube_path": track.get("mask_tube_path"),
        "bbox_tube_xywh": track.get("bbox_tube_xywh"),
        "track_quality_score": quality,
        "content_domain": domain,
        "style_domain": domain_meta.get("style_domain", domain),
        "domain_confidence": domain_meta.get("domain_confidence", 0.0),
        "semantic_bucket": semantic_bucket,
        "surface_subtype": surface_subtype,
        "coarse_group": rule["coarse_group"] if rule else "unsupported",
        "eligible": bool(rule),
        "eligibility_reason": "supported_candidate_class" if rule else "unsupported_candidate_class",
        "editable_operations": [],
    }
    if not rule:
        return base
    for op, class_weight in rule["operations"].items():
        op_cfg = OPERATION_REGISTRY[op]
        semantic_ok = not (op == "object_swap" and semantic_bucket == "generic_unknown")
        op_weight = op_cfg["global_weight"] * float(class_weight) * quality
        status = "eligible" if semantic_ok else "text_only_or_manual_review_only"
        base["editable_operations"].append({
            "operation": op,
            "operation_weight": round(op_weight, 6),
            "class_compatibility": float(class_weight),
            "quality_factor": quality,
            "status": status,
            "route_candidates": op_cfg["routes"],
        })
    base["track_total_weight"] = round(sum(item["operation_weight"] for item in base["editable_operations"]), 6)
    return base


def donor_key(operation: str, rec: Mapping[str, Any]) -> Optional[Tuple[str, ...]]:
    domain = str(rec.get("content_domain", "unknown"))
    if domain == "unknown":
        return None
    if operation == "person_appearance_swap":
        return (operation, domain, "human")
    if operation == "object_swap":
        bucket = str(rec.get("semantic_bucket", "generic_unknown"))
        if bucket in {"generic_unknown", "unknown"}:
            return None
        return (operation, domain, bucket)
    if operation == "surface_content_edit":
        subtype = str(rec.get("surface_subtype") or "unknown")
        if subtype == "unknown":
            return None
        return (operation, domain, subtype)
    return None


def pair_score(target: Mapping[str, Any], donor: Mapping[str, Any]) -> float:
    tq = safe_float(target.get("track_quality_score"), 0.0)
    dq = safe_float(donor.get("track_quality_score"), 0.0)
    return round(math.sqrt(max(0.0, tq) * max(0.0, dq)), 6)


def make_pair_id(operation: str, target: Mapping[str, Any], donor: Mapping[str, Any]) -> str:
    raw = f"{operation}|{target['video_id']}|{target['track_id']}|{donor['video_id']}|{donor['track_id']}"
    return f"pair_{operation}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--domains", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-donors-per-target", type=int, default=12)
    parser.add_argument("--allow-unknown-domain", action="store_true", help="Not recommended. Default behavior refuses donor pairing for unknown domains.")
    args = parser.parse_args()

    tracks_payload = json.loads(args.tracks.read_text(encoding="utf-8"))
    raw_tracks = tracks_payload.get("tracks", tracks_payload if isinstance(tracks_payload, list) else None)
    if not isinstance(raw_tracks, list):
        raise ValueError("--tracks must be a JSON object with a 'tracks' list, or a JSON list")
    domains = load_domains(args.domains)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "generator_registry_v1.json", {"schema_version": "dataA_v1_generator_registry", "generators": GENERATOR_REGISTRY})
    write_json(args.out_dir / "operation_registry_v1.json", {"schema_version": "dataA_v1_operation_registry", "operations": OPERATION_REGISTRY})
    write_json(args.out_dir / "candidate_operation_rules_v1.json", {
        "schema_version": "dataA_v1_candidate_operation_rules",
        "rules": CANDIDATE_OPERATION_RULES,
        "object_bucket_rules": [{"bucket": b, "keywords": list(w)} for b, w in OBJECT_BUCKET_RULES],
        "surface_subtype_rules": SURFACE_SUBTYPE_RULES,
    })

    catalog_tracks: List[Dict[str, Any]] = []
    missing_domain: Counter[str] = Counter()
    for raw in raw_tracks:
        if not isinstance(raw, dict) or not raw.get("video_id") or not raw.get("track_id"):
            continue
        vid = str(raw["video_id"])
        domain_meta = domains.get(vid)
        if domain_meta is None:
            missing_domain[vid] += 1
            domain_meta = {"content_domain": "unknown", "style_domain": "unknown", "domain_confidence": 0.0}
        catalog_tracks.append(build_track_record(raw, domain_meta))

    catalog_payload = {
        "schema_version": "dataA_v1_track_editability_catalog",
        "source_track_file": str(args.tracks),
        "source_track_count": len(raw_tracks),
        "catalog_track_count": len(catalog_tracks),
        "domain_index_file": str(args.domains),
        "tracks": catalog_tracks,
    }
    write_json(args.out_dir / "track_editability_catalog_v1.json", catalog_payload)

    donor_index: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    target_ops: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for rec in catalog_tracks:
        if not rec.get("eligible"):
            continue
        domain = rec.get("content_domain", "unknown")
        if domain == "unknown" and not args.allow_unknown_domain:
            continue
        for op in rec.get("editable_operations", []):
            key = donor_key(op["operation"], rec)
            if key is None:
                continue
            donor_index[key].append(rec)
            target_ops.append((rec, op))

    pairs: List[Dict[str, Any]] = []
    per_operation_pairs: Counter[str] = Counter()
    for target, op in target_ops:
        key = donor_key(op["operation"], target)
        if key is None:
            continue
        candidates = [
            donor for donor in donor_index[key]
            if donor["video_id"] != target["video_id"] and donor["track_id"] != target["track_id"]
        ]
        candidates.sort(key=lambda d: (-pair_score(target, d), stable_int(f"{target['track_id']}|{d['track_id']}")))
        for donor in candidates[: args.max_donors_per_target]:
            operation = op["operation"]
            pairs.append({
                "pair_id": make_pair_id(operation, target, donor),
                "operation": operation,
                "target": {
                    "video_id": target["video_id"],
                    "video_path": target.get("video_path"),
                    "track_id": target["track_id"],
                    "candidate_class": target["candidate_class"],
                    "canonical_concept": target["canonical_concept"],
                    "mask_tube_path": target.get("mask_tube_path"),
                    "bbox_tube_xywh": target.get("bbox_tube_xywh"),
                },
                "donor": {
                    "video_id": donor["video_id"],
                    "video_path": donor.get("video_path"),
                    "track_id": donor["track_id"],
                    "candidate_class": donor["candidate_class"],
                    "canonical_concept": donor["canonical_concept"],
                    "mask_tube_path": donor.get("mask_tube_path"),
                    "bbox_tube_xywh": donor.get("bbox_tube_xywh"),
                    "reference_frame_strategy": "deferred_pick_largest_visible_mask_frame",
                },
                "compatibility": {
                    "content_domain": target["content_domain"],
                    "style_domain_target": target["style_domain"],
                    "style_domain_donor": donor["style_domain"],
                    "semantic_bucket": target["semantic_bucket"],
                    "surface_subtype": target.get("surface_subtype"),
                    "same_source_video": False,
                    "pair_score": pair_score(target, donor),
                },
                "route_candidates": [route for route in op["route_candidates"] if "reference" in route["route_id"]],
                "status": "candidate_pair",
            })
            per_operation_pairs[operation] += 1

    pair_payload = {
        "schema_version": "dataA_v1_donor_pair_pool",
        "source_catalog": str(args.out_dir / "track_editability_catalog_v1.json"),
        "pairing_policy": {
            "directed_pair": "target source video A receives donor reference from source video B",
            "same_video_donor_forbidden": True,
            "unknown_domain_donor_pairing": bool(args.allow_unknown_domain),
            "max_donors_per_target": args.max_donors_per_target,
            "reference_crop_generation": "deferred; choose donor frame with largest visible mask after plan selection",
        },
        "pair_count": len(pairs),
        "pairs": pairs,
    }
    write_json(args.out_dir / "donor_pair_pool_v1.json", pair_payload)

    stats = {
        "raw_track_count": len(raw_tracks),
        "catalog_track_count": len(catalog_tracks),
        "eligible_track_count": sum(bool(r.get("eligible")) for r in catalog_tracks),
        "domain_counts": dict(Counter(r.get("content_domain", "unknown") for r in catalog_tracks)),
        "candidate_class_counts": dict(Counter(r.get("candidate_class", "unknown") for r in catalog_tracks)),
        "semantic_bucket_counts": dict(Counter(r.get("semantic_bucket", "unknown") for r in catalog_tracks)),
        "operation_attachment_counts": dict(Counter(op["operation"] for r in catalog_tracks for op in r.get("editable_operations", []))),
        "pair_counts_by_operation": dict(per_operation_pairs),
        "videos_missing_domain_labels": len(missing_domain),
        "missing_domain_track_count": sum(missing_domain.values()),
    }
    write_json(args.out_dir / "pairing_stats_v1.json", stats)

    print("Wrote:")
    for name in (
        "generator_registry_v1.json", "operation_registry_v1.json", "candidate_operation_rules_v1.json",
        "track_editability_catalog_v1.json", "donor_pair_pool_v1.json", "pairing_stats_v1.json",
    ):
        print(" -", args.out_dir / name)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
