"""Plan and track-bank normalization for Data A v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .common import DataAError
from .path_resolver import PathResolver, ResolvedPath


CASE_LIST_KEYS = ("cases", "items", "entries", "samples", "plan")
TRACK_LIST_KEYS = ("tracks", "items", "entries", "records", "data")
TARGET_KEYS = ("target", "target_track", "source", "source_track", "target_ref")
DONOR_KEYS = ("donor", "donor_track", "reference", "reference_track", "donor_ref")
TRACK_ID_KEYS = ("track_id", "target_track_id", "donor_track_id", "id")
VIDEO_ID_KEYS = ("video_id", "source_video_id")
VIDEO_PATH_KEYS = ("video_path", "source_video_path", "path")
MASK_PATH_KEYS = ("mask_tube_path", "mask_path")
OPERATION_KEYS = ("operation", "edit_type", "operation_type")
ROUTE_KEYS = ("generator_route", "route_id", "route")


@dataclass
class TrackRef:
    role: str
    track_id: Optional[str]
    video_id: Optional[str]
    video_path: Optional[str]
    mask_tube_path: Optional[str]
    candidate_id: Optional[str] = None
    candidate_class: Optional[str] = None
    canonical_concept: Optional[str] = None
    display_phrase: Optional[str] = None
    region_family: Optional[str] = None
    content_domain: Optional[str] = None
    style_domain: Optional[str] = None
    bbox_tube_xywh: Any = None
    raw: Dict[str, Any] = field(default_factory=dict)
    path: Optional[ResolvedPath] = None
    mask_npz: Optional[Dict[str, Any]] = None


@dataclass
class CanonicalCaseSpec:
    case_id: str
    operation: Optional[str]
    generator_route: Optional[str]
    target: TrackRef
    donor: Optional[TrackRef]
    sampling_meta: Dict[str, Any] = field(default_factory=dict)
    plan_source: str = ""
    raw_case: Dict[str, Any] = field(default_factory=dict)


def optional_str(value: Any) -> Optional[str]:
    return None if value in (None, "") else str(value)


def first_value(record: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def as_records(payload: Any, keys: Sequence[str], label: str) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        records = next((payload[k] for k in keys if isinstance(payload.get(k), list)), None)
        if records is None:
            values = list(payload.values())
            if values and all(isinstance(v, Mapping) for v in values):
                records = values
            else:
                raise DataAError(f"Cannot find a list of {label} records; expected one of: {', '.join(keys)}")
    else:
        raise DataAError(f"{label} JSON must be a list or object")

    output: List[Dict[str, Any]] = []
    for index, item in enumerate(records):
        if not isinstance(item, Mapping):
            raise DataAError(f"{label} record at index {index} is not an object")
        output.append(dict(item))
    return output


def build_track_index(track_bank: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
    by_track: Dict[str, Dict[str, Any]] = {}
    by_candidate: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(as_records(track_bank, TRACK_LIST_KEYS, "track-bank")):
        track_id = record.get("track_id")
        candidate_id = record.get("candidate_id")
        if not track_id and not candidate_id:
            raise DataAError(f"track-bank record {index} has neither track_id nor candidate_id")
        if track_id:
            key = str(track_id)
            if key in by_track:
                raise DataAError(f"duplicate track_id in track bank: {key}")
            by_track[key] = record
        if candidate_id:
            by_candidate.setdefault(str(candidate_id), record)
    return {"track_id": by_track, "candidate_id": by_candidate}


def _resolve_ref_record(
    case: Mapping[str, Any], role: str, track_index: Mapping[str, Mapping[str, Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    nested_keys = TARGET_KEYS if role == "target" else DONOR_KEYS
    prefix = "target" if role == "target" else "donor"
    nested = next((dict(case[k]) for k in nested_keys if isinstance(case.get(k), Mapping)), None)

    track_ids: List[str] = []
    candidate_ids: List[str] = []
    if nested:
        nested_track_id = first_value(nested, TRACK_ID_KEYS)
        if nested_track_id:
            track_ids.append(str(nested_track_id))
        if nested.get("candidate_id"):
            candidate_ids.append(str(nested["candidate_id"]))
    for key in (f"{prefix}_track_id", f"{prefix}_id"):
        if case.get(key):
            track_ids.append(str(case[key]))
    if case.get(f"{prefix}_candidate_id"):
        candidate_ids.append(str(case[f"{prefix}_candidate_id"]))

    base = next((track_index["track_id"][key] for key in track_ids if key in track_index["track_id"]), None)
    if base is None:
        base = next((track_index["candidate_id"][key] for key in candidate_ids if key in track_index["candidate_id"]), None)

    merged = dict(base or {})
    if nested:
        merged.update(nested)
    for key, value in case.items():
        if key.startswith(prefix + "_"):
            merged.setdefault(key[len(prefix) + 1 :], value)
    return merged or None


def track_ref(role: str, record: Optional[Mapping[str, Any]], resolver: PathResolver) -> Optional[TrackRef]:
    if record is None:
        return None
    raw = dict(record)
    mask_path = first_value(raw, MASK_PATH_KEYS)
    return TrackRef(
        role=role,
        track_id=optional_str(first_value(raw, TRACK_ID_KEYS)),
        video_id=optional_str(first_value(raw, VIDEO_ID_KEYS)),
        video_path=optional_str(first_value(raw, VIDEO_PATH_KEYS)),
        mask_tube_path=optional_str(mask_path),
        candidate_id=optional_str(raw.get("candidate_id")),
        candidate_class=optional_str(raw.get("candidate_class")),
        canonical_concept=optional_str(raw.get("canonical_concept")),
        display_phrase=optional_str(raw.get("display_phrase")),
        region_family=optional_str(raw.get("region_family")),
        content_domain=optional_str(raw.get("content_domain")),
        style_domain=optional_str(raw.get("style_domain")),
        bbox_tube_xywh=raw.get("bbox_tube_xywh") or raw.get("bbox_tube") or raw.get("bboxes"),
        raw=raw,
        path=resolver.resolve(optional_str(mask_path)),
    )


def normalize_cases(
    plan: Any,
    track_index: Mapping[str, Mapping[str, Dict[str, Any]]],
    resolver: PathResolver,
    plan_source: str,
) -> List[CanonicalCaseSpec]:
    normalized: List[CanonicalCaseSpec] = []
    seen_case_ids: set[str] = set()
    for index, case in enumerate(as_records(plan, CASE_LIST_KEYS, "plan")):
        case_id = optional_str(first_value(case, ("case_id", "pair_id", "id")))
        if not case_id:
            raise DataAError(f"plan case {index} lacks case_id")
        if case_id in seen_case_ids:
            raise DataAError(f"duplicate case_id in plan: {case_id}")
        seen_case_ids.add(case_id)

        target = track_ref("target", _resolve_ref_record(case, "target", track_index), resolver)
        donor = track_ref("donor", _resolve_ref_record(case, "donor", track_index), resolver)
        if target is None:
            target = TrackRef("target", None, None, None, None)
        normalized.append(
            CanonicalCaseSpec(
                case_id=case_id,
                operation=optional_str(first_value(case, OPERATION_KEYS)),
                generator_route=optional_str(first_value(case, ROUTE_KEYS)),
                target=target,
                donor=donor,
                sampling_meta=dict(case.get("sampling_meta") or {}),
                plan_source=plan_source,
                raw_case=dict(case),
            )
        )
    return normalized


def schema_errors(case: CanonicalCaseSpec) -> List[str]:
    errors: List[str] = []
    if not case.operation:
        errors.append("missing_operation")
    if not case.target.track_id:
        errors.append("unresolved_target_track")
    if not case.target.video_id:
        errors.append("missing_target_video_id")
    if not case.target.mask_tube_path:
        errors.append("missing_target_mask_path")
    if case.donor:
        if not case.donor.track_id:
            errors.append("unresolved_donor_track")
        if not case.donor.video_id:
            errors.append("missing_donor_video_id")
        if not case.donor.mask_tube_path:
            errors.append("missing_donor_mask_path")
        if case.target.video_id and case.donor.video_id and case.target.video_id == case.donor.video_id:
            errors.append("target_and_donor_same_video")
    return errors


def serialize_track(track: Optional[TrackRef], *, include_raw: bool = False) -> Optional[Dict[str, Any]]:
    if track is None:
        return None
    result = {
        "role": track.role,
        "track_id": track.track_id,
        "video_id": track.video_id,
        "video_path": track.video_path,
        "mask_tube_path": track.mask_tube_path,
        "candidate_id": track.candidate_id,
        "candidate_class": track.candidate_class,
        "canonical_concept": track.canonical_concept,
        "display_phrase": track.display_phrase,
        "region_family": track.region_family,
        "content_domain": track.content_domain,
        "style_domain": track.style_domain,
        "bbox_tube_xywh": track.bbox_tube_xywh,
        "path": track.path,
        "mask_npz": track.mask_npz,
    }
    if include_raw:
        result["raw"] = track.raw
    return result


def serialize_case(case: CanonicalCaseSpec, *, include_raw: bool = False) -> Dict[str, Any]:
    result = {
        "case_id": case.case_id,
        "operation": case.operation,
        "generator_route": case.generator_route,
        "target": serialize_track(case.target, include_raw=include_raw),
        "donor": serialize_track(case.donor, include_raw=include_raw),
        "sampling_meta": case.sampling_meta,
        "plan_source": case.plan_source,
    }
    if include_raw:
        result["raw_case"] = case.raw_case
    return result

