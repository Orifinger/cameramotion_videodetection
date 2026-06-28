#!/usr/bin/env python3
"""Audit frozen VACE Stage-1 cases before any video generation.

P1 only verifies plan/track references and mask-tube accessibility. It never
calls VACE, changes the frozen plan, creates media, or re-samples cases.

Inputs:
  * VACE stage-1 quota plan JSON;
  * enriched SAM3 track-bank JSON;
  * optional server-local path-mapping JSON.

Output:
  A JSON report with a canonicalized case view and a persistence state for every
  target/donor mask: readable_persistent, readable_volatile,
  mapped_but_unverified, or missing.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("numpy is required: pip install numpy") from exc


SCHEMA_VERSION = "dataA_v1_vace_stage1_preflight_v1"
VOLATILE_DEFAULT_PREFIXES = ("/tmp", "/var/tmp")
CASE_LIST_KEYS = ("cases", "items", "entries", "samples", "plan")
TRACK_LIST_KEYS = ("tracks", "items", "entries", "records", "data")
TARGET_KEYS = ("target", "target_track", "source", "source_track", "target_ref")
DONOR_KEYS = ("donor", "donor_track", "reference", "reference_track", "donor_ref")
TRACK_ID_KEYS = ("track_id", "target_track_id", "donor_track_id", "id")
VIDEO_ID_KEYS = ("video_id", "source_video_id")
VIDEO_PATH_KEYS = ("video_path", "source_video_path", "path")
MASK_PATH_KEYS = ("mask_tube_path", "mask_path")


class PreflightError(ValueError):
    """Raised for an unambiguous input/configuration failure."""


@dataclass
class ResolvedPath:
    original_path: Optional[str]
    resolved_path: Optional[str]
    state: str
    mapping_rule: Optional[str] = None
    exists: bool = False
    is_volatile: bool = False
    note: Optional[str] = None


@dataclass
class TrackRef:
    role: str
    track_id: Optional[str]
    video_id: Optional[str]
    video_path: Optional[str]
    mask_tube_path: Optional[str]
    candidate_class: Optional[str] = None
    canonical_concept: Optional[str] = None
    display_phrase: Optional[str] = None
    region_family: Optional[str] = None
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
    plan_source: str
    raw_case: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PathRule:
    source_prefix: str
    persistent_prefix: str
    status: str = "planned_or_verified"


def _normalize_prefix(value: str) -> str:
    return str(value).replace("\\", "/").rstrip("/") or "/"


def _path_has_prefix(path: str, prefix: str) -> bool:
    path_n = _normalize_prefix(path)
    prefix_n = _normalize_prefix(prefix)
    return path_n == prefix_n or path_n.startswith(prefix_n + "/")


def _is_local_path(path: str) -> bool:
    return not path.startswith(("oss://", "s3://", "http://", "https://"))


def _optional_str(value: Any) -> Optional[str]:
    return None if value in (None, "") else str(value)


def _first_value(record: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise PreflightError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PreflightError(f"Invalid JSON in {path}: {exc}") from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


def _as_records(payload: Any, keys: Sequence[str], label: str) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        records = next((payload[k] for k in keys if isinstance(payload.get(k), list)), None)
        if records is None:
            values = list(payload.values())
            if values and all(isinstance(v, Mapping) for v in values):
                records = values
            else:
                raise PreflightError(
                    f"Cannot find a list of {label} records; expected one of: {', '.join(keys)}"
                )
    else:
        raise PreflightError(f"{label} JSON must be a list or object")

    output: List[Dict[str, Any]] = []
    for index, item in enumerate(records):
        if not isinstance(item, Mapping):
            raise PreflightError(f"{label} record at index {index} is not an object")
        output.append(dict(item))
    return output


class PathResolver:
    """Resolve mask paths without altering original track-bank records."""

    def __init__(self, mapping: Optional[Mapping[str, Any]] = None) -> None:
        mapping = mapping or {}
        self.volatile_prefixes = tuple(
            _normalize_prefix(p)
            for p in mapping.get("volatile_prefixes", VOLATILE_DEFAULT_PREFIXES)
        )
        self.rules: List[PathRule] = []
        for entry in mapping.get("rules", []):
            if not isinstance(entry, Mapping):
                continue
            source = entry.get("source_prefix")
            destination = entry.get("persistent_prefix")
            if source and destination:
                self.rules.append(
                    PathRule(
                        source_prefix=_normalize_prefix(str(source)),
                        persistent_prefix=str(destination).rstrip("/"),
                        status=str(entry.get("status", "planned_or_verified")),
                    )
                )
        self.rules.sort(key=lambda rule: len(rule.source_prefix), reverse=True)
        self.explicit = {
            str(source): str(destination)
            for source, destination in dict(mapping.get("explicit_overrides", {})).items()
        }

    def resolve(self, raw_path: Optional[str]) -> ResolvedPath:
        if not raw_path:
            return ResolvedPath(None, None, "missing", note="mask_tube_path is absent")

        raw_path = str(raw_path)
        raw_is_local = _is_local_path(raw_path)
        raw_exists = bool(raw_is_local and Path(raw_path).is_file())
        raw_volatile = self._is_volatile(raw_path)

        # A directly readable persistent path is authoritative. A directly
        # readable /tmp path remains volatile unless a separately readable
        # persistent mapping has been supplied.
        if raw_exists and not raw_volatile:
            return ResolvedPath(
                original_path=raw_path,
                resolved_path=raw_path,
                state="readable_persistent",
                exists=True,
                note="original path is directly readable",
            )

        mapped_path, mapping_rule = self._lookup_mapping(raw_path)
        if mapped_path:
            mapped = self._mapped(raw_path, mapped_path, mapping_rule or "mapping")
            if mapped.state == "readable_persistent":
                return mapped
            if raw_exists and raw_volatile:
                return ResolvedPath(
                    original_path=raw_path,
                    resolved_path=raw_path,
                    state="readable_volatile",
                    mapping_rule=mapping_rule,
                    exists=True,
                    is_volatile=True,
                    note=f"volatile source readable; persistent mapping is not verified: {mapped_path}",
                )
            return mapped

        if raw_exists:
            return ResolvedPath(
                original_path=raw_path,
                resolved_path=raw_path,
                state="readable_volatile",
                exists=True,
                is_volatile=True,
                note="original path is directly readable but volatile",
            )

        return ResolvedPath(
            original_path=raw_path,
            resolved_path=None,
            state="missing",
            is_volatile=raw_volatile,
            note="path is not readable and no mapping matched",
        )

    def _lookup_mapping(self, raw_path: str) -> tuple[Optional[str], Optional[str]]:
        if raw_path in self.explicit:
            return self.explicit[raw_path], "explicit_override"
        for rule in self.rules:
            if _path_has_prefix(raw_path, rule.source_prefix):
                suffix = raw_path[len(rule.source_prefix):].lstrip("/\\")
                destination = rule.persistent_prefix.rstrip("/") + "/" + suffix.replace("\\", "/")
                return destination, f"rule:{rule.source_prefix}"
        return None, None

    def _mapped(self, original: str, mapped: str, mapping_rule: str) -> ResolvedPath:
        exists = bool(_is_local_path(mapped) and Path(mapped).is_file())
        if exists:
            return ResolvedPath(
                original_path=original,
                resolved_path=mapped,
                state="readable_persistent",
                mapping_rule=mapping_rule,
                exists=True,
                note="mapped persistent path is readable",
            )
        return ResolvedPath(
            original_path=original,
            resolved_path=mapped,
            state="mapped_but_unverified",
            mapping_rule=mapping_rule,
            exists=False,
            note="mapping exists but this runtime cannot verify its destination",
        )

    def _is_volatile(self, path: str) -> bool:
        normalized = _normalize_prefix(path)
        return any(_path_has_prefix(normalized, prefix) for prefix in self.volatile_prefixes)


def build_track_index(track_bank: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
    by_track: Dict[str, Dict[str, Any]] = {}
    by_candidate: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(_as_records(track_bank, TRACK_LIST_KEYS, "track-bank")):
        track_id = record.get("track_id")
        candidate_id = record.get("candidate_id")
        if not track_id and not candidate_id:
            raise PreflightError(f"track-bank record {index} has neither track_id nor candidate_id")
        if track_id:
            key = str(track_id)
            if key in by_track:
                raise PreflightError(f"duplicate track_id in track bank: {key}")
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
        nested_track_id = _first_value(nested, TRACK_ID_KEYS)
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


def _track_ref(role: str, record: Optional[Mapping[str, Any]], resolver: PathResolver) -> Optional[TrackRef]:
    if record is None:
        return None
    raw = dict(record)
    mask_path = _first_value(raw, MASK_PATH_KEYS)
    return TrackRef(
        role=role,
        track_id=_optional_str(_first_value(raw, TRACK_ID_KEYS)),
        video_id=_optional_str(_first_value(raw, VIDEO_ID_KEYS)),
        video_path=_optional_str(_first_value(raw, VIDEO_PATH_KEYS)),
        mask_tube_path=_optional_str(mask_path),
        candidate_class=_optional_str(raw.get("candidate_class")),
        canonical_concept=_optional_str(raw.get("canonical_concept")),
        display_phrase=_optional_str(raw.get("display_phrase")),
        region_family=_optional_str(raw.get("region_family")),
        raw=raw,
        path=resolver.resolve(_optional_str(mask_path)),
    )


def normalize_cases(
    plan: Any,
    track_index: Mapping[str, Mapping[str, Dict[str, Any]]],
    resolver: PathResolver,
    plan_source: str,
) -> List[CanonicalCaseSpec]:
    normalized: List[CanonicalCaseSpec] = []
    seen_case_ids: set[str] = set()
    for index, case in enumerate(_as_records(plan, CASE_LIST_KEYS, "plan")):
        case_id = _optional_str(_first_value(case, ("case_id", "pair_id", "id")))
        if not case_id:
            raise PreflightError(f"plan case {index} lacks case_id")
        if case_id in seen_case_ids:
            raise PreflightError(f"duplicate case_id in plan: {case_id}")
        seen_case_ids.add(case_id)

        target = _track_ref("target", _resolve_ref_record(case, "target", track_index), resolver)
        donor = _track_ref("donor", _resolve_ref_record(case, "donor", track_index), resolver)
        if target is None:
            target = TrackRef("target", None, None, None, None)
        normalized.append(
            CanonicalCaseSpec(
                case_id=case_id,
                operation=_optional_str(_first_value(case, ("operation", "edit_type", "operation_type"))),
                generator_route=_optional_str(_first_value(case, ("generator_route", "route_id", "route"))),
                target=target,
                donor=donor,
                plan_source=plan_source,
                raw_case=dict(case),
            )
        )
    return normalized


def inspect_mask_npz(resolved: ResolvedPath) -> Dict[str, Any]:
    """Check the SAM3 [N_visible], [N_visible,H,W] tube contract read-only."""
    result: Dict[str, Any] = {
        "checked": False, "valid": False, "reason": None,
        "frame_indices_count": None, "mask_count": None,
        "height": None, "width": None, "dtype": None,
        "frame_indices_strictly_increasing": None, "mask_nonempty_ratio": None,
    }
    if resolved.state not in {"readable_persistent", "readable_volatile"} or not resolved.resolved_path:
        result["reason"] = f"not locally readable ({resolved.state})"
        return result
    try:
        with np.load(Path(resolved.resolved_path), allow_pickle=False) as archive:
            if "frame_indices" not in archive or "masks" not in archive:
                result["reason"] = "npz must contain frame_indices and masks"
                return result
            frame_indices = archive["frame_indices"]
            masks = archive["masks"]
    except Exception as exc:
        result["reason"] = f"cannot read npz: {type(exc).__name__}: {exc}"
        return result

    result.update({
        "checked": True,
        "frame_indices_count": int(frame_indices.shape[0]) if frame_indices.ndim >= 1 else 0,
        "mask_count": int(masks.shape[0]) if masks.ndim >= 1 else 0,
        "dtype": str(masks.dtype),
    })
    if frame_indices.ndim != 1:
        result["reason"] = f"frame_indices must be 1D, got {frame_indices.shape}"
        return result
    if masks.ndim != 3:
        result["reason"] = f"masks must have [N,H,W], got {masks.shape}"
        return result
    if frame_indices.shape[0] != masks.shape[0]:
        result["reason"] = "frame_indices and masks have different N"
        return result
    if masks.shape[0] == 0:
        result["reason"] = "empty mask tube"
        return result

    result["height"] = int(masks.shape[1])
    result["width"] = int(masks.shape[2])
    increasing = bool(np.all(np.diff(frame_indices.astype(np.int64)) > 0)) if len(frame_indices) > 1 else True
    nonempty = np.any(masks > 0, axis=(1, 2))
    result["frame_indices_strictly_increasing"] = increasing
    result["mask_nonempty_ratio"] = float(nonempty.mean())
    if not increasing:
        result["reason"] = "frame_indices are not strictly increasing"
        return result
    if not bool(nonempty.all()):
        result["reason"] = "at least one visible-frame mask is empty"
        return result
    result.update(valid=True, reason="ok")
    return result


def _case_schema_errors(case: CanonicalCaseSpec) -> List[str]:
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


def _path_blockers(role: str, path: ResolvedPath, mask: Mapping[str, Any]) -> List[str]:
    if path.state == "missing":
        return [f"{role}_missing_mask"]
    if path.state == "readable_volatile":
        return [f"{role}_volatile_mask"]
    if path.state == "mapped_but_unverified":
        return [f"{role}_mapped_but_unverified"]
    if path.state == "readable_persistent" and not bool(mask.get("valid")):
        return [f"{role}_invalid_npz"]
    return []


def _serialize_track(track: Optional[TrackRef]) -> Optional[Dict[str, Any]]:
    if track is None:
        return None
    result = asdict(track)
    result.pop("raw", None)  # avoid copying potentially large bbox tubes into report
    return result


def _serialize_case(case: CanonicalCaseSpec) -> Dict[str, Any]:
    return {
        "case_id": case.case_id,
        "operation": case.operation,
        "generator_route": case.generator_route,
        "target": _serialize_track(case.target),
        "donor": _serialize_track(case.donor),
        "plan_source": case.plan_source,
    }


def audit_case(case: CanonicalCaseSpec) -> Dict[str, Any]:
    blockers = _case_schema_errors(case)
    target_path = case.target.path or ResolvedPath(None, None, "missing")
    case.target.mask_npz = inspect_mask_npz(target_path)
    blockers.extend(_path_blockers("target", target_path, case.target.mask_npz))

    if case.donor:
        donor_path = case.donor.path or ResolvedPath(None, None, "missing")
        case.donor.mask_npz = inspect_mask_npz(donor_path)
        blockers.extend(_path_blockers("donor", donor_path, case.donor.mask_npz))

    if any(item.endswith("missing_mask") or item.endswith("missing_mask_path") for item in blockers):
        status = "blocked_missing_mask"
    elif any("volatile" in item for item in blockers):
        status = "blocked_volatile_mask"
    elif any("mapped_but_unverified" in item for item in blockers):
        status = "blocked_mapped_but_unverified"
    elif any("invalid_npz" in item for item in blockers):
        status = "blocked_invalid_mask_npz"
    elif blockers:
        status = "blocked_schema_error"
    else:
        status = "preflight_passed"

    return {
        "case_id": case.case_id,
        "operation": case.operation,
        "generator_route": case.generator_route,
        "stage_status": status,
        "blockers": blockers,
        "target": _serialize_track(case.target),
        "donor": _serialize_track(case.donor),
        "canonical_case_spec": _serialize_case(case),
    }


def build_report(plan_path: Path, track_bank_path: Path, mapping_path: Optional[Path]) -> Dict[str, Any]:
    plan = _read_json(plan_path)
    track_bank = _read_json(track_bank_path)
    mapping = _read_json(mapping_path) if mapping_path else {}
    track_index = build_track_index(track_bank)
    cases = normalize_cases(plan, track_index, PathResolver(mapping), str(plan_path))
    audited = [audit_case(case) for case in cases]

    status_counts: Dict[str, int] = {}
    operation_counts: Dict[str, int] = {}
    for result in audited:
        status_counts[result["stage_status"]] = status_counts.get(result["stage_status"], 0) + 1
        operation = result["operation"] or "<missing>"
        operation_counts[operation] = operation_counts.get(operation, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "plan": str(plan_path),
            "track_bank": str(track_bank_path),
            "path_mapping": str(mapping_path) if mapping_path else None,
        },
        "summary": {
            "case_count": len(audited),
            "status_counts": status_counts,
            "operation_counts": operation_counts,
            "track_bank_index": {
                "track_id_count": len(track_index["track_id"]),
                "candidate_id_count": len(track_index["candidate_id"]),
            },
        },
        "cases": audited,
    }


def _self_test() -> int:
    """Synthetic, no-project-data smoke test for parser and NPZ validation."""
    with tempfile.TemporaryDirectory(prefix=".dataa_v1_preflight_", dir=str(Path.cwd())) as temp_dir:
        root = Path(temp_dir)
        mask_path = root / "masks" / "track_a.npz"
        mask_path.parent.mkdir(parents=True)
        np.savez_compressed(
            mask_path,
            frame_indices=np.array([10, 11, 12], dtype=np.int32),
            masks=np.ones((3, 4, 5), dtype=np.uint8),
        )
        tracks = {"tracks": [
            {"track_id": "track_a", "candidate_id": "cand_a", "video_id": "video_a", "video_path": "/data/a.mp4", "mask_tube_path": str(mask_path), "candidate_class": "bounded_object", "canonical_concept": "mug"},
            {"track_id": "track_b", "candidate_id": "cand_b", "video_id": "video_b", "video_path": "/data/b.mp4", "mask_tube_path": str(mask_path), "candidate_class": "bounded_object", "canonical_concept": "bottle"},
        ]}
        plan = {"cases": [{
            "case_id": "smoke_0001", "operation": "object_swap",
            "generator_route": "vace14b_masktrack_reference_swap",
            "target_track_id": "track_a", "donor_track_id": "track_b",
        }]}
        plan_path, tracks_path = root / "plan.json", root / "tracks.json"
        _write_json(plan_path, plan)
        _write_json(tracks_path, tracks)
        report = build_report(plan_path, tracks_path, None)
        assert report["summary"]["case_count"] == 1
        assert report["cases"][0]["stage_status"] == "preflight_passed", report
    print("self-test passed")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, help="Frozen VACE stage-1 quota plan JSON")
    parser.add_argument("--track-bank", type=Path, help="sam3_quality_tracks_enriched.json")
    parser.add_argument("--path-mapping", type=Path, default=None, help="Optional server-local path mapping JSON")
    parser.add_argument("--output", type=Path, default=None, help="Output preflight report JSON")
    parser.add_argument("--strict", action="store_true", help="Return non-zero unless every case preflight passes")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic self-test; no project data needed")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return _self_test()
    if not args.plan or not args.track_bank or not args.output:
        raise SystemExit("--plan, --track-bank and --output are required unless --self-test is used")
    try:
        report = build_report(args.plan, args.track_bank, args.path_mapping)
    except PreflightError as exc:
        print(f"preflight configuration error: {exc}", file=sys.stderr)
        return 2
    _write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    failures = report["summary"]["case_count"] - report["summary"]["status_counts"].get("preflight_passed", 0)
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
