#!/usr/bin/env python3
"""Audit frozen VACE Stage-1 cases before any video generation.

P1 verifies plan/track references and mask-tube accessibility. It never calls
VACE, changes the frozen plan, creates media, or re-samples cases. A readable
local mask under /tmp is an allowed runtime input; its volatile storage state is
recorded in the report but does not itself block generation.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.common import DataAError, read_json, utc_now_iso, write_json
from scripts.dataa_v1.mask_io import inspect_mask_npz
from scripts.dataa_v1.path_resolver import PathResolver, ResolvedPath
from scripts.dataa_v1.schema import (
    CanonicalCaseSpec,
    build_track_index,
    normalize_cases,
    schema_errors,
    serialize_case,
    serialize_track,
)


SCHEMA_VERSION = "dataA_v1_vace_stage1_preflight_v1"


def load_cases_for_audit(plan_path: Path, track_bank_path: Path, mapping_path: Optional[Path]) -> List[CanonicalCaseSpec]:
    plan = read_json(plan_path)
    track_bank = read_json(track_bank_path)
    mapping = read_json(mapping_path) if mapping_path else {}
    track_index = build_track_index(track_bank)
    return normalize_cases(plan, track_index, PathResolver(mapping), str(plan_path))


def _path_blockers(role: str, path: ResolvedPath, mask: Mapping[str, Any]) -> List[str]:
    if path.state == "missing":
        return [f"{role}_missing_mask"]
    if path.state == "mapped_but_unverified":
        return [f"{role}_mapped_but_unverified"]
    if path.state in {"readable_persistent", "readable_volatile"} and not bool(mask.get("valid")):
        return [f"{role}_invalid_npz"]
    # /tmp is the intended high-throughput runtime cache for CameraBench video
    # and SAM3 mask tubes. Its state remains visible in the serialized report,
    # but readable volatile input is not a generation blocker.
    return []


def audit_case(case: CanonicalCaseSpec) -> Dict[str, Any]:
    blockers = schema_errors(case)
    target_path = case.target.path or ResolvedPath(None, None, "missing")
    case.target.mask_npz = inspect_mask_npz(target_path)
    blockers.extend(_path_blockers("target", target_path, case.target.mask_npz))

    if case.donor:
        donor_path = case.donor.path or ResolvedPath(None, None, "missing")
        case.donor.mask_npz = inspect_mask_npz(donor_path)
        blockers.extend(_path_blockers("donor", donor_path, case.donor.mask_npz))

    if any(item.endswith("missing_mask") or item.endswith("missing_mask_path") for item in blockers):
        status = "blocked_missing_mask"
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
        "target": serialize_track(case.target),
        "donor": serialize_track(case.donor),
        "canonical_case_spec": serialize_case(case),
    }


def build_report(plan_path: Path, track_bank_path: Path, mapping_path: Optional[Path]) -> Dict[str, Any]:
    track_bank = read_json(track_bank_path)
    track_index = build_track_index(track_bank)
    cases = load_cases_for_audit(plan_path, track_bank_path, mapping_path)
    audited = [audit_case(case) for case in cases]

    status_counts: Dict[str, int] = {}
    operation_counts: Dict[str, int] = {}
    for result in audited:
        status_counts[result["stage_status"]] = status_counts.get(result["stage_status"], 0) + 1
        operation = result["operation"] or "<missing>"
        operation_counts[operation] = operation_counts.get(operation, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now_iso(),
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
        persistent = root / "persistent"
        mask_path = persistent / "masks" / "track_a.npz"
        mask_path.parent.mkdir(parents=True)
        np.savez_compressed(
            mask_path,
            frame_indices=np.array([10, 11, 12], dtype=np.int32),
            masks=np.ones((3, 4, 5), dtype=np.uint8),
        )
        invalid_path = persistent / "masks" / "bad_dtype.npz"
        np.savez_compressed(
            invalid_path,
            frame_indices=np.array([10, 11, 12], dtype=np.int64),
            masks=np.ones((3, 4, 5), dtype=np.uint8),
        )
        tracks = {"tracks": [
            {"track_id": "track_a", "candidate_id": "cand_a", "video_id": "video_a", "video_path": "/data/a.mp4", "mask_tube_path": str(mask_path), "candidate_class": "bounded_object", "canonical_concept": "mug"},
            {"track_id": "track_b", "candidate_id": "cand_b", "video_id": "video_b", "video_path": "/data/b.mp4", "mask_tube_path": str(mask_path), "candidate_class": "bounded_object", "canonical_concept": "bottle"},
            {"track_id": "track_bad", "candidate_id": "cand_bad", "video_id": "video_c", "video_path": "/data/c.mp4", "mask_tube_path": str(invalid_path)},
        ]}
        plan = {"cases": [
            {
                "case_id": "smoke_0001",
                "operation": "object_swap",
                "generator_route": "vace14b_masktrack_reference_swap",
                "target_track_id": "track_a",
                "donor_track_id": "track_b",
            },
            {
                "case_id": "smoke_bad_dtype",
                "operation": "object_swap",
                "target_track_id": "track_bad",
                "donor_track_id": "track_b",
            },
        ]}
        plan_path, tracks_path = root / "plan.json", root / "tracks.json"
        write_json(plan_path, plan)
        write_json(tracks_path, tracks)
        report = build_report(plan_path, tracks_path, None)
        assert report["summary"]["case_count"] == 2
        assert report["cases"][0]["stage_status"] == "preflight_passed", report
        assert report["cases"][1]["stage_status"] == "blocked_invalid_mask_npz", report
        assert "int32" in report["cases"][1]["target"]["mask_npz"]["reason"], report

        volatile_resolved = ResolvedPath(
            original_path="/tmp/sam3/track.npz",
            resolved_path="/tmp/sam3/track.npz",
            state="readable_volatile",
            exists=True,
            is_volatile=True,
        )
        valid_mask = {"valid": True}
        invalid_mask = {"valid": False}
        assert _path_blockers("target", volatile_resolved, valid_mask) == []
        assert _path_blockers("target", volatile_resolved, invalid_mask) == ["target_invalid_npz"]
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
    except DataAError as exc:
        print(f"preflight configuration error: {exc}", file=sys.stderr)
        return 2
    write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    failures = report["summary"]["case_count"] - report["summary"]["status_counts"].get("preflight_passed", 0)
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
