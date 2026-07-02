#!/usr/bin/env python3
"""Freeze a VACE execution plan from a materialized pairing dataset index."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.common import DataAError, read_json, utc_now_iso, write_json
from scripts.dataa_v1.execution_plan import validate_execution_cases
from scripts.dataa_v1.path_resolver import PathResolver
from scripts.dataa_v1.schema import normalize_cases


DEFAULT_PAIRING_INDEX = Path("res/dataA_v1/dataset_v2/pairing_dataset_index.json")
DEFAULT_OUT_PLAN = Path("res/dataA_v1/plans/frozen_dataset_v2_vace13b_plan.json")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_project_path(path: Path | str | None) -> Path | None:
    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else _repo_root() / value


def _clean(value: Any, *, max_len: int = 500) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())[:max_len]


def _existing_or_fallback(primary: str | None, fallback: str | None) -> str | None:
    if primary and Path(primary).is_file():
        return primary
    return fallback or primary


def _target_record(pair: Mapping[str, Any]) -> dict[str, Any]:
    target = pair.get("target")
    if not isinstance(target, Mapping):
        raise DataAError(f"pair missing target object: {pair.get('case_id')}")
    pair_dir = Path(str(pair.get("pair_dir") or ""))
    materialized_video = str(pair_dir / "source_video.mp4") if pair_dir else ""
    materialized_mask = str(pair_dir / "target_mask_raw.npz") if pair_dir else ""
    return {
        "track_id": target.get("track_id"),
        "candidate_id": target.get("candidate_id"),
        "video_id": target.get("video_id"),
        "video_path": _existing_or_fallback(materialized_video, target.get("video_path") or target.get("video_path_original")),
        "mask_tube_path": _existing_or_fallback(materialized_mask, target.get("mask_tube_path") or target.get("mask_tube_path_original")),
        "candidate_class": target.get("candidate_class"),
        "canonical_concept": target.get("canonical_concept"),
        "display_phrase": target.get("display_phrase"),
        "region_family": target.get("region_family"),
        "taxonomy_label": target.get("taxonomy_label"),
        "compatibility_group": target.get("compatibility_group"),
        "inventory_entity": target.get("inventory_entity") or {},
    }


def _donor_record(pair: Mapping[str, Any]) -> dict[str, Any] | None:
    donor = pair.get("donor")
    if not isinstance(donor, Mapping):
        return None
    pair_dir = Path(str(pair.get("pair_dir") or ""))
    materialized_mask = str(pair_dir / "donor_mask_raw.npz") if pair_dir else ""
    return {
        "track_id": donor.get("track_id"),
        "candidate_id": donor.get("candidate_id"),
        "video_id": donor.get("video_id"),
        "video_path": donor.get("video_path"),
        "mask_tube_path": _existing_or_fallback(materialized_mask, donor.get("mask_tube_path")),
        "candidate_class": donor.get("candidate_class"),
        "canonical_concept": donor.get("canonical_concept"),
        "display_phrase": donor.get("display_phrase"),
        "region_family": donor.get("region_family"),
        "taxonomy_label": donor.get("taxonomy_label"),
        "compatibility_group": donor.get("compatibility_group"),
        "inventory_entity": donor.get("inventory_entity") or {},
    }


def _model_plan(model_name: str, profile: str, size: str) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "profile": profile,
        "size": size,
        "route": "dataset_v2_pairing",
    }


def _case_from_pair(pair: Mapping[str, Any], *, model_name: str, profile: str, size: str) -> dict[str, Any]:
    case_id = _clean(pair.get("case_id"), max_len=120)
    if not case_id:
        raise DataAError("pair missing case_id")
    operation = _clean(pair.get("operation"), max_len=80)
    route = _clean(pair.get("generator_route"), max_len=120)
    target = _target_record(pair)
    donor = _donor_record(pair)
    sampling_meta = {
        "schema_version": "dataA_v1_dataset_v2_sampling_meta",
        "subject_first_source": "pairing_dataset_v2",
        "dataset_v2": {
            "pair_dir": pair.get("pair_dir"),
            "reference_image_path": pair.get("reference_image_path"),
            "reference_alpha_path": pair.get("reference_alpha_path"),
            "donor_match": pair.get("donor_match") or {},
        },
        "taxonomy": {
            "target_label": target.get("taxonomy_label"),
            "target_group": target.get("compatibility_group"),
            "donor_label": None if donor is None else donor.get("taxonomy_label"),
            "donor_group": None if donor is None else donor.get("compatibility_group"),
        },
        "mask_policy": pair.get("mask_policy") or {},
        "vace_model_plan": _model_plan(model_name, profile, size),
    }
    if operation == "surface_attribute_edit":
        sampling_meta["artifact_policy"] = {
            "artifact_type": "surface_text_degradation",
            "policy_source": "dataset_v2_surface_route",
            "description": "degrade fine text or markings on the masked carrier surface",
        }
    case = {
        "case_id": case_id,
        "operation": operation,
        "generator_route": route,
        "target": target,
        "sampling_meta": sampling_meta,
    }
    if donor is not None:
        case["donor"] = donor
    return case


def build_plan(
    *,
    pairing_index: Path,
    out_plan: Path,
    model_name: str,
    profile: str,
    size: str,
    max_cases: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    index = read_json(pairing_index)
    pairs = index.get("pairs") if isinstance(index, Mapping) else None
    if not isinstance(pairs, list):
        raise DataAError("pairing dataset index must contain pairs[]")
    cases = [_case_from_pair(pair, model_name=model_name, profile=profile, size=size) for pair in pairs if isinstance(pair, Mapping)]
    if max_cases is not None:
        cases = cases[:max_cases]
    normalized = normalize_cases({"cases": cases}, {"track_id": {}, "candidate_id": {}}, PathResolver({}), str(out_plan))
    validation = validate_execution_cases(normalized, donor_reuse_limit=1)
    payload = {
        "schema_version": "dataA_v1_frozen_dataset_v2_vace_execution_plan",
        "generated_at_utc": utc_now_iso(),
        "pairing_dataset_index": str(pairing_index),
        "dataset_root": index.get("dataset_root"),
        "model_name": model_name,
        "profile": profile,
        "size": size,
        "selection_summary": {
            "case_count": len(cases),
            "operation_counts": dict(Counter(str(case.get("operation")) for case in cases)),
            "target_taxonomy_counts": dict(Counter(str((case.get("target") or {}).get("taxonomy_label")) for case in cases).most_common()),
            "donor_taxonomy_counts": dict(Counter(str(((case.get("donor") or {}) or {}).get("taxonomy_label") or "<none>") for case in cases).most_common()),
        },
        "validation": validation,
        "cases": cases,
    }
    if not dry_run:
        write_json(out_plan, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing-index", type=Path, default=DEFAULT_PAIRING_INDEX)
    parser.add_argument("--out-plan", type=Path, default=DEFAULT_OUT_PLAN)
    parser.add_argument("--model-name", default="vace-1.3B")
    parser.add_argument("--profile", default="production_480")
    parser.add_argument("--size", default="480p")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_plan(
            pairing_index=_resolve_project_path(args.pairing_index) or args.pairing_index,
            out_plan=_resolve_project_path(args.out_plan) or args.out_plan,
            model_name=str(args.model_name),
            profile=str(args.profile),
            size=str(args.size),
            max_cases=args.max_cases,
            dry_run=bool(args.dry_run),
        )
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    validation = payload["validation"]
    print(
        "dataset_v2_vace_plan "
        f"dry_run={args.dry_run} cases={validation['case_count']} "
        f"valid={validation['valid']} out={args.out_plan}"
    )
    if not validation["valid"]:
        print(f"validation_errors={validation['errors'][:20]}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
