"""Case manifest construction and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from .common import DataAError, write_json


REQUIRED_MANIFEST_KEYS = {
    "case_id",
    "stage_status",
    "operation",
    "generator_route",
    "target",
    "donor",
    "source_clip",
    "canonical_vace_profile",
    "mask_layers",
    "mask_processing_parameters",
    "prompt",
    "vace_command",
    "preflight",
}


def build_case_manifest(**kwargs: Any) -> Dict[str, Any]:
    manifest = dict(kwargs)
    missing = REQUIRED_MANIFEST_KEYS - set(manifest)
    if missing:
        raise DataAError(f"case manifest missing required keys: {sorted(missing)}")
    return manifest


def validate_manifest_payload(manifest: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    missing = REQUIRED_MANIFEST_KEYS - set(manifest)
    if missing:
        errors.append(f"missing_keys:{','.join(sorted(missing))}")
    prompt = manifest.get("prompt") or {}
    if not prompt.get("model_prompt") or not prompt.get("control_prompt"):
        errors.append("missing_prompt")
    masks = manifest.get("mask_layers") or {}
    for key in ("M_raw", "M_edit", "M_gen", "M_alpha"):
        if key not in masks:
            errors.append(f"missing_mask_layer:{key}")
    if manifest.get("stage_status") not in {
        "planned",
        "blocked_missing_mask",
        "blocked_volatile_mask",
        "blocked_low_visibility",
        "blocked_clip_selection_failure",
        "blocked_mask_video_mismatch",
        "blocked_donor_reference_failure",
        "blocked_vace_generation_failure",
        "blocked_packaging_failure",
        "blocked_schema_error",
        "blocked_invalid_mask_npz",
        "blocked_mapped_but_unverified",
        "packed",
    }:
        errors.append(f"invalid_stage_status:{manifest.get('stage_status')}")
    return errors


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    errors = validate_manifest_payload(manifest)
    if errors:
        raise DataAError(f"invalid case manifest: {errors}")
    write_json(path, manifest)
