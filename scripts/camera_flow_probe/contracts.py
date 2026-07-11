"""Strict data contracts for the 40step_v3 camera-flow probe."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.dataa_v1.common import DataAError, utc_now_iso, write_json


SCHEMA_VERSION = "dataA_camera_flow_probe_manifest_v1"
CASE_RE = re.compile(r"(dataA_v1_(?:(?:dataset_v2|textedit_reserve)_)?\d+)")


@dataclass(frozen=True)
class RunSpec:
    name: str
    root: Path
    expected_cases: int
    vace_model: str


FINAL_RUN_SPECS = (
    RunSpec(
        "vace14b_reused",
        Path("/tmp/cameramotion_det/dataA_v1/vace14b/dataa_v1_subject_first_vace14b_finalv1"),
        198,
        "vace14b",
    ),
    RunSpec(
        "vace13b_dataset_40step_v3",
        Path("/tmp/cameramotion_det/dataA_v1/vace13b/dataa_v1_dataset_v2_vace13b_40step_v3"),
        714,
        "vace13b",
    ),
    RunSpec(
        "vace13b_textedit_40step_v3",
        Path("/tmp/cameramotion_det/dataA_v1/vace13b/dataa_v1_textedit_reserve_vace13b_40step_v3"),
        168,
        "vace13b",
    ),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataAError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise DataAError(f"JSONL row is not an object at {path}:{line_no}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").rstrip("/")


def case_id_from_value(value: Any) -> str | None:
    match = CASE_RE.search(str(value or "").replace("\\", "/"))
    return match.group(1) if match else None


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _walk_strings(child)
    elif isinstance(value, str):
        yield value


def load_case_ids(path: Path) -> set[str]:
    if path.suffix.lower() == ".jsonl":
        payload: Any = read_jsonl(path)
    else:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataAError(f"cannot read split file {path}: {exc}") from exc
    case_ids = {case_id for text in _walk_strings(payload) if (case_id := case_id_from_value(text))}
    if not case_ids:
        raise DataAError(f"no Data A case ids found in split file: {path}")
    return case_ids


def camera_bucket(labels: Sequence[str]) -> str:
    normalized = {str(label).strip().casefold() for label in labels}
    if "complex-motion" in normalized:
        return "complex-motion"
    if "minor-motion" in normalized:
        return "minor-motion"
    if "no-motion" in normalized or "static" in normalized:
        return "no-motion"
    return "unknown"


def load_camera_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        normalized = normalize_path(row.get("path"))
        case_id = case_id_from_value(normalized)
        role = "real" if normalized.endswith("/real") else "fake" if normalized.endswith("/fake") else ""
        if not case_id or not role:
            continue
        key = (case_id, role)
        if key in output:
            raise DataAError(f"duplicate camera record: {case_id}:{role}")
        labels = sorted({str(value).strip() for value in row.get("labels") or [] if str(value).strip()})
        output[key] = {
            "labels": labels,
            "caption": str(row.get("caption") or "").strip(),
            "path": normalized,
        }
    return output


def _record_path(record: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = normalize_path(record.get(key))
        if value:
            return value
    return ""


def _match_run(path: str, run_specs: Sequence[RunSpec]) -> RunSpec | None:
    normalized = normalize_path(path)
    for spec in run_specs:
        root = normalize_path(spec.root)
        if normalized == root or normalized.startswith(root + "/"):
            return spec
    return None


def _require_file(path: str, label: str, case_id: str) -> None:
    if not path or not Path(path).is_file():
        raise DataAError(f"missing {label} for {case_id}: {path}")


def build_probe_manifest(
    *,
    records_jsonl: Path,
    camera_jsonl: Path,
    test_split: Path,
    out_jsonl: Path,
    out_summary: Path,
    run_specs: Sequence[RunSpec] = FINAL_RUN_SPECS,
    expected_cases: int | None = 1080,
    expected_test_cases: int | None = 321,
    check_files: bool = False,
    strict_final_contract: bool = True,
) -> dict[str, Any]:
    records = read_jsonl(records_jsonl)
    camera = load_camera_records(camera_jsonl)
    requested_test_ids = load_case_ids(test_split)
    manifest: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    motion_counts: Counter[str] = Counter()
    camera_missing: list[str] = []

    for record in records:
        case_id = str(record.get("case_id") or "").strip()
        if not case_id:
            raise DataAError("grounded-CoT record is missing case_id")
        if case_id in seen:
            raise DataAError(f"duplicate grounded-CoT case: {case_id}")
        seen.add(case_id)

        real_video = _record_path(record, "real_video")
        fake_video = _record_path(record, "fake_video")
        mask_npz = _record_path(record, "mask_npz")
        case_manifest = _record_path(record, "case_manifest_path", "case_manifest")
        source_spec = _match_run(case_manifest or fake_video, run_specs)
        if source_spec is None:
            raise DataAError(
                f"case outside the approved 40step_v3 run roots: {case_id}: "
                f"{case_manifest or fake_video}"
            )
        declared_model = str(record.get("vace_model") or "").strip().casefold()
        if declared_model and declared_model != source_spec.vace_model:
            raise DataAError(
                f"VACE model/run mismatch for {case_id}: declared={declared_model} "
                f"root={source_spec.vace_model}"
            )
        if check_files:
            _require_file(real_video, "real video", case_id)
            _require_file(fake_video, "fake video", case_id)
            _require_file(mask_npz, "mask NPZ", case_id)
            _require_file(case_manifest, "case manifest", case_id)

        real_camera = camera.get((case_id, "real"))
        fake_camera = camera.get((case_id, "fake"))
        if real_camera is None or fake_camera is None:
            camera_missing.append(case_id)
            labels: list[str] = []
            camera_consistent: bool | None = None
        else:
            labels = list(real_camera["labels"])
            camera_consistent = labels == list(fake_camera["labels"])
            if strict_final_contract and not camera_consistent:
                raise DataAError(f"real/fake camera labels disagree for {case_id}")

        split = "test" if case_id in requested_test_ids else "train"
        bucket = camera_bucket(labels)
        source_counts[source_spec.name] += 1
        model_counts[source_spec.vace_model] += 1
        split_counts[split] += 1
        motion_counts[bucket] += 1
        manifest.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": case_id,
                "dataset_split": split,
                "source_name": source_spec.name,
                "run_root": str(source_spec.root),
                "vace_model": source_spec.vace_model,
                "operation": record.get("operation"),
                "real_video": real_video,
                "fake_video": fake_video,
                "mask_npz": mask_npz,
                "case_manifest": case_manifest,
                "edit_time_range_source_sec": record.get("edit_time_range_source_sec")
                or record.get("edit_time_range"),
                "edit_bbox_1000": record.get("edit_bbox") or record.get("evidence_bbox"),
                "camera_labels": labels,
                "camera_caption": "" if real_camera is None else real_camera["caption"],
                "camera_pair_consistent": camera_consistent,
                "motion_bucket": bucket,
            }
        )

    missing_test_ids = sorted(requested_test_ids - seen)
    actual_test_ids = seen & requested_test_ids
    if strict_final_contract and missing_test_ids:
        raise DataAError(
            f"test split contains {len(missing_test_ids)} cases absent from 40step_v3 records: "
            f"{missing_test_ids[:20]}"
        )
    if strict_final_contract and camera_missing:
        raise DataAError(
            f"camera labels missing for {len(camera_missing)} cases: {sorted(camera_missing)[:20]}"
        )
    if expected_cases is not None and len(manifest) != expected_cases:
        raise DataAError(f"case count mismatch: expected={expected_cases} actual={len(manifest)}")
    if expected_test_cases is not None and len(actual_test_ids) != expected_test_cases:
        raise DataAError(
            f"test case count mismatch: expected={expected_test_cases} actual={len(actual_test_ids)}"
        )
    if strict_final_contract:
        for spec in run_specs:
            actual = source_counts[spec.name]
            if actual != spec.expected_cases:
                raise DataAError(
                    f"source count mismatch for {spec.name}: expected={spec.expected_cases} actual={actual}"
                )

    manifest.sort(key=lambda row: str(row["case_id"]))
    write_jsonl(out_jsonl, manifest)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now_iso(),
        "records_jsonl": str(records_jsonl),
        "camera_jsonl": str(camera_jsonl),
        "test_split": str(test_split),
        "out_jsonl": str(out_jsonl),
        "strict_final_contract": bool(strict_final_contract),
        "check_files": bool(check_files),
        "case_count": len(manifest),
        "source_counts": dict(source_counts),
        "vace_model_counts": dict(model_counts),
        "split_counts": dict(split_counts),
        "motion_bucket_counts": dict(motion_counts),
        "camera_missing_count": len(camera_missing),
        "camera_missing_cases": sorted(camera_missing),
        "test_cases_absent_from_records": missing_test_ids,
        "manifest_sha256": sha256(out_jsonl),
    }
    write_json(out_summary, summary)
    return summary
