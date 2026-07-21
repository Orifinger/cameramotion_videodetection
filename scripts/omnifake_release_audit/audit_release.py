#!/usr/bin/env python3
"""Audit the public Omni-Fake video release before using it in experiments.

The audit deliberately produces two independent decisions:

1. Whether the release is usable for Real/Fake classification.
2. Whether it contains enough paired/localized supervision to support claims
   about a forensic evidence bottleneck.

The Hub stage only inspects repository metadata.  The local stage inspects
downloaded parquet/archive contents and decodes a stratified video sample.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RELEASES = {
    "set": {
        "repo_id": "JamalLee/Omni-Fake-SET",
        "expected_video_count": 260_000,
        "expected_labels": {"real", "full_synthetic", "tampered"},
        "required_patterns": {
            "real_full_archive": "data/Video/video-set.7z.*",
            "tampered_parquet": "data/Video/train-*.parquet",
        },
    },
    "ood": {
        "repo_id": "JamalLee/Omni-Fake-OOD",
        "expected_video_count": 22_000,
        "expected_labels": {"real", "full_synthetic", "tampered"},
        "required_patterns": {
            "real_full_archive": "data/Video/video-ood.7z",
            "tampered_parquet": "data/Video/test-*.parquet",
        },
    },
}

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
LABEL_FIELDS = {"label", "class", "target"}
GENERATOR_FIELDS = {"generator", "model", "source_model", "aigc_model"}
PAIR_FIELDS = {
    "pair_id", "source_id", "source_video", "source_path", "original_id",
    "original_video", "original_path", "real_id", "real_video", "parent_id",
}
MASK_FIELDS = {
    "mask", "masks", "mask_path", "mask_video", "spatial_mask",
    "tamper_mask", "edit_mask", "segmentation", "segmentation_mask",
}
BBOX_FIELDS = {"bbox", "bboxes", "boxes", "bounding_box", "bounding_boxes"}
TEMPORAL_FIELDS = {
    "timestamp", "timestamps", "time_span", "time_spans", "interval",
    "intervals", "temporal_mask", "tamper_intervals", "start_time", "end_time",
}
VIDEO_FIELDS = {"video", "video_path", "path", "file", "filepath", "filename"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (bytes, bytearray, list, tuple, dict, set)):
        return bool(value)
    return True


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "fake": "full_synthetic",
        "fully_synthetic": "full_synthetic",
        "synthetic": "full_synthetic",
        "partial": "tampered",
        "partially_manipulated": "tampered",
        "manipulated": "tampered",
    }
    return aliases.get(label, label)


def normalized_name(value: str) -> str:
    stem = Path(value).stem.casefold()
    return re.sub(r"[^a-z0-9]+", "", stem)


def first_present(row: Mapping[str, Any], names: set[str]) -> Any:
    lowered = {str(key).casefold(): key for key in row}
    for name in names:
        key = lowered.get(name)
        if key is not None and nonempty(row.get(key)):
            return row.get(key)
    return None


def field_names_matching(columns: Sequence[str], candidates: set[str]) -> list[str]:
    return sorted(column for column in columns if column.casefold() in candidates)


def request_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "omnifake-release-audit/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.load(response)


def hub_tree(repo_id: str) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi  # type: ignore

        rows = []
        for item in HfApi().list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            recursive=True,
            expand=True,
        ):
            rows.append({
                "path": str(getattr(item, "path", "")),
                "type": type(item).__name__,
                "size": int(getattr(item, "size", 0) or 0),
                "blob_id": str(getattr(item, "blob_id", "") or ""),
            })
        return rows
    except ImportError:
        encoded = urllib.parse.quote(repo_id, safe="/")
        url = f"https://huggingface.co/api/datasets/{encoded}/tree/main?recursive=true&expand=true"
        result = request_json(url)
        if not isinstance(result, list):
            raise RuntimeError(f"unexpected Hub tree response for {repo_id}")
        return [dict(item) for item in result if isinstance(item, Mapping)]


def audit_hub_release(kind: str) -> dict[str, Any]:
    contract = RELEASES[kind]
    repo_id = str(contract["repo_id"])
    encoded = urllib.parse.quote(repo_id, safe="/")
    info = request_json(f"https://huggingface.co/api/datasets/{encoded}")
    tree = hub_tree(repo_id)
    paths = [str(item.get("path", "")) for item in tree]
    pattern_counts = {
        name: sum(fnmatch.fnmatch(path, pattern) for path in paths)
        for name, pattern in contract["required_patterns"].items()
    }
    card_data = info.get("cardData") or {}
    license_name = str(card_data.get("license") or "").casefold()
    checks = {
        "repo_accessible": bool(info.get("id") or info.get("_id")),
        "license_is_cc_by_4_0": license_name in {"cc-by-4.0", "cc_by_4_0"},
        "real_full_archive_present": pattern_counts["real_full_archive"] > 0,
        "tampered_parquet_present": pattern_counts["tampered_parquet"] > 0,
    }
    video_files = [item for item in tree if str(item.get("path", "")).startswith("data/Video/")]
    return {
        "release": kind,
        "repo_id": repo_id,
        "revision": str(info.get("sha") or ""),
        "license": license_name,
        "expected_video_count_from_card": contract["expected_video_count"],
        "repository_file_count": len(tree),
        "video_release_file_count": len(video_files),
        "video_release_size_bytes_known": sum(int(item.get("size", 0) or 0) for item in video_files),
        "required_pattern_counts": pattern_counts,
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
        "video_files": sorted(video_files, key=lambda item: str(item.get("path", ""))),
    }


def run_hub(args: argparse.Namespace) -> None:
    releases = {}
    failures = {}
    for kind in ("set", "ood"):
        try:
            releases[kind] = audit_hub_release(kind)
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            failures[kind] = f"{type(exc).__name__}: {exc}"
    summary = {
        "schema_version": "omnifake_hub_release_audit_v1",
        "question": "Are the public Omni-Fake video repositories accessible and complete enough to begin local auditing?",
        "releases": releases,
        "failures": failures,
        "status": "passed" if len(releases) == 2 and not failures and all(
            item["status"] == "passed" for item in releases.values()
        ) else "failed",
        "does_not_establish": "Hub metadata alone does not establish video decodability, pair identity, masks, localization labels, or SET/OOD disjointness.",
    }
    out = Path(args.out_dir)
    write_json(out / "omnifake_hub_release_audit.json", summary)
    for kind, release in releases.items():
        write_csv(out / f"omnifake_{kind}_hub_files.csv", release["video_files"])
    terminal_summary = {
        **summary,
        "releases": {
            kind: {key: value for key, value in release.items() if key != "video_files"}
            for kind, release in releases.items()
        },
    }
    print(json.dumps(terminal_summary, ensure_ascii=False, indent=2))
    if args.fail_on_audit and summary["status"] != "passed":
        raise SystemExit(2)


def parquet_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.parquet") if path.is_file())


def media_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES)


def read_parquet_sample(paths: Sequence[Path], max_rows: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for the local parquet audit") from exc

    rows: list[dict[str, Any]] = []
    schemas: dict[str, list[str]] = {}
    total_rows = 0
    errors = {}
    for path in paths:
        try:
            parquet = pq.ParquetFile(path)
            columns = list(parquet.schema_arrow.names)
            schemas[str(path)] = columns
            total_rows += int(parquet.metadata.num_rows)
            remaining = max(0, max_rows - len(rows))
            if remaining:
                for batch in parquet.iter_batches(batch_size=min(1024, remaining)):
                    for row in batch.to_pylist():
                        if isinstance(row, Mapping):
                            item = dict(row)
                            item["__parquet_path"] = str(path)
                            rows.append(item)
                            if len(rows) >= max_rows:
                                break
                    if len(rows) >= max_rows:
                        break
        except Exception as exc:  # noqa: BLE001
            errors[str(path)] = f"{type(exc).__name__}: {exc}"
    return rows, {
        "num_parquet_files": len(paths),
        "total_rows_from_metadata": total_rows,
        "schemas": schemas,
        "errors": errors,
    }


def infer_label_from_path(path: Path) -> str:
    parts = [part.casefold().replace("-", "_") for part in path.parts]
    if "real" in parts:
        return "real"
    if "tampered" in parts or "partialedit" in parts or "partial_edit" in parts:
        return "tampered"
    if "fake" in parts or "full_synthetic" in parts:
        return "full_synthetic"
    return "unknown"


def row_video_payload(row: Mapping[str, Any]) -> tuple[str | None, bytes | None]:
    value = first_present(row, VIDEO_FIELDS)
    if isinstance(value, Mapping):
        path = value.get("path")
        payload = value.get("bytes")
        return (str(path) if nonempty(path) else None, bytes(payload) if payload else None)
    if isinstance(value, str):
        return value, None
    return None, None


def resolve_video_path(reference: str, roots: Sequence[Path], parquet_path: str | None = None) -> Path | None:
    candidate = Path(reference)
    if candidate.is_file():
        return candidate
    search_roots = list(roots)
    if parquet_path:
        search_roots.insert(0, Path(parquet_path).parent)
    for root in search_roots:
        joined = root / reference
        if joined.is_file():
            return joined
    basename = candidate.name
    if basename:
        for root in roots:
            matches = list(root.rglob(basename))
            if len(matches) == 1 and matches[0].is_file():
                return matches[0]
    return None


def ffprobe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration", "-of", "json", str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    if completed.returncode != 0:
        return {"ok": False, "error": completed.stderr.strip()[-1000:]}
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"invalid ffprobe JSON: {exc}"}
    streams = data.get("streams") or []
    if not streams:
        return {"ok": False, "error": "no video stream"}
    stream = streams[0]
    duration = stream.get("duration") or (data.get("format") or {}).get("duration")
    return {
        "ok": True,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration": float(duration) if duration not in (None, "N/A") else None,
        "nb_frames": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
        "avg_frame_rate": str(stream.get("avg_frame_rate") or ""),
    }


def sample_stratified(candidates: Sequence[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        groups[(str(item.get("label", "unknown")), str(item.get("generator", "unknown")))].append(item)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    keys = sorted(groups)
    while len(selected) < count and keys:
        next_keys = []
        for key in keys:
            values = groups[key]
            if values and len(selected) < count:
                selected.append(values.pop())
            if values:
                next_keys.append(key)
        keys = next_keys
    return selected


def extract_filename(row: Mapping[str, Any]) -> str:
    value = first_present(row, {"filename", "file_name", "video_id", "id"})
    if isinstance(value, str):
        return value
    reference, _ = row_video_payload(row)
    return reference or ""


def audit_local_release(kind: str, root: Path, max_parquet_rows: int, decode_samples: int, seed: int) -> dict[str, Any]:
    parquets = parquet_files(root)
    videos = media_files(root)
    rows, parquet_audit = read_parquet_sample(parquets, max_parquet_rows) if parquets else ([], {
        "num_parquet_files": 0,
        "total_rows_from_metadata": 0,
        "schemas": {},
        "errors": {},
    })
    all_columns = sorted({column for columns in parquet_audit["schemas"].values() for column in columns})
    pair_fields = field_names_matching(all_columns, PAIR_FIELDS)
    mask_fields = field_names_matching(all_columns, MASK_FIELDS)
    bbox_fields = field_names_matching(all_columns, BBOX_FIELDS)
    temporal_fields = field_names_matching(all_columns, TEMPORAL_FIELDS)

    label_counts = Counter()
    generator_counts = Counter()
    pair_covered = mask_covered = bbox_covered = temporal_covered = 0
    names = []
    candidates: list[dict[str, Any]] = []
    for row in rows:
        label = normalize_label(first_present(row, LABEL_FIELDS))
        generator = str(first_present(row, GENERATOR_FIELDS) or "unknown")
        if label:
            label_counts[label] += 1
        generator_counts[generator] += 1
        pair_covered += any(nonempty(row.get(field)) for field in pair_fields)
        mask_covered += any(nonempty(row.get(field)) for field in mask_fields)
        bbox_covered += any(nonempty(row.get(field)) for field in bbox_fields)
        temporal_covered += any(nonempty(row.get(field)) for field in temporal_fields)
        filename = extract_filename(row)
        if filename:
            names.append(filename)
        reference, payload = row_video_payload(row)
        path = resolve_video_path(reference, [root], str(row.get("__parquet_path") or "")) if reference else None
        if path or payload:
            candidates.append({
                "label": label or "unknown",
                "generator": generator,
                "filename": filename,
                "path": str(path) if path else "",
                "payload": payload,
            })

    known_paths = {str(item.get("path")) for item in candidates if item.get("path")}
    for path in videos:
        if str(path) in known_paths:
            continue
        label = infer_label_from_path(path)
        generator = path.parent.name if label == "full_synthetic" else "unknown"
        candidates.append({
            "label": label,
            "generator": generator,
            "filename": path.name,
            "path": str(path),
            "payload": None,
        })
        names.append(path.name)
        if label != "unknown":
            label_counts[label] += 1

    decode_items = []
    ffprobe_available = shutil.which("ffprobe") is not None
    for index, item in enumerate(sample_stratified(candidates, decode_samples, seed)):
        path = Path(str(item.get("path") or "")) if item.get("path") else None
        cleanup = None
        if (path is None or not path.is_file()) and item.get("payload"):
            handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            handle.write(item["payload"])
            handle.close()
            path = Path(handle.name)
            cleanup = path
        if path is None or not path.is_file():
            result = {"ok": False, "error": "video payload/path unavailable"}
        elif not ffprobe_available:
            result = {"ok": False, "error": "ffprobe not installed"}
        else:
            result = ffprobe_video(path)
        if cleanup:
            cleanup.unlink(missing_ok=True)
        decode_items.append({
            "index": index,
            "label": item.get("label"),
            "generator": item.get("generator"),
            "filename": item.get("filename"),
            **result,
        })

    sampled_rows = len(rows)
    decode_ok = sum(bool(item.get("ok")) for item in decode_items)
    expected_labels = set(RELEASES[kind]["expected_labels"])
    found_labels = {label for label, count in label_counts.items() if count > 0}
    binary_checks = {
        "release_root_exists": root.is_dir(),
        "has_parquet_or_extracted_videos": bool(parquets or videos),
        "parquet_files_readable": not parquet_audit["errors"],
        "all_three_labels_observed_in_available_content": expected_labels <= found_labels,
        "decode_sample_available": bool(decode_items),
        "decode_success_rate_at_least_95pct": bool(decode_items) and decode_ok / len(decode_items) >= 0.95,
    }
    paired_spatial_evidence_checks = {
        "source_or_pair_field_present": bool(pair_fields),
        "source_or_pair_coverage_at_least_80pct": sampled_rows > 0 and pair_covered / sampled_rows >= 0.80,
        "true_mask_field_present": bool(mask_fields),
        "true_mask_coverage_at_least_80pct": sampled_rows > 0 and mask_covered / sampled_rows >= 0.80,
    }
    temporal_evidence_checks = {
        "temporal_field_present": bool(temporal_fields),
        "temporal_coverage_at_least_80pct": sampled_rows > 0 and temporal_covered / sampled_rows >= 0.80,
    }
    filename_hash = hashlib.sha256("\n".join(sorted(set(names))).encode("utf-8")).hexdigest()
    return {
        "release": kind,
        "root": str(root),
        "scope": "downloaded_content_only",
        "file_inventory": {
            "parquet_files": len(parquets),
            "extracted_video_files": len(videos),
            "parquet_rows_from_metadata": parquet_audit["total_rows_from_metadata"],
            "parquet_rows_sampled": sampled_rows,
        },
        "schema": {
            "columns": all_columns,
            "pair_fields": pair_fields,
            "mask_fields": mask_fields,
            "bbox_fields": bbox_fields,
            "temporal_fields": temporal_fields,
        },
        "sample_coverage": {
            "source_or_pair": pair_covered / sampled_rows if sampled_rows else 0.0,
            "true_mask": mask_covered / sampled_rows if sampled_rows else 0.0,
            "bbox": bbox_covered / sampled_rows if sampled_rows else 0.0,
            "temporal": temporal_covered / sampled_rows if sampled_rows else 0.0,
        },
        "labels_in_sample_or_paths": dict(label_counts),
        "generators_in_sample": dict(generator_counts.most_common()),
        "decode": {
            "requested": decode_samples,
            "attempted": len(decode_items),
            "succeeded": decode_ok,
            "success_rate": decode_ok / len(decode_items) if decode_items else 0.0,
            "items": decode_items,
        },
        "parquet_errors": parquet_audit["errors"],
        "filename_count": len(set(names)),
        "filename_sha256": filename_hash,
        "binary_detection": {
            "checks": binary_checks,
            "status": "usable" if all(binary_checks.values()) else "insufficient_or_partial",
        },
        "paired_spatial_evidence_supervision": {
            "checks": paired_spatial_evidence_checks,
            "status": "usable" if all(paired_spatial_evidence_checks.values()) else "not_established",
        },
        "temporal_evidence_supervision": {
            "checks": temporal_evidence_checks,
            "status": "usable" if all(temporal_evidence_checks.values()) else "not_established",
        },
        "filenames": sorted(set(names)),
    }


def overlap_summary(set_names: Sequence[str], ood_names: Sequence[str]) -> dict[str, Any]:
    set_exact = {Path(name).name.casefold() for name in set_names if name}
    ood_exact = {Path(name).name.casefold() for name in ood_names if name}
    set_normal = {normalized_name(name) for name in set_names if normalized_name(name)}
    ood_normal = {normalized_name(name) for name in ood_names if normalized_name(name)}
    exact = sorted(set_exact & ood_exact)
    normalized = sorted(set_normal & ood_normal)
    return {
        "exact_basename_overlap_count": len(exact),
        "normalized_stem_overlap_count": len(normalized),
        "first_exact_overlaps": exact[:100],
        "first_normalized_overlaps": normalized[:100],
        "status": "passed" if not exact else "failed",
        "warning": "Filename checks do not replace perceptual duplicate detection on the complete decoded release.",
    }


def run_local(args: argparse.Namespace) -> None:
    set_result = audit_local_release(
        "set", Path(args.set_root), args.max_parquet_rows, args.decode_samples, args.seed
    )
    ood_result = audit_local_release(
        "ood", Path(args.ood_root), args.max_parquet_rows, args.decode_samples, args.seed + 1
    )
    overlap = overlap_summary(set_result.pop("filenames"), ood_result.pop("filenames"))
    binary_usable = (
        set_result["binary_detection"]["status"] == "usable"
        and ood_result["binary_detection"]["status"] == "usable"
        and overlap["status"] == "passed"
    )
    evidence_training_usable = set_result["paired_spatial_evidence_supervision"]["status"] == "usable"
    evidence_evaluation_usable = ood_result["paired_spatial_evidence_supervision"]["status"] == "usable"
    if binary_usable and evidence_training_usable:
        decision = "usable_for_binary_and_evidence_supervision"
    elif binary_usable:
        decision = "usable_for_binary_only"
    else:
        decision = "not_yet_usable_or_only_partially_downloaded"
    summary = {
        "schema_version": "omnifake_local_release_audit_v1",
        "question": "Can the downloaded Omni-Fake video release support binary detection and/or paired localized evidence supervision?",
        "set": set_result,
        "ood": ood_result,
        "set_ood_overlap": overlap,
        "evidence_contract": {
            "training_supervision_available_in_set": evidence_training_usable,
            "held_out_evidence_evaluation_available_in_ood": evidence_evaluation_usable,
        },
        "decision": decision,
        "status": "passed" if binary_usable else "incomplete",
        "does_not_establish": "A sampled local audit does not prove semantic annotation quality or absence of perceptual duplicates in unscanned media.",
        "next_action": (
            "Use Omni-Fake for binary training and separately source verified paired/mask supervision."
            if decision == "usable_for_binary_only"
            else "Freeze the release contract only after reviewing the detailed checks."
        ),
    }
    out = Path(args.out_dir)
    write_json(out / "omnifake_local_release_audit.json", summary)
    write_csv(out / "omnifake_set_decode_items.csv", set_result["decode"]["items"])
    write_csv(out / "omnifake_ood_decode_items.csv", ood_result["decode"]["items"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_audit and not binary_usable:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    hub = subparsers.add_parser("hub", help="Audit the public Hugging Face repository metadata")
    hub.add_argument("--out-dir", required=True)
    hub.add_argument("--fail-on-audit", action="store_true")
    hub.set_defaults(func=run_hub)

    local = subparsers.add_parser("local", help="Audit downloaded parquet files and extracted videos")
    local.add_argument("--set-root", required=True)
    local.add_argument("--ood-root", required=True)
    local.add_argument("--out-dir", required=True)
    local.add_argument("--max-parquet-rows", type=int, default=2_000)
    local.add_argument("--decode-samples", type=int, default=150)
    local.add_argument("--seed", type=int, default=20260721)
    local.add_argument("--fail-on-audit", action="store_true")
    local.set_defaults(func=run_local)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
