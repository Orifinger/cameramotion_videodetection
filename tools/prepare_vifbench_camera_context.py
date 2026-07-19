#!/usr/bin/env python3
"""Canonicalize predicted ViF-Bench camera context and audit prompt parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


CAMERA_TEMPLATE = (
    "<camera_motion>\n"
    "<labels>{camera_labels}</labels>\n"
    "<caption>{camera_caption}</caption>\n"
    "</camera_motion>"
)
KEY_FIELDS = (
    "video_id",
    "path",
    "frame_dir_path",
    "frame_dir",
    "video_path",
)
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def normalize(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.rstrip("/")


def strip_video_suffix(value: str) -> str:
    path = PurePosixPath(value)
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        return value
    return str(path.with_suffix(""))


def relative_after_marker(value: str, marker: str = "test_normalized") -> str:
    parts = PurePosixPath(value).parts
    try:
        index = parts.index(marker)
    except ValueError:
        return ""
    return "/".join(parts[index + 1 :])


def last_parts(value: str, count: int) -> str:
    parts = PurePosixPath(value).parts
    return "/".join(parts[-count:]) if len(parts) >= count else value


def video_id_from_frame_dir(frame_dir: str) -> str:
    relative = relative_after_marker(frame_dir)
    if relative:
        return relative
    path = PurePosixPath(frame_dir)
    return f"{path.parent.name}/{path.name}"


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = [json.loads(line) for line in handle if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, Mapping):
            for field in ("data", "items", "records", "predictions"):
                candidate = payload.get(field)
                if isinstance(candidate, list):
                    payload = candidate
                    break
    if not isinstance(payload, list):
        raise ValueError(f"camera context must be a JSON list or JSONL: {path}")
    rows = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise ValueError(f"camera row {index} is not an object")
        rows.append(dict(row))
    return rows


def read_vif_index(index_dir: Path, expected_ranks: int) -> list[dict[str, str]]:
    files = sorted(index_dir.glob("test_index.rank*.json"))
    if len(files) != expected_ranks:
        raise ValueError(
            f"expected {expected_ranks} index shards under {index_dir}, found {len(files)}"
        )
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"index shard is not an object: {path}")
        for source, frame_dirs in payload.items():
            if not isinstance(frame_dirs, list):
                raise ValueError(f"index entry {source!r} is not a list: {path}")
            for raw_frame_dir in frame_dirs:
                frame_dir = normalize(raw_frame_dir)
                if "full-videos" in frame_dir:
                    continue
                video_id = video_id_from_frame_dir(frame_dir)
                if video_id in seen:
                    raise ValueError(f"duplicate ViF video_id across index shards: {video_id}")
                seen.add(video_id)
                rows.append(
                    {
                        "video_id": video_id,
                        "frame_dir_path": frame_dir,
                        "aigc_model_name": str(source),
                    }
                )
    if not rows:
        raise ValueError(f"no ViF-Bench samples found under {index_dir}")
    return rows


def context_signature(row: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    labels = row.get("labels")
    if not isinstance(labels, list):
        raise ValueError("camera row labels must be a list")
    normalized_labels = tuple(str(label).strip() for label in labels if str(label).strip())
    caption = str(row.get("caption", "")).strip()
    if not normalized_labels:
        raise ValueError("camera row has no non-empty labels")
    if not caption:
        raise ValueError("camera row has an empty caption")
    return normalized_labels, caption


def row_values(row: Mapping[str, Any]) -> list[str]:
    values = []
    for field in KEY_FIELDS:
        value = normalize(row.get(field))
        if value:
            values.append(strip_video_suffix(value))
    return list(dict.fromkeys(values))


def query_tiers(values: Iterable[str]) -> dict[str, set[str]]:
    exact: set[str] = set()
    relative: set[str] = set()
    tail2: set[str] = set()
    basename: set[str] = set()
    for raw_value in values:
        value = strip_video_suffix(normalize(raw_value))
        if not value:
            continue
        exact.add(value)
        rel = relative_after_marker(value)
        if rel:
            relative.add(rel)
        tail2.add(last_parts(value, 2))
        basename.add(PurePosixPath(value).name)
    return {
        "exact": exact,
        "relative_to_test_normalized": relative,
        "last_two_path_parts": tail2,
        "basename": basename,
    }


def build_maps(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, set[int]]]:
    maps: dict[str, dict[str, set[int]]] = {
        name: defaultdict(set)
        for name in ("exact", "relative_to_test_normalized", "last_two_path_parts", "basename")
    }
    for index, row in enumerate(rows):
        for tier, keys in query_tiers(row_values(row)).items():
            for key in keys:
                maps[tier][key].add(index)
    return maps


def resolve_row(
    expected: Mapping[str, str],
    raw_rows: Sequence[Mapping[str, Any]],
    maps: Mapping[str, Mapping[str, set[int]]],
) -> tuple[int | None, str, list[int]]:
    queries = query_tiers([expected["video_id"], expected["frame_dir_path"]])
    for tier in ("exact", "relative_to_test_normalized", "last_two_path_parts", "basename"):
        candidates: set[int] = set()
        for key in queries[tier]:
            candidates.update(maps[tier].get(key, set()))
        if not candidates:
            continue
        if len(candidates) == 1:
            return next(iter(candidates)), tier, []
        signatures = {context_signature(raw_rows[index]) for index in candidates}
        if len(signatures) == 1:
            return min(candidates), f"{tier}_identical_duplicate", sorted(candidates)
        return None, f"{tier}_ambiguous", sorted(candidates)
    return None, "unmatched", []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(args: argparse.Namespace) -> None:
    expected = read_vif_index(args.index_dir, args.expected_ranks)
    raw_rows = read_json_or_jsonl(args.camera_json)
    invalid_rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        try:
            context_signature(row)
            if not row_values(row):
                raise ValueError(f"none of {KEY_FIELDS} is present")
        except ValueError as exc:
            invalid_rows.append({"row_index": index, "reason": str(exc)})
    if invalid_rows:
        raise ValueError(f"invalid camera rows: {invalid_rows[:10]}")

    maps = build_maps(raw_rows)
    canonical: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    methods: Counter[str] = Counter()
    usage: Counter[int] = Counter()
    for item in expected:
        row_index, method, candidates = resolve_row(item, raw_rows, maps)
        methods[method] += 1
        if row_index is None:
            failure = {
                **item,
                "reason": method,
                "candidate_row_indices": candidates,
            }
            if candidates:
                ambiguous.append(failure)
            else:
                unmatched.append(failure)
            continue
        usage[row_index] += 1
        labels, caption = context_signature(raw_rows[row_index])
        source_values = row_values(raw_rows[row_index])
        canonical.append(
            {
                **item,
                "path": item["frame_dir_path"],
                "labels": list(labels),
                "caption": caption,
                "source_camera_row_index": row_index,
                "source_camera_key": source_values[0],
                "match_method": method,
            }
        )

    reused_rows = {str(index): count for index, count in usage.items() if count > 1}
    coverage = len(canonical) / len(expected)
    write_jsonl(args.output_jsonl, canonical)
    summary = {
        "schema_version": "vifbench_predicted_camera_context_v1",
        "camera_context_kind": "predicted CameraBench labels+caption; not gold labels",
        "index_dir": str(args.index_dir),
        "raw_camera_json": str(args.camera_json),
        "canonical_camera_jsonl": str(args.output_jsonl),
        "expected_vif_samples": len(expected),
        "raw_camera_rows": len(raw_rows),
        "matched_samples": len(canonical),
        "coverage": coverage,
        "min_required_coverage": args.min_coverage,
        "match_method_counts": dict(methods),
        "unmatched_count": len(unmatched),
        "ambiguous_count": len(ambiguous),
        "reused_source_camera_rows": reused_rows,
        "first_unmatched": unmatched[:30],
        "first_ambiguous": ambiguous[:30],
        "raw_camera_sha256": sha256(args.camera_json),
        "canonical_camera_sha256": sha256(args.output_jsonl),
        "status": "passed" if coverage >= args.min_coverage and not ambiguous and not reused_rows else "failed",
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "passed":
        raise SystemExit(2)


def decode_prompt_file(path: Path) -> str:
    raw = path.read_text(encoding="utf-8").rstrip("\r\n")
    return (
        raw.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\\", "\\")
    )


def audit_prompts(args: argparse.Namespace) -> None:
    detection = json.loads(args.detection_json.read_text(encoding="utf-8-sig"))
    if not isinstance(detection, list) or not detection:
        raise ValueError(f"empty DataB detection JSON: {args.detection_json}")
    messages = detection[0].get("messages", [])
    source_system = next(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "system"
    )
    source_user = next(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "user"
    )
    system_prompt = decode_prompt_file(args.system_prompt_file)
    no_camera_suffix = decode_prompt_file(args.no_camera_suffix_file)
    with_camera_suffix = decode_prompt_file(args.with_camera_suffix_file)
    expected_with_camera = no_camera_suffix + "\n\n" + CAMERA_TEMPLATE
    checks = {
        "system_prompt_matches_DataB_training": system_prompt == source_system,
        "no_camera_suffix_matches_DataB_training": source_user.endswith(no_camera_suffix),
        "camera_suffix_is_exact_training_append": with_camera_suffix == expected_with_camera,
        "camera_labels_placeholder_once": with_camera_suffix.count("{camera_labels}") == 1,
        "camera_caption_placeholder_once": with_camera_suffix.count("{camera_caption}") == 1,
        "no_extra_camera_instruction": "Use camera motion" not in with_camera_suffix,
    }
    summary = {
        "schema_version": "datab_explicit_camera_vifbench_prompt_audit_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "detection_json": str(args.detection_json),
        "system_prompt_file": str(args.system_prompt_file),
        "no_camera_suffix_file": str(args.no_camera_suffix_file),
        "with_camera_suffix_file": str(args.with_camera_suffix_file),
        "camera_append_template": CAMERA_TEMPLATE.replace("\n", "\\n"),
    }
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "passed":
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--index-dir", type=Path, required=True)
    prepare_parser.add_argument("--camera-json", type=Path, required=True)
    prepare_parser.add_argument("--output-jsonl", type=Path, required=True)
    prepare_parser.add_argument("--summary-json", type=Path, required=True)
    prepare_parser.add_argument("--expected-ranks", type=int, default=16)
    prepare_parser.add_argument("--min-coverage", type=float, default=1.0)
    prepare_parser.set_defaults(func=prepare)

    audit_parser = subparsers.add_parser("audit-prompts")
    audit_parser.add_argument("--detection-json", type=Path, required=True)
    audit_parser.add_argument("--system-prompt-file", type=Path, required=True)
    audit_parser.add_argument("--no-camera-suffix-file", type=Path, required=True)
    audit_parser.add_argument("--with-camera-suffix-file", type=Path, required=True)
    audit_parser.add_argument("--output-json", type=Path, required=True)
    audit_parser.set_defaults(func=audit_prompts)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
