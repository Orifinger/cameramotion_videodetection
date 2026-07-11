#!/usr/bin/env python3
"""Merge reused VACE-14B CoT rows with regenerated VACE-1.3B CoT rows."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.common import DataAError, utc_now_iso, write_json


VACE14_CASE_RE = re.compile(r"(?<![A-Za-z0-9_])(dataA_v1_\d{5})(?![A-Za-z0-9_])")
VACE13_CASE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(dataA_v1_(?:dataset_v2|textedit_reserve)_\d+)(?![A-Za-z0-9_])"
)


def _read_sft(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataAError(f"cannot read SFT JSON {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise DataAError(f"SFT JSON must contain a list: {path}")
    rows = [row for row in payload if isinstance(row, dict)]
    if len(rows) != len(payload):
        raise DataAError(f"SFT JSON contains non-object rows: {path}")
    return rows


def _row_text(row: Mapping[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _case_id(row: Mapping[str, Any], pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(_row_text(row))
    return match.group(1) if match else None


def _assistant_text(row: Mapping[str, Any]) -> str:
    messages = row.get("messages") or []
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return ""
    parts: list[str] = []
    for message in messages:
        if isinstance(message, Mapping) and str(message.get("role") or "") == "assistant":
            parts.append(str(message.get("content") or ""))
    return "\n".join(parts)


def _role(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("role") or "").strip().lower()
    if explicit in {"real", "fake"}:
        return explicit
    sample_key = str(row.get("sample_key") or "").strip().lower()
    if sample_key.endswith(":real"):
        return "real"
    if sample_key.endswith(":fake"):
        return "fake"
    images = row.get("images") or []
    image_text = "/".join(str(item).replace("\\", "/") for item in images) if isinstance(images, list) else ""
    has_real = "/real/" in image_text
    has_fake = "/fake/" in image_text
    if has_real != has_fake:
        return "real" if has_real else "fake"
    answer = _assistant_text(row).lower()
    if "<answer>real</answer>" in answer:
        return "real"
    if "<answer>fake</answer>" in answer:
        return "fake"
    raise DataAError("cannot determine real/fake role for SFT row")


def _rewrite_images(row: Mapping[str, Any], *, case_id: str, role: str, frame_root: Path) -> dict[str, Any]:
    images = row.get("images")
    if not isinstance(images, list) or not images:
        raise DataAError(f"missing images for {case_id}:{role}")
    rewritten: list[str] = []
    for source in images:
        destination = frame_root / case_id / role / Path(str(source)).name
        if not destination.is_file():
            raise DataAError(f"missing remapped frame for {case_id}:{role}: {destination}")
        rewritten.append(str(destination))
    output = copy.deepcopy(dict(row))
    output["images"] = rewritten
    return output


def merge_sft(
    *,
    old_sft: Path,
    new_sft: Path,
    frame_root: Path,
    out_path: Path,
    summary_path: Path,
    expected_old14_rows: int,
    expected_new13_rows: int,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    old_rows = _read_sft(old_sft)
    new_rows = _read_sft(new_sft)
    selected: list[tuple[dict[str, Any], str, str, str]] = []
    for row in old_rows:
        case_id = _case_id(row, VACE14_CASE_RE)
        if case_id:
            selected.append((row, case_id, _role(row), "vace14b_reused"))
    old14_count = len(selected)
    for row in new_rows:
        case_id = _case_id(row, VACE13_CASE_RE)
        if case_id:
            selected.append((row, case_id, _role(row), "vace13b_40step"))
    new13_count = len(selected) - old14_count

    if old14_count != expected_old14_rows:
        raise DataAError(f"old VACE-14B row count mismatch: expected={expected_old14_rows} actual={old14_count}")
    if new13_count != expected_new13_rows:
        raise DataAError(f"new VACE-1.3B row count mismatch: expected={expected_new13_rows} actual={new13_count}")

    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    source_counts = {"vace14b_reused": 0, "vace13b_40step": 0}
    for row, case_id, role, source_name in selected:
        key = (case_id, role)
        if key in seen:
            raise DataAError(f"duplicate case/role in merged SFT: {case_id}:{role}")
        seen.add(key)
        merged.append(_rewrite_images(row, case_id=case_id, role=role, frame_root=frame_root))
        source_counts[source_name] += 1

    summary = {
        "schema_version": "dataA_v1_grounded_cot_sft_merge_v1",
        "created_at_utc": utc_now_iso(),
        "dry_run": dry_run,
        "old_sft": str(old_sft),
        "new_sft": str(new_sft),
        "frame_root": str(frame_root),
        "out_path": str(out_path),
        "row_count": len(merged),
        "case_count": len({case_id for case_id, _ in seen}),
        "source_counts": source_counts,
        "duplicate_count": 0,
        "missing_frame_count": 0,
    }
    if not dry_run:
        if out_path.exists() and not overwrite:
            raise DataAError(f"output already exists; pass --overwrite to replace it: {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_json(summary_path, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-sft", type=Path, required=True, help="Previous mixed SFT JSON containing reusable 14B rows.")
    parser.add_argument("--new-sft", type=Path, required=True, help="New 40-step 1.3B SFT JSON.")
    parser.add_argument("--frame-root", type=Path, required=True, help="Unified frame root containing all 14B and 1.3B cases.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--expected-old14-rows", type=int, default=396)
    parser.add_argument("--expected-new13-rows", type=int, default=1764)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = merge_sft(
            old_sft=args.old_sft,
            new_sft=args.new_sft,
            frame_root=args.frame_root,
            out_path=args.out,
            summary_path=args.out_summary,
            expected_old14_rows=args.expected_old14_rows,
            expected_new13_rows=args.expected_new13_rows,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        f"grounded_cot_sft_merge dry_run={summary['dry_run']} rows={summary['row_count']} "
        f"cases={summary['case_count']} sources={summary['source_counts']} out={summary['out_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
