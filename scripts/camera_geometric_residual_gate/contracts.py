"""Shared data contracts for the camera geometric-residual gate."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "camera_geometric_residual_manifest_v1"
FEATURE_SCHEMA_VERSION = "camera_geometric_residual_features_v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANSWER_RE = re.compile(r"<answer>\s*(Real|Fake)\s*</answer>", re.IGNORECASE)
COARSE_LABELS = {
    "complex-motion": "complex-motion",
    "minor-motion": "minor-motion",
    "no-motion": "static/no-motion",
    "static": "static/no-motion",
}


def normalize_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.rstrip("/")


def path_key(value: Any) -> str:
    return normalize_path(value).casefold()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            payload: Any = [json.loads(line) for line in handle if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, Mapping):
            for field in ("data", "items", "records", "predictions"):
                if isinstance(payload.get(field), list):
                    payload = payload[field]
                    break
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list or JSONL records: {path}")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} is not an object: {path}")
        rows.append(dict(row))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def stable_unit(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def feature_filename(sample_id: str) -> str:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]
    return f"{digest}.npz"


def camera_bucket(labels: Sequence[Any]) -> str:
    coarse = {
        COARSE_LABELS[label]
        for raw in labels
        if (label := str(raw).strip().casefold()) in COARSE_LABELS
    }
    if not coarse:
        return "unknown"
    if len(coarse) > 1:
        return "ambiguous"
    return next(iter(coarse))


def assistant_answer(row: Mapping[str, Any]) -> str:
    messages = row.get("messages") or row.get("conversations") or []
    for message in reversed(messages if isinstance(messages, list) else []):
        role = str(message.get("role", message.get("from", ""))).casefold()
        if role not in {"assistant", "gpt"}:
            continue
        content = str(message.get("content", message.get("value", "")))
        match = ANSWER_RE.search(content)
        if match:
            return match.group(1).title()
    raise ValueError("detection row has no valid <answer>Real/Fake</answer>")


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value))


def frame_paths_in_directory(path: Path) -> list[str]:
    values = [child for child in path.iterdir() if child.is_file() and child.suffix.casefold() in IMAGE_SUFFIXES]
    values.sort(key=lambda child: natural_key(child.name))
    return [normalize_path(child) for child in values]


def source_from_datab_path(value: str) -> str:
    normalized = path_key(value)
    if "/1genbuster/" in normalized:
        return "GenBuster-select1"
    if "/2genbuster/" in normalized:
        return "GenBuster-select2"
    if "/1vif4k/" in normalized:
        return "ViF-CoT-4K"
    return "other"


def label_from_vif_path(value: str) -> str:
    parts = {part.casefold() for part in PurePosixPath(normalize_path(value)).parts}
    has_real = "real" in parts
    has_fake = "fake" in parts
    if has_real == has_fake:
        raise ValueError(f"cannot infer exactly one Real/Fake label from ViF path: {value}")
    return "Real" if has_real else "Fake"


def compact_counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))
