"""Shared data contracts for CTNE Gate 1."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MANIFEST_SCHEMA_VERSION = "camera_ctne_manifest_v1"
FEATURE_SCHEMA_VERSION = "camera_ctne_transition_features_v1"
MODEL_SCHEMA_VERSION = "camera_ctne_flow_v1"
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


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value))


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
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
        raise ValueError(f"expected JSON list or JSONL objects: {path}")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} is not an object: {path}")
        rows.append(dict(row))
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_json_or_jsonl(path)


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
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24] + ".npz"


def dataset_slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "dataset").casefold()).strip("_")
    return text or "dataset"


def camera_bucket(labels: Sequence[Any]) -> str:
    values = {
        COARSE_LABELS[label]
        for raw in labels
        if (label := str(raw).strip().casefold()) in COARSE_LABELS
    }
    if not values:
        return "unknown"
    if len(values) > 1:
        return "ambiguous"
    return next(iter(values))


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
    raise ValueError("detection row has no <answer>Real/Fake</answer>")


def frame_paths_in_directory(path: Path) -> list[str]:
    values = [child for child in path.iterdir() if child.is_file() and child.suffix.casefold() in IMAGE_SUFFIXES]
    values.sort(key=lambda child: natural_key(child.name))
    return [normalize_path(child) for child in values]


def label_from_path(value: Any) -> str:
    parts = {part.casefold() for part in PurePosixPath(normalize_path(value)).parts}
    has_real = "real" in parts
    has_fake = "fake" in parts
    if has_real == has_fake:
        raise ValueError(f"cannot infer exactly one Real/Fake label from path: {value}")
    return "Real" if has_real else "Fake"


def source_from_datab_path(value: Any) -> str:
    normalized = path_key(value)
    if "/1genbuster/" in normalized:
        return "GenBuster-select1"
    if "/2genbuster/" in normalized:
        return "GenBuster-select2"
    if "/1vif4k/" in normalized:
        return "ViF-CoT-4K"
    return "other"


def source_from_labeled_path(value: Any) -> str:
    parts = list(PurePosixPath(normalize_path(value)).parts)
    lowered = [part.casefold() for part in parts]
    if "fake" in lowered:
        index = lowered.index("fake")
        return parts[index + 1] if index + 1 < len(parts) else "fake"
    if "real" in lowered:
        index = lowered.index("real")
        candidate = parts[index + 1] if index + 1 < len(parts) else "real"
        return candidate if candidate.casefold() not in {"real", "frames"} else "real"
    return "unknown"


def frame_count_bin(count: int) -> str:
    if count < 3:
        return "lt3"
    if count <= 7:
        return "3-7"
    if count <= 15:
        return "8-15"
    if count <= 31:
        return "16-31"
    return "32+"


def compact_counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))


def frame_count_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray([int(row.get("frame_count", 0)) for row in rows], dtype=np.int64)
    if values.size == 0:
        return {"count": 0, "histogram": {}}
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "max": int(values.max()),
        "histogram": dict(sorted(Counter(int(value) for value in values).items())),
        "bin_counts": dict(sorted(Counter(frame_count_bin(int(value)) for value in values).items())),
    }


def camera_path_aliases(value: Any) -> set[str]:
    normalized = normalize_path(value)
    if not normalized:
        return set()
    path = PurePosixPath(normalized)
    parent = str(path.parent).replace("\\", "/")
    leaf = path.name
    aliases = {path_key(normalized)}
    aliases.add(path_key(f"{parent}/{leaf.replace(' ', '_')}"))
    return aliases


def camera_sidecar_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        labels = [str(value).strip() for value in row.get("labels") or [] if str(value).strip()]
        normalized = {
            "path": normalize_path(row.get("path") or row.get("frame_dir_path")),
            "labels": labels,
            "caption": str(row.get("caption") or "").strip(),
            "motion_bucket": camera_bucket(labels),
        }
        for alias in camera_path_aliases(normalized["path"]):
            candidates.setdefault(alias, []).append(normalized)
    output: dict[str, dict[str, Any]] = {}
    for alias, values in candidates.items():
        signatures = {(tuple(value["labels"]), value["caption"]) for value in values}
        if len(signatures) == 1:
            output[alias] = values[0]
    return output


def lookup_camera_row(mapping: Mapping[str, dict[str, Any]], frame_dir: Any) -> dict[str, Any] | None:
    for alias in camera_path_aliases(frame_dir):
        if alias in mapping:
            return dict(mapping[alias])
    return None


def video_id_from_frame_dir(value: Any) -> str:
    normalized = normalize_path(value)
    marker = "/parsed_frames/parsed_frames/"
    lowered = normalized.casefold()
    if marker in lowered:
        index = lowered.index(marker) + len(marker)
        relative = normalized[index:]
        parts = PurePosixPath(relative).parts
        if len(parts) >= 3 and parts[0].casefold() in {"real", "fake"}:
            return "/".join(parts[1:])
    return PurePosixPath(normalized).name


def read_vif_index(index_dir: Path, expected_ranks: int = 16) -> list[dict[str, str]]:
    files = sorted(index_dir.glob("test_index.rank*.json"))
    if expected_ranks > 0 and len(files) != expected_ranks:
        raise ValueError(f"expected {expected_ranks} ViF index shards, found {len(files)} under {index_dir}")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"ViF index shard is not an object: {path}")
        for source, frame_dirs in payload.items():
            if not isinstance(frame_dirs, list):
                raise ValueError(f"ViF index entry is not a list: {path} {source}")
            for raw in frame_dirs:
                frame_dir = normalize_path(raw)
                if "full-videos" in frame_dir.casefold():
                    continue
                video_id = video_id_from_frame_dir(frame_dir)
                if video_id in seen:
                    raise ValueError(f"duplicate ViF video_id: {video_id}")
                seen.add(video_id)
                rows.append(
                    {
                        "video_id": video_id,
                        "frame_dir_path": frame_dir,
                        "source_name": str(source),
                    }
                )
    if not rows:
        raise ValueError(f"no ViF samples found under {index_dir}")
    return rows


def validate_feature_archive(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        schema = str(archive["schema_version"].item())
        if schema != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"feature schema mismatch: {path}: {schema}")
        camera = np.asarray(archive["camera_context"], dtype=np.float32)
        evidence = np.asarray(archive["temporal_evidence"], dtype=np.float32)
        if camera.ndim != 2 or evidence.ndim != 2 or camera.shape[0] != evidence.shape[0]:
            raise ValueError(f"invalid feature shapes: {path}: camera={camera.shape} evidence={evidence.shape}")
        if camera.shape[0] < 1 or not np.isfinite(camera).all() or not np.isfinite(evidence).all():
            raise ValueError(f"empty or non-finite feature archive: {path}")
        return {
            "sample_id": str(archive["sample_id"].item()),
            "label": int(archive["label"].item()),
            "frame_count": int(archive["frame_count"].item()),
            "transition_count": int(camera.shape[0]),
            "camera_dim": int(camera.shape[1]),
            "evidence_dim": int(evidence.shape[1]),
        }
