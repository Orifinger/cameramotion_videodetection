"""Shared contracts for the forensic temporal expert gate."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANSWER_RE = re.compile(r"<answer>\s*(Real|Fake)\s*</answer>", re.IGNORECASE)


def normalize_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.rstrip("/")


def path_key(value: Any) -> str:
    return normalize_path(value).casefold()


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    )


def stable_hash(value: str, seed: int = 0) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def feature_filename(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24] + ".npz"


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
        raise ValueError(f"expected JSON list or JSONL records: {path}")
    output: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} is not an object: {path}")
        output.append(dict(row))
    return output


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


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
    raise ValueError("detection row has no canonical Real/Fake answer")


def common_frame_directory(images: Sequence[str]) -> str:
    if not images:
        raise ValueError("sample has no image paths")
    parents = {
        path_key(PurePosixPath(normalize_path(value)).parent) for value in images
    }
    if len(parents) != 1:
        raise ValueError(f"sample spans multiple frame directories: {sorted(parents)}")
    return normalize_path(PurePosixPath(normalize_path(images[0])).parent)


def frame_paths_in_directory(path: Path) -> list[str]:
    values = [
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.casefold() in IMAGE_SUFFIXES
    ]
    values.sort(key=lambda child: natural_key(child.name))
    return [normalize_path(value) for value in values]


def label_from_path(value: Any) -> str:
    parts = {part.casefold() for part in PurePosixPath(normalize_path(value)).parts}
    has_real = "real" in parts
    has_fake = "fake" in parts
    if has_real == has_fake:
        raise ValueError(f"cannot infer one Real/Fake label from path: {value}")
    return "Real" if has_real else "Fake"


def source_and_split_from_datab(frame_dir: str) -> tuple[str, str]:
    text = path_key(frame_dir)
    parts = list(PurePosixPath(normalize_path(frame_dir)).parts)
    lower = [part.casefold() for part in parts]
    if "genbuster-200k" in text:
        split = "unknown"
        if "parsed_frames" in lower:
            index = max(i for i, value in enumerate(lower) if value == "parsed_frames")
            if index + 1 < len(parts):
                split = parts[index + 1]
        return "GenBuster-200K", split
    if "vif-cot-4k" in text or "/1vif4k/" in text:
        source = "unknown"
        if "parsed_frames" in lower:
            index = max(i for i, value in enumerate(lower) if value == "parsed_frames")
            if index + 1 < len(parts):
                source = parts[index + 1]
        return "ViF-CoT-4K", source
    return "other", "unknown"


def generator_from_labeled_path(frame_dir: str, label_name: str) -> str:
    parts = list(PurePosixPath(normalize_path(frame_dir)).parts)
    lower = [part.casefold() for part in parts]
    label = label_name.casefold()
    if label in lower:
        index = lower.index(label)
        if index + 1 < len(parts):
            candidate = parts[index + 1]
            if candidate.casefold() not in {"real", "fake", "frames"}:
                return candidate
    return "real" if label == "real" else "unknown_fake"


def video_id_from_frame_dir(frame_dir: str) -> str:
    text = normalize_path(frame_dir)
    lowered = text.casefold()
    for marker in ("/parsed_frames/parsed_frames/", "/test_normalized/"):
        if marker in lowered:
            text = text[lowered.index(marker) + len(marker) :]
            break
    parts = list(PurePosixPath(text.lstrip("/")).parts)
    if len(parts) >= 3 and parts[0].casefold() in {"real", "fake"}:
        parts = parts[1:]
    if len(parts) < 2:
        return "/".join(parts)
    if parts[0].casefold() == "real":
        parts[0] = "real"
    return "/".join(parts)


def group_identity(frame_dir: str, source_name: str) -> str:
    leaf = PurePosixPath(normalize_path(frame_dir)).name
    return f"{source_name}:{leaf}"


def read_vif_index(index_dir: Path, expected_ranks: int = 16) -> list[dict[str, str]]:
    files = sorted(index_dir.glob("test_index.rank*.json"))
    if expected_ranks > 0 and len(files) != expected_ranks:
        raise ValueError(
            f"expected {expected_ranks} ViF index shards, found {len(files)} under {index_dir}"
        )
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
                    raise ValueError(f"duplicate ViF video id: {video_id}")
                seen.add(video_id)
                rows.append(
                    {
                        "video_id": video_id,
                        "frame_dir_path": frame_dir,
                        "generator_name": str(source),
                    }
                )
    if not rows:
        raise ValueError(f"no ViF records found under {index_dir}")
    return rows


def compact_counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))

