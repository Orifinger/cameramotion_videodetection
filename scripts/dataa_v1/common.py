"""Shared helpers for Data A v1 VACE packaging.

This module intentionally contains no model or generation code.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any, Mapping


class DataAError(ValueError):
    """Raised for deterministic Data A packager failures."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataAError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataAError(f"Invalid JSON in {path}: {exc}") from exc


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def normalize_prefix(value: str) -> str:
    return str(value).replace("\\", "/").rstrip("/") or "/"


def path_has_prefix(path: str, prefix: str) -> bool:
    path_n = normalize_prefix(path)
    prefix_n = normalize_prefix(prefix)
    return path_n == prefix_n or path_n.startswith(prefix_n + "/")


def is_local_path(path: str) -> bool:
    return not path.startswith(("oss://", "s3://", "http://", "https://"))
