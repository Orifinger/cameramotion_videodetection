"""Runtime path resolution for Data A mask tubes.

Path mapping never mutates the source track bank. It only reports whether a
runtime can read the original or mapped path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional

from .common import is_local_path, normalize_prefix, path_has_prefix


VOLATILE_DEFAULT_PREFIXES = ("/tmp", "/var/tmp")


@dataclass
class ResolvedPath:
    original_path: Optional[str]
    resolved_path: Optional[str]
    state: str
    mapping_rule: Optional[str] = None
    exists: bool = False
    is_volatile: bool = False
    note: Optional[str] = None


@dataclass
class PathRule:
    source_prefix: str
    persistent_prefix: str
    status: str = "planned_or_verified"


class PathResolver:
    """Resolve mask paths without claiming unverified persistence."""

    def __init__(self, mapping: Optional[Mapping[str, Any]] = None) -> None:
        mapping = mapping or {}
        self.volatile_prefixes = tuple(
            normalize_prefix(p) for p in mapping.get("volatile_prefixes", VOLATILE_DEFAULT_PREFIXES)
        )
        self.rules: List[PathRule] = []
        for entry in mapping.get("rules", []):
            if not isinstance(entry, Mapping):
                continue
            source = entry.get("source_prefix")
            destination = entry.get("persistent_prefix")
            if source and destination:
                self.rules.append(
                    PathRule(
                        source_prefix=normalize_prefix(str(source)),
                        persistent_prefix=str(destination).rstrip("/"),
                        status=str(entry.get("status", "planned_or_verified")),
                    )
                )
        self.rules.sort(key=lambda rule: len(rule.source_prefix), reverse=True)
        self.explicit = {
            str(source): str(destination)
            for source, destination in dict(mapping.get("explicit_overrides", {})).items()
        }

    def resolve(self, raw_path: Optional[str]) -> ResolvedPath:
        if not raw_path:
            return ResolvedPath(None, None, "missing", note="mask_tube_path is absent")

        raw_path = str(raw_path)
        raw_is_local = is_local_path(raw_path)
        raw_exists = bool(raw_is_local and Path(raw_path).is_file())
        raw_volatile = self.is_volatile(raw_path)

        if raw_exists and not raw_volatile:
            return ResolvedPath(raw_path, raw_path, "readable_persistent", exists=True, note="original path is directly readable")

        mapped_path, mapping_rule = self.lookup_mapping(raw_path)
        if mapped_path:
            mapped = self._mapped(raw_path, mapped_path, mapping_rule or "mapping")
            if mapped.state == "readable_persistent":
                return mapped
            if raw_exists and raw_volatile:
                return ResolvedPath(
                    original_path=raw_path,
                    resolved_path=raw_path,
                    state="readable_volatile",
                    mapping_rule=mapping_rule,
                    exists=True,
                    is_volatile=True,
                    note=f"volatile source readable; persistent mapping is not verified: {mapped_path}",
                )
            return mapped

        if raw_exists:
            return ResolvedPath(
                original_path=raw_path,
                resolved_path=raw_path,
                state="readable_volatile" if raw_volatile else "readable_persistent",
                exists=True,
                is_volatile=raw_volatile,
                note="original path is directly readable",
            )

        return ResolvedPath(
            original_path=raw_path,
            resolved_path=None,
            state="missing",
            is_volatile=raw_volatile,
            note="path is not readable and no mapping matched",
        )

    def lookup_mapping(self, raw_path: str) -> tuple[Optional[str], Optional[str]]:
        if raw_path in self.explicit:
            return self.explicit[raw_path], "explicit_override"
        raw_norm = normalize_prefix(raw_path)
        for rule in self.rules:
            if path_has_prefix(raw_norm, rule.source_prefix):
                suffix = raw_norm[len(rule.source_prefix):].lstrip("/\\")
                destination = rule.persistent_prefix.rstrip("/") + "/" + suffix.replace("\\", "/")
                return destination, f"rule:{rule.source_prefix}"
        return None, None

    def _mapped(self, original: str, mapped: str, mapping_rule: str) -> ResolvedPath:
        exists = bool(is_local_path(mapped) and Path(mapped).is_file())
        if exists:
            return ResolvedPath(
                original_path=original,
                resolved_path=mapped,
                state="readable_persistent",
                mapping_rule=mapping_rule,
                exists=True,
                note="mapped persistent path is readable",
            )
        return ResolvedPath(
            original_path=original,
            resolved_path=mapped,
            state="mapped_but_unverified",
            mapping_rule=mapping_rule,
            exists=False,
            note="mapping exists but this runtime cannot verify its destination",
        )

    def is_volatile(self, path: str) -> bool:
        normalized = normalize_prefix(path)
        return any(path_has_prefix(normalized, prefix) for prefix in self.volatile_prefixes)

