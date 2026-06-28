#!/usr/bin/env python3
"""Freeze a full Data A v1 VACE execution plan from existing candidate records.

This script does not discover objects, re-run tracking, re-pair donors, or
sample new cases. It normalizes the provided plan/track-bank records, audits
mask availability, writes passed cases to the frozen execution plan, and writes
all blocked cases to a companion blockers report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.audit_vace_stage1 import audit_case, load_cases_for_audit
from scripts.dataa_v1.common import DataAError, utc_now_iso, write_json
from scripts.dataa_v1.execution_plan import validate_execution_cases
from scripts.dataa_v1.schema import CanonicalCaseSpec, serialize_case


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_project_path(path: Path | str | None) -> Path | None:
    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else _repo_root() / value


def _freeze_case(case: CanonicalCaseSpec, audit: dict[str, Any]) -> dict[str, Any]:
    payload = serialize_case(case, include_raw=True)
    sampling_meta = dict(payload.get("sampling_meta") or {})
    sampling_meta.update(
        {
            "frozen": True,
            "frozen_at_utc": utc_now_iso(),
            "freeze_source": "scripts/dataa_v1/freeze_full_vace_execution_plan.py",
            "preflight_stage_status": audit["stage_status"],
            "preflight_blockers": audit["blockers"],
        }
    )
    payload["sampling_meta"] = sampling_meta
    return payload


def freeze_full_plan(
    *,
    source_plan: Path,
    track_bank: Path,
    output: Path,
    path_mapping: Optional[Path] = None,
    blockers_output: Optional[Path] = None,
    include_blocked: bool = False,
) -> dict[str, Any]:
    cases = load_cases_for_audit(source_plan, track_bank, path_mapping)
    frozen_cases: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    passed_specs: list[CanonicalCaseSpec] = []
    for case in cases:
        audit = audit_case(case)
        case_validation = validate_execution_cases([case])
        if not case_validation["valid"]:
            audit["stage_status"] = "blocked_plan_validation_failure"
            audit["blockers"] = list(audit.get("blockers") or []) + case_validation["errors"]
            audit["execution_plan_validation"] = case_validation
        if audit["stage_status"] == "preflight_passed":
            frozen_cases.append(_freeze_case(case, audit))
            passed_specs.append(case)
        else:
            blockers.append(audit)
            if include_blocked:
                frozen_cases.append(_freeze_case(case, audit))

    validation = validate_execution_cases(passed_specs)
    plan_payload = {
        "schema_version": "dataA_v1_frozen_full_vace_execution_plan_v1",
        "generated_at_utc": utc_now_iso(),
        "source_plan": str(source_plan),
        "track_bank": str(track_bank),
        "path_mapping": str(path_mapping) if path_mapping else None,
        "case_count": len(frozen_cases),
        "preflight_passed_count": len(passed_specs),
        "blocked_count": len(blockers),
        "include_blocked": include_blocked,
        "validation": validation,
        "cases": frozen_cases,
    }
    write_json(output, plan_payload)

    blocker_path = blockers_output or output.with_name(output.stem + ".blockers.json")
    write_json(
        blocker_path,
        {
            "schema_version": "dataA_v1_frozen_full_vace_blockers_v1",
            "generated_at_utc": utc_now_iso(),
            "source_plan": str(source_plan),
            "track_bank": str(track_bank),
            "path_mapping": str(path_mapping) if path_mapping else None,
            "blocked_count": len(blockers),
            "blockers": blockers,
        },
    )
    return {
        "output": str(output),
        "blockers_output": str(blocker_path),
        "case_count": len(frozen_cases),
        "preflight_passed_count": len(passed_specs),
        "blocked_count": len(blockers),
        "validation": validation,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", "--plan", required=True, type=Path)
    parser.add_argument("--track-bank", required=True, type=Path)
    parser.add_argument("--path-mapping", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("res/dataA_v1/plans/frozen_full_vace_execution_plan.json"),
    )
    parser.add_argument("--blockers-output", type=Path, default=None)
    parser.add_argument("--include-blocked", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = freeze_full_plan(
            source_plan=_resolve_project_path(args.source_plan) or args.source_plan,
            track_bank=_resolve_project_path(args.track_bank) or args.track_bank,
            output=_resolve_project_path(args.output) or args.output,
            path_mapping=_resolve_project_path(args.path_mapping),
            blockers_output=_resolve_project_path(args.blockers_output),
            include_blocked=bool(args.include_blocked),
        )
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        "frozen_full_plan="
        f"{summary['output']} cases={summary['case_count']} "
        f"passed={summary['preflight_passed_count']} blocked={summary['blocked_count']}"
    )
    if not summary["validation"]["valid"]:
        print(f"validation_errors={summary['validation']['errors']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
