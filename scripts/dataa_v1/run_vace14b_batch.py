#!/usr/bin/env python3
"""Production Data A v1 VACE-14B batch entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.common import DataAError, read_json, utc_now_iso, write_json
from scripts.dataa_v1.config import apply_cli_overrides, load_config
from scripts.dataa_v1.execution_plan import (
    MissingFrozenFullExecutionPlan,
    discover_full_execution_plan,
    load_execution_plan,
    require_full_plan_for_execute,
)
from scripts.dataa_v1.run_state import RunPaths, RunState
from scripts.dataa_v1.runtime_preflight import check_runtime
from scripts.dataa_v1.schema import serialize_case
from scripts.dataa_v1.topology import build_topology, shard_cases, topology_payload, validate_topology_for_resume
from scripts.dataa_v1.worker_commands import build_worker_command


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_project_path(path: Path | str | None) -> Path | None:
    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else _repo_root() / value


def _default_run_id() -> str:
    return "dataa_v1_vace14b_full_" + utc_now_iso().replace(":", "").replace("-", "").split(".")[0]


def _resolve_execution_plan(args: argparse.Namespace, config: Dict[str, Any], *, execute: bool) -> Path | None:
    path_value = args.execution_plan or config.get("execution", {}).get("full_execution_plan")
    path = _resolve_project_path(path_value) if path_value else discover_full_execution_plan(_repo_root())
    if execute:
        return require_full_plan_for_execute(path)
    return path if path and path.is_file() else None


def _validation_error_case_id(error: str, case_ids: set[str]) -> str | None:
    parts = str(error).split(":")
    for part in parts[1:4]:
        if part in case_ids:
            return part
    return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lineage_key(entry: Dict[str, Any]) -> str:
    text = "|".join(
        [
            str(entry.get("execution_plan_sha256") or ""),
            str(entry.get("case_filter") or {}),
            ",".join(str(case_id) for case_id in entry.get("runnable_case_ids") or []),
        ]
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_plan_lineage(
    *,
    paths: RunPaths,
    run_id: str,
    execution_plan_path: Path,
    plan: Any,
    runnable_case_ids: Sequence[str],
    blocked_case_errors: Mapping[str, list[str]],
    shards: Mapping[int, Sequence[str]],
    args: argparse.Namespace,
) -> None:
    raw = plan.raw if isinstance(plan.raw, dict) else {}
    plan_cases = list(plan.cases)
    plan_case_ids = [case.case_id for case in plan_cases]
    continuation_case_ids = [
        case.case_id
        for case in plan_cases
        if isinstance(case.sampling_meta, dict) and case.sampling_meta.get("continuation")
    ]
    model_counts = Counter(
        str(((case.sampling_meta or {}).get("vace_model_plan") or {}).get("model_name") or "<missing>")
        for case in plan_cases
    )
    route_counts = Counter(str(case.generator_route or "<missing>") for case in plan_cases)
    subject_sources = Counter(str((case.sampling_meta or {}).get("subject_first_source") or "<missing>") for case in plan_cases)
    entry = {
        "first_seen_at_utc": utc_now_iso(),
        "last_seen_at_utc": utc_now_iso(),
        "invocation_count": 1,
        "execution_plan": str(execution_plan_path),
        "execution_plan_sha256": _file_sha256(execution_plan_path),
        "plan_schema_version": raw.get("schema_version"),
        "plan_generated_at_utc": raw.get("generated_at_utc"),
        "base_plan": raw.get("base_plan"),
        "track_bank": raw.get("track_bank"),
        "path_mapping": raw.get("path_mapping"),
        "run_roots": raw.get("run_roots") or [],
        "completed_case_ids_from_plan": raw.get("completed_case_ids") or [],
        "completed_video_ids_from_plan": raw.get("completed_video_ids") or [],
        "case_filter": {"case_id": args.case_id, "max_cases": args.max_cases},
        "case_count": len(plan_case_ids),
        "runnable_case_count": len(runnable_case_ids),
        "blocked_case_count": len(blocked_case_errors),
        "continuation_case_count": len(continuation_case_ids),
        "case_ids": plan_case_ids,
        "runnable_case_ids": list(runnable_case_ids),
        "blocked_case_ids": sorted(blocked_case_errors),
        "continuation_case_ids": continuation_case_ids,
        "operation_counts": dict(plan.validation.get("operation_counts") or {}),
        "route_counts": dict(route_counts),
        "model_counts": dict(model_counts),
        "subject_first_source_counts": dict(subject_sources),
        "shards": {str(worker_id): list(case_ids) for worker_id, case_ids in shards.items()},
    }
    entry["lineage_key"] = _lineage_key(entry)

    path = paths.coordinator_dir / "plan_lineage.json"
    if path.is_file():
        lineage = read_json(path)
    else:
        lineage = {
            "schema_version": "dataA_v1_run_plan_lineage_v1",
            "run_id": run_id,
            "run_root": str(paths.run_root),
            "created_at_utc": utc_now_iso(),
            "plans": [],
            "case_to_plan": {},
        }
    lineage["updated_at_utc"] = utc_now_iso()
    lineage["run_id"] = run_id
    lineage["run_root"] = str(paths.run_root)
    plans = list(lineage.get("plans") or [])
    existing_index = next((idx for idx, item in enumerate(plans) if item.get("lineage_key") == entry["lineage_key"]), None)
    if existing_index is None:
        plans.append(entry)
    else:
        previous = dict(plans[existing_index])
        entry["first_seen_at_utc"] = previous.get("first_seen_at_utc") or entry["first_seen_at_utc"]
        entry["invocation_count"] = int(previous.get("invocation_count") or 1) + 1
        plans[existing_index] = entry
    lineage["plans"] = plans

    case_to_plan: Dict[str, Any] = dict(lineage.get("case_to_plan") or {})
    for case in plan_cases:
        if case.case_id not in set(runnable_case_ids):
            continue
        case_to_plan[case.case_id] = {
            "execution_plan": str(execution_plan_path),
            "execution_plan_sha256": entry["execution_plan_sha256"],
            "lineage_key": entry["lineage_key"],
            "operation": case.operation,
            "generator_route": case.generator_route,
            "target_video_id": case.target.video_id,
            "target_track_id": case.target.track_id,
            "donor_video_id": None if case.donor is None else case.donor.video_id,
            "donor_track_id": None if case.donor is None else case.donor.track_id,
            "is_continuation_case": case.case_id in set(continuation_case_ids),
            "continuation": (case.sampling_meta or {}).get("continuation"),
            "mask_policy": (case.sampling_meta or {}).get("mask_policy"),
            "vace_model_plan": (case.sampling_meta or {}).get("vace_model_plan"),
        }
    lineage["case_to_plan"] = case_to_plan
    write_json(path, lineage)


def plan_batch(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = _resolve_project_path(args.config)
    config = apply_cli_overrides(
        load_config(config_path),
        execution_plan=args.execution_plan,
        checkpoint_dir=args.checkpoint_dir,
        run_id=args.run_id,
        oss_prefix=args.oss_prefix,
        resume=args.resume,
        allow_reshard=args.allow_reshard,
        topology=args.topology,
        workers_per_gpu=getattr(args, "workers_per_gpu", None),
    )
    execute = bool(args.execute)
    execution_plan_path = _resolve_execution_plan(args, config, execute=execute)
    run_id = config["run"].get("run_id") or args.run_id or _default_run_id()
    config["run"]["run_id"] = run_id
    topology = build_topology(config["gpu"])
    paths = RunPaths.from_root(Path(config["run"]["tmp_root"]), run_id)
    state = RunState(paths, run_id=run_id, topology=topology_payload(topology))
    previous = state.load() if args.resume else None
    validate_topology_for_resume((previous or {}).get("topology"), topology, allow_reshard=bool(config["execution"].get("allow_reshard")))

    runtime_report = check_runtime(config, topology, execution_plan=execution_plan_path, strict=False)
    if execute:
        critical_checks = {
            "vace_vendored_source",
            "checkpoint_dir",
            "ffmpeg",
            "ffprobe",
            "torchrun",
            "execution_plan",
            "tmp_root_writable",
        }
        failed = [
            f"{name}={item.get('detail')}"
            for name, item in (runtime_report.get("checks") or {}).items()
            if name in critical_checks and not bool(item.get("ok"))
        ]
        if failed:
            raise DataAError(f"runtime preflight failed before worker launch: {failed}")
    plan_report: Dict[str, Any] = {
        "run_id": run_id,
        "execute": execute,
        "run_root": str(paths.run_root),
        "coordinator_dir": str(paths.coordinator_dir),
        "execution_plan": str(execution_plan_path) if execution_plan_path else None,
        "full_plan_search_result": "found" if execution_plan_path else "missing_frozen_full_execution_plan",
        "topology": topology_payload(topology),
        "runtime_preflight": runtime_report,
        "worker_commands": [],
        "shards": {},
    }
    if execution_plan_path:
        track_bank = _resolve_project_path(args.track_bank) if args.track_bank else None
        path_mapping = _resolve_project_path(args.path_mapping) if args.path_mapping else None
        plan = load_execution_plan(execution_plan_path=execution_plan_path, track_bank_path=track_bank, path_mapping_path=path_mapping)
        plan_report["execution_plan_validation"] = plan.validation
        case_by_id = {case.case_id: case for case in plan.cases}
        blocked_case_errors: Dict[str, list[str]] = {}
        fatal_validation_errors: list[str] = []
        if not plan.validation["valid"]:
            all_case_ids = set(case_by_id)
            for error in plan.validation["errors"]:
                case_id = _validation_error_case_id(str(error), all_case_ids)
                if case_id and bool(config["execution"].get("block_invalid_cases", True)):
                    blocked_case_errors.setdefault(case_id, []).append(str(error))
                else:
                    fatal_validation_errors.append(str(error))
            if fatal_validation_errors and config["execution"].get("strict", True):
                raise DataAError(f"full execution plan validation failed: {fatal_validation_errors}")
            if blocked_case_errors:
                blocked_report = {
                    "status": "blocked_plan_validation_failure",
                    "execution_plan": str(execution_plan_path),
                    "blocked_count": len(blocked_case_errors),
                    "blocked_cases": [
                        {
                            "case_id": case_id,
                            "errors": errors,
                            "case": serialize_case(case_by_id[case_id], include_raw=True),
                        }
                        for case_id, errors in sorted(blocked_case_errors.items())
                    ],
                }
                write_json(paths.coordinator_dir / "blocked_execution_plan_cases.json", blocked_report)
                plan_report["blocked_execution_plan_cases"] = blocked_report
                for case_id, errors in blocked_case_errors.items():
                    state.append_status(case_id, "blocked_plan_validation_failure", worker_id=None, detail={"errors": errors})
        case_ids = [case.case_id for case in plan.cases]
        if args.case_id:
            case_ids = [case_id for case_id in case_ids if case_id == args.case_id]
        case_ids = [case_id for case_id in case_ids if case_id not in blocked_case_errors]
        if args.max_cases is not None:
            case_ids = case_ids[: args.max_cases]
        if not case_ids and args.execute:
            raise DataAError("no runnable cases after execution plan blockers were removed")
        shards = shard_cases(case_ids, run_id, topology)
        plan_report["shards"] = {str(worker_id): ids for worker_id, ids in shards.items()}
        paths.coordinator_dir.mkdir(parents=True, exist_ok=True)
        _write_plan_lineage(
            paths=paths,
            run_id=run_id,
            execution_plan_path=execution_plan_path,
            plan=plan,
            runnable_case_ids=case_ids,
            blocked_case_errors=blocked_case_errors,
            shards=shards,
            args=args,
        )
        plan_report["plan_lineage"] = str(paths.coordinator_dir / "plan_lineage.json")
        for group in topology.groups:
            shard_payload = {
                "run_id": run_id,
                "worker_id": group.worker_id,
                "topology": topology_payload(topology),
                "execution_plan": str(execution_plan_path),
                "track_bank": str(track_bank) if track_bank else None,
                "path_mapping": str(path_mapping) if path_mapping else None,
                "profile": str(config["vace"].get("profile", "production_720")),
                "ffmpeg_bin": str(config["vace"].get("ffmpeg_bin", "ffmpeg")),
                "ffprobe_bin": str(config["vace"].get("ffprobe_bin", "ffprobe")),
                "vace_size": str(config["vace"].get("size") or ("480p" if config["vace"].get("profile") == "smoke_480" else "720p")),
                "cases": [
                    {
                        "case_id": case_id,
                        "attempt_dir": str(paths.attempt_dir(group.worker_id, case_id)),
                        "case": serialize_case(case_by_id[case_id], include_raw=True),
                    }
                    for case_id in shards[group.worker_id]
                ],
            }
            shard_path = paths.coordinator_dir / f"worker_{group.worker_id:02d}_shard.json"
            write_json(shard_path, shard_payload)
            plan_report["worker_commands"].append(
                build_worker_command(
                    group=group,
                    config_path=config_path,
                    shard_path=shard_path,
                    run_id=run_id,
                    torchrun_bin=str(config["vace"].get("torchrun_bin", "torchrun")),
                )
            )
    write_json(paths.coordinator_dir / "batch_plan.json", plan_report)
    state.save(state.load())
    return plan_report


def launch_workers(plan: Dict[str, Any]) -> int:
    commands = plan.get("worker_commands", [])
    procs = []
    logs_dir = Path(str(plan.get("coordinator_dir") or Path(str(plan.get("run_root", "."))) / "coordinator")) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for command in commands:
        env = None
        if command.get("env"):
            import os

            env = os.environ.copy()
            env.update(command["env"])
        else:
            import os

            env = os.environ.copy()
        worker_id = int(command.get("worker_id", len(procs)))
        env["PYTHONUNBUFFERED"] = "1"
        env["TORCHELASTIC_ERROR_FILE"] = str(logs_dir / f"worker_{worker_id:02d}.torchelastic_error.json")
        print(f"launching worker_{worker_id:02d}")
        procs.append((worker_id, subprocess.Popen(command["argv"], env=env)))
    failed: list[tuple[int, int]] = []
    for worker_id, proc in procs:
        code = proc.wait()
        if code != 0:
            failed.append((worker_id, code))
    if not failed:
        return 0
    for worker_id, code in failed:
        print(f"worker_{worker_id:02d} exited with code {code}", file=sys.stderr)
    return 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-plan", type=Path, default=None)
    parser.add_argument("--track-bank", type=Path, default=None)
    parser.add_argument("--path-mapping", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/dataa_v1/vace14b_production.yaml"))
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--oss-prefix", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--allow-reshard", action="store_true", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--topology", default=None, help="GPU topology as worker_groups x gpus_per_worker, e.g. 4x4, 2x8, 16x1.")
    parser.add_argument(
        "--workers-per-gpu",
        "--batch-size",
        dest="workers_per_gpu",
        type=int,
        default=None,
        help="Launch this many worker groups on each physical GPU group. --batch-size is an alias for this worker-level concurrency.",
    )
    parser.add_argument("--launch-workers", action="store_true", help="Actually launch persistent torchrun worker groups after planning.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        plan = plan_batch(args)
    except MissingFrozenFullExecutionPlan as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except DataAError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"run_id={plan['run_id']} topology={plan['topology']['name']} execution_plan={plan['execution_plan']}")
    if plan["full_plan_search_result"] != "found":
        print("missing_frozen_full_execution_plan", file=sys.stderr)
    if args.execute and args.launch_workers:
        return launch_workers(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
