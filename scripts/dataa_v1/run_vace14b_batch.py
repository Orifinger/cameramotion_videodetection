#!/usr/bin/env python3
"""Production Data A v1 VACE-14B batch entrypoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dataa_v1.common import DataAError, utc_now_iso, write_json
from scripts.dataa_v1.config import apply_cli_overrides, load_config
from scripts.dataa_v1.execution_plan import (
    MissingFrozenFullExecutionPlan,
    discover_full_execution_plan,
    load_execution_plan,
    require_full_plan_for_execute,
)
from scripts.dataa_v1.run_state import RunPaths, RunState
from scripts.dataa_v1.runtime_preflight import check_runtime
from scripts.dataa_v1.topology import build_topology, shard_cases, topology_payload, validate_topology_for_resume
from scripts.dataa_v1.worker_commands import build_worker_command


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_run_id() -> str:
    return "dataa_v1_vace14b_full_" + utc_now_iso().replace(":", "").replace("-", "").split(".")[0]


def _resolve_execution_plan(args: argparse.Namespace, config: Dict[str, Any], *, execute: bool) -> Path | None:
    path_value = args.execution_plan or config.get("execution", {}).get("full_execution_plan")
    path = Path(path_value) if path_value else discover_full_execution_plan(_repo_root())
    if execute:
        return require_full_plan_for_execute(path)
    return path if path and path.is_file() else None


def plan_batch(args: argparse.Namespace) -> Dict[str, Any]:
    config = apply_cli_overrides(
        load_config(args.config),
        execution_plan=args.execution_plan,
        checkpoint_dir=args.checkpoint_dir,
        run_id=args.run_id,
        oss_prefix=args.oss_prefix,
        resume=args.resume,
        allow_reshard=args.allow_reshard,
        topology=args.topology,
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
    plan_report: Dict[str, Any] = {
        "run_id": run_id,
        "execute": execute,
        "execution_plan": str(execution_plan_path) if execution_plan_path else None,
        "full_plan_search_result": "found" if execution_plan_path else "missing_frozen_full_execution_plan",
        "topology": topology_payload(topology),
        "runtime_preflight": runtime_report,
        "worker_commands": [],
        "shards": {},
    }
    if execution_plan_path:
        track_bank = Path(args.track_bank) if args.track_bank else None
        path_mapping = Path(args.path_mapping) if args.path_mapping else None
        plan = load_execution_plan(execution_plan_path=execution_plan_path, track_bank_path=track_bank, path_mapping_path=path_mapping)
        plan_report["execution_plan_validation"] = plan.validation
        if not plan.validation["valid"] and config["execution"].get("strict", True):
            raise DataAError(f"full execution plan validation failed: {plan.validation['errors']}")
        case_ids = [case.case_id for case in plan.cases]
        if args.case_id:
            case_ids = [case_id for case_id in case_ids if case_id == args.case_id]
        if args.max_cases is not None:
            case_ids = case_ids[: args.max_cases]
        shards = shard_cases(case_ids, run_id, topology)
        plan_report["shards"] = {str(worker_id): ids for worker_id, ids in shards.items()}
        paths.coordinator_dir.mkdir(parents=True, exist_ok=True)
        for group in topology.groups:
            shard_payload = {
                "run_id": run_id,
                "worker_id": group.worker_id,
                "topology": topology_payload(topology),
                "cases": [
                    {
                        "case_id": case_id,
                        "vace_job": {
                            "case_id": case_id,
                            "source_clip": str(paths.attempt_dir(group.worker_id, case_id) / "source_clip.mp4"),
                            "target_mask_gen_video": str(paths.attempt_dir(group.worker_id, case_id) / "target_mask_gen.mp4"),
                            "model_prompt": "",
                            "output_path": str(paths.attempt_dir(group.worker_id, case_id) / "generated_raw.mp4"),
                            "donor_reference": None,
                            "frame_count": 81,
                            "size": "720p",
                            "seed": int(config["vace"].get("seed", 20260629)),
                        },
                    }
                    for case_id in shards[group.worker_id]
                ],
            }
            shard_path = paths.coordinator_dir / f"worker_{group.worker_id:02d}_shard.json"
            write_json(shard_path, shard_payload)
            plan_report["worker_commands"].append(
                build_worker_command(
                    group=group,
                    config_path=args.config,
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
    for command in commands:
        env = None
        if command.get("env"):
            import os

            env = os.environ.copy()
            env.update(command["env"])
        procs.append(subprocess.Popen(command["argv"], env=env))
    exit_codes = [proc.wait() for proc in procs]
    return 0 if all(code == 0 for code in exit_codes) else 1


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
    parser.add_argument("--topology", choices=["4x4", "2x8"], default=None)
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
