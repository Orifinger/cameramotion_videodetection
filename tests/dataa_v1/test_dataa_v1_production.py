from __future__ import annotations

from pathlib import Path
from argparse import Namespace
import shutil

import numpy as np
import pytest

from scripts.dataa_v1.common import read_json, write_json
from scripts.dataa_v1.config import DEFAULT_CONFIG, apply_cli_overrides
from scripts.dataa_v1.execution_plan import load_execution_plan, validate_execution_cases
from scripts.dataa_v1.gpu_telemetry import aggregate_vram_ratio, parse_nvidia_smi_csv, summarize_telemetry
from scripts.dataa_v1.mask_video import write_mask_video_ffmpeg
from scripts.dataa_v1.oss_sync import mark_ready_to_upload, should_trigger_upload, upload_case_bundle
from scripts.dataa_v1.qa import batch_summary, compare_video_metadata
from scripts.dataa_v1.run_state import RunPaths, RunState
from scripts.dataa_v1.run_vace14b_batch import _resolve_project_path, plan_batch
from scripts.dataa_v1.topology import build_topology, shard_cases, stable_worker_id
from scripts.dataa_v1.worker_commands import build_worker_command


def _full_plan(path: Path, *, duplicate_target: bool = False) -> Path:
    target_b = "target_video_a" if duplicate_target else "target_video_b"
    payload = {
        "cases": [
            {
                "case_id": "case_a",
                "operation": "object_swap",
                "generator_route": "vace14b_masktrack_reference_swap",
                "target": {"track_id": "ta", "video_id": "target_video_a", "mask_tube_path": "/m/ta.npz"},
                "donor": {"track_id": "da", "video_id": "donor_video_a", "mask_tube_path": "/m/da.npz"},
                "sampling_meta": {"frozen": True},
            },
            {
                "case_id": "case_b",
                "operation": "object_attribute_edit",
                "generator_route": "vace14b_masktrack_text_edit",
                "target": {"track_id": "tb", "video_id": target_b, "mask_tube_path": "/m/tb.npz"},
                "sampling_meta": {"frozen": True},
            },
        ]
    }
    write_json(path, payload)
    return path


def test_full_execution_plan_validation_and_target_uniqueness(tmp_path: Path) -> None:
    plan = load_execution_plan(execution_plan_path=_full_plan(tmp_path / "plan.json"), track_bank_path=None, path_mapping_path=None)
    assert plan.validation["valid"]
    assert plan.validation["case_count"] == 2

    bad = load_execution_plan(execution_plan_path=_full_plan(tmp_path / "bad.json", duplicate_target=True), track_bank_path=None, path_mapping_path=None)
    assert not bad.validation["valid"]
    assert any(error.startswith("target_video_reused") for error in bad.validation["errors"])


def test_donor_reuse_limit_validation(tmp_path: Path) -> None:
    plan = load_execution_plan(execution_plan_path=_full_plan(tmp_path / "plan.json"), track_bank_path=None, path_mapping_path=None)
    validation = validate_execution_cases(plan.cases, donor_reuse_limit=0)
    assert any(error.startswith("donor_reuse_limit_exceeded") for error in validation["errors"])


def test_default_topology_4x4_and_fallback_2x8() -> None:
    topology = build_topology(DEFAULT_CONFIG["gpu"], available_gpu_count=16)
    assert topology.name == "4x4"
    assert len(topology.groups) == 4
    assert topology.groups[0].cuda_visible_devices == (0, 1, 2, 3)
    assert topology.groups[0].nproc_per_node == 4
    assert topology.groups[0].ulysses_size == 4

    cfg = apply_cli_overrides(DEFAULT_CONFIG, topology="2x8")
    fallback = build_topology(cfg["gpu"], available_gpu_count=16)
    assert fallback.name == "2x8"
    assert fallback.is_fallback
    assert fallback.groups[0].nproc_per_node == 8


def test_topology_workers_per_gpu_reuses_physical_devices(tmp_path: Path) -> None:
    cfg = dict(DEFAULT_CONFIG["gpu"])
    cfg.update({"worker_groups": 16, "gpus_per_worker": 1, "workers_per_gpu": 2})
    topology = build_topology(cfg, available_gpu_count=16)
    assert topology.name == "16x1w2"
    assert topology.total_gpus == 16
    assert topology.worker_groups == 32
    assert len(topology.groups) == 32
    assert topology.groups[0].cuda_visible_devices == (0,)
    assert topology.groups[1].cuda_visible_devices == (0,)
    assert topology.groups[2].cuda_visible_devices == (1,)

    command = build_worker_command(
        group=topology.groups[0],
        config_path=tmp_path / "config.yaml",
        shard_path=tmp_path / "shard.json",
        run_id="run_1",
    )
    assert "--nproc_per_node=1" in command["argv"]
    assert command["vace_distributed_flags"] == {"dit_fsdp": False, "t5_fsdp": False, "ulysses_size": 1, "ring_size": 1}


def test_topology_batch_size_alias_sets_workers_per_gpu() -> None:
    cfg = dict(DEFAULT_CONFIG["gpu"])
    cfg.update({"worker_groups": 16, "gpus_per_worker": 1, "batch_size": 2})
    topology = build_topology(cfg, available_gpu_count=16)
    assert topology.name == "16x1w2"
    assert topology.worker_groups == 32


def test_project_relative_execution_plan_resolves_under_repo_res() -> None:
    resolved = _resolve_project_path("res/dataA_v1/plans/frozen_full_vace_execution_plan.json")
    assert resolved is not None
    assert resolved.is_absolute()
    assert resolved.as_posix().endswith("/res/dataA_v1/plans/frozen_full_vace_execution_plan.json")


def test_deterministic_sharding_is_stable() -> None:
    topology = build_topology(DEFAULT_CONFIG["gpu"])
    cases = [f"case_{i:04d}" for i in range(16)]
    first = shard_cases(cases, "run_1", topology)
    second = shard_cases(reversed(cases), "run_1", topology)
    assert first == second
    assert set().union(*[set(v) for v in first.values()]) == set(cases)
    assert stable_worker_id("case_0001", "run_1", topology) in first


def test_worker_command_contract_uses_4x4_flags(tmp_path: Path) -> None:
    topology = build_topology(DEFAULT_CONFIG["gpu"])
    command = build_worker_command(
        group=topology.groups[0],
        config_path=tmp_path / "config.yaml",
        shard_path=tmp_path / "shard.json",
        run_id="run_1",
    )
    assert "--nproc_per_node=4" in command["argv"]
    assert command["vace_distributed_flags"] == {"dit_fsdp": True, "t5_fsdp": True, "ulysses_size": 4, "ring_size": 1}
    assert command["contract"]["model_loads_per_worker_group"] == 1
    assert command["contract"]["per_case_torchrun_forbidden"]


def test_upload_receipt_and_failure_does_not_delete_local_artifact(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    artifact = attempt / "case_manifest.json"
    artifact.write_text("{}\n", encoding="utf-8")
    mark_ready_to_upload(attempt)
    receipt = upload_case_bundle(
        attempt_dir=attempt,
        oss_dest="oss://bucket/run/worker_00/attempts/case_a",
        upload_command="missing_ossutil64",
        run_id="run",
        case_id="case_a",
        worker_id=0,
        execute=False,
    )
    assert receipt["status"] == "upload_planned"
    assert artifact.is_file()
    assert receipt["inventory"]["file_count"] >= 2


def test_upload_interval_by_case_count_and_time() -> None:
    assert should_trigger_upload(completed_since_upload=8, minutes_since_upload=0, every_completed_cases=8, every_minutes=30)
    assert should_trigger_upload(completed_since_upload=0, minutes_since_upload=31, every_completed_cases=8, every_minutes=30)
    assert not should_trigger_upload(completed_since_upload=7, minutes_since_upload=29, every_completed_cases=8, every_minutes=30)


def test_telemetry_aggregate_vram_ratio_and_floor() -> None:
    samples = parse_nvidia_smi_csv("0, 48000, 96000, 75, 60\n1, 24000, 96000, 50, 55\n")
    assert aggregate_vram_ratio(samples) == pytest.approx(72000 / 192000)
    summary = summarize_telemetry(samples, min_aggregate_vram_ratio=0.50)
    assert not summary["vram_floor_met"]


def test_run_state_resume_skips_uploaded_terminal_case(tmp_path: Path) -> None:
    paths = RunPaths.from_root(tmp_path, "run")
    state = RunState(paths, run_id="run", topology={"name": "4x4"})
    state.append_status("case_a", "uploaded_verified", worker_id=0, detail={"upload_receipt": "receipt"})
    assert state.should_skip_case("case_a")
    assert not state.should_skip_case("case_b")


def test_run_state_recovers_from_invalid_json_using_status_log(tmp_path: Path) -> None:
    paths = RunPaths.from_root(tmp_path, "run")
    state = RunState(paths, run_id="run", topology={"name": "4x4"})
    state.append_status("case_a", "uploaded_verified", worker_id=0, detail={"upload_receipt": "receipt"})
    paths.run_state_path.write_text("", encoding="utf-8")

    recovered = state.load()

    assert recovered["cases"]["case_a"]["status"] == "uploaded_verified"
    assert recovered["invalid_run_state_backup"].endswith(".json")
    assert state.should_skip_case("case_a")


def test_batch_plan_lineage_accumulates_plan_index_for_same_run_id(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                f"  tmp_root: {tmp_path.as_posix()}/runs",
                "gpu:",
                "  worker_groups: 1",
                "  gpus_per_worker: 1",
                "  fallback_topologies: []",
                "upload:",
                "  enabled: false",
                "vace:",
                f"  repo_dir: {tmp_path.as_posix()}",
                f"  checkpoint_dir: {tmp_path.as_posix()}",
                "  torchrun_bin: torchrun",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plan_a = tmp_path / "plan_a.json"
    plan_b = tmp_path / "plan_b.json"
    write_json(
        plan_a,
        {
            "schema_version": "test_plan_a",
            "cases": [
                {
                    "case_id": "case_a",
                    "operation": "object_attribute_edit",
                    "generator_route": "vace14b_masktrack_text_edit",
                    "target": {"track_id": "ta", "video_id": "video_a", "mask_tube_path": "/m/ta.npz"},
                    "sampling_meta": {"frozen": True},
                }
            ],
        },
    )
    write_json(
        plan_b,
        {
            "schema_version": "test_plan_b",
            "cases": [
                {
                    "case_id": "case_b",
                    "operation": "person_appearance_swap",
                    "generator_route": "vace14b_masktrack_reference_swap",
                    "target": {"track_id": "tb", "video_id": "video_b", "mask_tube_path": "/m/tb.npz"},
                    "donor": {"track_id": "db", "video_id": "donor_b", "mask_tube_path": "/m/db.npz"},
                    "sampling_meta": {
                        "frozen": True,
                        "continuation": {"strategy": "person_preferred"},
                        "mask_policy": {"variant_type": "dilated", "person_bbox_disabled": True},
                    },
                }
            ],
        },
    )

    base_args = {
        "track_bank": None,
        "path_mapping": None,
        "config": config_path,
        "checkpoint_dir": None,
        "run_id": "same_run",
        "oss_prefix": None,
        "execute": False,
        "allow_reshard": True,
        "case_id": None,
        "max_cases": None,
        "topology": None,
        "launch_workers": False,
    }
    plan_batch(Namespace(**base_args, execution_plan=plan_a, resume=False))
    plan_batch(Namespace(**base_args, execution_plan=plan_b, resume=True))

    lineage = read_json(tmp_path / "runs" / "same_run" / "coordinator" / "plan_lineage.json")
    assert lineage["run_id"] == "same_run"
    assert len(lineage["plans"]) == 2
    assert lineage["case_to_plan"]["case_a"]["execution_plan"].endswith("plan_a.json")
    assert lineage["case_to_plan"]["case_b"]["execution_plan"].endswith("plan_b.json")
    assert lineage["case_to_plan"]["case_b"]["is_continuation_case"] is True


def test_qa_summary_counts() -> None:
    meta = compare_video_metadata(
        {"fps": 16, "frame_count": 81, "height": 720, "width": 1280},
        {"fps": 16, "frame_count": 81, "height": 720, "width": 1280},
    )
    assert meta["compatible"]
    summary = batch_summary([{"status": "accepted"}, {"status": "rejected_generation_failure"}])
    assert summary["accepted_count"] == 1
    assert summary["rejected_count_by_reason"]["rejected_generation_failure"] == 1


def test_real_media_mask_video_requires_ffmpeg_or_skips(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg unavailable; real-media mask video test skipped explicitly")
    masks = np.zeros((3, 16, 16), dtype=np.uint8)
    masks[:, 4:8, 4:8] = 1
    report = write_mask_video_ffmpeg(tmp_path / "mask.mp4", masks, fps=16, ffmpeg_bin=ffmpeg)
    assert report["backend"] == "ffmpeg_libx264rgb_lossless"
