from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.dataa_v1.build_subject_first_execution_plan import build_subject_first_plan
from scripts.dataa_v1.build_continuation_execution_plan import build_continuation_plan
from scripts.dataa_v1.build_subject_first_execution_plan import _is_person_track
from scripts.dataa_v1.common import write_json
from scripts.dataa_v1.execution_plan import load_execution_plan
from scripts.dataa_v1.schema import TrackRef
from scripts.dataa_v1.subject_selection import (
    evaluate_tracks,
    load_selection_config,
    select_subjects_by_video,
)


def _mask(path: Path, frames: np.ndarray, box: tuple[int, int, int, int], *, h: int = 100, w: int = 100) -> str:
    masks = np.zeros((len(frames), h, w), dtype=np.uint8)
    x, y, bw, bh = box
    masks[:, y : y + bh, x : x + bw] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frame_indices=frames.astype(np.int32), masks=masks)
    return str(path)


def _track(
    tmp_path: Path,
    *,
    video_id: str,
    track_id: str,
    box: tuple[int, int, int, int],
    frames: np.ndarray | None = None,
    quality: float = 0.9,
    candidate_class: str = "bounded_object",
) -> dict[str, object]:
    frames = np.arange(0, 60, dtype=np.int32) if frames is None else frames
    return {
        "track_id": track_id,
        "candidate_id": f"{track_id}_candidate",
        "video_id": video_id,
        "video_path": f"/synthetic/{video_id}.mp4",
        "source_fps": 30.0,
        "mask_tube_path": _mask(tmp_path / f"{track_id}.npz", frames, box),
        "candidate_class": candidate_class,
        "canonical_concept": candidate_class,
        "track_quality_score": quality,
    }


def _config(**overrides: object) -> dict[str, object]:
    config = load_selection_config(None)
    config.update(overrides)
    return config


def _select(records: list[dict[str, object]], config: dict[str, object] | None = None):
    cfg = config or load_selection_config(None)
    evaluated = evaluate_tracks(records, cfg)
    return evaluated, select_subjects_by_video(evaluated, cfg)


def test_small_centered_object_fails_universal_gate(tmp_path: Path) -> None:
    records = [_track(tmp_path, video_id="v1", track_id="small", box=(48, 48, 3, 3))]
    evaluated, selections = _select(records)
    assert selections["v1"].selected is None
    assert evaluated[0].selection_status == "ineligible_small_or_weak"
    assert "median_mask_area_ratio_below_universal_threshold" in evaluated[0].rejection_tags


def test_primary_secondary_sampling_is_seeded_and_reproducible(tmp_path: Path) -> None:
    records = [
        _track(tmp_path, video_id="v1", track_id="primary", box=(35, 35, 30, 30), quality=0.95),
        _track(tmp_path, video_id="v1", track_id="secondary", box=(10, 10, 22, 22), quality=0.8),
    ]
    cfg = _config(random_seed=7, primary_probability=0.0)
    _evaluated_a, selected_a = _select(records, cfg)
    _evaluated_b, selected_b = _select(records, cfg)
    assert selected_a["v1"].selected is not None
    assert selected_b["v1"].selected is not None
    assert selected_a["v1"].selected.track_id == selected_b["v1"].selected.track_id
    assert selected_a["v1"].selected.selection_role == "eligible_secondary"


def test_parallel_evaluation_matches_single_process(tmp_path: Path) -> None:
    records = [
        _track(tmp_path, video_id="v1", track_id="primary", box=(35, 35, 30, 30), quality=0.95),
        _track(tmp_path, video_id="v1", track_id="secondary", box=(10, 10, 22, 22), quality=0.8),
        _track(tmp_path, video_id="v2", track_id="other", box=(30, 30, 28, 28), quality=0.9),
    ]
    cfg = load_selection_config(None)
    single = evaluate_tracks(records, cfg, num_workers=1)
    parallel = evaluate_tracks(records, cfg, num_workers=2)
    assert [item.track_id for item in parallel] == [item.track_id for item in single]
    assert [round(item.subject_score, 8) for item in parallel] == [round(item.subject_score, 8) for item in single]
    assert select_subjects_by_video(parallel, cfg)["v1"].selected.track_id == select_subjects_by_video(single, cfg)["v1"].selected.track_id


def test_no_secondary_uses_primary(tmp_path: Path) -> None:
    records = [_track(tmp_path, video_id="v1", track_id="primary", box=(35, 35, 30, 30))]
    _evaluated, selections = _select(records)
    assert selections["v1"].selected is not None
    assert selections["v1"].selected.track_id == "primary"
    assert selections["v1"].selected.selection_role == "fallback_primary"


def test_no_primary_produces_no_case(tmp_path: Path) -> None:
    records = [_track(tmp_path, video_id="v1", track_id="tiny", box=(50, 50, 2, 2))]
    _evaluated, selections = _select(records)
    assert selections["v1"].primary is None
    assert selections["v1"].selected is None


def test_subject_selection_accepts_verified_mask_alias(tmp_path: Path) -> None:
    track = _track(tmp_path, video_id="v1", track_id="target", box=(35, 35, 30, 30))
    alias_path = track.pop("mask_tube_path")
    track["mask_npz_path"] = alias_path
    evaluated, selections = _select([track])
    assert selections["v1"].selected is not None
    assert evaluated[0].record["mask_tube_path"] == alias_path
    assert evaluated[0].record["subject_selection_mask_path_key"] == "mask_npz_path"


def test_invalid_mask_tube_records_detailed_reason(tmp_path: Path) -> None:
    bad_mask = tmp_path / "bad.npz"
    np.savez_compressed(bad_mask, wrong=np.zeros((1,), dtype=np.uint8))
    track = _track(tmp_path, video_id="v1", track_id="target", box=(35, 35, 30, 30))
    track["mask_tube_path"] = str(bad_mask)
    evaluated, selections = _select([track])
    assert selections["v1"].selected is None
    assert evaluated[0].rejection_tags == ["invalid_mask_tube:npz_missing_frame_indices_or_masks"]
    assert "npz must contain frame_indices and masks" in evaluated[0].rejection_reasons[0]


def test_large_edge_low_quality_does_not_crush_stable_subject(tmp_path: Path) -> None:
    records = [
        _track(tmp_path, video_id="v1", track_id="edge", box=(0, 0, 45, 90), quality=0.05),
        _track(tmp_path, video_id="v1", track_id="stable", box=(35, 30, 30, 35), quality=0.95),
    ]
    _evaluated, selections = _select(records)
    assert selections["v1"].primary is not None
    assert selections["v1"].primary.track_id == "stable"


def test_high_iou_nested_track_not_secondary(tmp_path: Path) -> None:
    records = [
        _track(tmp_path, video_id="v1", track_id="outer", box=(25, 25, 40, 40), quality=0.95),
        _track(tmp_path, video_id="v1", track_id="inner", box=(28, 28, 34, 34), quality=0.9),
    ]
    evaluated, selections = _select(records)
    assert selections["v1"].primary is not None
    duplicate = next(item for item in evaluated if item.track_id != selections["v1"].primary.track_id)
    assert duplicate.selection_status == "duplicate_secondary_high_iou"
    assert "duplicate_with_primary_subject" in duplicate.rejection_tags


def test_subject_first_plan_loads_and_validates(tmp_path: Path) -> None:
    target = _track(tmp_path, video_id="target_video", track_id="target", box=(35, 35, 30, 30))
    donor = _track(tmp_path, video_id="donor_video", track_id="donor", box=(20, 20, 25, 25))
    track_bank = tmp_path / "tracks.json"
    base_plan = tmp_path / "base_plan.json"
    out_plan = tmp_path / "frozen_subject_first.json"
    out_coverage = tmp_path / "coverage.json"
    out_reserve = tmp_path / "reserve.json"
    write_json(track_bank, {"tracks": [target, donor]})
    write_json(
        base_plan,
        {
            "cases": [
                {
                    "case_id": "case_0001",
                    "operation": "object_swap",
                    "generator_route": "vace14b_masktrack_reference_swap",
                    "target_track_id": "target",
                    "donor_track_id": "donor",
                }
            ]
        },
    )
    summary = build_subject_first_plan(
        track_bank=track_bank,
        selection_config=None,
        base_plan=base_plan,
        out_catalog=tmp_path / "catalog.json",
        out_audit_json=tmp_path / "audit.json",
        out_audit_csv=tmp_path / "audit.csv",
        out_plan=out_plan,
        out_coverage_plan=out_coverage,
        out_reserve_plan=out_reserve,
    )
    assert summary["validation"]["valid"]
    plan = load_execution_plan(execution_plan_path=out_plan, track_bank_path=None, path_mapping_path=None)
    assert plan.validation["valid"]
    assert plan.cases[0].sampling_meta["target_selection"]["selection_role"] in {"fallback_primary", "primary_subject"}
    assert plan.cases[0].sampling_meta["mask_policy"]["variant_type"] in {"sam3_shape", "dilated", "expanded_bbox", "closing", "erode_then_dilate"}
    assert plan.cases[0].sampling_meta["vace_model_plan"]["model_name"] == "vace-14B"


def test_subject_first_writes_coverage_and_reserve_plans(tmp_path: Path) -> None:
    clean = _track(tmp_path, video_id="clean_video", track_id="clean", box=(35, 35, 30, 30))
    relaxed = _track(tmp_path, video_id="relaxed_video", track_id="relaxed", box=(49, 49, 4, 4))
    repair_target = _track(tmp_path, video_id="repair_video", track_id="repair_target", box=(35, 35, 30, 30), candidate_class="vehicle")
    same_video_donor = _track(tmp_path, video_id="repair_video", track_id="same_video_donor", box=(20, 20, 25, 25), candidate_class="vehicle")
    repair_donor = _track(tmp_path, video_id="donor_video", track_id="repair_donor", box=(20, 20, 25, 25), candidate_class="vehicle")
    coverage_only = _track(tmp_path, video_id="coverage_only_video", track_id="coverage_only", box=(35, 35, 30, 30))
    track_bank = tmp_path / "tracks.json"
    base_plan = tmp_path / "base_plan.json"
    out_plan = tmp_path / "clean.json"
    out_coverage = tmp_path / "coverage.json"
    out_reserve = tmp_path / "reserve.json"
    write_json(track_bank, {"tracks": [clean, relaxed, repair_target, same_video_donor, repair_donor, coverage_only]})
    write_json(
        base_plan,
        {
            "cases": [
                {
                    "case_id": "case_clean",
                    "operation": "object_attribute_edit",
                    "generator_route": "vace14b_masktrack_text_edit",
                    "target_track_id": "clean",
                },
                {
                    "case_id": "case_relaxed",
                    "operation": "object_attribute_edit",
                    "generator_route": "vace14b_masktrack_text_edit",
                    "target_track_id": "relaxed",
                },
                {
                    "case_id": "case_repair",
                    "operation": "object_swap",
                    "generator_route": "vace14b_masktrack_reference_swap",
                    "target_track_id": "repair_target",
                    "donor_track_id": "same_video_donor",
                },
            ]
        },
    )
    summary = build_subject_first_plan(
        track_bank=track_bank,
        selection_config=None,
        base_plan=base_plan,
        out_catalog=tmp_path / "catalog.json",
        out_audit_json=tmp_path / "audit.json",
        out_audit_csv=tmp_path / "audit.csv",
        out_plan=out_plan,
        out_coverage_plan=out_coverage,
        out_reserve_plan=out_reserve,
        num_workers=1,
    )
    assert summary["validation"]["valid"]
    assert summary["coverage_validation"]["valid"]
    assert summary["reserve_validation"]["valid"]

    clean_payload = load_execution_plan(execution_plan_path=out_plan, track_bank_path=None, path_mapping_path=None)
    coverage_payload = load_execution_plan(execution_plan_path=out_coverage, track_bank_path=None, path_mapping_path=None)
    reserve_payload = load_execution_plan(execution_plan_path=out_reserve, track_bank_path=None, path_mapping_path=None)
    assert {case.case_id for case in clean_payload.cases} == {"case_clean", "case_relaxed", "case_repair"}
    assert {case.case_id for case in reserve_payload.cases} == {case.case_id for case in coverage_payload.cases} - {case.case_id for case in clean_payload.cases}
    repaired = next(case for case in coverage_payload.cases if case.case_id == "case_repair")
    assert repaired.donor is not None
    assert repaired.donor.video_id == "donor_video"
    relaxed_case = next(case for case in coverage_payload.cases if case.case_id == "case_relaxed")
    assert relaxed_case.sampling_meta["target_selection"]["quality_tier"] == "area_gate_fallback_largest"
    assert relaxed_case.sampling_meta["mask_policy"]["variant_type"] in {"sam3_shape", "dilated", "expanded_bbox", "closing", "erode_then_dilate"}
    assert any(case.case_id.startswith("dataA_v1_subject_first_coverage_") for case in coverage_payload.cases)


def test_operation_fallback_does_not_keep_object_swap_on_person(tmp_path: Path) -> None:
    person = _track(tmp_path, video_id="v1", track_id="person", box=(20, 20, 40, 50), candidate_class="human")
    donor = _track(tmp_path, video_id="donor", track_id="donor_person", box=(20, 20, 40, 50), candidate_class="human")
    track_bank = tmp_path / "tracks.json"
    base_plan = tmp_path / "base_plan.json"
    out_plan = tmp_path / "plan.json"
    write_json(track_bank, {"tracks": [person, donor]})
    write_json(
        base_plan,
        {
            "cases": [
                {
                    "case_id": "case_repair_person",
                    "operation": "object_swap",
                    "generator_route": "vace14b_masktrack_reference_swap",
                    "target_track_id": "person",
                    "donor_track_id": "donor_person",
                }
            ]
        },
    )
    summary = build_subject_first_plan(
        track_bank=track_bank,
        selection_config=None,
        base_plan=base_plan,
        out_catalog=tmp_path / "catalog.json",
        out_audit_json=tmp_path / "audit.json",
        out_audit_csv=tmp_path / "audit.csv",
        out_plan=out_plan,
        out_coverage_plan=tmp_path / "coverage.json",
        out_reserve_plan=tmp_path / "reserve.json",
        num_workers=1,
    )
    assert summary["validation"]["valid"]
    plan = load_execution_plan(execution_plan_path=out_plan, track_bank_path=None, path_mapping_path=None)
    case = plan.cases[0]
    assert case.operation == "person_appearance_swap"
    assert case.generator_route == "vace14b_masktrack_reference_swap"
    assert case.sampling_meta["operation_repair"]["original_operation"] == "object_swap"
    assert case.sampling_meta["mask_policy"]["person_bbox_disabled"] is True
    assert case.sampling_meta["mask_policy"]["variant_type"] != "expanded_bbox"


def test_continuation_skips_completed_and_prefers_person_swap(tmp_path: Path) -> None:
    completed_object = _track(tmp_path, video_id="done_video", track_id="done_object", box=(35, 35, 30, 30), candidate_class="bounded_object")
    remaining_object = _track(tmp_path, video_id="remain_video", track_id="remain_object", box=(45, 45, 8, 8), candidate_class="bounded_object")
    remaining_person = _track(tmp_path, video_id="remain_video", track_id="remain_person", box=(20, 20, 40, 50), candidate_class="human")
    donor_person = _track(tmp_path, video_id="donor_video", track_id="donor_person", box=(20, 20, 40, 50), candidate_class="human")
    track_bank = tmp_path / "tracks.json"
    base_plan = tmp_path / "base_plan.json"
    run_root = tmp_path / "run"
    attempt = run_root / "worker_00" / "attempts" / "case_done"
    attempt.mkdir(parents=True)
    write_json(attempt / "case_manifest.json", {"case_id": "case_done", "target": {"video_id": "done_video"}})
    write_json(attempt / "generation_result.json", {"status": "generated", "full_video": {"status": "ok"}})
    (attempt / "full_real.mp4").write_bytes(b"x")
    (attempt / "full_fake.mp4").write_bytes(b"x")
    write_json(track_bank, {"tracks": [completed_object, remaining_object, remaining_person, donor_person]})
    write_json(
        base_plan,
        {
            "cases": [
                {
                    "case_id": "case_done",
                    "operation": "object_swap",
                    "generator_route": "vace14b_masktrack_reference_swap",
                    "target_track_id": "done_object",
                    "donor_track_id": "donor_person",
                },
                {
                    "case_id": "case_remaining",
                    "operation": "object_swap",
                    "generator_route": "vace14b_masktrack_reference_swap",
                    "target_track_id": "remain_object",
                    "donor_track_id": "donor_person",
                },
            ]
        },
    )

    summary = build_continuation_plan(
        track_bank=track_bank,
        base_plan=base_plan,
        run_roots=[run_root],
        selection_config=None,
        out_plan=tmp_path / "continuation.json",
        out_vace14b_plan=tmp_path / "continuation_14b.json",
        out_vace13b_plan=tmp_path / "continuation_13b.json",
        out_audit=tmp_path / "continuation_audit.json",
        num_workers=1,
    )
    assert summary["validation"]["valid"]
    plan = load_execution_plan(execution_plan_path=tmp_path / "continuation.json", track_bank_path=None, path_mapping_path=None)
    assert [case.case_id for case in plan.cases] == ["case_remaining"]
    case = plan.cases[0]
    assert case.operation == "person_appearance_swap"
    assert case.target.track_id == "remain_person"
    assert case.sampling_meta["continuation"]["strategy"] == "person_preferred"
    assert case.sampling_meta["mask_policy"]["person_bbox_disabled"] is True
    assert case.sampling_meta["mask_policy"]["variant_type"] != "expanded_bbox"


def test_track_label_helpers_accept_trackref() -> None:
    track = TrackRef(
        role="donor",
        track_id="d1",
        video_id="v2",
        video_path=None,
        mask_tube_path="/m/d1.npz",
        candidate_class="human",
        raw={"candidate_class": "human"},
    )
    assert _is_person_track(track)


def test_subject_first_dry_run_writes_audit_not_plan(tmp_path: Path) -> None:
    track = _track(tmp_path, video_id="v1", track_id="target", box=(35, 35, 30, 30))
    track_bank = tmp_path / "tracks.json"
    out_plan = tmp_path / "plan.json"
    out_coverage = tmp_path / "coverage.json"
    out_reserve = tmp_path / "reserve.json"
    write_json(track_bank, {"tracks": [track]})
    summary = build_subject_first_plan(
        track_bank=track_bank,
        selection_config=None,
        base_plan=None,
        out_catalog=tmp_path / "catalog.json",
        out_audit_json=tmp_path / "audit.json",
        out_audit_csv=tmp_path / "audit.csv",
        out_plan=out_plan,
        out_coverage_plan=out_coverage,
        out_reserve_plan=out_reserve,
        dry_run=True,
    )
    assert summary["summary"]["videos_with_primary"] == 1
    assert (tmp_path / "audit.json").is_file()
    assert not out_plan.exists()
    assert not out_coverage.exists()
    assert not out_reserve.exists()
