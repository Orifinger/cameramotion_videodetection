from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.dataa_v1.build_subject_first_execution_plan import build_subject_first_plan
from scripts.dataa_v1.common import write_json
from scripts.dataa_v1.execution_plan import load_execution_plan
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
    )
    assert summary["validation"]["valid"]
    plan = load_execution_plan(execution_plan_path=out_plan, track_bank_path=None, path_mapping_path=None)
    assert plan.validation["valid"]
    assert plan.cases[0].sampling_meta["target_selection"]["selection_role"] in {"fallback_primary", "primary_subject"}


def test_subject_first_dry_run_writes_audit_not_plan(tmp_path: Path) -> None:
    track = _track(tmp_path, video_id="v1", track_id="target", box=(35, 35, 30, 30))
    track_bank = tmp_path / "tracks.json"
    out_plan = tmp_path / "plan.json"
    write_json(track_bank, {"tracks": [track]})
    summary = build_subject_first_plan(
        track_bank=track_bank,
        selection_config=None,
        base_plan=None,
        out_catalog=tmp_path / "catalog.json",
        out_audit_json=tmp_path / "audit.json",
        out_audit_csv=tmp_path / "audit.csv",
        out_plan=out_plan,
        dry_run=True,
    )
    assert summary["summary"]["videos_with_primary"] == 1
    assert (tmp_path / "audit.json").is_file()
    assert not out_plan.exists()
