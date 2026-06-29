from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.dataa_v1.audit_vace_stage1 import build_report
from scripts.dataa_v1.clip_selection import VaceProfile, select_clip
from scripts.dataa_v1.donor_reference import choose_donor_frame
from scripts.dataa_v1.mask_io import align_masks_to_canonical, load_mask_tube
from scripts.dataa_v1.mask_processing import apply_effective_mask_policy, process_masks
from scripts.dataa_v1.mask_video import validate_mask_video_roundtrip, write_mask_video
from scripts.dataa_v1.package_vace_case import package_case
from scripts.dataa_v1.path_resolver import PathResolver
from scripts.dataa_v1.common import write_json


def _tube(path: Path, frames: np.ndarray, *, h: int = 24, w: int = 32, offset: int = 3) -> Path:
    masks = np.zeros((len(frames), h, w), dtype=np.uint8)
    for i in range(len(frames)):
        x = offset + (i % 4)
        masks[i, 5:15, x : x + 8] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frame_indices=frames.astype(np.int32), masks=masks)
    return path


def _plan_and_tracks(tmp_path: Path) -> tuple[Path, Path]:
    target_mask = _tube(tmp_path / "persistent" / "target.npz", np.arange(0, 180, dtype=np.int32))
    donor_mask = _tube(tmp_path / "persistent" / "donor.npz", np.arange(10, 80, dtype=np.int32), offset=10)
    plan = {
        "cases": [
            {
                "case_id": "case_0001",
                "operation": "object_swap",
                "generator_route": "vace14b_masktrack_reference_swap",
                "target_track_id": "target_track",
                "donor_track_id": "donor_track",
                "sampling_meta": {
                    "mask_policy": {
                        "variant_type": "sam3_shape",
                        "seed": 1,
                        "dilation_radius_px": 8,
                        "bbox_expand_ratio": 1.15,
                        "person_bbox_disabled": False,
                    },
                    "vace_model_plan": {
                        "model_name": "vace-14B",
                        "size": "720p",
                        "profile": "production_720",
                        "offload_model": False,
                        "t5_cpu": False,
                    },
                },
            }
        ]
    }
    tracks = {
        "tracks": [
            {
                "track_id": "target_track",
                "candidate_id": "target_candidate",
                "video_id": "video_target",
                "video_path": "/data/target.mp4",
                "mask_tube_path": str(target_mask),
                "canonical_concept": "red cup",
                "display_phrase": "a red cup",
                "content_domain": "indoor tabletop",
                "style_domain": "realistic",
            },
            {
                "track_id": "donor_track",
                "candidate_id": "donor_candidate",
                "video_id": "video_donor",
                "video_path": "/data/donor.mp4",
                "mask_tube_path": str(donor_mask),
                "canonical_concept": "blue bottle",
                "display_phrase": "a blue bottle",
            },
        ]
    }
    plan_path = tmp_path / "plan.json"
    tracks_path = tmp_path / "tracks.json"
    write_json(plan_path, plan)
    write_json(tracks_path, tracks)
    return plan_path, tracks_path


def test_p1_preflight_strict_npz_and_track_lookup(tmp_path: Path) -> None:
    plan_path, tracks_path = _plan_and_tracks(tmp_path)
    report = build_report(plan_path, tracks_path, None)
    case = report["cases"][0]
    assert case["stage_status"] == "preflight_passed"
    assert case["target"]["track_id"] == "target_track"
    assert case["donor"]["track_id"] == "donor_track"
    assert case["target"]["mask_npz"]["frame_indices_dtype"] == "int32"
    assert case["target"]["mask_npz"]["dtype"] == "uint8"


def test_path_mapping_does_not_claim_tmp_persistence(tmp_path: Path) -> None:
    tmp_mask = _tube(tmp_path / "tmp" / "track.npz", np.arange(3, dtype=np.int32))
    mapping = {
        "volatile_prefixes": [str(tmp_path / "tmp")],
        "rules": [{"source_prefix": str(tmp_path / "tmp"), "persistent_prefix": "oss://bucket/masks"}],
    }
    resolved = PathResolver(mapping).resolve(str(tmp_mask))
    assert resolved.state == "readable_volatile"
    assert resolved.resolved_path == str(tmp_mask)


def test_clip_selection_and_mask_alignment(tmp_path: Path) -> None:
    tube = load_mask_tube(_tube(tmp_path / "tube.npz", np.arange(0, 180, dtype=np.int32)))
    clip = select_clip(tube, source_fps=30.0, profile=VaceProfile())
    assert clip.duration_seconds == 6
    assert clip.canonical_frame_count == 81
    assert clip.canonical_fps == 13.5
    aligned, meta = align_masks_to_canonical(tube, clip.canonical_to_source_frames)
    assert aligned.shape == (81, 24, 32)
    assert meta["frame_mapping"][0]["source_frame_index"] == 0


def test_gap_alignment_zero_fills_invisible_frames(tmp_path: Path) -> None:
    tube = load_mask_tube(_tube(tmp_path / "gap.npz", np.array([0, 1, 2, 10, 11, 12], dtype=np.int32)))
    clip = select_clip(tube, source_fps=6.0, profile=VaceProfile())
    aligned, meta = align_masks_to_canonical(tube, clip.canonical_to_source_frames)
    assert meta["zero_filled_gap_frame_count"] > 0
    assert any(aligned[index].sum() == 0 for index in meta["zero_filled_gap_frames"])


def test_mask_video_roundtrip_validation(tmp_path: Path) -> None:
    tube = load_mask_tube(_tube(tmp_path / "tube.npz", np.arange(0, 81, dtype=np.int32)))
    masks, _params = process_masks(tube.masks)
    video_path = tmp_path / "mask.mp4"
    write_mask_video(video_path, masks["M_gen"], fps=16, allow_synthetic_npz_fallback=True)
    report = validate_mask_video_roundtrip(video_path, masks["M_gen"], allow_synthetic_npz_fallback=True)
    assert report["status"] == "ok"
    assert report["thresholded_iou_min"] == 1.0
    assert report["backend"] in {"decoded_video", "synthetic_npz_sidecar"}


def test_effective_mask_policy_is_frozen_and_can_expand_bbox(tmp_path: Path) -> None:
    tube = load_mask_tube(_tube(tmp_path / "tube.npz", np.arange(0, 17, dtype=np.int32)))
    masks, _params = process_masks(tube.masks)
    effective, meta = apply_effective_mask_policy(
        masks,
        {
            "variant_type": "expanded_bbox",
            "seed": 7,
            "dilation_radius_px": 8,
            "bbox_expand_ratio": 1.25,
            "person_bbox_disabled": False,
        },
    )
    assert effective["M_gen"].shape == masks["M_gen"].shape
    assert meta["mask_policy"]["variant_type"] == "expanded_bbox"
    assert meta["effective_mask_area_stats"]["mean"] > meta["base_gen_area_stats"]["mean"]


def test_donor_reference_scoring(tmp_path: Path) -> None:
    tube = load_mask_tube(_tube(tmp_path / "donor.npz", np.arange(10, 20, dtype=np.int32), offset=8))
    choice = choose_donor_frame(tube)
    assert 10 <= choice.frame_index <= 19
    assert choice.bbox_xywh[2] > 0
    assert "area_score" in choice.components


def test_package_case_dry_run_manifest_complete(tmp_path: Path) -> None:
    plan_path, tracks_path = _plan_and_tracks(tmp_path)
    out_dir = tmp_path / "attempts"
    manifest = package_case(
        plan=plan_path,
        track_bank=tracks_path,
        case_id="case_0001",
        out_dir=out_dir,
        source_fps=30.0,
        synthetic_media=False,
    )
    assert manifest["stage_status"] == "planned"
    assert manifest["mask_video"]["roundtrip"]["status"] == "not_run_in_dry_run"
    assert "model_prompt" in manifest["prompt"]
    assert manifest["sampling_meta"]["mask_policy"]["variant_type"] == "sam3_shape"
    assert (out_dir / "case_0001" / "case_manifest.json").is_file()
    assert (out_dir / "case_0001" / "vace_command.json").is_file()
