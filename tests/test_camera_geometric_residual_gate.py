from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.camera_geometric_residual_gate.build_manifest import build_datab
from scripts.camera_geometric_residual_gate.contracts import camera_bucket, label_from_vif_path


def _detection_row(frame_dir: Path, answer: str) -> dict:
    return {
        "images": [str(frame_dir / f"{index:03d}.png") for index in range(8)],
        "messages": [
            {"role": "user", "content": "inspect"},
            {"role": "assistant", "content": f"<think>evidence</think><answer>{answer}</answer>"},
        ],
    }


def test_camera_bucket_marks_conflicting_coarse_labels() -> None:
    assert camera_bucket(["no-motion", "static", "regular-speed"]) == "static/no-motion"
    assert camera_bucket(["complex-motion", "fast-speed"]) == "complex-motion"
    assert camera_bucket(["minor-motion", "no-motion"]) == "ambiguous"
    assert camera_bucket(["regular-speed"]) == "unknown"


def test_vif_label_is_inferred_from_path_component() -> None:
    assert label_from_vif_path("/tmp/parsed_frames/Real/source/id") == "Real"
    assert label_from_vif_path("/tmp/parsed_frames/Fake/source/id") == "Fake"


def test_datab_manifest_deduplicates_same_video(tmp_path: Path) -> None:
    real_dir = tmp_path / "1vif4k" / "parsed_frames" / "Real" / "real-a"
    fake_dir = tmp_path / "1genbuster" / "parsed_frames" / "Fake" / "fake-a"
    detection = [
        _detection_row(real_dir, "Real"),
        _detection_row(fake_dir, "Fake"),
        _detection_row(fake_dir, "Fake"),
    ]
    camera = [
        {"path": str(real_dir), "labels": ["no-motion", "static"], "caption": "Static camera."},
        {"path": str(fake_dir), "labels": ["complex-motion"], "caption": "Moving camera."},
    ]
    detection_path = tmp_path / "detection.json"
    camera_path = tmp_path / "camera.jsonl"
    output_path = tmp_path / "manifest.jsonl"
    summary_path = tmp_path / "summary.json"
    detection_path.write_text(json.dumps(detection), encoding="utf-8")
    camera_path.write_text("".join(json.dumps(row) + "\n" for row in camera), encoding="utf-8")
    summary = build_datab(
        argparse.Namespace(
            detection_json=detection_path,
            camera_jsonl=camera_path,
            output_jsonl=output_path,
            summary_json=summary_path,
            val_ratio=0.15,
            seed=7,
            min_frames=8,
            check_files=False,
        )
    )
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary["matched_detection_rows"] == 3
    assert summary["unique_manifest_samples"] == 2
    assert summary["duplicate_frame_directory_groups"] == 1
    assert sorted(row["duplicate_detection_rows"] for row in rows) == [1, 2]
    assert all(row["camera_annotation_kind"].endswith("stratification only") for row in rows)


def test_motion_blocks_have_equal_finite_dimensions() -> None:
    pytest.importorskip("cv2")
    from scripts.camera_geometric_residual_gate.features import MOTION_BLOCK_DIM, build_motion_blocks

    height = width = 64
    pairs = 4
    forward = np.zeros((pairs, height, width, 2), dtype=np.float32)
    backward = np.zeros_like(forward)
    for index in range(pairs):
        dx = 1.0 + index * 0.75
        forward[index, ..., 0] = dx
        backward[index, ..., 0] = -dx
        forward[index, 24:40, 24:40, 1] += 0.3 * (index + 1)
    blocks, quality = build_motion_blocks(forward, backward, grid_step=8)
    assert set(blocks) == {"raw_motion", "geometry_residual", "wrong_geometry"}
    assert all(value.shape == (MOTION_BLOCK_DIM,) for value in blocks.values())
    assert all(np.isfinite(value).all() for value in blocks.values())
    assert not np.allclose(blocks["geometry_residual"], blocks["wrong_geometry"])
    assert quality["num_frame_pairs"] == pairs
