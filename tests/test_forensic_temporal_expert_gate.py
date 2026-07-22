from __future__ import annotations

import numpy as np

from scripts.forensic_temporal_expert_gate.build_manifest import (
    _audit_duplicate_frames,
    assign_group_folds,
)
from scripts.forensic_temporal_expert_gate.complementarity import canonical_video_id
from scripts.forensic_temporal_expert_gate.contracts import resized_shape, stable_hash
from scripts.forensic_temporal_expert_gate.metrics import (
    best_balanced_threshold,
    classification_metrics,
)


def test_native_resize_is_patch_aligned_without_upscaling() -> None:
    width, height = resized_shape(
        1280, 720, patch_size=14, max_pixels=262144, max_side=672
    )
    assert width % 14 == 0
    assert height % 14 == 0
    assert width * height <= 262144
    assert max(width, height) <= 672
    assert width <= 1280 and height <= 720
    assert abs(width / height - 1280 / 720) < 0.05


def test_split_hash_is_deterministic() -> None:
    assert stable_hash("sample", 13) == stable_hash("sample", 13)
    assert stable_hash("sample", 13) != stable_hash("sample", 14)


def test_group_fold_assignment_never_splits_a_group() -> None:
    rows = []
    for group in range(20):
        for label in ("Real", "Fake"):
            rows.append(
                {
                    "group_id": f"group-{group}",
                    "label_name": label,
                    "source_dataset": "source",
                    "generator_name": "real" if label == "Real" else "generator",
                }
            )
    assign_group_folds(rows, folds=5, seed=7)
    by_group: dict[str, set[int]] = {}
    for row in rows:
        by_group.setdefault(row["group_id"], set()).add(row["fold"])
    assert all(len(folds) == 1 for folds in by_group.values())
    assert set(row["fold"] for row in rows) == set(range(5))


def test_metrics_and_threshold_are_balanced() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.4, 0.6, 0.9])
    threshold = best_balanced_threshold(labels, scores)
    report = classification_metrics(
        labels, scores, threshold, ["real", "real", "g1", "g2"]
    )
    assert report["balanced_accuracy"] == 1.0
    assert report["generator_macro_balanced_accuracy"] == 1.0


def test_vif_id_alignment_contract() -> None:
    assert canonical_video_id(
        "vifbench:/tmp/x/parsed_frames/parsed_frames/Fake/HunyuanVideo-I2V/a.mp4"
    ) == "HunyuanVideo-I2V/a.mp4"
    assert canonical_video_id(
        "/tmp/x/parsed_frames/parsed_frames/Real/real/b.mp4"
    ) == "real/b.mp4"
def test_duplicate_frame_records_are_retained_and_linked() -> None:
    rows = [
        {
            "sample_id": "row-0",
            "frame_paths": ["/x/001.png", "/x/002.png"],
            "label": 1,
        },
        {
            "sample_id": "row-1",
            "frame_paths": ["/x/001.png", "/x/002.png"],
            "label": 1,
        },
    ]
    assert _audit_duplicate_frames(rows) == 1
    assert len(rows) == 2
    assert rows[1]["duplicate_of_sample_id"] == "row-0"
