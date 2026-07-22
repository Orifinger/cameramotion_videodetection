from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.forensic_temporal_expert_gate.build_manifest import assign_group_folds
from scripts.forensic_temporal_expert_gate.complementarity import canonical_video_id
from scripts.forensic_temporal_expert_gate.data import permutation
from scripts.forensic_temporal_expert_gate.extract_features import resized_shape
from scripts.forensic_temporal_expert_gate.metrics import best_balanced_threshold, classification_metrics
from scripts.forensic_temporal_expert_gate.model import ForensicTemporalExpert, ModelConfig


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


def test_variable_length_permutation_is_deterministic() -> None:
    first = permutation(17, "sample", 13, epoch=2)
    second = permutation(17, "sample", 13, epoch=2)
    changed = permutation(17, "sample", 13, epoch=3)
    np.testing.assert_array_equal(first, second)
    assert sorted(first.tolist()) == list(range(17))
    assert not np.array_equal(first, changed)


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


def test_static_head_is_invariant_to_frame_reversal_with_variable_lengths() -> None:
    torch.manual_seed(3)
    model = ForensicTemporalExpert(
        ModelConfig(input_dim=8, hidden_dim=12, dropout=0.0, mode="static")
    ).eval()
    cls = torch.randn(2, 5, 8)
    patches = torch.randn(2, 5, 16, 8)
    lengths = torch.tensor([5, 3])
    reversed_cls = cls.clone()
    reversed_patches = patches.clone()
    for index, length in enumerate(lengths.tolist()):
        reversed_cls[index, :length] = cls[index, :length].flip(0)
        reversed_patches[index, :length] = patches[index, :length].flip(0)
    with torch.no_grad():
        original = model(cls, patches, lengths)
        reversed_output = model(reversed_cls, reversed_patches, lengths)
    torch.testing.assert_close(original, reversed_output, rtol=0, atol=1e-6)


@pytest.mark.parametrize("mode", ["ordered", "shuffled"])
def test_temporal_model_accepts_11_16_17_frame_batches(mode: str) -> None:
    model = ForensicTemporalExpert(
        ModelConfig(input_dim=8, hidden_dim=12, dropout=0.0, mode=mode)
    ).eval()
    cls = torch.randn(3, 17, 8)
    patches = torch.randn(3, 17, 16, 8)
    lengths = torch.tensor([11, 16, 17])
    with torch.no_grad():
        output = model(cls, patches, lengths)
    assert output.shape == (3,)
    assert torch.isfinite(output).all()


def test_metrics_and_threshold_are_balanced() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.4, 0.6, 0.9])
    threshold = best_balanced_threshold(labels, scores)
    report = classification_metrics(labels, scores, threshold, ["real", "real", "g1", "g2"])
    assert report["balanced_accuracy"] == 1.0
    assert report["generator_macro_balanced_accuracy"] == 1.0


def test_vif_id_alignment_contract() -> None:
    assert canonical_video_id(
        "vifbench:/tmp/x/parsed_frames/parsed_frames/Fake/HunyuanVideo-I2V/a.mp4"
    ) == "HunyuanVideo-I2V/a.mp4"
    assert canonical_video_id(
        "/tmp/x/parsed_frames/parsed_frames/Real/real/b.mp4"
    ) == "real/b.mp4"
