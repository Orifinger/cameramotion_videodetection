from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.camera_discriminative_gate.data import (
    PackedSequences,
    build_packed_sequences,
    fit_supervised_preprocessor,
)


def _feature_rows(tmp_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(7)
    specs = [
        ("train-real-a", "train", 0, 3),
        ("train-fake-a", "train", 1, 5),
        ("train-real-b", "train", 0, 7),
        ("train-fake-b", "train", 1, 9),
        ("val-real", "val", 0, 4),
        ("val-fake", "val", 1, 6),
    ]
    for sample_id, split, label, transitions in specs:
        feature_path = tmp_path / f"{sample_id}.npz"
        camera = rng.normal(size=(transitions, 4)).astype(np.float32)
        evidence = rng.normal(loc=0.25 * label, size=(transitions, 12)).astype(np.float32)
        np.savez_compressed(feature_path, camera_context=camera, temporal_evidence=evidence)
        rows.append(
            {
                "sample_id": sample_id,
                "dataset_name": "synthetic",
                "dataset_split": split,
                "label": label,
                "generator_name": "real" if label == 0 else "fake-model",
                "motion_bucket": "minor-motion",
                "frame_count_bin": str(transitions + 1),
                "feature_path": str(feature_path),
            }
        )
    return rows


def test_preprocessor_and_pack_preserve_variable_lengths(tmp_path: Path) -> None:
    rows = _feature_rows(tmp_path)
    train = [row for row in rows if row["dataset_split"] == "train"]
    preprocessor, summary = fit_supervised_preprocessor(
        train,
        pca_dim=6,
        fit_transitions_per_video=4,
        seed=11,
        clip_value=10.0,
    )
    packed = build_packed_sequences(rows, preprocessor)

    assert summary["fit_real_videos"] == 2
    assert summary["fit_fake_videos"] == 2
    assert [packed.sequence(index)[0].shape[0] for index in range(len(packed))] == [3, 5, 7, 9, 4, 6]
    assert packed.camera.shape[1] == 4
    assert packed.evidence.shape[1] == 6
    assert np.isfinite(packed.camera).all()
    assert np.isfinite(packed.evidence).all()

    npz_path = tmp_path / "packed.npz"
    rows_path = tmp_path / "rows.jsonl"
    packed.save(npz_path, rows_path)
    restored = PackedSequences.load(npz_path, rows_path)
    subset = restored.subset([1, 4])
    assert subset.offsets.tolist() == [0, 5, 9]
    assert subset.labels.tolist() == [1, 0]


def test_preprocessor_rejects_one_class(tmp_path: Path) -> None:
    rows = [row for row in _feature_rows(tmp_path) if int(row["label"]) == 0]
    with pytest.raises(ValueError, match="both Real and Fake"):
        fit_supervised_preprocessor(
            rows,
            pca_dim=4,
            fit_transitions_per_video=4,
            seed=11,
            clip_value=10.0,
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch unavailable")
def test_model_accepts_padded_variable_length_batch(tmp_path: Path) -> None:
    try:
        import torch
        from scripts.camera_discriminative_gate.model import CameraFiLMClassifier, collate_indices
    except (ImportError, OSError) as exc:
        pytest.skip(f"torch runtime unavailable: {exc}")

    rows = _feature_rows(tmp_path)
    preprocessor, _ = fit_supervised_preprocessor(
        [row for row in rows if row["dataset_split"] == "train"],
        pca_dim=6,
        fit_transitions_per_video=4,
        seed=11,
        clip_value=10.0,
    )
    packed = build_packed_sequences(rows, preprocessor)
    evidence, camera, mask, labels = collate_indices(
        packed,
        [0, 3],
        mode="matched",
        device=torch.device("cpu"),
    )
    model = CameraFiLMClassifier(evidence_dim=6, camera_dim=4, hidden_dim=16, dropout=0.0)
    logits = model(evidence, camera, mask)

    assert evidence.shape == (2, 9, 6)
    assert camera.shape == (2, 9, 4)
    assert mask.sum(dim=1).tolist() == [3, 9]
    assert labels.tolist() == [0.0, 1.0]
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
