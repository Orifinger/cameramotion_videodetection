"""Leakage-safe preprocessing shared by all CTNE controls."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.camera_ctne_gate1.contracts import MODEL_SCHEMA_VERSION


def load_feature_arrays(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(str(row["feature_path"])), allow_pickle=False) as archive:
        camera = np.asarray(archive["camera_context"], dtype=np.float32)
        evidence = np.asarray(archive["temporal_evidence"], dtype=np.float32)
    if camera.ndim != 2 or evidence.ndim != 2 or camera.shape[0] != evidence.shape[0]:
        raise ValueError(f"invalid feature arrays for {row.get('sample_id')}: {camera.shape} {evidence.shape}")
    return camera, evidence


def deterministic_indices(sample_id: str, count: int, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or count <= maximum:
        return np.arange(count, dtype=np.int64)
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return np.sort(rng.choice(count, size=maximum, replace=False)).astype(np.int64)


def per_video_transition_weights(lengths: Sequence[int]) -> np.ndarray:
    parts = []
    for length in lengths:
        if length <= 0:
            raise ValueError("every video must have at least one transition")
        parts.append(np.full(length, 1.0 / length, dtype=np.float32))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[0] < 1 or target_length < 1:
        raise ValueError(f"invalid sequence resampling request: {sequence.shape} -> {target_length}")
    if sequence.shape[0] == target_length:
        return sequence.copy()
    source = np.linspace(0.0, 1.0, sequence.shape[0])
    target = np.linspace(0.0, 1.0, target_length)
    output = np.stack([np.interp(target, source, sequence[:, index]) for index in range(sequence.shape[1])], axis=1)
    return output.astype(np.float32)


def camera_video_summary(camera: np.ndarray) -> np.ndarray:
    camera = np.asarray(camera, dtype=np.float32)
    if camera.ndim != 2 or camera.shape[0] < 1:
        raise ValueError("camera sequence must be [T,C] with T>=1")
    return np.concatenate(
        [
            camera.mean(axis=0),
            camera.std(axis=0),
            np.quantile(camera, 0.10, axis=0),
            np.quantile(camera, 0.90, axis=0),
        ]
    ).astype(np.float32)


@dataclass
class CTNEPreprocessor:
    camera_mean: np.ndarray
    camera_scale: np.ndarray
    evidence_mean: np.ndarray
    evidence_scale: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    pca_explained_variance_ratio: np.ndarray

    @property
    def camera_dim(self) -> int:
        return int(self.camera_mean.size)

    @property
    def evidence_dim(self) -> int:
        return int(self.pca_components.shape[0])

    def transform(self, camera: np.ndarray, evidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        camera = np.asarray(camera, dtype=np.float32)
        evidence = np.asarray(evidence, dtype=np.float32)
        if camera.shape[0] != evidence.shape[0]:
            raise ValueError("camera and evidence transition counts differ")
        camera_scaled = (camera - self.camera_mean) / self.camera_scale
        evidence_scaled = (evidence - self.evidence_mean) / self.evidence_scale
        evidence_pca = (evidence_scaled - self.pca_mean) @ self.pca_components.T
        return camera_scaled.astype(np.float32), evidence_pca.astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            schema_version=np.asarray(MODEL_SCHEMA_VERSION),
            camera_mean=self.camera_mean,
            camera_scale=self.camera_scale,
            evidence_mean=self.evidence_mean,
            evidence_scale=self.evidence_scale,
            pca_mean=self.pca_mean,
            pca_components=self.pca_components,
            pca_explained_variance_ratio=self.pca_explained_variance_ratio,
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "CTNEPreprocessor":
        with np.load(path, allow_pickle=False) as archive:
            schema = str(archive["schema_version"].item())
            if schema != MODEL_SCHEMA_VERSION:
                raise ValueError(f"preprocessor schema mismatch: {schema}")
            return cls(
                camera_mean=np.asarray(archive["camera_mean"], dtype=np.float32),
                camera_scale=np.asarray(archive["camera_scale"], dtype=np.float32),
                evidence_mean=np.asarray(archive["evidence_mean"], dtype=np.float32),
                evidence_scale=np.asarray(archive["evidence_scale"], dtype=np.float32),
                pca_mean=np.asarray(archive["pca_mean"], dtype=np.float32),
                pca_components=np.asarray(archive["pca_components"], dtype=np.float32),
                pca_explained_variance_ratio=np.asarray(archive["pca_explained_variance_ratio"], dtype=np.float32),
            )


def fit_preprocessor(
    real_train_rows: Sequence[Mapping[str, Any]],
    *,
    pca_dim: int,
    max_transitions_per_video: int,
    seed: int,
) -> tuple[CTNEPreprocessor, dict[str, Any]]:
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:  # pragma: no cover - checked in server preflight
        raise RuntimeError("scikit-learn is required for CTNE PCA preprocessing") from exc
    camera_parts: list[np.ndarray] = []
    evidence_parts: list[np.ndarray] = []
    for row in real_train_rows:
        camera, evidence = load_feature_arrays(row)
        indices = deterministic_indices(
            str(row["sample_id"]),
            camera.shape[0],
            max_transitions_per_video,
            seed,
        )
        camera_parts.append(camera[indices])
        evidence_parts.append(evidence[indices])
    if not camera_parts:
        raise ValueError("no real training videos available for preprocessing")
    camera = np.concatenate(camera_parts).astype(np.float32)
    evidence = np.concatenate(evidence_parts).astype(np.float32)
    camera_mean = camera.mean(axis=0)
    camera_scale = camera.std(axis=0)
    camera_scale[camera_scale < 1e-6] = 1.0
    evidence_mean = evidence.mean(axis=0)
    evidence_scale = evidence.std(axis=0)
    evidence_scale[evidence_scale < 1e-6] = 1.0
    evidence_scaled = (evidence - evidence_mean) / evidence_scale
    components = max(1, min(int(pca_dim), evidence_scaled.shape[0] - 1, evidence_scaled.shape[1]))
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    pca.fit(evidence_scaled)
    preprocessor = CTNEPreprocessor(
        camera_mean=camera_mean.astype(np.float32),
        camera_scale=camera_scale.astype(np.float32),
        evidence_mean=evidence_mean.astype(np.float32),
        evidence_scale=evidence_scale.astype(np.float32),
        pca_mean=np.asarray(pca.mean_, dtype=np.float32),
        pca_components=np.asarray(pca.components_, dtype=np.float32),
        pca_explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float32),
    )
    summary = {
        "real_training_videos": len(real_train_rows),
        "sampled_training_transitions": int(camera.shape[0]),
        "raw_camera_dim": int(camera.shape[1]),
        "raw_evidence_dim": int(evidence.shape[1]),
        "pca_evidence_dim": components,
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "max_transitions_per_video_for_fit": int(max_transitions_per_video),
        "fit_uses_real_train_only": True,
    }
    return preprocessor, summary
