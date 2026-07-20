"""Leakage-safe preprocessing and variable-length sequence packing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.camera_ctne_gate1.contracts import normalize_path, read_jsonl, write_json, write_jsonl
from scripts.camera_ctne_gate1.preprocessing import load_feature_arrays, resample_sequence
from scripts.camera_discriminative_gate import SCHEMA_VERSION


def _safe_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


@dataclass
class SupervisedPreprocessor:
    camera_mean: np.ndarray
    camera_scale: np.ndarray
    evidence_mean: np.ndarray
    evidence_scale: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    pca_explained_variance_ratio: np.ndarray
    clip_value: float = 10.0

    @property
    def camera_dim(self) -> int:
        return int(self.camera_mean.size)

    @property
    def evidence_dim(self) -> int:
        return int(self.pca_components.shape[0])

    def fingerprint(self) -> str:
        digest = hashlib.sha256(SCHEMA_VERSION.encode("ascii"))
        for value in (
            self.camera_mean,
            self.camera_scale,
            self.evidence_mean,
            self.evidence_scale,
            self.pca_mean,
            self.pca_components,
        ):
            digest.update(np.ascontiguousarray(value).tobytes())
        digest.update(str(self.clip_value).encode("ascii"))
        return digest.hexdigest()

    def transform(self, camera: np.ndarray, evidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        camera = np.asarray(camera, dtype=np.float32)
        evidence = np.asarray(evidence, dtype=np.float32)
        if camera.ndim != 2 or evidence.ndim != 2 or camera.shape[0] != evidence.shape[0]:
            raise ValueError(f"invalid aligned transition arrays: {camera.shape} {evidence.shape}")
        camera_scaled = (camera - self.camera_mean) / self.camera_scale
        evidence_scaled = (evidence - self.evidence_mean) / self.evidence_scale
        evidence_projected = (evidence_scaled - self.pca_mean) @ self.pca_components.T
        camera_scaled = np.clip(camera_scaled, -self.clip_value, self.clip_value)
        evidence_projected = np.clip(evidence_projected, -self.clip_value, self.clip_value)
        return camera_scaled.astype(np.float32), evidence_projected.astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            schema_version=np.asarray(SCHEMA_VERSION),
            camera_mean=self.camera_mean,
            camera_scale=self.camera_scale,
            evidence_mean=self.evidence_mean,
            evidence_scale=self.evidence_scale,
            pca_mean=self.pca_mean,
            pca_components=self.pca_components,
            pca_explained_variance_ratio=self.pca_explained_variance_ratio,
            clip_value=np.asarray(self.clip_value, dtype=np.float32),
            fingerprint=np.asarray(self.fingerprint()),
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "SupervisedPreprocessor":
        with np.load(path, allow_pickle=False) as archive:
            schema = str(archive["schema_version"].item())
            if schema != SCHEMA_VERSION:
                raise ValueError(f"preprocessor schema mismatch: {schema}")
            value = cls(
                camera_mean=np.asarray(archive["camera_mean"], dtype=np.float32),
                camera_scale=np.asarray(archive["camera_scale"], dtype=np.float32),
                evidence_mean=np.asarray(archive["evidence_mean"], dtype=np.float32),
                evidence_scale=np.asarray(archive["evidence_scale"], dtype=np.float32),
                pca_mean=np.asarray(archive["pca_mean"], dtype=np.float32),
                pca_components=np.asarray(archive["pca_components"], dtype=np.float32),
                pca_explained_variance_ratio=np.asarray(
                    archive["pca_explained_variance_ratio"], dtype=np.float32
                ),
                clip_value=float(archive["clip_value"].item()),
            )
            expected = str(archive["fingerprint"].item())
        if value.fingerprint() != expected:
            raise ValueError(f"preprocessor fingerprint mismatch: {path}")
        return value


def fit_supervised_preprocessor(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    pca_dim: int,
    fit_transitions_per_video: int,
    seed: int,
    clip_value: float,
) -> tuple[SupervisedPreprocessor, dict[str, Any]]:
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:  # pragma: no cover - server preflight checks this
        raise RuntimeError("scikit-learn is required") from exc
    if fit_transitions_per_video < 2:
        raise ValueError("fit_transitions_per_video must be at least 2")
    camera_parts: list[np.ndarray] = []
    evidence_parts: list[np.ndarray] = []
    labels: list[int] = []
    for row in train_rows:
        camera, evidence = load_feature_arrays(row)
        camera_parts.append(resample_sequence(camera, fit_transitions_per_video))
        evidence_parts.append(resample_sequence(evidence, fit_transitions_per_video))
        labels.append(int(row["label"]))
    if not camera_parts or len(set(labels)) != 2:
        raise ValueError("preprocessor requires both Real and Fake DataB training videos")
    camera = np.concatenate(camera_parts).astype(np.float32)
    evidence = np.concatenate(evidence_parts).astype(np.float32)
    camera_mean, camera_scale = _safe_scale(camera)
    evidence_mean, evidence_scale = _safe_scale(evidence)
    evidence_scaled = (evidence - evidence_mean) / evidence_scale
    components = max(1, min(int(pca_dim), evidence_scaled.shape[0] - 1, evidence_scaled.shape[1]))
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    pca.fit(evidence_scaled)
    preprocessor = SupervisedPreprocessor(
        camera_mean=camera_mean,
        camera_scale=camera_scale,
        evidence_mean=evidence_mean,
        evidence_scale=evidence_scale,
        pca_mean=np.asarray(pca.mean_, dtype=np.float32),
        pca_components=np.asarray(pca.components_, dtype=np.float32),
        pca_explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float32),
        clip_value=float(clip_value),
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "fit_scope": "DataB train Real and Fake only",
        "fit_videos": len(train_rows),
        "fit_real_videos": int(sum(label == 0 for label in labels)),
        "fit_fake_videos": int(sum(label == 1 for label in labels)),
        "fit_transitions_per_video": int(fit_transitions_per_video),
        "equal_video_contribution_to_preprocessing": True,
        "raw_camera_dim": int(camera.shape[1]),
        "raw_evidence_dim": int(evidence.shape[1]),
        "projected_evidence_dim": int(components),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "clip_value": float(clip_value),
        "seed": int(seed),
        "preprocessor_fingerprint": preprocessor.fingerprint(),
    }
    return preprocessor, summary


@dataclass
class PackedSequences:
    camera: np.ndarray
    evidence: np.ndarray
    offsets: np.ndarray
    labels: np.ndarray
    rows: list[dict[str, Any]]
    preprocessor_fingerprint: str

    def __len__(self) -> int:
        return int(self.labels.size)

    def sequence(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return self.camera[start:end], self.evidence[start:end]

    def subset(self, indices: Sequence[int]) -> "PackedSequences":
        camera_parts: list[np.ndarray] = []
        evidence_parts: list[np.ndarray] = []
        offsets = [0]
        rows: list[dict[str, Any]] = []
        labels: list[int] = []
        for index in indices:
            camera, evidence = self.sequence(int(index))
            camera_parts.append(camera)
            evidence_parts.append(evidence)
            offsets.append(offsets[-1] + camera.shape[0])
            rows.append(dict(self.rows[int(index)]))
            labels.append(int(self.labels[int(index)]))
        return PackedSequences(
            camera=np.concatenate(camera_parts).astype(np.float32),
            evidence=np.concatenate(evidence_parts).astype(np.float32),
            offsets=np.asarray(offsets, dtype=np.int64),
            labels=np.asarray(labels, dtype=np.int64),
            rows=rows,
            preprocessor_fingerprint=self.preprocessor_fingerprint,
        )

    def save(self, npz_path: Path, rows_path: Path) -> None:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = npz_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            schema_version=np.asarray(SCHEMA_VERSION),
            camera=self.camera,
            evidence=self.evidence,
            offsets=self.offsets,
            labels=self.labels,
            preprocessor_fingerprint=np.asarray(self.preprocessor_fingerprint),
        )
        temporary.replace(npz_path)
        write_jsonl(rows_path, self.rows)

    @classmethod
    def load(cls, npz_path: Path, rows_path: Path) -> "PackedSequences":
        rows = read_jsonl(rows_path)
        with np.load(npz_path, allow_pickle=False) as archive:
            schema = str(archive["schema_version"].item())
            if schema != SCHEMA_VERSION:
                raise ValueError(f"packed sequence schema mismatch: {schema}")
            value = cls(
                camera=np.asarray(archive["camera"], dtype=np.float32),
                evidence=np.asarray(archive["evidence"], dtype=np.float32),
                offsets=np.asarray(archive["offsets"], dtype=np.int64),
                labels=np.asarray(archive["labels"], dtype=np.int64),
                rows=rows,
                preprocessor_fingerprint=str(archive["preprocessor_fingerprint"].item()),
            )
        value.validate()
        return value

    def validate(self) -> None:
        if self.camera.ndim != 2 or self.evidence.ndim != 2:
            raise ValueError("packed camera/evidence arrays must be two-dimensional")
        if self.camera.shape[0] != self.evidence.shape[0]:
            raise ValueError("packed camera/evidence transition counts differ")
        if self.offsets.shape != (len(self.rows) + 1,) or self.labels.shape != (len(self.rows),):
            raise ValueError("packed offsets, labels, and metadata lengths differ")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != self.camera.shape[0]:
            raise ValueError("packed offsets do not consume transitions")
        if np.any(np.diff(self.offsets) < 1):
            raise ValueError("every packed video must have at least one transition")
        if not np.isfinite(self.camera).all() or not np.isfinite(self.evidence).all():
            raise ValueError("packed arrays contain non-finite values")
        for index, row in enumerate(self.rows):
            if int(row["label"]) != int(self.labels[index]):
                raise ValueError(f"packed label mismatch for {row.get('sample_id')}")


def build_packed_sequences(
    rows: Sequence[Mapping[str, Any]],
    preprocessor: SupervisedPreprocessor,
) -> PackedSequences:
    camera_parts: list[np.ndarray] = []
    evidence_parts: list[np.ndarray] = []
    offsets = [0]
    metadata: list[dict[str, Any]] = []
    labels: list[int] = []
    for row in rows:
        camera, evidence = load_feature_arrays(row)
        camera_scaled, evidence_projected = preprocessor.transform(camera, evidence)
        camera_parts.append(camera_scaled)
        evidence_parts.append(evidence_projected)
        offsets.append(offsets[-1] + camera_scaled.shape[0])
        metadata.append(dict(row))
        labels.append(int(row["label"]))
    if not camera_parts:
        raise ValueError("no feature rows available for sequence packing")
    result = PackedSequences(
        camera=np.concatenate(camera_parts).astype(np.float32),
        evidence=np.concatenate(evidence_parts).astype(np.float32),
        offsets=np.asarray(offsets, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        rows=metadata,
        preprocessor_fingerprint=preprocessor.fingerprint(),
    )
    result.validate()
    return result


def prepare_datab(
    *,
    feature_index_jsonl: Path,
    output_dir: Path,
    pca_dim: int,
    fit_transitions_per_video: int,
    seed: int,
    clip_value: float,
) -> dict[str, Any]:
    rows = read_jsonl(feature_index_jsonl)
    train_rows = [row for row in rows if str(row.get("dataset_split")) == "train"]
    val_rows = [row for row in rows if str(row.get("dataset_split")) == "val"]
    if not train_rows or not val_rows:
        raise ValueError(f"need DataB train/val rows, found {len(train_rows)} and {len(val_rows)}")
    preprocessor, preprocessing_summary = fit_supervised_preprocessor(
        train_rows,
        pca_dim=pca_dim,
        fit_transitions_per_video=fit_transitions_per_video,
        seed=seed,
        clip_value=clip_value,
    )
    packed = build_packed_sequences(rows, preprocessor)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save(output_dir / "preprocessor.npz")
    packed.save(output_dir / "datab_sequences.npz", output_dir / "datab_rows.jsonl")
    split_counts = {
        split: int(sum(str(row.get("dataset_split")) == split for row in rows))
        for split in ("train", "val")
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "feature_index_jsonl": normalize_path(feature_index_jsonl),
        "output_dir": normalize_path(output_dir),
        "records": len(rows),
        "split_counts": split_counts,
        "label_counts": {
            "Real": int(sum(int(row["label"]) == 0 for row in rows)),
            "Fake": int(sum(int(row["label"]) == 1 for row in rows)),
        },
        "total_transitions": int(packed.camera.shape[0]),
        "camera_dim": int(packed.camera.shape[1]),
        "evidence_dim": int(packed.evidence.shape[1]),
        **preprocessing_summary,
    }
    write_json(output_dir / "prepare_summary.json", summary)
    return summary
