"""Equal-capacity temporal classifier with continuous-camera FiLM modulation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from scripts.camera_ctne_gate1.contracts import write_json
from scripts.camera_ctne_gate1.preprocessing import resample_sequence
from scripts.camera_discriminative_gate import SCHEMA_VERSION
from scripts.camera_discriminative_gate.data import PackedSequences


class CameraFiLMClassifier(nn.Module):
    def __init__(
        self,
        *,
        evidence_dim: int,
        camera_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.evidence_dim = int(evidence_dim)
        self.camera_dim = int(camera_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.evidence_encoder = nn.Sequential(
            nn.Linear(evidence_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.camera_encoder = nn.Sequential(
            nn.Linear(camera_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.modulation_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, evidence: torch.Tensor, camera: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 3 or camera.ndim != 3 or mask.ndim != 2:
            raise ValueError("evidence/camera/mask must be [B,T,D], [B,T,C], [B,T]")
        hidden = self.evidence_encoder(evidence)
        gamma, beta = self.camera_encoder(camera).chunk(2, dim=-1)
        gamma = 0.5 * torch.tanh(gamma)
        modulated = self.modulation_norm((1.0 + gamma) * hidden + beta)
        valid = mask.unsqueeze(-1)
        attention_logits = self.attention(modulated).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, torch.finfo(attention_logits.dtype).min)
        attention_weights = torch.softmax(attention_logits, dim=1).unsqueeze(-1)
        attention_pool = (modulated * attention_weights).sum(dim=1)
        denominator = valid.sum(dim=1).clamp_min(1)
        mean_pool = (modulated * valid).sum(dim=1) / denominator
        max_pool = modulated.masked_fill(~valid, torch.finfo(modulated.dtype).min).max(dim=1).values
        return self.classifier(torch.cat([attention_pool, mean_pool, max_pool], dim=-1)).squeeze(-1)


def model_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def state_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_model(model: CameraFiLMClassifier, output_dir: Path, config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {"schema_version": SCHEMA_VERSION, **config})
    temporary = output_dir / "model.tmp.pt"
    torch.save(model.state_dict(), temporary)
    temporary.replace(output_dir / "model.pt")


def load_model(output_dir: Path, device: torch.device) -> tuple[CameraFiLMClassifier, dict[str, Any]]:
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"model schema mismatch under {output_dir}")
    model = CameraFiLMClassifier(
        evidence_dim=int(config["evidence_dim"]),
        camera_dim=int(config["camera_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    )
    try:
        state = torch.load(output_dir / "model.pt", map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old torch compatibility
        state = torch.load(output_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, config


def _mode_arrays(
    evidence: np.ndarray,
    camera: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "matched":
        return evidence, camera
    if mode == "zero_camera":
        return evidence, np.zeros_like(camera)
    if mode == "camera_only":
        return np.zeros_like(evidence), camera
    raise ValueError(f"unknown input mode: {mode}")


def collate_indices(
    packed: PackedSequences,
    indices: Sequence[int],
    *,
    mode: str,
    device: torch.device,
    camera_overrides: Sequence[np.ndarray] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences: list[tuple[np.ndarray, np.ndarray]] = []
    maximum = 0
    for batch_position, index in enumerate(indices):
        camera, evidence = packed.sequence(int(index))
        if camera_overrides is not None:
            camera = np.asarray(camera_overrides[batch_position], dtype=np.float32)
            if camera.shape[0] != evidence.shape[0]:
                camera = resample_sequence(camera, evidence.shape[0])
        evidence, camera = _mode_arrays(evidence, camera, mode=mode)
        sequences.append((evidence, camera))
        maximum = max(maximum, evidence.shape[0])
    batch = len(sequences)
    evidence_batch = np.zeros((batch, maximum, packed.evidence.shape[1]), dtype=np.float32)
    camera_batch = np.zeros((batch, maximum, packed.camera.shape[1]), dtype=np.float32)
    mask = np.zeros((batch, maximum), dtype=bool)
    for position, (evidence, camera) in enumerate(sequences):
        length = evidence.shape[0]
        evidence_batch[position, :length] = evidence
        camera_batch[position, :length] = camera
        mask[position, :length] = True
    labels = packed.labels[np.asarray(indices, dtype=np.int64)].astype(np.float32)
    return (
        torch.from_numpy(evidence_batch).to(device, non_blocking=True),
        torch.from_numpy(camera_batch).to(device, non_blocking=True),
        torch.from_numpy(mask).to(device, non_blocking=True),
        torch.from_numpy(labels).to(device, non_blocking=True),
    )


@torch.inference_mode()
def score_model(
    model: CameraFiLMClassifier,
    packed: PackedSequences,
    *,
    mode: str,
    device: torch.device,
    batch_size: int,
    camera_overrides: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    for start in range(0, len(packed), batch_size):
        indices = list(range(start, min(len(packed), start + batch_size)))
        overrides = None if camera_overrides is None else camera_overrides[start : start + len(indices)]
        evidence, camera, mask, _ = collate_indices(
            packed,
            indices,
            mode=mode,
            device=device,
            camera_overrides=overrides,
        )
        scores.append(model(evidence, camera, mask).float().cpu().numpy())
    return np.concatenate(scores).astype(np.float64)
