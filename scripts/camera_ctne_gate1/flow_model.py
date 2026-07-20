"""Conditional normalizing-flow model used by CTNE Gate 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from scripts.camera_ctne_gate1.contracts import MODEL_SCHEMA_VERSION, write_json


def build_flow(
    *,
    evidence_dim: int,
    context_dim: int,
    hidden_features: int,
    num_blocks: int,
    transform_blocks: int,
) -> torch.nn.Module:
    try:
        from nflows.distributions.normal import StandardNormal
        from nflows.flows.base import Flow
        from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.permutations import RandomPermutation
    except ImportError as exc:  # pragma: no cover - checked by server preflight
        raise RuntimeError("nflows==0.14 is required for CTNE Gate 1") from exc
    transforms = []
    for _ in range(transform_blocks):
        transforms.extend(
            [
                RandomPermutation(features=evidence_dim),
                MaskedAffineAutoregressiveTransform(
                    features=evidence_dim,
                    hidden_features=hidden_features,
                    context_features=context_dim,
                    num_blocks=num_blocks,
                    use_residual_blocks=True,
                    random_mask=False,
                    activation=torch.nn.functional.relu,
                    dropout_probability=0.0,
                    use_batch_norm=False,
                ),
            ]
        )
    return Flow(CompositeTransform(transforms), StandardNormal([evidence_dim]))


def save_flow(model: torch.nn.Module, output_dir: Path, config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": MODEL_SCHEMA_VERSION, **config}
    write_json(output_dir / "config.json", payload)
    temporary = output_dir / "model.tmp.pt"
    torch.save(model.state_dict(), temporary)
    temporary.replace(output_dir / "model.pt")


def load_flow(output_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError(f"flow schema mismatch under {output_dir}")
    model = build_flow(
        evidence_dim=int(config["evidence_dim"]),
        context_dim=int(config["context_dim"]),
        hidden_features=int(config["hidden_features"]),
        num_blocks=int(config["num_blocks"]),
        transform_blocks=int(config["transform_blocks"]),
    )
    try:
        state = torch.load(output_dir / "model.pt", map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old torch compatibility
        state = torch.load(output_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, config
