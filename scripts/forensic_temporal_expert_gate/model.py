"""Small temporal experts trained on frozen DINOv2 frame tokens."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


MODEL_SCHEMA_VERSION = "forensic_temporal_expert_model_v1"


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    hidden_dim: int = 192
    dropout: float = 0.15
    mode: str = "ordered"


class FrameEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.token_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.spatial_score = nn.Linear(hidden_dim, 1)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, cls_tokens: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        cls = self.token_projection(cls_tokens)
        patches = self.token_projection(patch_tokens)
        attention = torch.softmax(self.spatial_score(patches).squeeze(-1), dim=-1)
        attended = (patches * attention.unsqueeze(-1)).sum(dim=-2)
        mean = patches.mean(dim=-2)
        maximum = patches.amax(dim=-2)
        return self.output(torch.cat([cls, attended, mean, maximum], dim=-1))


class ForensicTemporalExpert(nn.Module):
    """Order-invariant static head or order-sensitive temporal head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.mode not in {"static", "ordered", "shuffled"}:
            raise ValueError(f"unsupported mode: {config.mode}")
        self.config = config
        self.frame_encoder = FrameEncoder(
            config.input_dim, config.hidden_dim, config.dropout
        )
        if config.mode == "static":
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_dim * 3, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )
        else:
            self.temporal_input = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.gru = nn.GRU(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_dim * 4, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )

    def forward(
        self,
        cls_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        frames = self.frame_encoder(cls_tokens, patch_tokens)
        steps = torch.arange(frames.shape[1], device=frames.device)[None, :]
        mask = steps < lengths[:, None]
        if self.config.mode == "static":
            masked = frames * mask.unsqueeze(-1)
            mean = masked.sum(dim=1) / lengths.clamp_min(1).unsqueeze(-1)
            maximum = frames.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1)
            centered = (frames - mean.unsqueeze(1)).abs() * mask.unsqueeze(-1)
            variation = centered.sum(dim=1) / lengths.clamp_min(1).unsqueeze(-1)
            return self.classifier(torch.cat([mean, maximum, variation], dim=-1)).squeeze(-1)

        previous = torch.cat([frames[:, :1], frames[:, :-1]], dim=1)
        temporal = self.temporal_input(torch.cat([frames, (frames - previous).abs()], dim=-1))
        packed = pack_padded_sequence(
            temporal,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=frames.shape[1]
        )
        masked = output * mask.unsqueeze(-1)
        mean = masked.sum(dim=1) / lengths.clamp_min(1).unsqueeze(-1)
        maximum = output.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1)
        return self.classifier(torch.cat([mean, maximum], dim=-1)).squeeze(-1)


def save_model(
    model: ForensicTemporalExpert,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    from scripts.forensic_temporal_expert_gate.contracts import write_json

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "model.tmp.pt"
    torch.save(model.state_dict(), temporary)
    temporary.replace(output_dir / "model.pt")
    write_json(
        output_dir / "config.json",
        {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model": asdict(model.config),
            **metadata,
        },
    )


def load_model(
    output_dir: Path, device: torch.device
) -> tuple[ForensicTemporalExpert, dict[str, Any]]:
    import json

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError(f"model schema mismatch: {output_dir}")
    model = ForensicTemporalExpert(ModelConfig(**config["model"]))
    try:
        state = torch.load(output_dir / "model.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(output_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config
