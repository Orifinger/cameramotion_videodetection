"""Offline RAFT and DINOv2 loaders used by the probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from scripts.camera_flow_probe.geometry import CanvasGeometry, canvas_geometry, resize_and_pad


def _batches(length: int, batch_size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, length, max(1, batch_size)):
        yield start, min(length, start + max(1, batch_size))


def resolve_hf_model_root(path: Path) -> Path:
    if (path / "config.json").is_file() and (path / "model.safetensors").is_file():
        return path
    matches = [candidate.parent for candidate in path.rglob("config.json") if (candidate.parent / "model.safetensors").is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one local Hugging Face model under {path}, found {len(matches)}")
    return matches[0]


def _unwrap_state_dict(value: Any) -> dict[str, torch.Tensor]:
    if isinstance(value, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            nested = value.get(key)
            if isinstance(nested, dict):
                value = nested
                break
    if not isinstance(value, dict):
        raise ValueError("checkpoint does not contain a state dict")
    output: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        normalized = str(key)
        if normalized.startswith("module."):
            normalized = normalized[len("module.") :]
        output[normalized] = tensor
    return output


class TorchvisionRaft:
    def __init__(
        self,
        checkpoint: Path,
        *,
        device: torch.device,
        long_side: int = 512,
        batch_size: int = 4,
    ) -> None:
        from torchvision.models.optical_flow import raft_large

        self.device = device
        self.long_side = int(long_side)
        self.batch_size = int(batch_size)
        self.model = raft_large(weights=None, progress=False)
        try:
            checkpoint_value = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint_value = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(_unwrap_state_dict(checkpoint_value), strict=True)
        self.model.to(device).eval()

    @torch.inference_mode()
    def infer_pairs(
        self,
        frames: np.ndarray,
        *,
        backward: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None, CanvasGeometry]:
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] < 2:
            raise ValueError(f"frames must be [T,H,W,3] with T>=2, got {frames.shape}")
        geometry = canvas_geometry(
            int(frames.shape[1]),
            int(frames.shape[2]),
            long_side=self.long_side,
            multiple=8,
        )
        prepared = np.stack([resize_and_pad(frame, geometry) for frame in frames])
        tensor = torch.from_numpy(prepared).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)

        def run(first: torch.Tensor, second: torch.Tensor) -> np.ndarray:
            outputs: list[np.ndarray] = []
            for start, end in _batches(first.shape[0], self.batch_size):
                flow = self.model(
                    first[start:end].to(self.device, non_blocking=True),
                    second[start:end].to(self.device, non_blocking=True),
                )[-1]
                outputs.append(flow.permute(0, 2, 3, 1).float().cpu().numpy())
            return np.concatenate(outputs, axis=0)

        forward = run(tensor[:-1], tensor[1:])
        reverse = run(tensor[1:], tensor[:-1]) if backward else None
        return forward, reverse, geometry


class DinoV2Extractor:
    def __init__(
        self,
        model_path: Path,
        *,
        device: torch.device,
        long_side: int = 518,
        batch_size: int = 16,
    ) -> None:
        from transformers import AutoModel

        self.root = resolve_hf_model_root(model_path)
        self.device = device
        self.long_side = int(long_side)
        self.batch_size = int(batch_size)
        self.model = AutoModel.from_pretrained(
            self.root,
            local_files_only=True,
            trust_remote_code=False,
        ).to(device).eval()
        self.patch_size = int(getattr(self.model.config, "patch_size", 14))
        self.register_tokens = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        preprocessor_path = self.root / "preprocessor_config.json"
        preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8")) if preprocessor_path.is_file() else {}
        self.mean = torch.tensor(preprocessor.get("image_mean", [0.485, 0.456, 0.406])).view(1, 3, 1, 1)
        self.std = torch.tensor(preprocessor.get("image_std", [0.229, 0.224, 0.225])).view(1, 3, 1, 1)

    @torch.inference_mode()
    def extract(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, CanvasGeometry]:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"frames must be [T,H,W,3], got {frames.shape}")
        geometry = canvas_geometry(
            int(frames.shape[1]),
            int(frames.shape[2]),
            long_side=self.long_side,
            multiple=self.patch_size,
        )
        prepared = np.stack([resize_and_pad(frame, geometry) for frame in frames])
        tensor = torch.from_numpy(prepared).permute(0, 3, 1, 2).float().div_(255.0)
        tensor = (tensor - self.mean) / self.std
        cls_outputs: list[np.ndarray] = []
        patch_outputs: list[np.ndarray] = []
        grid_height = geometry.canvas_height // self.patch_size
        grid_width = geometry.canvas_width // self.patch_size
        for start, end in _batches(tensor.shape[0], self.batch_size):
            values = tensor[start:end].to(self.device, non_blocking=True)
            try:
                output = self.model(pixel_values=values, interpolate_pos_encoding=True)
            except TypeError:
                output = self.model(pixel_values=values)
            hidden = output.last_hidden_state.float()
            cls_outputs.append(hidden[:, 0].cpu().numpy())
            patches = hidden[:, 1 + self.register_tokens :]
            expected = grid_height * grid_width
            if patches.shape[1] != expected:
                raise ValueError(
                    f"DINO patch count mismatch: expected={expected} actual={patches.shape[1]} "
                    f"canvas={(geometry.canvas_height, geometry.canvas_width)} patch={self.patch_size}"
                )
            patches = patches.reshape(-1, grid_height, grid_width, patches.shape[-1]).permute(0, 3, 1, 2)
            patch_outputs.append(patches.cpu().numpy())
        return np.concatenate(cls_outputs), np.concatenate(patch_outputs), geometry
