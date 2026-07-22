"""Lazy NPZ dataset and variable-length collation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def permutation(length: int, key: str, seed: int, epoch: int = 0) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}:{epoch}:{key}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return rng.permutation(length)


class FeatureDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        order: str = "ordered",
        seed: int = 0,
    ) -> None:
        if order not in {"ordered", "shuffled"}:
            raise ValueError(f"unknown order: {order}")
        self.rows = [dict(row) for row in rows]
        self.order = order
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with np.load(Path(row["feature_path"]), allow_pickle=False) as archive:
            cls = np.asarray(archive["cls_tokens"], dtype=np.float32)
            patches = np.asarray(archive["patch_tokens"], dtype=np.float32)
        if cls.shape[0] != patches.shape[0] or cls.shape[0] < 2:
            raise ValueError(f"invalid feature length for {row['sample_id']}")
        if self.order == "shuffled":
            indices = permutation(cls.shape[0], str(row["sample_id"]), self.seed, self.epoch)
            cls = cls[indices]
            patches = patches[indices]
        return {
            "cls_tokens": cls,
            "patch_tokens": patches,
            "label": int(row["label"]),
            "row": row,
        }


def collate_features(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor(
        [int(item["cls_tokens"].shape[0]) for item in items], dtype=torch.long
    )
    max_length = int(lengths.max())
    input_dim = int(items[0]["cls_tokens"].shape[-1])
    grid_tokens = int(items[0]["patch_tokens"].shape[-2])
    cls = torch.zeros((len(items), max_length, input_dim), dtype=torch.float32)
    patches = torch.zeros(
        (len(items), max_length, grid_tokens, input_dim), dtype=torch.float32
    )
    for index, item in enumerate(items):
        length = int(lengths[index])
        cls[index, :length] = torch.from_numpy(item["cls_tokens"])
        patches[index, :length] = torch.from_numpy(item["patch_tokens"])
    return {
        "cls_tokens": cls,
        "patch_tokens": patches,
        "lengths": lengths,
        "labels": torch.tensor([int(item["label"]) for item in items], dtype=torch.float32),
        "rows": [dict(item["row"]) for item in items],
    }
