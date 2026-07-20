#!/usr/bin/env python3
"""Train matched, zero-camera, and camera-only supervised video classifiers."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_ctne_gate1.contracts import normalize_path, write_json
from scripts.camera_discriminative_gate import SCHEMA_VERSION
from scripts.camera_discriminative_gate.data import PackedSequences
from scripts.camera_discriminative_gate.model import (
    CameraFiLMClassifier,
    collate_indices,
    model_parameter_count,
    save_model,
    score_model,
    state_fingerprint,
)

MODES = ("matched", "zero_camera", "camera_only")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _indices_for_split(packed: PackedSequences, split: str) -> list[int]:
    return [index for index, row in enumerate(packed.rows) if str(row.get("dataset_split")) == split]


def _batch_order(indices: Sequence[int], seed: int, epoch: int) -> np.ndarray:
    rng = np.random.default_rng(seed * 100003 + epoch)
    return rng.permutation(np.asarray(indices, dtype=np.int64))


def _subset_for_scoring(packed: PackedSequences, indices: Sequence[int]) -> PackedSequences:
    return packed.subset(indices)


def train_one(
    *,
    packed: PackedSequences,
    mode: str,
    seed: int,
    output_dir: Path,
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
) -> dict[str, Any]:
    train_indices = _indices_for_split(packed, "train")
    val_indices = _indices_for_split(packed, "val")
    if not train_indices or not val_indices:
        raise ValueError("packed DataB sequences require train and val splits")
    train_labels = packed.labels[np.asarray(train_indices, dtype=np.int64)]
    if np.unique(train_labels).size != 2:
        raise ValueError("training split requires both classes")
    negative = int((train_labels == 0).sum())
    positive = int((train_labels == 1).sum())
    pos_weight = torch.tensor([negative / max(positive, 1)], dtype=torch.float32, device=device)
    validation = _subset_for_scoring(packed, val_indices)
    _seed_everything(seed)
    model = CameraFiLMClassifier(
        evidence_dim=packed.evidence.shape[1],
        camera_dim=packed.camera.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    initial_fingerprint = state_fingerprint(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_auc = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        numerator = 0.0
        count = 0
        order = _batch_order(train_indices, seed, epoch)
        for start in range(0, order.size, batch_size):
            indices = order[start : start + batch_size].tolist()
            evidence, camera, mask, labels = collate_indices(
                packed,
                indices,
                mode=mode,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(evidence, camera, mask)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss for mode={mode}, seed={seed}, epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            numerator += float(loss.detach().cpu()) * len(indices)
            count += len(indices)
        scheduler.step()
        val_scores = score_model(
            model,
            validation,
            mode=mode,
            device=device,
            batch_size=batch_size,
        )
        val_auc = float(roc_auc_score(validation.labels, val_scores))
        train_loss = numerator / max(count, 1)
        if not np.isfinite(val_auc) or not np.isfinite(train_loss):
            raise FloatingPointError(f"non-finite metric for mode={mode}, seed={seed}, epoch={epoch}")
        item = {
            "epoch": epoch,
            "train_bce": train_loss,
            "val_roc_auc": val_auc,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_sec": time.time() - started,
        }
        history.append(item)
        print(json.dumps({"mode": mode, "seed": seed, **item}), flush=True)
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_epoch = epoch
            stale = 0
            save_model(
                model,
                output_dir,
                {
                    "mode": mode,
                    "seed": seed,
                    "evidence_dim": int(packed.evidence.shape[1]),
                    "camera_dim": int(packed.camera.shape[1]),
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "parameter_count": model_parameter_count(model),
                    "initial_state_fingerprint": initial_fingerprint,
                    "preprocessor_fingerprint": packed.preprocessor_fingerprint,
                    "best_epoch": best_epoch,
                    "best_val_roc_auc": best_auc,
                },
            )
        else:
            stale += 1
        if stale >= patience:
            break
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "mode": mode,
        "seed": seed,
        "output_dir": normalize_path(output_dir),
        "train_videos": len(train_indices),
        "val_videos": len(val_indices),
        "train_real": negative,
        "train_fake": positive,
        "pos_weight": float(pos_weight.item()),
        "parameter_count": model_parameter_count(model),
        "initial_state_fingerprint": initial_fingerprint,
        "best_epoch": best_epoch,
        "best_val_roc_auc": best_auc,
        "epochs_ran": len(history),
        "elapsed_sec": time.time() - started,
        "history": history,
    }
    write_json(output_dir / "training_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-npz", type=Path, required=True)
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 37, 73])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--job-index", type=int, default=0)
    parser.add_argument("--job-count", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.job_count < 1 or not 0 <= args.job_index < args.job_count:
        raise ValueError("job-index must be in [0, job-count)")
    packed = PackedSequences.load(args.packed_npz, args.rows_jsonl)
    if args.max_samples > 0 and len(packed) > args.max_samples:
        train = _indices_for_split(packed, "train")[: max(2, args.max_samples * 4 // 5)]
        val = _indices_for_split(packed, "val")[: max(2, args.max_samples // 5)]
        packed = packed.subset(train + val)
    jobs = [(mode, seed) for seed in args.seeds for mode in args.modes]
    selected = [job for index, job in enumerate(jobs) if index % args.job_count == args.job_index]
    device = torch.device(args.device)
    summaries = []
    for mode, seed in selected:
        summaries.append(
            train_one(
                packed=packed,
                mode=mode,
                seed=seed,
                output_dir=args.output_dir / "models" / f"seed_{seed}" / mode,
                device=device,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip=args.gradient_clip,
            )
        )
    write_json(
        args.output_dir / "jobs" / f"job_{args.job_index:02d}_of_{args.job_count:02d}.json",
        {
            "schema_version": SCHEMA_VERSION,
            "job_index": args.job_index,
            "job_count": args.job_count,
            "selected_jobs": [{"mode": mode, "seed": seed} for mode, seed in selected],
            "summaries": summaries,
            "pid": os.getpid(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
