#!/usr/bin/env python3
"""Train static, ordered, and shuffled frozen-DINO experts on DataB."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.forensic_temporal_expert_gate.contracts import (
    normalize_path,
    read_json_or_jsonl,
    write_json,
)
from scripts.forensic_temporal_expert_gate.data import FeatureDataset, collate_features
from scripts.forensic_temporal_expert_gate.metrics import (
    best_balanced_threshold,
    classification_metrics,
    sigmoid,
)
from scripts.forensic_temporal_expert_gate.model import (
    ForensicTemporalExpert,
    ModelConfig,
    save_model,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(
    rows: Sequence[Mapping[str, Any]],
    *,
    order: str,
    seed: int,
    batch_size: int,
    shuffle: bool,
) -> tuple[FeatureDataset, DataLoader[dict[str, Any]]]:
    dataset = FeatureDataset(rows, order=order, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    return dataset, DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_features,
        generator=generator,
        drop_last=False,
    )


def forward_batch(
    model: ForensicTemporalExpert,
    batch: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    return model(
        batch["cls_tokens"].to(device, non_blocking=True),
        batch["patch_tokens"].to(device, non_blocking=True),
        batch["lengths"].to(device, non_blocking=True),
    )


@torch.inference_mode()
def predict(
    model: ForensicTemporalExpert,
    data_loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    generators: list[str] = []
    for batch in data_loader:
        logits.append(forward_batch(model, batch, device).float().cpu().numpy())
        labels.append(batch["labels"].numpy())
        generators.extend(str(row.get("generator_name", "unknown")) for row in batch["rows"])
    return np.concatenate(logits), np.concatenate(labels).astype(np.int64), generators


def train_one(
    *,
    mode: str,
    seed: int,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
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
    seed_everything(seed)
    input_dim = int(train_rows[0]["feature_hidden_size"])
    model = ForensicTemporalExpert(
        ModelConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            mode=mode,
        )
    ).to(device)
    order = "shuffled" if mode == "shuffled" else "ordered"
    train_dataset, train_loader = loader(
        train_rows,
        order=order,
        seed=seed,
        batch_size=batch_size,
        shuffle=True,
    )
    validation_dataset, validation_loader = loader(
        validation_rows,
        order=order,
        seed=seed + 100_000,
        batch_size=batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    best_auc = float("-inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch)
        validation_dataset.set_epoch(0)
        model.train()
        numerator = 0.0
        count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = forward_batch(model, batch, device)
            labels = batch["labels"].to(device, non_blocking=True)
            loss = loss_function(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            numerator += float(loss.detach().cpu()) * labels.numel()
            count += labels.numel()
        scheduler.step()
        validation_logits, validation_labels, generators = predict(
            model, validation_loader, device
        )
        scores = sigmoid(validation_logits)
        threshold = best_balanced_threshold(validation_labels, scores)
        metrics = classification_metrics(
            validation_labels, scores, threshold, generators
        )
        auc = float(metrics["roc_auc"])
        item = {
            "epoch": epoch,
            "train_loss": numerator / max(count, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": metrics,
            "elapsed_sec": time.time() - started,
        }
        history.append(item)
        print(json.dumps({"mode": mode, "seed": seed, **item}), flush=True)
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_epoch = epoch
            stale = 0
            save_model(
                model,
                output_dir,
                {
                    "mode": mode,
                    "seed": seed,
                    "best_epoch": epoch,
                    "selection_metric": "DataB fold-0 validation AUROC",
                    "validation_threshold": threshold,
                    "validation_metrics": metrics,
                    "train_records": len(train_rows),
                    "validation_records": len(validation_rows),
                    "validation_fold": 0,
                },
            )
        else:
            stale += 1
        if stale >= patience:
            break
    summary = {
        "status": "completed",
        "mode": mode,
        "seed": seed,
        "output_dir": normalize_path(output_dir),
        "best_epoch": best_epoch,
        "best_validation_roc_auc": best_auc,
        "epochs_ran": len(history),
        "elapsed_sec": time.time() - started,
        "history": history,
    }
    write_json(output_dir / "training_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-index-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 37, 73])
    parser.add_argument(
        "--modes", nargs="+", choices=("static", "ordered", "shuffled"), default=["static", "ordered", "shuffled"]
    )
    parser.add_argument("--job-index", type=int, default=0)
    parser.add_argument("--job-count", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.job_count < 1 or not 0 <= args.job_index < args.job_count:
        raise ValueError("job-index must be in [0, job-count)")
    rows = read_json_or_jsonl(args.feature_index_jsonl)
    train_rows = [row for row in rows if int(row["fold"]) != 0]
    validation_rows = [row for row in rows if int(row["fold"]) == 0]
    if not train_rows or not validation_rows:
        raise ValueError("DataB grouped train/validation folds are empty")
    jobs = [(mode, seed) for seed in args.seeds for mode in args.modes]
    selected = [job for index, job in enumerate(jobs) if index % args.job_count == args.job_index]
    device = torch.device(args.device)
    summaries = []
    for mode, seed in selected:
        summaries.append(
            train_one(
                mode=mode,
                seed=seed,
                train_rows=train_rows,
                validation_rows=validation_rows,
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
            "job_index": args.job_index,
            "job_count": args.job_count,
            "selected_jobs": [{"mode": mode, "seed": seed} for mode, seed in selected],
            "summaries": summaries,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
