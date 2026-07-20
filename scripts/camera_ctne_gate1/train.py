#!/usr/bin/env python3
"""Train matched-camera and equal-capacity unconditional CTNE flows."""

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
from torch.utils.data import DataLoader, TensorDataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.camera_ctne_gate1.contracts import MODEL_SCHEMA_VERSION, normalize_path, read_jsonl, write_json
from scripts.camera_ctne_gate1.flow_model import build_flow, save_flow
from scripts.camera_ctne_gate1.preprocessing import (
    CTNEPreprocessor,
    fit_preprocessor,
    load_feature_arrays,
    per_video_transition_weights,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _real_rows(rows: Sequence[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows if str(row.get("dataset_split")) == split and int(row.get("label", -1)) == 0]
    output.sort(key=lambda row: str(row["sample_id"]))
    return output


def _load_transformed(
    rows: Sequence[Mapping[str, Any]],
    preprocessor: CTNEPreprocessor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    cameras: list[np.ndarray] = []
    evidence: list[np.ndarray] = []
    lengths: list[int] = []
    for row in rows:
        camera_raw, evidence_raw = load_feature_arrays(row)
        camera, projected = preprocessor.transform(camera_raw, evidence_raw)
        cameras.append(camera)
        evidence.append(projected)
        lengths.append(camera.shape[0])
    if not cameras:
        raise ValueError("no transformed real-video transitions")
    return (
        np.concatenate(cameras).astype(np.float32),
        np.concatenate(evidence).astype(np.float32),
        per_video_transition_weights(lengths),
        lengths,
    )


@torch.inference_mode()
def _validation_loss(
    model: torch.nn.Module,
    camera: np.ndarray,
    evidence: np.ndarray,
    lengths: Sequence[int],
    *,
    mode: str,
    device: torch.device,
    batch_size: int,
) -> float:
    context = np.zeros_like(camera) if mode == "unconditional" else camera
    losses: list[np.ndarray] = []
    for start in range(0, evidence.shape[0], batch_size):
        end = min(evidence.shape[0], start + batch_size)
        y = torch.from_numpy(evidence[start:end]).to(device)
        c = torch.from_numpy(context[start:end]).to(device)
        losses.append((-model.log_prob(y, context=c)).float().cpu().numpy())
    values = np.concatenate(losses)
    video_losses: list[float] = []
    offset = 0
    for length in lengths:
        video_losses.append(float(values[offset : offset + length].mean()))
        offset += length
    return float(np.mean(video_losses))


def train_one(
    *,
    mode: str,
    seed: int,
    preprocessor: CTNEPreprocessor,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    device: torch.device,
    hidden_features: int,
    num_blocks: int,
    transform_blocks: int,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
) -> dict[str, Any]:
    if mode not in {"matched", "unconditional"}:
        raise ValueError(f"unknown CTNE mode: {mode}")
    _seed_everything(seed)
    train_camera, train_evidence, train_weights, train_lengths = _load_transformed(train_rows, preprocessor)
    val_camera, val_evidence, _, val_lengths = _load_transformed(val_rows, preprocessor)
    if mode == "unconditional":
        train_camera = np.zeros_like(train_camera)
    dataset = TensorDataset(
        torch.from_numpy(train_evidence),
        torch.from_numpy(train_camera),
        torch.from_numpy(train_weights),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0, drop_last=False)
    _seed_everything(seed)
    model = build_flow(
        evidence_dim=preprocessor.evidence_dim,
        context_dim=preprocessor.camera_dim,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        transform_blocks=transform_blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        numerator = 0.0
        denominator = 0.0
        for y_cpu, c_cpu, weight_cpu in loader:
            y = y_cpu.to(device, non_blocking=True)
            context = c_cpu.to(device, non_blocking=True)
            weights = weight_cpu.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            nll = -model.log_prob(y, context=context)
            loss = (nll * weights).sum() / weights.sum().clamp_min(1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            numerator += float((nll.detach() * weights).sum().cpu())
            denominator += float(weights.sum().cpu())
        scheduler.step()
        train_loss = numerator / max(denominator, 1e-8)
        model.eval()
        val_loss = _validation_loss(
            model,
            val_camera,
            val_evidence,
            val_lengths,
            mode=mode,
            device=device,
            batch_size=batch_size,
        )
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise FloatingPointError(
                f"non-finite loss for mode={mode}, seed={seed}, epoch={epoch}: "
                f"train={train_loss}, val={val_loss}"
            )
        item = {
            "epoch": epoch,
            "train_video_weighted_nll": train_loss,
            "val_video_mean_nll": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_sec": time.time() - started,
        }
        history.append(item)
        print(json.dumps({"mode": mode, "seed": seed, **item}), flush=True)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            save_flow(
                model,
                output_dir,
                {
                    "mode": mode,
                    "seed": seed,
                    "evidence_dim": preprocessor.evidence_dim,
                    "context_dim": preprocessor.camera_dim,
                    "hidden_features": hidden_features,
                    "num_blocks": num_blocks,
                    "transform_blocks": transform_blocks,
                    "best_epoch": best_epoch,
                    "best_val_video_mean_nll": best_val,
                    "preprocessor_path": normalize_path(output_dir.parents[2] / "preprocessor.npz"),
                },
            )
        else:
            stale += 1
        if stale >= patience:
            break
    summary = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "mode": mode,
        "seed": seed,
        "status": "completed",
        "output_dir": normalize_path(output_dir),
        "train_real_videos": len(train_rows),
        "val_real_videos": len(val_rows),
        "train_transitions": int(sum(train_lengths)),
        "val_transitions": int(sum(val_lengths)),
        "each_video_total_training_weight": 1.0,
        "best_epoch": best_epoch,
        "best_val_video_mean_nll": best_val,
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
    parser.add_argument("--modes", nargs="+", choices=("matched", "unconditional"), default=["matched", "unconditional"])
    parser.add_argument("--job-index", type=int, default=0)
    parser.add_argument("--job-count", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-preprocessor", action="store_true")
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--max-transitions-per-video", type=int, default=64)
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--transform-blocks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.job_count < 1 or not 0 <= args.job_index < args.job_count:
        raise ValueError("job-index must be in [0, job-count)")
    rows = read_jsonl(args.feature_index_jsonl)
    train_rows = _real_rows(rows, "train")
    val_rows = _real_rows(rows, "val")
    if not train_rows or not val_rows:
        raise ValueError(f"need real train and val videos, found train={len(train_rows)} val={len(val_rows)}")
    preprocessor_path = args.output_dir / "preprocessor.npz"
    preprocessor_summary_path = args.output_dir / "preprocessor_summary.json"
    if args.reuse_preprocessor or preprocessor_path.is_file():
        preprocessor = CTNEPreprocessor.load(preprocessor_path)
    else:
        preprocessor, preprocessing_summary = fit_preprocessor(
            train_rows,
            pca_dim=args.pca_dim,
            max_transitions_per_video=args.max_transitions_per_video,
            seed=min(args.seeds),
        )
        preprocessor.save(preprocessor_path)
        write_json(
            preprocessor_summary_path,
            {
                "schema_version": MODEL_SCHEMA_VERSION,
                "feature_index_jsonl": normalize_path(args.feature_index_jsonl),
                "preprocessor_path": normalize_path(preprocessor_path),
                **preprocessing_summary,
            },
        )
    if args.prepare_only:
        print(f"Prepared shared CTNE preprocessor: {preprocessor_path}")
        return 0
    device = torch.device(args.device)
    jobs = [(mode, seed) for seed in args.seeds for mode in args.modes]
    selected_jobs = [job for index, job in enumerate(jobs) if index % args.job_count == args.job_index]
    summaries = []
    for mode, seed in selected_jobs:
        summaries.append(
            train_one(
                mode=mode,
                seed=seed,
                preprocessor=preprocessor,
                train_rows=train_rows,
                val_rows=val_rows,
                output_dir=args.output_dir / "models" / f"seed_{seed}" / mode,
                device=device,
                hidden_features=args.hidden_features,
                num_blocks=args.num_blocks,
                transform_blocks=args.transform_blocks,
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
            "schema_version": MODEL_SCHEMA_VERSION,
            "job_index": args.job_index,
            "job_count": args.job_count,
            "selected_jobs": [{"mode": mode, "seed": seed} for mode, seed in selected_jobs],
            "summaries": summaries,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
