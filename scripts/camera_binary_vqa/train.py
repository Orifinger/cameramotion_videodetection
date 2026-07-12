#!/usr/bin/env python3
"""Train a balanced binary camera-motion VQA LoRA with a wall-clock guard."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from scripts.camera_binary_vqa.runtime import (
    attach_new_lora,
    cleanup_distributed,
    init_distributed,
    load_model,
    load_processor,
    next_record,
    prepare_sft_batch,
    read_jsonl,
    set_seed,
    write_json,
)


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--max-wall-seconds", type=float, default=21600.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--video-max-pixels", type=int, default=16384)
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--cpu-threads-per-rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_adapter(model: torch.nn.Module, processor: Any, path: Path, state: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    unwrap(model).save_pretrained(path, safe_serialization=True)
    processor.save_pretrained(path)
    write_json(path / "camera_binary_vqa_training_state.json", state)


def synchronized_stop(rank: int, device: torch.device, should_stop: bool) -> bool:
    value = torch.tensor([1 if should_stop else 0], device=device, dtype=torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(value, src=0)
    return bool(int(value.item()))


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = init_distributed()
    torch.set_num_threads(max(1, args.cpu_threads_per_rank))
    set_seed(args.seed, rank)
    device = torch.device("cuda", local_rank)
    rows = read_jsonl(args.train_jsonl)
    if not rows:
        raise ValueError("empty binary camera VQA training set")
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = math.ceil(len(rows) / world_size)
    planned_steps = steps_per_epoch * args.num_epochs
    processor = load_processor(args.model_path)
    model = load_model(args.model_path, args.attn_implementation, torch.bfloat16)
    model = attach_new_lora(
        model, args.lora_rank, args.lora_alpha, args.lora_dropout, LORA_TARGETS
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    warmup_steps = max(1, round(planned_steps * args.warmup_ratio))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, planned_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    model.train()
    started = time.time()
    log_path = output_dir / "trainer_log.jsonl"
    completed_steps = 0
    stop_reason = "planned_steps_completed"

    for step in range(planned_steps):
        elapsed = time.time() - started
        wall_stop = step >= steps_per_epoch and elapsed >= args.max_wall_seconds
        if synchronized_stop(rank, device, rank == 0 and wall_stop):
            stop_reason = "max_wall_seconds_reached"
            break
        sample = next_record(rows, step, rank, world_size, args.seed + 101)
        optimizer.zero_grad(set_to_none=True)
        batch = prepare_sft_batch(
            sample,
            processor,
            device,
            args.video_max_pixels,
            args.video_fps,
        )
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite camera VQA loss at step {step + 1}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
        optimizer.step()
        scheduler.step()
        completed_steps = step + 1

        metrics = torch.tensor(
            [loss.detach().float(), torch.as_tensor(grad_norm, device=device).float()],
            device=device,
        )
        if world_size > 1:
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics /= world_size
        if rank == 0 and (completed_steps == 1 or completed_steps % args.logging_steps == 0):
            log_row = {
                "step": completed_steps,
                "planned_steps": planned_steps,
                "steps_per_epoch": steps_per_epoch,
                "effective_epoch": completed_steps / steps_per_epoch,
                "loss": float(metrics[0]),
                "grad_norm": float(metrics[1]),
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - started,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_row, ensure_ascii=False) + "\n")
            print(json.dumps(log_row, ensure_ascii=False), flush=True)

        if completed_steps == steps_per_epoch:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                save_adapter(
                    model,
                    processor,
                    output_dir / "checkpoint-epoch-1",
                    {
                        "step": completed_steps,
                        "effective_epoch": 1.0,
                        "base_model": args.model_path,
                        "train_jsonl": args.train_jsonl,
                    },
                )
            if world_size > 1:
                dist.barrier()

    if world_size > 1:
        dist.barrier()
    elapsed = time.time() - started
    state = {
        "base_model": args.model_path,
        "train_jsonl": args.train_jsonl,
        "num_train_records": len(rows),
        "world_size": world_size,
        "steps_per_epoch": steps_per_epoch,
        "planned_epochs": args.num_epochs,
        "planned_steps": planned_steps,
        "completed_steps": completed_steps,
        "effective_epochs": completed_steps / steps_per_epoch,
        "effective_records_seen": completed_steps * world_size,
        "stop_reason": stop_reason,
        "max_wall_seconds": args.max_wall_seconds,
        "elapsed_seconds": elapsed,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "video_fps": args.video_fps,
        "video_max_pixels": args.video_max_pixels,
    }
    if rank == 0:
        save_adapter(model, processor, output_dir / "final", state)
        write_json(output_dir / "all_results.json", state)
    cleanup_distributed()


if __name__ == "__main__":
    main()
