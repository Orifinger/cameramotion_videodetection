#!/usr/bin/env python3
"""Train a short camera-motion SFT LoRA on the fixed DataA train identities."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist

from scripts.camera_pretext_transfer.runtime import prepare_sft_batch
from scripts.caspr_gate1.runtime import (
    attach_adapter, attach_new_lora, cleanup_distributed, init_distributed, load_model, load_processor,
    next_record, read_jsonl, set_seed, write_json,
)

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--initial-adapter-path",
        help="Optional camera LoRA to continue training with the same camera objective.",
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=24)
    parser.add_argument("--logging-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_adapter(model: torch.nn.Module, processor: object, path: Path, state: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    unwrap(model).save_pretrained(path, safe_serialization=True)
    processor.save_pretrained(path)
    write_json(path / "camera_sft_state.json", state)


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = init_distributed()
    set_seed(args.seed, rank)
    device = torch.device("cuda", local_rank)
    rows = read_jsonl(args.train_jsonl)
    if not rows:
        raise ValueError("empty camera SFT training set")
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    processor = load_processor(args.model_path)
    model = load_model(args.model_path, args.attn_implementation, torch.bfloat16)
    if args.initial_adapter_path:
        model = attach_adapter(model, args.initial_adapter_path, is_trainable=True)
    else:
        model = attach_new_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout, LORA_TARGETS)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_steps = max(1, round(args.max_steps * args.warmup_ratio))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, args.max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )
    model.train()
    started = time.time()
    log_path = output_dir / "trainer_log.jsonl"
    for step in range(args.max_steps):
        sample = next_record(rows, step, rank, world_size, args.seed + 101)
        optimizer.zero_grad(set_to_none=True)
        batch = prepare_sft_batch([sample], processor, device, args.max_pixels)
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite camera SFT loss at step {step + 1}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
        optimizer.step()
        scheduler.step()
        metrics = torch.tensor([loss.detach().float(), torch.as_tensor(grad_norm, device=device).float()], device=device)
        if world_size > 1:
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics /= world_size
        if rank == 0 and ((step + 1) % args.logging_steps == 0 or step == 0):
            row = {
                "step": step + 1, "max_steps": args.max_steps, "loss": float(metrics[0]),
                "grad_norm": float(metrics[1]), "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - started,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if args.save_steps > 0 and (step + 1) % args.save_steps == 0:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                save_adapter(model, processor, output_dir / f"checkpoint-{step + 1}", {
                    "step": step + 1, "base_model": args.model_path, "train_jsonl": args.train_jsonl,
                    "world_size": world_size, "initial_adapter_path": args.initial_adapter_path,
                })
            if world_size > 1:
                dist.barrier()
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        state = {
            "base_model": args.model_path, "train_jsonl": args.train_jsonl,
            "initial_adapter_path": args.initial_adapter_path,
            "num_train_records": len(rows), "max_steps": args.max_steps, "world_size": world_size,
            "effective_records_seen": args.max_steps * world_size,
            "elapsed_seconds": time.time() - started,
        }
        save_adapter(model, processor, output_dir / "final", state)
        write_json(output_dir / "all_results.json", state)
    cleanup_distributed()


if __name__ == "__main__":
    main()
