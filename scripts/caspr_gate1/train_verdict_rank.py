#!/usr/bin/env python3
"""Train the matched CASPR Gate 1 verdict control or pair-ranking adapter."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as functional

from scripts.caspr_gate1.runtime import (
    attach_adapter,
    attach_new_lora,
    binary_verdict_loss,
    candidate_token_ids,
    cleanup_distributed,
    init_distributed,
    load_model,
    load_processor,
    next_record,
    prepare_batch,
    read_jsonl,
    set_seed,
    verdict_scores,
    write_json,
)

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--initial-adapter-path",
        help="Optional camera-pretext LoRA to continue training instead of creating a new LoRA.",
    )
    parser.add_argument("--train-pairs-jsonl", required=True)
    parser.add_argument("--datab-replay-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("control", "pair_rank"), required=True)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--pair-loss-weight", type=float, default=0.2)
    parser.add_argument("--pair-margin", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_adapter(model: torch.nn.Module, processor: object, path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    unwrap(model).save_pretrained(path, safe_serialization=True)
    processor.save_pretrained(path)
    write_json(path / "caspr_training_state.json", payload)


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = init_distributed()
    set_seed(args.seed, rank)
    device = torch.device("cuda", local_rank)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    pairs = read_jsonl(args.train_pairs_jsonl)
    replay = read_jsonl(args.datab_replay_jsonl)
    processor = load_processor(args.model_path)
    tokenizer = processor.tokenizer
    real_token_id, fake_token_id = candidate_token_ids(tokenizer)
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
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, args.max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )
    model.train()
    log_path = output_dir / "trainer_log.jsonl"
    pair_draw = replay_draw = 0
    started = time.time()

    for step in range(args.max_steps):
        is_pair_step = step % 2 == 0
        optimizer.zero_grad(set_to_none=True)
        if is_pair_step:
            record = next_record(pairs, pair_draw, rank, world_size, args.seed + 101)
            pair_draw += 1
            samples = [record["real"], record["fake"]]
            batch = prepare_batch(samples, processor, device, args.max_pixels)
            outputs = model(**batch, use_cache=False)
            scores = verdict_scores(outputs.logits, real_token_id, fake_token_id)
            binary_loss = binary_verdict_loss(scores, ["Real", "Fake"])
            pair_loss = functional.softplus(args.pair_margin - (scores[1] - scores[0]))
            loss = binary_loss
            if args.mode == "pair_rank":
                loss = loss + args.pair_loss_weight * pair_loss
            real_score, fake_score = scores[0].detach(), scores[1].detach()
        else:
            sample = next_record(replay, replay_draw, rank, world_size, args.seed + 202)
            replay_draw += 1
            batch = prepare_batch([sample], processor, device, args.max_pixels)
            outputs = model(**batch, use_cache=False)
            scores = verdict_scores(outputs.logits, real_token_id, fake_token_id)
            binary_loss = binary_verdict_loss(scores, [str(sample["label"])])
            pair_loss = torch.zeros((), device=device)
            loss = binary_loss
            real_score = fake_score = torch.tensor(float("nan"), device=device)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
        optimizer.step()
        scheduler.step()

        metrics = torch.tensor(
            [
                loss.detach().float(),
                binary_loss.detach().float(),
                pair_loss.detach().float(),
                real_score.float(),
                fake_score.float(),
                torch.as_tensor(grad_norm, device=device).float(),
            ],
            device=device,
        )
        if world_size > 1:
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics /= world_size
        if rank == 0 and ((step + 1) % args.logging_steps == 0 or step == 0):
            row = {
                "step": step + 1,
                "max_steps": args.max_steps,
                "step_kind": "dataa_pair" if is_pair_step else "datab_replay",
                "mode": args.mode,
                "loss": float(metrics[0]),
                "binary_loss": float(metrics[1]),
                "pair_loss": float(metrics[2]),
                "mean_real_score": None if not is_pair_step else float(metrics[3]),
                "mean_fake_score": None if not is_pair_step else float(metrics[4]),
                "grad_norm": float(metrics[5]),
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - started,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)

        should_save = args.save_steps > 0 and (step + 1) % args.save_steps == 0
        if should_save:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                save_adapter(
                    model,
                    processor,
                    output_dir / f"checkpoint-{step + 1}",
                    {
                        "step": step + 1,
                        "mode": args.mode,
                        "base_model": args.model_path,
                        "initial_adapter_path": args.initial_adapter_path,
                    },
                )
            if world_size > 1:
                dist.barrier()

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        state = {
            "mode": args.mode,
            "base_model": args.model_path,
            "initial_adapter_path": args.initial_adapter_path,
            "train_pairs_jsonl": args.train_pairs_jsonl,
            "datab_replay_jsonl": args.datab_replay_jsonl,
            "max_steps": args.max_steps,
            "world_size": world_size,
            "pair_steps": (args.max_steps + 1) // 2,
            "replay_steps": args.max_steps // 2,
            "pair_loss_weight": args.pair_loss_weight if args.mode == "pair_rank" else 0.0,
            "pair_margin": args.pair_margin,
            "candidate_token_ids": {"Real": real_token_id, "Fake": fake_token_id},
            "elapsed_seconds": time.time() - started,
        }
        save_adapter(model, processor, output_dir / "final", state)
        write_json(output_dir / "all_results.json", state)
    cleanup_distributed()


if __name__ == "__main__":
    main()
