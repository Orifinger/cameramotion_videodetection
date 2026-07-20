#!/usr/bin/env python3
"""Score Real/Fake answer tokens for archived ViF-Bench Qwen responses.

The script rebuilds the deployed ViF-Bench prompt through its own ViFBench
class, teacher-forces the archived response prefix up to ``<answer>``, and
records the two candidate logits. It does not regenerate or replace the
historical hard prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ANSWER_RE = re.compile(r"<answer>\s*(Real|Fake)\b", re.IGNORECASE)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def normalize_path(value: Any) -> str:
    return re.sub(r"/+", "/", str(value or "").strip().replace("\\", "/"))


def canonical_video_id(value: Any) -> str:
    text = normalize_path(value).lstrip("/")
    lowered = text.casefold()
    for marker in ("/parsed_frames/parsed_frames/", "/test_normalized/"):
        if marker in lowered:
            text = text[lowered.index(marker) + len(marker) :]
            break
    parts = text.split("/")
    if len(parts) >= 3 and parts[0].casefold() in {"real", "fake"}:
        parts = parts[1:]
    if len(parts) < 2:
        raise ValueError(f"cannot derive video ID from {value!r}")
    if parts[0].casefold() == "real":
        parts[0] = "real"
    return "/".join(parts)


def load_historical_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("historical prediction JSON must contain a list")
    rows: dict[str, dict[str, Any]] = {}
    for raw in payload:
        if not isinstance(raw, Mapping):
            continue
        video_id = canonical_video_id(raw.get("video_id"))
        if video_id in rows:
            raise ValueError(f"duplicate historical video ID: {video_id}")
        rows[video_id] = dict(raw)
    if not rows:
        raise ValueError(f"no historical predictions loaded from {path}")
    return rows


def answer_token_contract(tokenizer: Any, response: str) -> dict[str, Any]:
    """Locate the single-token Real/Fake substitution in an archived response."""
    match = ANSWER_RE.search(response)
    if not match:
        return {"valid": False, "reason": "missing_canonical_answer_tag"}
    archived_answer = match.group(1).capitalize()
    alternate_answer = "Fake" if archived_answer == "Real" else "Real"
    counterfactual = response[: match.start(1)] + alternate_answer + response[match.end(1) :]
    actual_ids = tokenizer.encode(response, add_special_tokens=False)
    alternate_ids = tokenizer.encode(counterfactual, add_special_tokens=False)
    common_prefix = 0
    for actual, alternate in zip(actual_ids, alternate_ids):
        if actual != alternate:
            break
        common_prefix += 1
    contract_valid = (
        common_prefix < len(actual_ids)
        and common_prefix < len(alternate_ids)
        and len(actual_ids) == len(alternate_ids)
        and actual_ids[common_prefix + 1 :] == alternate_ids[common_prefix + 1 :]
    )
    if not contract_valid:
        return {
            "valid": False,
            "reason": "real_fake_substitution_is_not_one_token",
            "archived_answer": archived_answer,
            "actual_token_count": len(actual_ids),
            "alternate_token_count": len(alternate_ids),
            "common_prefix_tokens": common_prefix,
        }
    actual_token_id = int(actual_ids[common_prefix])
    alternate_token_id = int(alternate_ids[common_prefix])
    real_token_id = actual_token_id if archived_answer == "Real" else alternate_token_id
    fake_token_id = actual_token_id if archived_answer == "Fake" else alternate_token_id
    return {
        "valid": True,
        "archived_answer": archived_answer,
        "answer_prefix_token_count": common_prefix,
        "real_token_id": real_token_id,
        "fake_token_id": fake_token_id,
        "real_token_text": tokenizer.decode([real_token_id]),
        "fake_token_text": tokenizer.decode([fake_token_id]),
        "response_token_ids": actual_ids,
    }


def build_user_content(frame_paths: Sequence[str], user_prompt: str) -> list[dict[str, Any]]:
    normalized = [path.replace("file://", "", 1) if path.startswith("file://") else path for path in frame_paths]
    frame_iter = iter(normalized)
    content: list[dict[str, Any]] = []
    parts = re.split(r"<image>", user_prompt)
    for index, text_part in enumerate(parts):
        if text_part.strip():
            content.append({"type": "text", "text": text_part})
        if index < len(parts) - 1:
            try:
                content.append({"type": "image", "image": next(frame_iter)})
            except StopIteration as exc:
                raise ValueError("more <image> placeholders than frame paths") from exc
    try:
        next(frame_iter)
    except StopIteration:
        return content
    raise ValueError("more frame paths than <image> placeholders")


def load_prompt_bench(v4train_eval_dir: Path, index_json: Path, scratch_dir: Path) -> Any:
    sys.path.insert(0, str(v4train_eval_dir))
    from utils.ViFBench import ViFBench  # type: ignore

    class PromptOnlyBench(ViFBench):
        def load_model(self) -> None:
            self.model = None

        def run_inference(self, frame_paths: list[str], user_prompt: str) -> str:
            raise RuntimeError("PromptOnlyBench does not run generation")

    bench = PromptOnlyBench(
        index_json=str(index_json),
        model_path="prompt-only",
        model_name=f"prompt-contract-{index_json.stem}",
        save_dir=str(scratch_dir),
    )
    bench._load_data()
    return bench


def prompt_contract_hash(bench: Any) -> str:
    payload = f"{bench.SYSTEM_PROMPT}\0{bench.user_prompt_suffix}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoProcessor

    eval_dir = Path(args.v4train_eval_dir)
    required = [
        eval_dir / "utils" / "ViFBench.py",
        eval_dir / "models" / "Qwen3_VL.py",
        Path(args.index_json),
        Path(args.model_path) / "config.json",
        Path(args.historical_predictions),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")
    history = load_historical_predictions(args.historical_predictions)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    examples = {
        answer: answer_token_contract(
            processor.tokenizer,
            f"<think>audit</think>\n<answer>{answer}</answer>",
        )
        for answer in ("Real", "Fake")
    }
    if not all(item.get("valid") for item in examples.values()):
        raise RuntimeError(f"Real/Fake tokenizer contract failed: {examples}")
    bench = load_prompt_bench(eval_dir, Path(args.index_json), Path(args.output_path).parent / "prompt_audit")
    first_task = bench.all_tasks[0] if bench.all_tasks else None
    if first_task is None:
        raise RuntimeError(f"no ViF-Bench tasks loaded from {args.index_json}")
    user_prompt, frames = bench._build_user_prompt(first_task["frame_dir_path"], first_task)
    if not user_prompt or not frames:
        raise RuntimeError("first ViF-Bench task could not build its frame prompt")
    return {
        "status": "passed",
        "historical_prediction_rows": len(history),
        "rank_tasks": len(bench.all_tasks),
        "first_task_video_id": first_task["video_id"],
        "first_task_frames": len(frames),
        "prompt_contract_sha256": prompt_contract_hash(bench),
        "token_contract_examples": {
            key: {field: value for field, value in item.items() if field != "response_token_ids"}
            for key, item in examples.items()
        },
    }


def load_model_and_processor(model_path: str, attn_implementation: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor, model.device


def score_one(
    *,
    task: Mapping[str, Any],
    historical: Mapping[str, Any],
    bench: Any,
    model: Any,
    processor: Any,
    device: Any,
) -> dict[str, Any]:
    import torch

    response = str(historical.get("response", ""))
    contract = answer_token_contract(processor.tokenizer, response)
    if not contract.get("valid"):
        return {
            "status": "invalid_answer_token_contract",
            "error": contract.get("reason"),
            "answer_token_contract": contract,
        }
    user_prompt, frame_paths = bench._build_user_prompt(task["frame_dir_path"], task)
    if not user_prompt or not frame_paths:
        return {"status": "invalid_frame_prompt", "error": "could_not_build_frame_prompt"}
    messages = [
        {"role": "system", "content": [{"type": "text", "text": bench.SYSTEM_PROMPT}]},
        {"role": "user", "content": build_user_content(frame_paths, user_prompt)},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    response_ids = contract.pop("response_token_ids")
    prefix_count = int(contract["answer_prefix_token_count"])
    answer_prefix = torch.tensor(
        [response_ids[:prefix_count]], dtype=inputs["input_ids"].dtype
    )
    inputs["input_ids"] = torch.cat([inputs["input_ids"], answer_prefix], dim=1)
    if "attention_mask" in inputs:
        prefix_mask = torch.ones(
            (inputs["attention_mask"].shape[0], prefix_count),
            dtype=inputs["attention_mask"].dtype,
        )
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], prefix_mask], dim=1)
    inputs = inputs.to(device)
    forward_kwargs = dict(inputs)
    forward_kwargs["use_cache"] = False
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        forward_kwargs["logits_to_keep"] = 1
    with torch.inference_mode():
        output = model(**forward_kwargs)
    logits = output.logits[0, -1].float()
    real_logit = float(logits[int(contract["real_token_id"])].item())
    fake_logit = float(logits[int(contract["fake_token_id"])].item())
    pair_probability = torch.softmax(
        torch.tensor([real_logit, fake_logit], dtype=torch.float64), dim=0
    )
    fake_probability = float(pair_probability[1].item())
    scored_answer = "Fake" if fake_logit > real_logit else "Real"
    return {
        "status": "ok",
        "num_frames": len(frame_paths),
        "prompt_tokens": int(inputs["input_ids"].shape[1] - prefix_count),
        "answer_prefix_tokens": prefix_count,
        "real_logit": real_logit,
        "fake_logit": fake_logit,
        "fake_minus_real_logit_margin": fake_logit - real_logit,
        "fake_pair_probability": fake_probability,
        "scored_answer": scored_answer,
        "score_matches_archived_answer": scored_answer == contract["archived_answer"],
        "answer_token_contract": contract,
    }


def run(args: argparse.Namespace) -> None:
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        result = preflight(args)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    history = load_historical_predictions(args.historical_predictions)
    bench = load_prompt_bench(
        Path(args.v4train_eval_dir), Path(args.index_json), output_path.parent / "prompt_cache"
    )
    if args.max_samples > 0:
        bench.all_tasks = bench.all_tasks[: args.max_samples]
    processed: set[str] = set()
    if output_path.is_file() and not args.overwrite:
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    processed.add(str(json.loads(line).get("video_id", "")))
    mode = "w" if args.overwrite else "a"
    model, processor, device = load_model_and_processor(args.model_path, args.attn_implementation)
    started = time.time()
    written = 0
    with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for task in bench.all_tasks:
            video_id = canonical_video_id(task["video_id"])
            if video_id in processed:
                continue
            row: dict[str, Any] = {
                "video_id": video_id,
                "aigc_model_name": task["aigc_model_name"],
                "gt": task["gt"],
                "archived_answer": None,
                "prompt_contract_sha256": prompt_contract_hash(bench),
                "historical_predictions": str(args.historical_predictions),
            }
            historical = history.get(video_id)
            sample_started = time.time()
            if historical is None:
                row.update({"status": "missing_historical_prediction"})
            else:
                row["archived_answer"] = historical.get("answer")
                try:
                    row.update(
                        score_one(
                            task=task,
                            historical=historical,
                            bench=bench,
                            model=model,
                            processor=processor,
                            device=device,
                        )
                    )
                except Exception as exc:
                    row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            row["elapsed_seconds"] = time.time() - sample_started
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            if written % 10 == 0:
                elapsed = time.time() - started
                print(
                    f"pid={os.getpid()} scored={written} elapsed={elapsed:.1f}s "
                    f"last={video_id}",
                    flush=True,
                )
    print(f"saved={output_path} new_rows={written}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--historical-predictions", required=True)
    parser.add_argument("--v4train-eval-dir", required=True)
    parser.add_argument("--index-json", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
