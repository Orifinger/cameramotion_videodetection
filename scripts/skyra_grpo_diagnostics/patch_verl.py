#!/usr/bin/env python3
"""Apply idempotent Qwen3-VL and diagnostic-metric fixes to verl 2c9e19e."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


QWEN_MARKER = "# SKYRA_GRPO_DIAGNOSTICS_QWEN3_IMAGE_PROCESSOR"
CHUNK_MARKER = "# SKYRA_GRPO_DIAGNOSTICS_PPU_CHUNKED_PREFILL"
METRIC_MARKER = "# SKYRA_GRPO_DIAGNOSTICS_REWARD_METRICS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-root", required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-json")
    return parser.parse_args()


def backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".skyra_diag.bak")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def patch_dataset(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if QWEN_MARKER in text:
        return False
    old = '        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:\n'
    if old not in text:
        raise RuntimeError("Qwen image-processor condition was not found; verl source does not match the expected commit")
    new = (
        f"        {QWEN_MARKER}\n"
        "        if self.processor is not None and any(\n"
        "            name in self.processor.image_processor.__class__.__name__\n"
        '            for name in ("Qwen2VLImageProcessor", "Qwen3VLImageProcessor")\n'
        "        ):\n"
    )
    backup(path)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def patch_rollout(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if CHUNK_MARKER in text:
        return False
    old = "        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:\n"
    if old not in text:
        raise RuntimeError("chunked-prefill guard was not found; verl source does not match the expected commit")
    new = (
        f"        {CHUNK_MARKER}\n"
        "        if max_num_batched_tokens < max_model_len and not self.config.enable_chunked_prefill:\n"
    )
    backup(path)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def patch_metrics(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if METRIC_MARKER in text:
        return False
    old = (
        "                        if reward_extra_infos_dict:\n"
        "                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})\n"
    )
    if old not in text:
        raise RuntimeError("reward-extra insertion point was not found; verl source does not match the expected commit")
    new = old + (
        f"\n                        {METRIC_MARKER}\n"
        "                        for reward_key, reward_values in reward_extra_infos_dict.items():\n"
        "                            try:\n"
        "                                reward_array = np.asarray(reward_values, dtype=np.float64)\n"
        "                            except (TypeError, ValueError):\n"
        "                                continue\n"
        "                            reward_array = reward_array[np.isfinite(reward_array)]\n"
        "                            if reward_array.size:\n"
        "                                metrics[f\"reward_extra/{reward_key}/mean\"] = float(reward_array.mean())\n"
        "                                metrics[f\"reward_extra/{reward_key}/std\"] = float(reward_array.std())\n"
        "\n"
        "                        group_scores = reward_tensor.sum(dim=-1).detach().float().cpu().numpy()\n"
        "                        grouped_scores = defaultdict(list)\n"
        "                        for group_uid, group_score in zip(batch.non_tensor_batch[\"uid\"], group_scores, strict=True):\n"
        "                            grouped_scores[group_uid].append(float(group_score))\n"
        "                        group_stds = np.asarray(\n"
        "                            [np.std(values) for values in grouped_scores.values()], dtype=np.float64\n"
        "                        )\n"
        "                        if group_stds.size:\n"
        "                            metrics[\"reward_extra/grpo_zero_std_group_rate\"] = float(\n"
        "                                np.mean(group_stds <= 1e-12)\n"
        "                            )\n"
        "                            metrics[\"reward_extra/grpo_group_reward_std_mean\"] = float(group_stds.mean())\n"
    )
    backup(path)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    root = Path(args.verl_root).resolve()
    paths = {
        "dataset": root / "verl/utils/dataset/rl_dataset.py",
        "rollout": root / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py",
        "trainer": root / "verl/trainer/ppo/ray_trainer.py",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    changed = {}
    if not args.check:
        changed = {
            "dataset": patch_dataset(paths["dataset"]),
            "rollout": patch_rollout(paths["rollout"]),
            "trainer": patch_metrics(paths["trainer"]),
        }

    checks = {
        "qwen3_image_processor": QWEN_MARKER in paths["dataset"].read_text(encoding="utf-8"),
        "ppu_chunked_prefill": CHUNK_MARKER in paths["rollout"].read_text(encoding="utf-8"),
        "reward_metrics": METRIC_MARKER in paths["trainer"].read_text(encoding="utf-8"),
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "verl_root": str(root),
        "changed": changed,
        "checks": checks,
    }
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
