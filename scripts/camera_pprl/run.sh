#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${STAGE:-preflight}"
PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/input/workflow_58770161/workspace/test/ms_swift/ms-swift-main}"
SWIFT_BIN="${SWIFT_BIN:-swift}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_GPUS="${NUM_GPUS:-16}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

BASE_MODEL="${BASE_MODEL:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
JOINT_ROOT="${JOINT_ROOT:-/tmp/1res/camera_joint_sft_gate}"
CORRECT_SFT_ADAPTER="${CORRECT_SFT_ADAPTER:-${JOINT_ROOT}/train/correct_camera}"
JOINT_DATA_ROOT="${JOINT_DATA_ROOT:-${JOINT_ROOT}/data}"
WORK_ROOT="${WORK_ROOT:-/tmp/1res/camera_pprl/correct_camera_1024}"
PERSIST_ROOT="${PERSIST_ROOT:-${PROJECT_ROOT}/res/camera_pprl/correct_camera_1024}"
OSS_ROOT="${OSS_ROOT:-oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_pprl/correct_camera_1024}"

DATA_ROOT="${WORK_ROOT}/data"
MODEL_ROOT="${WORK_ROOT}/models"
TRAIN_ROOT="${WORK_ROOT}/train"
CAMERA_PRED_ROOT="${WORK_ROOT}/camera_predictions"
CAMERA_EVAL_ROOT="${WORK_ROOT}/camera_eval"
VIF_DIRECT_ROOT="${WORK_ROOT}/vif_direct"
VIF_RECOVERY_ROOT="${WORK_ROOT}/vif_recovery"
ARTIFACT_ROOT="${WORK_ROOT}/artifacts"
PREFLIGHT_ROOT="${WORK_ROOT}/preflight"
PIPELINE_LOG="${WORK_ROOT}/pipeline.log"

SFT_MERGED_MODEL="${SFT_MERGED_MODEL:-${MODEL_ROOT}/correct_camera_joint_sft_merged}"
PPRL_OUTPUT="${PPRL_OUTPUT:-${TRAIN_ROOT}/camera_pprl}"
PPRL_ADAPTER="${PPRL_ADAPTER:-${ARTIFACT_ROOT}/camera_pprl_adapter}"
PPRL_MERGED_MODEL="${PPRL_MERGED_MODEL:-${MODEL_ROOT}/camera_pprl_merged}"
RECOVERY_OUTPUT="${RECOVERY_OUTPUT:-${TRAIN_ROOT}/detection_recovery}"
RECOVERY_ADAPTER="${RECOVERY_ADAPTER:-${ARTIFACT_ROOT}/detection_recovery_adapter}"
RECOVERY_MERGED_MODEL="${RECOVERY_MERGED_MODEL:-${MODEL_ROOT}/camera_pprl_detection_recovery_merged}"
WARM_COMPACT_ADAPTER="${ARTIFACT_ROOT}/correct_camera_joint_sft_adapter"

PPRL_DATA="${DATA_ROOT}/camera_pprl_train_1024.json"
PPRL_SMOKE_DATA="${DATA_ROOT}/camera_pprl_smoke_32.json"
PPRL_DATA_SUMMARY="${DATA_ROOT}/camera_pprl_data_summary.json"
CAMERA_TRAIN_SOURCE="${JOINT_DATA_ROOT}/camera_train_correct.json"
CAMERA_DEV_MATCHED="${JOINT_DATA_ROOT}/camera_dev_matched_frames.jsonl"
CAMERA_DEV_OPPOSITE="${JOINT_DATA_ROOT}/camera_dev_opposite_frames.jsonl"
CAMERA_DEV_NO_FRAMES="${JOINT_DATA_ROOT}/camera_dev_no_frames.jsonl"

DATAA_DETECTION_JSON="${DATAA_DETECTION_JSON:-${PROJECT_ROOT}/res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json}"
DATAA_CAMERA_JSONL="${DATAA_CAMERA_JSONL:-${PROJECT_ROOT}/camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl}"
DATAB_DETECTION_JSON="${DATAB_DETECTION_JSON:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json}"
DATAB_CAMERA_JSONL="${DATAB_CAMERA_JSONL:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl}"
DETECTION_ONLY_VIF_REFERENCE="${DETECTION_ONLY_VIF_REFERENCE:-${PROJECT_ROOT}/res/camera_joint_sft_gate/vif_four_model_compare/branches/detection-only/eval/camera_adapter_vifbench_eval.json}"

PPRL_RECORDS="${PPRL_RECORDS:-1024}"
PPRL_SMOKE_RECORDS="${PPRL_SMOKE_RECORDS:-32}"
PPRL_LORA_RANK="${PPRL_LORA_RANK:-32}"
PPRL_LORA_ALPHA="${PPRL_LORA_ALPHA:-64}"
PPRL_LR="${PPRL_LR:-1e-6}"
PPRL_EPOCHS="${PPRL_EPOCHS:-1.0}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
PPRL_TEMPERATURE="${PPRL_TEMPERATURE:-1.0}"
PPRL_BETA="${PPRL_BETA:-0.04}"
PPRL_MAX_RESAMPLE_TIMES="${PPRL_MAX_RESAMPLE_TIMES:-3}"
PPRL_SMOKE_MAX_ZERO_STD_RATE="${PPRL_SMOKE_MAX_ZERO_STD_RATE:-0.80}"
RECOVERY_LORA_RANK="${RECOVERY_LORA_RANK:-16}"
RECOVERY_LORA_ALPHA="${RECOVERY_LORA_ALPHA:-32}"
RECOVERY_LR="${RECOVERY_LR:-5e-6}"
RECOVERY_EPOCHS="${RECOVERY_EPOCHS:-0.5}"
MAX_LENGTH="${MAX_LENGTH:-49152}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
VLLM_TP="${VLLM_TP:-4}"
VLLM_MEMORY="${VLLM_MEMORY:-0.45}"
VLLM_ENABLE_LORA="${VLLM_ENABLE_LORA:-1}"

AUTO_UPLOAD_OSS="${AUTO_UPLOAD_OSS:-1}"
KEEP_ALIVE_AFTER_RUN="${KEEP_ALIVE_AFTER_RUN:-0}"
KEEP_ALIVE_SCRIPT="${KEEP_ALIVE_SCRIPT:-/input/training/keep.sh}"
REBUILD_DATA="${REBUILD_DATA:-0}"
REBUILD_MERGED="${REBUILD_MERGED:-0}"
RETRAIN_PPRL="${RETRAIN_PPRL:-0}"
RETRAIN_RECOVERY="${RETRAIN_RECOVERY:-0}"

mkdir -p "${WORK_ROOT}" "${PERSIST_ROOT}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

export PYTHONPATH="${REPO_ROOT}:${MS_SWIFT_ROOT}:${PYTHONPATH:-}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing file: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing directory: $1" >&2
    exit 2
  fi
}

adapter_check() {
  require_dir "$1"
  require_file "$1/adapter_config.json"
  if [[ ! -f "$1/adapter_model.safetensors" && ! -f "$1/adapter_model.bin" ]]; then
    echo "Missing adapter weights under $1" >&2
    exit 2
  fi
}

help_has() {
  local help_file="$1"
  local option="$2"
  grep -q -- "${option}" "${help_file}"
}

training_type_option() {
  local help_file="$1"
  if help_has "${help_file}" "--train_type"; then
    echo "--train_type"
  elif help_has "${help_file}" "--tuner_type"; then
    echo "--tuner_type"
  else
    echo "Neither --train_type nor --tuner_type is available in ${help_file}" >&2
    exit 2
  fi
}

persist_small_results() {
  mkdir -p "${PERSIST_ROOT}"
  if [[ -d "${PREFLIGHT_ROOT}" ]]; then
    mkdir -p "${PERSIST_ROOT}/preflight"
    find "${PREFLIGHT_ROOT}" -maxdepth 1 -type f -exec cp -a {} "${PERSIST_ROOT}/preflight/" \;
  fi
  for file in \
    "${PPRL_DATA}" \
    "${PPRL_SMOKE_DATA}" \
    "${PPRL_DATA_SUMMARY}" \
    "${WORK_ROOT}/smoke/smoke_reward_audit.json" \
    "${CAMERA_EVAL_ROOT}/warm_joint_sft.json" \
    "${CAMERA_EVAL_ROOT}/camera_pprl.json" \
    "${CAMERA_EVAL_ROOT}/detection_recovery.json" \
    "${WORK_ROOT}/camera_pprl_final_summary.json"
  do
    if [[ -f "${file}" ]]; then
      cp -a "${file}" "${PERSIST_ROOT}/"
    fi
  done
  for stage in camera_pprl detection_recovery; do
    local source_dir="${TRAIN_ROOT}/${stage}"
    local target_dir="${PERSIST_ROOT}/train/${stage}"
    if [[ -d "${source_dir}" ]]; then
      mkdir -p "${target_dir}"
      find "${source_dir}" -maxdepth 2 -type f \
        \( -name 'trainer_log.jsonl' -o -name 'all_results.json' -o -name 'logging.jsonl' \
           -o -name 'train_results.json' -o -name 'args.json' \) \
        -exec cp -a {} "${target_dir}/" \;
    fi
  done
  cp -a "${PIPELINE_LOG}" "${PERSIST_ROOT}/" 2>/dev/null || true
}

upload_artifact() {
  local source="$1"
  local name="$2"
  local marker="${WORK_ROOT}/.oss_uploaded_${name}"
  if [[ "${AUTO_UPLOAD_OSS}" != "1" || ! -d "${source}" ]]; then
    return
  fi
  if [[ -f "${marker}" ]]; then
    echo "Reusable artifact already uploaded: ${OSS_ROOT}/${name}/"
    return
  fi
  command -v ossutil64 >/dev/null
  echo "Uploading reusable artifact: ${source} -> ${OSS_ROOT}/${name}/"
  ossutil64 cp -r "${source}/" "${OSS_ROOT}/${name}/"
  touch "${marker}"
}

archive_on_exit() {
  local status=$?
  trap - EXIT
  set +e
  persist_small_results
  if [[ -f "${PPRL_ADAPTER}/adapter_config.json" ]]; then
    upload_artifact "${PPRL_ADAPTER}" "camera_pprl_adapter"
  fi
  if [[ -f "${RECOVERY_ADAPTER}/adapter_config.json" ]]; then
    upload_artifact "${RECOVERY_ADAPTER}" "detection_recovery_adapter"
  fi
  echo "Pipeline exit status: ${status}"
  echo "Persistent small results: ${PERSIST_ROOT}"
  echo "OSS root: ${OSS_ROOT}/"
  exit "${status}"
}
trap archive_on_exit EXIT

preflight() {
  require_dir "${MS_SWIFT_ROOT}"
  require_dir "${BASE_MODEL}"
  require_file "${BASE_MODEL}/config.json"
  adapter_check "${CORRECT_SFT_ADAPTER}"
  require_file "${DATAA_DETECTION_JSON}"
  require_file "${DATAA_CAMERA_JSONL}"
  require_file "${DATAB_DETECTION_JSON}"
  require_file "${DATAB_CAMERA_JSONL}"
  require_file "rl/camera_detection_rewards.py"
  require_file "tools/build_camera_pprl_binary.py"
  require_file "tools/audit_camera_pprl_smoke.py"
  require_file "scripts/camera_detection_retention/run_vifbench.sh"
  require_file "${KEEP_ALIVE_SCRIPT}"
  command -v "${SWIFT_BIN}" >/dev/null
  command -v torchrun >/dev/null
  command -v ossutil64 >/dev/null
  mkdir -p "${PREFLIGHT_ROOT}"
  (
    cd "${MS_SWIFT_ROOT}"
    "${SWIFT_BIN}" rlhf --help > "${PREFLIGHT_ROOT}/swift_rlhf_help.txt"
    "${SWIFT_BIN}" sft --help > "${PREFLIGHT_ROOT}/swift_sft_help.txt"
  )
  for option in --rlhf_type --external_plugins --reward_funcs --reward_weights \
    --num_generations --use_vllm --vllm_mode --vllm_tensor_parallel_size \
    --freeze_vit --freeze_aligner --loss_type --sleep_level \
    --dynamic_sample --max_resample_times
  do
    if ! help_has "${PREFLIGHT_ROOT}/swift_rlhf_help.txt" "${option}"; then
      echo "Current ms-swift rlhf CLI is missing required option: ${option}" >&2
      exit 2
    fi
  done
  training_type_option "${PREFLIGHT_ROOT}/swift_rlhf_help.txt" >/dev/null
  training_type_option "${PREFLIGHT_ROOT}/swift_sft_help.txt" >/dev/null
  "${PYTHON_BIN}" - <<PY
import importlib.metadata
import json
import os
from pathlib import Path

import torch
import transformers
import vllm
import swift
from rl.camera_detection_rewards import orms

assert torch.cuda.device_count() == ${NUM_GPUS}, (torch.cuda.device_count(), ${NUM_GPUS})
assert ${NUM_GPUS} % ${VLLM_TP} == 0, (${NUM_GPUS}, ${VLLM_TP})
assert (${NUM_GPUS} * 1 * 1) % ${NUM_GENERATIONS} == 0, (${NUM_GPUS}, ${NUM_GENERATIONS})
assert "camera_binary_acc" in orms
assert "camera_binary_format" in orms
payload = {
    "status": "passed",
    "swift_file": swift.__file__,
    "swift_distribution_version": None,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "vllm": getattr(vllm, "__version__", "unknown"),
    "gpus": torch.cuda.device_count(),
    "base_model": "${BASE_MODEL}",
    "correct_sft_adapter": "${CORRECT_SFT_ADAPTER}",
    "vllm_tensor_parallel_size": ${VLLM_TP},
    "num_generations": ${NUM_GENERATIONS},
    "detection_only_vif_reference": "${DETECTION_ONLY_VIF_REFERENCE}",
    "detection_only_vif_reference_present": Path("${DETECTION_ONLY_VIF_REFERENCE}").is_file(),
}
try:
    payload["swift_distribution_version"] = importlib.metadata.version("ms-swift")
except importlib.metadata.PackageNotFoundError:
    pass
Path("${PREFLIGHT_ROOT}/environment_audit.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

dataset = Path("${DATAB_DETECTION_JSON}")
with dataset.open("r", encoding="utf-8-sig") as handle:
    rows = json.load(handle)
if not isinstance(rows, list) or not rows:
    raise ValueError(f"DataB detection replay is not a non-empty JSON list: {dataset}")
image_refs = [
    str(path)
    for row in rows
    if isinstance(row, dict)
    for path in row.get("images", [])
]
missing = [path for path in image_refs if not Path(path).is_file()]
audit = {
    "status": "passed" if not missing else "failed",
    "dataset": str(dataset),
    "records": len(rows),
    "image_references": len(image_refs),
    "missing_image_references": len(missing),
    "first_missing": missing[:20],
}
Path("${PREFLIGHT_ROOT}/datab_replay_asset_audit.json").write_text(
    json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(audit, ensure_ascii=False, indent=2))
if missing:
    raise SystemExit(2)
PY
  STAGE=preflight \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  MODEL_PATH="${BASE_MODEL}" \
  ADAPTER_PATH="${CORRECT_SFT_ADAPTER}" \
  RUN_ROOT="${PREFLIGHT_ROOT}/vifbench" \
  PERSIST_ROOT="${PERSIST_ROOT}/preflight/vifbench" \
  NUM_GPUS="${NUM_GPUS}" \
  KEEP_ALIVE_AFTER_RUN=0 \
  bash scripts/camera_detection_retention/run_vifbench.sh
  mkdir -p "${PERSIST_ROOT}/preflight"
  cp -a "${PREFLIGHT_ROOT}/environment_audit.json" \
    "${PREFLIGHT_ROOT}/datab_replay_asset_audit.json" \
    "${PERSIST_ROOT}/preflight/"
  echo "Preflight passed. No model weights, inference, or training were run."
}

build_joint_data_if_needed() {
  if [[ -f "${CAMERA_TRAIN_SOURCE}" && -f "${CAMERA_DEV_MATCHED}" \
        && -f "${CAMERA_DEV_OPPOSITE}" && -f "${CAMERA_DEV_NO_FRAMES}" ]]; then
    return
  fi
  mkdir -p "${JOINT_DATA_ROOT}"
  "${PYTHON_BIN}" tools/build_camera_joint_sft_gate.py \
    --dataa-detection-json "${DATAA_DETECTION_JSON}" \
    --dataa-camera-jsonl "${DATAA_CAMERA_JSONL}" \
    --datab-detection-json "${DATAB_DETECTION_JSON}" \
    --datab-camera-jsonl "${DATAB_CAMERA_JSONL}" \
    --out-dir "${JOINT_DATA_ROOT}" \
    --test-ratio 0.30 \
    --expected-dataa-cases 1080 \
    --seed 20260713 \
    --check-images
}

build_data() {
  build_joint_data_if_needed
  if [[ "${REBUILD_DATA}" != "1" && -f "${PPRL_DATA_SUMMARY}" \
        && -f "${PPRL_DATA}" && -f "${PPRL_SMOKE_DATA}" ]]; then
    echo "Reusing Camera-PPRL data: ${PPRL_DATA}"
    return
  fi
  mkdir -p "${DATA_ROOT}"
  "${PYTHON_BIN}" tools/build_camera_pprl_binary.py \
    --input-json "${CAMERA_TRAIN_SOURCE}" \
    --output-json "${PPRL_DATA}" \
    --smoke-json "${PPRL_SMOKE_DATA}" \
    --summary-json "${PPRL_DATA_SUMMARY}" \
    --max-records "${PPRL_RECORDS}" \
    --smoke-records "${PPRL_SMOKE_RECORDS}" \
    --seed 20260715 \
    --check-images
  persist_small_results
}

merge_adapter_atomic() {
  local model_path="$1"
  local adapter_path="$2"
  local destination="$3"
  if [[ "${REBUILD_MERGED}" != "1" && -f "${destination}/.merge_complete" \
        && -f "${destination}/config.json" ]]; then
    echo "Reusing merged model: ${destination}"
    return
  fi
  adapter_check "${adapter_path}"
  local parent
  local build_dir
  parent="$(dirname "${destination}")"
  mkdir -p "${parent}"
  build_dir="$(mktemp -d "${parent}/merge.build.XXXXXX")"
  "${PYTHON_BIN}" -m scripts.caspr_gate1.merge_adapter \
    --model-path "${model_path}" \
    --adapter-path "${adapter_path}" \
    --output-dir "${build_dir}"
  MERGED_AUDIT_PATH="${build_dir}" "${PYTHON_BIN}" - <<'PY'
import os
from transformers import AutoConfig, AutoProcessor

path = os.environ["MERGED_AUDIT_PATH"]
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
text_config = getattr(config, "text_config", None)
if text_config is not None and getattr(text_config, "rope_scaling", None) is None:
    raise ValueError("merged text_config.rope_scaling is None")
AutoProcessor.from_pretrained(path, trust_remote_code=True)
print("Merged model config and processor audit: OK")
PY
  touch "${build_dir}/.merge_complete"
  if [[ -e "${destination}" ]]; then
    mv "${destination}" "${destination}.incomplete.$(date +%Y%m%d_%H%M%S)"
  fi
  mv "${build_dir}" "${destination}"
}

resolve_adapter_dir() {
  local root="$1"
  if [[ -f "${root}/adapter_config.json" ]]; then
    echo "${root}"
    return
  fi
  local config
  config="$(find "${root}" -type f -name adapter_config.json | sort -V | tail -n 1)"
  if [[ -z "${config}" ]]; then
    echo "No adapter_config.json found under ${root}" >&2
    exit 2
  fi
  dirname "${config}"
}

compact_adapter() {
  local source_root="$1"
  local destination="$2"
  local source
  source="$(resolve_adapter_dir "${source_root}")"
  if [[ -d "${destination}" ]]; then
    mv "${destination}" "${destination}.old.$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "${destination}"
  find "${source}" -maxdepth 1 -type f -exec cp -a {} "${destination}/" \;
  adapter_check "${destination}"
  echo "Compact adapter: ${source} -> ${destination}"
}

archive_warm_start() {
  if [[ ! -f "${WARM_COMPACT_ADAPTER}/adapter_config.json" ]]; then
    compact_adapter "${CORRECT_SFT_ADAPTER}" "${WARM_COMPACT_ADAPTER}"
  fi
  upload_artifact "${WARM_COMPACT_ADAPTER}" "correct_camera_joint_sft_adapter"
}

append_optional_grpo_args() {
  local help_file="$1"
  local array_name="$2"
  local -n target_array="${array_name}"
  if help_has "${help_file}" "--importance_sampling_level"; then
    target_array+=(--importance_sampling_level sequence)
  fi
  if help_has "${help_file}" "--dynamic_sample"; then
    target_array+=(--dynamic_sample true)
  fi
  if help_has "${help_file}" "--max_resample_times"; then
    target_array+=(--max_resample_times "${PPRL_MAX_RESAMPLE_TIMES}")
  fi
  if help_has "${help_file}" "--log_entropy"; then
    target_array+=(--log_entropy true)
  fi
  if [[ "${VLLM_ENABLE_LORA}" == "1" ]] \
      && help_has "${help_file}" "--vllm_enable_lora" \
      && help_has "${help_file}" "--vllm_max_lora_rank"; then
    target_array+=(--vllm_enable_lora true --vllm_max_lora_rank "${PPRL_LORA_RANK}")
  fi
}

run_grpo() {
  local dataset="$1"
  local output_dir="$2"
  local save_steps="$3"
  local help_file="${PREFLIGHT_ROOT}/swift_rlhf_help.txt"
  require_file "${help_file}"
  local type_option
  type_option="$(training_type_option "${help_file}")"
  local -a args=(
    rlhf
    --rlhf_type grpo
    --model "${SFT_MERGED_MODEL}"
    --external_plugins "${REPO_ROOT}/rl/camera_detection_rewards.py"
    --reward_funcs camera_binary_acc camera_binary_format
    --reward_weights 0.9 0.1
    --dataset "${dataset}"
    --split_dataset_ratio 0
    "${type_option}" lora
    --lora_rank "${PPRL_LORA_RANK}"
    --lora_alpha "${PPRL_LORA_ALPHA}"
    --target_modules all-linear
    --freeze_vit true
    --freeze_aligner true
    --torch_dtype bfloat16
    --attn_impl flash_attn
    --max_length "${MAX_LENGTH}"
    --max_completion_length 32
    --num_train_epochs "${PPRL_EPOCHS}"
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps 1
    --learning_rate "${PPRL_LR}"
    --lr_scheduler_type cosine
    --warmup_ratio 0.03
    --save_steps "${save_steps}"
    --save_total_limit 2
    --logging_steps 1
    --dataset_num_proc 16
    --dataloader_num_workers 4
    --num_generations "${NUM_GENERATIONS}"
    --temperature "${PPRL_TEMPERATURE}"
    --top_p 1.0
    --beta "${PPRL_BETA}"
    --loss_type grpo
    --use_vllm true
    --vllm_mode colocate
    --vllm_gpu_memory_utilization "${VLLM_MEMORY}"
    --vllm_tensor_parallel_size "${VLLM_TP}"
    --vllm_max_model_len "${MAX_LENGTH}"
    --sleep_level 1
    --log_completions true
    --report_to tensorboard
    --output_dir "${output_dir}"
  )
  append_optional_grpo_args "${help_file}" args
  (
    cd "${MS_SWIFT_ROOT}"
    env \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      NPROC_PER_NODE="${NUM_GPUS}" \
      MASTER_PORT="${MASTER_PORT:-29615}" \
      MAX_PIXELS="${MAX_PIXELS}" \
      IMAGE_MAX_TOKEN_NUM=1024 \
      PYTHONPATH="${PYTHONPATH}" \
      "${SWIFT_BIN}" "${args[@]}"
  )
}

run_smoke() {
  require_file "${PPRL_SMOKE_DATA}"
  require_dir "${SFT_MERGED_MODEL}"
  local smoke_root="${WORK_ROOT}/smoke"
  if [[ -d "${smoke_root}" ]]; then
    mv "${smoke_root}" "${smoke_root}.old.$(date +%Y%m%d_%H%M%S)"
  fi
  if ! PPRL_EPOCHS=1.0 run_grpo "${PPRL_SMOKE_DATA}" "${smoke_root}" 9999; then
    if [[ "${VLLM_ENABLE_LORA}" != "1" ]]; then
      return 1
    fi
    echo "Smoke failed with vLLM LoRA-only sync; retrying once with full weight sync."
    VLLM_ENABLE_LORA=0
    if [[ -d "${smoke_root}" ]]; then
      mv "${smoke_root}" "${smoke_root}.lora_sync_failed.$(date +%Y%m%d_%H%M%S)"
    fi
    PPRL_EPOCHS=1.0 run_grpo "${PPRL_SMOKE_DATA}" "${smoke_root}" 9999
  fi
  resolve_adapter_dir "${smoke_root}" >/dev/null
  "${PYTHON_BIN}" tools/audit_camera_pprl_smoke.py \
    --train-dir "${smoke_root}" \
    --output-json "${smoke_root}/smoke_reward_audit.json" \
    --max-zero-std-rate "${PPRL_SMOKE_MAX_ZERO_STD_RATE}" \
    --min-log-points 2
  persist_small_results
  echo "Camera-PPRL distributed smoke passed."
}

train_pprl() {
  if [[ "${RETRAIN_PPRL}" != "1" && -f "${PPRL_ADAPTER}/adapter_config.json" ]]; then
    echo "Reusing completed Camera-PPRL adapter: ${PPRL_ADAPTER}"
    return
  fi
  require_file "${PPRL_DATA}"
  require_dir "${SFT_MERGED_MODEL}"
  if [[ -d "${PPRL_OUTPUT}" ]]; then
    mv "${PPRL_OUTPUT}" "${PPRL_OUTPUT}.old.$(date +%Y%m%d_%H%M%S)"
  fi
  run_grpo "${PPRL_DATA}" "${PPRL_OUTPUT}" 128
  compact_adapter "${PPRL_OUTPUT}" "${PPRL_ADAPTER}"
  persist_small_results
  upload_artifact "${PPRL_ADAPTER}" "camera_pprl_adapter"
}

score_camera_model() {
  local name="$1"
  local model_path="$2"
  local adapter_path="$3"
  local prediction_dir="${CAMERA_PRED_ROOT}/${name}"
  local eval_json="${CAMERA_EVAL_ROOT}/${name}.json"
  adapter_check "${adapter_path}"
  require_file "${CAMERA_DEV_MATCHED}"
  require_file "${CAMERA_DEV_OPPOSITE}"
  require_file "${CAMERA_DEV_NO_FRAMES}"
  if [[ -d "${prediction_dir}" ]]; then
    mv "${prediction_dir}" "${prediction_dir}.old.$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "${CAMERA_EVAL_ROOT}"
  torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    -m scripts.camera_joint_sft_gate.score_binary \
    --model-path "${model_path}" \
    --adapter-path "${adapter_path}" \
    --condition "matched_frames=${CAMERA_DEV_MATCHED}" \
    --condition "opposite_frames=${CAMERA_DEV_OPPOSITE}" \
    --condition "no_frames=${CAMERA_DEV_NO_FRAMES}" \
    --output-dir "${prediction_dir}" \
    --model-stage "${name}" \
    --max-pixels "${MAX_PIXELS}" \
    --seed 20260715
  "${PYTHON_BIN}" -m scripts.camera_binary_vqa.evaluate \
    --gold "matched_frames=${CAMERA_DEV_MATCHED}" \
    --gold "opposite_frames=${CAMERA_DEV_OPPOSITE}" \
    --gold "no_frames=${CAMERA_DEV_NO_FRAMES}" \
    --predictions-dir "${prediction_dir}" \
    --model-stage "${name}" \
    --output-json "${eval_json}"
  persist_small_results
}

evaluate_warm_and_pprl_camera() {
  score_camera_model warm_joint_sft "${BASE_MODEL}" "${CORRECT_SFT_ADAPTER}"
  score_camera_model camera_pprl "${SFT_MERGED_MODEL}" "${PPRL_ADAPTER}"
}

run_vif_direct() {
  adapter_check "${PPRL_ADAPTER}"
  STAGE=all \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  MODEL_PATH="${SFT_MERGED_MODEL}" \
  ADAPTER_PATH="${PPRL_ADAPTER}" \
  RUN_ROOT="${VIF_DIRECT_ROOT}" \
  PERSIST_ROOT="${PERSIST_ROOT}/vif_direct" \
  MERGED_MODEL_DIR="${PPRL_MERGED_MODEL}" \
  BASE_MODEL_NAME="Qwen3-VL-8B-correct-camera-joint-sft-vif" \
  CAMERA_MODEL_NAME="Qwen3-VL-8B-camera-pprl-vif" \
  NUM_GPUS="${NUM_GPUS}" \
  PARALLEL_MODELS=1 \
  SKIP_BASE_INFERENCE=0 \
  KEEP_ALIVE_AFTER_RUN=0 \
  bash scripts/camera_detection_retention/run_vifbench.sh
}

train_detection_recovery() {
  if [[ "${RETRAIN_RECOVERY}" != "1" && -f "${RECOVERY_ADAPTER}/adapter_config.json" ]]; then
    echo "Reusing completed detection-recovery adapter: ${RECOVERY_ADAPTER}"
    return
  fi
  if [[ ! -f "${PPRL_MERGED_MODEL}/.merge_complete" ]]; then
    merge_adapter_atomic "${SFT_MERGED_MODEL}" "${PPRL_ADAPTER}" "${PPRL_MERGED_MODEL}"
  fi
  require_dir "${PPRL_MERGED_MODEL}"
  require_file "${DATAB_DETECTION_JSON}"
  local help_file="${PREFLIGHT_ROOT}/swift_sft_help.txt"
  local type_option
  type_option="$(training_type_option "${help_file}")"
  if [[ -d "${RECOVERY_OUTPUT}" ]]; then
    mv "${RECOVERY_OUTPUT}" "${RECOVERY_OUTPUT}.old.$(date +%Y%m%d_%H%M%S)"
  fi
  local -a args=(
    sft
    --model "${PPRL_MERGED_MODEL}"
    --dataset "${DATAB_DETECTION_JSON}"
    --split_dataset_ratio 0
    "${type_option}" lora
    --lora_rank "${RECOVERY_LORA_RANK}"
    --lora_alpha "${RECOVERY_LORA_ALPHA}"
    --target_modules all-linear
    --freeze_vit true
    --freeze_aligner true
    --torch_dtype bfloat16
    --attn_impl flash_attn
    --max_length "${MAX_LENGTH}"
    --num_train_epochs "${RECOVERY_EPOCHS}"
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps 1
    --learning_rate "${RECOVERY_LR}"
    --lr_scheduler_type cosine
    --warmup_ratio 0.03
    --save_steps 100
    --save_total_limit 2
    --logging_steps 10
    --dataset_num_proc 16
    --dataloader_num_workers 4
    --report_to tensorboard
    --output_dir "${RECOVERY_OUTPUT}"
  )
  (
    cd "${MS_SWIFT_ROOT}"
    env \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      NPROC_PER_NODE="${NUM_GPUS}" \
      MASTER_PORT="${RECOVERY_MASTER_PORT:-29625}" \
      MAX_PIXELS="${MAX_PIXELS}" \
      IMAGE_MAX_TOKEN_NUM=1024 \
      PYTHONPATH="${PYTHONPATH}" \
      "${SWIFT_BIN}" "${args[@]}"
  )
  compact_adapter "${RECOVERY_OUTPUT}" "${RECOVERY_ADAPTER}"
  persist_small_results
  upload_artifact "${RECOVERY_ADAPTER}" "detection_recovery_adapter"
}

evaluate_recovery_camera() {
  score_camera_model detection_recovery "${PPRL_MERGED_MODEL}" "${RECOVERY_ADAPTER}"
}

run_vif_recovery() {
  adapter_check "${RECOVERY_ADAPTER}"
  STAGE=all \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  MODEL_PATH="${PPRL_MERGED_MODEL}" \
  ADAPTER_PATH="${RECOVERY_ADAPTER}" \
  RUN_ROOT="${VIF_RECOVERY_ROOT}" \
  PERSIST_ROOT="${PERSIST_ROOT}/vif_recovery" \
  MERGED_MODEL_DIR="${RECOVERY_MERGED_MODEL}" \
  BASE_MODEL_NAME="Qwen3-VL-8B-camera-pprl-vif-repeat" \
  CAMERA_MODEL_NAME="Qwen3-VL-8B-camera-pprl-detection-recovery-vif" \
  NUM_GPUS="${NUM_GPUS}" \
  PARALLEL_MODELS=1 \
  SKIP_BASE_INFERENCE=0 \
  KEEP_ALIVE_AFTER_RUN=0 \
  bash scripts/camera_detection_retention/run_vifbench.sh
}

summarize_all() {
  local summary="${WORK_ROOT}/camera_pprl_final_summary.json"
  local -a args=(
    --warm-camera-eval "${CAMERA_EVAL_ROOT}/warm_joint_sft.json" \
    --pprl-camera-eval "${CAMERA_EVAL_ROOT}/camera_pprl.json" \
    --recovery-camera-eval "${CAMERA_EVAL_ROOT}/detection_recovery.json" \
    --direct-vif-base-eval "${VIF_DIRECT_ROOT}/eval/base_vifbench_eval.json" \
    --direct-vif-pprl-eval "${VIF_DIRECT_ROOT}/eval/camera_adapter_vifbench_eval.json" \
    --recovery-vif-base-eval "${VIF_RECOVERY_ROOT}/eval/base_vifbench_eval.json" \
    --recovery-vif-model-eval "${VIF_RECOVERY_ROOT}/eval/camera_adapter_vifbench_eval.json" \
    --output-json "${summary}"
  )
  if [[ -f "${DETECTION_ONLY_VIF_REFERENCE}" ]]; then
    args+=(--detection-only-vif-reference "${DETECTION_ONLY_VIF_REFERENCE}")
  else
    echo "Detection-only ViF reference is not available yet; final status will remain pending if the PPRL core gate passes."
  fi
  "${PYTHON_BIN}" -m scripts.camera_pprl.summarize "${args[@]}"
  persist_small_results
}

run_all() {
  preflight
  build_data
  archive_warm_start
  merge_adapter_atomic "${BASE_MODEL}" "${CORRECT_SFT_ADAPTER}" "${SFT_MERGED_MODEL}"
  run_smoke
  train_pprl
  evaluate_warm_and_pprl_camera
  run_vif_direct
  train_detection_recovery
  evaluate_recovery_camera
  run_vif_recovery
  summarize_all
}

echo "=== 正确相机二元前置强化学习与检测恢复分阶段验证 ==="
echo "stage=${STAGE}"
echo "base_model=${BASE_MODEL}"
echo "correct_camera_sft_adapter=${CORRECT_SFT_ADAPTER}"
echo "work_root=${WORK_ROOT}"
echo "pprl_records=${PPRL_RECORDS}; num_generations=${NUM_GENERATIONS}"
echo "detection_camera_text_at_inference=false"

case "${STAGE}" in
  preflight) preflight ;;
  build) build_data ;;
  archive_warm_start) archive_warm_start ;;
  merge_warm_start) merge_adapter_atomic "${BASE_MODEL}" "${CORRECT_SFT_ADAPTER}" "${SFT_MERGED_MODEL}" ;;
  smoke) run_smoke ;;
  train_pprl) train_pprl ;;
  eval_camera_pprl) evaluate_warm_and_pprl_camera ;;
  vif_pprl) run_vif_direct ;;
  train_recovery) train_detection_recovery ;;
  eval_camera_recovery) evaluate_recovery_camera ;;
  vif_recovery) run_vif_recovery ;;
  summarize) summarize_all ;;
  all) run_all ;;
  *)
    echo "Unknown STAGE=${STAGE}" >&2
    exit 2
    ;;
esac

if [[ "${KEEP_ALIVE_AFTER_RUN}" == "1" ]]; then
  persist_small_results
  trap - EXIT
  echo "All requested stages completed. Starting ${KEEP_ALIVE_SCRIPT}."
  exec bash "${KEEP_ALIVE_SCRIPT}"
fi
