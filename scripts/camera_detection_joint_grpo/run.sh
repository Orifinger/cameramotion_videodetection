#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${STAGE:-preflight}"
PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/input/workflow_58770161/workspace/test/ms_swift/ms-swift-main}"
V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR:-/input/workflow_58770161/workspace/test/test_selfcot/Skyra/eval}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SWIFT_BIN="${SWIFT_BIN:-swift}"
NUM_GPUS="${NUM_GPUS:-16}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
WORLD_SIZE="$((NUM_GPUS * WORKERS_PER_GPU))"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

BASE_MODEL="${BASE_MODEL:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
JOINT_SFT_ROOT="${JOINT_SFT_ROOT:-/tmp/1res/camera_joint_sft_gate}"
CAMERA_SFT_ADAPTER="${CAMERA_SFT_ADAPTER:-${JOINT_SFT_ROOT}/train/correct_camera}"
JOINT_SOURCE_DATA="${JOINT_SOURCE_DATA:-${JOINT_SFT_ROOT}/data}"

WORK_ROOT="${WORK_ROOT:-/tmp/1res/camera_detection_joint_grpo/v1}"
PERSIST_ROOT="${PERSIST_ROOT:-${PROJECT_ROOT}/res/camera_detection_joint_grpo/v1}"
OSS_ROOT="${OSS_ROOT:-oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_detection_joint_grpo/v1}"
DATA_ROOT="${WORK_ROOT}/data"
MODEL_ROOT="${WORK_ROOT}/models"
TRAIN_ROOT="${WORK_ROOT}/train"
ADAPTER_ROOT="${WORK_ROOT}/artifacts"
PREFLIGHT_ROOT="${WORK_ROOT}/preflight"
SMOKE_ROOT="${WORK_ROOT}/smoke"
DATAA_EVAL_ROOT="${WORK_ROOT}/dataa_eval"
VIF_ROOT="${WORK_ROOT}/vif_eval"
PIPELINE_LOG="${WORK_ROOT}/pipeline.log"

DATAA_DETECTION_JSON="${DATAA_DETECTION_JSON:-${PROJECT_ROOT}/res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json}"
DATAA_CAMERA_JSONL="${DATAA_CAMERA_JSONL:-${PROJECT_ROOT}/camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl}"
DATAB_DETECTION_JSON="${DATAB_DETECTION_JSON:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json}"
DATAB_CAMERA_JSONL="${DATAB_CAMERA_JSONL:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl}"

DATAA_TRAIN="${JOINT_SOURCE_DATA}/dataa_train_detection.json"
DATAA_TEST="${JOINT_SOURCE_DATA}/dataa_test_detection.json"
DATAB_REPLAY="${JOINT_SOURCE_DATA}/datab_detection_replay.json"
WARM_SFT_DATA="${DATA_ROOT}/joint_sft_warmup.json"
WARM_SFT_SMOKE_DATA="${DATA_ROOT}/joint_sft_warmup_smoke.json"
CORRECT_DATA="${DATA_ROOT}/joint_grpo_correct_camera.json"
SHUFFLED_DATA="${DATA_ROOT}/joint_grpo_shuffled_camera.json"
DETECTION_ONLY_DATA="${DATA_ROOT}/joint_grpo_detection_only.json"
DATAA_EVAL_DATA="${DATA_ROOT}/dataa_test_joint_detection.json"
DATA_SUMMARY="${DATA_ROOT}/camera_detection_joint_grpo_data_summary.json"

CAMERA_START_MODEL="${CAMERA_START_MODEL:-${MODEL_ROOT}/camera_capable_start}"
WARM_SFT_OUTPUT="${TRAIN_ROOT}/joint_format_sft"
WARM_SFT_ADAPTER="${WARM_SFT_ADAPTER:-${ADAPTER_ROOT}/joint_format_sft_adapter}"
WARM_MODEL="${WARM_MODEL:-${MODEL_ROOT}/joint_format_sft_merged}"

SFT_LORA_RANK="${SFT_LORA_RANK:-16}"
SFT_LORA_ALPHA="${SFT_LORA_ALPHA:-32}"
SFT_LR="${SFT_LR:-2e-6}"
SFT_EPOCHS="${SFT_EPOCHS:-1.0}"
GRPO_LORA_RANK="${GRPO_LORA_RANK:-16}"
GRPO_LORA_ALPHA="${GRPO_LORA_ALPHA:-32}"
GRPO_LR="${GRPO_LR:-8e-7}"
GRPO_EPOCHS="${GRPO_EPOCHS:-1.0}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
GRPO_TEMPERATURE="${GRPO_TEMPERATURE:-1.0}"
GRPO_BETA="${GRPO_BETA:-0.04}"
MAX_RESAMPLE_TIMES="${MAX_RESAMPLE_TIMES:-3}"
GRPO_SMOKE_MAX_ZERO_STD_RATE="${GRPO_SMOKE_MAX_ZERO_STD_RATE:-0.80}"
MAX_LENGTH="${MAX_LENGTH:-49152}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-160}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
VLLM_TP="${VLLM_TP:-4}"
VLLM_MEMORY="${VLLM_MEMORY:-0.45}"
VLLM_ENABLE_LORA="${VLLM_ENABLE_LORA:-1}"

SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-${REPO_ROOT}/prompts/camera_detection_joint_grpo/system_prompt.txt}"
USER_PROMPT_SUFFIX_FILE="${USER_PROMPT_SUFFIX_FILE:-${REPO_ROOT}/prompts/camera_detection_joint_grpo/user_suffix.txt}"
AUTO_UPLOAD_OSS="${AUTO_UPLOAD_OSS:-1}"
KEEP_ALIVE_AFTER_RUN="${KEEP_ALIVE_AFTER_RUN:-0}"
KEEP_ALIVE_SCRIPT="${KEEP_ALIVE_SCRIPT:-/input/training/keep.sh}"
REBUILD_DATA="${REBUILD_DATA:-0}"
RETRAIN_SFT="${RETRAIN_SFT:-0}"
RETRAIN_GRPO="${RETRAIN_GRPO:-0}"
REINFER="${REINFER:-0}"

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
  grep -q -- "$2" "$1"
}

training_type_option() {
  if help_has "$1" --train_type; then
    echo --train_type
  elif help_has "$1" --tuner_type; then
    echo --tuner_type
  else
    echo "No explicit LoRA type option is exposed in $1; using the ms-swift default LoRA tuner." >&2
    echo ""
  fi
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
  local source
  source="$(resolve_adapter_dir "$1")"
  local destination="$2"
  if [[ -e "${destination}" ]]; then
    mv "${destination}" "${destination}.old.$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "${destination}"
  find "${source}" -maxdepth 1 -type f -exec cp -a {} "${destination}/" \;
  adapter_check "${destination}"
}

merge_adapter_atomic() {
  local model_path="$1"
  local adapter_path="$2"
  local destination="$3"
  if [[ -f "${destination}/.merge_complete" && -f "${destination}/config.json" ]]; then
    echo "Reusing merged model: ${destination}"
    return
  fi
  adapter_check "${adapter_path}"
  local parent build_dir
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

branch_data() {
  case "$1" in
    correct_camera) echo "${CORRECT_DATA}" ;;
    shuffled_camera) echo "${SHUFFLED_DATA}" ;;
    detection_only) echo "${DETECTION_ONLY_DATA}" ;;
    *) echo "Unknown branch: $1" >&2; exit 2 ;;
  esac
}

branch_adapter() {
  echo "${ADAPTER_ROOT}/$1_adapter"
}

branch_output() {
  echo "${TRAIN_ROOT}/$1"
}

branch_model() {
  echo "${MODEL_ROOT}/$1_merged"
}

upload_adapter() {
  local source="$1"
  local name="$2"
  local marker="${source}/.oss_upload_complete"
  if [[ "${AUTO_UPLOAD_OSS}" != "1" || -f "${marker}" ]]; then
    return
  fi
  adapter_check "${source}"
  echo "Uploading reusable adapter: ${source} -> ${OSS_ROOT}/${name}/"
  ossutil64 cp -r "${source}/" "${OSS_ROOT}/${name}/"
  touch "${marker}"
}

persist_small_results() {
  mkdir -p "${PERSIST_ROOT}"
  for file in "${DATA_SUMMARY}" "${PIPELINE_LOG}" \
    "${DATAA_EVAL_ROOT}/camera_detection_joint_grpo_dataa_summary.json" \
    "${VIF_ROOT}/camera_detection_joint_grpo_vif_summary.json"
  do
    if [[ -f "${file}" ]]; then
      cp -a "${file}" "${PERSIST_ROOT}/"
    fi
  done
  if [[ -d "${PREFLIGHT_ROOT}" ]]; then
    mkdir -p "${PERSIST_ROOT}/preflight"
    find "${PREFLIGHT_ROOT}" -maxdepth 1 -type f -exec cp -a {} "${PERSIST_ROOT}/preflight/" \;
  fi
  for root in "${SMOKE_ROOT}" "${TRAIN_ROOT}"; do
    if [[ -d "${root}" ]]; then
      while IFS= read -r file; do
        local relative target
        relative="${file#${WORK_ROOT}/}"
        target="${PERSIST_ROOT}/${relative}"
        mkdir -p "$(dirname "${target}")"
        cp -a "${file}" "${target}"
      done < <(find "${root}" -type f \( -name 'trainer_log.jsonl' -o -name 'logging.jsonl' \
        -o -name 'all_results.json' -o -name '*audit*.json' -o -name 'args.json' \))
    fi
  done
  for root in "${DATAA_EVAL_ROOT}" "${VIF_ROOT}"; do
    if [[ -d "${root}" ]]; then
      while IFS= read -r file; do
        local relative target
        relative="${file#${WORK_ROOT}/}"
        target="${PERSIST_ROOT}/${relative}"
        mkdir -p "$(dirname "${target}")"
        cp -a "${file}" "${target}"
      done < <(find "${root}" -type f \( -name '*summary.json' -o -name '*eval.json' -o -name '*.csv' \))
    fi
  done
}

archive_on_exit() {
  local status=$?
  trap - EXIT
  set +e
  persist_small_results
  if [[ -f "${WARM_SFT_ADAPTER}/adapter_config.json" ]]; then
    upload_adapter "${WARM_SFT_ADAPTER}" joint_format_sft_adapter
  fi
  for branch in correct_camera detection_only shuffled_camera; do
    local adapter
    adapter="$(branch_adapter "${branch}")"
    if [[ -f "${adapter}/adapter_config.json" ]]; then
      upload_adapter "${adapter}" "${branch}_adapter"
    fi
  done
  echo "Pipeline exit status: ${status}"
  echo "Persistent small results: ${PERSIST_ROOT}"
  echo "OSS root: ${OSS_ROOT}/"
  exit "${status}"
}
trap archive_on_exit EXIT

preflight() {
  require_dir "${BASE_MODEL}"
  require_file "${BASE_MODEL}/config.json"
  adapter_check "${CAMERA_SFT_ADAPTER}"
  require_dir "${MS_SWIFT_ROOT}"
  require_file "${DATAA_DETECTION_JSON}"
  require_file "${DATAA_CAMERA_JSONL}"
  require_file "${DATAB_DETECTION_JSON}"
  require_file "${DATAB_CAMERA_JSONL}"
  require_file "tools/build_camera_detection_joint_grpo.py"
  require_file "tools/build_camera_joint_sft_gate.py"
  require_file "tools/audit_camera_pprl_smoke.py"
  require_file "rl/camera_detection_rewards.py"
  require_file "scripts/__init__.py"
  require_file "scripts/caspr_gate1/merge_adapter.py"
  require_file "scripts/caspr_gate1/runtime.py"
  require_file "scripts/camera_joint_sft_gate/summarize_dataa.py"
  require_file "scripts/camera_joint_sft_gate/summarize_vif_four_model.py"
  require_file "scripts/camera_detection_retention/run_vifbench.sh"
  require_file "scripts/camera_detection_retention/vifbench_retention.py"
  require_file "${SYSTEM_PROMPT_FILE}"
  require_file "${USER_PROMPT_SUFFIX_FILE}"
  require_file "${V4TRAIN_EVAL_DIR}/infer_dataa.py"
  require_file "${V4TRAIN_EVAL_DIR}/eval_dataa.py"
  require_file "${PROJECT_ROOT}/eval/v4train-main/eval/infer2_5_3.sh"
  require_file "${KEEP_ALIVE_SCRIPT}"
  command -v "${SWIFT_BIN}" >/dev/null
  command -v torchrun >/dev/null
  if [[ "${AUTO_UPLOAD_OSS}" == "1" ]]; then command -v ossutil64 >/dev/null; fi
  mkdir -p "${PREFLIGHT_ROOT}"
  (
    cd "${MS_SWIFT_ROOT}"
    "${SWIFT_BIN}" sft --help > "${PREFLIGHT_ROOT}/swift_sft_help.txt"
    "${SWIFT_BIN}" rlhf --help > "${PREFLIGHT_ROOT}/swift_rlhf_help.txt"
  )
  for option in --rlhf_type --external_plugins --reward_funcs --reward_weights \
    --num_generations --use_vllm --vllm_mode --vllm_tensor_parallel_size \
    --freeze_vit --freeze_aligner --loss_type --dynamic_sample --max_resample_times
  do
    if ! help_has "${PREFLIGHT_ROOT}/swift_rlhf_help.txt" "${option}"; then
      echo "Current ms-swift rlhf CLI is missing required option: ${option}" >&2
      exit 2
    fi
  done
  training_type_option "${PREFLIGHT_ROOT}/swift_sft_help.txt" >/dev/null
  training_type_option "${PREFLIGHT_ROOT}/swift_rlhf_help.txt" >/dev/null
  "${PYTHON_BIN}" - <<PY
import json
import importlib.util
from pathlib import Path
import torch, transformers, swift, vllm
from rl.camera_detection_rewards import orms

repo_root = Path("${REPO_ROOT}").resolve()
project_modules = (
    "scripts.caspr_gate1.merge_adapter",
    "scripts.caspr_gate1.runtime",
    "scripts.camera_detection_joint_grpo.summarize",
    "scripts.camera_detection_retention.vifbench_retention",
    "scripts.camera_joint_sft_gate.summarize_dataa",
    "scripts.camera_joint_sft_gate.summarize_vif_four_model",
)
module_origins = {}
for name in project_modules:
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin, f"Cannot resolve project module: {name}"
    origin = Path(spec.origin).resolve()
    assert repo_root in origin.parents, f"{name} resolved outside project: {origin}"
    module_origins[name] = str(origin)

assert torch.cuda.device_count() == ${NUM_GPUS}, (torch.cuda.device_count(), ${NUM_GPUS})
assert ${NUM_GPUS} % ${VLLM_TP} == 0
assert (${NUM_GPUS} * 1) % ${NUM_GENERATIONS} == 0
for name in ("joint_detection_acc", "camera_set_f1", "joint_output_format"):
    assert name in orms, name
payload = {
    "status": "passed",
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "swift_file": swift.__file__,
    "vllm": getattr(vllm, "__version__", "unknown"),
    "gpus": torch.cuda.device_count(),
    "project_module_origins": module_origins,
    "reward_weights": {
        "correct_or_shuffled": [0.65, 0.30, 0.05],
        "detection_only": [0.95, 0.05],
    },
    "detection_reward_dominates_all_auxiliary_rewards": 0.65 > 0.30 + 0.05,
}
Path("${PREFLIGHT_ROOT}/environment_audit.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
  echo "Preflight passed. It checked files, project module resolution, CLI options, reward registration, and 16 GPUs only."
}

build_joint_source_if_needed() {
  if [[ -f "${DATAA_TRAIN}" && -f "${DATAA_TEST}" && -f "${DATAB_REPLAY}" ]]; then
    return
  fi
  mkdir -p "${JOINT_SOURCE_DATA}"
  "${PYTHON_BIN}" tools/build_camera_joint_sft_gate.py \
    --dataa-detection-json "${DATAA_DETECTION_JSON}" \
    --dataa-camera-jsonl "${DATAA_CAMERA_JSONL}" \
    --datab-detection-json "${DATAB_DETECTION_JSON}" \
    --datab-camera-jsonl "${DATAB_CAMERA_JSONL}" \
    --out-dir "${JOINT_SOURCE_DATA}" \
    --test-ratio 0.30 \
    --expected-dataa-cases 1080 \
    --seed 20260713 \
    --check-images
}

build_data() {
  build_joint_source_if_needed
  if [[ "${REBUILD_DATA}" != "1" && -f "${DATA_SUMMARY}" && -f "${CORRECT_DATA}" ]]; then
    echo "Reusing joint camera-detection data: ${DATA_ROOT}"
    return
  fi
  mkdir -p "${DATA_ROOT}"
  "${PYTHON_BIN}" tools/build_camera_detection_joint_grpo.py \
    --dataa-train-json "${DATAA_TRAIN}" \
    --dataa-test-json "${DATAA_TEST}" \
    --dataa-camera-jsonl "${DATAA_CAMERA_JSONL}" \
    --datab-replay-json "${DATAB_REPLAY}" \
    --datab-camera-jsonl "${DATAB_CAMERA_JSONL}" \
    --out-dir "${DATA_ROOT}" \
    --dataa-records 512 \
    --datab-records 512 \
    --smoke-records 64 \
    --seed 20260715 \
    --check-images
  persist_small_results
}

prepare_camera_start() {
  merge_adapter_atomic "${BASE_MODEL}" "${CAMERA_SFT_ADAPTER}" "${CAMERA_START_MODEL}"
}

run_sft() {
  local dataset="$1" output="$2" epochs="$3"
  local help_file="${PREFLIGHT_ROOT}/swift_sft_help.txt"
  require_file "${help_file}"
  local type_option
  type_option="$(training_type_option "${help_file}")"
  local -a tuning_args=()
  if [[ -n "${type_option}" ]]; then tuning_args=("${type_option}" lora); fi
  local -a args=(
    sft
    --model "${CAMERA_START_MODEL}"
    --dataset "${dataset}"
    --split_dataset_ratio 0
    "${tuning_args[@]}"
    --lora_rank "${SFT_LORA_RANK}"
    --lora_alpha "${SFT_LORA_ALPHA}"
    --target_modules all-linear
    --freeze_vit true
    --freeze_aligner true
    --torch_dtype bfloat16
    --attn_impl flash_attn
    --max_length "${MAX_LENGTH}"
    --num_train_epochs "${epochs}"
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 1
    --learning_rate "${SFT_LR}"
    --lr_scheduler_type cosine
    --warmup_ratio 0.03
    --save_steps 64
    --save_total_limit 2
    --logging_steps 5
    --dataset_num_proc 16
    --dataloader_num_workers 4
    --report_to tensorboard
    --output_dir "${output}"
  )
  (
    cd "${MS_SWIFT_ROOT}"
    env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" NPROC_PER_NODE="${NUM_GPUS}" \
      MASTER_PORT="${SFT_MASTER_PORT:-29715}" MAX_PIXELS="${MAX_PIXELS}" \
      IMAGE_MAX_TOKEN_NUM=1024 PYTHONPATH="${PYTHONPATH}" \
      "${SWIFT_BIN}" "${args[@]}"
  )
}

smoke_sft() {
  prepare_camera_start
  require_file "${WARM_SFT_SMOKE_DATA}"
  local output="${SMOKE_ROOT}/joint_format_sft"
  if [[ -d "${output}" ]]; then mv "${output}" "${output}.old.$(date +%Y%m%d_%H%M%S)"; fi
  run_sft "${WARM_SFT_SMOKE_DATA}" "${output}" 1.0
  resolve_adapter_dir "${output}" >/dev/null
  echo "Joint-format SFT smoke passed."
}

train_warm_sft() {
  prepare_camera_start
  if [[ "${RETRAIN_SFT}" != "1" && -f "${WARM_SFT_ADAPTER}/adapter_config.json" ]]; then
    echo "Reusing common joint-format SFT adapter: ${WARM_SFT_ADAPTER}"
  else
    if [[ -d "${WARM_SFT_OUTPUT}" ]]; then
      mv "${WARM_SFT_OUTPUT}" "${WARM_SFT_OUTPUT}.old.$(date +%Y%m%d_%H%M%S)"
    fi
    run_sft "${WARM_SFT_DATA}" "${WARM_SFT_OUTPUT}" "${SFT_EPOCHS}"
    compact_adapter "${WARM_SFT_OUTPUT}" "${WARM_SFT_ADAPTER}"
    upload_adapter "${WARM_SFT_ADAPTER}" joint_format_sft_adapter
  fi
  merge_adapter_atomic "${CAMERA_START_MODEL}" "${WARM_SFT_ADAPTER}" "${WARM_MODEL}"
}

append_optional_grpo_args() {
  local help_file="$1" array_name="$2"
  local -n target_array="${array_name}"
  if help_has "${help_file}" --importance_sampling_level; then
    target_array+=(--importance_sampling_level sequence)
  fi
  target_array+=(--dynamic_sample true --max_resample_times "${MAX_RESAMPLE_TIMES}")
  if help_has "${help_file}" --log_entropy; then target_array+=(--log_entropy true); fi
  if [[ "${VLLM_ENABLE_LORA}" == "1" ]] \
      && help_has "${help_file}" --vllm_enable_lora \
      && help_has "${help_file}" --vllm_max_lora_rank; then
    target_array+=(--vllm_enable_lora true --vllm_max_lora_rank "${GRPO_LORA_RANK}")
  fi
}

run_grpo() {
  local branch="$1" dataset="$2" output="$3" epochs="$4"
  local help_file="${PREFLIGHT_ROOT}/swift_rlhf_help.txt"
  require_file "${help_file}"
  local type_option
  type_option="$(training_type_option "${help_file}")"
  local -a tuning_args=()
  if [[ -n "${type_option}" ]]; then tuning_args=("${type_option}" lora); fi
  local -a rewards weights
  if [[ "${branch}" == "detection_only" ]]; then
    rewards=(joint_detection_acc joint_output_format)
    weights=(0.95 0.05)
  else
    rewards=(joint_detection_acc camera_set_f1 joint_output_format)
    weights=(0.65 0.30 0.05)
  fi
  local -a args=(
    rlhf
    --rlhf_type grpo
    --model "${WARM_MODEL}"
    --external_plugins "${REPO_ROOT}/rl/camera_detection_rewards.py"
    --reward_funcs "${rewards[@]}"
    --reward_weights "${weights[@]}"
    --dataset "${dataset}"
    --split_dataset_ratio 0
    "${tuning_args[@]}"
    --lora_rank "${GRPO_LORA_RANK}"
    --lora_alpha "${GRPO_LORA_ALPHA}"
    --target_modules all-linear
    --freeze_vit true
    --freeze_aligner true
    --torch_dtype bfloat16
    --attn_impl flash_attn
    --max_length "${MAX_LENGTH}"
    --max_completion_length "${MAX_COMPLETION_LENGTH}"
    --num_train_epochs "${epochs}"
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 1
    --learning_rate "${GRPO_LR}"
    --lr_scheduler_type cosine
    --warmup_ratio 0.03
    --save_steps 32
    --save_total_limit 2
    --logging_steps 1
    --dataset_num_proc 16
    --dataloader_num_workers 4
    --num_generations "${NUM_GENERATIONS}"
    --temperature "${GRPO_TEMPERATURE}"
    --top_p 1.0
    --beta "${GRPO_BETA}"
    --loss_type grpo
    --use_vllm true
    --vllm_mode colocate
    --vllm_gpu_memory_utilization "${VLLM_MEMORY}"
    --vllm_tensor_parallel_size "${VLLM_TP}"
    --vllm_max_model_len "${MAX_LENGTH}"
    --sleep_level 1
    --log_completions true
    --report_to tensorboard
    --output_dir "${output}"
  )
  append_optional_grpo_args "${help_file}" args
  (
    cd "${MS_SWIFT_ROOT}"
    env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" NPROC_PER_NODE="${NUM_GPUS}" \
      MASTER_PORT="${GRPO_MASTER_PORT:-29725}" MAX_PIXELS="${MAX_PIXELS}" \
      IMAGE_MAX_TOKEN_NUM=1024 PYTHONPATH="${PYTHONPATH}" \
      "${SWIFT_BIN}" "${args[@]}"
  )
}

smoke_grpo() {
  train_warm_sft
  local branch="${BRANCH:-correct_camera}"
  local dataset="${DATA_ROOT}/joint_grpo_${branch}_smoke.json"
  local output="${SMOKE_ROOT}/grpo_${branch}"
  require_file "${dataset}"
  if [[ -d "${output}" ]]; then mv "${output}" "${output}.old.$(date +%Y%m%d_%H%M%S)"; fi
  if ! run_grpo "${branch}" "${dataset}" "${output}" 1.0; then
    if [[ "${VLLM_ENABLE_LORA}" != "1" ]]; then return 1; fi
    echo "GRPO smoke failed with vLLM LoRA sync; retrying once with full weight sync."
    VLLM_ENABLE_LORA=0
    if [[ -d "${output}" ]]; then mv "${output}" "${output}.sync_failed.$(date +%Y%m%d_%H%M%S)"; fi
    run_grpo "${branch}" "${dataset}" "${output}" 1.0
  fi
  resolve_adapter_dir "${output}" >/dev/null
  "${PYTHON_BIN}" tools/audit_camera_pprl_smoke.py \
    --train-dir "${output}" \
    --output-json "${output}/reward_variance_audit.json" \
    --max-zero-std-rate "${GRPO_SMOKE_MAX_ZERO_STD_RATE}" \
    --min-log-points 2
  persist_small_results
}

train_branch() {
  local branch="$1"
  train_warm_sft
  local dataset output adapter
  dataset="$(branch_data "${branch}")"
  output="$(branch_output "${branch}")"
  adapter="$(branch_adapter "${branch}")"
  if [[ "${RETRAIN_GRPO}" != "1" && -f "${adapter}/adapter_config.json" ]]; then
    echo "Reusing ${branch} adapter: ${adapter}"
    return
  fi
  if [[ -d "${output}" ]]; then mv "${output}" "${output}.old.$(date +%Y%m%d_%H%M%S)"; fi
  run_grpo "${branch}" "${dataset}" "${output}" "${GRPO_EPOCHS}"
  compact_adapter "${output}" "${adapter}"
  upload_adapter "${adapter}" "${branch}_adapter"
  persist_small_results
}

merge_branch() {
  local branch="$1"
  merge_adapter_atomic "${WARM_MODEL}" "$(branch_adapter "${branch}")" "$(branch_model "${branch}")"
}

infer_dataa_model() {
  local name="$1" model_path="$2"
  local prediction_dir="${DATAA_EVAL_ROOT}/${name}/predictions"
  local eval_dir="${DATAA_EVAL_ROOT}/${name}/eval"
  local eval_json="${eval_dir}/${name}_summary.json"
  if [[ "${REINFER}" != "1" && -f "${eval_json}" ]]; then
    echo "Reusing DataA evaluation: ${eval_json}"
    return
  fi
  require_file "${DATAA_EVAL_DATA}"
  require_dir "${model_path}"
  mkdir -p "${prediction_dir}" "${eval_dir}"
  export PYTHON_BIN V4TRAIN_EVAL_DIR DATAA_EVAL_DATA WORLD_SIZE NUM_GPUS MAX_PIXELS
  export name model_path prediction_dir
  seq 0 "$((WORLD_SIZE - 1))" | xargs -n1 -P "${WORLD_SIZE}" bash -lc '
    rank="$1"
    device_id="$((rank % NUM_GPUS))"
    CUDA_VISIBLE_DEVICES="${device_id}" "${PYTHON_BIN}" "${V4TRAIN_EVAL_DIR}/infer_dataa.py" \
      --sft_json "${DATAA_EVAL_DATA}" \
      --model_path "${model_path}" \
      --model_name "${name}-rank${rank}" \
      --save_dir "${prediction_dir}/rank_${rank}" \
      --rank "${rank}" \
      --world_size "${WORLD_SIZE}" \
      --max_new_tokens 192 \
      --prompt_mode record \
      --image_max_pixels "${MAX_PIXELS}" \
      --overwrite
  ' _
  "${PYTHON_BIN}" "${V4TRAIN_EVAL_DIR}/eval_dataa.py" \
    --gt_json "${DATAA_EVAL_DATA}" \
    --pred_json "${prediction_dir}" \
    --out_dir "${eval_dir}" \
    --output_prefix "${name}"
}

eval_dataa_all() {
  train_warm_sft
  infer_dataa_model warm_start "${WARM_MODEL}"
  for branch in correct_camera detection_only shuffled_camera; do
    merge_branch "${branch}"
    infer_dataa_model "${branch}" "$(branch_model "${branch}")"
  done
  local output="${DATAA_EVAL_ROOT}/camera_detection_joint_grpo_dataa_summary.json"
  "${PYTHON_BIN}" -m scripts.camera_detection_joint_grpo.summarize dataa \
    --warm-eval "${DATAA_EVAL_ROOT}/warm_start/eval/warm_start_summary.json" \
    --correct-eval "${DATAA_EVAL_ROOT}/correct_camera/eval/correct_camera_summary.json" \
    --detection-only-eval "${DATAA_EVAL_ROOT}/detection_only/eval/detection_only_summary.json" \
    --shuffled-eval "${DATAA_EVAL_ROOT}/shuffled_camera/eval/shuffled_camera_summary.json" \
    --output-json "${output}"
  persist_small_results
}

run_vif_branch() {
  local branch="$1"
  train_warm_sft
  adapter_check "$(branch_adapter "${branch}")"
  local run_root="${VIF_ROOT}/${branch}"
  local skip_base=0
  local base_pred="${run_root}/inference/base/splitresults"
  if [[ "${branch}" != "correct_camera" ]]; then
    skip_base=1
    base_pred="${VIF_ROOT}/correct_camera/inference/base/splitresults"
    require_dir "${base_pred}"
  fi
  STAGE=all \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  MODEL_PATH="${WARM_MODEL}" \
  ADAPTER_PATH="$(branch_adapter "${branch}")" \
  RUN_ROOT="${run_root}" \
  PERSIST_ROOT="${PERSIST_ROOT}/vif_eval/${branch}" \
  MERGED_MODEL_DIR="$(branch_model "${branch}")" \
  SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE}" \
  USER_PROMPT_SUFFIX_FILE="${USER_PROMPT_SUFFIX_FILE}" \
  BASE_MODEL_NAME="Qwen3-VL-8B-joint-warm-vif" \
  CAMERA_MODEL_NAME="Qwen3-VL-8B-${branch}-joint-grpo-vif" \
  NUM_GPUS="${NUM_GPUS}" \
  SKIP_BASE_INFERENCE="${skip_base}" \
  BASE_PRED_DIR="${base_pred}" \
  PARALLEL_MODELS=1 \
  KEEP_ALIVE_AFTER_RUN=0 \
  bash scripts/camera_detection_retention/run_vifbench.sh
}

summarize_vif() {
  local output="${VIF_ROOT}/camera_detection_joint_grpo_vif_summary.json"
  "${PYTHON_BIN}" -m scripts.camera_detection_joint_grpo.summarize vif \
    --warm-eval "${VIF_ROOT}/correct_camera/eval/base_vifbench_eval.json" \
    --correct-eval "${VIF_ROOT}/correct_camera/eval/camera_adapter_vifbench_eval.json" \
    --detection-only-eval "${VIF_ROOT}/detection_only/eval/camera_adapter_vifbench_eval.json" \
    --shuffled-eval "${VIF_ROOT}/shuffled_camera/eval/camera_adapter_vifbench_eval.json" \
    --output-json "${output}"
  persist_small_results
}

run_all_dataa() {
  preflight
  build_data
  smoke_sft
  train_warm_sft
  BRANCH=correct_camera smoke_grpo
  train_branch correct_camera
  train_branch detection_only
  train_branch shuffled_camera
  eval_dataa_all
}

run_all_full() {
  run_all_dataa
  run_all_vif
}

run_all_vif() {
  run_vif_branch correct_camera
  run_vif_branch detection_only
  run_vif_branch shuffled_camera
  summarize_vif
}

launch_keepalive() {
  if [[ "${KEEP_ALIVE_AFTER_RUN}" != "1" ]]; then return; fi
  persist_small_results
  trap - EXIT
  echo "Requested stages completed. Starting ${KEEP_ALIVE_SCRIPT}."
  exec bash "${KEEP_ALIVE_SCRIPT}"
}

echo "=== 检测主导的相机中间变量联合 SFT/GRPO 门 ==="
echo "stage=${STAGE}"
echo "base_model=${BASE_MODEL}"
echo "camera_sft_adapter=${CAMERA_SFT_ADAPTER}"
echo "work_root=${WORK_ROOT}"
echo "primary_endpoint=Real/Fake; camera_text_at_inference=false"

case "${STAGE}" in
  preflight) preflight ;;
  build) build_data ;;
  merge_camera_start) prepare_camera_start ;;
  smoke_sft) smoke_sft ;;
  train_warm_sft) train_warm_sft ;;
  smoke_grpo) smoke_grpo ;;
  train_correct) train_branch correct_camera ;;
  train_detection_only) train_branch detection_only ;;
  train_shuffled) train_branch shuffled_camera ;;
  eval_dataa) eval_dataa_all ;;
  vif_correct) run_vif_branch correct_camera ;;
  vif_detection_only) run_vif_branch detection_only ;;
  vif_shuffled) run_vif_branch shuffled_camera ;;
  summarize_vif) summarize_vif ;;
  vif_all) run_all_vif ;;
  all_dataa) run_all_dataa ;;
  all_full) run_all_full ;;
  *)
    echo "Unknown STAGE=${STAGE}" >&2
    exit 2
    ;;
esac

launch_keepalive
