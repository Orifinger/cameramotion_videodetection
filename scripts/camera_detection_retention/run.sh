#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${STAGE:-all}"
PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_GPUS="${NUM_GPUS:-16}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
WORLD_SIZE="$((NUM_GPUS * WORKERS_PER_GPU))"
MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
ADAPTER_PATH="${ADAPTER_PATH:-/tmp/1res/dataa_camera_binary_vqa/detection_checkpoint_start/train/final}"
RUN_ROOT="${RUN_ROOT:-/tmp/1res/camera_detection_retention/detection_checkpoint_start}"
PERSIST_ROOT="${PERSIST_ROOT:-${PROJECT_ROOT}/res/camera_detection_retention/detection_checkpoint_start}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-${RUN_ROOT}/models/camera_binary_merged}"
MERGED_MODEL_MARKER="${MERGED_MODEL_DIR}/.merge_complete"
V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR:-/input/workflow_58770161/workspace/test/test_selfcot/Skyra/eval}"

DATAA_DETECTION_JSON="${DATAA_DETECTION_JSON:-${PROJECT_ROOT}/res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json}"
DATAA_DEV_SPLIT_JSON="${DATAA_DEV_SPLIT_JSON:-${PROJECT_ROOT}/tools/data/camera_motion_splits/dataA_test.json}"
DATA_DIR="${RUN_ROOT}/data"
DATAA_DEV_JSON="${DATA_DIR}/dataa_40step_v3_fixed_dev_detection.json"
DATA_SUMMARY="${DATA_DIR}/dataa_40step_v3_fixed_dev_detection_summary.json"
INFER_ROOT="${RUN_ROOT}/inference"
EVAL_ROOT="${RUN_ROOT}/eval"
BASE_EVAL="${EVAL_ROOT}/base/dataa_detection_base_summary.json"
CAMERA_EVAL="${EVAL_ROOT}/camera_adapter/dataa_detection_camera_adapter_summary.json"
GATE_SUMMARY="${EVAL_ROOT}/camera_detection_retention_summary.json"
LOG_PATH="${RUN_ROOT}/pipeline.log"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-262144}"
OVERWRITE="${OVERWRITE:-1}"
REBUILD_MERGED="${REBUILD_MERGED:-0}"
KEEP_ALIVE_AFTER_RUN="${KEEP_ALIVE_AFTER_RUN:-1}"
KEEP_ALIVE_SCRIPT="${KEEP_ALIVE_SCRIPT:-/input/training/keep.sh}"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

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

persist_small_results() {
  mkdir -p "${PERSIST_ROOT}"
  if [[ -f "${DATA_SUMMARY}" ]]; then
    mkdir -p "${PERSIST_ROOT}/data"
    cp -a "${DATA_SUMMARY}" "${PERSIST_ROOT}/data/"
  fi
  if [[ -d "${EVAL_ROOT}" ]]; then
    mkdir -p "${PERSIST_ROOT}/eval"
    cp -a "${EVAL_ROOT}/." "${PERSIST_ROOT}/eval/"
  fi
  cp -a "${LOG_PATH}" "${PERSIST_ROOT}/" 2>/dev/null || true
}

archive_on_exit() {
  local status=$?
  trap - EXIT
  set +e
  persist_small_results
  echo "Pipeline exit status: ${status}"
  echo "Persistent small results: ${PERSIST_ROOT}"
  exit "${status}"
}
trap archive_on_exit EXIT

build_data() {
  require_file "${DATAA_DETECTION_JSON}"
  require_file "${DATAA_DEV_SPLIT_JSON}"
  "${PYTHON_BIN}" -m scripts.camera_detection_retention.build_data \
    --detection-json "${DATAA_DETECTION_JSON}" \
    --fixed-dev-json "${DATAA_DEV_SPLIT_JSON}" \
    --output-json "${DATAA_DEV_JSON}" \
    --summary-json "${DATA_SUMMARY}" \
    --check-images
}

preflight() {
  require_dir "${MODEL_PATH}"
  require_file "${MODEL_PATH}/config.json"
  require_dir "${ADAPTER_PATH}"
  require_file "${ADAPTER_PATH}/adapter_config.json"
  if [[ ! -f "${ADAPTER_PATH}/adapter_model.safetensors" && ! -f "${ADAPTER_PATH}/adapter_model.bin" ]]; then
    echo "Missing adapter weights under ${ADAPTER_PATH}" >&2
    exit 2
  fi
  require_file "${V4TRAIN_EVAL_DIR}/infer_dataa.py"
  require_file "${V4TRAIN_EVAL_DIR}/eval_dataa.py"
  require_file "${KEEP_ALIVE_SCRIPT}"
  build_data
  "${PYTHON_BIN}" -c "import torch, transformers, peft; assert torch.cuda.device_count() == ${NUM_GPUS}; print('GPU and Python runtime: OK')"
  echo "Preflight passed. No model inference was run."
}

merge_camera_adapter() {
  if [[ "${REBUILD_MERGED}" != "1" && -f "${MERGED_MODEL_MARKER}" && -f "${MERGED_MODEL_DIR}/config.json" ]]; then
    echo "Reusing merged model: ${MERGED_MODEL_DIR}"
    return
  fi
  require_dir "${MODEL_PATH}"
  require_dir "${ADAPTER_PATH}"
  local build_parent="${RUN_ROOT}/models"
  local build_dir
  mkdir -p "${build_parent}"
  build_dir="$(mktemp -d "${build_parent}/camera_binary_merged.build.XXXXXX")"
  "${PYTHON_BIN}" -m scripts.caspr_gate1.merge_adapter \
    --model-path "${MODEL_PATH}" \
    --adapter-path "${ADAPTER_PATH}" \
    --output-dir "${build_dir}"
  MERGED_MODEL_AUDIT_PATH="${build_dir}" "${PYTHON_BIN}" -c '
import os
from transformers import AutoConfig, AutoProcessor
path = os.environ["MERGED_MODEL_AUDIT_PATH"]
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
text_config = getattr(config, "text_config", None)
if text_config is not None and getattr(text_config, "rope_scaling", None) is None:
    raise ValueError("merged text_config.rope_scaling is None")
AutoProcessor.from_pretrained(path, trust_remote_code=True)
print("Merged model config and processor audit: OK")
'
  touch "${build_dir}/.merge_complete"
  if [[ -e "${MERGED_MODEL_DIR}" ]]; then
    local incomplete_backup="${MERGED_MODEL_DIR}.incomplete.$(date +%Y%m%d_%H%M%S)"
    echo "Moving existing unverified merged directory to: ${incomplete_backup}"
    mv "${MERGED_MODEL_DIR}" "${incomplete_backup}"
  fi
  mv "${build_dir}" "${MERGED_MODEL_DIR}"
}

run_inference() {
  local name="$1"
  local model_path="$2"
  local save_dir="${INFER_ROOT}/${name}"
  require_file "${DATAA_DEV_JSON}"
  require_dir "${model_path}"
  mkdir -p "${save_dir}"
  export PYTHON_BIN V4TRAIN_EVAL_DIR DATAA_DEV_JSON MAX_NEW_TOKENS IMAGE_MAX_PIXELS OVERWRITE WORLD_SIZE NUM_GPUS
  export name model_path save_dir
  seq 0 "$((WORLD_SIZE - 1))" | xargs -n1 -P "${WORLD_SIZE}" bash -lc '
    rank="$1"
    device_id="$((rank % NUM_GPUS))"
    cmd=(
      "${PYTHON_BIN}" "${V4TRAIN_EVAL_DIR}/infer_dataa.py"
      --sft_json "${DATAA_DEV_JSON}"
      --model_path "${model_path}"
      --model_name "${name}-rank${rank}"
      --save_dir "${save_dir}/rank_${rank}"
      --rank "${rank}"
      --world_size "${WORLD_SIZE}"
      --max_new_tokens "${MAX_NEW_TOKENS}"
      --prompt_mode record
      --image_max_pixels "${IMAGE_MAX_PIXELS}"
    )
    if [[ "${OVERWRITE}" == "1" ]]; then cmd+=(--overwrite); fi
    CUDA_VISIBLE_DEVICES="${device_id}" "${cmd[@]}"
  ' _
}

evaluate_one() {
  local name="$1"
  local prefix="$2"
  mkdir -p "${EVAL_ROOT}/${name}"
  "${PYTHON_BIN}" "${V4TRAIN_EVAL_DIR}/eval_dataa.py" \
    --gt_json "${DATAA_DEV_JSON}" \
    --pred_json "${INFER_ROOT}/${name}" \
    --out_dir "${EVAL_ROOT}/${name}" \
    --output_prefix "${prefix}" \
    --compute_iou
}

evaluate_all() {
  evaluate_one base dataa_detection_base
  evaluate_one camera_adapter dataa_detection_camera_adapter
  "${PYTHON_BIN}" -m scripts.camera_detection_retention.summarize \
    --base-eval "${BASE_EVAL}" \
    --camera-eval "${CAMERA_EVAL}" \
    --output-json "${GATE_SUMMARY}"
  persist_small_results
}

launch_keepalive() {
  if [[ "${KEEP_ALIVE_AFTER_RUN}" != "1" ]]; then
    return
  fi
  require_file "${KEEP_ALIVE_SCRIPT}"
  persist_small_results
  trap - EXIT
  echo "Retention diagnostic completed. Starting keepalive: ${KEEP_ALIVE_SCRIPT}"
  exec bash "${KEEP_ALIVE_SCRIPT}"
}

echo "=== Camera VQA adapter detection-retention diagnostic ==="
echo "stage=${STAGE}"
echo "base_model=${MODEL_PATH}"
echo "camera_adapter=${ADAPTER_PATH}"
echo "run_root=${RUN_ROOT}"
echo "gpus=${NUM_GPUS} workers_per_gpu=${WORKERS_PER_GPU} inference_shards=${WORLD_SIZE}"

case "${STAGE}" in
  preflight)
    preflight
    ;;
  build)
    build_data
    ;;
  merge)
    merge_camera_adapter
    ;;
  infer_base)
    run_inference base "${MODEL_PATH}"
    ;;
  infer_camera)
    run_inference camera_adapter "${MERGED_MODEL_DIR}"
    ;;
  eval)
    evaluate_all
    ;;
  all)
    preflight
    merge_camera_adapter
    run_inference base "${MODEL_PATH}"
    run_inference camera_adapter "${MERGED_MODEL_DIR}"
    evaluate_all
    launch_keepalive
    ;;
  *)
    echo "STAGE must be preflight, build, merge, infer_base, infer_camera, eval, or all" >&2
    exit 2
    ;;
esac
