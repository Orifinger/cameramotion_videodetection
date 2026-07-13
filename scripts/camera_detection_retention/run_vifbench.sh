#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${STAGE:-all}"
PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_GPUS="${NUM_GPUS:-16}"
MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
ADAPTER_PATH="${ADAPTER_PATH:-/tmp/1res/dataa_camera_binary_vqa/detection_checkpoint_start/train/final}"
RUN_ROOT="${RUN_ROOT:-/tmp/1res/camera_detection_retention/vifbench_detection_checkpoint_start}"
PERSIST_ROOT="${PERSIST_ROOT:-${PROJECT_ROOT}/res/camera_detection_retention/vifbench_detection_checkpoint_start}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-${RUN_ROOT}/models/camera_binary_merged}"
V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR:-${PROJECT_ROOT}/eval/v4train-main/eval}"
INFER_SCRIPT="${INFER_SCRIPT:-${V4TRAIN_EVAL_DIR}/infer2_5_3.sh}"
OFFICIAL_EVAL_PY="${OFFICIAL_EVAL_PY:-${V4TRAIN_EVAL_DIR}/eval.py}"
PROMPT_DIR="${PROMPT_DIR:-${V4TRAIN_EVAL_DIR}/prompts/camera_context}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-${PROMPT_DIR}/datab_detection_system_prompt.txt}"
USER_PROMPT_SUFFIX_FILE="${USER_PROMPT_SUFFIX_FILE:-${PROMPT_DIR}/datab_no_camera_user_suffix.txt}"

if [[ -z "${INDEX_DIR:-}" ]]; then
  INDEX_DIR="${V4TRAIN_EVAL_DIR}/test_index_splits/splits_16"
  V4TRAIN_ROOT_INDEX_DIR="$(dirname "${V4TRAIN_EVAL_DIR}")/test_index_splits/splits_16"
  if [[ ! -d "${INDEX_DIR}" && -d "${V4TRAIN_ROOT_INDEX_DIR}" ]]; then
    INDEX_DIR="${V4TRAIN_ROOT_INDEX_DIR}"
  fi
fi

BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen3-VL-8B-detection-base-vifbench-retention}"
CAMERA_MODEL_NAME="${CAMERA_MODEL_NAME:-Qwen3-VL-8B-camera-adapter-vifbench-retention}"
INFERENCE_ROOT="${RUN_ROOT}/inference"
BASE_PRED_DIR="${INFERENCE_ROOT}/base/splitresults"
CAMERA_PRED_DIR="${INFERENCE_ROOT}/camera_adapter/splitresults"
COMBINED_DIR="${RUN_ROOT}/combined_predictions"
BASE_MERGED_JSON="${COMBINED_DIR}/base.json"
CAMERA_MERGED_JSON="${COMBINED_DIR}/camera_adapter.json"
PREFLIGHT_DIR="${RUN_ROOT}/preflight"
PREFLIGHT_SUMMARY="${PREFLIGHT_DIR}/vifbench_retention_preflight.json"
EVAL_ROOT="${RUN_ROOT}/eval"
BASE_EVAL_JSON="${EVAL_ROOT}/base_vifbench_eval.json"
CAMERA_EVAL_JSON="${EVAL_ROOT}/camera_adapter_vifbench_eval.json"
GATE_SUMMARY="${EVAL_ROOT}/vifbench_camera_adapter_retention_summary.json"
LOG_PATH="${RUN_ROOT}/pipeline.log"

REBUILD_MERGED="${REBUILD_MERGED:-0}"
PARALLEL_MODELS="${PARALLEL_MODELS:-1}"
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
  if [[ -d "${PREFLIGHT_DIR}" ]]; then
    mkdir -p "${PERSIST_ROOT}/preflight"
    cp -a "${PREFLIGHT_DIR}/." "${PERSIST_ROOT}/preflight/"
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

preflight() {
  require_dir "${MODEL_PATH}"
  require_file "${MODEL_PATH}/config.json"
  require_dir "${ADAPTER_PATH}"
  require_file "${ADAPTER_PATH}/adapter_config.json"
  if [[ ! -f "${ADAPTER_PATH}/adapter_model.safetensors" && ! -f "${ADAPTER_PATH}/adapter_model.bin" ]]; then
    echo "Missing adapter weights under ${ADAPTER_PATH}" >&2
    exit 2
  fi
  require_file "${INFER_SCRIPT}"
  require_file "${OFFICIAL_EVAL_PY}"
  require_dir "${INDEX_DIR}"
  require_file "${SYSTEM_PROMPT_FILE}"
  require_file "${USER_PROMPT_SUFFIX_FILE}"
  require_file "${KEEP_ALIVE_SCRIPT}"
  mkdir -p "${PREFLIGHT_DIR}"
  "${PYTHON_BIN}" -m scripts.camera_detection_retention.vifbench_retention audit \
    --index-dir "${INDEX_DIR}" \
    --system-prompt-file "${SYSTEM_PROMPT_FILE}" \
    --user-prompt-suffix-file "${USER_PROMPT_SUFFIX_FILE}" \
    --output-json "${PREFLIGHT_SUMMARY}" \
    --expected-ranks "${NUM_GPUS}" \
    --check-frame-dirs
  "${PYTHON_BIN}" -c "import numpy, peft, sklearn, torch, transformers; assert torch.cuda.device_count() == ${NUM_GPUS}; print('GPU and Python runtime: OK')"
  echo "Preflight passed. No model inference was run."
}

merge_camera_adapter() {
  if [[ "${REBUILD_MERGED}" != "1" && -f "${MERGED_MODEL_DIR}/config.json" ]]; then
    echo "Reusing merged model: ${MERGED_MODEL_DIR}"
    return
  fi
  "${PYTHON_BIN}" -m scripts.caspr_gate1.merge_adapter \
    --model-path "${MODEL_PATH}" \
    --adapter-path "${ADAPTER_PATH}" \
    --output-dir "${MERGED_MODEL_DIR}"
}

infer_one() {
  local model_path="$1"
  local model_name="$2"
  local save_dir="$3"
  local infer_log="$4"
  mkdir -p "${save_dir}" "$(dirname "${infer_log}")"
  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    WORLD_SIZE="${NUM_GPUS}" \
    PROMPT_MODE=no_camera \
    MODEL_PATH="${model_path}" \
    MODEL_NAME="${model_name}" \
    SAVE_DIR="${save_dir}" \
    INDEX_DIR="${INDEX_DIR}" \
    PROMPT_DIR="${PROMPT_DIR}" \
    SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE}" \
    USER_PROMPT_SUFFIX_FILE="${USER_PROMPT_SUFFIX_FILE}" \
    bash "${INFER_SCRIPT}" > "${infer_log}" 2>&1
}

run_inference_pair() {
  require_dir "${MERGED_MODEL_DIR}"
  if [[ "${PARALLEL_MODELS}" == "1" ]]; then
    echo "Launching base and camera-adapter inference concurrently: two model processes per GPU."
    infer_one "${MODEL_PATH}" "${BASE_MODEL_NAME}" "${BASE_PRED_DIR}" "${INFERENCE_ROOT}/base/inference.log" &
    local base_pid=$!
    infer_one "${MERGED_MODEL_DIR}" "${CAMERA_MODEL_NAME}" "${CAMERA_PRED_DIR}" "${INFERENCE_ROOT}/camera_adapter/inference.log" &
    local camera_pid=$!
    set +e
    wait "${base_pid}"
    local base_status=$?
    wait "${camera_pid}"
    local camera_status=$?
    set -e
    if [[ "${base_status}" != "0" || "${camera_status}" != "0" ]]; then
      echo "Inference failed: base_status=${base_status}, camera_status=${camera_status}" >&2
      echo "Base log: ${INFERENCE_ROOT}/base/inference.log" >&2
      echo "Camera log: ${INFERENCE_ROOT}/camera_adapter/inference.log" >&2
      exit 1
    fi
  else
    echo "Running base and camera-adapter inference sequentially."
    infer_one "${MODEL_PATH}" "${BASE_MODEL_NAME}" "${BASE_PRED_DIR}" "${INFERENCE_ROOT}/base/inference.log"
    infer_one "${MERGED_MODEL_DIR}" "${CAMERA_MODEL_NAME}" "${CAMERA_PRED_DIR}" "${INFERENCE_ROOT}/camera_adapter/inference.log"
  fi
}

evaluate_all() {
  mkdir -p "${EVAL_ROOT}" "${COMBINED_DIR}"
  "${PYTHON_BIN}" -m scripts.camera_detection_retention.vifbench_retention evaluate \
    --index-dir "${INDEX_DIR}" \
    --base-prediction-dir "${BASE_PRED_DIR}" \
    --camera-prediction-dir "${CAMERA_PRED_DIR}" \
    --base-merged-json "${BASE_MERGED_JSON}" \
    --camera-merged-json "${CAMERA_MERGED_JSON}" \
    --base-eval-json "${BASE_EVAL_JSON}" \
    --camera-eval-json "${CAMERA_EVAL_JSON}" \
    --output-json "${GATE_SUMMARY}" \
    --expected-ranks "${NUM_GPUS}"

  "${PYTHON_BIN}" "${OFFICIAL_EVAL_PY}" --json_file_path "${BASE_MERGED_JSON}" \
    | tee "${EVAL_ROOT}/base_official_eval.log"
  "${PYTHON_BIN}" "${OFFICIAL_EVAL_PY}" --json_file_path "${CAMERA_MERGED_JSON}" \
    | tee "${EVAL_ROOT}/camera_adapter_official_eval.log"
  cp -a "${BASE_MERGED_JSON%.json}_paired_metrics_transposed.csv" \
    "${EVAL_ROOT}/base_official_paired_metrics.csv"
  cp -a "${CAMERA_MERGED_JSON%.json}_paired_metrics_transposed.csv" \
    "${EVAL_ROOT}/camera_adapter_official_paired_metrics.csv"
  persist_small_results
}

launch_keepalive() {
  if [[ "${KEEP_ALIVE_AFTER_RUN}" != "1" ]]; then
    return
  fi
  persist_small_results
  trap - EXIT
  echo "VIF-Bench retention diagnostic completed. Starting keepalive: ${KEEP_ALIVE_SCRIPT}"
  exec bash "${KEEP_ALIVE_SCRIPT}"
}

echo "=== VIF-Bench camera-adapter detection-retention diagnostic ==="
echo "stage=${STAGE}"
echo "base_model=${MODEL_PATH}"
echo "camera_adapter=${ADAPTER_PATH}"
echo "run_root=${RUN_ROOT}"
echo "prompt_mode=no_camera"
echo "parallel_models=${PARALLEL_MODELS}; processes_per_gpu=$((PARALLEL_MODELS + 1))"

case "${STAGE}" in
  preflight)
    preflight
    ;;
  merge)
    merge_camera_adapter
    ;;
  infer)
    run_inference_pair
    ;;
  eval)
    evaluate_all
    ;;
  all)
    preflight
    merge_camera_adapter
    run_inference_pair
    evaluate_all
    launch_keepalive
    ;;
  *)
    echo "STAGE must be preflight, merge, infer, eval, or all" >&2
    exit 2
    ;;
esac
