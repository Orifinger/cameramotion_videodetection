#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${STAGE:-preflight}"
PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_GPUS="${NUM_GPUS:-16}"
WORK_ROOT="${WORK_ROOT:-/tmp/1res/datab_explicit_camera_sft/v1/vifbench}"
PERSIST_ROOT="${PERSIST_ROOT:-${PROJECT_ROOT}/res/datab_explicit_camera_sft/v1/vifbench}"
NO_CAMERA_MODEL="${NO_CAMERA_MODEL:-/tmp/1res/datab_explicit_camera_sft/v1/train/no_camera}"
WITH_CAMERA_MODEL="${WITH_CAMERA_MODEL:-/tmp/1res/datab_explicit_camera_sft/v1/train/with_camera}"
DATAB_DETECTION_JSON="${DATAB_DETECTION_JSON:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json}"
VIF_CAMERA_JSON="${VIF_CAMERA_JSON:-/input/workflow_58770161/workspace/test/camb/camerabench_outputs/vifbench_cameramotion_labels_v2/datab_cameramotion_labels_v2.jsonl}"

V4TRAIN_ROOT="${V4TRAIN_ROOT:-${PROJECT_ROOT}/eval/v4train-main}"
V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR:-${V4TRAIN_ROOT}/eval}"
INFER_SCRIPT="${INFER_SCRIPT:-${V4TRAIN_EVAL_DIR}/infer2_5_3.sh}"
INFERENCE_PY="${INFERENCE_PY:-${V4TRAIN_EVAL_DIR}/inference.py}"
OFFICIAL_EVAL_PY="${OFFICIAL_EVAL_PY:-${V4TRAIN_EVAL_DIR}/eval.py}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-${V4TRAIN_EVAL_DIR}/prompts/camera_context/datab_detection_system_prompt.txt}"
NO_CAMERA_SUFFIX_FILE="${NO_CAMERA_SUFFIX_FILE:-${SCRIPT_DIR}/prompts/no_camera_user_suffix.txt}"
WITH_CAMERA_SUFFIX_FILE="${WITH_CAMERA_SUFFIX_FILE:-${SCRIPT_DIR}/prompts/with_camera_user_suffix.txt}"

if [[ -z "${INDEX_DIR:-}" ]]; then
  INDEX_DIR="${V4TRAIN_EVAL_DIR}/test_index_splits/splits_16"
  if [[ ! -d "${INDEX_DIR}" && -d "${V4TRAIN_ROOT}/test_index_splits/splits_16" ]]; then
    INDEX_DIR="${V4TRAIN_ROOT}/test_index_splits/splits_16"
  fi
fi

DATA_ROOT="${WORK_ROOT}/data"
CANONICAL_CAMERA_JSONL="${DATA_ROOT}/vifbench_predicted_camera_context.jsonl"
CAMERA_CONTEXT_SUMMARY="${DATA_ROOT}/vifbench_predicted_camera_context_summary.json"
PROMPT_AUDIT="${DATA_ROOT}/vifbench_prompt_parity_audit.json"
INFERENCE_ROOT="${WORK_ROOT}/inference"
NO_CAMERA_PRED_DIR="${INFERENCE_ROOT}/no_camera/splitresults"
WITH_CAMERA_PRED_DIR="${INFERENCE_ROOT}/with_camera/splitresults"
EVAL_ROOT="${WORK_ROOT}/eval"
NO_CAMERA_MERGED="${EVAL_ROOT}/no_camera_predictions.json"
WITH_CAMERA_MERGED="${EVAL_ROOT}/with_camera_predictions.json"
NO_CAMERA_EVAL="${EVAL_ROOT}/no_camera_vifbench_eval.json"
WITH_CAMERA_EVAL="${EVAL_ROOT}/with_camera_vifbench_eval.json"
COMPARISON_JSON="${EVAL_ROOT}/explicit_camera_vifbench_comparison.json"
LOG_PATH="${WORK_ROOT}/pipeline.log"

PARALLEL_MODELS="${PARALLEL_MODELS:-1}"
KEEP_ALIVE_AFTER_RUN="${KEEP_ALIVE_AFTER_RUN:-0}"
KEEP_ALIVE_SCRIPT="${KEEP_ALIVE_SCRIPT:-/input/training/keep.sh}"

mkdir -p "${WORK_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

require_file() {
  [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 2; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 2; }
}

require_model() {
  require_dir "$1"
  require_file "$1/config.json"
  if ! find "$1" -maxdepth 1 -type f \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' \) | grep -q .; then
    echo "No full-model weights found directly under: $1" >&2
    exit 2
  fi
}

persist_small_results() {
  mkdir -p "${PERSIST_ROOT}"
  for file in "${PROMPT_AUDIT}" "${CAMERA_CONTEXT_SUMMARY}" "${CANONICAL_CAMERA_JSONL}"; do
    [[ -f "${file}" ]] && cp -a "${file}" "${PERSIST_ROOT}/"
  done
  if [[ -d "${EVAL_ROOT}" ]]; then
    mkdir -p "${PERSIST_ROOT}/eval"
    find "${EVAL_ROOT}" -maxdepth 1 -type f ! -name '*_predictions.json' \
      -exec cp -a {} "${PERSIST_ROOT}/eval/" \;
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

common_preflight() {
  require_file "${INFER_SCRIPT}"
  require_file "${INFERENCE_PY}"
  require_file "${OFFICIAL_EVAL_PY}"
  require_dir "${INDEX_DIR}"
  require_file "${DATAB_DETECTION_JSON}"
  require_file "${SYSTEM_PROMPT_FILE}"
  require_file "${NO_CAMERA_SUFFIX_FILE}"
  require_file "${WITH_CAMERA_SUFFIX_FILE}"
  grep -q -- '--camera_context_jsonl' "${INFERENCE_PY}" || {
    echo "${INFERENCE_PY} does not contain the required camera-context patch." >&2
    exit 2
  }
  grep -q 'CAMERA_CONTEXT_JSONL' "${INFER_SCRIPT}" || {
    echo "${INFER_SCRIPT} does not contain the required camera-context patch." >&2
    exit 2
  }
  mkdir -p "${DATA_ROOT}"
  "${PYTHON_BIN}" -m tools.prepare_vifbench_camera_context audit-prompts \
    --detection-json "${DATAB_DETECTION_JSON}" \
    --system-prompt-file "${SYSTEM_PROMPT_FILE}" \
    --no-camera-suffix-file "${NO_CAMERA_SUFFIX_FILE}" \
    --with-camera-suffix-file "${WITH_CAMERA_SUFFIX_FILE}" \
    --output-json "${PROMPT_AUDIT}"
  "${PYTHON_BIN}" -c "import torch; assert torch.cuda.device_count() == ${NUM_GPUS}; print('GPU runtime: OK')"
}

prepare_camera_context() {
  [[ -n "${VIF_CAMERA_JSON}" ]] || {
    echo "Set VIF_CAMERA_JSON to the completed ViF-Bench CameraBench labels+caption JSON/JSONL." >&2
    exit 2
  }
  require_file "${VIF_CAMERA_JSON}"
  mkdir -p "${DATA_ROOT}"
  "${PYTHON_BIN}" -m tools.prepare_vifbench_camera_context prepare \
    --index-dir "${INDEX_DIR}" \
    --camera-json "${VIF_CAMERA_JSON}" \
    --output-jsonl "${CANONICAL_CAMERA_JSONL}" \
    --summary-json "${CAMERA_CONTEXT_SUMMARY}" \
    --expected-ranks "${NUM_GPUS}" \
    --min-coverage 1.0
}

preflight_no_camera() {
  common_preflight
  require_model "${NO_CAMERA_MODEL}"
  echo "No-camera inference preflight passed. No model inference was run."
}

preflight_camera() {
  common_preflight
  require_model "${WITH_CAMERA_MODEL}"
  prepare_camera_context
  echo "Predicted-camera inference preflight passed with full sidecar coverage."
}

preflight_all() {
  common_preflight
  require_model "${NO_CAMERA_MODEL}"
  require_model "${WITH_CAMERA_MODEL}"
  prepare_camera_context
  echo "Paired ViF-Bench preflight passed. No model inference was run."
}

infer_one() {
  local prompt_mode="$1"
  local model_path="$2"
  local model_name="$3"
  local save_dir="$4"
  local suffix_file="$5"
  local camera_jsonl="${6:-}"
  local log_path="$7"
  mkdir -p "${save_dir}" "$(dirname "${log_path}")"
  local command=(
    env
    PYTHON_BIN="${PYTHON_BIN}"
    WORLD_SIZE="${NUM_GPUS}"
    PROMPT_MODE="${prompt_mode}"
    MODEL_PATH="${model_path}"
    MODEL_NAME="${model_name}"
    SAVE_DIR="${save_dir}"
    INDEX_DIR="${INDEX_DIR}"
    SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE}"
    USER_PROMPT_SUFFIX_FILE="${suffix_file}"
  )
  if [[ -n "${camera_jsonl}" ]]; then
    command+=(CAMERA_CONTEXT_JSONL="${camera_jsonl}")
  fi
  command+=(bash "${INFER_SCRIPT}")
  "${command[@]}" > "${log_path}" 2>&1
}

infer_no_camera() {
  infer_one no_camera "${NO_CAMERA_MODEL}" \
    Qwen3-VL-8B-DataB5739-no-camera-vifbench \
    "${NO_CAMERA_PRED_DIR}" "${NO_CAMERA_SUFFIX_FILE}" "" \
    "${INFERENCE_ROOT}/no_camera/inference.log"
}

infer_with_camera() {
  require_file "${CANONICAL_CAMERA_JSONL}"
  # The patched external script calls this mode gold_camera. The supplied context is predicted.
  infer_one gold_camera "${WITH_CAMERA_MODEL}" \
    Qwen3-VL-8B-DataB5739-predicted-camera-vifbench \
    "${WITH_CAMERA_PRED_DIR}" "${WITH_CAMERA_SUFFIX_FILE}" \
    "${CANONICAL_CAMERA_JSONL}" "${INFERENCE_ROOT}/with_camera/inference.log"
}

infer_both() {
  if [[ "${PARALLEL_MODELS}" == "1" ]]; then
    echo "Launching both full-SFT models concurrently: two inference processes per GPU."
    infer_no_camera &
    local no_camera_pid=$!
    infer_with_camera &
    local with_camera_pid=$!
    set +e
    wait "${no_camera_pid}"; local no_camera_status=$?
    wait "${with_camera_pid}"; local with_camera_status=$?
    set -e
    if [[ "${no_camera_status}" != "0" || "${with_camera_status}" != "0" ]]; then
      echo "Inference failed: no_camera=${no_camera_status}, with_camera=${with_camera_status}" >&2
      exit 1
    fi
  else
    infer_no_camera
    infer_with_camera
  fi
}

evaluate_one() {
  local prediction_dir="$1"
  local merged_json="$2"
  local eval_json="$3"
  local official_log="$4"
  "${PYTHON_BIN}" -m scripts.camera_detection_retention.vifbench_retention evaluate-one \
    --index-dir "${INDEX_DIR}" \
    --prediction-dir "${prediction_dir}" \
    --merged-json "${merged_json}" \
    --eval-json "${eval_json}" \
    --expected-ranks "${NUM_GPUS}"
  "${PYTHON_BIN}" "${OFFICIAL_EVAL_PY}" --json_file_path "${merged_json}" \
    | tee "${official_log}"
}

evaluate_all() {
  require_dir "${NO_CAMERA_PRED_DIR}"
  require_dir "${WITH_CAMERA_PRED_DIR}"
  require_file "${CAMERA_CONTEXT_SUMMARY}"
  mkdir -p "${EVAL_ROOT}"
  evaluate_one "${NO_CAMERA_PRED_DIR}" "${NO_CAMERA_MERGED}" \
    "${NO_CAMERA_EVAL}" "${EVAL_ROOT}/no_camera_official_eval.log"
  evaluate_one "${WITH_CAMERA_PRED_DIR}" "${WITH_CAMERA_MERGED}" \
    "${WITH_CAMERA_EVAL}" "${EVAL_ROOT}/with_camera_official_eval.log"
  "${PYTHON_BIN}" -m tools.compare_datab_explicit_camera_vifbench \
    --no-camera-eval "${NO_CAMERA_EVAL}" \
    --with-camera-eval "${WITH_CAMERA_EVAL}" \
    --camera-context-summary "${CAMERA_CONTEXT_SUMMARY}" \
    --output-json "${COMPARISON_JSON}"
  persist_small_results
}

launch_keepalive() {
  [[ "${KEEP_ALIVE_AFTER_RUN}" == "1" ]] || return
  require_file "${KEEP_ALIVE_SCRIPT}"
  persist_small_results
  trap - EXIT
  echo "ViF-Bench stage completed; starting keep-alive: ${KEEP_ALIVE_SCRIPT}"
  exec bash "${KEEP_ALIVE_SCRIPT}"
}

echo "=== DataB 显式 Camera labels+caption 的 ViF-Bench 配对推理 ==="
echo "stage=${STAGE}"
echo "no_camera_model=${NO_CAMERA_MODEL}"
echo "with_camera_model=${WITH_CAMERA_MODEL}"
echo "work_root=${WORK_ROOT}"
echo "camera_context=predicted CameraBench labels+caption; never called gold"

case "${STAGE}" in
  preflight) preflight_all ;;
  preflight_no_camera) preflight_no_camera ;;
  preflight_camera) preflight_camera ;;
  prepare) common_preflight; prepare_camera_context ;;
  infer_no_camera) preflight_no_camera; infer_no_camera; launch_keepalive ;;
  infer_with_camera) preflight_camera; infer_with_camera; launch_keepalive ;;
  infer_both) preflight_all; infer_both; launch_keepalive ;;
  eval) evaluate_all ;;
  all) preflight_all; infer_both; evaluate_all; launch_keepalive ;;
  *)
    echo "STAGE must be preflight, preflight_no_camera, preflight_camera, prepare, infer_no_camera, infer_with_camera, infer_both, eval, or all" >&2
    exit 2
    ;;
esac
