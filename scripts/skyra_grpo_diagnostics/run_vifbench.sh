#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${STAGE:-all}"
PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_GPUS="${NUM_GPUS:-16}"
RUN_NAME="${RUN_NAME:-paper_asymmetric_inspection_evalable_100step_retry1}"
TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-/tmp/1res/skyra_grpo_diagnostics/${RUN_NAME}}"
RUN_ROOT="${RUN_ROOT:-${TRAIN_RUN_ROOT}/vifbench}"
PERSIST_ROOT="${PERSIST_ROOT:-${PROJECT_ROOT}/res/skyra_grpo_diagnostics/${RUN_NAME}/vifbench}"

STEP50_MODEL="${STEP50_MODEL:-${TRAIN_RUN_ROOT}/merged_step_50}"
STEP100_MODEL="${STEP100_MODEL:-${TRAIN_RUN_ROOT}/merged_step_100}"
BASE_EVAL_JSON="${BASE_EVAL_JSON:-${PROJECT_ROOT}/res/camera_detection_retention/vifbench_detection_checkpoint_start/eval/base_vifbench_eval.json}"

V4TRAIN_ROOT="${V4TRAIN_ROOT:-${PROJECT_ROOT}/eval/v4train-main}"
V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR:-${V4TRAIN_ROOT}/eval}"
INFER_SCRIPT="${INFER_SCRIPT:-${V4TRAIN_EVAL_DIR}/infer2_5_3.sh}"
OFFICIAL_EVAL_PY="${OFFICIAL_EVAL_PY:-${V4TRAIN_EVAL_DIR}/eval.py}"
INDEX_DIR="${INDEX_DIR:-${V4TRAIN_ROOT}/test_index_splits/splits_16}"
PROMPT_DIR="${PROMPT_DIR:-${V4TRAIN_EVAL_DIR}/prompts/camera_context}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-${PROMPT_DIR}/datab_detection_system_prompt.txt}"
USER_PROMPT_SUFFIX_FILE="${USER_PROMPT_SUFFIX_FILE:-${PROMPT_DIR}/datab_no_camera_user_suffix.txt}"

PARALLEL_MODELS="${PARALLEL_MODELS:-1}"
KEEP_ALIVE_AFTER_RUN="${KEEP_ALIVE_AFTER_RUN:-1}"
KEEP_ALIVE_SCRIPT="${KEEP_ALIVE_SCRIPT:-/input/training/keep.sh}"

STEP50_PRED_DIR="${RUN_ROOT}/inference/step50/splitresults"
STEP100_PRED_DIR="${RUN_ROOT}/inference/step100/splitresults"
EVAL_ROOT="${RUN_ROOT}/eval"
STEP50_MERGED_JSON="${EVAL_ROOT}/step50_predictions.json"
STEP100_MERGED_JSON="${EVAL_ROOT}/step100_predictions.json"
STEP50_EVAL_JSON="${EVAL_ROOT}/step50_vifbench_eval.json"
STEP100_EVAL_JSON="${EVAL_ROOT}/step100_vifbench_eval.json"
COMPARISON_JSON="${EVAL_ROOT}/vifbench_grpo_checkpoint_comparison.json"
LOG_PATH="${RUN_ROOT}/pipeline.log"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

require_file() {
  [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 2; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 2; }
}

persist_small_results() {
  mkdir -p "${PERSIST_ROOT}"
  if [[ -f "${RUN_ROOT}/vifbench_preflight.json" ]]; then
    cp -a "${RUN_ROOT}/vifbench_preflight.json" "${PERSIST_ROOT}/"
  fi
  if [[ -d "${EVAL_ROOT}" ]]; then
    find "${EVAL_ROOT}" -maxdepth 1 -type f \
      ! -name '*_predictions.json' -exec cp -a {} "${PERSIST_ROOT}/" \;
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
  require_dir "${STEP50_MODEL}"
  require_file "${STEP50_MODEL}/config.json"
  require_dir "${STEP100_MODEL}"
  require_file "${STEP100_MODEL}/config.json"
  require_file "${BASE_EVAL_JSON}"
  require_file "${INFER_SCRIPT}"
  require_file "${OFFICIAL_EVAL_PY}"
  require_dir "${INDEX_DIR}"
  require_file "${SYSTEM_PROMPT_FILE}"
  require_file "${USER_PROMPT_SUFFIX_FILE}"
  require_file "${KEEP_ALIVE_SCRIPT}"
  "${PYTHON_BIN}" -m scripts.skyra_grpo_diagnostics.vifbench_eval audit \
    --index-dir "${INDEX_DIR}" \
    --system-prompt-file "${SYSTEM_PROMPT_FILE}" \
    --user-prompt-suffix-file "${USER_PROMPT_SUFFIX_FILE}" \
    --output-json "${RUN_ROOT}/vifbench_preflight.json" \
    --expected-ranks "${NUM_GPUS}" \
    --check-frame-dirs
  "${PYTHON_BIN}" -c "import torch; assert torch.cuda.device_count() == ${NUM_GPUS}; print('GPU runtime: OK')"
  echo "Preflight passed. Base is reused from the prior identical-prompt VIF-Bench run."
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

run_inference() {
  if [[ "${PARALLEL_MODELS}" == "1" ]]; then
    echo "Running GRPO step 50 and step 100 concurrently: two inference processes per GPU."
    infer_one "${STEP50_MODEL}" "Qwen3-VL-8B-GRPO-step50-vifbench" \
      "${STEP50_PRED_DIR}" "${RUN_ROOT}/inference/step50/inference.log" &
    local step50_pid=$!
    infer_one "${STEP100_MODEL}" "Qwen3-VL-8B-GRPO-step100-vifbench" \
      "${STEP100_PRED_DIR}" "${RUN_ROOT}/inference/step100/inference.log" &
    local step100_pid=$!
    set +e
    wait "${step50_pid}"; local step50_status=$?
    wait "${step100_pid}"; local step100_status=$?
    set -e
    if [[ "${step50_status}" != "0" || "${step100_status}" != "0" ]]; then
      echo "Inference failed: step50=${step50_status}, step100=${step100_status}" >&2
      exit 1
    fi
  else
    echo "Running GRPO step 50 and step 100 sequentially."
    infer_one "${STEP50_MODEL}" "Qwen3-VL-8B-GRPO-step50-vifbench" \
      "${STEP50_PRED_DIR}" "${RUN_ROOT}/inference/step50/inference.log"
    infer_one "${STEP100_MODEL}" "Qwen3-VL-8B-GRPO-step100-vifbench" \
      "${STEP100_PRED_DIR}" "${RUN_ROOT}/inference/step100/inference.log"
  fi
}

evaluate_one() {
  local prediction_dir="$1"
  local merged_json="$2"
  local eval_json="$3"
  local official_log="$4"
  "${PYTHON_BIN}" -m scripts.skyra_grpo_diagnostics.vifbench_eval evaluate-one \
    --index-dir "${INDEX_DIR}" \
    --prediction-dir "${prediction_dir}" \
    --merged-json "${merged_json}" \
    --eval-json "${eval_json}" \
    --expected-ranks "${NUM_GPUS}"
  "${PYTHON_BIN}" "${OFFICIAL_EVAL_PY}" --json_file_path "${merged_json}" \
    | tee "${official_log}"
}

evaluate_all() {
  mkdir -p "${EVAL_ROOT}"
  evaluate_one "${STEP50_PRED_DIR}" "${STEP50_MERGED_JSON}" \
    "${STEP50_EVAL_JSON}" "${EVAL_ROOT}/step50_official_eval.log"
  evaluate_one "${STEP100_PRED_DIR}" "${STEP100_MERGED_JSON}" \
    "${STEP100_EVAL_JSON}" "${EVAL_ROOT}/step100_official_eval.log"
  "${PYTHON_BIN}" -m scripts.skyra_grpo_diagnostics.compare_vifbench \
    --base-json "${BASE_EVAL_JSON}" \
    --step50-json "${STEP50_EVAL_JSON}" \
    --step100-json "${STEP100_EVAL_JSON}" \
    --output-json "${COMPARISON_JSON}"
  for step in step50 step100; do
    local generated_csv="${EVAL_ROOT}/${step}_predictions_paired_metrics_transposed.csv"
    if [[ -f "${generated_csv}" ]]; then
      cp -a "${generated_csv}" "${EVAL_ROOT}/${step}_official_paired_metrics.csv"
    fi
  done
  persist_small_results
}

launch_keepalive() {
  [[ "${KEEP_ALIVE_AFTER_RUN}" == "1" ]] || return
  persist_small_results
  trap - EXIT
  echo "VIF-Bench evaluation completed. Starting keepalive: ${KEEP_ALIVE_SCRIPT}"
  exec bash "${KEEP_ALIVE_SCRIPT}"
}

echo "=== Full VIF-Bench evaluation for saved GRPO checkpoints ==="
echo "stage=${STAGE}"
echo "step50_model=${STEP50_MODEL}"
echo "step100_model=${STEP100_MODEL}"
echo "base_eval=${BASE_EVAL_JSON}"
echo "run_root=${RUN_ROOT}"
echo "prompt_mode=no_camera"

case "${STAGE}" in
  preflight)
    preflight
    ;;
  infer)
    run_inference
    ;;
  eval)
    evaluate_all
    ;;
  all)
    preflight
    run_inference
    evaluate_all
    launch_keepalive
    ;;
  *)
    echo "STAGE must be preflight, infer, eval, or all" >&2
    exit 2
    ;;
esac
