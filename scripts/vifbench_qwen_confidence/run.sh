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
V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR:-${PROJECT_ROOT}/eval/v4train-main/eval}"
INDEX_DIR="${INDEX_DIR:-${V4TRAIN_EVAL_DIR}/test_index_splits/splits_16}"
if [[ ! -d "${INDEX_DIR}" && -d "$(dirname "${V4TRAIN_EVAL_DIR}")/test_index_splits/splits_16" ]]; then
  INDEX_DIR="$(dirname "${V4TRAIN_EVAL_DIR}")/test_index_splits/splits_16"
fi

HISTORICAL_PREDICTIONS="${HISTORICAL_PREDICTIONS:-/input/workflow_58770161/workspace/test/test_selfcot/Skyra/res/v4vif_2766busterall_trainall/v4vif_2766busterall_trainall-3vl8b-vifbench/Qwen3-VL-v4vif_2766busterall_trainall-vifbench.json}"
EXPERT_ITEMS_CSV="${EXPERT_ITEMS_CSV:-${PROJECT_ROOT}/res/camera_discriminative_gate/v1/eval/vifbench/camera_discriminative_gate_items.csv}"
RUN_ROOT="${RUN_ROOT:-/tmp/1res/vifbench_qwen_confidence_fusion/v1}"
PERSIST_ROOT="${PERSIST_ROOT:-${PROJECT_ROOT}/res/vifbench_qwen_confidence_fusion/v1}"
SCORE_DIR="${RUN_ROOT}/confidence_shards"
PREFLIGHT_DIR="${RUN_ROOT}/preflight"
EVAL_DIR="${RUN_ROOT}/eval"
LOG_PATH="${RUN_ROOT}/pipeline.log"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
OVERWRITE="${OVERWRITE:-0}"
KEEP_ALIVE_AFTER_RUN="${KEEP_ALIVE_AFTER_RUN:-0}"
KEEP_ALIVE_SCRIPT="${KEEP_ALIVE_SCRIPT:-/input/training/keep.sh}"

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
  if [[ -d "${PREFLIGHT_DIR}" ]]; then
    mkdir -p "${PERSIST_ROOT}/preflight"
    cp -a "${PREFLIGHT_DIR}/." "${PERSIST_ROOT}/preflight/"
  fi
  if [[ -d "${SCORE_DIR}" ]]; then
    mkdir -p "${PERSIST_ROOT}/confidence_shards"
    cp -a "${SCORE_DIR}"/rank_*.jsonl "${PERSIST_ROOT}/confidence_shards/" 2>/dev/null || true
  fi
  if [[ -d "${EVAL_DIR}" ]]; then
    mkdir -p "${PERSIST_ROOT}/eval"
    cp -a "${EVAL_DIR}/." "${PERSIST_ROOT}/eval/"
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

common_checks() {
  require_dir "${MODEL_PATH}"
  require_file "${MODEL_PATH}/config.json"
  require_dir "${V4TRAIN_EVAL_DIR}"
  require_file "${V4TRAIN_EVAL_DIR}/utils/ViFBench.py"
  require_file "${V4TRAIN_EVAL_DIR}/models/Qwen3_VL.py"
  require_dir "${INDEX_DIR}"
  require_file "${HISTORICAL_PREDICTIONS}"
  require_file "${EXPERT_ITEMS_CSV}"
  require_file "${SCRIPT_DIR}/score_historical_answers.py"
  require_file "${REPO_ROOT}/tools/audit_vifbench_confidence_fusion.py"
}

preflight() {
  common_checks
  mkdir -p "${PREFLIGHT_DIR}"
  "${PYTHON_BIN}" -c "import torch, transformers, sklearn; assert torch.cuda.device_count() == ${NUM_GPUS}; print('GPU and Python runtime: OK')"
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" \
    "${SCRIPT_DIR}/score_historical_answers.py" \
    --model-path "${MODEL_PATH}" \
    --historical-predictions "${HISTORICAL_PREDICTIONS}" \
    --v4train-eval-dir "${V4TRAIN_EVAL_DIR}" \
    --index-json "${INDEX_DIR}/test_index.rank0.json" \
    --output-path "${PREFLIGHT_DIR}/confidence_preflight.json" \
    --preflight-only
  echo "Preflight passed. No Qwen forward pass was run."
}

score_all_ranks() {
  common_checks
  mkdir -p "${SCORE_DIR}"
  export PYTHON_BIN REPO_ROOT SCRIPT_DIR MODEL_PATH HISTORICAL_PREDICTIONS
  export V4TRAIN_EVAL_DIR INDEX_DIR SCORE_DIR ATTN_IMPLEMENTATION OVERWRITE
  seq 0 "$((NUM_GPUS - 1))" | xargs -n1 -P "${NUM_GPUS}" bash -lc '
    rank="$1"
    command=(
      "${PYTHON_BIN}" "${SCRIPT_DIR}/score_historical_answers.py"
      --model-path "${MODEL_PATH}"
      --historical-predictions "${HISTORICAL_PREDICTIONS}"
      --v4train-eval-dir "${V4TRAIN_EVAL_DIR}"
      --index-json "${INDEX_DIR}/test_index.rank${rank}.json"
      --output-path "${SCORE_DIR}/rank_$(printf "%02d" "${rank}").jsonl"
      --attn-implementation "${ATTN_IMPLEMENTATION}"
    )
    if [[ "${OVERWRITE}" == "1" ]]; then
      command+=(--overwrite)
    fi
    CUDA_VISIBLE_DEVICES="${rank}" PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 "${command[@]}"
  ' _
  echo "All confidence shards completed: ${SCORE_DIR}"
}

audit_fusion() {
  common_checks
  require_dir "${SCORE_DIR}"
  mkdir -p "${EVAL_DIR}"
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" \
    tools/audit_vifbench_confidence_fusion.py \
    --confidence-scores "${SCORE_DIR}" \
    --expert-items-csv "${EXPERT_ITEMS_CSV}" \
    --output-dir "${EVAL_DIR}"
  persist_small_results
}

launch_keepalive() {
  if [[ "${KEEP_ALIVE_AFTER_RUN}" != "1" ]]; then
    return
  fi
  require_file "${KEEP_ALIVE_SCRIPT}"
  persist_small_results
  trap - EXIT
  echo "Confidence-fusion audit completed. Starting keepalive: ${KEEP_ALIVE_SCRIPT}"
  exec bash "${KEEP_ALIVE_SCRIPT}"
}

echo "=== ViF-Bench 强检测答案置信度与时序/相机专家融合诊断 ==="
echo "stage=${STAGE}"
echo "model=${MODEL_PATH}"
echo "historical_predictions=${HISTORICAL_PREDICTIONS}"
echo "expert_items=${EXPERT_ITEMS_CSV}"
echo "run_root=${RUN_ROOT}"
echo "The historical hard answers are never regenerated or replaced."

case "${STAGE}" in
  preflight)
    preflight
    ;;
  score)
    score_all_ranks
    ;;
  audit)
    audit_fusion
    ;;
  all)
    preflight
    score_all_ranks
    audit_fusion
    launch_keepalive
    ;;
  *)
    echo "STAGE must be preflight, score, audit, or all" >&2
    exit 2
    ;;
esac
