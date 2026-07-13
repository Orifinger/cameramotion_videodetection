#!/usr/bin/env bash
set -euo pipefail

STAGE="${STAGE:-build}"
PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATAB_JSON="${DATAB_JSON:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json}"
MODEL_PATH="${MODEL_PATH:-/tmp/1res/models/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ}"
WORK_ROOT="${WORK_ROOT:-/tmp/1res/datab_deepfakejudge_gate}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SEED="${SEED:-20260713}"
CHECK_IMAGES="${CHECK_IMAGES:-1}"

DATA_DIR="${WORK_ROOT}/data"
PRED_DIR="${WORK_ROOT}/predictions"
EVAL_DIR="${WORK_ROOT}/eval"
INPUT_JSONL="${DATA_DIR}/datab_deepfakejudge_gate.jsonl"
INPUT_SUMMARY="${DATA_DIR}/datab_deepfakejudge_gate_build_summary.json"
PREDICTIONS_JSONL="${PRED_DIR}/predictions.jsonl"
EVAL_SUMMARY="${EVAL_DIR}/datab_deepfakejudge_gate_summary.json"
EVAL_ITEMS="${EVAL_DIR}/datab_deepfakejudge_gate_items.csv"

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

preflight() {
  require_file "${DATAB_JSON}"
  require_dir "${MODEL_PATH}"
  require_file "${MODEL_PATH}/config.json"
  require_file "${PROJECT_ROOT}/tools/build_datab_deepfakejudge_gate.py"
  require_file "${PROJECT_ROOT}/tools/eval_datab_deepfakejudge_gate.py"
  require_file "${PROJECT_ROOT}/scripts/datab_deepfakejudge/infer_pointwise.py"
  "${PYTHON_BIN}" - <<'PY'
import importlib

for name in ("torch", "transformers", "qwen_vl_utils"):
    module = importlib.import_module(name)
    print(f"OK import: {name} {getattr(module, '__version__', '')}")
PY
  echo "OK DataB: ${DATAB_JSON}"
  echo "OK model: ${MODEL_PATH}"
  echo "Validation output: ${WORK_ROOT}"
}

build() {
  mkdir -p "${DATA_DIR}"
  if [[ "${CHECK_IMAGES}" == "1" ]]; then
    "${PYTHON_BIN}" tools/build_datab_deepfakejudge_gate.py \
      --datab-json "${DATAB_JSON}" \
      --output-jsonl "${INPUT_JSONL}" \
      --summary-json "${INPUT_SUMMARY}" \
      --mode gate \
      --sample-size "${SAMPLE_SIZE}" \
      --seed "${SEED}" \
      --check-images
  else
    "${PYTHON_BIN}" tools/build_datab_deepfakejudge_gate.py \
      --datab-json "${DATAB_JSON}" \
      --output-jsonl "${INPUT_JSONL}" \
      --summary-json "${INPUT_SUMMARY}" \
      --mode gate \
      --sample-size "${SAMPLE_SIZE}" \
      --seed "${SEED}"
  fi
}

infer() {
  require_file "${INPUT_JSONL}"
  require_dir "${MODEL_PATH}"
  rm -rf "${PRED_DIR}"
  mkdir -p "${PRED_DIR}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.datab_deepfakejudge.infer_pointwise \
    --model-path "${MODEL_PATH}" \
    --input-jsonl "${INPUT_JSONL}" \
    --output-dir "${PRED_DIR}" \
    --max-pixels "${MAX_PIXELS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}"
}

evaluate() {
  require_file "${PREDICTIONS_JSONL}"
  require_file "${INPUT_SUMMARY}"
  mkdir -p "${EVAL_DIR}"
  "${PYTHON_BIN}" tools/eval_datab_deepfakejudge_gate.py \
    --predictions-jsonl "${PREDICTIONS_JSONL}" \
    --input-summary-json "${INPUT_SUMMARY}" \
    --output-json "${EVAL_SUMMARY}" \
    --output-csv "${EVAL_ITEMS}"
}

cd "${PROJECT_ROOT}"

case "${STAGE}" in
  preflight) preflight ;;
  build) build ;;
  infer) infer ;;
  eval) evaluate ;;
  all)
    preflight
    build
    infer
    evaluate
    ;;
  *)
    echo "Unknown STAGE=${STAGE}; use preflight, build, infer, eval, or all" >&2
    exit 2
    ;;
esac
