#!/usr/bin/env bash
set -euo pipefail

V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR:-/input/workflow_58770161/workspace/test/test_selfcot/Skyra/eval}"
GT_JSON="${GT_JSON:-/tmp/1res/counterfactual_gate/data/dataa_counterfactual_eval_local_only.json}"
MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-8B-gate1-pair-baseline}"
SAVE_DIR="${SAVE_DIR:-/tmp/1res/gate1_pair_eval/${MODEL_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
WORLD_SIZE="${WORLD_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OVERWRITE="${OVERWRITE:-0}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-262144}"

if [[ ! -f "${V4TRAIN_EVAL_DIR}/infer_dataa.py" ]]; then
  echo "Missing ${V4TRAIN_EVAL_DIR}/infer_dataa.py" >&2
  exit 2
fi
if [[ ! -f "${GT_JSON}" || ! -d "${MODEL_PATH}" ]]; then
  echo "Missing GT JSON or model: GT_JSON=${GT_JSON} MODEL_PATH=${MODEL_PATH}" >&2
  exit 2
fi

mkdir -p "${SAVE_DIR}"
export PYTHON_BIN V4TRAIN_EVAL_DIR GT_JSON MODEL_PATH MODEL_NAME SAVE_DIR WORLD_SIZE
export MAX_NEW_TOKENS MAX_SAMPLES OVERWRITE IMAGE_MAX_PIXELS

seq 0 "$((WORLD_SIZE - 1))" | xargs -n1 -P "${WORLD_SIZE}" bash -lc '
  RANK="$1"
  CMD=(
    "${PYTHON_BIN}" "${V4TRAIN_EVAL_DIR}/infer_dataa.py"
    --sft_json "${GT_JSON}"
    --model_path "${MODEL_PATH}"
    --model_name "${MODEL_NAME}-rank${RANK}"
    --save_dir "${SAVE_DIR}/rank_${RANK}"
    --rank "${RANK}"
    --world_size "${WORLD_SIZE}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --prompt_mode record
    --image_max_pixels "${IMAGE_MAX_PIXELS}"
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then CMD+=(--max_samples "${MAX_SAMPLES}"); fi
  if [[ "${OVERWRITE}" == "1" ]]; then CMD+=(--overwrite); fi
  CUDA_VISIBLE_DEVICES="${RANK}" "${CMD[@]}"
' _

echo "Predictions: ${SAVE_DIR}"
