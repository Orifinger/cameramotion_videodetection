#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"

V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR:-/input/workflow_58770161/workspace/test/test_selfcot/Skyra/eval}"
ABLATION_DIR="${ABLATION_DIR:-/tmp/1res/dataa_camera_context_ablation}"
PREFIX="${PREFIX:-dataa_test}"
VARIANT="${VARIANT:-all}"
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-8B-camera-context-ablation}"
SAVE_ROOT="${SAVE_ROOT:-${ABLATION_DIR}/${MODEL_NAME}}"
COMPUTE_IOU="${COMPUTE_IOU:-0}"
MATCH_TYPE="${MATCH_TYPE:-0}"

if [[ ! -f "${V4TRAIN_EVAL_DIR}/eval_dataa.py" ]]; then
  echo "Missing ${V4TRAIN_EVAL_DIR}/eval_dataa.py"
  exit 2
fi

if [[ "${VARIANT}" == "all" ]]; then
  VARIANTS=(no_camera gold_camera shuffled_camera null_camera)
else
  VARIANTS=(${VARIANT})
fi

for variant in "${VARIANTS[@]}"; do
  echo "=== Eval variant: ${variant} ==="
  CMD=(
    "${PYTHON_BIN}" "${V4TRAIN_EVAL_DIR}/eval_dataa.py"
    --gt_json "${ABLATION_DIR}/${PREFIX}_${variant}.json"
    --pred_json "${SAVE_ROOT}/${variant}"
    --out_dir "${SAVE_ROOT}/${variant}/eval"
    --output_prefix "dataa_${variant}_eval"
  )
  if [[ "${COMPUTE_IOU}" == "1" ]]; then
    CMD+=(--compute_iou)
  fi
  if [[ "${MATCH_TYPE}" == "1" ]]; then
    CMD+=(--match_type)
  fi
  "${CMD[@]}"
done
