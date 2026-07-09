#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"

PAIR_JSON="${PAIR_JSON:-/tmp/1res/dataa_pair_selection_probe/dataa_pair_selection_200.json}"
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-8B-v4vif-dataa-pair-selection}"
SAVE_DIR="${SAVE_DIR:-/tmp/1res/dataa_pair_selection_probe/${MODEL_NAME}}"
PRED_JSON="${PRED_JSON:-${1:-${SAVE_DIR}}}"
OUT_DIR="${OUT_DIR:-${SAVE_DIR}/eval}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-dataa_pair_selection_eval}"

"${PYTHON_BIN}" "${REPO_ROOT}/tools/eval_dataa_pair_selection.py" \
  --gt_json "${PAIR_JSON}" \
  --pred_json "${PRED_JSON}" \
  --out_dir "${OUT_DIR}" \
  --output_prefix "${OUTPUT_PREFIX}"
