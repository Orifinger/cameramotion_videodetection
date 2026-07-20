#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERT_ITEMS_CSV="${EXPERT_ITEMS_CSV:-${PROJECT_ROOT}/res/camera_discriminative_gate/v1/eval/vifbench/camera_discriminative_gate_items.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/res/vifbench_residual_complementarity/v1}"
EXPERT_SCORE_COLUMN="${EXPERT_SCORE_COLUMN:-evidence_only_score}"
EXPERT_PREDICTION_COLUMN="${EXPERT_PREDICTION_COLUMN:-evidence_only_prediction}"

resolve_qwen_predictions() {
  local candidate
  for candidate in \
    "${PROJECT_ROOT}/res/camera_detection_retention/vifbench_detection_checkpoint_start/combined_predictions/base.json" \
    "${PROJECT_ROOT}/res/camera_detection_retention/vifbench_detection_checkpoint_start/eval/combined/base_predictions.json" \
    "/tmp/1res/camera_detection_retention/vifbench_detection_checkpoint_start/combined_predictions/base.json" \
    "/tmp/1res/camera_detection_retention/vifbench_detection_checkpoint_start/inference/base/splitresults"
  do
    if [[ -f "${candidate}" || -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

if [[ -z "${QWEN_PREDICTIONS:-}" ]]; then
  if ! QWEN_PREDICTIONS="$(resolve_qwen_predictions)"; then
    echo "Cannot find the strict-prompt Qwen ViF-Bench predictions." >&2
    echo "Set QWEN_PREDICTIONS to base.json or its rank-result directory." >&2
    exit 2
  fi
fi

if [[ ! -f "${QWEN_PREDICTIONS}" && ! -d "${QWEN_PREDICTIONS}" ]]; then
  echo "Missing Qwen predictions: ${QWEN_PREDICTIONS}" >&2
  exit 2
fi
if [[ ! -f "${EXPERT_ITEMS_CSV}" ]]; then
  echo "Missing temporal-expert item CSV: ${EXPERT_ITEMS_CSV}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

echo "=== ViF-Bench residual-error complementarity audit ==="
echo "Qwen predictions: ${QWEN_PREDICTIONS}"
echo "Temporal expert:  ${EXPERT_ITEMS_CSV}"
echo "Expert score:     ${EXPERT_SCORE_COLUMN}"
echo "Output:           ${OUTPUT_DIR}"
echo "CPU-only analysis: no Qwen inference and no model training."

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" \
  tools/audit_vifbench_residual_complementarity.py \
  --qwen-predictions "${QWEN_PREDICTIONS}" \
  --expert-items-csv "${EXPERT_ITEMS_CSV}" \
  --expert-score-column "${EXPERT_SCORE_COLUMN}" \
  --expert-prediction-column "${EXPERT_PREDICTION_COLUMN}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@" \
  | tee "${OUTPUT_DIR}/run.log"

echo "Summary: ${OUTPUT_DIR}/vifbench_residual_complementarity_summary.json"
echo "Items:   ${OUTPUT_DIR}/vifbench_residual_complementarity_items.csv"
echo "Report:  ${OUTPUT_DIR}/vifbench_residual_complementarity_report.md"