#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"

GT_JSON="${GT_JSON:-/tmp/1res/counterfactual_gate/data/dataa_counterfactual_eval_local_only.json}"
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-8B-gate1-pair-baseline}"
PRED_JSON="${PRED_JSON:-/tmp/1res/gate1_pair_eval/${MODEL_NAME}}"
OUT_DIR="${OUT_DIR:-${PRED_JSON}/eval}"
FAIL_ON_GATE="${FAIL_ON_GATE:-0}"

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/eval/eval_dataa_counterfactual_pair_gate.py"
  --gt-json "${GT_JSON}"
  --pred-json "${PRED_JSON}"
  --out-dir "${OUT_DIR}"
)
if [[ "${FAIL_ON_GATE}" == "1" ]]; then CMD+=(--fail-on-gate); fi
"${CMD[@]}"
