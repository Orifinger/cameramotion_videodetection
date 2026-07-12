#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
ADAPTER_PATH="${ADAPTER_PATH:?Set ADAPTER_PATH to control/final or pair_rank/final}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for the merged model}"

"${PYTHON_BIN}" -m scripts.caspr_gate1.merge_adapter \
  --model-path "${MODEL_PATH}" \
  --adapter-path "${ADAPTER_PATH}" \
  --output-dir "${OUTPUT_DIR}"
