#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR:-${PROJECT_ROOT}/eval/v4train-main/eval}"
INDEX_DIR="${INDEX_DIR:-${V4TRAIN_EVAL_DIR}/test_index_splits/splits_16}"
if [[ ! -d "${INDEX_DIR}" && -d "$(dirname "${V4TRAIN_EVAL_DIR}")/test_index_splits/splits_16" ]]; then
  INDEX_DIR="$(dirname "${V4TRAIN_EVAL_DIR}")/test_index_splits/splits_16"
fi
HISTORICAL_PREDICTIONS="${HISTORICAL_PREDICTIONS:-/input/workflow_58770161/workspace/test/test_selfcot/Skyra/res/v4vif_2766busterall_trainall/v4vif_2766busterall_trainall-3vl8b-vifbench/Qwen3-VL-v4vif_2766busterall_trainall-vifbench.json}"
SMOKE_ROOT="${SMOKE_ROOT:-/tmp/1res/vifbench_qwen_confidence_fusion/v1/smoke}"
OUTPUT_PATH="${SMOKE_ROOT}/rank_00.jsonl"

mkdir -p "${SMOKE_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 \
"${PYTHON_BIN}" "${SCRIPT_DIR}/score_historical_answers.py" \
  --model-path "${MODEL_PATH}" \
  --historical-predictions "${HISTORICAL_PREDICTIONS}" \
  --v4train-eval-dir "${V4TRAIN_EVAL_DIR}" \
  --index-json "${INDEX_DIR}/test_index.rank0.json" \
  --output-path "${OUTPUT_PATH}" \
  --max-samples 2 \
  --overwrite

"${PYTHON_BIN}" -c '
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
assert len(rows) == 2, rows
assert all(row.get("status") == "ok" for row in rows), rows
assert all(row.get("score_matches_archived_answer") for row in rows), rows
print("Smoke passed: two historical answers reproduced by answer-token logits.")
' "${OUTPUT_PATH}"

echo "Smoke output: ${OUTPUT_PATH}"
