#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-/input/workflow_58770161/workspace/test/test_selfcot/LlamaFactory/LlamaFactory}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"
MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
ADAPTER_PATH="${ADAPTER_PATH:-/tmp/1res/gate1_pair_dpo_local_only}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-/tmp/1res/gate1_pair_dpo_local_only_merged}"
CONFIG="${REPO_ROOT}/configs/gate1/qwen3vl8b_pair_dpo_merge.yaml"

if [[ ! -d "${MODEL_PATH}" || ! -d "${ADAPTER_PATH}" ]]; then
  echo "Missing model or adapter: MODEL_PATH=${MODEL_PATH} ADAPTER_PATH=${ADAPTER_PATH}" >&2
  exit 2
fi

cd "${LLAMAFACTORY_ROOT}"
"${LLAMAFACTORY_CLI}" export "${CONFIG}" \
  model_name_or_path="${MODEL_PATH}" \
  adapter_name_or_path="${ADAPTER_PATH}" \
  export_dir="${MERGED_MODEL_DIR}"

echo "Merged model: ${MERGED_MODEL_DIR}"
