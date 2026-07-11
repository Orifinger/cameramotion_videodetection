#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"

LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-/input/workflow_58770161/workspace/test/test_selfcot/LlamaFactory/LlamaFactory}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"
LLAMAFACTORY_DATA_DIR="${LLAMAFACTORY_DATA_DIR:-${LLAMAFACTORY_ROOT}/data}"
SOURCE_JSON="${SOURCE_JSON:-/tmp/1res/counterfactual_gate/data/dataa_counterfactual_dpo_local_only.json}"
MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${LLAMAFACTORY_ROOT}/examples/deepspeed/ds_z2_config.json}"
TRAIN_MODE="${TRAIN_MODE:-smoke}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-262144}"
CUTOFF_LEN="${CUTOFF_LEN:-49152}"

if [[ ! -d "${LLAMAFACTORY_ROOT}" ]]; then
  echo "Missing LlamaFactory root: ${LLAMAFACTORY_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_JSON}" ]]; then
  echo "Missing Gate 1 preference data: ${SOURCE_JSON}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing initial checkpoint: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "Missing DeepSpeed config: ${DEEPSPEED_CONFIG}" >&2
  exit 2
fi
if ! command -v "${LLAMAFACTORY_CLI}" >/dev/null 2>&1; then
  echo "Cannot find ${LLAMAFACTORY_CLI}; activate the environment used for prior LlamaFactory SFT." >&2
  exit 2
fi

"${PYTHON_BIN}" "${REPO_ROOT}/tools/install_gate1_llamafactory_data.py" \
  --source-json "${SOURCE_JSON}" \
  --llamafactory-data-dir "${LLAMAFACTORY_DATA_DIR}" \
  --smoke-samples 64 \
  --seed 20260711 \
  --check-image-files

case "${TRAIN_MODE}" in
  smoke)
    CONFIG="${REPO_ROOT}/configs/gate1/qwen3vl8b_pair_dpo_smoke.yaml"
    DATASET_NAME="dataa_counterfactual_dpo_local_only_smoke"
    OUTPUT_DIR="${OUTPUT_DIR:-/tmp/1res/gate1_pair_dpo_smoke}"
    ;;
  full)
    CONFIG="${REPO_ROOT}/configs/gate1/qwen3vl8b_pair_dpo_full.yaml"
    DATASET_NAME="dataa_counterfactual_dpo_local_only"
    OUTPUT_DIR="${OUTPUT_DIR:-/tmp/1res/gate1_pair_dpo_local_only}"
    ;;
  *)
    echo "TRAIN_MODE must be smoke or full, got: ${TRAIN_MODE}" >&2
    exit 2
    ;;
esac

echo "=== Gate 1 LlamaFactory DPO ==="
echo "mode=${TRAIN_MODE}"
echo "model=${MODEL_PATH}"
echo "dataset=${DATASET_NAME}"
echo "output=${OUTPUT_DIR}"
echo "gpus=${CUDA_VISIBLE_DEVICES}"

cd "${LLAMAFACTORY_ROOT}"
export CUDA_VISIBLE_DEVICES
export FORCE_TORCHRUN=1
"${LLAMAFACTORY_CLI}" train "${CONFIG}" \
  model_name_or_path="${MODEL_PATH}" \
  dataset_dir="${LLAMAFACTORY_DATA_DIR}" \
  dataset="${DATASET_NAME}" \
  deepspeed="${DEEPSPEED_CONFIG}" \
  image_max_pixels="${IMAGE_MAX_PIXELS}" \
  cutoff_len="${CUTOFF_LEN}" \
  output_dir="${OUTPUT_DIR}"
