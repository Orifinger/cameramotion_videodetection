#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${STAGE:-preflight}"
PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
MODEL_PATH="${MODEL_PATH:-/home/admin/Qwen3-VL-8B-Instruct}"
WORK_ROOT="${WORK_ROOT:-/tmp/1res/datab_explicit_camera_sft/v1}"
DATA_DIR="${WORK_ROOT}/data"
TRAIN_ROOT="${WORK_ROOT}/train"
CONFIG_ROOT="${WORK_ROOT}/configs"
PERSIST_ROOT="${PERSIST_ROOT:-${PROJECT_ROOT}/res/datab_explicit_camera_sft/v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"
FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
EXPECTED_GPUS="${EXPECTED_GPUS:-16}"
CHECK_IMAGES="${CHECK_IMAGES:-1}"
KEEP_ALIVE_AFTER_RUN="${KEEP_ALIVE_AFTER_RUN:-0}"
KEEP_ALIVE_SCRIPT="${KEEP_ALIVE_SCRIPT:-/input/training/keep.sh}"

DATAB_DETECTION_JSON="${DATAB_DETECTION_JSON:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json}"
DATAB_CAMERA_JSONL="${DATAB_CAMERA_JSONL:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl}"

if [[ -z "${LLAMAFACTORY_ROOT:-}" ]]; then
  for candidate in \
    /input/workflow_58770161/workspace/test/test_selfcot/LlamaFactory/LlamaFactory \
    /input/workflow_58770161/workspace/test/test_selfcot/Skyra/train/LLaMA-Factory \
    /input/training/LlamaFactory/LlamaFactory
  do
    if [[ -d "${candidate}" ]]; then
      LLAMAFACTORY_ROOT="${candidate}"
      break
    fi
  done
fi
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-/input/workflow_58770161/workspace/test/test_selfcot/LlamaFactory/LlamaFactory}"

if [[ -z "${LLAMAFACTORY_DATA_DIR:-}" ]]; then
  for candidate in \
    /input/workflow_58770161/workspace/test/test_selfcot/Skyra/train/LLaMA-Factory/data \
    "${LLAMAFACTORY_ROOT}/data" \
    /input/training/LlamaFactory/LlamaFactory/data
  do
    if [[ -f "${candidate}/dataset_info.json" ]]; then
      LLAMAFACTORY_DATA_DIR="${candidate}"
      break
    fi
  done
fi
LLAMAFACTORY_DATA_DIR="${LLAMAFACTORY_DATA_DIR:-${LLAMAFACTORY_ROOT}/data}"

if [[ -z "${DEEPSPEED_CONFIG:-}" ]]; then
  for candidate in \
    "${LLAMAFACTORY_ROOT}/examples/deepspeed/ds_z2_config.json" \
    /input/workflow_58770161/workspace/test/test_selfcot/LlamaFactory/LlamaFactory/examples/deepspeed/ds_z2_config.json
  do
    if [[ -f "${candidate}" ]]; then
      DEEPSPEED_CONFIG="${candidate}"
      break
    fi
  done
fi
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${LLAMAFACTORY_ROOT}/examples/deepspeed/ds_z2_config.json}"

mkdir -p "${WORK_ROOT}"

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

require_layout() {
  require_dir "${MODEL_PATH}"
  require_file "${MODEL_PATH}/config.json"
  require_file "${DATAB_DETECTION_JSON}"
  require_file "${DATAB_CAMERA_JSONL}"
  require_dir "${LLAMAFACTORY_ROOT}"
  require_dir "${LLAMAFACTORY_DATA_DIR}"
  require_file "${LLAMAFACTORY_DATA_DIR}/dataset_info.json"
  require_file "${DEEPSPEED_CONFIG}"
  command -v "${LLAMAFACTORY_CLI}" >/dev/null
}

persist_small_results() {
  mkdir -p "${PERSIST_ROOT}"
  for file in \
    "${DATA_DIR}/datab_explicit_camera_sft_data_summary.json" \
    "${DATA_DIR}/datab_sft_pair_manifest.jsonl" \
    "${DATA_DIR}/llamafactory_install_summary.json"
  do
    if [[ -f "${file}" ]]; then
      cp -a "${file}" "${PERSIST_ROOT}/"
    fi
  done
  if [[ -d "${CONFIG_ROOT}" ]]; then
    mkdir -p "${PERSIST_ROOT}/configs"
    find "${CONFIG_ROOT}" -maxdepth 1 -type f -name '*.yaml' -exec cp -a {} "${PERSIST_ROOT}/configs/" \;
  fi
}

preflight() {
  require_layout
  "${PYTHON_BIN}" - "${EXPECTED_GPUS}" <<'PY'
import sys
import torch
import transformers
import yaml

expected = int(sys.argv[1])
actual = torch.cuda.device_count()
if actual != expected:
    raise SystemExit(f"expected {expected} GPUs, found {actual}")
print("Python dependencies: OK")
print("transformers:", transformers.__version__)
print("gpus:", actual)
PY
  echo "Preflight passed. No data was written and no model inference was run."
}

build_data() {
  require_layout
  mkdir -p "${DATA_DIR}"
  image_args=()
  if [[ "${CHECK_IMAGES}" == "1" ]]; then
    image_args+=(--check-images)
  fi
  "${PYTHON_BIN}" -m tools.build_datab_explicit_camera_sft \
    --detection-json "${DATAB_DETECTION_JSON}" \
    --camera-jsonl "${DATAB_CAMERA_JSONL}" \
    --out-dir "${DATA_DIR}" \
    --expected-detection-records 6766 \
    --expected-camera-records 5639 \
    --expected-matched-records 5739 \
    "${image_args[@]}"
  "${PYTHON_BIN}" -m tools.install_datab_explicit_camera_sft \
    --source-dir "${DATA_DIR}" \
    --llamafactory-data-dir "${LLAMAFACTORY_DATA_DIR}" \
    --expected-records 5739 \
    --smoke-samples 96 \
    --seed 20260718 \
    "${image_args[@]}"
  persist_small_results
}

branch_dataset() {
  case "$1" in
    no_camera) echo "datab_explicit_camera_no_camera" ;;
    with_camera) echo "datab_explicit_camera_labels_caption" ;;
    *) echo "Unknown branch: $1" >&2; exit 2 ;;
  esac
}

render_config() {
  local branch="$1"
  local mode="${2:-full}"
  local template="configs/datab_explicit_camera_sft/train_template.yaml"
  local dataset output destination
  dataset="$(branch_dataset "${branch}")"
  output="${TRAIN_ROOT}/${branch}"
  destination="${CONFIG_ROOT}/train_${branch}.yaml"
  if [[ "${mode}" == "smoke" ]]; then
    dataset="${dataset}_smoke"
    output="${WORK_ROOT}/smoke/${branch}"
    destination="${CONFIG_ROOT}/smoke_${branch}.yaml"
  fi
  require_file "${template}"
  mkdir -p "${CONFIG_ROOT}"
  sed \
    -e "s|__MODEL_PATH__|${MODEL_PATH}|g" \
    -e "s|__DEEPSPEED_CONFIG__|${DEEPSPEED_CONFIG}|g" \
    -e "s|__DATASET_DIR__|${LLAMAFACTORY_DATA_DIR}|g" \
    -e "s|__DATASET_NAME__|${dataset}|g" \
    -e "s|__OUTPUT_DIR__|${output}|g" \
    "${template}" > "${destination}"
  if [[ "${mode}" == "smoke" ]]; then
    sed -i \
      -e 's|num_train_epochs: 5.0|max_steps: 2|' \
      -e 's|save_steps: 500|save_strategy: "no"|' \
      -e 's|overwrite_output_dir: false|overwrite_output_dir: true|' \
      "${destination}"
  fi
  echo "${destination}"
}

train_branch() {
  local branch="$1"
  local mode="${2:-full}"
  local config
  config="$(render_config "${branch}" "${mode}")"
  FORCE_TORCHRUN="${FORCE_TORCHRUN}" "${LLAMAFACTORY_CLI}" train "${config}"
  persist_small_results
}

smoke() {
  train_branch no_camera smoke
  train_branch with_camera smoke
}

echo "=== DataB 显式 Camera labels+caption 检测 SFT ==="
echo "stage=${STAGE}"
echo "model=${MODEL_PATH}"
echo "work_root=${WORK_ROOT}"
echo "paired_train_records=5739"

case "${STAGE}" in
  preflight) preflight ;;
  build) build_data ;;
  smoke) smoke ;;
  train_no_camera) train_branch no_camera ;;
  train_with_camera) train_branch with_camera ;;
  train_both)
    train_branch no_camera
    train_branch with_camera
    ;;
  *)
    echo "Unknown STAGE=${STAGE}" >&2
    exit 2
    ;;
esac

if [[ "${KEEP_ALIVE_AFTER_RUN}" == "1" && "${STAGE}" == train_* ]]; then
  require_file "${KEEP_ALIVE_SCRIPT}"
  echo "Training stage completed; starting keep-alive: ${KEEP_ALIVE_SCRIPT}"
  exec bash "${KEEP_ALIVE_SCRIPT}"
fi
