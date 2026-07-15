#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${STAGE:-preflight}"
PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
HARD_ROUTE_ROOT="${HARD_ROUTE_ROOT:-/tmp/1res/camera_hard_route_gate/v1}"
HARD_ROUTE_DATA_DIR="${HARD_ROUTE_DATA_DIR:-${HARD_ROUTE_ROOT}/data}"
BINARY_AUDIT_SUMMARY="${BINARY_AUDIT_SUMMARY:-${HARD_ROUTE_ROOT}/routes/dataa_binary_route_summary.json}"
CAMERA_ROUTE_ADAPTER="${CAMERA_ROUTE_ADAPTER:-${HARD_ROUTE_ROOT}/train/router}"

WORK_ROOT="${WORK_ROOT:-/tmp/1res/camera_binary_route_gate/v1}"
DATA_DIR="${WORK_ROOT}/data"
TRAIN_ROOT="${WORK_ROOT}/train"
ROUTE_ROOT="${WORK_ROOT}/routes"
VIF_ROOT="${WORK_ROOT}/vifbench"
CONFIG_DIR="${WORK_ROOT}/configs"
PERSIST_ROOT="${PERSIST_ROOT:-${PROJECT_ROOT}/res/camera_binary_route_gate/v1}"

SHARED_ADAPTER="${SHARED_ADAPTER:-${TRAIN_ROOT}/shared}"
NO_MOTION_ADAPTER="${NO_MOTION_ADAPTER:-${TRAIN_ROOT}/no_motion}"
MOTION_ADAPTER="${MOTION_ADAPTER:-${TRAIN_ROOT}/motion}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
CHECK_IMAGES="${CHECK_IMAGES:-1}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"
FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
KEEP_ALIVE_AFTER_RUN="${KEEP_ALIVE_AFTER_RUN:-0}"
KEEP_ALIVE_SCRIPT="${KEEP_ALIVE_SCRIPT:-/input/training/keep.sh}"

V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR:-${PROJECT_ROOT}/eval/v4train-main/eval}"
OFFICIAL_EVAL_PY="${OFFICIAL_EVAL_PY:-${V4TRAIN_EVAL_DIR}/eval.py}"
if [[ -z "${INDEX_DIR:-}" ]]; then
  INDEX_DIR="${V4TRAIN_EVAL_DIR}/test_index_splits/splits_16"
  alternate_index="$(dirname "${V4TRAIN_EVAL_DIR}")/test_index_splits/splits_16"
  if [[ ! -d "${INDEX_DIR}" && -d "${alternate_index}" ]]; then
    INDEX_DIR="${alternate_index}"
  fi
fi

if [[ -z "${LLAMAFACTORY_ROOT:-}" ]]; then
  for candidate in \
    /input/workflow_58770161/workspace/test/test_selfcot/LlamaFactory/LlamaFactory \
    /input/workflow_58770161/workspace/test/test_selfcot/Skyra/train/LLaMA-Factory \
    /input/training/LlamaFactory/LlamaFactory
  do
    if [[ -f "${candidate}/examples/deepspeed/ds_z2_config.json" ]]; then
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

VIF_ROUTE_INPUT="${ROUTE_ROOT}/vifbench_route_questions.jsonl"
VIF_ROUTE_INPUT_SUMMARY="${ROUTE_ROOT}/vifbench_route_input_summary.json"
VIF_ROUTE_SCORE_DIR="${ROUTE_ROOT}/vifbench_scores"
VIF_THREE_ROUTE_MANIFEST="${ROUTE_ROOT}/vifbench_three_class_route_manifest.jsonl"
VIF_THREE_ROUTE_SUMMARY="${ROUTE_ROOT}/vifbench_three_class_route_summary.json"
VIF_BINARY_ROUTE_MANIFEST="${ROUTE_ROOT}/vifbench_binary_route_manifest.jsonl"
VIF_BINARY_ROUTE_SUMMARY="${ROUTE_ROOT}/vifbench_binary_route_summary.json"

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

adapter_check() {
  require_dir "$1"
  require_file "$1/adapter_config.json"
  if [[ ! -f "$1/adapter_model.safetensors" && ! -f "$1/adapter_model.bin" ]]; then
    echo "Missing adapter weights under $1" >&2
    exit 2
  fi
}

persist_small_results() {
  mkdir -p "${PERSIST_ROOT}/data" "${PERSIST_ROOT}/routes" "${PERSIST_ROOT}/eval"
  for file in \
    "${DATA_DIR}/camera_binary_route_data_summary.json" \
    "${DATA_DIR}/llamafactory_install_summary.json"
  do
    if [[ -f "${file}" ]]; then
      cp -a "${file}" "${PERSIST_ROOT}/data/"
    fi
  done
  for file in \
    "${VIF_ROUTE_INPUT_SUMMARY}" \
    "${VIF_THREE_ROUTE_MANIFEST}" \
    "${VIF_THREE_ROUTE_SUMMARY}" \
    "${VIF_BINARY_ROUTE_MANIFEST}" \
    "${VIF_BINARY_ROUTE_SUMMARY}"
  do
    if [[ -f "${file}" ]]; then
      cp -a "${file}" "${PERSIST_ROOT}/routes/"
    fi
  done
  if [[ -d "${VIF_ROOT}/composed" ]]; then
    find "${VIF_ROOT}/composed" -maxdepth 1 -type f \
      \( -name '*summary.json' -o -name '*gate.json' -o -name '*.csv' -o -name '*.log' \) \
      -exec cp -a {} "${PERSIST_ROOT}/eval/" \;
  fi
}

preflight() {
  require_dir "${MODEL_PATH}"
  require_file "${MODEL_PATH}/config.json"
  require_file "${BINARY_AUDIT_SUMMARY}"
  adapter_check "${CAMERA_ROUTE_ADAPTER}"
  for name in shared no_motion minor_motion complex_motion; do
    require_file "${HARD_ROUTE_DATA_DIR}/hard_route_${name}.json"
  done
  require_file "${LLAMAFACTORY_ROOT}/examples/deepspeed/ds_z2_config.json"
  require_file "${LLAMAFACTORY_DATA_DIR}/dataset_info.json"
  require_file "${OFFICIAL_EVAL_PY}"
  require_dir "${INDEX_DIR}"
  require_file "${KEEP_ALIVE_SCRIPT}"
  command -v "${LLAMAFACTORY_CLI}" >/dev/null
  "${PYTHON_BIN}" - "${BINARY_AUDIT_SUMMARY}" "${NPROC_PER_NODE}" <<'PY'
import json
import sys
import torch

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if summary.get("status") != "passed" or not all(summary.get("checks", {}).values()):
    raise SystemExit("binary DataA route audit did not pass")
expected_gpus = int(sys.argv[2])
if torch.cuda.device_count() != expected_gpus:
    raise SystemExit(f"expected {expected_gpus} GPUs, found {torch.cuda.device_count()}")
print("Binary route audit and GPU count: OK")
PY
  "${PYTHON_BIN}" -m scripts.camera_binary_route_gate.route --help >/dev/null
  echo "Preflight passed. No model loading, training, or inference was run."
}

build_data() {
  mkdir -p "${DATA_DIR}"
  image_args=()
  if [[ "${CHECK_IMAGES}" == "1" ]]; then
    image_args+=(--check-images)
  fi
  "${PYTHON_BIN}" -m tools.build_camera_binary_route_gate \
    --hard-route-data-dir "${HARD_ROUTE_DATA_DIR}" \
    --binary-audit-summary "${BINARY_AUDIT_SUMMARY}" \
    --out-dir "${DATA_DIR}" \
    --seed 20260715 \
    "${image_args[@]}"
  "${PYTHON_BIN}" -m tools.install_camera_binary_route_gate \
    --source-dir "${DATA_DIR}" \
    --llamafactory-data-dir "${LLAMAFACTORY_DATA_DIR}" \
    --smoke-samples 96 \
    --seed 20260715 \
    "${image_args[@]}"
  persist_small_results
}

branch_spec() {
  case "$1" in
    shared) echo "camera_binary_route_shared|${SHARED_ADAPTER}" ;;
    no_motion) echo "camera_binary_route_no_motion|${NO_MOTION_ADAPTER}" ;;
    motion) echo "camera_binary_route_motion|${MOTION_ADAPTER}" ;;
    *) echo "Unknown binary training branch: $1" >&2; exit 2 ;;
  esac
}

render_config() {
  local branch="$1"
  local template="configs/camera_hard_route_gate/train_template.yaml"
  local spec dataset_name output_dir destination
  spec="$(branch_spec "${branch}")"
  dataset_name="${spec%%|*}"
  output_dir="${spec#*|}"
  destination="${CONFIG_DIR}/train_${branch}.yaml"
  require_file "${template}"
  mkdir -p "${CONFIG_DIR}"
  sed \
    -e "s|__MODEL_PATH__|${MODEL_PATH}|g" \
    -e "s|__DEEPSPEED_CONFIG__|${LLAMAFACTORY_ROOT}/examples/deepspeed/ds_z2_config.json|g" \
    -e "s|__DATASET_DIR__|${LLAMAFACTORY_DATA_DIR}|g" \
    -e "s|__DATASET_NAME__|${dataset_name}|g" \
    -e "s|__OUTPUT_DIR__|${output_dir}|g" \
    "${template}" > "${destination}"
  echo "${destination}"
}

train_branch() {
  local config
  config="$(render_config "$1")"
  FORCE_TORCHRUN="${FORCE_TORCHRUN}" "${LLAMAFACTORY_CLI}" train "${config}"
}

smoke_train() {
  local template="configs/camera_hard_route_gate/train_smoke.yaml"
  local destination="${CONFIG_DIR}/train_smoke.yaml"
  require_file "${template}"
  mkdir -p "${CONFIG_DIR}"
  sed \
    -e "s|__MODEL_PATH__|${MODEL_PATH}|g" \
    -e "s|__DEEPSPEED_CONFIG__|${LLAMAFACTORY_ROOT}/examples/deepspeed/ds_z2_config.json|g" \
    -e "s|__DATASET_DIR__|${LLAMAFACTORY_DATA_DIR}|g" \
    -e "s|__OUTPUT_DIR__|${WORK_ROOT}/smoke|g" \
    -e "s|camera_hard_route_shared_smoke|camera_binary_route_shared_smoke|g" \
    "${template}" > "${destination}"
  FORCE_TORCHRUN="${FORCE_TORCHRUN}" "${LLAMAFACTORY_CLI}" train "${destination}"
}

build_vif_route_inputs() {
  mkdir -p "${ROUTE_ROOT}"
  "${PYTHON_BIN}" -m scripts.camera_hard_route_gate.route_manifest build-vif-inputs \
    --index-dir "${INDEX_DIR}" \
    --output-jsonl "${VIF_ROUTE_INPUT}" \
    --summary-json "${VIF_ROUTE_INPUT_SUMMARY}" \
    --expected-ranks 16 \
    --expected-frames 16 \
    --check-frame-dirs \
    --require-timestamps
  persist_small_results
}

score_vif_route() {
  adapter_check "${CAMERA_ROUTE_ADAPTER}"
  require_file "${VIF_ROUTE_INPUT}"
  rm -rf "${VIF_ROUTE_SCORE_DIR}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.camera_joint_sft_gate.score_binary \
    --model-path "${MODEL_PATH}" \
    --adapter-path "${CAMERA_ROUTE_ADAPTER}" \
    --condition "route=${VIF_ROUTE_INPUT}" \
    --output-dir "${VIF_ROUTE_SCORE_DIR}" \
    --model-stage vifbench_binary_route_source_scores \
    --max-pixels "${MAX_PIXELS}" \
    --seed 20260715
}

aggregate_vif_route() {
  "${PYTHON_BIN}" -m scripts.camera_hard_route_gate.route_manifest aggregate \
    --input-jsonl "${VIF_ROUTE_INPUT}" \
    --prediction-dir "${VIF_ROUTE_SCORE_DIR}" \
    --output-manifest "${VIF_THREE_ROUTE_MANIFEST}" \
    --summary-json "${VIF_THREE_ROUTE_SUMMARY}" \
    --min-top-probability 0.0 \
    --min-margin 0.0
  "${PYTHON_BIN}" -m scripts.camera_binary_route_gate.route map-manifest \
    --input-manifest "${VIF_THREE_ROUTE_MANIFEST}" \
    --output-manifest "${VIF_BINARY_ROUTE_MANIFEST}" \
    --output-summary "${VIF_BINARY_ROUTE_SUMMARY}"
  persist_small_results
}

run_vif_branch() {
  local name="$1"
  local adapter="$2"
  adapter_check "${adapter}"
  local branch_root="${VIF_ROOT}/${name}"
  if [[ "${name}" == "shared" ]]; then
    STAGE=all \
    MODEL_PATH="${MODEL_PATH}" \
    ADAPTER_PATH="${adapter}" \
    RUN_ROOT="${branch_root}" \
    PERSIST_ROOT="${PERSIST_ROOT}/vif_branches/${name}" \
    INDEX_DIR="${INDEX_DIR}" \
    CAMERA_MODEL_NAME="Qwen3-VL-8B-binary-route-${name}" \
    PARALLEL_MODELS=1 \
    KEEP_ALIVE_AFTER_RUN=0 \
    bash scripts/camera_detection_retention/run_vifbench.sh
  else
    STAGE=adapter_only \
    MODEL_PATH="${MODEL_PATH}" \
    ADAPTER_PATH="${adapter}" \
    RUN_ROOT="${branch_root}" \
    PERSIST_ROOT="${PERSIST_ROOT}/vif_branches/${name}" \
    INDEX_DIR="${INDEX_DIR}" \
    CAMERA_MODEL_NAME="Qwen3-VL-8B-binary-route-${name}" \
    KEEP_ALIVE_AFTER_RUN=0 \
    bash scripts/camera_detection_retention/run_vifbench.sh
  fi
}

compose_vif() {
  require_file "${VIF_BINARY_ROUTE_MANIFEST}"
  local shared_predictions="${VIF_ROOT}/shared/inference/camera_adapter/splitresults"
  local no_predictions="${VIF_ROOT}/no_motion/inference/camera_adapter/splitresults"
  local motion_predictions="${VIF_ROOT}/motion/inference/camera_adapter/splitresults"
  require_dir "${shared_predictions}"
  require_dir "${no_predictions}"
  require_dir "${motion_predictions}"
  local output_dir="${VIF_ROOT}/composed"
  mkdir -p "${output_dir}"
  for mode in shared predicted wrong; do
    "${PYTHON_BIN}" -m scripts.camera_binary_route_gate.route compose \
      --index-dir "${INDEX_DIR}" \
      --route-manifest "${VIF_BINARY_ROUTE_MANIFEST}" \
      --expert "no-motion=${no_predictions}" \
      --expert "motion=${motion_predictions}" \
      --shared-prediction-dir "${shared_predictions}" \
      --route-mode "${mode}" \
      --output-predictions "${output_dir}/${mode}_predictions.json" \
      --output-summary "${output_dir}/${mode}_summary.json" \
      --expected-ranks 16
    "${PYTHON_BIN}" "${OFFICIAL_EVAL_PY}" \
      --json_file_path "${output_dir}/${mode}_predictions.json" \
      | tee "${output_dir}/${mode}_official_eval.log"
  done
  "${PYTHON_BIN}" -m scripts.camera_binary_route_gate.route summarize \
    --base-eval "${VIF_ROOT}/shared/eval/base_vifbench_eval.json" \
    --shared-summary "${output_dir}/shared_summary.json" \
    --predicted-summary "${output_dir}/predicted_summary.json" \
    --wrong-summary "${output_dir}/wrong_summary.json" \
    --route-summary "${VIF_BINARY_ROUTE_SUMMARY}" \
    --output-json "${output_dir}/camera_binary_route_gate.json"
  persist_small_results
}

echo "=== 静止/有运动二路硬路由检测专家验证 ==="
echo "stage=${STAGE}"
echo "base_model=${MODEL_PATH}"
echo "router_adapter=${CAMERA_ROUTE_ADAPTER}"
echo "work_root=${WORK_ROOT}"
echo "camera_text_at_detection_inference=false"

case "${STAGE}" in
  preflight) preflight ;;
  build) build_data ;;
  smoke) smoke_train ;;
  train_shared) train_branch shared ;;
  train_no_motion) train_branch no_motion ;;
  train_motion) train_branch motion ;;
  train_experts)
    train_branch no_motion
    train_branch motion
    ;;
  train_all)
    train_branch shared
    train_branch no_motion
    train_branch motion
    ;;
  build_vif_route_inputs) build_vif_route_inputs ;;
  score_vif_route) score_vif_route ;;
  aggregate_vif_route) aggregate_vif_route ;;
  build_vif_route)
    build_vif_route_inputs
    score_vif_route
    aggregate_vif_route
    ;;
  vif_shared) run_vif_branch shared "${SHARED_ADAPTER}" ;;
  vif_no_motion) run_vif_branch no_motion "${NO_MOTION_ADAPTER}" ;;
  vif_motion) run_vif_branch motion "${MOTION_ADAPTER}" ;;
  vif_experts)
    run_vif_branch no_motion "${NO_MOTION_ADAPTER}"
    run_vif_branch motion "${MOTION_ADAPTER}"
    ;;
  vif_all)
    run_vif_branch shared "${SHARED_ADAPTER}"
    run_vif_branch no_motion "${NO_MOTION_ADAPTER}"
    run_vif_branch motion "${MOTION_ADAPTER}"
    compose_vif
    ;;
  compose_vif) compose_vif ;;
  *)
    echo "Unknown STAGE=${STAGE}" >&2
    exit 2
    ;;
esac

persist_small_results
if [[ "${KEEP_ALIVE_AFTER_RUN}" == "1" ]]; then
  require_file "${KEEP_ALIVE_SCRIPT}"
  echo "Stage ${STAGE} completed. Starting ${KEEP_ALIVE_SCRIPT}."
  exec bash "${KEEP_ALIVE_SCRIPT}"
fi
