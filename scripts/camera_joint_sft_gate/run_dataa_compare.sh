#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${STAGE:-all}"
MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
WORK_ROOT="${WORK_ROOT:-/tmp/1res/camera_joint_sft_gate}"
RUN_ROOT="${RUN_ROOT:-${WORK_ROOT}/dataa_four_model_compare}"
PERSIST_ROOT="${PERSIST_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_joint_sft_gate/dataa_four_model_compare}"
DATAA_SPLIT="${DATAA_SPLIT:-${WORK_ROOT}/data/dataa_test_detection.json}"
DETECTION_ONLY_ADAPTER="${DETECTION_ONLY_ADAPTER:-${WORK_ROOT}/train/detection_only}"
CORRECT_ADAPTER="${CORRECT_ADAPTER:-${WORK_ROOT}/train/correct_camera}"
FLIPPED_ADAPTER="${FLIPPED_ADAPTER:-${WORK_ROOT}/train/shuffled_camera}"
V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR:-/input/workflow_58770161/workspace/test/test_selfcot/Skyra/eval}"
KEEP_ALIVE_AFTER_RUN="${KEEP_ALIVE_AFTER_RUN:-0}"
KEEP_ALIVE_SCRIPT="${KEEP_ALIVE_SCRIPT:-/input/training/keep.sh}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${V4TRAIN_EVAL_DIR}/infer_dataa.py" ]]; then
  V4TRAIN_EVAL_DIR="${REPO_ROOT}/eval/v4train-main/eval"
fi

mkdir -p "${RUN_ROOT}"

branch_root() {
  echo "${RUN_ROOT}/$1"
}

branch_eval() {
  echo "$(branch_root "$1")/eval/camera_adapter/dataa_detection_camera_adapter_summary.json"
}

run_detection_only() {
  STAGE=all \
  MODEL_PATH="${MODEL_PATH}" \
  ADAPTER_PATH="${DETECTION_ONLY_ADAPTER}" \
  DATAA_DEV_SPLIT_JSON="${DATAA_SPLIT}" \
  RUN_ROOT="$(branch_root detection_only)" \
  PERSIST_ROOT="${PERSIST_ROOT}/detection_only" \
  V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR}" \
  KEEP_ALIVE_AFTER_RUN=0 \
  bash scripts/camera_detection_retention/run.sh
}

run_adapter_only() {
  local name="$1"
  local adapter="$2"
  local root
  root="$(branch_root "${name}")"
  for stage in preflight merge infer_camera; do
    STAGE="${stage}" \
    MODEL_PATH="${MODEL_PATH}" \
    ADAPTER_PATH="${adapter}" \
    DATAA_DEV_SPLIT_JSON="${DATAA_SPLIT}" \
    RUN_ROOT="${root}" \
    PERSIST_ROOT="${PERSIST_ROOT}/${name}" \
    V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR}" \
    KEEP_ALIVE_AFTER_RUN=0 \
    bash scripts/camera_detection_retention/run.sh
  done
  mkdir -p "${root}/eval/camera_adapter"
  "${PYTHON_BIN}" "${V4TRAIN_EVAL_DIR}/eval_dataa.py" \
    --gt_json "${root}/data/dataa_40step_v3_fixed_dev_detection.json" \
    --pred_json "${root}/inference/camera_adapter" \
    --out_dir "${root}/eval/camera_adapter" \
    --output_prefix dataa_detection_camera_adapter \
    --compute_iou
  mkdir -p "${PERSIST_ROOT}/${name}/eval"
  cp -a "${root}/eval/." "${PERSIST_ROOT}/${name}/eval/"
}

summarize() {
  local output="${RUN_ROOT}/dataa_four_model_detection_gate_summary.json"
  "${PYTHON_BIN}" -m scripts.camera_joint_sft_gate.summarize_dataa \
    --base-eval "$(branch_root detection_only)/eval/base/dataa_detection_base_summary.json" \
    --detection-only-eval "$(branch_eval detection_only)" \
    --correct-camera-eval "$(branch_eval correct_camera)" \
    --flipped-camera-eval "$(branch_eval flipped_camera)" \
    --output-json "${output}"
  mkdir -p "${PERSIST_ROOT}"
  cp -a "${output}" "${PERSIST_ROOT}/"
}

case "${STAGE}" in
  detection_only) run_detection_only ;;
  correct_camera) run_adapter_only correct_camera "${CORRECT_ADAPTER}" ;;
  flipped_camera) run_adapter_only flipped_camera "${FLIPPED_ADAPTER}" ;;
  summarize) summarize ;;
  all)
    run_detection_only
    run_adapter_only correct_camera "${CORRECT_ADAPTER}"
    run_adapter_only flipped_camera "${FLIPPED_ADAPTER}"
    summarize
    ;;
  *)
    echo "STAGE must be detection_only, correct_camera, flipped_camera, summarize, or all" >&2
    exit 2
    ;;
esac

if [[ "${KEEP_ALIVE_AFTER_RUN}" == "1" ]]; then
  exec bash "${KEEP_ALIVE_SCRIPT}"
fi
