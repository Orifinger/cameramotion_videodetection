#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${STAGE:-all}"
STEPS_TEXT="${STEPS:-698 1396}"
read -r -a STEPS_ARRAY <<< "${STEPS_TEXT}"
MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
JOINT_ROOT="${JOINT_ROOT:-/tmp/1res/camera_joint_sft_gate}"
RUN_ROOT="${RUN_ROOT:-${JOINT_ROOT}/checkpoint_window}"
PERSIST_ROOT="${PERSIST_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_joint_sft_gate/checkpoint_window}"
CAMERA_DEV="${CAMERA_DEV:-${JOINT_ROOT}/data/camera_dev_matched_frames.jsonl}"
DATAA_DEV="${DATAA_DEV:-${JOINT_ROOT}/data/dataa_test_detection.json}"
V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR:-/input/workflow_58770161/workspace/test/test_selfcot/Skyra/eval}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
NUM_GPUS="${NUM_GPUS:-16}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
KEEP_ALIVE_AFTER_RUN="${KEEP_ALIVE_AFTER_RUN:-0}"
KEEP_ALIVE_SCRIPT="${KEEP_ALIVE_SCRIPT:-/input/training/keep.sh}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${V4TRAIN_EVAL_DIR}/infer_dataa.py" ]]; then
  V4TRAIN_EVAL_DIR="${REPO_ROOT}/eval/v4train-main/eval"
fi

mkdir -p "${RUN_ROOT}" "${PERSIST_ROOT}"

adapter_path() {
  local branch="$1"
  local step="$2"
  echo "${JOINT_ROOT}/train/${branch}/checkpoint-${step}"
}

score_camera() {
  local branch="$1"
  local step="$2"
  local adapter
  local output_dir
  adapter="$(adapter_path "${branch}" "${step}")"
  output_dir="${RUN_ROOT}/camera_predictions/step_${step}_${branch}"
  rm -rf "${output_dir}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.camera_joint_sft_gate.score_binary \
    --model-path "${MODEL_PATH}" \
    --adapter-path "${adapter}" \
    --condition "matched_frames=${CAMERA_DEV}" \
    --output-dir "${output_dir}" \
    --model-stage "step_${step}_${branch}" \
    --max-pixels "${MAX_PIXELS}" \
    --seed 20260713
  mkdir -p "${RUN_ROOT}/camera_eval"
  "${PYTHON_BIN}" -m scripts.camera_binary_vqa.evaluate \
    --gold "matched_frames=${CAMERA_DEV}" \
    --predictions-dir "${output_dir}" \
    --model-stage "step_${step}_${branch}" \
    --output-json "${RUN_ROOT}/camera_eval/step_${step}_${branch}.json"
}

run_dataa() {
  local branch="$1"
  local step="$2"
  local adapter
  local root
  adapter="$(adapter_path "${branch}" "${step}")"
  root="${RUN_ROOT}/dataa/step_${step}/${branch}"
  mkdir -p "${root}/data"
  cp -a "${DATAA_DEV}" "${root}/data/dataa_40step_v3_fixed_dev_detection.json"
  for stage in merge infer_camera; do
    STAGE="${stage}" \
    MODEL_PATH="${MODEL_PATH}" \
    ADAPTER_PATH="${adapter}" \
    RUN_ROOT="${root}" \
    V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR}" \
    NUM_GPUS="${NUM_GPUS}" \
    WORKERS_PER_GPU="${WORKERS_PER_GPU}" \
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
}

run_camera_all() {
  local step
  for step in "${STEPS_ARRAY[@]}"; do
    score_camera detection_only "${step}"
    score_camera correct_camera "${step}"
  done
}

run_dataa_all() {
  local step
  for step in "${STEPS_ARRAY[@]}"; do
    run_dataa detection_only "${step}"
    run_dataa correct_camera "${step}"
  done
}

summarize() {
  local output="${RUN_ROOT}/checkpoint_window_summary.json"
  "${PYTHON_BIN}" -m scripts.camera_joint_sft_gate.summarize_checkpoint_window \
    --root "${RUN_ROOT}" \
    --steps "${STEPS_ARRAY[@]}" \
    --output-json "${output}"
  mkdir -p "${PERSIST_ROOT}/camera_eval" "${PERSIST_ROOT}/dataa_eval"
  cp -a "${RUN_ROOT}/camera_eval/." "${PERSIST_ROOT}/camera_eval/"
  local step
  local branch
  for step in "${STEPS_ARRAY[@]}"; do
    for branch in detection_only correct_camera; do
      cp -a \
        "${RUN_ROOT}/dataa/step_${step}/${branch}/eval/camera_adapter/dataa_detection_camera_adapter_summary.json" \
        "${PERSIST_ROOT}/dataa_eval/step_${step}_${branch}.json"
    done
  done
  cp -a "${output}" "${PERSIST_ROOT}/"
}

case "${STAGE}" in
  camera) run_camera_all ;;
  dataa) run_dataa_all ;;
  summarize) summarize ;;
  all)
    run_camera_all
    run_dataa_all
    summarize
    ;;
  *)
    echo "STAGE must be camera, dataa, summarize, or all" >&2
    exit 2
    ;;
esac

if [[ "${KEEP_ALIVE_AFTER_RUN}" == "1" ]]; then
  exec bash "${KEEP_ALIVE_SCRIPT}"
fi
