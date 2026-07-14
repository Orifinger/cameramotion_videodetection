#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${STAGE:-all}"
PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
JOINT_ROOT="${JOINT_ROOT:-/tmp/1res/camera_joint_sft_gate}"
RUN_ROOT="${RUN_ROOT:-${JOINT_ROOT}/vif_four_model_compare}"
PERSIST_ROOT="${PERSIST_ROOT:-${PROJECT_ROOT}/res/camera_joint_sft_gate/vif_four_model_compare}"
DETECTION_ONLY_ADAPTER="${DETECTION_ONLY_ADAPTER:-${JOINT_ROOT}/train/detection_only}"
CORRECT_ADAPTER="${CORRECT_ADAPTER:-${JOINT_ROOT}/train/correct_camera}"
SHUFFLED_ADAPTER="${SHUFFLED_ADAPTER:-${JOINT_ROOT}/train/shuffled_camera}"
VIF_RUNNER="${VIF_RUNNER:-scripts/camera_detection_retention/run_vifbench.sh}"
SHARED_BASE_PRED_DIR="${SHARED_BASE_PRED_DIR:-${RUN_ROOT}/shared/base_predictions}"
NUM_GPUS="${NUM_GPUS:-16}"
KEEP_ALIVE_AFTER_RUN="${KEEP_ALIVE_AFTER_RUN:-0}"
KEEP_ALIVE_SCRIPT="${KEEP_ALIVE_SCRIPT:-/input/training/keep.sh}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DETECTION_RUN="${RUN_ROOT}/detection_only"
CORRECT_RUN="${RUN_ROOT}/correct_camera"
SHUFFLED_RUN="${RUN_ROOT}/shuffled_camera"
FINAL_SUMMARY="${RUN_ROOT}/vif_four_model_detection_gate_summary.json"
LAUNCH_LOG="${RUN_ROOT}/pipeline.log"

mkdir -p "${RUN_ROOT}" "${PERSIST_ROOT}"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

run_branch() {
  local name="$1"
  local adapter="$2"
  local root="$3"
  local skip_base="$4"
  local parallel_models="$5"
  local nested_stage="${6:-all}"
  STAGE="${nested_stage}" \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  MODEL_PATH="${MODEL_PATH}" \
  ADAPTER_PATH="${adapter}" \
  RUN_ROOT="${root}" \
  PERSIST_ROOT="${PERSIST_ROOT}/branches/${name}" \
  BASE_PRED_DIR="${SHARED_BASE_PRED_DIR}" \
  BASE_MODEL_NAME="Qwen3-VL-8B-detection-base-vif-dev" \
  CAMERA_MODEL_NAME="Qwen3-VL-8B-${name}-vif-dev" \
  NUM_GPUS="${NUM_GPUS}" \
  SKIP_BASE_INFERENCE="${skip_base}" \
  PARALLEL_MODELS="${parallel_models}" \
  KEEP_ALIVE_AFTER_RUN=0 \
  bash "${VIF_RUNNER}"
}

preflight_all() {
  run_branch detection-only "${DETECTION_ONLY_ADAPTER}" "${DETECTION_RUN}" 0 1 preflight
  run_branch correct-camera "${CORRECT_ADAPTER}" "${CORRECT_RUN}" 1 0 preflight
  run_branch flipped-camera "${SHUFFLED_ADAPTER}" "${SHUFFLED_RUN}" 1 0 preflight
}

run_detection_pair() {
  echo "=== Pair 1/2: base checkpoint + detection-only control ==="
  run_branch detection-only "${DETECTION_ONLY_ADAPTER}" "${DETECTION_RUN}" 0 1 all
}

run_camera_pair() {
  if [[ ! -d "${SHARED_BASE_PRED_DIR}" ]]; then
    echo "Missing shared base predictions: ${SHARED_BASE_PRED_DIR}" >&2
    echo "Run STAGE=detection_pair first." >&2
    exit 2
  fi
  echo "=== Pair 2/2: correct-camera + flipped-camera controls ==="
  echo "Merging the two adapters sequentially to limit host-memory pressure."
  run_branch correct-camera "${CORRECT_ADAPTER}" "${CORRECT_RUN}" 1 0 merge
  run_branch flipped-camera "${SHUFFLED_ADAPTER}" "${SHUFFLED_RUN}" 1 0 merge
  run_branch correct-camera "${CORRECT_ADAPTER}" "${CORRECT_RUN}" 1 0 infer &
  local correct_pid=$!
  run_branch flipped-camera "${SHUFFLED_ADAPTER}" "${SHUFFLED_RUN}" 1 0 infer &
  local flipped_pid=$!
  set +e
  wait "${correct_pid}"
  local correct_status=$?
  wait "${flipped_pid}"
  local flipped_status=$?
  set -e
  if [[ "${correct_status}" != "0" || "${flipped_status}" != "0" ]]; then
    echo "Camera pair failed: correct_status=${correct_status}, flipped_status=${flipped_status}" >&2
    exit 1
  fi
  run_branch correct-camera "${CORRECT_ADAPTER}" "${CORRECT_RUN}" 1 0 eval
  run_branch flipped-camera "${SHUFFLED_ADAPTER}" "${SHUFFLED_RUN}" 1 0 eval
}

summarize() {
  "${PYTHON_BIN}" -m scripts.camera_joint_sft_gate.summarize_vif_four_model \
    --base-eval "${DETECTION_RUN}/eval/base_vifbench_eval.json" \
    --detection-only-eval "${DETECTION_RUN}/eval/camera_adapter_vifbench_eval.json" \
    --correct-camera-eval "${CORRECT_RUN}/eval/camera_adapter_vifbench_eval.json" \
    --flipped-camera-eval "${SHUFFLED_RUN}/eval/camera_adapter_vifbench_eval.json" \
    --output-json "${FINAL_SUMMARY}"
  cp -a "${FINAL_SUMMARY}" "${PERSIST_ROOT}/"
  cp -a "${LAUNCH_LOG}" "${PERSIST_ROOT}/" 2>/dev/null || true
}

echo "=== ViF-Bench four-model development comparison ==="
echo "stage=${STAGE}"
echo "base_model=${MODEL_PATH}"
echo "run_root=${RUN_ROOT}"
echo "shared_base_predictions=${SHARED_BASE_PRED_DIR}"
echo "camera_context_at_inference=false"

case "${STAGE}" in
  preflight) preflight_all ;;
  detection_pair) run_detection_pair ;;
  camera_pair) run_camera_pair ;;
  summarize) summarize ;;
  all)
    run_detection_pair
    run_camera_pair
    summarize
    ;;
  *)
    echo "STAGE must be preflight, detection_pair, camera_pair, summarize, or all" >&2
    exit 2
    ;;
esac

if [[ "${KEEP_ALIVE_AFTER_RUN}" == "1" ]]; then
  exec bash "${KEEP_ALIVE_SCRIPT}"
fi
