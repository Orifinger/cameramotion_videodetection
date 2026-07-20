#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
WORK_ROOT="${WORK_ROOT:-/tmp/1res/camera_discriminative_gate/v1}"
META_ROOT="${META_ROOT:-${PROJECT_ROOT}/res/camera_discriminative_gate/v1}"
STAGE="${STAGE:-eval_vif}"
NUM_GPUS="${NUM_GPUS:-16}"

VIF_FEATURE_INDEX="${VIF_FEATURE_INDEX:-${PROJECT_ROOT}/res/camera_ctne_gate1/v1/features/vifbench_feature_index.jsonl}"
MODEL_BUNDLE="${META_ROOT}/model_bundle"
CALIBRATION_DIR="${META_ROOT}/calibration"
VIF_DATA_DIR="${WORK_ROOT}/vifbench_data"
VIF_OUTPUT_DIR="${META_ROOT}/eval/vifbench"

cd "${PROJECT_ROOT}"
mkdir -p "${WORK_ROOT}" "${META_ROOT}/preflight" "${VIF_OUTPUT_DIR}"

preflight() {
  python -m scripts.camera_discriminative_gate.preflight \
    --role evaluate \
    --feature-index-jsonl "${VIF_FEATURE_INDEX}" \
    --model-root "${MODEL_BUNDLE}" \
    --calibration-dir "${CALIBRATION_DIR}" \
    --expected-gpus "${NUM_GPUS}" \
    --output-json "${META_ROOT}/preflight/server_b.json"
}

eval_vif() {
  CUDA_VISIBLE_DEVICES=0 python -m scripts.camera_discriminative_gate.evaluate external \
    --feature-index-jsonl "${VIF_FEATURE_INDEX}" \
    --model-root "${MODEL_BUNDLE}" \
    --calibration-dir "${CALIBRATION_DIR}" \
    --packed-npz "${VIF_DATA_DIR}/vifbench_sequences.npz" \
    --rows-jsonl "${VIF_DATA_DIR}/vifbench_rows.jsonl" \
    --dataset-name "ViF-Bench" \
    --output-dir "${VIF_OUTPUT_DIR}" \
    --batch-size 256 \
    --bootstrap-iterations 2000 \
    --device cuda
}

case "${STAGE}" in
  preflight) preflight ;;
  eval_vif) preflight; eval_vif ;;
  *) echo "Unknown STAGE=${STAGE}" >&2; exit 2 ;;
esac

echo "Server B stage completed: ${STAGE}"
echo "Persistent summary: ${VIF_OUTPUT_DIR}/camera_discriminative_gate_summary.json"
echo "Persistent per-item metrics: ${VIF_OUTPUT_DIR}/camera_discriminative_gate_items.csv"
echo "Ephemeral prepared ViF arrays: ${VIF_DATA_DIR}"

if [[ "${KEEP_ALIVE_AFTER_RUN:-0}" == "1" ]]; then
  exec bash /input/training/keep.sh
fi
