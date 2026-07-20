#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
WORK_ROOT="${WORK_ROOT:-/tmp/1res/camera_ctne_gate1/v1}"
META_ROOT="${META_ROOT:-${PROJECT_ROOT}/res/camera_ctne_gate1/v1}"
STAGE="${STAGE:-all}"
NUM_GPUS="${NUM_GPUS:-16}"
MAX_FRAMES="${MAX_FRAMES:-0}"

DATAB_DETECTION_JSON="${DATAB_DETECTION_JSON:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json}"
DATAB_CAMERA_JSONL="${DATAB_CAMERA_JSONL:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl}"
RAFT_CHECKPOINT="${RAFT_CHECKPOINT:-/home/admin/raft_large_C_T_SKHT_V2-ff5fadd5.pth}"
DINOV2_MODEL="${DINOV2_MODEL:-/home/admin/dinov2-small}"

DATAB_MANIFEST="${META_ROOT}/data/datab_manifest.jsonl"
DATAB_MANIFEST_SUMMARY="${META_ROOT}/data/datab_manifest_summary.json"
DATAB_FEATURE_ROOT="${WORK_ROOT}/datab_features"
DATAB_FEATURE_INDEX="${META_ROOT}/features/datab_feature_index.jsonl"
DATAB_FEATURE_AUDIT="${META_ROOT}/features/datab_feature_audit.json"
FLOW_WORK_ROOT="${WORK_ROOT}/flow_model"
MODEL_BUNDLE="${META_ROOT}/model_bundle"
CALIBRATION_DIR="${META_ROOT}/calibration"

cd "${PROJECT_ROOT}"
mkdir -p "${WORK_ROOT}" "${META_ROOT}/preflight" "${META_ROOT}/data" "${META_ROOT}/features"

run_preflight_extract() {
  python -m scripts.camera_ctne_gate1.preflight \
    --role extract \
    --expected-gpus "${NUM_GPUS}" \
    --raft-checkpoint "${RAFT_CHECKPOINT}" \
    --dinov2-model "${DINOV2_MODEL}" \
    --required-file "${DATAB_DETECTION_JSON}" \
    --required-file "${DATAB_CAMERA_JSONL}" \
    --output-json "${META_ROOT}/preflight/server_a_extract.json"
}

build_datab() {
  python -m scripts.camera_ctne_gate1.build_manifest datab \
    --detection-json "${DATAB_DETECTION_JSON}" \
    --camera-jsonl "${DATAB_CAMERA_JSONL}" \
    --val-ratio 0.20 \
    --seed 20260720 \
    --check-files \
    --output-jsonl "${DATAB_MANIFEST}" \
    --summary-json "${DATAB_MANIFEST_SUMMARY}"
}

extract_datab() {
  OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    -m scripts.camera_ctne_gate1.extract_features \
    --manifest-jsonl "${DATAB_MANIFEST}" \
    --output-dir "${DATAB_FEATURE_ROOT}" \
    --raft-checkpoint "${RAFT_CHECKPOINT}" \
    --dinov2-model "${DINOV2_MODEL}" \
    --split all \
    --max-frames "${MAX_FRAMES}" \
    --chunk-frames 32 \
    --raft-batch-size 4 \
    --dino-batch-size 16
}

smoke_datab() {
  CUDA_VISIBLE_DEVICES=0 python -m scripts.camera_ctne_gate1.extract_features \
    --manifest-jsonl "${DATAB_MANIFEST}" \
    --output-dir "${WORK_ROOT}/smoke_datab" \
    --raft-checkpoint "${RAFT_CHECKPOINT}" \
    --dinov2-model "${DINOV2_MODEL}" \
    --split all \
    --max-frames "${MAX_FRAMES}" \
    --chunk-frames 32 \
    --max-cases 6 \
    --raft-batch-size 2 \
    --dino-batch-size 8 \
    --overwrite
}

audit_datab() {
  python -m scripts.camera_ctne_gate1.audit_features \
    --manifest-jsonl "${DATAB_MANIFEST}" \
    --feature-root "${DATAB_FEATURE_ROOT}" \
    --output-index-jsonl "${DATAB_FEATURE_INDEX}" \
    --output-summary-json "${DATAB_FEATURE_AUDIT}" \
    --min-coverage 0.98
}

train_flows() {
  python -m scripts.camera_ctne_gate1.preflight \
    --role train \
    --expected-gpus "${NUM_GPUS}" \
    --required-file "${DATAB_FEATURE_INDEX}" \
    --output-json "${META_ROOT}/preflight/server_a_train.json"

  python -m scripts.camera_ctne_gate1.train \
    --feature-index-jsonl "${DATAB_FEATURE_INDEX}" \
    --output-dir "${FLOW_WORK_ROOT}" \
    --prepare-only

  local job_count=6
  local pids=()
  local logs="${WORK_ROOT}/train_logs"
  mkdir -p "${logs}"
  for job_index in $(seq 0 $((job_count - 1))); do
    CUDA_VISIBLE_DEVICES="${job_index}" python -m scripts.camera_ctne_gate1.train \
      --feature-index-jsonl "${DATAB_FEATURE_INDEX}" \
      --output-dir "${FLOW_WORK_ROOT}" \
      --reuse-preprocessor \
      --job-index "${job_index}" \
      --job-count "${job_count}" \
      > "${logs}/job_${job_index}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "At least one CTNE flow training job failed. Inspect ${logs}." >&2
    return 2
  fi

  mkdir -p "${MODEL_BUNDLE}/models"
  cp -a "${FLOW_WORK_ROOT}/preprocessor.npz" "${MODEL_BUNDLE}/"
  cp -a "${FLOW_WORK_ROOT}/preprocessor_summary.json" "${MODEL_BUNDLE}/"
  cp -a "${FLOW_WORK_ROOT}/models/." "${MODEL_BUNDLE}/models/"
}

calibrate_datab() {
  python -m scripts.camera_ctne_gate1.calibrate \
    --model-root "${MODEL_BUNDLE}" \
    --feature-index-jsonl "${DATAB_FEATURE_INDEX}" \
    --output-dir "${CALIBRATION_DIR}"
}

case "${STAGE}" in
  preflight) run_preflight_extract ;;
  build) build_datab ;;
  smoke) smoke_datab ;;
  extract) extract_datab ;;
  audit) audit_datab ;;
  train) train_flows ;;
  calibrate) calibrate_datab ;;
  all)
    run_preflight_extract
    build_datab
    extract_datab
    audit_datab
    train_flows
    calibrate_datab
    ;;
  *) echo "Unknown STAGE=${STAGE}" >&2; exit 2 ;;
esac

echo "Server A stage completed: ${STAGE}"
echo "Persistent model bundle: ${MODEL_BUNDLE}"
echo "Persistent calibration: ${CALIBRATION_DIR}"
echo "Ephemeral DataB features: ${DATAB_FEATURE_ROOT}"

if [[ "${KEEP_ALIVE_AFTER_RUN:-0}" == "1" ]]; then
  exec bash /input/training/keep.sh
fi
