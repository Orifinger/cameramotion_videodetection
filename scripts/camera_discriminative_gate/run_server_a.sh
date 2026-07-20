#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
WORK_ROOT="${WORK_ROOT:-/tmp/1res/camera_discriminative_gate/v1}"
META_ROOT="${META_ROOT:-${PROJECT_ROOT}/res/camera_discriminative_gate/v1}"
STAGE="${STAGE:-all}"
NUM_GPUS="${NUM_GPUS:-16}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

DATAB_FEATURE_INDEX="${DATAB_FEATURE_INDEX:-${PROJECT_ROOT}/res/camera_ctne_gate1/v1/features/datab_feature_index.jsonl}"
DATA_DIR="${WORK_ROOT}/data"
PACKED_NPZ="${DATA_DIR}/datab_sequences.npz"
ROWS_JSONL="${DATA_DIR}/datab_rows.jsonl"
TRAIN_ROOT="${WORK_ROOT}/train"
MODEL_BUNDLE="${META_ROOT}/model_bundle"
CALIBRATION_DIR="${META_ROOT}/calibration"

cd "${PROJECT_ROOT}"
mkdir -p "${WORK_ROOT}" "${META_ROOT}/preflight" "${META_ROOT}/training" "${CALIBRATION_DIR}"

preflight() {
  python -m scripts.camera_discriminative_gate.preflight \
    --role train \
    --feature-index-jsonl "${DATAB_FEATURE_INDEX}" \
    --expected-gpus "${NUM_GPUS}" \
    --output-json "${META_ROOT}/preflight/server_a.json"
}

prepare_data() {
  if [[ "${FORCE_PREPARE}" != "1" \
        && -f "${DATA_DIR}/preprocessor.npz" \
        && -f "${PACKED_NPZ}" \
        && -f "${ROWS_JSONL}" \
        && -f "${DATA_DIR}/prepare_summary.json" ]]; then
    echo "Reusing prepared variable-length DataB sequences: ${DATA_DIR}"
    return
  fi
  python -m scripts.camera_discriminative_gate.prepare \
    --feature-index-jsonl "${DATAB_FEATURE_INDEX}" \
    --output-dir "${DATA_DIR}" \
    --pca-dim 64 \
    --fit-transitions-per-video 16 \
    --seed 20260720 \
    --clip-value 10
}

smoke() {
  prepare_data
  CUDA_VISIBLE_DEVICES=0 python -m scripts.camera_discriminative_gate.train \
    --packed-npz "${PACKED_NPZ}" \
    --rows-jsonl "${ROWS_JSONL}" \
    --output-dir "${WORK_ROOT}/smoke" \
    --modes matched \
    --seeds 13 \
    --epochs 2 \
    --patience 2 \
    --batch-size 256 \
    --device cuda
}

train_all() {
  prepare_data
  local complete
  complete="$(find "${TRAIN_ROOT}/models" -path '*/model.pt' -type f 2>/dev/null | wc -l || true)"
  if [[ "${FORCE_TRAIN}" != "1" && "${complete}" -eq 9 ]]; then
    echo "Reusing nine completed supervised classifier models: ${TRAIN_ROOT}/models"
  else
    local pids=()
    local failed=0
    local logs="${WORK_ROOT}/train_logs"
    mkdir -p "${logs}"
    for job_index in $(seq 0 8); do
      CUDA_VISIBLE_DEVICES="${job_index}" python -m scripts.camera_discriminative_gate.train \
        --packed-npz "${PACKED_NPZ}" \
        --rows-jsonl "${ROWS_JSONL}" \
        --output-dir "${TRAIN_ROOT}" \
        --job-index "${job_index}" \
        --job-count 9 \
        --hidden-dim 128 \
        --dropout 0.10 \
        --batch-size 128 \
        --epochs 50 \
        --patience 8 \
        --learning-rate 3e-4 \
        --weight-decay 1e-4 \
        --device cuda \
        > "${logs}/job_${job_index}.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        failed=1
      fi
    done
    if [[ "${failed}" -ne 0 ]]; then
      echo "At least one supervised classifier training job failed. Inspect ${logs}." >&2
      return 2
    fi
  fi

  mkdir -p "${MODEL_BUNDLE}/models"
  cp -a "${DATA_DIR}/preprocessor.npz" "${MODEL_BUNDLE}/"
  cp -a "${DATA_DIR}/prepare_summary.json" "${MODEL_BUNDLE}/"
  cp -a "${TRAIN_ROOT}/models/." "${MODEL_BUNDLE}/models/"
  find "${TRAIN_ROOT}/models" -name training_summary.json -type f -exec cp --parents '{}' "${META_ROOT}/training" \;
}

calibrate() {
  python -m scripts.camera_discriminative_gate.evaluate calibrate \
    --packed-npz "${PACKED_NPZ}" \
    --rows-jsonl "${ROWS_JSONL}" \
    --model-root "${MODEL_BUNDLE}" \
    --output-dir "${CALIBRATION_DIR}" \
    --validation-split val \
    --batch-size 256 \
    --device cuda
}

case "${STAGE}" in
  preflight) preflight ;;
  prepare) prepare_data ;;
  smoke) preflight; smoke ;;
  train) preflight; train_all ;;
  calibrate) calibrate ;;
  all) preflight; prepare_data; train_all; calibrate ;;
  *) echo "Unknown STAGE=${STAGE}" >&2; exit 2 ;;
esac

echo "Server A stage completed: ${STAGE}"
echo "Persistent model bundle: ${MODEL_BUNDLE}"
echo "Persistent DataB calibration: ${CALIBRATION_DIR}/calibration.json"
echo "Ephemeral prepared DataB arrays: ${DATA_DIR}"

if [[ "${KEEP_ALIVE_AFTER_RUN:-0}" == "1" ]]; then
  exec bash /input/training/keep.sh
fi
