#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
WORK_ROOT="${WORK_ROOT:-/tmp/1res/forensic_temporal_expert_gate/v1/server_a}"
META_ROOT="${META_ROOT:-${PROJECT_ROOT}/res/forensic_temporal_expert_gate/v1}"
STAGE="${STAGE:-all}"
NUM_GPUS="${NUM_GPUS:-16}"
DINOV2_MODEL="${DINOV2_MODEL:-/home/admin/dinov2-small}"
DATAB_JSON="${DATAB_JSON:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json}"

MANIFEST="${META_ROOT}/data/datab_full_6766_grouped_folds.jsonl"
MANIFEST_SUMMARY="${META_ROOT}/data/datab_full_6766_grouped_folds_summary.json"
FEATURE_ROOT="${WORK_ROOT}/datab_features"
FEATURE_INDEX="${META_ROOT}/features/datab_feature_index.jsonl"
FEATURE_AUDIT="${META_ROOT}/features/datab_feature_audit.json"
TRAIN_ROOT="${WORK_ROOT}/training"
MODEL_BUNDLE="${META_ROOT}/model_bundle"
LOG_PATH="${WORK_ROOT}/pipeline.log"

cd "${PROJECT_ROOT}"
mkdir -p "${WORK_ROOT}" "${META_ROOT}/preflight" "${META_ROOT}/data" "${META_ROOT}/features"
exec > >(tee -a "${LOG_PATH}") 2>&1

persist_small() {
  mkdir -p "${META_ROOT}/logs"
  cp -a "${LOG_PATH}" "${META_ROOT}/logs/server_a_pipeline.log" 2>/dev/null || true
}
finish() {
  local status=$?
  trap - EXIT
  set +e
  persist_small
  echo "Pipeline exit status: ${status}"
  echo "Persistent metadata: ${META_ROOT}"
  exit "${status}"
}
trap finish EXIT

preflight() {
  python -m scripts.forensic_temporal_expert_gate.preflight \
    --dinov2-model "${DINOV2_MODEL}" \
    --required-file "${DATAB_JSON}" \
    --expected-gpus "${NUM_GPUS}" \
    --output-json "${META_ROOT}/preflight/server_a.json"
}

build() {
  python -m scripts.forensic_temporal_expert_gate.build_manifest datab \
    --detection-json "${DATAB_JSON}" \
    --output-jsonl "${MANIFEST}" \
    --summary-json "${MANIFEST_SUMMARY}" \
    --expected-records 6766 \
    --folds 5 \
    --seed 20260722 \
    --check-files
}

smoke() {
  CUDA_VISIBLE_DEVICES=0 python -m scripts.forensic_temporal_expert_gate.smoke_model \
    --output-json "${META_ROOT}/preflight/server_a_model_smoke.json"
  CUDA_VISIBLE_DEVICES=0 python -m scripts.forensic_temporal_expert_gate.extract_features \
    --manifest-jsonl "${MANIFEST}" \
    --output-dir "${WORK_ROOT}/smoke" \
    --dinov2-model "${DINOV2_MODEL}" \
    --max-cases 8 \
    --batch-size 8 \
    --overwrite
}

extract_all() {
  OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    -m scripts.forensic_temporal_expert_gate.extract_features \
    --manifest-jsonl "${MANIFEST}" \
    --output-dir "${FEATURE_ROOT}" \
    --dinov2-model "${DINOV2_MODEL}" \
    --batch-size 16
}

audit() {
  python -m scripts.forensic_temporal_expert_gate.audit_features \
    --manifest-jsonl "${MANIFEST}" \
    --feature-root "${FEATURE_ROOT}" \
    --output-index-jsonl "${FEATURE_INDEX}" \
    --output-summary-json "${FEATURE_AUDIT}" \
    --min-coverage 0.99
}

train_all() {
  local job_count=9
  local pids=()
  local failed=0
  mkdir -p "${WORK_ROOT}/train_logs"
  for job in $(seq 0 $((job_count - 1))); do
    CUDA_VISIBLE_DEVICES="${job}" OMP_NUM_THREADS=4 \
      python -m scripts.forensic_temporal_expert_gate.train \
      --feature-index-jsonl "${FEATURE_INDEX}" \
      --output-dir "${TRAIN_ROOT}" \
      --job-index "${job}" \
      --job-count "${job_count}" \
      --patience 0 \
      > "${WORK_ROOT}/train_logs/job_${job}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "At least one training job failed; inspect ${WORK_ROOT}/train_logs" >&2
    return 2
  fi
  rm -rf "${MODEL_BUNDLE}.building"
  mkdir -p "${MODEL_BUNDLE}.building"
  cp -a "${TRAIN_ROOT}/models" "${MODEL_BUNDLE}.building/"
  cp -a "${TRAIN_ROOT}/jobs" "${MODEL_BUNDLE}.building/"
  python - "${MODEL_BUNDLE}.building/complete.json" "${FEATURE_AUDIT}" <<'PY'
import json, sys
from pathlib import Path
payload = {
    "status": "completed",
    "models": 9,
    "modes": ["static", "ordered", "shuffled"],
    "seeds": [13, 37, 73],
    "feature_audit": sys.argv[2],
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  rm -rf "${MODEL_BUNDLE}"
  mv "${MODEL_BUNDLE}.building" "${MODEL_BUNDLE}"
}

echo "=== 服务器 A：DataB 原生尺度 DINO 时序专家训练 ==="
echo "stage=${STAGE}"
echo "work_root=${WORK_ROOT}"
echo "persistent_root=${META_ROOT}"
echo "All 6766 DataB rows are retained; original GenBuster train/test is metadata only."

case "${STAGE}" in
  preflight) preflight ;;
  build) build ;;
  smoke) smoke ;;
  extract) extract_all ;;
  audit) audit ;;
  train) train_all ;;
  all)
    preflight
    build
    smoke
    extract_all
    audit
    train_all
    ;;
  *) echo "STAGE must be preflight, build, smoke, extract, audit, train, or all" >&2; exit 2 ;;
esac

persist_small
echo "Server A completed: ${STAGE}"
echo "Persistent model bundle: ${MODEL_BUNDLE}"
echo "Ephemeral DataB features: ${FEATURE_ROOT}"

if [[ "${KEEP_ALIVE_AFTER_RUN:-0}" == "1" ]]; then
  trap - EXIT
  exec bash /input/training/keep.sh
fi
