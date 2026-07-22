#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
WORK_ROOT="${WORK_ROOT:-/tmp/1res/forensic_temporal_expert_gate/v1/server_b}"
META_ROOT="${META_ROOT:-${PROJECT_ROOT}/res/forensic_temporal_expert_gate/v1}"
STAGE="${STAGE:-all}"
NUM_GPUS="${NUM_GPUS:-16}"
DINOV2_MODEL="${DINOV2_MODEL:-/home/admin/dinov2-small}"
V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR:-${PROJECT_ROOT}/eval/v4train-main/eval}"
INDEX_DIR="${INDEX_DIR:-${V4TRAIN_EVAL_DIR}/test_index_splits/splits_16}"
if [[ ! -d "${INDEX_DIR}" && -d "$(dirname "${V4TRAIN_EVAL_DIR}")/test_index_splits/splits_16" ]]; then
  INDEX_DIR="$(dirname "${V4TRAIN_EVAL_DIR}")/test_index_splits/splits_16"
fi
HISTORICAL_QWEN="${HISTORICAL_QWEN:-/input/workflow_58770161/workspace/test/test_selfcot/Skyra/res/v4vif_2766busterall_trainall/v4vif_2766busterall_trainall-3vl8b-vifbench/Qwen3-VL-v4vif_2766busterall_trainall-vifbench.json}"
QWEN_CONFIDENCE="${QWEN_CONFIDENCE:-${PROJECT_ROOT}/res/vifbench_qwen_confidence_fusion/v1/confidence_shards}"

MANIFEST="${META_ROOT}/data/vifbench_development_manifest.jsonl"
MANIFEST_SUMMARY="${META_ROOT}/data/vifbench_development_manifest_summary.json"
FEATURE_ROOT="${WORK_ROOT}/vifbench_features"
FEATURE_INDEX="${META_ROOT}/features/vifbench_feature_index.jsonl"
FEATURE_AUDIT="${META_ROOT}/features/vifbench_feature_audit.json"
MODEL_BUNDLE="${META_ROOT}/model_bundle"
EVAL_DIR="${META_ROOT}/eval"
GATE1_SUMMARY="${EVAL_DIR}/forensic_temporal_expert_gate1_summary.json"
EXPERT_ITEMS="${EVAL_DIR}/forensic_temporal_expert_vifbench_items.csv"
LOG_PATH="${WORK_ROOT}/pipeline.log"
WAIT_SECONDS="${WAIT_SECONDS:-43200}"

cd "${PROJECT_ROOT}"
mkdir -p "${WORK_ROOT}" "${META_ROOT}/preflight" "${META_ROOT}/data" "${META_ROOT}/features" "${EVAL_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

persist_small() {
  mkdir -p "${META_ROOT}/logs"
  cp -a "${LOG_PATH}" "${META_ROOT}/logs/server_b_pipeline.log" 2>/dev/null || true
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
    --required-dir "${INDEX_DIR}" \
    --required-file "${HISTORICAL_QWEN}" \
    --expected-gpus "${NUM_GPUS}" \
    --output-json "${META_ROOT}/preflight/server_b.json"
}

build() {
  python -m scripts.forensic_temporal_expert_gate.build_manifest vif \
    --index-dir "${INDEX_DIR}" \
    --output-jsonl "${MANIFEST}" \
    --summary-json "${MANIFEST_SUMMARY}" \
    --expected-ranks 16 \
    --expected-records 3160 \
    --check-files
}

smoke() {
  CUDA_VISIBLE_DEVICES=0 python -m scripts.forensic_temporal_expert_gate.smoke_model \
    --output-json "${META_ROOT}/preflight/server_b_model_smoke.json"
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

wait_for_models() {
  local deadline=$((SECONDS + WAIT_SECONDS))
  while [[ ! -f "${MODEL_BUNDLE}/complete.json" ]]; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for ${MODEL_BUNDLE}/complete.json" >&2
      return 2
    fi
    echo "Waiting for server A model bundle: ${MODEL_BUNDLE}"
    sleep 30
  done
}

evaluate_gate1() {
  set +e
  CUDA_VISIBLE_DEVICES=0 python -m scripts.forensic_temporal_expert_gate.evaluate \
    --feature-index-jsonl "${FEATURE_INDEX}" \
    --model-root "${MODEL_BUNDLE}" \
    --output-dir "${EVAL_DIR}" \
    --expected-records 3160 \
    --min-coverage 0.99
  local status=$?
  set -e
  echo "Gate 1 diagnostic exit status: ${status}; Gate 2 will still be computed for diagnosis."
}

evaluate_gate2() {
  set +e
  python -m scripts.forensic_temporal_expert_gate.complementarity \
    --gate1-summary "${GATE1_SUMMARY}" \
    --expert-items-csv "${EXPERT_ITEMS}" \
    --historical-qwen-predictions "${HISTORICAL_QWEN}" \
    --qwen-confidence "${QWEN_CONFIDENCE}" \
    --output-dir "${EVAL_DIR}"
  local status=$?
  set -e
  echo "Gate 2 diagnostic exit status: ${status}"
}

echo "=== 服务器 B：ViF-Bench 时序因果与 Qwen 互补性开发门 ==="
echo "stage=${STAGE}"
echo "work_root=${WORK_ROOT}"
echo "persistent_root=${META_ROOT}"
echo "GenBuster Closed Benchmark is not read by this script."

case "${STAGE}" in
  preflight) preflight ;;
  build) build ;;
  smoke) smoke ;;
  extract) extract_all ;;
  audit) audit ;;
  evaluate)
    wait_for_models
    evaluate_gate1
    evaluate_gate2
    ;;
  all)
    preflight
    build
    smoke
    extract_all
    audit
    wait_for_models
    evaluate_gate1
    evaluate_gate2
    ;;
  *) echo "STAGE must be preflight, build, smoke, extract, audit, evaluate, or all" >&2; exit 2 ;;
esac

persist_small
echo "Server B completed: ${STAGE}"
echo "Persistent evaluation: ${EVAL_DIR}"
echo "Ephemeral ViF features: ${FEATURE_ROOT}"

if [[ "${KEEP_ALIVE_AFTER_RUN:-0}" == "1" ]]; then
  trap - EXIT
  exec bash /input/training/keep.sh
fi
