#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
WORK_ROOT="${WORK_ROOT:-/tmp/1res/camera_ctne_gate1/v1}"
META_ROOT="${META_ROOT:-${PROJECT_ROOT}/res/camera_ctne_gate1/v1}"
STAGE="${STAGE:-all_vif}"
NUM_GPUS="${NUM_GPUS:-16}"
MAX_FRAMES="${MAX_FRAMES:-0}"
RAFT_CHECKPOINT="${RAFT_CHECKPOINT:-/home/admin/raft_large_C_T_SKHT_V2-ff5fadd5.pth}"
DINOV2_MODEL="${DINOV2_MODEL:-/home/admin/dinov2-small}"

DATAB_MANIFEST="${META_ROOT}/data/datab_manifest.jsonl"
MODEL_BUNDLE="${META_ROOT}/model_bundle"
CALIBRATION_DIR="${META_ROOT}/calibration"

VIF_INDEX_DIR="${VIF_INDEX_DIR:-${PROJECT_ROOT}/eval/v4train-main/test_index_splits/splits_16}"
VIF_CAMERA_JSONL="${VIF_CAMERA_JSONL:-/input/workflow_58770161/workspace/test/camb/camerabench_outputs/vifbench_cameramotion_labels_v2/datab_cameramotion_labels_v2.jsonl}"
VIF_MANIFEST="${META_ROOT}/data/vifbench_manifest.jsonl"
VIF_MANIFEST_SUMMARY="${META_ROOT}/data/vifbench_manifest_summary.json"
VIF_FEATURE_ROOT="${WORK_ROOT}/vifbench_features"
VIF_FEATURE_INDEX="${META_ROOT}/features/vifbench_feature_index.jsonl"
VIF_FEATURE_AUDIT="${META_ROOT}/features/vifbench_feature_audit.json"

GENBUSTER_FRAME_ROOT="${GENBUSTER_FRAME_ROOT:-}"
GENBUSTER_MANIFEST="${META_ROOT}/data/genbuster_benchmark_manifest.jsonl"
GENBUSTER_MANIFEST_SUMMARY="${META_ROOT}/data/genbuster_benchmark_manifest_summary.json"
GENBUSTER_FEATURE_ROOT="${WORK_ROOT}/genbuster_benchmark_features"
GENBUSTER_FEATURE_INDEX="${META_ROOT}/features/genbuster_benchmark_feature_index.jsonl"
GENBUSTER_FEATURE_AUDIT="${META_ROOT}/features/genbuster_benchmark_feature_audit.json"

cd "${PROJECT_ROOT}"
mkdir -p "${WORK_ROOT}" "${META_ROOT}/preflight" "${META_ROOT}/data" "${META_ROOT}/features" "${META_ROOT}/overlap"

preflight_extract() {
  python -m scripts.camera_ctne_gate1.preflight \
    --role extract \
    --expected-gpus "${NUM_GPUS}" \
    --raft-checkpoint "${RAFT_CHECKPOINT}" \
    --dinov2-model "${DINOV2_MODEL}" \
    --output-json "${META_ROOT}/preflight/server_b_extract.json"
}

build_vif() {
  python -m scripts.camera_ctne_gate1.build_manifest vif \
    --index-dir "${VIF_INDEX_DIR}" \
    --expected-ranks 16 \
    --camera-jsonl "${VIF_CAMERA_JSONL}" \
    --check-files \
    --output-jsonl "${VIF_MANIFEST}" \
    --summary-json "${VIF_MANIFEST_SUMMARY}"
  python -m scripts.camera_ctne_gate1.audit_overlap \
    --train-manifest-jsonl "${DATAB_MANIFEST}" \
    --external-manifest-jsonl "${VIF_MANIFEST}" \
    --output-json "${META_ROOT}/overlap/datab_vs_vifbench.json"
}

extract_vif() {
  OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    -m scripts.camera_ctne_gate1.extract_features \
    --manifest-jsonl "${VIF_MANIFEST}" \
    --output-dir "${VIF_FEATURE_ROOT}" \
    --raft-checkpoint "${RAFT_CHECKPOINT}" \
    --dinov2-model "${DINOV2_MODEL}" \
    --split test \
    --max-frames "${MAX_FRAMES}" \
    --chunk-frames 32
}

audit_vif() {
  python -m scripts.camera_ctne_gate1.audit_features \
    --manifest-jsonl "${VIF_MANIFEST}" \
    --feature-root "${VIF_FEATURE_ROOT}" \
    --output-index-jsonl "${VIF_FEATURE_INDEX}" \
    --output-summary-json "${VIF_FEATURE_AUDIT}" \
    --min-coverage 0.98
}

build_genbuster() {
  if [[ -z "${GENBUSTER_FRAME_ROOT}" ]]; then
    echo "Set GENBUSTER_FRAME_ROOT to the extracted GenBuster-200K benchmark frame root." >&2
    return 2
  fi
  python -m scripts.camera_ctne_gate1.build_manifest tree \
    --frame-root "${GENBUSTER_FRAME_ROOT}" \
    --dataset-name "GenBuster benchmark" \
    --split test \
    --check-files \
    --output-jsonl "${GENBUSTER_MANIFEST}" \
    --summary-json "${GENBUSTER_MANIFEST_SUMMARY}"
  python -m scripts.camera_ctne_gate1.audit_overlap \
    --train-manifest-jsonl "${DATAB_MANIFEST}" \
    --external-manifest-jsonl "${GENBUSTER_MANIFEST}" \
    --output-json "${META_ROOT}/overlap/datab_vs_genbuster_benchmark.json"
}

extract_genbuster() {
  OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    -m scripts.camera_ctne_gate1.extract_features \
    --manifest-jsonl "${GENBUSTER_MANIFEST}" \
    --output-dir "${GENBUSTER_FEATURE_ROOT}" \
    --raft-checkpoint "${RAFT_CHECKPOINT}" \
    --dinov2-model "${DINOV2_MODEL}" \
    --split test \
    --max-frames "${MAX_FRAMES}" \
    --chunk-frames 32
}

audit_genbuster() {
  python -m scripts.camera_ctne_gate1.audit_features \
    --manifest-jsonl "${GENBUSTER_MANIFEST}" \
    --feature-root "${GENBUSTER_FEATURE_ROOT}" \
    --output-index-jsonl "${GENBUSTER_FEATURE_INDEX}" \
    --output-summary-json "${GENBUSTER_FEATURE_AUDIT}" \
    --min-coverage 0.98
}

eval_vif() {
  python -m scripts.camera_ctne_gate1.preflight \
    --role evaluate \
    --expected-gpus "${NUM_GPUS}" \
    --required-file "${MODEL_BUNDLE}/preprocessor.npz" \
    --required-file "${CALIBRATION_DIR}/calibration.json" \
    --required-file "${VIF_FEATURE_INDEX}" \
    --output-json "${META_ROOT}/preflight/server_b_eval_vif.json"
  python -m scripts.camera_ctne_gate1.evaluate_external \
    --model-root "${MODEL_BUNDLE}" \
    --calibration-dir "${CALIBRATION_DIR}" \
    --test-index-jsonl "${VIF_FEATURE_INDEX}" \
    --dataset-name "ViF-Bench" \
    --output-dir "${META_ROOT}/eval/vifbench"
}

eval_genbuster() {
  python -m scripts.camera_ctne_gate1.evaluate_external \
    --model-root "${MODEL_BUNDLE}" \
    --calibration-dir "${CALIBRATION_DIR}" \
    --test-index-jsonl "${GENBUSTER_FEATURE_INDEX}" \
    --dataset-name "GenBuster benchmark" \
    --output-dir "${META_ROOT}/eval/genbuster_benchmark"
}

combine_external() {
  python -m scripts.camera_ctne_gate1.combine_external \
    --vif-summary "${META_ROOT}/eval/vifbench/ctne_gate1_summary.json" \
    --genbuster-summary "${META_ROOT}/eval/genbuster_benchmark/ctne_gate1_summary.json" \
    --output-json "${META_ROOT}/eval/ctne_gate1_final_decision.json"
}

case "${STAGE}" in
  preflight) preflight_extract ;;
  build_vif) build_vif ;;
  extract_vif) extract_vif ;;
  audit_vif) audit_vif ;;
  eval_vif) eval_vif ;;
  all_vif) preflight_extract; build_vif; extract_vif; audit_vif ;;
  build_genbuster) build_genbuster ;;
  extract_genbuster) extract_genbuster ;;
  audit_genbuster) audit_genbuster ;;
  eval_genbuster) eval_genbuster ;;
  all_genbuster) preflight_extract; build_genbuster; extract_genbuster; audit_genbuster ;;
  combine) combine_external ;;
  *) echo "Unknown STAGE=${STAGE}" >&2; exit 2 ;;
esac

echo "Server B stage completed: ${STAGE}"
echo "Ephemeral feature roots: ${VIF_FEATURE_ROOT} ${GENBUSTER_FEATURE_ROOT}"
echo "Persistent metadata root: ${META_ROOT}"

if [[ "${KEEP_ALIVE_AFTER_RUN:-0}" == "1" ]]; then
  exec bash /input/training/keep.sh
fi
