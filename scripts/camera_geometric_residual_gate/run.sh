#!/usr/bin/env bash
set -euo pipefail

STAGE=${STAGE:-preflight}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
RUN_NAME=${RUN_NAME:-v1}
OUT=${OUT:-/tmp/1res/camera_geometric_residual_gate/${RUN_NAME}}
META_DIR=${META_DIR:-${ROOT}/res/camera_geometric_residual_gate/${RUN_NAME}}

DATAB_DETECTION_JSON=${DATAB_DETECTION_JSON:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json}
DATAB_CAMERA_JSONL=${DATAB_CAMERA_JSONL:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl}
VIF_INDEX_DIR=${VIF_INDEX_DIR:-${ROOT}/eval/v4train-main/test_index_splits/splits_16}
VIF_CAMERA_JSONL=${VIF_CAMERA_JSONL:-/input/workflow_58770161/workspace/test/camb/camerabench_outputs/vifbench_cameramotion_labels_v2/datab_cameramotion_labels_v2.jsonl}
RAFT_CHECKPOINT=${RAFT_CHECKPOINT:-/home/admin/raft_large_C_T_SKHT_V2-ff5fadd5.pth}
DINO_MODEL=${DINO_MODEL:-/home/admin/dinov2-small}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
KEEP_ALIVE_AFTER_RUN=${KEEP_ALIVE_AFTER_RUN:-0}

DATA_DIR=${OUT}/data
FEATURE_ROOT=${OUT}/features
EVAL_DIR=${OUT}/eval
DATAB_MANIFEST=${DATA_DIR}/datab_geometric_residual_manifest.jsonl
VIF_CANONICAL_CAMERA=${DATA_DIR}/vifbench_predicted_camera_context.jsonl
VIF_MANIFEST=${DATA_DIR}/vifbench_geometric_residual_manifest.jsonl

mkdir -p "${DATA_DIR}" "${FEATURE_ROOT}" "${EVAL_DIR}" "${META_DIR}"
export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
cd "${ROOT}"

persist_small_results() {
  mkdir -p "${META_DIR}/data" "${META_DIR}/eval" "${META_DIR}/audits"
  find "${DATA_DIR}" -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' \) -exec cp -f {} "${META_DIR}/data/" \; 2>/dev/null || true
  find "${EVAL_DIR}" -maxdepth 1 -type f \( -name '*.json' -o -name '*.csv' \) -exec cp -f {} "${META_DIR}/eval/" \; 2>/dev/null || true
  find "${OUT}/audits" -maxdepth 1 -type f -name '*.json' -exec cp -f {} "${META_DIR}/audits/" \; 2>/dev/null || true
}

finish() {
  status=$?
  trap - EXIT
  persist_small_results
  echo "Pipeline exit status: ${status}"
  echo "Persistent small results: ${META_DIR}"
  echo "Compact validation features remain disposable under: ${FEATURE_ROOT}"
  if [[ "${KEEP_ALIVE_AFTER_RUN}" == "1" ]]; then
    echo "Starting /input/training/keep.sh after the experiment finished."
    exec bash /input/training/keep.sh
  fi
  exit "${status}"
}
trap finish EXIT

require_file() {
  [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 2; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 2; }
}

build_manifests() {
  python -m tools.prepare_vifbench_camera_context prepare \
    --index-dir "${VIF_INDEX_DIR}" \
    --camera-json "${VIF_CAMERA_JSONL}" \
    --output-jsonl "${VIF_CANONICAL_CAMERA}" \
    --summary-json "${DATA_DIR}/vifbench_camera_context_summary.json" \
    --expected-ranks 16 \
    --min-coverage 1.0

  python -m scripts.camera_geometric_residual_gate.build_manifest datab \
    --detection-json "${DATAB_DETECTION_JSON}" \
    --camera-jsonl "${DATAB_CAMERA_JSONL}" \
    --output-jsonl "${DATAB_MANIFEST}" \
    --summary-json "${DATA_DIR}/datab_geometric_residual_manifest_summary.json" \
    --check-files

  python -m scripts.camera_geometric_residual_gate.build_manifest vif \
    --index-dir "${VIF_INDEX_DIR}" \
    --canonical-camera-jsonl "${VIF_CANONICAL_CAMERA}" \
    --output-jsonl "${VIF_MANIFEST}" \
    --summary-json "${DATA_DIR}/vifbench_geometric_residual_manifest_summary.json" \
    --expected-ranks 16 \
    --min-coverage 1.0 \
    --check-files
}

preflight() {
  require_file "${DATAB_DETECTION_JSON}"
  require_file "${DATAB_CAMERA_JSONL}"
  require_file "${VIF_CAMERA_JSONL}"
  require_file "${RAFT_CHECKPOINT}"
  require_dir "${DINO_MODEL}"
  require_dir "${VIF_INDEX_DIR}"
  python - "${NPROC_PER_NODE}" <<'PY'
import cv2
import numpy
import sys
import torch
import transformers

expected = int(sys.argv[1])
actual = torch.cuda.device_count()
assert actual >= expected, f"need {expected} GPUs, found {actual}"
print("python imports: OK")
print("opencv:", cv2.__version__)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("gpus:", torch.cuda.device_count())
PY
  build_manifests
  python - "${DATAB_MANIFEST}" "${VIF_MANIFEST}" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    assert rows
    assert all("dataa" not in str(row).casefold() for row in rows)
    assert all(row.get("camera_annotation_kind", "").endswith("stratification only") for row in rows)
    print(path, len(rows), "rows; DataA excluded; camera text excluded from classifier input")
PY
  echo "Preflight passed. No model inference was run."
}

extract_manifest() {
  manifest=$1
  max_samples=${2:-0}
  args=(
    --manifest-jsonl "${manifest}"
    --output-dir "${FEATURE_ROOT}"
    --raft-checkpoint "${RAFT_CHECKPOINT}"
    --dinov2-model "${DINO_MODEL}"
    --max-frames 16
  )
  if [[ "${max_samples}" -gt 0 ]]; then
    args+=(--max-samples "${max_samples}")
  fi
  torchrun --standalone --nproc-per-node="${NPROC_PER_NODE}" \
    -m scripts.camera_geometric_residual_gate.extract_features "${args[@]}"
}

audit_manifest() {
  manifest=$1
  name=$2
  max_samples=${3:-0}
  mkdir -p "${OUT}/audits"
  args=(
    --manifest-jsonl "${manifest}"
    --feature-root "${FEATURE_ROOT}"
    --output-json "${OUT}/audits/${name}_feature_audit.json"
    --min-coverage 0.99
  )
  if [[ "${max_samples}" -gt 0 ]]; then
    args+=(--max-samples "${max_samples}")
  fi
  python -m scripts.camera_geometric_residual_gate.audit_features "${args[@]}"
}

run_eval() {
  set +e
  python -m scripts.camera_geometric_residual_gate.train_gate \
    --train-manifest "${DATAB_MANIFEST}" \
    --test-manifest "${VIF_MANIFEST}" \
    --feature-root "${FEATURE_ROOT}" \
    --output-dir "${EVAL_DIR}" \
    --device cuda:0
  status=$?
  set -e
  persist_small_results
  return "${status}"
}

echo "=== 相机条件化几何残差最小验证 ==="
echo "stage=${STAGE}"
echo "DataA/CoT targets are excluded; camera labels are stratification-only."
echo "work_root=${OUT}"

case "${STAGE}" in
  preflight)
    preflight
    ;;
  build)
    build_manifests
    ;;
  smoke)
    preflight
    NPROC_PER_NODE=1 extract_manifest "${DATAB_MANIFEST}" 8
    NPROC_PER_NODE=1 extract_manifest "${VIF_MANIFEST}" 8
    audit_manifest "${DATAB_MANIFEST}" datab_smoke 8
    audit_manifest "${VIF_MANIFEST}" vif_smoke 8
    ;;
  extract)
    [[ -f "${DATAB_MANIFEST}" && -f "${VIF_MANIFEST}" ]] || build_manifests
    extract_manifest "${DATAB_MANIFEST}"
    extract_manifest "${VIF_MANIFEST}"
    audit_manifest "${DATAB_MANIFEST}" datab_full
    audit_manifest "${VIF_MANIFEST}" vif_full
    ;;
  eval)
    audit_manifest "${DATAB_MANIFEST}" datab_full
    audit_manifest "${VIF_MANIFEST}" vif_full
    run_eval
    ;;
  all)
    preflight
    extract_manifest "${DATAB_MANIFEST}"
    extract_manifest "${VIF_MANIFEST}"
    audit_manifest "${DATAB_MANIFEST}" datab_full
    audit_manifest "${VIF_MANIFEST}" vif_full
    run_eval
    ;;
  *)
    echo "Usage: STAGE={preflight|build|smoke|extract|eval|all} bash $0" >&2
    exit 2
    ;;
esac
