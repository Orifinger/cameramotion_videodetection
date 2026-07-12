#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preflight}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
OUT=${OUT:-/tmp/1res/camera_flow_probe_40step_v3}
META_DIR=${META_DIR:-${ROOT}/res/camera_flow_probe_40step_v3}

RECORDS_JSONL=${RECORDS_JSONL:-${ROOT}/res/dataA_v1/autolabel/dataa_vace_grounded_cot_v4_records_40step_v3.jsonl}
CAMERA_JSONL=${CAMERA_JSONL:-${ROOT}/camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl}
TEST_SPLIT=${TEST_SPLIT:-${ROOT}/tools/data/camera_motion_splits/dataA_test.json}
RAFT_CHECKPOINT=${RAFT_CHECKPOINT:-/home/admin/raft_large_C_T_SKHT_V2-ff5fadd5.pth}
DINO_MODEL=${DINO_MODEL:-/home/admin/dinov2-small}
SEA_RAFT_MODEL=${SEA_RAFT_MODEL:-/home/admin/MemorySlices/Tartan-C-T-TSKH-spring540x960-M}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}

MANIFEST=${META_DIR}/dataa_camera_flow_probe_manifest_40step_v3.jsonl
MANIFEST_SUMMARY=${META_DIR}/dataa_camera_flow_probe_manifest_40step_v3_summary.json
FEATURE_ROOT=${OUT}/full

mkdir -p "${META_DIR}"

export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}
cd "${ROOT}"

build_manifest() {
  python -m scripts.camera_flow_probe.build_manifest \
    --records-jsonl "${RECORDS_JSONL}" \
    --camera-jsonl "${CAMERA_JSONL}" \
    --test-split "${TEST_SPLIT}" \
    --out-jsonl "${MANIFEST}" \
    --out-summary "${MANIFEST_SUMMARY}" \
    --check-files
}

weight_preflight() {
  python -m scripts.camera_flow_probe.weight_preflight \
    --raft-checkpoint "${RAFT_CHECKPOINT}" \
    --dinov2-model "${DINO_MODEL}" \
    --sea-raft-model "${SEA_RAFT_MODEL}" \
    --out-json "${META_DIR}/weight_preflight.json"
}

case "${MODE}" in
  preflight)
    build_manifest
    weight_preflight
    ;;
  smoke)
    build_manifest
    weight_preflight
    SMOKE_MANIFEST=${OUT}/data/dataa_camera_flow_probe_smoke.jsonl
    python -m scripts.camera_flow_probe.select_manifest \
      --manifest-jsonl "${MANIFEST}" \
      --out-jsonl "${SMOKE_MANIFEST}" \
      --summary-json "${OUT}/data/dataa_camera_flow_probe_smoke_summary.json" \
      --split train \
      --per-source-motion 1
    torchrun --standalone --nproc-per-node=1 \
      -m scripts.camera_flow_probe.extract_features \
      --manifest-jsonl "${SMOKE_MANIFEST}" \
      --output-dir "${OUT}/smoke" \
      --raft-checkpoint "${RAFT_CHECKPOINT}" \
      --dinov2-model "${DINO_MODEL}" \
      --target-fps 8 \
      --window-frames 16 \
      --stride-frames 8 \
      --max-sampled-frames 24 \
      --max-windows 2
    python -m scripts.camera_flow_probe.summarize_extraction \
      --manifest-jsonl "${SMOKE_MANIFEST}" \
      --feature-dir "${OUT}/smoke/features" \
      --out-json "${OUT}/smoke/extraction_audit.json"
    python -m scripts.camera_flow_probe.visualize_flow \
      --manifest-jsonl "${SMOKE_MANIFEST}" \
      --output-dir "${OUT}/smoke/visualizations" \
      --raft-checkpoint "${RAFT_CHECKPOINT}" \
      --target-fps 8 \
      --max-cases 12
    ;;
  extract)
    test -f "${MANIFEST}" || build_manifest
    torchrun --standalone --nproc-per-node="${NPROC_PER_NODE}" \
      -m scripts.camera_flow_probe.extract_features \
      --manifest-jsonl "${MANIFEST}" \
      --output-dir "${FEATURE_ROOT}" \
      --raft-checkpoint "${RAFT_CHECKPOINT}" \
      --dinov2-model "${DINO_MODEL}" \
      --target-fps 8 \
      --window-frames 16 \
      --stride-frames 8
    ;;
  audit)
    python -m scripts.camera_flow_probe.summarize_extraction \
      --manifest-jsonl "${MANIFEST}" \
      --feature-dir "${FEATURE_ROOT}/features" \
      --out-json "${FEATURE_ROOT}/extraction_audit.json"
    cp "${FEATURE_ROOT}/extraction_audit.json" "${META_DIR}/full_extraction_audit.json"
    ;;
  probe)
    python -m scripts.camera_flow_probe.train_probe \
      --manifest-jsonl "${MANIFEST}" \
      --feature-dir "${FEATURE_ROOT}/features" \
      --output-dir "${META_DIR}/probe" \
      --device cuda:0
    ;;
  *)
    echo "Usage: $0 {preflight|smoke|extract|audit|probe}" >&2
    exit 2
    ;;
esac
