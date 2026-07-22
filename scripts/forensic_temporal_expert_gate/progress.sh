#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-both}"
PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
META_ROOT="${META_ROOT:-${PROJECT_ROOT}/res/forensic_temporal_expert_gate/v1}"
A_ROOT="${A_ROOT:-/tmp/1res/forensic_temporal_expert_gate/v1/server_a}"
B_ROOT="${B_ROOT:-/tmp/1res/forensic_temporal_expert_gate/v1/server_b}"

count_features() {
  local root="$1"
  find "${root}/features" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l
}

if [[ "${ROLE}" == "A" || "${ROLE}" == "both" ]]; then
  echo "=== 服务器 A ==="
  echo "DataB features: $(count_features "${A_ROOT}/datab_features") / 6766"
  echo "Complete models: $(find "${META_ROOT}/model_bundle/models" -name model.pt 2>/dev/null | wc -l) / 9"
  if [[ -f "${A_ROOT}/pipeline.log" ]]; then
    tail -n 5 "${A_ROOT}/pipeline.log"
  fi
fi

if [[ "${ROLE}" == "B" || "${ROLE}" == "both" ]]; then
  echo "=== 服务器 B ==="
  echo "ViF features: $(count_features "${B_ROOT}/vifbench_features") / 3160"
  test -f "${META_ROOT}/eval/forensic_temporal_expert_gate1_summary.json" && echo "Gate 1 summary: READY" || echo "Gate 1 summary: pending"
  test -f "${META_ROOT}/eval/forensic_temporal_expert_gate2_summary.json" && echo "Gate 2 summary: READY" || echo "Gate 2 summary: pending"
  if [[ -f "${B_ROOT}/pipeline.log" ]]; then
    tail -n 5 "${B_ROOT}/pipeline.log"
  fi
fi
