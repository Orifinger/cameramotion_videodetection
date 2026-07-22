#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
KEEP_ALIVE_AFTER_RUN="${KEEP_ALIVE_AFTER_RUN:-1}"
KEEP_ALIVE_SCRIPT="${KEEP_ALIVE_SCRIPT:-/input/training/keep.sh}"
A_SCRIPT="${PROJECT_ROOT}/scripts/forensic_temporal_expert_gate/run_server_a.sh"
B_SCRIPT="${PROJECT_ROOT}/scripts/forensic_temporal_expert_gate/run_server_b.sh"

cd "${PROJECT_ROOT}"

echo "=== 单服务器串行执行原生尺度 DINO 时序专家两层验证 ==="
echo "第一阶段：DataB 特征提取与三组等步数训练"
KEEP_ALIVE_AFTER_RUN=0 STAGE=all bash "${A_SCRIPT}"

echo "第二阶段：ViF-Bench 特征提取与两层验收"
KEEP_ALIVE_AFTER_RUN=0 STAGE=all bash "${B_SCRIPT}"

echo "单服务器全部阶段执行完成。"
echo "持久化结果：${PROJECT_ROOT}/res/forensic_temporal_expert_gate/v1"

if [[ "${KEEP_ALIVE_AFTER_RUN}" == "1" ]]; then
  exec bash "${KEEP_ALIVE_SCRIPT}"
fi
