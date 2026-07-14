#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}
RUN_NAME=${RUN_NAME:-paper_asymmetric_inspection_evalable_100step}
RUN_ROOT=${RUN_ROOT:-/tmp/1res/skyra_grpo_diagnostics/${RUN_NAME}}
PERSIST_ROOT=${PERSIST_ROOT:-${ROOT}/res/skyra_grpo_diagnostics/${RUN_NAME}}
OSS_DEST=${OSS_DEST:-oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/skyra_grpo_diagnostics/${RUN_NAME}/}
MODEL_PATH=${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}
KEEP_ALIVE_AFTER_RUN=${KEEP_ALIVE_AFTER_RUN:-1}
KEEP_ALIVE_SCRIPT=${KEEP_ALIVE_SCRIPT:-/input/training/keep.sh}

mkdir -p "${RUN_ROOT}" "${PERSIST_ROOT}"
exec > >(tee -a "${PERSIST_ROOT}/launcher.log") 2>&1

echo "=== DataB paper-reward GRPO evaluable rerun ==="
echo "run_name=${RUN_NAME}"
echo "model_path=${MODEL_PATH}"
echo "run_root=${RUN_ROOT}"
echo "persistent_small_results=${PERSIST_ROOT}"
echo "oss_destination=${OSS_DEST}"
echo "checkpoints=global_step_50,global_step_100"

set +e
env \
  STAGE=all \
  RUN_MODE=formal \
  RUN_NAME="${RUN_NAME}" \
  RUN_ROOT="${RUN_ROOT}" \
  PERSIST_ROOT="${PERSIST_ROOT}" \
  MODEL_PATH="${MODEL_PATH}" \
  REWARD_VARIANT=paper_asymmetric_inspection \
  TOTAL_STEPS=100 \
  TRAIN_BATCH_SIZE=16 \
  GROUP_SIZE=8 \
  MAX_RESPONSE_LENGTH=768 \
  SAVE_FREQ=50 \
  KEEP_ALIVE_AFTER_RUN=0 \
  bash "${ROOT}/scripts/skyra_grpo_diagnostics/run.sh"
train_status=$?
set -e
echo "${train_status}" > "${PERSIST_ROOT}/training_exit_status.txt"

checkpoint_status=0
inventory="${PERSIST_ROOT}/checkpoint_inventory.txt"
: > "${inventory}"
for step in 50 100; do
  checkpoint_dir="${RUN_ROOT}/checkpoints/global_step_${step}"
  if [[ ! -d "${checkpoint_dir}/actor" ]]; then
    echo "MISSING global_step_${step}/actor" | tee -a "${inventory}"
    checkpoint_status=1
    continue
  fi
  echo "FOUND global_step_${step}/actor" | tee -a "${inventory}"
  du -sh "${checkpoint_dir}" | tee -a "${inventory}"
  find "${checkpoint_dir}/actor" -maxdepth 1 -type f -printf '%f\n' | sort | tee -a "${inventory}"
done
echo "${checkpoint_status}" > "${PERSIST_ROOT}/checkpoint_audit_exit_status.txt"

set +e
ossutil64 cp -r "${RUN_ROOT}/" "${OSS_DEST}" 2>&1 | tee "${PERSIST_ROOT}/oss_upload.log"
upload_status=${PIPESTATUS[0]}
set -e
echo "${upload_status}" > "${PERSIST_ROOT}/oss_upload_exit_status.txt"

final_status=0
if [[ "${train_status}" -ne 0 || "${checkpoint_status}" -ne 0 || "${upload_status}" -ne 0 ]]; then
  final_status=1
fi
echo "${final_status}" > "${PERSIST_ROOT}/pipeline_exit_status.txt"

echo "training_exit_status=${train_status}"
echo "checkpoint_audit_exit_status=${checkpoint_status}"
echo "oss_upload_exit_status=${upload_status}"
echo "pipeline_exit_status=${final_status}"

if [[ "${KEEP_ALIVE_AFTER_RUN}" == "1" ]]; then
  if [[ ! -f "${KEEP_ALIVE_SCRIPT}" ]]; then
    echo "Missing keepalive script: ${KEEP_ALIVE_SCRIPT}" >&2
    exit 2
  fi
  echo "Experiment and OSS upload finished. Starting keepalive: ${KEEP_ALIVE_SCRIPT}"
  exec bash "${KEEP_ALIVE_SCRIPT}"
fi

exit "${final_status}"
