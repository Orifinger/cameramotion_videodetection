#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
STAGE="${STAGE:-all}"

PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
WORK_ROOT="${WORK_ROOT:-/tmp/1res/caspr_gate1}"
DATA_DIR="${DATA_DIR:-${WORK_ROOT}/data}"
CONTROL_DIR="${CONTROL_DIR:-${WORK_ROOT}/train/control}"
METHOD_DIR="${METHOD_DIR:-${WORK_ROOT}/train/pair_rank}"
SCORE_ROOT="${SCORE_ROOT:-${WORK_ROOT}/scores}"
EVAL_DIR="${EVAL_DIR:-${WORK_ROOT}/eval}"

DATAA_DETECTION_JSON="${DATAA_DETECTION_JSON:-${PROJECT_ROOT}/res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json}"
DATAA_CAMERA_JSONL="${DATAA_CAMERA_JSONL:-${PROJECT_ROOT}/camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl}"
DATAA_TRAIN_JSON="${DATAA_TRAIN_JSON:-${PROJECT_ROOT}/tools/data/camera_motion_splits/dataA_train.json}"
DATAA_DEV_JSON="${DATAA_DEV_JSON:-${PROJECT_ROOT}/tools/data/camera_motion_splits/dataA_test.json}"
USE_EXPLICIT_DATAA_TRAIN="${USE_EXPLICIT_DATAA_TRAIN:-0}"
DATAB_DETECTION_JSON="${DATAB_DETECTION_JSON:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json}"

TRAIN_PAIRS="${DATA_DIR}/dataa_train_pairs_256.jsonl"
DATAB_REPLAY="${DATA_DIR}/datab_replay_512.jsonl"
DEV_PAIRS="${DATA_DIR}/dataa_dev_pairs.jsonl"
MAX_STEPS="${MAX_STEPS:-64}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
CHECK_IMAGES="${CHECK_IMAGES:-1}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing file: $1" >&2
    exit 2
  fi
}

build_data() {
  local check_args=()
  local train_args=()
  if [[ "${CHECK_IMAGES}" == "1" ]]; then check_args+=(--check-images); fi
  if [[ "${USE_EXPLICIT_DATAA_TRAIN}" == "1" ]]; then
    require_file "${DATAA_TRAIN_JSON}"
    train_args+=(--dataa-train-json "${DATAA_TRAIN_JSON}")
  fi
  "${PYTHON_BIN}" tools/build_caspr_gate1_data.py \
    --dataa-detection-json "${DATAA_DETECTION_JSON}" \
    --dataa-camera-jsonl "${DATAA_CAMERA_JSONL}" \
    --dataa-dev-json "${DATAA_DEV_JSON}" \
    --datab-detection-json "${DATAB_DETECTION_JSON}" \
    --out-dir "${DATA_DIR}" \
    --num-train-pairs 256 \
    --num-datab-replay 512 \
    --frames-per-video 16 \
    --seed 20260712 \
    "${train_args[@]}" \
    "${check_args[@]}"
}

train_one() {
  local mode="$1"
  local output_dir="$2"
  require_file "${TRAIN_PAIRS}"
  require_file "${DATAB_REPLAY}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.caspr_gate1.train_verdict_rank \
    --model-path "${MODEL_PATH}" \
    --train-pairs-jsonl "${TRAIN_PAIRS}" \
    --datab-replay-jsonl "${DATAB_REPLAY}" \
    --output-dir "${output_dir}" \
    --mode "${mode}" \
    --max-steps "${MAX_STEPS}" \
    --learning-rate 2e-5 \
    --pair-loss-weight 0.2 \
    --pair-margin 0.5 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --max-pixels "${MAX_PIXELS}" \
    --save-steps 32 \
    --seed 20260712
}

score_one() {
  local name="$1"
  local adapter="$2"
  local output_dir="${SCORE_ROOT}/${name}"
  require_file "${DEV_PAIRS}"
  if [[ ! -d "${adapter}" ]]; then
    echo "Missing adapter: ${adapter}" >&2
    exit 2
  fi
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.caspr_gate1.score_pairs \
    --model-path "${MODEL_PATH}" \
    --adapter-path "${adapter}" \
    --pairs-jsonl "${DEV_PAIRS}" \
    --output-dir "${output_dir}" \
    --model-name "${name}" \
    --max-pixels "${MAX_PIXELS}" \
    --seed 20260712
}

score_base() {
  require_file "${DEV_PAIRS}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.caspr_gate1.score_pairs \
    --model-path "${MODEL_PATH}" \
    --pairs-jsonl "${DEV_PAIRS}" \
    --output-dir "${SCORE_ROOT}/base" \
    --model-name "base_detection_checkpoint" \
    --max-pixels "${MAX_PIXELS}" \
    --seed 20260712
}

evaluate() {
  "${PYTHON_BIN}" -m scripts.caspr_gate1.eval_gate \
    --control-scores "${SCORE_ROOT}/control" \
    --method-scores "${SCORE_ROOT}/pair_rank" \
    --baseline-scores "${SCORE_ROOT}/base" \
    --out-dir "${EVAL_DIR}" \
    --bootstrap-repeats 1000 \
    --seed 20260712
}

case "${STAGE}" in
  build)
    build_data
    ;;
  smoke)
    if [[ ! -f "${TRAIN_PAIRS}" ]]; then build_data; fi
    MAX_STEPS=2 NPROC_PER_NODE=1 train_one pair_rank "${WORK_ROOT}/smoke/pair_rank"
    ;;
  train_control)
    train_one control "${CONTROL_DIR}"
    ;;
  train_method)
    train_one pair_rank "${METHOD_DIR}"
    ;;
  score_control)
    score_one control "${CONTROL_DIR}/final"
    ;;
  score_base)
    score_base
    ;;
  score_method)
    score_one pair_rank "${METHOD_DIR}/final"
    ;;
  eval)
    evaluate
    ;;
  all)
    build_data
    train_one control "${CONTROL_DIR}"
    train_one pair_rank "${METHOD_DIR}"
    score_base
    score_one control "${CONTROL_DIR}/final"
    score_one pair_rank "${METHOD_DIR}/final"
    evaluate
    ;;
  *)
    echo "STAGE must be build, smoke, train_control, train_method, score_base, score_control, score_method, eval, or all" >&2
    exit 2
    ;;
esac
