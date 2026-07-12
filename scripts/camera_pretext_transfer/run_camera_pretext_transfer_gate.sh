#!/usr/bin/env bash
set -euo pipefail

STAGE="${STAGE:-build}"
PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
WORK_ROOT="${WORK_ROOT:-/tmp/1res/camera_pretext_transfer_gate}"
DATA_DIR="${WORK_ROOT}/data"
CAMERA_ROOT="${WORK_ROOT}/camera_sft"
CAMERA_PRED_ROOT="${WORK_ROOT}/camera_predictions"
CAMERA_EVAL_ROOT="${WORK_ROOT}/camera_eval"
TRANSFER_ROOT="${WORK_ROOT}/detection_transfer"
CASPR_ROOT="${CASPR_ROOT:-/tmp/1res/caspr_gate1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
CAMERA_STEPS="${CAMERA_STEPS:-48}"
CAMERA_SAVE_STEPS="${CAMERA_SAVE_STEPS:-24}"
CAMERA_CHECKPOINT_STEP="${CAMERA_CHECKPOINT_STEP:-48}"
VISUAL_CONTROL_STEP="${VISUAL_CONTROL_STEP:-96}"
CHECK_IMAGES="${CHECK_IMAGES:-1}"

DATAA_DETECTION_JSON="${DATAA_DETECTION_JSON:-${PROJECT_ROOT}/res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json}"
DATAA_CAMERA_JSONL="${DATAA_CAMERA_JSONL:-${PROJECT_ROOT}/camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl}"
DATAA_DEV_JSON="${DATAA_DEV_JSON:-${PROJECT_ROOT}/tools/data/camera_motion_splits/dataA_test.json}"
CAMERA_TRAIN_CORRECT="${DATA_DIR}/camera_train_correct.jsonl"
CAMERA_TRAIN_SHUFFLED="${DATA_DIR}/camera_train_shuffled.jsonl"
CAMERA_DEV_CANONICAL="${DATA_DIR}/camera_dev_canonical.jsonl"
CAMERA_DEV_PARAPHRASED="${DATA_DIR}/camera_dev_paraphrased.jsonl"
CAMERA_DEV_SHUFFLED_FRAMES="${DATA_DIR}/camera_dev_shuffled_frames.jsonl"
TRAIN_PAIRS="${CASPR_ROOT}/data/dataa_train_pairs_256.jsonl"
DATAB_REPLAY="${CASPR_ROOT}/data/datab_replay_512.jsonl"
DEV_PAIRS="${CASPR_ROOT}/data/dataa_dev_pairs.jsonl"
NO_PRETEXT_SCORES="${CASPR_ROOT}/scores/pair_rank"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing file: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing directory: $1" >&2
    exit 2
  fi
}

build_data() {
  if [[ "${CHECK_IMAGES}" == "1" ]]; then
    "${PYTHON_BIN}" tools/build_camera_pretext_transfer_gate.py \
      --dataa-detection-json "${DATAA_DETECTION_JSON}" \
      --dataa-camera-jsonl "${DATAA_CAMERA_JSONL}" \
      --dataa-dev-json "${DATAA_DEV_JSON}" \
      --out-dir "${DATA_DIR}" \
      --seed 20260712 \
      --check-images
  else
    "${PYTHON_BIN}" tools/build_camera_pretext_transfer_gate.py \
      --dataa-detection-json "${DATAA_DETECTION_JSON}" \
      --dataa-camera-jsonl "${DATAA_CAMERA_JSONL}" \
      --dataa-dev-json "${DATAA_DEV_JSON}" \
      --out-dir "${DATA_DIR}" \
      --seed 20260712
  fi
}

train_camera() {
  local target_kind="$1"
  local train_jsonl="$2"
  local initial_adapter="${3:-}"
  local output_dir="${CAMERA_ROOT}/${target_kind}"
  require_file "${train_jsonl}"
  if [[ -n "${initial_adapter}" ]]; then
    require_dir "${initial_adapter}"
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
      -m scripts.camera_pretext_transfer.train_camera_sft \
      --model-path "${MODEL_PATH}" \
      --initial-adapter-path "${initial_adapter}" \
      --train-jsonl "${train_jsonl}" \
      --output-dir "${output_dir}" \
      --max-steps "${CAMERA_STEPS}" \
      --learning-rate 1e-5 \
      --max-pixels "${MAX_PIXELS}" \
      --save-steps "${CAMERA_SAVE_STEPS}" \
      --seed 20260712
  else
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
      -m scripts.camera_pretext_transfer.train_camera_sft \
      --model-path "${MODEL_PATH}" \
      --train-jsonl "${train_jsonl}" \
      --output-dir "${output_dir}" \
      --max-steps "${CAMERA_STEPS}" \
      --learning-rate 1e-5 \
      --lora-rank 32 \
      --lora-alpha 64 \
      --lora-dropout 0.05 \
      --max-pixels "${MAX_PIXELS}" \
      --save-steps "${CAMERA_SAVE_STEPS}" \
      --seed 20260712
  fi
}

infer_camera() {
  local name="$1"
  local adapter="$2"
  local eval_jsonl="$3"
  local output_dir="${CAMERA_PRED_ROOT}/${name}"
  require_file "${eval_jsonl}"
  if [[ -n "${adapter}" ]]; then
    require_dir "${adapter}"
  fi
  rm -rf "${output_dir}"
  if [[ -n "${adapter}" ]]; then
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
      -m scripts.camera_pretext_transfer.infer_camera \
      --model-path "${MODEL_PATH}" \
      --adapter-path "${adapter}" \
      --eval-jsonl "${eval_jsonl}" \
      --output-dir "${output_dir}" \
      --model-name "${name}" \
      --max-pixels "${MAX_PIXELS}" \
      --seed 20260712
  else
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
      -m scripts.camera_pretext_transfer.infer_camera \
      --model-path "${MODEL_PATH}" \
      --eval-jsonl "${eval_jsonl}" \
      --output-dir "${output_dir}" \
      --model-name "${name}" \
      --max-pixels "${MAX_PIXELS}" \
      --seed 20260712
  fi
}

eval_camera_one() {
  local name="$1"
  local gold="$2"
  "${PYTHON_BIN}" -m scripts.camera_pretext_transfer.eval_camera \
    --gold-jsonl "${gold}" \
    --predictions "${CAMERA_PRED_ROOT}/${name}" \
    --output-json "${CAMERA_EVAL_ROOT}/${name}.json" \
    --model-name "${name}"
}

eval_stage1() {
  mkdir -p "${CAMERA_EVAL_ROOT}"
  eval_camera_one base "${CAMERA_DEV_CANONICAL}"
  for step in 24 48; do
    eval_camera_one "correct_${step}" "${CAMERA_DEV_CANONICAL}"
    eval_camera_one "shuffled_${step}" "${CAMERA_DEV_CANONICAL}"
    "${PYTHON_BIN}" -m scripts.camera_pretext_transfer.eval_stage1_gate \
      --base-summary "${CAMERA_EVAL_ROOT}/base.json" \
      --correct-summary "${CAMERA_EVAL_ROOT}/correct_${step}.json" \
      --shuffled-summary "${CAMERA_EVAL_ROOT}/shuffled_${step}.json" \
      --output-json "${CAMERA_EVAL_ROOT}/stage1_gate_step_${step}.json"
  done
}

train_transfer() {
  local target_kind="$1"
  local initial_adapter="${CAMERA_ROOT}/${target_kind}/checkpoint-${CAMERA_CHECKPOINT_STEP}"
  require_dir "${initial_adapter}"
  require_file "${TRAIN_PAIRS}"
  require_file "${DATAB_REPLAY}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.caspr_gate1.train_verdict_rank \
    --model-path "${MODEL_PATH}" \
    --initial-adapter-path "${initial_adapter}" \
    --train-pairs-jsonl "${TRAIN_PAIRS}" \
    --datab-replay-jsonl "${DATAB_REPLAY}" \
    --output-dir "${TRANSFER_ROOT}/${target_kind}" \
    --mode pair_rank \
    --max-steps 64 \
    --learning-rate 2e-5 \
    --pair-loss-weight 0.2 \
    --pair-margin 0.5 \
    --max-pixels "${MAX_PIXELS}" \
    --save-steps 32 \
    --seed 20260712
}

score_transfer() {
  local target_kind="$1"
  local adapter="${TRANSFER_ROOT}/${target_kind}/final"
  local output_dir="${TRANSFER_ROOT}/scores/${target_kind}"
  require_dir "${adapter}"
  require_file "${DEV_PAIRS}"
  rm -rf "${output_dir}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.caspr_gate1.score_pairs \
    --model-path "${MODEL_PATH}" \
    --adapter-path "${adapter}" \
    --pairs-jsonl "${DEV_PAIRS}" \
    --output-dir "${output_dir}" \
    --model-name "camera_${target_kind}_then_pair_rank" \
    --max-pixels "${MAX_PIXELS}" \
    --seed 20260712
}

eval_stage2() {
  require_dir "${NO_PRETEXT_SCORES}"
  "${PYTHON_BIN}" -m scripts.camera_pretext_transfer.eval_stage2_transfer \
    --no-pretext-scores "${NO_PRETEXT_SCORES}" \
    --correct-camera-scores "${TRANSFER_ROOT}/scores/correct" \
    --shuffled-camera-scores "${TRANSFER_ROOT}/scores/shuffled" \
    --output-json "${TRANSFER_ROOT}/stage2_transfer_gate.json" \
    --bootstrap-repeats 1000 \
    --seed 20260712
}

case "${STAGE}" in
  build) build_data ;;
  smoke_correct)
    require_file "${CAMERA_TRAIN_CORRECT}"
    CAMERA_STEPS=2 NPROC_PER_NODE=1 train_camera correct_smoke "${CAMERA_TRAIN_CORRECT}"
    ;;
  train_correct) train_camera correct "${CAMERA_TRAIN_CORRECT}" ;;
  train_shuffled) train_camera shuffled "${CAMERA_TRAIN_SHUFFLED}" ;;
  infer_camera_base) infer_camera base "" "${CAMERA_DEV_CANONICAL}" ;;
  infer_camera_correct_24) infer_camera correct_24 "${CAMERA_ROOT}/correct/checkpoint-24" "${CAMERA_DEV_CANONICAL}" ;;
  infer_camera_correct_48) infer_camera correct_48 "${CAMERA_ROOT}/correct/checkpoint-48" "${CAMERA_DEV_CANONICAL}" ;;
  infer_camera_shuffled_24) infer_camera shuffled_24 "${CAMERA_ROOT}/shuffled/checkpoint-24" "${CAMERA_DEV_CANONICAL}" ;;
  infer_camera_shuffled_48) infer_camera shuffled_48 "${CAMERA_ROOT}/shuffled/checkpoint-48" "${CAMERA_DEV_CANONICAL}" ;;
  eval_stage1) eval_stage1 ;;
  train_correct_clean_4epoch)
    CAMERA_STEPS=192 CAMERA_SAVE_STEPS=48 train_camera correct_clean_4epoch "${CAMERA_TRAIN_CORRECT}"
    ;;
  train_shuffled_clean_4epoch)
    CAMERA_STEPS=192 CAMERA_SAVE_STEPS=48 train_camera shuffled_clean_4epoch "${CAMERA_TRAIN_SHUFFLED}"
    ;;
  infer_camera_clean_4epoch)
    for step in 48 96 144 192; do
      infer_camera "correct_clean_${step}" \
        "${CAMERA_ROOT}/correct_clean_4epoch/checkpoint-${step}" "${CAMERA_DEV_CANONICAL}"
      infer_camera "shuffled_clean_${step}" \
        "${CAMERA_ROOT}/shuffled_clean_4epoch/checkpoint-${step}" "${CAMERA_DEV_CANONICAL}"
    done
    ;;
  eval_stage1_clean_4epoch)
    mkdir -p "${CAMERA_EVAL_ROOT}"
    eval_camera_one base "${CAMERA_DEV_CANONICAL}"
    for step in 48 96 144 192; do
      eval_camera_one "correct_clean_${step}" "${CAMERA_DEV_CANONICAL}"
      eval_camera_one "shuffled_clean_${step}" "${CAMERA_DEV_CANONICAL}"
      "${PYTHON_BIN}" -m scripts.camera_pretext_transfer.eval_stage1_gate \
        --base-summary "${CAMERA_EVAL_ROOT}/base.json" \
        --correct-summary "${CAMERA_EVAL_ROOT}/correct_clean_${step}.json" \
        --shuffled-summary "${CAMERA_EVAL_ROOT}/shuffled_clean_${step}.json" \
        --output-json "${CAMERA_EVAL_ROOT}/stage1_gate_clean_step_${step}.json"
    done
    "${PYTHON_BIN}" -m scripts.camera_pretext_transfer.summarize_stage1_curve \
      --eval-dir "${CAMERA_EVAL_ROOT}" \
      --output-json "${CAMERA_EVAL_ROOT}/stage1_clean_4epoch_curve.json"
    ;;
  build_stage1_visual_control)
    "${PYTHON_BIN}" -m scripts.camera_pretext_transfer.build_shuffled_frame_eval \
      --canonical-dev-jsonl "${CAMERA_DEV_CANONICAL}" \
      --output-jsonl "${CAMERA_DEV_SHUFFLED_FRAMES}" \
      --summary-json "${DATA_DIR}/camera_dev_shuffled_frames_summary.json"
    ;;
  infer_stage1_visual_control)
    infer_camera "correct_clean_${VISUAL_CONTROL_STEP}_shuffled_frames" \
      "${CAMERA_ROOT}/correct_clean_4epoch/checkpoint-${VISUAL_CONTROL_STEP}" \
      "${CAMERA_DEV_SHUFFLED_FRAMES}"
    ;;
  eval_stage1_visual_control)
    eval_camera_one "correct_clean_${VISUAL_CONTROL_STEP}" "${CAMERA_DEV_CANONICAL}"
    eval_camera_one "correct_clean_${VISUAL_CONTROL_STEP}_shuffled_frames" \
      "${CAMERA_DEV_SHUFFLED_FRAMES}"
    "${PYTHON_BIN}" -m scripts.camera_pretext_transfer.eval_visual_dependency \
      --matched-summary "${CAMERA_EVAL_ROOT}/correct_clean_${VISUAL_CONTROL_STEP}.json" \
      --shuffled-frame-summary \
        "${CAMERA_EVAL_ROOT}/correct_clean_${VISUAL_CONTROL_STEP}_shuffled_frames.json" \
      --frame-control-summary "${DATA_DIR}/camera_dev_shuffled_frames_summary.json" \
      --output-json "${CAMERA_EVAL_ROOT}/stage1_visual_dependency_step_${VISUAL_CONTROL_STEP}.json"
    ;;
  continue_correct_96)
    CAMERA_STEPS=48 train_camera correct_96 "${CAMERA_TRAIN_CORRECT}" \
      "${CAMERA_ROOT}/correct/checkpoint-48"
    ;;
  continue_shuffled_96)
    CAMERA_STEPS=48 train_camera shuffled_96 "${CAMERA_TRAIN_SHUFFLED}" \
      "${CAMERA_ROOT}/shuffled/checkpoint-48"
    ;;
  infer_camera_correct_96)
    infer_camera correct_96 "${CAMERA_ROOT}/correct_96/final" "${CAMERA_DEV_CANONICAL}"
    ;;
  infer_camera_shuffled_96)
    infer_camera shuffled_96 "${CAMERA_ROOT}/shuffled_96/final" "${CAMERA_DEV_CANONICAL}"
    ;;
  eval_stage1_96)
    eval_camera_one correct_96 "${CAMERA_DEV_CANONICAL}"
    eval_camera_one shuffled_96 "${CAMERA_DEV_CANONICAL}"
    "${PYTHON_BIN}" -m scripts.camera_pretext_transfer.eval_stage1_gate \
      --base-summary "${CAMERA_EVAL_ROOT}/base.json" \
      --correct-summary "${CAMERA_EVAL_ROOT}/correct_96.json" \
      --shuffled-summary "${CAMERA_EVAL_ROOT}/shuffled_96.json" \
      --output-json "${CAMERA_EVAL_ROOT}/stage1_gate_step_96.json"
    ;;
  infer_camera_paraphrase)
    infer_camera "correct_${CAMERA_CHECKPOINT_STEP}_paraphrased" \
      "${CAMERA_ROOT}/correct/checkpoint-${CAMERA_CHECKPOINT_STEP}" "${CAMERA_DEV_PARAPHRASED}"
    eval_camera_one "correct_${CAMERA_CHECKPOINT_STEP}_paraphrased" "${CAMERA_DEV_PARAPHRASED}"
    ;;
  train_transfer_correct) train_transfer correct ;;
  train_transfer_shuffled) train_transfer shuffled ;;
  score_transfer_correct) score_transfer correct ;;
  score_transfer_shuffled) score_transfer shuffled ;;
  eval_stage2) eval_stage2 ;;
  *)
    echo "Unknown STAGE=${STAGE}. See docs/camera_pretext_transfer_validation_20260712.md" >&2
    exit 2
    ;;
esac
