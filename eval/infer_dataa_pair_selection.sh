#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"

V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR:-/input/workflow_58770161/workspace/test/test_selfcot/Skyra/eval}"
DATAA_DETECTION_JSON="${DATAA_DETECTION_JSON:-/input/workflow_58770161/workspace/test/cameramotion_det/detection/dataa_vace_grounded_cot_instruct_tp8x2_sft_all.json}"
DATAA_CAMERA_JSONL="${DATAA_CAMERA_JSONL:-/input/workflow_58770161/workspace/test/cameramotion_det/camera/camerajson/dataa_cameramotion_labels_v2.jsonl}"

PAIR_JSON="${PAIR_JSON:-/tmp/1res/dataa_pair_selection_probe/dataa_pair_selection_200.json}"
MAX_PAIRS="${MAX_PAIRS:-200}"
PAIR_MAX_FRAMES_PER_VIDEO="${PAIR_MAX_FRAMES_PER_VIDEO:-8}"
SEED="${SEED:-20260709}"
REBUILD_PAIR_JSON="${REBUILD_PAIR_JSON:-0}"

MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall/checkpoint-2115}"
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-8B-v4vif-dataa-pair-selection}"
SAVE_DIR="${SAVE_DIR:-/tmp/1res/dataa_pair_selection_probe/${MODEL_NAME}}"
WORLD_SIZE="${WORLD_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OVERWRITE="${OVERWRITE:-0}"
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-262144}"

if [[ ! -f "${V4TRAIN_EVAL_DIR}/infer_dataa.py" ]]; then
  echo "Missing ${V4TRAIN_EVAL_DIR}/infer_dataa.py"
  echo "Set V4TRAIN_EVAL_DIR to the directory that contains v4train/eval/infer_dataa.py."
  exit 2
fi

if [[ "${REBUILD_PAIR_JSON}" == "1" || ! -f "${PAIR_JSON}" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/build_dataa_pair_region_pretext.py" \
    --detection-json "${DATAA_DETECTION_JSON}" \
    --camera-jsonl "${DATAA_CAMERA_JSONL}" \
    --out "${PAIR_JSON}" \
    --task pair \
    --max-pairs "${MAX_PAIRS}" \
    --pair-max-frames-per-video "${PAIR_MAX_FRAMES_PER_VIDEO}" \
    --seed "${SEED}"
fi

mkdir -p "${SAVE_DIR}"

export PYTHON_BIN V4TRAIN_EVAL_DIR PAIR_JSON MODEL_PATH MODEL_NAME SAVE_DIR WORLD_SIZE
export MAX_NEW_TOKENS MAX_SAMPLES OVERWRITE IMAGE_MIN_PIXELS IMAGE_MAX_PIXELS

seq 0 "$((WORLD_SIZE - 1))" | xargs -n1 -P "${WORLD_SIZE}" bash -lc '
  RANK="$1"
  CMD=(
    "${PYTHON_BIN}" "${V4TRAIN_EVAL_DIR}/infer_dataa.py"
    --sft_json "${PAIR_JSON}"
    --model_path "${MODEL_PATH}"
    --model_name "${MODEL_NAME}-rank${RANK}"
    --save_dir "${SAVE_DIR}/rank_${RANK}"
    --rank "${RANK}"
    --world_size "${WORLD_SIZE}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --prompt_mode record
  )

  if [[ -n "${MAX_SAMPLES}" ]]; then
    CMD+=(--max_samples "${MAX_SAMPLES}")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    CMD+=(--overwrite)
  fi
  if [[ -n "${IMAGE_MIN_PIXELS}" ]]; then
    CMD+=(--image_min_pixels "${IMAGE_MIN_PIXELS}")
  fi
  if [[ -n "${IMAGE_MAX_PIXELS}" ]]; then
    CMD+=(--image_max_pixels "${IMAGE_MAX_PIXELS}")
  fi

  CUDA_VISIBLE_DEVICES="${RANK}" "${CMD[@]}"
' _
