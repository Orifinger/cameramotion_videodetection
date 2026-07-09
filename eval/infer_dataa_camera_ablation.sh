#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"

V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR:-/input/workflow_58770161/workspace/test/test_selfcot/Skyra/eval}"
DATAA_TEST_JSON="${DATAA_TEST_JSON:-/input/workflow_58770161/workspace/test/cameramotion_det/tools/data/camera_motion_splits/dataA_test.json}"
DATAA_CAMERA_JSONL="${DATAA_CAMERA_JSONL:-/input/workflow_58770161/workspace/test/cameramotion_det/camera/camerajson/dataa_cameramotion_labels_v2.jsonl}"
ABLATION_DIR="${ABLATION_DIR:-/tmp/1res/dataa_camera_context_ablation}"
PREFIX="${PREFIX:-dataa_test}"
VARIANT="${VARIANT:-all}"
SEED="${SEED:-20260709}"
REBUILD_JSON="${REBUILD_JSON:-0}"
DROP_MISSING_CAMERA="${DROP_MISSING_CAMERA:-1}"
MAX_RECORDS="${MAX_RECORDS:-0}"

MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall/checkpoint-2115}"
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-8B-camera-context-ablation}"
SAVE_ROOT="${SAVE_ROOT:-${ABLATION_DIR}/${MODEL_NAME}}"
WORLD_SIZE="${WORLD_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OVERWRITE="${OVERWRITE:-0}"
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-262144}"
PROMPT_MODE="${PROMPT_MODE:-record}"

if [[ ! -f "${V4TRAIN_EVAL_DIR}/infer_dataa.py" ]]; then
  echo "Missing ${V4TRAIN_EVAL_DIR}/infer_dataa.py"
  exit 2
fi

if [[ "${REBUILD_JSON}" == "1" || ! -f "${ABLATION_DIR}/${PREFIX}_gold_camera.json" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/build_dataa_camera_context_ablation.py" \
    --input-json "${DATAA_TEST_JSON}" \
    --camera-jsonl "${DATAA_CAMERA_JSONL}" \
    --out-dir "${ABLATION_DIR}" \
    --prefix "${PREFIX}" \
    --seed "${SEED}" \
    --max-records "${MAX_RECORDS}" \
    $(if [[ "${DROP_MISSING_CAMERA}" == "1" ]]; then echo --drop-missing-camera; fi)
fi

if [[ "${VARIANT}" == "all" ]]; then
  VARIANTS=(no_camera gold_camera shuffled_camera null_camera)
else
  VARIANTS=(${VARIANT})
fi

run_variant() {
  local variant="$1"
  local sft_json="${ABLATION_DIR}/${PREFIX}_${variant}.json"
  local save_dir="${SAVE_ROOT}/${variant}"
  mkdir -p "${save_dir}"

  export PYTHON_BIN V4TRAIN_EVAL_DIR MODEL_PATH MODEL_NAME WORLD_SIZE MAX_NEW_TOKENS MAX_SAMPLES OVERWRITE IMAGE_MIN_PIXELS IMAGE_MAX_PIXELS PROMPT_MODE
  export sft_json save_dir variant

  seq 0 "$((WORLD_SIZE - 1))" | xargs -n1 -P "${WORLD_SIZE}" bash -lc '
    RANK="$1"
    CMD=(
      "${PYTHON_BIN}" "${V4TRAIN_EVAL_DIR}/infer_dataa.py"
      --sft_json "${sft_json}"
      --model_path "${MODEL_PATH}"
      --model_name "${MODEL_NAME}-${variant}-rank${RANK}"
      --save_dir "${save_dir}/rank_${RANK}"
      --rank "${RANK}"
      --world_size "${WORLD_SIZE}"
      --max_new_tokens "${MAX_NEW_TOKENS}"
      --prompt_mode "${PROMPT_MODE}"
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
}

for variant in "${VARIANTS[@]}"; do
  echo "=== Inference variant: ${variant} ==="
  run_variant "${variant}"
done
