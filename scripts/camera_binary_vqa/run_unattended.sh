#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${STAGE:-all}"
PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
RUN_NAME="${RUN_NAME:-detection_checkpoint_start}"
RUN_ROOT="${RUN_ROOT:-/tmp/1res/dataa_camera_binary_vqa/${RUN_NAME}}"
PERSIST_ROOT="${PERSIST_ROOT:-${PROJECT_ROOT}/res/dataa_camera_binary_vqa/${RUN_NAME}}"
OSS_URI="${OSS_URI:-oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/dataa_camera_binary_vqa/${RUN_NAME}/}"
AUTO_UPLOAD_OSS="${AUTO_UPLOAD_OSS:-1}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
MAX_TRAIN_WALL_SECONDS="${MAX_TRAIN_WALL_SECONDS:-16200}"
VIDEO_FPS="${VIDEO_FPS:-8}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-16384}"
CPU_THREADS_PER_RANK="${CPU_THREADS_PER_RANK:-4}"
SEED="${SEED:-20260713}"

MANIFEST_JSONL="${MANIFEST_JSONL:-${PROJECT_ROOT}/res/camera_flow_probe_40step_v3/dataa_camera_flow_probe_manifest_40step_v3.jsonl}"
DATA_DIR="${RUN_ROOT}/data"
TRAIN_DIR="${RUN_ROOT}/train"
SCORE_ROOT="${RUN_ROOT}/scores"
EVAL_DIR="${RUN_ROOT}/eval"
LOG_PATH="${RUN_ROOT}/pipeline.log"

TRAIN_JSONL="${DATA_DIR}/train_balanced.jsonl"
DEV_MATCHED="${DATA_DIR}/dev_matched_video.jsonl"
DEV_OPPOSITE="${DATA_DIR}/dev_opposite_label_video.jsonl"
DEV_NO_VIDEO="${DATA_DIR}/dev_no_video.jsonl"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

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

persist_small_results() {
  mkdir -p "${PERSIST_ROOT}"
  if [[ -f "${DATA_DIR}/data_summary.json" ]]; then
    mkdir -p "${PERSIST_ROOT}/data"
    cp -a "${DATA_DIR}/data_summary.json" "${PERSIST_ROOT}/data/"
  fi
  if [[ -d "${EVAL_DIR}" ]]; then
    mkdir -p "${PERSIST_ROOT}/eval"
    cp -a "${EVAL_DIR}/." "${PERSIST_ROOT}/eval/"
  fi
  if [[ -d "${SCORE_ROOT}" ]]; then
    mkdir -p "${PERSIST_ROOT}/scores"
    cp -a "${SCORE_ROOT}/." "${PERSIST_ROOT}/scores/"
  fi
  if [[ -f "${TRAIN_DIR}/all_results.json" ]]; then
    mkdir -p "${PERSIST_ROOT}/train"
    cp -a "${TRAIN_DIR}/all_results.json" "${PERSIST_ROOT}/train/"
  fi
  if [[ -f "${TRAIN_DIR}/trainer_log.jsonl" ]]; then
    mkdir -p "${PERSIST_ROOT}/train"
    cp -a "${TRAIN_DIR}/trainer_log.jsonl" "${PERSIST_ROOT}/train/"
  fi
  cp -a "${LOG_PATH}" "${PERSIST_ROOT}/" 2>/dev/null || true
}

archive_on_exit() {
  local status=$?
  trap - EXIT
  set +e
  persist_small_results
  if [[ "${AUTO_UPLOAD_OSS}" == "1" && "${STAGE}" != "preflight" ]]; then
    ossutil64 cp -r "${RUN_ROOT}/" "${OSS_URI}"
  fi
  echo "Pipeline exit status: ${status}"
  echo "Persistent small results: ${PERSIST_ROOT}"
  echo "OSS destination: ${OSS_URI}"
  exit "${status}"
}
trap archive_on_exit EXIT

build_data() {
  require_file "${MANIFEST_JSONL}"
  "${PYTHON_BIN}" -m scripts.camera_binary_vqa.build_data \
    --manifest-jsonl "${MANIFEST_JSONL}" \
    --output-dir "${DATA_DIR}" \
    --max-dev-per-class 64 \
    --min-train-per-class 8 \
    --min-dev-per-class 4 \
    --minimum-eligible-labels 20 \
    --world-size "${NPROC_PER_NODE}" \
    --seed "${SEED}" \
    --check-videos
}

score_base() {
  mkdir -p "${SCORE_ROOT}/base"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.camera_binary_vqa.score \
    --model-path "${MODEL_PATH}" \
    --condition "matched_video=${DEV_MATCHED}" \
    --output-dir "${SCORE_ROOT}/base" \
    --model-stage base \
    --video-fps "${VIDEO_FPS}" \
    --video-max-pixels "${VIDEO_MAX_PIXELS}" \
    --cpu-threads-per-rank "${CPU_THREADS_PER_RANK}" \
    --seed "${SEED}"
}

train_model() {
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.camera_binary_vqa.train \
    --model-path "${MODEL_PATH}" \
    --train-jsonl "${TRAIN_JSONL}" \
    --output-dir "${TRAIN_DIR}" \
    --num-epochs "${NUM_EPOCHS}" \
    --max-wall-seconds "${MAX_TRAIN_WALL_SECONDS}" \
    --learning-rate 2e-4 \
    --lora-rank 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05 \
    --video-fps "${VIDEO_FPS}" \
    --video-max-pixels "${VIDEO_MAX_PIXELS}" \
    --cpu-threads-per-rank "${CPU_THREADS_PER_RANK}" \
    --seed "${SEED}"
}

score_epoch1() {
  require_dir "${TRAIN_DIR}/checkpoint-epoch-1"
  mkdir -p "${SCORE_ROOT}/epoch1"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.camera_binary_vqa.score \
    --model-path "${MODEL_PATH}" \
    --adapter-path "${TRAIN_DIR}/checkpoint-epoch-1" \
    --condition "matched_video=${DEV_MATCHED}" \
    --output-dir "${SCORE_ROOT}/epoch1" \
    --model-stage epoch1 \
    --video-fps "${VIDEO_FPS}" \
    --video-max-pixels "${VIDEO_MAX_PIXELS}" \
    --cpu-threads-per-rank "${CPU_THREADS_PER_RANK}" \
    --seed "${SEED}"
}

score_final() {
  require_dir "${TRAIN_DIR}/final"
  mkdir -p "${SCORE_ROOT}/final"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.camera_binary_vqa.score \
    --model-path "${MODEL_PATH}" \
    --adapter-path "${TRAIN_DIR}/final" \
    --condition "matched_video=${DEV_MATCHED}" \
    --condition "opposite_label_video=${DEV_OPPOSITE}" \
    --condition "no_video=${DEV_NO_VIDEO}" \
    --output-dir "${SCORE_ROOT}/final" \
    --model-stage final \
    --video-fps "${VIDEO_FPS}" \
    --video-max-pixels "${VIDEO_MAX_PIXELS}" \
    --cpu-threads-per-rank "${CPU_THREADS_PER_RANK}" \
    --seed "${SEED}"
}

evaluate_all() {
  mkdir -p "${EVAL_DIR}"
  "${PYTHON_BIN}" -m scripts.camera_binary_vqa.evaluate \
    --gold "matched_video=${DEV_MATCHED}" \
    --predictions-dir "${SCORE_ROOT}/base" \
    --model-stage base \
    --output-json "${EVAL_DIR}/base.json"
  "${PYTHON_BIN}" -m scripts.camera_binary_vqa.evaluate \
    --gold "matched_video=${DEV_MATCHED}" \
    --predictions-dir "${SCORE_ROOT}/epoch1" \
    --model-stage epoch1 \
    --output-json "${EVAL_DIR}/epoch1.json"
  "${PYTHON_BIN}" -m scripts.camera_binary_vqa.evaluate \
    --gold "matched_video=${DEV_MATCHED}" \
    --gold "opposite_label_video=${DEV_OPPOSITE}" \
    --gold "no_video=${DEV_NO_VIDEO}" \
    --predictions-dir "${SCORE_ROOT}/final" \
    --model-stage final \
    --output-json "${EVAL_DIR}/final.json"
  "${PYTHON_BIN}" -m scripts.camera_binary_vqa.summarize_gate \
    --base-eval "${EVAL_DIR}/base.json" \
    --epoch1-eval "${EVAL_DIR}/epoch1.json" \
    --final-eval "${EVAL_DIR}/final.json" \
    --training-state "${TRAIN_DIR}/all_results.json" \
    --output-json "${EVAL_DIR}/gate_summary.json"
  persist_small_results
}

preflight() {
  build_data
  mkdir -p "${RUN_ROOT}/preflight"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m scripts.camera_binary_vqa.score \
    --model-path "${MODEL_PATH}" \
    --condition "matched_video=${DEV_MATCHED}" \
    --output-dir "${RUN_ROOT}/preflight" \
    --model-stage preflight \
    --max-samples-per-condition 2 \
    --video-fps "${VIDEO_FPS}" \
    --video-max-pixels "${VIDEO_MAX_PIXELS}" \
    --cpu-threads-per-rank "${CPU_THREADS_PER_RANK}" \
    --seed "${SEED}"
}

export OMP_NUM_THREADS="${CPU_THREADS_PER_RANK}"
export TOKENIZERS_PARALLELISM=false

require_dir "${MODEL_PATH}"
echo "=== DataA balanced binary camera VQA unattended gate ==="
echo "stage=${STAGE} run_name=${RUN_NAME}"
echo "model_path=${MODEL_PATH}"
echo "run_root=${RUN_ROOT}"
echo "manifest=${MANIFEST_JSONL}"
echo "gpus=${NPROC_PER_NODE} fps=${VIDEO_FPS} epochs=${NUM_EPOCHS} train_wall=${MAX_TRAIN_WALL_SECONDS}s"

case "${STAGE}" in
  preflight)
    preflight
    ;;
  build)
    build_data
    ;;
  train)
    require_file "${TRAIN_JSONL}"
    train_model
    ;;
  score)
    require_file "${DEV_MATCHED}"
    score_base
    score_epoch1
    score_final
    ;;
  eval)
    evaluate_all
    ;;
  all)
    if [[ -f "${RUN_ROOT}/COMPLETED" ]]; then
      echo "Run already completed: ${RUN_ROOT}" >&2
      exit 2
    fi
    build_data
    score_base
    train_model
    score_epoch1
    score_final
    evaluate_all
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "${RUN_ROOT}/COMPLETED"
    ;;
  *)
    echo "Unknown STAGE=${STAGE}; expected preflight, build, train, score, eval, or all" >&2
    exit 2
    ;;
esac
