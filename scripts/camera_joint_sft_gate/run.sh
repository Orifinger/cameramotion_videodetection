#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${STAGE:-preflight}"
PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
WORK_ROOT="${WORK_ROOT:-/tmp/1res/camera_joint_sft_gate}"
DATA_DIR="${WORK_ROOT}/data"
TRAIN_ROOT="${WORK_ROOT}/train"
PRED_ROOT="${WORK_ROOT}/camera_predictions"
EVAL_ROOT="${WORK_ROOT}/camera_eval"
READINESS_ROOT="${WORK_ROOT}/rl_readiness"
PERSIST_ROOT="${PERSIST_ROOT:-${PROJECT_ROOT}/res/camera_joint_sft_gate}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
CHECK_IMAGES="${CHECK_IMAGES:-1}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"
FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"

if [[ -z "${LLAMAFACTORY_ROOT:-}" ]]; then
  for candidate in \
    /input/workflow_58770161/workspace/test/test_selfcot/LlamaFactory/LlamaFactory \
    /input/workflow_58770161/workspace/test/test_selfcot/Skyra/train/LLaMA-Factory \
    /input/training/LlamaFactory/LlamaFactory
  do
    if [[ -f "${candidate}/examples/deepspeed/ds_z2_config.json" ]]; then
      LLAMAFACTORY_ROOT="${candidate}"
      break
    fi
  done
fi
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-/input/workflow_58770161/workspace/test/test_selfcot/LlamaFactory/LlamaFactory}"
if [[ -z "${LLAMAFACTORY_DATA_DIR:-}" ]]; then
  for candidate in \
    /input/workflow_58770161/workspace/test/test_selfcot/Skyra/train/LLaMA-Factory/data \
    "${LLAMAFACTORY_ROOT}/data" \
    /input/training/LlamaFactory/LlamaFactory/data
  do
    if [[ -f "${candidate}/dataset_info.json" ]]; then
      LLAMAFACTORY_DATA_DIR="${candidate}"
      break
    fi
  done
fi
LLAMAFACTORY_DATA_DIR="${LLAMAFACTORY_DATA_DIR:-${LLAMAFACTORY_ROOT}/data}"

DATAA_DETECTION_JSON="${DATAA_DETECTION_JSON:-${PROJECT_ROOT}/res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json}"
DATAA_CAMERA_JSONL="${DATAA_CAMERA_JSONL:-${PROJECT_ROOT}/camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl}"
DATAB_DETECTION_JSON="${DATAB_DETECTION_JSON:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json}"
DATAB_CAMERA_JSONL="${DATAB_CAMERA_JSONL:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl}"

CAMERA_DEV_MATCHED="${DATA_DIR}/camera_dev_matched_frames.jsonl"
CAMERA_DEV_OPPOSITE="${DATA_DIR}/camera_dev_opposite_frames.jsonl"
CAMERA_DEV_NO_FRAMES="${DATA_DIR}/camera_dev_no_frames.jsonl"
DETECTION_ONLY_ADAPTER="${DETECTION_ONLY_ADAPTER:-${TRAIN_ROOT}/detection_only}"
CORRECT_ADAPTER="${CORRECT_ADAPTER:-${TRAIN_ROOT}/correct_camera}"
SHUFFLED_ADAPTER="${SHUFFLED_ADAPTER:-${TRAIN_ROOT}/shuffled_camera}"
RUNTIME_CONFIG_DIR="${WORK_ROOT}/configs"
KEEP_ALIVE_AFTER_RUN="${KEEP_ALIVE_AFTER_RUN:-0}"
KEEP_ALIVE_SCRIPT="${KEEP_ALIVE_SCRIPT:-/input/training/keep.sh}"

mkdir -p "${WORK_ROOT}"

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

adapter_file_check() {
  require_dir "$1"
  require_file "$1/adapter_config.json"
  if [[ ! -f "$1/adapter_model.safetensors" && ! -f "$1/adapter_model.bin" ]]; then
    echo "Missing adapter weights under $1" >&2
    exit 2
  fi
}

require_llamafactory_layout() {
  require_dir "${LLAMAFACTORY_ROOT}"
  require_dir "${LLAMAFACTORY_DATA_DIR}"
  require_file "${LLAMAFACTORY_DATA_DIR}/dataset_info.json"
  require_file "${LLAMAFACTORY_ROOT}/examples/deepspeed/ds_z2_config.json"
}

persist_small_results() {
  mkdir -p "${PERSIST_ROOT}"
  for file in \
    "${DATA_DIR}/camera_joint_sft_data_summary.json" \
    "${DATA_DIR}/dataa_40step_v3_split_manifest.jsonl" \
    "${DATA_DIR}/llamafactory_install_summary.json" \
    "${EVAL_ROOT}/joint_sft_camera_gate_summary.json"
  do
    if [[ -f "${file}" ]]; then
      cp -a "${file}" "${PERSIST_ROOT}/"
    fi
  done
  if [[ -d "${EVAL_ROOT}" ]]; then
    mkdir -p "${PERSIST_ROOT}/camera_eval"
    find "${EVAL_ROOT}" -maxdepth 1 -type f -name '*.json' -exec cp -a {} "${PERSIST_ROOT}/camera_eval/" \;
  fi
  if [[ -d "${READINESS_ROOT}" ]]; then
    mkdir -p "${PERSIST_ROOT}/rl_readiness"
    find "${READINESS_ROOT}" -maxdepth 1 -type f -name '*.json' -exec cp -a {} "${PERSIST_ROOT}/rl_readiness/" \;
  fi
}

preflight() {
  echo "LlamaFactory root: ${LLAMAFACTORY_ROOT}"
  echo "LlamaFactory data: ${LLAMAFACTORY_DATA_DIR}"
  require_dir "${MODEL_PATH}"
  require_file "${MODEL_PATH}/config.json"
  require_file "${DATAA_DETECTION_JSON}"
  require_file "${DATAA_CAMERA_JSONL}"
  require_file "${DATAB_DETECTION_JSON}"
  require_file "${DATAB_CAMERA_JSONL}"
  require_llamafactory_layout
  command -v "${LLAMAFACTORY_CLI}" >/dev/null
  "${PYTHON_BIN}" -c 'import peft, qwen_vl_utils, torch, transformers, yaml; print("Python dependencies: OK"); print("transformers:", transformers.__version__); print("gpus:", torch.cuda.device_count())'
  echo "Preflight passed. No model loading or inference was run."
}

build_data() {
  require_llamafactory_layout
  mkdir -p "${DATA_DIR}"
  if [[ "${CHECK_IMAGES}" == "1" ]]; then
    "${PYTHON_BIN}" tools/build_camera_joint_sft_gate.py \
      --dataa-detection-json "${DATAA_DETECTION_JSON}" \
      --dataa-camera-jsonl "${DATAA_CAMERA_JSONL}" \
      --datab-detection-json "${DATAB_DETECTION_JSON}" \
      --datab-camera-jsonl "${DATAB_CAMERA_JSONL}" \
      --out-dir "${DATA_DIR}" \
      --test-ratio 0.30 \
      --expected-dataa-cases 1080 \
      --seed 20260713 \
      --check-images
    "${PYTHON_BIN}" tools/install_camera_joint_sft_gate.py \
      --source-dir "${DATA_DIR}" \
      --llamafactory-data-dir "${LLAMAFACTORY_DATA_DIR}" \
      --smoke-samples 96 \
      --seed 20260713 \
      --check-images
  else
    "${PYTHON_BIN}" tools/build_camera_joint_sft_gate.py \
      --dataa-detection-json "${DATAA_DETECTION_JSON}" \
      --dataa-camera-jsonl "${DATAA_CAMERA_JSONL}" \
      --datab-detection-json "${DATAB_DETECTION_JSON}" \
      --datab-camera-jsonl "${DATAB_CAMERA_JSONL}" \
      --out-dir "${DATA_DIR}" \
      --test-ratio 0.30 \
      --expected-dataa-cases 1080 \
      --seed 20260713
    "${PYTHON_BIN}" tools/install_camera_joint_sft_gate.py \
      --source-dir "${DATA_DIR}" \
      --llamafactory-data-dir "${LLAMAFACTORY_DATA_DIR}" \
      --smoke-samples 96 \
      --seed 20260713
  fi
  persist_small_results
}

render_config() {
  local name="$1"
  local source="configs/camera_joint_sft_gate/${name}.yaml"
  local destination="${RUNTIME_CONFIG_DIR}/${name}.yaml"
  require_file "${source}"
  mkdir -p "${RUNTIME_CONFIG_DIR}"
  sed \
    -e "s|model_name_or_path: .*|model_name_or_path: ${MODEL_PATH}|" \
    -e "s|dataset_dir: .*|dataset_dir: ${LLAMAFACTORY_DATA_DIR}|" \
    -e "s|deepspeed: .*|deepspeed: ${LLAMAFACTORY_ROOT}/examples/deepspeed/ds_z2_config.json|" \
    -e "s|/tmp/1res/camera_joint_sft_gate|${WORK_ROOT}|g" \
    "${source}" > "${destination}"
  echo "${destination}"
}

train_branch() {
  local branch="$1"
  local config
  config="$(render_config "train_${branch}")"
  FORCE_TORCHRUN="${FORCE_TORCHRUN}" "${LLAMAFACTORY_CLI}" train "${config}"
}

score_camera() {
  local name="$1"
  local adapter="$2"
  local output_dir="${PRED_ROOT}/${name}"
  adapter_file_check "${adapter}"
  require_file "${CAMERA_DEV_MATCHED}"
  require_file "${CAMERA_DEV_OPPOSITE}"
  require_file "${CAMERA_DEV_NO_FRAMES}"
  rm -rf "${output_dir}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.camera_joint_sft_gate.score_binary \
    --model-path "${MODEL_PATH}" \
    --adapter-path "${adapter}" \
    --condition "matched_frames=${CAMERA_DEV_MATCHED}" \
    --condition "opposite_frames=${CAMERA_DEV_OPPOSITE}" \
    --condition "no_frames=${CAMERA_DEV_NO_FRAMES}" \
    --output-dir "${output_dir}" \
    --model-stage "${name}" \
    --max-pixels "${MAX_PIXELS}" \
    --seed 20260713
}

evaluate_camera() {
  local name="$1"
  mkdir -p "${EVAL_ROOT}"
  "${PYTHON_BIN}" -m scripts.camera_binary_vqa.evaluate \
    --gold "matched_frames=${CAMERA_DEV_MATCHED}" \
    --gold "opposite_frames=${CAMERA_DEV_OPPOSITE}" \
    --gold "no_frames=${CAMERA_DEV_NO_FRAMES}" \
    --predictions-dir "${PRED_ROOT}/${name}" \
    --model-stage "${name}" \
    --output-json "${EVAL_ROOT}/${name}.json"
}

infer_and_evaluate_all_camera() {
  score_camera detection_only "${DETECTION_ONLY_ADAPTER}"
  evaluate_camera detection_only
  score_camera correct_camera "${CORRECT_ADAPTER}"
  evaluate_camera correct_camera
  score_camera shuffled_camera "${SHUFFLED_ADAPTER}"
  evaluate_camera shuffled_camera
  persist_small_results
}

sample_readiness() {
  local name="$1"
  local adapter="$2"
  local output_dir="${READINESS_ROOT}/${name}_rollouts"
  adapter_file_check "${adapter}"
  rm -rf "${output_dir}"
  mkdir -p "${READINESS_ROOT}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.camera_joint_sft_gate.sample_rollouts \
    --model-path "${MODEL_PATH}" \
    --adapter-path "${adapter}" \
    --eval-jsonl "${CAMERA_DEV_MATCHED}" \
    --output-dir "${output_dir}" \
    --model-name "${name}" \
    --rollouts-per-sample 8 \
    --max-new-tokens 8 \
    --max-pixels "${MAX_PIXELS}" \
    --seed 20260713
  "${PYTHON_BIN}" -m scripts.camera_joint_sft_gate.evaluate_readiness \
    --gold-jsonl "${CAMERA_DEV_MATCHED}" \
    --rollouts "${output_dir}" \
    --output-json "${READINESS_ROOT}/${name}.json" \
    --expected-k 8
  persist_small_results
}

summarize_gate() {
  require_file "${EVAL_ROOT}/detection_only.json"
  require_file "${EVAL_ROOT}/correct_camera.json"
  require_file "${EVAL_ROOT}/shuffled_camera.json"
  require_file "${READINESS_ROOT}/correct_camera.json"
  if [[ -f "${READINESS_ROOT}/shuffled_camera.json" ]]; then
    "${PYTHON_BIN}" -m scripts.camera_joint_sft_gate.summarize \
      --detection-only-camera-eval "${EVAL_ROOT}/detection_only.json" \
      --correct-camera-eval "${EVAL_ROOT}/correct_camera.json" \
      --shuffled-camera-eval "${EVAL_ROOT}/shuffled_camera.json" \
      --correct-readiness "${READINESS_ROOT}/correct_camera.json" \
      --shuffled-readiness "${READINESS_ROOT}/shuffled_camera.json" \
      --output-json "${EVAL_ROOT}/joint_sft_camera_gate_summary.json"
  else
    "${PYTHON_BIN}" -m scripts.camera_joint_sft_gate.summarize \
      --detection-only-camera-eval "${EVAL_ROOT}/detection_only.json" \
      --correct-camera-eval "${EVAL_ROOT}/correct_camera.json" \
      --shuffled-camera-eval "${EVAL_ROOT}/shuffled_camera.json" \
      --correct-readiness "${READINESS_ROOT}/correct_camera.json" \
      --output-json "${EVAL_ROOT}/joint_sft_camera_gate_summary.json"
  fi
  persist_small_results
}

run_dataa_retention() {
  local name="$1"
  local adapter="$2"
  adapter_file_check "${adapter}"
  STAGE=all \
  MODEL_PATH="${MODEL_PATH}" \
  ADAPTER_PATH="${adapter}" \
  DATAA_DEV_SPLIT_JSON="${DATA_DIR}/dataa_test_detection.json" \
  RUN_ROOT="${WORK_ROOT}/dataa_retention/${name}" \
  PERSIST_ROOT="${PERSIST_ROOT}/dataa_retention/${name}" \
  KEEP_ALIVE_AFTER_RUN=0 \
  bash scripts/camera_detection_retention/run.sh
}

run_vif_retention() {
  local name="$1"
  local adapter="$2"
  adapter_file_check "${adapter}"
  STAGE=all \
  MODEL_PATH="${MODEL_PATH}" \
  ADAPTER_PATH="${adapter}" \
  RUN_ROOT="${WORK_ROOT}/vif_retention/${name}" \
  PERSIST_ROOT="${PERSIST_ROOT}/vif_retention/${name}" \
  KEEP_ALIVE_AFTER_RUN=0 \
  bash scripts/camera_detection_retention/run_vifbench.sh
}

case "${STAGE}" in
  preflight) preflight ;;
  build) build_data ;;
  smoke)
    config="$(render_config train_smoke)"
    FORCE_TORCHRUN="${FORCE_TORCHRUN}" "${LLAMAFACTORY_CLI}" train "${config}"
    ;;
  train_detection_only) train_branch detection_only ;;
  train_correct_camera) train_branch correct_camera ;;
  train_shuffled_camera) train_branch shuffled_camera ;;
  eval_camera_all) infer_and_evaluate_all_camera ;;
  readiness_correct) sample_readiness correct_camera "${CORRECT_ADAPTER}" ;;
  readiness_shuffled) sample_readiness shuffled_camera "${SHUFFLED_ADAPTER}" ;;
  summarize) summarize_gate ;;
  dataa_detection_only) run_dataa_retention detection_only "${DETECTION_ONLY_ADAPTER}" ;;
  dataa_correct_camera) run_dataa_retention correct_camera "${CORRECT_ADAPTER}" ;;
  dataa_shuffled_camera) run_dataa_retention shuffled_camera "${SHUFFLED_ADAPTER}" ;;
  vif_detection_only) run_vif_retention detection_only "${DETECTION_ONLY_ADAPTER}" ;;
  vif_correct_camera) run_vif_retention correct_camera "${CORRECT_ADAPTER}" ;;
  vif_shuffled_camera) run_vif_retention shuffled_camera "${SHUFFLED_ADAPTER}" ;;
  *)
    echo "Unknown STAGE=${STAGE}. See docs/camera_joint_sft_gate_execution_20260713.md" >&2
    exit 2
    ;;
esac

if [[ "${KEEP_ALIVE_AFTER_RUN}" == "1" ]]; then
  require_file "${KEEP_ALIVE_SCRIPT}"
  echo "Stage ${STAGE} completed. Starting ${KEEP_ALIVE_SCRIPT}."
  exec bash "${KEEP_ALIVE_SCRIPT}"
fi
