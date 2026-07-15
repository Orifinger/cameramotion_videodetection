#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${STAGE:-preflight}"
PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
MODEL_PATH="${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}"
WORK_ROOT="${WORK_ROOT:-/tmp/1res/camera_hard_route_gate/v1}"
DATA_DIR="${WORK_ROOT}/data"
TRAIN_ROOT="${WORK_ROOT}/train"
ROUTE_ROOT="${WORK_ROOT}/routes"
VIF_ROOT="${WORK_ROOT}/vifbench"
PERSIST_ROOT="${PERSIST_ROOT:-${PROJECT_ROOT}/res/camera_hard_route_gate/v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
CHECK_IMAGES="${CHECK_IMAGES:-1}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"
FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
KEEP_ALIVE_AFTER_RUN="${KEEP_ALIVE_AFTER_RUN:-0}"
KEEP_ALIVE_SCRIPT="${KEEP_ALIVE_SCRIPT:-/input/training/keep.sh}"

DATAA_DETECTION_JSON="${DATAA_DETECTION_JSON:-${PROJECT_ROOT}/res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json}"
DATAA_CAMERA_JSONL="${DATAA_CAMERA_JSONL:-${PROJECT_ROOT}/camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl}"
DATAB_DETECTION_JSON="${DATAB_DETECTION_JSON:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json}"
DATAB_CAMERA_JSONL="${DATAB_CAMERA_JSONL:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl}"

V4TRAIN_EVAL_DIR="${V4TRAIN_EVAL_DIR:-${PROJECT_ROOT}/eval/v4train-main/eval}"
OFFICIAL_EVAL_PY="${OFFICIAL_EVAL_PY:-${V4TRAIN_EVAL_DIR}/eval.py}"
if [[ -z "${INDEX_DIR:-}" ]]; then
  INDEX_DIR="${V4TRAIN_EVAL_DIR}/test_index_splits/splits_16"
  alternate_index="$(dirname "${V4TRAIN_EVAL_DIR}")/test_index_splits/splits_16"
  if [[ ! -d "${INDEX_DIR}" && -d "${alternate_index}" ]]; then
    INDEX_DIR="${alternate_index}"
  fi
fi

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

SHARED_ADAPTER="${SHARED_ADAPTER:-${TRAIN_ROOT}/shared}"
NO_MOTION_ADAPTER="${NO_MOTION_ADAPTER:-${TRAIN_ROOT}/no_motion}"
MINOR_MOTION_ADAPTER="${MINOR_MOTION_ADAPTER:-${TRAIN_ROOT}/minor_motion}"
COMPLEX_MOTION_ADAPTER="${COMPLEX_MOTION_ADAPTER:-${TRAIN_ROOT}/complex_motion}"
CAMERA_ROUTE_ADAPTER="${CAMERA_ROUTE_ADAPTER:-${TRAIN_ROOT}/router}"
RUNTIME_CONFIG_DIR="${WORK_ROOT}/configs"

DATAA_ROUTE_INPUT="${DATA_DIR}/dataa_route_dev_questions.jsonl"
DATAA_ROUTE_PRED_DIR="${ROUTE_ROOT}/dataa_scores"
DATAA_ROUTE_MANIFEST="${ROUTE_ROOT}/dataa_route_manifest.jsonl"
DATAA_ROUTE_SUMMARY="${ROUTE_ROOT}/dataa_route_summary.json"
VIF_ROUTE_INPUT="${ROUTE_ROOT}/vifbench_route_questions.jsonl"
VIF_ROUTE_INPUT_SUMMARY="${ROUTE_ROOT}/vifbench_route_input_summary.json"
VIF_ROUTE_PRED_DIR="${ROUTE_ROOT}/vifbench_scores"
VIF_ROUTE_MANIFEST="${ROUTE_ROOT}/vifbench_route_manifest.jsonl"
VIF_ROUTE_SUMMARY="${ROUTE_ROOT}/vifbench_route_summary.json"
MIN_ROUTE_PROBABILITY="${MIN_ROUTE_PROBABILITY:-0.0}"
MIN_ROUTE_MARGIN="${MIN_ROUTE_MARGIN:-0.0}"

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

adapter_check() {
  require_dir "$1"
  require_file "$1/adapter_config.json"
  if [[ ! -f "$1/adapter_model.safetensors" && ! -f "$1/adapter_model.bin" ]]; then
    echo "Missing adapter weights under $1" >&2
    exit 2
  fi
}

persist_small_results() {
  mkdir -p "${PERSIST_ROOT}/data" "${PERSIST_ROOT}/routes" "${PERSIST_ROOT}/eval"
  for file in \
    "${DATA_DIR}/camera_hard_route_data_summary.json" \
    "${DATA_DIR}/dataa_hard_route_split_manifest.jsonl" \
    "${DATA_DIR}/dataa_route_dev_gold.jsonl" \
    "${DATA_DIR}/llamafactory_install_summary.json"
  do
    if [[ -f "${file}" ]]; then
      cp -a "${file}" "${PERSIST_ROOT}/data/"
    fi
  done
  for file in \
    "${DATAA_ROUTE_MANIFEST}" \
    "${DATAA_ROUTE_SUMMARY}" \
    "${VIF_ROUTE_INPUT_SUMMARY}" \
    "${VIF_ROUTE_MANIFEST}" \
    "${VIF_ROUTE_SUMMARY}"
  do
    if [[ -f "${file}" ]]; then
      cp -a "${file}" "${PERSIST_ROOT}/routes/"
    fi
  done
  if [[ -d "${VIF_ROOT}/composed" ]]; then
    find "${VIF_ROOT}/composed" -maxdepth 1 -type f \
      \( -name '*summary.json' -o -name '*gate.json' -o -name '*.csv' -o -name '*.log' \) \
      -exec cp -a {} "${PERSIST_ROOT}/eval/" \;
  fi
}

preflight() {
  require_dir "${MODEL_PATH}"
  require_file "${MODEL_PATH}/config.json"
  require_file "${DATAA_DETECTION_JSON}"
  require_file "${DATAA_CAMERA_JSONL}"
  require_file "${DATAB_DETECTION_JSON}"
  require_file "${DATAB_CAMERA_JSONL}"
  require_file "${LLAMAFACTORY_ROOT}/examples/deepspeed/ds_z2_config.json"
  require_file "${LLAMAFACTORY_DATA_DIR}/dataset_info.json"
  require_file "${OFFICIAL_EVAL_PY}"
  require_dir "${INDEX_DIR}"
  command -v "${LLAMAFACTORY_CLI}" >/dev/null
  "${PYTHON_BIN}" -c 'import peft, qwen_vl_utils, sklearn, torch, transformers, yaml; print("Python dependencies: OK"); print("transformers:", transformers.__version__); print("gpus:", torch.cuda.device_count())'
  echo "Preflight passed. No model loading, training, or inference was run."
}

build_data() {
  mkdir -p "${DATA_DIR}"
  image_args=()
  if [[ "${CHECK_IMAGES}" == "1" ]]; then
    image_args+=(--check-images)
  fi
  "${PYTHON_BIN}" tools/build_camera_hard_route_gate.py \
    --dataa-detection-json "${DATAA_DETECTION_JSON}" \
    --dataa-camera-jsonl "${DATAA_CAMERA_JSONL}" \
    --datab-detection-json "${DATAB_DETECTION_JSON}" \
    --datab-camera-jsonl "${DATAB_CAMERA_JSONL}" \
    --out-dir "${DATA_DIR}" \
    --test-ratio 0.30 \
    --expected-dataa-cases 1080 \
    --seed 20260715 \
    "${image_args[@]}"
  "${PYTHON_BIN}" tools/install_camera_hard_route_gate.py \
    --source-dir "${DATA_DIR}" \
    --llamafactory-data-dir "${LLAMAFACTORY_DATA_DIR}" \
    --smoke-samples 96 \
    --seed 20260715 \
    "${image_args[@]}"
  persist_small_results
}

branch_spec() {
  case "$1" in
    shared) echo "camera_hard_route_shared|${SHARED_ADAPTER}" ;;
    no_motion) echo "camera_hard_route_no_motion|${NO_MOTION_ADAPTER}" ;;
    minor_motion) echo "camera_hard_route_minor_motion|${MINOR_MOTION_ADAPTER}" ;;
    complex_motion) echo "camera_hard_route_complex_motion|${COMPLEX_MOTION_ADAPTER}" ;;
    *) echo "Unknown training branch: $1" >&2; exit 2 ;;
  esac
}

render_config() {
  local branch="$1"
  local template="configs/camera_hard_route_gate/train_template.yaml"
  local dataset_name output_dir spec destination
  spec="$(branch_spec "${branch}")"
  dataset_name="${spec%%|*}"
  output_dir="${spec#*|}"
  destination="${RUNTIME_CONFIG_DIR}/train_${branch}.yaml"
  require_file "${template}"
  mkdir -p "${RUNTIME_CONFIG_DIR}"
  sed \
    -e "s|__MODEL_PATH__|${MODEL_PATH}|g" \
    -e "s|__DEEPSPEED_CONFIG__|${LLAMAFACTORY_ROOT}/examples/deepspeed/ds_z2_config.json|g" \
    -e "s|__DATASET_DIR__|${LLAMAFACTORY_DATA_DIR}|g" \
    -e "s|__DATASET_NAME__|${dataset_name}|g" \
    -e "s|__OUTPUT_DIR__|${output_dir}|g" \
    "${template}" > "${destination}"
  echo "${destination}"
}

train_branch() {
  local branch="$1"
  local config
  config="$(render_config "${branch}")"
  FORCE_TORCHRUN="${FORCE_TORCHRUN}" "${LLAMAFACTORY_CLI}" train "${config}"
}

train_router() {
  local template="configs/camera_hard_route_gate/train_router.yaml"
  local destination="${RUNTIME_CONFIG_DIR}/train_router.yaml"
  require_file "${template}"
  mkdir -p "${RUNTIME_CONFIG_DIR}"
  sed \
    -e "s|__MODEL_PATH__|${MODEL_PATH}|g" \
    -e "s|__DEEPSPEED_CONFIG__|${LLAMAFACTORY_ROOT}/examples/deepspeed/ds_z2_config.json|g" \
    -e "s|__DATASET_DIR__|${LLAMAFACTORY_DATA_DIR}|g" \
    -e "s|__OUTPUT_DIR__|${CAMERA_ROUTE_ADAPTER}|g" \
    "${template}" > "${destination}"
  FORCE_TORCHRUN="${FORCE_TORCHRUN}" "${LLAMAFACTORY_CLI}" train "${destination}"
}

smoke_train() {
  local template="configs/camera_hard_route_gate/train_smoke.yaml"
  local destination="${RUNTIME_CONFIG_DIR}/train_smoke.yaml"
  require_file "${template}"
  mkdir -p "${RUNTIME_CONFIG_DIR}"
  sed \
    -e "s|__MODEL_PATH__|${MODEL_PATH}|g" \
    -e "s|__DEEPSPEED_CONFIG__|${LLAMAFACTORY_ROOT}/examples/deepspeed/ds_z2_config.json|g" \
    -e "s|__DATASET_DIR__|${LLAMAFACTORY_DATA_DIR}|g" \
    -e "s|__OUTPUT_DIR__|${WORK_ROOT}/smoke|g" \
    "${template}" > "${destination}"
  FORCE_TORCHRUN="${FORCE_TORCHRUN}" "${LLAMAFACTORY_CLI}" train "${destination}"
}

score_routes() {
  local input_jsonl="$1"
  local output_dir="$2"
  local model_stage="$3"
  adapter_check "${CAMERA_ROUTE_ADAPTER}"
  require_file "${input_jsonl}"
  rm -rf "${output_dir}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    -m scripts.camera_joint_sft_gate.score_binary \
    --model-path "${MODEL_PATH}" \
    --adapter-path "${CAMERA_ROUTE_ADAPTER}" \
    --condition "route=${input_jsonl}" \
    --output-dir "${output_dir}" \
    --model-stage "${model_stage}" \
    --max-pixels "${MAX_PIXELS}" \
    --seed 20260715
}

score_dataa_routes() {
  score_routes "${DATAA_ROUTE_INPUT}" "${DATAA_ROUTE_PRED_DIR}" dataa_three_class_route
}

aggregate_dataa_routes() {
  "${PYTHON_BIN}" -m scripts.camera_hard_route_gate.route_manifest aggregate \
    --input-jsonl "${DATAA_ROUTE_INPUT}" \
    --prediction-dir "${DATAA_ROUTE_PRED_DIR}" \
    --output-manifest "${DATAA_ROUTE_MANIFEST}" \
    --summary-json "${DATAA_ROUTE_SUMMARY}" \
    --min-top-probability "${MIN_ROUTE_PROBABILITY}" \
    --min-margin "${MIN_ROUTE_MARGIN}"
  persist_small_results
}

dataa_route_calibration() {
  score_dataa_routes
  aggregate_dataa_routes
}

build_vif_route_inputs() {
  mkdir -p "${ROUTE_ROOT}"
  "${PYTHON_BIN}" -m scripts.camera_hard_route_gate.route_manifest build-vif-inputs \
    --index-dir "${INDEX_DIR}" \
    --output-jsonl "${VIF_ROUTE_INPUT}" \
    --summary-json "${VIF_ROUTE_INPUT_SUMMARY}" \
    --expected-ranks 16 \
    --expected-frames 16 \
    --check-frame-dirs \
    --require-timestamps
  persist_small_results
}

score_vif_routes() {
  score_routes "${VIF_ROUTE_INPUT}" "${VIF_ROUTE_PRED_DIR}" vifbench_three_class_route
}

aggregate_vif_routes() {
  "${PYTHON_BIN}" -m scripts.camera_hard_route_gate.route_manifest aggregate \
    --input-jsonl "${VIF_ROUTE_INPUT}" \
    --prediction-dir "${VIF_ROUTE_PRED_DIR}" \
    --output-manifest "${VIF_ROUTE_MANIFEST}" \
    --summary-json "${VIF_ROUTE_SUMMARY}" \
    --min-top-probability "${MIN_ROUTE_PROBABILITY}" \
    --min-margin "${MIN_ROUTE_MARGIN}"
  persist_small_results
}

run_vif_branch() {
  local name="$1"
  local adapter="$2"
  adapter_check "${adapter}"
  local branch_root="${VIF_ROOT}/${name}"
  if [[ "${name}" == "shared" ]]; then
    STAGE=all \
    MODEL_PATH="${MODEL_PATH}" \
    ADAPTER_PATH="${adapter}" \
    RUN_ROOT="${branch_root}" \
    PERSIST_ROOT="${PERSIST_ROOT}/vif_branches/${name}" \
    INDEX_DIR="${INDEX_DIR}" \
    CAMERA_MODEL_NAME="Qwen3-VL-8B-hard-route-${name}" \
    PARALLEL_MODELS=1 \
    KEEP_ALIVE_AFTER_RUN=0 \
    bash scripts/camera_detection_retention/run_vifbench.sh
  else
    STAGE=adapter_only \
    MODEL_PATH="${MODEL_PATH}" \
    ADAPTER_PATH="${adapter}" \
    RUN_ROOT="${branch_root}" \
    PERSIST_ROOT="${PERSIST_ROOT}/vif_branches/${name}" \
    INDEX_DIR="${INDEX_DIR}" \
    CAMERA_MODEL_NAME="Qwen3-VL-8B-hard-route-${name}" \
    KEEP_ALIVE_AFTER_RUN=0 \
    bash scripts/camera_detection_retention/run_vifbench.sh
  fi
}

compose_vif() {
  require_file "${VIF_ROUTE_MANIFEST}"
  local shared_predictions="${VIF_ROOT}/shared/inference/camera_adapter/splitresults"
  local no_predictions="${VIF_ROOT}/no_motion/inference/camera_adapter/splitresults"
  local minor_predictions="${VIF_ROOT}/minor_motion/inference/camera_adapter/splitresults"
  local complex_predictions="${VIF_ROOT}/complex_motion/inference/camera_adapter/splitresults"
  require_dir "${shared_predictions}"
  require_dir "${no_predictions}"
  require_dir "${minor_predictions}"
  require_dir "${complex_predictions}"
  local output_dir="${VIF_ROOT}/composed"
  mkdir -p "${output_dir}"
  for mode in shared predicted cyclic; do
    "${PYTHON_BIN}" -m scripts.camera_hard_route_gate.route_manifest compose \
      --index-dir "${INDEX_DIR}" \
      --route-manifest "${VIF_ROUTE_MANIFEST}" \
      --expert "no-motion=${no_predictions}" \
      --expert "minor-motion=${minor_predictions}" \
      --expert "complex-motion=${complex_predictions}" \
      --shared-prediction-dir "${shared_predictions}" \
      --route-mode "${mode}" \
      --output-predictions "${output_dir}/${mode}_predictions.json" \
      --output-summary "${output_dir}/${mode}_summary.json" \
      --expected-ranks 16
    "${PYTHON_BIN}" "${OFFICIAL_EVAL_PY}" \
      --json_file_path "${output_dir}/${mode}_predictions.json" \
      | tee "${output_dir}/${mode}_official_eval.log"
  done
  "${PYTHON_BIN}" -m scripts.camera_hard_route_gate.route_manifest summarize \
    --base-eval "${VIF_ROOT}/shared/eval/base_vifbench_eval.json" \
    --shared-summary "${output_dir}/shared_summary.json" \
    --predicted-summary "${output_dir}/predicted_summary.json" \
    --cyclic-summary "${output_dir}/cyclic_summary.json" \
    --output-json "${output_dir}/camera_hard_route_gate.json"
  persist_small_results
}

echo "=== 三分类相机运动硬路由检测专家验证 ==="
echo "stage=${STAGE}"
echo "base_model=${MODEL_PATH}"
echo "work_root=${WORK_ROOT}"
echo "camera_text_at_detection_inference=false"

case "${STAGE}" in
  preflight) preflight ;;
  build) build_data ;;
  smoke) smoke_train ;;
  train_router) train_router ;;
  train_shared) train_branch shared ;;
  train_no_motion) train_branch no_motion ;;
  train_minor_motion) train_branch minor_motion ;;
  train_complex_motion) train_branch complex_motion ;;
  train_experts)
    train_branch no_motion
    train_branch minor_motion
    train_branch complex_motion
    ;;
  train_all)
    train_branch shared
    train_branch no_motion
    train_branch minor_motion
    train_branch complex_motion
    ;;
  score_dataa_route) score_dataa_routes ;;
  aggregate_dataa_route) aggregate_dataa_routes ;;
  calibrate_dataa_route) dataa_route_calibration ;;
  build_vif_route_inputs) build_vif_route_inputs ;;
  score_vif_route) score_vif_routes ;;
  aggregate_vif_route) aggregate_vif_routes ;;
  vif_shared) run_vif_branch shared "${SHARED_ADAPTER}" ;;
  vif_no_motion) run_vif_branch no_motion "${NO_MOTION_ADAPTER}" ;;
  vif_minor_motion) run_vif_branch minor_motion "${MINOR_MOTION_ADAPTER}" ;;
  vif_complex_motion) run_vif_branch complex_motion "${COMPLEX_MOTION_ADAPTER}" ;;
  vif_experts)
    run_vif_branch no_motion "${NO_MOTION_ADAPTER}"
    run_vif_branch minor_motion "${MINOR_MOTION_ADAPTER}"
    run_vif_branch complex_motion "${COMPLEX_MOTION_ADAPTER}"
    ;;
  compose_vif) compose_vif ;;
  *)
    echo "Unknown STAGE=${STAGE}" >&2
    exit 2
    ;;
esac

if [[ "${KEEP_ALIVE_AFTER_RUN}" == "1" ]]; then
  require_file "${KEEP_ALIVE_SCRIPT}"
  echo "Stage ${STAGE} completed. Starting ${KEEP_ALIVE_SCRIPT}."
  exec bash "${KEEP_ALIVE_SCRIPT}"
fi
