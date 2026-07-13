#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}
MODEL_PATH=${MODEL_PATH:-/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115}
DATAB_JSON=${DATAB_JSON:-/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json}
VERL_ROOT=${VERL_ROOT:-${ROOT}/third_party/verl-2c9e19ef2f0619a2e9e9d4fc813dab8e717e3ab9}
DATA_DIR=${DATA_DIR:-${ROOT}/res/skyra_grpo_diagnostics/data}
REWARD_FILE=${REWARD_FILE:-${ROOT}/scripts/skyra_grpo_diagnostics/reward.py}
STAGE=${STAGE:-prepare}
RUN_MODE=${RUN_MODE:-smoke}
REWARD_VARIANT=${REWARD_VARIANT:-paper_asymmetric_inspection}
SEED=${SEED:-20260714}

if [[ "${RUN_MODE}" == "smoke" ]]; then
  TOTAL_STEPS=${TOTAL_STEPS:-1}
  GROUP_SIZE=${GROUP_SIZE:-4}
  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
  MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
else
  TOTAL_STEPS=${TOTAL_STEPS:-40}
  GROUP_SIZE=${GROUP_SIZE:-8}
  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
  MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-768}
fi

TRAJECTORY_BATCH_SIZE=$((TRAIN_BATCH_SIZE * GROUP_SIZE))
RUN_NAME=${RUN_NAME:-${REWARD_VARIANT}_${RUN_MODE}_${TOTAL_STEPS}step}
RUN_ROOT=${RUN_ROOT:-/tmp/1res/skyra_grpo_diagnostics/${RUN_NAME}}
PERSIST_ROOT=${PERSIST_ROOT:-${ROOT}/res/skyra_grpo_diagnostics/${RUN_NAME}}
TRAIN_PARQUET=${DATA_DIR}/datab_grpo_train.parquet
VAL_PARQUET=${DATA_DIR}/datab_grpo_validation.parquet

mkdir -p "${DATA_DIR}" "${RUN_ROOT}" "${PERSIST_ROOT}"

build_data() {
  python3 "${ROOT}/scripts/skyra_grpo_diagnostics/build_datab_verl.py" \
    --input-json "${DATAB_JSON}" \
    --output-dir "${DATA_DIR}" \
    --validation-per-class 256 \
    --seed "${SEED}"
}

patch_verl() {
  python3 "${ROOT}/scripts/skyra_grpo_diagnostics/patch_verl.py" \
    --verl-root "${VERL_ROOT}" \
    --output-json "${PERSIST_ROOT}/verl_patch_audit.json"
}

reward_tests() {
  cd "${ROOT}"
  python3 -m unittest discover -s tests -p 'test_skyra_grpo_*.py'
}

dataset_preflight() {
  python3 "${ROOT}/scripts/skyra_grpo_diagnostics/patch_verl.py" \
    --verl-root "${VERL_ROOT}" --check
  python3 "${ROOT}/scripts/skyra_grpo_diagnostics/preflight_verl_dataset.py" \
    --verl-root "${VERL_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --train-parquet "${TRAIN_PARQUET}" \
    --output-json "${PERSIST_ROOT}/verl_dataset_preflight.json"
}

summarize() {
  python3 "${ROOT}/scripts/skyra_grpo_diagnostics/summarize_rollouts.py" \
    --rollout-dir "${RUN_ROOT}/rollouts" \
    --output-dir "${PERSIST_ROOT}" \
    --reward-variant "${REWARD_VARIANT}"
}

train() {
  export PYTHONPATH="${VERL_ROOT}:${ROOT}:${PYTHONPATH:-}"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export VLLM_USE_V1=1
  export VLLM_ALLREDUCE_USE_SYMM_MEM=0
  export TOKENIZERS_PARALLELISM=false
  export TENSORBOARD_DIR="${PERSIST_ROOT}/tensorboard"
  export RAY_TMPDIR="${RUN_ROOT}/ray"
  mkdir -p "${TENSORBOARD_DIR}" "${RUN_ROOT}/rollouts" "${RUN_ROOT}/checkpoints" "${RAY_TMPDIR}"

  printf '%s\n' \
    "run_name=${RUN_NAME}" \
    "run_mode=${RUN_MODE}" \
    "reward_variant=${REWARD_VARIANT}" \
    "model_path=${MODEL_PATH}" \
    "train_parquet=${TRAIN_PARQUET}" \
    "total_steps=${TOTAL_STEPS}" \
    "train_batch_size=${TRAIN_BATCH_SIZE}" \
    "group_size=${GROUP_SIZE}" \
    "trajectory_batch_size=${TRAJECTORY_BATCH_SIZE}" \
    "seed=${SEED}" > "${PERSIST_ROOT}/run_manifest.txt"

  set +e
  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size=32 \
    data.max_prompt_length=12288 \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.image_key=images \
    data.shuffle=True \
    data.seed="${SEED}" \
    data.dataloader_num_workers=8 \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAJECTORY_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=6144 \
    actor_rollout_ref.rollout.max_model_len=13056 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    +actor_rollout_ref.rollout.limit_images=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.n="${GROUP_SIZE}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True' \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs={min_pixels:3136,max_pixels:262144}' \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward_model.reward_manager=naive \
    reward_model.launch_reward_fn_async=False \
    custom_reward_function.path="${REWARD_FILE}" \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.reward_variant="${REWARD_VARIANT}" \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=skyra_grpo_diagnostics \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=1 \
    trainer.balance_batch=True \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.rollout_data_dir="${RUN_ROOT}/rollouts" \
    trainer.default_local_dir="${RUN_ROOT}/checkpoints" \
    2>&1 | tee "${PERSIST_ROOT}/train.log"
  status=${PIPESTATUS[0]}
  set -e

  echo "${status}" > "${PERSIST_ROOT}/exit_status.txt"
  if compgen -G "${RUN_ROOT}/rollouts/*.jsonl" > /dev/null; then
    summarize || true
  fi
  if [[ "${KEEP_ALIVE_AFTER_RUN:-0}" == "1" ]]; then
    bash /input/training/keep.sh
  fi
  return "${status}"
}

case "${STAGE}" in
  build_data) build_data ;;
  patch_verl) patch_verl ;;
  reward_tests) reward_tests ;;
  preflight) dataset_preflight ;;
  prepare) build_data; patch_verl; reward_tests; dataset_preflight ;;
  train) train ;;
  summarize) summarize ;;
  all) build_data; patch_verl; reward_tests; dataset_preflight; train ;;
  *) echo "Unknown STAGE=${STAGE}" >&2; exit 2 ;;
esac
