#!/usr/bin/env bash
# REVO training for Qwen3-1.7B-Base with a Qwen3-4B teacher.
# Default: 4 GPUs, 50 training steps.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-1.7B-Base}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-4B}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
N_GPUS=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_qwen3_1.7b_base_qwen3_4b}

export PYTHONPATH="$ROOT/verl:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true
export TTRL_MATH_VERIFY_WALL_TIMEOUT_SECONDS=2.0
export RAY_memory_usage_threshold=0.99
export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=${WANDB_MODE:-offline}
fi

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=token_reward_direct \
    data.shuffle=False \
    data.train_files=datasets/dapo-math-17k-ttrl.parquet \
    'data.val_files=[datasets/test_data/AIME25/test.parquet, datasets/test_data/AMC23/test.parquet, datasets/test_data/AIME24/test.parquet]' \
    data.train_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=7168 \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$STUDENT_MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.opd_current_samplek_no_candidate_is=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    +actor_rollout_ref.rollout.log_prob_temperature=1.0 \
    actor_rollout_ref.rollout.student_eos_prob_log_enable=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=7168 \
    +actor_rollout_ref.rollout.opd_loss_type=sample_k_reverse_kl \
    +actor_rollout_ref.rollout.log_prob_candidate_mode=sample_stu \
    +actor_rollout_ref.rollout.log_prob_top_k=16 \
    +actor_rollout_ref.rollout.sample_k_kl_plus_one=False \
    +actor_rollout_ref.rollout.reward_weight_mode=none \
    +actor_rollout_ref.rollout.reward_weight_normalize=True \
    actor_rollout_ref.rollout.opd_advantage_mode=current_kl_is \
    actor_rollout_ref.rollout.opd_samplek_candidate_aggregation=mean \
    actor_rollout_ref.rollout.opd_samplek_advantage_centering=leave_one_out \
    actor_rollout_ref.rollout.opd_samplek_loo_variance_filter_threshold=0.005 \
    actor_rollout_ref.rollout.opd_samplek_loo_variance_filter_mode=expectile \
    actor_rollout_ref.rollout.opd_samplek_loo_variance_filter_expectile_tau=0.75 \
    actor_rollout_ref.rollout.opd_terminal_aware_enable=True \
    actor_rollout_ref.rollout.opd_terminal_objective_mode=teacher_remap_only \
    actor_rollout_ref.rollout.opd_terminal_gate_coef=0.0 \
    actor_rollout_ref.rollout.opd_terminal_secondary_token_id=151645 \
    actor_rollout_ref.rollout.opd_terminal_teacher_remap_enable=True \
    actor_rollout_ref.rollout.adaptive_ppo_update_enable=True \
    actor_rollout_ref.rollout.adaptive_ppo_update_max_updates=10 \
    actor_rollout_ref.rollout.prefix_drift_enable=True \
    actor_rollout_ref.rollout.prefix_drift_method=prefix_geometric \
    actor_rollout_ref.rollout.prefix_drift_log_clip=1.3862943611198906 \
    actor_rollout_ref.rollout.prefix_drift_log_clip_mode=upper \
    reward_model.enable=True \
    reward_model.model.path="$TEACHER_MODEL" \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    +reward_model.model.dtype=fp32 \
    +reward_model.reward_kwargs.enable_format_reward=False \
    reward_model.use_dynamic_bsz=False \
    reward_model.micro_batch_size_per_gpu=24 \
    custom_reward_function.path=verl/verl/utils/reward_score/ttrl_math/__init__.py \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.log_val_generations=2 \
    trainer.project_name=OnPolicyDistillation \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.validation_data_dir="validation_log/$EXPERIMENT_NAME" \
    trainer.default_local_dir="checkpoints/$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=50 \
    trainer.save_freq=10 \
    "$@"
