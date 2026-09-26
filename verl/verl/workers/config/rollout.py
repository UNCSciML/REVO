# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.utils.profiler import ProfilerConfig

__all__ = [
    "SamplingConfig",
    "MultiTurnConfig",
    "CustomAsyncServerConfig",
    "AgentLoopConfig",
    "TraceConfig",
    "ServerConfig",
    "RolloutConfig",
]


@dataclass
class SamplingConfig(BaseConfig):
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    max_tokens: Optional[int] = None
    n: int = 1


@dataclass
class MultiTurnConfig(BaseConfig):
    _mutable_fields = {"max_assistant_turns", "max_user_turns"}

    enable: bool = False
    max_assistant_turns: Optional[int] = None
    tool_config_path: Optional[str] = None
    max_user_turns: Optional[int] = None
    max_parallel_calls: int = 1
    max_tool_response_length: int = 256
    tool_response_truncate_side: str = "middle"
    interaction_config_path: Optional[str] = None
    use_inference_chat_template: bool = False
    tokenization_sanity_check_mode: str = "strict"
    format: str = "hermes"
    num_repeat_rollouts: Optional[int] = None


@dataclass
class CustomAsyncServerConfig(BaseConfig):
    path: Optional[str] = None
    name: Optional[str] = None


@dataclass
class AgentLoopConfig(BaseConfig):
    num_workers: int = 8
    default_agent_loop: str = "single_turn_agent"
    agent_loop_config_path: Optional[str] = None
    custom_async_server: CustomAsyncServerConfig = field(default_factory=CustomAsyncServerConfig)


@dataclass
class TraceConfig(BaseConfig):
    backend: Optional[str] = None
    token2text: bool = False


@dataclass
class ServerConfig(BaseConfig):
    """
    Configuration for SGLang server when running in server mode
    """

    timeout: float = 60.0
    max_attempts: int = 3
    retry_delay: float = 2.0
    max_connections: int = 1000
    max_start_wait_time: float = 300.0


@dataclass
class RolloutConfig(BaseConfig):
    _mutable_fields = {"max_model_len", "load_format"}

    name: Optional[str] = MISSING
    mode: str = "sync"
    skip_tokenizer_init: bool = True

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    do_sample: bool = True
    n: int = 1

    # Early termination threshold for multi-turn rollout in sglang.
    # Abort remaining requests when (1 - over_sample_rate) * total_requests are completed.
    over_sample_rate: float = 0.0

    offpolicy_replay_enable: bool = False
    offpolicy_replay_current_prompt_batch_size: Optional[int] = None
    offpolicy_replay_sample_prompt_batch_size: int = 8
    offpolicy_replay_buffer_size: int = 200
    offpolicy_replay_seed: Optional[int] = None
    offpolicy_replay_skip_hit_budget: bool = False
    teacher_rollout_mix_enable: bool = False
    teacher_rollout_mix_path: Optional[str] = None
    teacher_rollout_mix_num_rollouts: int = 0
    teacher_rollout_mix_index_key: str = "global_index"
    teacher_rollout_mix_cache_index_field: str = "global_index"
    teacher_rollout_mix_selection: str = "first"
    teacher_rollout_mix_seed: Optional[int] = None

    prompt_length: int = 512
    response_length: int = 512

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    ignore_eos: bool = False
    enforce_eager: bool = True
    cudagraph_capture_sizes: Optional[list] = None
    free_cache_engine: bool = True
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 1
    max_num_batched_tokens: int = 8192

    # TODO: enable train_kwargs
    # train_sampling_config: SamplingConfig = field(default_factory=SamplingConfig)

    val_kwargs: SamplingConfig = field(default_factory=SamplingConfig)

    max_model_len: Optional[int] = None
    max_num_seqs: int = 1024

    # note that the logprob computation should belong to the actor
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 16384
    log_prob_temperature: Optional[float] = None
    student_eos_prob_log_enable: bool = False
    opd_advantage_mode: str = "fixed"
    opd_decomposed_prefix_is_mode: str = "cumulative_cap"
    opd_decomposed_prefix_is_min_weight: float = 0.25
    opd_decomposed_prefix_is_max_weight: float = 4.0
    opd_decomposed_proximal_coef: float = 1.0
    opd_q_mixture_enable: bool = False
    opd_q_mixture_teacher_advantage_mode: str = "proposal"
    opd_q_mixture_source_normalize_enable: bool = False
    opd_q_mixture_teacher_loss_lambda: Optional[float | str] = None
    opd_q_mixture_samplek_loss_coef: float = 1.0
    opd_q_mixture_diagnostics_enable: bool = False
    opd_q_mixture_prefix_reference_mode: str = "mixture"
    opd_q_mixture_teacher_prefix_method: str = "prefix"
    opd_q_mixture_teacher_prefix_log_clip: Optional[float | str] = None
    opd_q_mixture_teacher_prefix_log_clip_mode: Optional[str] = None
    opd_q_mixture_teacher_prefix_min_weight: Optional[float | str] = None
    opd_samplek_candidate_aggregation: str = "sum"
    opd_samplek_advantage_centering: str = "none"
    opd_samplek_loo_variance_filter_threshold: Optional[float | str] = None
    opd_samplek_loo_variance_filter_threshold_mode: str = "fixed"
    opd_samplek_loo_variance_filter_quantile: float = 0.7
    opd_samplek_loo_variance_filter_selection: str = "high"
    opd_samplek_loo_variance_filter_mode: str = "hard"
    opd_samplek_loo_variance_filter_soft_base_weight: float = 0.5
    opd_samplek_loo_variance_filter_soft_active_bonus: float = 1.0
    opd_samplek_loo_variance_filter_expectile_tau: float = 0.75
    opd_samplek_loo_variance_ratio_diagnostics_enable: bool = False
    opd_samplek_loo_variance_histogram_diagnostics_enable: bool = False
    opd_samplek_influence_clip: Optional[float | str] = None
    opd_teacher_deficit_residual_enable: bool = False
    opd_teacher_deficit_residual_k: int = 8
    opd_teacher_deficit_residual_coef: float = 0.0
    opd_diagnostic_forced_eos_enable: bool = False
    opd_terminal_aware_enable: bool = False
    opd_terminal_objective_mode: str = "anchor_kl"
    opd_terminal_anchor_mode: str = "behavior"
    opd_terminal_gate_coef: float = 1.0
    opd_terminal_secondary_token_id: Optional[int] = None
    opd_terminal_teacher_remap_enable: bool = False
    opd_terminal_kl_baseline_mode: str = "mc_loo"
    opd_terminal_teacher_remap_floor: float = 1e-18
    opd_terminal_topm: int = 0
    opd_samplek_eos_negative_relu_enable: bool = False
    opd_eos_future_enable: bool = False
    opd_eos_future_mode: str = "fixed_signed"
    opd_eos_future_horizon: int = 32
    opd_eos_future_coef: float = 1.0
    opd_sampled_token_eos_alignment_enable: bool = False
    opd_sampled_token_teacher_eos_token_id: Optional[int] = None
    opd_raw_advantage_clip: Optional[float | str] = None
    opd_samplek_entropy_coef: float = 0.0
    opd_sampled_token_proximal_coef: float = 0.0
    opd_sampled_token_proximal_mode: str = "reverse_kl"
    opd_sampled_token_max_entropy_coef: float = 0.0
    opd_sampled_token_proximal_prefix_weight_enable: bool = False
    opd_samplek_total_grad_norm: Optional[float | str] = None
    opd_dapo_format_penalty_enable: bool = False
    opd_dapo_format_penalty_coef: Optional[float | str] = 0.0
    opd_dapo_format_penalty_tail_tokens: int = 512
    adaptive_ppo_update_enable: bool = False
    adaptive_ppo_update_min_updates: int = 1
    adaptive_ppo_update_max_updates: int = 8
    adaptive_ppo_update_abs_log_ratio_threshold: Optional[float | str] = None
    adaptive_ppo_update_sampled_token_rkl_k3_threshold: Optional[float | str] = None
    adaptive_ppo_update_no_candidate_is: bool = True
    samplek_candidate_reuse_is_enable: bool = False
    samplek_candidate_refresh_interval: int = 5
    adaptive_ppo_update_samplek_probe_enable: bool = False
    prefix_drift_enable: bool = False
    prefix_drift_method: str = "diagnostic"
    prefix_drift_ess_target_fraction: float = 0.5
    prefix_drift_ess_bisection_steps: int = 8
    prefix_drift_log_clip: Optional[float | str] = 3.0
    prefix_drift_log_clip_mode: str = "symmetric"
    prefix_drift_ctpo_log_clip_base: Optional[float | str] = None
    prefix_drift_ctpo_log_clip_lower_base: Optional[float | str] = None
    prefix_drift_ctpo_log_clip_upper_base: Optional[float | str] = None
    prefix_drift_ctpo_log_clip_power: float = 0.5
    prefix_drift_normalize: str = "none"
    prefix_drift_position_beta: float = 0.0
    prefix_drift_position_hmax: Optional[int | str] = None
    prefix_drift_detach: bool = True
    ess_samplek_resample_enable: bool = False
    ess_samplek_resample_max_updates: int = 4
    ess_samplek_resample_mean_threshold: float = 0.5
    log_prob_top_k: int = 256
    log_prob_candidate_mode: str = "topk"
    sample_k_replacement: bool = True
    adaptive_head_tail_gamma: float = 0.5
    adaptive_head_tail_k2_min: int = 1
    adaptive_head_tail_negative_elu_enable: bool = False
    adaptive_head_tail_negative_elu_threshold: float = -1.0
    adaptive_head_tail_negative_elu_tau: float = 1.0
    sample_k_kl_plus_one: bool = True
    opd_loss_type: str = "sample_k_reverse_kl"
    chi_square_baseline: str = "mean"
    on_logprob_mse_clip: Optional[float | str] = None
    on_logprob_mse_center: bool = False
    on_logprob_mse_normalize: bool = False
    top_k_strategy: str = "only_stu"
    reward_weight_mode: str = "student_p"
    reward_weight_normalize: Optional[bool | str] = None
    teacher_temperature: float = 1.0
    teacher_temperature_anneal_enable: bool = False
    teacher_temperature_min: float = 1.0
    teacher_temperature_anneal_steps: int = 50
    teacher_temperature_anneal_schedule: str = "linear"

    disable_log_stats: bool = True

    multi_stage_wake_up: bool = False
    engine_kwargs: dict = field(default_factory=dict)

    calculate_log_probs: bool = False

    agent: AgentLoopConfig = field(default_factory=AgentLoopConfig)

    trace: TraceConfig = field(default_factory=TraceConfig)

    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)

    # Server configuration for sglang server mode
    server: ServerConfig = field(default_factory=ServerConfig)

    update_weights_bucket_megabytes: int = 512

    skip_rollout: bool = False

    skip_dump_dir: str = "/tmp/rollout_dump"

    profiler: Optional[ProfilerConfig] = None

    enable_chunked_prefill: bool = True

    enable_prefix_caching: bool = True

    load_format: str = "dummy"

    layered_summon: bool = False

    layer_name_map: dict = field(default_factory=dict)

    sglang_engine_mode: str = "local"

    limit_images: Optional[int] = None

    skip_tokenizer_init: bool = False

    def __post_init__(self):
        """Validate the rollout config"""
        if self.expert_parallel_size > 1:
            assert self.expert_parallel_size == (self.tensor_model_parallel_size * self.data_parallel_size), (
                "expert_parallel_size must be equal to tensor_model_parallel_size * data_parallel_size"
            )

        if self.pipeline_model_parallel_size > 1:
            if self.name == "vllm" or self.name == "sglang":
                raise NotImplementedError(
                    f"Current rollout {self.name=} not implemented pipeline_model_parallel_size > 1 yet."
                )
