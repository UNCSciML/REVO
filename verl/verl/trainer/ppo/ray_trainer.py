# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import math
import os
import uuid
from collections import defaultdict, deque
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.adaptive_update import (
    compute_samplek_probe_diagnostics,
    is_samplek_candidate_refresh_update,
    validate_samplek_candidate_reuse_configuration,
)
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.eos_future_correction import (
    EOS_FUTURE_MODE_DYNAMIC_H1_OFFPOLICY_CLIPPED,
    EOS_FUTURE_MODE_DYNAMIC_H1_REMAINING_HORIZON_RELU,
    EOS_FUTURE_MODE_DYNAMIC_H1_SQRT_REMAINING_HORIZON_RELU,
    EOS_FUTURE_MODE_FIXED_SIGNED,
    OPD_EOS_FUTURE_CORRECTION_MASK_KEY,
    OPD_EOS_FUTURE_VALUE_KEY,
    build_fixed_eos_future_values,
    normalize_eos_future_mode,
    validate_eos_future_correction_configuration,
)
from verl.trainer.ppo.forced_eos_diagnostic import (
    FORCED_EOS_ESTIMATOR_WEIGHTS_KEY,
    validate_forced_eos_diagnostic_configuration,
)
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.opd_decomposed import (
    OPD_Q_MIXTURE_ALPHA_KEY,
    OPD_Q_MIXTURE_IS_TEACHER_KEY,
    OPD_Q_MIXTURE_PRIOR_ALPHA_KEY,
    OPD_Q_MIXTURE_SOURCE_WEIGHTS_KEY,
    OPD_Q_MIXTURE_TEACHER_LOG_PROBS_KEY,
    OPD_ROLLOUT_REFERENCE_LOG_PROBS_KEY,
    apply_samplek_advantage_centering,
    compute_annealed_teacher_temperature,
    compute_q_mixture_source_normalization_weights,
    compute_samplek_loo_variance,
    compute_samplek_loo_variance_histogram_diagnostics,
    compute_samplek_loo_variance_ratio_diagnostics,
    compute_samplek_update0_loo_variance_quantile_threshold,
    compute_sampled_token_rkl_diagnostics,
    compute_teacher_temperature_anneal_progress,
    compute_trajectory_mixture_proposal,
    is_decomposed_pi_old_mode,
    normalize_q_mixture_teacher_advantage_mode,
    normalize_samplek_loo_variance_threshold_mode,
    normalize_teacher_temperature_anneal_schedule,
    requires_teacher_on_student_log_probs,
    validate_teacher_temperature_configuration,
)
from verl.trainer.ppo.prefix_drift import (
    PREFIX_DRIFT_RAW_WEIGHTS_KEY,
    PREFIX_DRIFT_REFERENCE_LOG_PROBS_KEY,
    PREFIX_DRIFT_WEIGHTS_KEY,
    compute_prefix_drift,
    compute_source_specific_prefix_drift,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.sampled_token_eos_alignment import (
    OPD_SAMPLED_TOKEN_EOS_TEACHER_PRIMARY_LOG_PROBS_KEY,
    OPD_SAMPLED_TOKEN_EOS_TEACHER_SECONDARY_LOG_PROBS_KEY,
    compute_sampled_token_eos_alignment_metrics,
    validate_sampled_token_eos_alignment_configuration,
)
from verl.trainer.ppo.student_eos_diagnostics import record_nonterminal_eos_probability_metric
from verl.trainer.ppo.terminal_aware_opd import (
    OPD_TERMINAL_BEHAVIOR_EOS_LOG_PROBS_KEY,
    OPD_TERMINAL_BEHAVIOR_SECONDARY_LOG_PROBS_KEY,
    OPD_TERMINAL_STUDENT_TOPM_IDS_KEY,
    OPD_TERMINAL_STUDENT_TOPM_LOG_PROBS_KEY,
    OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY,
    prepare_terminal_teacher_remap_only_objective,
    remap_terminal_teacher_trajectory_log_probs,
    validate_terminal_aware_configuration,
)
from verl.trainer.ppo.teacher_deficit_residual import (
    validate_teacher_deficit_residual_configuration,
)
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


ADAPTIVE_PPO_PROBE_IDS_KEY = "adaptive_ppo_probe_candidate_ids"
ADAPTIVE_PPO_PROBE_REFERENCE_LOG_PROBS_KEY = "adaptive_ppo_probe_reference_log_probs"
ADAPTIVE_PPO_PROBE_TEACHER_LOG_PROBS_KEY = "adaptive_ppo_probe_teacher_log_probs"
DAPO_ANSWER_FORMAT_MASK_KEY = "dapo_answer_format_mask"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "true_reward_score" in data.batch:
            adv_kwargs["true_reward_score"] = data.batch["true_reward_score"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        res = adv_estimator_fn(**adv_kwargs)
        if len(res) == 2:
            advantages, returns = res
        elif len(res) == 3:
            advantages, returns, extra_metrics = res
            for k, v in extra_metrics.items():
                data.batch[k] = v
        else:
            raise ValueError("Invalid return from adv_estimator_fn")

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self._sampled_token_eos_alignment_config()
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        replay_buffer_size = int(self.config.actor_rollout_ref.rollout.get("offpolicy_replay_buffer_size", 200))
        self._offpolicy_replay_buffer = deque(maxlen=max(0, replay_buffer_size))
        replay_seed = self.config.actor_rollout_ref.rollout.get("offpolicy_replay_seed", None)
        if isinstance(replay_seed, str) and replay_seed.strip().lower() in {"", "none", "null"}:
            replay_seed = None
        elif replay_seed is not None:
            replay_seed = int(replay_seed)
        self._offpolicy_replay_rng = np.random.default_rng(replay_seed)
        self._offpolicy_pending_prompt_batch: Optional[DataProto] = None
        teacher_mix_seed = self.config.actor_rollout_ref.rollout.get("teacher_rollout_mix_seed", None)
        if isinstance(teacher_mix_seed, str) and teacher_mix_seed.strip().lower() in {"", "none", "null"}:
            teacher_mix_seed = None
        elif teacher_mix_seed is not None:
            teacher_mix_seed = int(teacher_mix_seed)
        self._teacher_rollout_mix_rng = np.random.default_rng(teacher_mix_seed)
        if self._teacher_rollout_mix_enabled() and self._offpolicy_replay_enabled():
            raise ValueError("teacher_rollout_mix_enable and offpolicy_replay_enable cannot both be true.")
        validate_teacher_temperature_configuration(
            teacher_temperature=self.config.actor_rollout_ref.rollout.get("teacher_temperature", 1.0),
            anneal_enable=self._teacher_temperature_anneal_enabled(),
            minimum_temperature=self.config.actor_rollout_ref.rollout.get("teacher_temperature_min", 1.0),
            anneal_steps=self.config.actor_rollout_ref.rollout.get("teacher_temperature_anneal_steps", 50),
            teacher_rollout_mix_enable=self._teacher_rollout_mix_enabled(),
            schedule=self.config.actor_rollout_ref.rollout.get(
                "teacher_temperature_anneal_schedule", "linear"
            ),
        )
        if self._teacher_temperature_anneal_enabled() and not self.use_rm:
            raise ValueError(
                "teacher_temperature_anneal_enable=True requires the teacher reward model worker."
            )
        self._teacher_rollout_mix_cache = self._load_teacher_rollout_cache()
        if self._opd_q_mixture_source_normalize_enabled() and not self._opd_q_mixture_enabled():
            raise ValueError(
                "opd_q_mixture_source_normalize_enable=True requires opd_q_mixture_enable=True."
            )
        if (
            self._opd_q_mixture_teacher_loss_lambda() is not None
            and not self._opd_q_mixture_source_normalize_enabled()
        ):
            raise ValueError(
                "opd_q_mixture_teacher_loss_lambda requires "
                "opd_q_mixture_source_normalize_enable=True."
            )
        if (
            abs(self._opd_q_mixture_samplek_loss_coef() - 1.0) > 1e-12
            and not self._opd_q_mixture_source_normalize_enabled()
        ):
            raise ValueError(
                "opd_q_mixture_samplek_loss_coef requires "
                "opd_q_mixture_source_normalize_enable=True."
            )
        if self._opd_q_mixture_diagnostics_enabled() and not self._opd_q_prefix_samplek_enabled():
            raise ValueError(
                "opd_q_mixture_diagnostics_enable=True requires Q-prefix sample-k "
                "(opd_q_mixture_enable=True and opd_q_mixture_teacher_advantage_mode != proposal)."
            )
        if self._opd_q_mixture_enabled():
            if not self.use_rm:
                raise ValueError("opd_q_mixture_enable=True requires the teacher reward model worker.")
            if not self._teacher_rollout_mix_enabled():
                raise ValueError("opd_q_mixture_enable=True requires teacher_rollout_mix_enable=True.")
            rollout_config = self.config.actor_rollout_ref.rollout
            opd_mode = rollout_config.get("opd_advantage_mode", "fixed")
            q_advantage_mode = self._opd_q_mixture_teacher_advantage_mode()
            top_k = int(self.config.actor_rollout_ref.rollout.get("log_prob_top_k", 0))
            if q_advantage_mode == "proposal":
                if not is_decomposed_pi_old_mode(opd_mode):
                    raise ValueError(
                        "Q-mixture proposal teacher advantage requires "
                        "opd_advantage_mode=decomposed_pi_old."
                    )
                if top_k != 0:
                    raise ValueError(
                        "Q-mixture proposal teacher advantage supports sampled-token OPD only; "
                        f"log_prob_top_k must be 0, got {top_k}."
                    )
            else:
                normalized_opd_mode = str(opd_mode or "fixed").strip().lower().replace("-", "_")
                if normalized_opd_mode not in {"current_kl_is", "current_kl"}:
                    raise ValueError(
                        "Q-prefix sample-k teacher advantages require "
                        "opd_advantage_mode=current_kl_is."
                    )
                candidate_mode = str(rollout_config.get("log_prob_candidate_mode", "topk"))
                candidate_mode = candidate_mode.strip().lower().replace("-", "_")
                candidate_mode = {
                    "sample": "sample_stu",
                    "sample_student": "sample_stu",
                    "student_sample": "sample_stu",
                    "sample_k": "sample_stu",
                }.get(candidate_mode, candidate_mode)
                if top_k <= 0 or candidate_mode != "sample_stu":
                    raise ValueError(
                        "Q-prefix sample-k teacher advantages require "
                        "log_prob_candidate_mode=sample_stu and log_prob_top_k>0."
                    )
                if not self._adaptive_ppo_update_enabled() or not self._prefix_drift_enabled():
                    raise ValueError(
                        "Q-prefix sample-k teacher advantages require adaptive PPO updates "
                        "and prefix_drift_enable=True."
                    )
                if self._config_bool(rollout_config.get("sample_k_kl_plus_one", True)):
                    raise ValueError(
                        "Q-prefix sample-k current advantages implement A=log(pi_current/pi_teacher); "
                        "set sample_k_kl_plus_one=False."
                    )
            if self._opd_q_mixture_source_normalize_enabled():
                if not self._opd_q_prefix_samplek_enabled():
                    raise ValueError(
                        "opd_q_mixture_source_normalize_enable=True requires a Q-prefix "
                        "sample-k teacher advantage mode."
                    )
                loss_agg_mode = str(self.config.actor_rollout_ref.actor.get("loss_agg_mode", "token-mean"))
                if loss_agg_mode != "token-mean":
                    raise ValueError(
                        "Q-mixture source normalization requires actor.loss_agg_mode=token-mean, "
                        f"got {loss_agg_mode!r}."
                    )
            if self._opd_q_mixture_source_behavior_prefix_enabled():
                if not self._opd_q_prefix_samplek_enabled():
                    raise ValueError(
                        "opd_q_mixture_prefix_reference_mode=source_behavior requires "
                        "a Q-prefix sample-k teacher advantage mode."
                    )
                if not self._prefix_drift_enabled():
                    raise ValueError(
                        "opd_q_mixture_prefix_reference_mode=source_behavior requires "
                        "prefix_drift_enable=True."
                    )
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        train_batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        if self._offpolicy_replay_enabled():
            train_batch_size = self._offpolicy_replay_current_prompt_batch_size()
            replay_prompt_batch_size = self._offpolicy_replay_sample_prompt_batch_size()
            if train_batch_size <= 0:
                raise ValueError(f"offpolicy_replay current prompt batch size must be positive, got {train_batch_size}")
            if replay_prompt_batch_size < 0:
                raise ValueError(
                    f"offpolicy_replay sample prompt batch size must be non-negative, got {replay_prompt_batch_size}"
                )
            target_prompt_batch_size = int(self.config.data.train_batch_size)
            mixed_prompt_batch_size = train_batch_size + replay_prompt_batch_size
            if mixed_prompt_batch_size != target_prompt_batch_size:
                print(
                    "Warning: off-policy replay prompt mix does not match data.train_batch_size: "
                    f"{train_batch_size} current + {replay_prompt_batch_size} replay != {target_prompt_batch_size}"
                )
            print(
                "Off-policy replay enabled: "
                f"current_prompts={train_batch_size}, replay_prompts={replay_prompt_batch_size}, "
                f"buffer_size={self._offpolicy_replay_buffer.maxlen}, "
                f"skip_hit_budget={self._offpolicy_replay_skip_hit_budget()}"
            )

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    @staticmethod
    def _config_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @staticmethod
    def _config_optional_float(value, name: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip().lower()
            if value in {"", "none", "null", "auto"}:
                return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a float or null, got {value!r}") from exc

    @staticmethod
    def _config_optional_int(value, name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip().lower()
            if value in {"", "none", "null", "auto"}:
                return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an int or null, got {value!r}") from exc

    def _sampled_token_eos_alignment_config(self) -> tuple[bool, int | None, int | None]:
        rollout_config = self.config.actor_rollout_ref.rollout
        enabled = self._config_bool(
            rollout_config.get("opd_sampled_token_eos_alignment_enable", False)
        )
        teacher_eos_token_id = self._config_optional_int(
            rollout_config.get("opd_sampled_token_teacher_eos_token_id", None),
            "opd_sampled_token_teacher_eos_token_id",
        )
        student_eos_token_id = self.tokenizer.eos_token_id
        if isinstance(student_eos_token_id, list):
            student_eos_token_id = student_eos_token_id[0] if student_eos_token_id else None
        if student_eos_token_id is not None:
            student_eos_token_id = int(student_eos_token_id)
        validate_sampled_token_eos_alignment_configuration(
            enable=enabled,
            log_prob_top_k=int(rollout_config.get("log_prob_top_k", 0)),
            student_eos_token_id=student_eos_token_id,
            teacher_eos_token_id=teacher_eos_token_id,
            use_reward_model=self.use_rm,
        )
        return enabled, student_eos_token_id, teacher_eos_token_id

    def _dapo_answer_format_penalty_config(self) -> tuple[bool, float, int]:
        rollout_config = self.config.actor_rollout_ref.rollout
        coef = self._config_optional_float(
            rollout_config.get("opd_dapo_format_penalty_coef", 0.0),
            "actor_rollout_ref.rollout.opd_dapo_format_penalty_coef",
        )
        coef = 0.0 if coef is None else coef
        if not math.isfinite(coef) or coef < 0.0:
            raise ValueError("opd_dapo_format_penalty_coef must be finite and nonnegative.")

        enabled = (
            self._config_bool(rollout_config.get("opd_dapo_format_penalty_enable", False))
            or coef > 0.0
        )
        tail_tokens = self._config_optional_int(
            rollout_config.get("opd_dapo_format_penalty_tail_tokens", 512),
            "actor_rollout_ref.rollout.opd_dapo_format_penalty_tail_tokens",
        )
        tail_tokens = 512 if tail_tokens is None else tail_tokens
        if tail_tokens <= 0:
            raise ValueError("opd_dapo_format_penalty_tail_tokens must be positive.")
        return enabled, coef, tail_tokens

    def _set_dapo_answer_format_penalty_meta(self, batch: DataProto, metrics: dict | None = None) -> None:
        enabled, coef, tail_tokens = self._dapo_answer_format_penalty_config()
        batch.meta_info["dapo_answer_format_check_enable"] = enabled
        batch.meta_info["opd_dapo_format_penalty_enable"] = enabled
        batch.meta_info["opd_dapo_format_penalty_coef"] = coef
        batch.meta_info["opd_dapo_format_penalty_tail_tokens"] = tail_tokens
        if metrics is not None:
            metrics["opd_dapo_format_penalty/config_enable"] = float(enabled)
            metrics["opd_dapo_format_penalty/config_coef"] = float(coef)
            metrics["opd_dapo_format_penalty/config_tail_tokens"] = float(tail_tokens)

    @staticmethod
    def _reward_extra_vector_to_tensor(
        value,
        *,
        batch: DataProto,
        key: str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
        else:
            tensor = torch.as_tensor(value)
        expected_batch = batch.batch["responses"].shape[0]
        if tensor.dim() == 2 and tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)
        if tensor.dim() != 1:
            if tensor.numel() != expected_batch:
                raise ValueError(
                    f"{key} must be a batch vector, got shape={tuple(tensor.shape)} "
                    f"for batch size {expected_batch}."
                )
            tensor = tensor.reshape(expected_batch)
        if tensor.shape[0] != expected_batch:
            raise ValueError(
                f"{key} length must match batch size, got {tensor.shape[0]} vs {expected_batch}."
            )
        return tensor.to(device=batch.batch["responses"].device, dtype=dtype)

    def _attach_reward_extra_batch_tensors(
        self,
        batch: DataProto,
        reward_extra_infos_dict: dict,
        metrics: dict | None = None,
    ) -> None:
        for key in ("format_mask", DAPO_ANSWER_FORMAT_MASK_KEY):
            if key not in reward_extra_infos_dict:
                continue
            tensor = self._reward_extra_vector_to_tensor(
                reward_extra_infos_dict[key],
                batch=batch,
                key=key,
            )
            batch.batch[key] = tensor
            if metrics is not None and key == DAPO_ANSWER_FORMAT_MASK_KEY:
                ok_rate = tensor.float().mean().detach().item()
                metrics["format/dapo_answer_ok_rate"] = ok_rate
                metrics["format/dapo_answer_missing_rate"] = 1.0 - ok_rate

    def _offpolicy_replay_enabled(self) -> bool:
        return self._config_bool(self.config.actor_rollout_ref.rollout.get("offpolicy_replay_enable", False))

    def _adaptive_ppo_update_enabled(self) -> bool:
        return self._config_bool(self.config.actor_rollout_ref.rollout.get("adaptive_ppo_update_enable", False))

    def _samplek_candidate_reuse_enabled(self) -> bool:
        return self._config_bool(
            self.config.actor_rollout_ref.rollout.get("samplek_candidate_reuse_is_enable", False)
        )

    def _prefix_drift_enabled(self) -> bool:
        return self._config_bool(self.config.actor_rollout_ref.rollout.get("prefix_drift_enable", False))

    def _ess_samplek_resample_enabled(self) -> bool:
        return self._config_bool(self.config.actor_rollout_ref.rollout.get("ess_samplek_resample_enable", False))

    def _teacher_rollout_mix_enabled(self) -> bool:
        return self._config_bool(self.config.actor_rollout_ref.rollout.get("teacher_rollout_mix_enable", False))

    def _teacher_temperature_anneal_enabled(self) -> bool:
        return self._config_bool(
            self.config.actor_rollout_ref.rollout.get("teacher_temperature_anneal_enable", False)
        )

    def _teacher_temperature_anneal_schedule(self) -> str:
        return normalize_teacher_temperature_anneal_schedule(
            self.config.actor_rollout_ref.rollout.get(
                "teacher_temperature_anneal_schedule", "linear"
            )
        )

    def _current_teacher_temperature(self) -> float:
        rollout_config = self.config.actor_rollout_ref.rollout
        initial_temperature = float(rollout_config.get("teacher_temperature", 1.0))
        if not self._teacher_temperature_anneal_enabled():
            return initial_temperature
        return compute_annealed_teacher_temperature(
            step=self.global_steps,
            initial_temperature=initial_temperature,
            minimum_temperature=float(rollout_config.get("teacher_temperature_min", 1.0)),
            anneal_steps=int(rollout_config.get("teacher_temperature_anneal_steps", 50)),
            schedule=self._teacher_temperature_anneal_schedule(),
        )

    def _teacher_temperature_anneal_progress(self) -> float:
        if not self._teacher_temperature_anneal_enabled():
            return 0.0
        return compute_teacher_temperature_anneal_progress(
            step=self.global_steps,
            anneal_steps=int(
                self.config.actor_rollout_ref.rollout.get("teacher_temperature_anneal_steps", 50)
            ),
        )

    def _opd_q_mixture_enabled(self) -> bool:
        return self._config_bool(self.config.actor_rollout_ref.rollout.get("opd_q_mixture_enable", False))

    def _opd_q_mixture_teacher_advantage_mode(self) -> str:
        value = self.config.actor_rollout_ref.rollout.get(
            "opd_q_mixture_teacher_advantage_mode", "proposal"
        )
        return normalize_q_mixture_teacher_advantage_mode(value)

    def _opd_q_prefix_samplek_enabled(self) -> bool:
        return self._opd_q_mixture_enabled() and self._opd_q_mixture_teacher_advantage_mode() != "proposal"

    def _opd_q_mixture_source_normalize_enabled(self) -> bool:
        return self._config_bool(
            self.config.actor_rollout_ref.rollout.get("opd_q_mixture_source_normalize_enable", False)
        )

    def _opd_q_mixture_diagnostics_enabled(self) -> bool:
        return self._config_bool(
            self.config.actor_rollout_ref.rollout.get("opd_q_mixture_diagnostics_enable", False)
        )

    def _opd_q_mixture_prefix_reference_mode(self) -> str:
        value = str(
            self.config.actor_rollout_ref.rollout.get(
                "opd_q_mixture_prefix_reference_mode", "mixture"
            )
        )
        mode = value.strip().lower().replace("-", "_")
        aliases = {
            "q": "mixture",
            "proposal": "mixture",
            "trajectory_mixture": "mixture",
            "source": "source_behavior",
            "source_specific": "source_behavior",
            "behavior": "source_behavior",
            "source_behaviour": "source_behavior",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"mixture", "source_behavior"}:
            raise ValueError(
                "opd_q_mixture_prefix_reference_mode must be mixture or "
                f"source_behavior, got {value!r}."
            )
        return mode

    def _opd_q_mixture_source_behavior_prefix_enabled(self) -> bool:
        return self._opd_q_mixture_prefix_reference_mode() == "source_behavior"

    def _opd_q_mixture_teacher_prefix_method(self) -> str:
        return str(
            self.config.actor_rollout_ref.rollout.get(
                "opd_q_mixture_teacher_prefix_method", "prefix"
            )
        )

    def _opd_q_mixture_teacher_prefix_log_clip(self) -> Optional[float]:
        return self._config_optional_float(
            self.config.actor_rollout_ref.rollout.get(
                "opd_q_mixture_teacher_prefix_log_clip", None
            ),
            "opd_q_mixture_teacher_prefix_log_clip",
        )

    def _opd_q_mixture_teacher_prefix_log_clip_mode(self) -> Optional[str]:
        value = self.config.actor_rollout_ref.rollout.get(
            "opd_q_mixture_teacher_prefix_log_clip_mode", None
        )
        if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"}):
            return None
        return str(value)

    def _opd_q_mixture_teacher_prefix_min_weight(self) -> Optional[float]:
        return self._config_optional_float(
            self.config.actor_rollout_ref.rollout.get(
                "opd_q_mixture_teacher_prefix_min_weight", None
            ),
            "opd_q_mixture_teacher_prefix_min_weight",
        )

    def _opd_q_mixture_teacher_loss_lambda(self) -> Optional[float]:
        value = self.config.actor_rollout_ref.rollout.get("opd_q_mixture_teacher_loss_lambda", None)
        if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"}):
            return None
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "opd_q_mixture_teacher_loss_lambda must be null or a finite nonnegative value."
            )
        return value

    def _opd_q_mixture_samplek_loss_coef(self) -> float:
        value = float(
            self.config.actor_rollout_ref.rollout.get("opd_q_mixture_samplek_loss_coef", 1.0)
        )
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "opd_q_mixture_samplek_loss_coef must be finite and nonnegative."
            )
        return value

    @staticmethod
    def _normalize_teacher_rollout_index(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped.lstrip("-").isdigit():
                try:
                    return int(stripped)
                except ValueError:
                    return stripped
            return stripped
        return value

    @staticmethod
    def _extract_teacher_rollout_text(record: dict) -> str:
        if isinstance(record.get("output"), str):
            return record["output"]
        if isinstance(record.get("response"), str):
            return record["response"]
        messages = record.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "assistant":
                    content = message.get("content", "")
                    return content if isinstance(content, str) else str(content)
        return ""

    @staticmethod
    def _strip_teacher_rollout_artifacts(text: str) -> str:
        if not text:
            return text
        import re

        cleaned = re.sub(r"(?is)<think>\s*.*?</think>\s*", "", text)
        cleaned = cleaned.replace("<think>", "").replace("</think>", "")
        for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            cleaned = cleaned.replace(token, "")
        return cleaned.lstrip()

    def _load_teacher_rollout_cache(self) -> dict:
        if not self._teacher_rollout_mix_enabled():
            return {}

        rollout_config = self.config.actor_rollout_ref.rollout
        cache_path = rollout_config.get("teacher_rollout_mix_path", None)
        if cache_path is None or str(cache_path).strip().lower() in {"", "none", "null"}:
            raise ValueError("teacher_rollout_mix_enable=True requires teacher_rollout_mix_path.")

        cache_path = os.path.expanduser(str(cache_path))
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"teacher rollout mix cache not found: {cache_path}")

        index_field = str(rollout_config.get("teacher_rollout_mix_cache_index_field", "global_index"))
        cache: dict = defaultdict(list)
        with open(cache_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if index_field in record:
                    raw_index = record[index_field]
                elif "global_index" in record:
                    raw_index = record["global_index"]
                elif "index" in record:
                    raw_index = record["index"]
                else:
                    raise ValueError(
                        f"teacher rollout record at line {line_no} has no {index_field!r}, global_index, or index"
                    )
                text = self._strip_teacher_rollout_artifacts(self._extract_teacher_rollout_text(record))
                if not text:
                    continue
                rollout_index = record.get("rollout_index", len(cache[self._normalize_teacher_rollout_index(raw_index)]))
                cache[self._normalize_teacher_rollout_index(raw_index)].append(
                    {
                        "rollout_index": int(rollout_index),
                        "text": text,
                        "hit_max_tokens": bool(record.get("hit_max_tokens", False)),
                    }
                )

        for key in list(cache.keys()):
            cache[key] = sorted(cache[key], key=lambda item: item["rollout_index"])

        total_rollouts = sum(len(v) for v in cache.values())
        print(
            f"Loaded teacher rollout mix cache from {cache_path}: "
            f"{len(cache)} prompts, {total_rollouts} rollouts"
        )
        return dict(cache)

    def _teacher_rollout_mix_num_rollouts(self, available_count: int) -> int:
        configured = self._config_optional_int(
            self.config.actor_rollout_ref.rollout.get("teacher_rollout_mix_num_rollouts", 0),
            "teacher_rollout_mix_num_rollouts",
        )
        if configured is None or configured <= 0:
            return available_count
        return min(configured, available_count)

    def _select_teacher_rollouts_for_prompt(self, prompt_index) -> list[dict]:
        rollouts = self._teacher_rollout_mix_cache.get(self._normalize_teacher_rollout_index(prompt_index), [])
        if not rollouts:
            return []
        count = self._teacher_rollout_mix_num_rollouts(len(rollouts))
        if count <= 0:
            return []

        selection = str(
            self.config.actor_rollout_ref.rollout.get("teacher_rollout_mix_selection", "first")
        ).strip().lower()
        if selection == "first":
            return rollouts[:count]
        if selection == "random":
            chosen = self._teacher_rollout_mix_rng.choice(len(rollouts), size=count, replace=False)
            return [rollouts[int(i)] for i in chosen]
        raise ValueError(f"Unsupported teacher_rollout_mix_selection={selection!r}; expected first or random.")

    def _encode_teacher_rollout_response(
        self,
        text: str,
        response_length: int,
        *,
        append_eos: bool = True,
    ) -> tuple[list[int], bool]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        truncated = len(token_ids) > response_length
        token_ids = token_ids[:response_length]

        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        if append_eos and eos_token_id is not None and not truncated and len(token_ids) < response_length:
            if not token_ids or token_ids[-1] != eos_token_id:
                token_ids.append(eos_token_id)
        return token_ids, truncated

    def _ensure_teacher_rollout_mix_non_tensor_keys(self, batch: DataProto) -> None:
        row_count = len(batch)
        defaults = {
            "teacher_rollout_mix_source": "student",
            "teacher_rollout_mix_rollout_index": -1,
            "teacher_rollout_mix_global_index": -1,
        }
        for key, default in defaults.items():
            if key not in batch.non_tensor_batch:
                batch.non_tensor_batch[key] = np.full(row_count, default, dtype=object)

    def _build_teacher_rollout_batch(
        self,
        batch: DataProto,
        base_indices: list[int],
        rollout_texts: list[str],
        rollout_hit_max_tokens: list[bool],
        rollout_indices: list[int],
        prompt_indices: list,
    ) -> tuple[DataProto | None, int, int]:
        if not base_indices:
            return None, 0, 0

        if len(rollout_hit_max_tokens) != len(base_indices):
            raise ValueError(
                "teacher rollout hit-max metadata must match the selected rows, "
                f"got {len(rollout_hit_max_tokens)} flags for {len(base_indices)} rows."
            )

        if "prompts" not in batch.batch.keys():
            raise ValueError("teacher rollout mix requires rollout output batch key 'prompts'.")
        if "responses" not in batch.batch.keys():
            raise ValueError("teacher rollout mix requires rollout output batch key 'responses'.")

        first_tensor = next(iter(batch.batch.values()))
        device = first_tensor.device
        base_idx_tensor = torch.as_tensor(base_indices, dtype=torch.long, device=device)
        prompts = batch.batch["prompts"].index_select(0, base_idx_tensor)
        prompt_length = prompts.size(-1)
        response_length = batch.batch["responses"].size(-1)
        prompt_attention_mask = batch.batch["attention_mask"].index_select(0, base_idx_tensor)[:, :prompt_length]

        full_position_ids = batch.batch["position_ids"].index_select(0, base_idx_tensor)
        prompt_position_ids = full_position_ids[..., :prompt_length]

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if isinstance(pad_token_id, list):
            pad_token_id = pad_token_id[0]

        responses = torch.full(
            (len(base_indices), response_length),
            int(pad_token_id),
            dtype=prompts.dtype,
            device=device,
        )
        response_attention_mask = torch.zeros(
            (len(base_indices), response_length),
            dtype=prompt_attention_mask.dtype,
            device=device,
        )
        truncated_count = 0
        censored_count = 0
        for row_idx, (text, hit_max_tokens) in enumerate(zip(rollout_texts, rollout_hit_max_tokens, strict=True)):
            token_ids, truncated = self._encode_teacher_rollout_response(
                text,
                response_length,
                append_eos=not hit_max_tokens,
            )
            truncated_count += int(truncated)
            censored_count += int(hit_max_tokens)
            if token_ids:
                token_tensor = torch.as_tensor(token_ids, dtype=responses.dtype, device=device)
                responses[row_idx, : token_tensor.numel()] = token_tensor
                response_attention_mask[row_idx, : token_tensor.numel()] = 1

        input_ids = torch.cat([prompts, responses], dim=-1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)

        delta_position_id = torch.arange(
            1,
            response_length + 1,
            device=device,
            dtype=prompt_position_ids.dtype,
        )
        delta_position_id = delta_position_id.unsqueeze(0).expand(len(base_indices), -1)
        if prompt_position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(len(base_indices), 1, -1).expand(
                len(base_indices),
                prompt_position_ids.size(1),
                -1,
            )
        response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)

        teacher_tensors = {}
        for key in batch.batch.keys():
            if key == "prompts":
                teacher_tensors[key] = prompts
            elif key == "responses":
                teacher_tensors[key] = responses
            elif key == "input_ids":
                teacher_tensors[key] = input_ids
            elif key == "attention_mask":
                teacher_tensors[key] = attention_mask
            elif key == "position_ids":
                teacher_tensors[key] = position_ids
            elif key == "response_mask":
                teacher_tensors[key] = response_attention_mask
            elif key == "rollout_log_probs":
                teacher_tensors[key] = torch.zeros(
                    (len(base_indices), response_length),
                    dtype=batch.batch[key].dtype,
                    device=device,
                )
            else:
                value = batch.batch[key]
                if value.shape[0] != len(batch):
                    raise ValueError(f"teacher rollout mix cannot construct batch tensor key {key!r}.")
                teacher_tensors[key] = value.index_select(0, base_idx_tensor).clone()

        base_indices_np = np.asarray(base_indices)
        teacher_non_tensor = {}
        for key, values in batch.non_tensor_batch.items():
            teacher_non_tensor[key] = values[base_indices_np].copy()
        teacher_non_tensor["teacher_rollout_mix_source"] = np.full(len(base_indices), "teacher", dtype=object)
        teacher_non_tensor["teacher_rollout_mix_rollout_index"] = np.asarray(rollout_indices, dtype=object)
        teacher_non_tensor["teacher_rollout_mix_global_index"] = np.asarray(prompt_indices, dtype=object)

        teacher_batch = DataProto.from_dict(
            tensors=teacher_tensors,
            non_tensors=teacher_non_tensor,
            meta_info=batch.meta_info,
        )
        return teacher_batch, truncated_count, censored_count

    def _mix_teacher_rollout_batch(self, batch: DataProto, metrics: dict) -> DataProto:
        if not self._teacher_rollout_mix_enabled():
            batch.meta_info["num_repeat"] = int(self.config.actor_rollout_ref.rollout.n)
            return batch

        if not self._teacher_rollout_mix_cache:
            raise ValueError("teacher_rollout_mix_enable=True but the teacher rollout cache is empty.")

        student_n = int(self.config.actor_rollout_ref.rollout.n)
        if student_n <= 0:
            raise ValueError(f"actor_rollout_ref.rollout.n must be positive, got {student_n}")
        if len(batch) % student_n != 0:
            raise ValueError(
                f"teacher rollout mix expects student rows to be grouped by n={student_n}, got {len(batch)} rows"
            )

        index_key = str(self.config.actor_rollout_ref.rollout.get("teacher_rollout_mix_index_key", "global_index"))
        if index_key not in batch.non_tensor_batch:
            available = sorted(batch.non_tensor_batch.keys())
            raise ValueError(f"teacher rollout mix index key {index_key!r} not found in batch; available={available}")

        self._ensure_teacher_rollout_mix_non_tensor_keys(batch)

        prompt_count = len(batch) // student_n
        teacher_base_indices = []
        teacher_texts = []
        teacher_hit_max_tokens = []
        teacher_rollout_indices = []
        teacher_prompt_indices = []
        teacher_counts_by_prompt = []
        matched_prompt_count = 0
        missing_prompt_count = 0

        for prompt_idx in range(prompt_count):
            base_row = prompt_idx * student_n
            prompt_index = batch.non_tensor_batch[index_key][base_row]
            selected_rollouts = self._select_teacher_rollouts_for_prompt(prompt_index)
            teacher_counts_by_prompt.append(len(selected_rollouts))
            if not selected_rollouts:
                missing_prompt_count += 1
                continue
            matched_prompt_count += 1
            for rollout in selected_rollouts:
                teacher_base_indices.append(base_row)
                teacher_texts.append(rollout["text"])
                teacher_hit_max_tokens.append(rollout["hit_max_tokens"])
                teacher_rollout_indices.append(rollout["rollout_index"])
                teacher_prompt_indices.append(prompt_index)

        teacher_batch, truncated_count, censored_count = self._build_teacher_rollout_batch(
            batch=batch,
            base_indices=teacher_base_indices,
            rollout_texts=teacher_texts,
            rollout_hit_max_tokens=teacher_hit_max_tokens,
            rollout_indices=teacher_rollout_indices,
            prompt_indices=teacher_prompt_indices,
        )
        teacher_row_count = 0 if teacher_batch is None else len(teacher_batch)
        mixed_batch = batch if teacher_batch is None else DataProto.concat([batch, teacher_batch])

        if self._opd_q_mixture_enabled():
            prompt_alphas = np.asarray(
                [student_n / (student_n + teacher_count) for teacher_count in teacher_counts_by_prompt],
                dtype=np.float32,
            )
            student_alphas = np.repeat(prompt_alphas, student_n)
            if teacher_row_count > 0:
                teacher_alphas = np.concatenate(
                    [
                        np.full(teacher_count, prompt_alphas[prompt_idx], dtype=np.float32)
                        for prompt_idx, teacher_count in enumerate(teacher_counts_by_prompt)
                        if teacher_count > 0
                    ]
                )
            else:
                teacher_alphas = np.empty(0, dtype=np.float32)
            row_alphas = np.concatenate([student_alphas, teacher_alphas])
            if row_alphas.shape[0] != len(mixed_batch):
                raise RuntimeError(
                    "Q-mixture alpha construction did not match the mixed batch: "
                    f"{row_alphas.shape[0]} alphas for {len(mixed_batch)} rows."
                )

            first_tensor = next(iter(mixed_batch.batch.values()))
            mixed_batch.batch[OPD_Q_MIXTURE_ALPHA_KEY] = torch.as_tensor(
                row_alphas,
                dtype=torch.float32,
                device=first_tensor.device,
            )
            source_is_teacher = torch.zeros(len(mixed_batch), dtype=torch.bool, device=first_tensor.device)
            source_is_teacher[len(batch) :] = True
            mixed_batch.batch[OPD_Q_MIXTURE_IS_TEACHER_KEY] = source_is_teacher
            metrics.update(
                {
                    "opd_q_mixture/enabled": 1.0,
                    "opd_q_mixture/prompt_alpha_mean": float(prompt_alphas.mean()),
                    "opd_q_mixture/prompt_alpha_min": float(prompt_alphas.min()),
                    "opd_q_mixture/prompt_alpha_max": float(prompt_alphas.max()),
                    "opd_q_mixture/teacher_row_fraction": teacher_row_count / max(len(mixed_batch), 1),
                }
            )

        effective_num_repeat = student_n
        if prompt_count > 0 and matched_prompt_count == prompt_count and teacher_row_count % prompt_count == 0:
            effective_num_repeat = student_n + teacher_row_count // prompt_count
        mixed_batch.meta_info["num_repeat"] = effective_num_repeat
        metrics.update(
            {
                "teacher_rollout_mix/student_rows": len(batch),
                "teacher_rollout_mix/teacher_rows": teacher_row_count,
                "teacher_rollout_mix/matched_prompts": matched_prompt_count,
                "teacher_rollout_mix/missing_prompts": missing_prompt_count,
                "teacher_rollout_mix/cache_prompts": len(self._teacher_rollout_mix_cache),
                "teacher_rollout_mix/effective_num_repeat": effective_num_repeat,
                "teacher_rollout_mix/truncated_teacher_rows": truncated_count,
                "teacher_rollout_mix/censored_teacher_rows": censored_count,
            }
        )
        return mixed_batch

    def _patch_teacher_rollout_mix_rollout_log_probs(self, batch: DataProto) -> None:
        if not self._teacher_rollout_mix_enabled():
            return
        if "teacher_rollout_mix_source" not in batch.non_tensor_batch:
            return
        if "rollout_log_probs" not in batch.batch.keys() or "old_log_probs" not in batch.batch.keys():
            return

        source = batch.non_tensor_batch["teacher_rollout_mix_source"]
        teacher_mask_np = np.asarray([value == "teacher" for value in source], dtype=bool)
        if not teacher_mask_np.any():
            return
        teacher_mask = torch.as_tensor(teacher_mask_np, dtype=torch.bool, device=batch.batch["old_log_probs"].device)
        batch.batch["rollout_log_probs"][teacher_mask] = batch.batch["old_log_probs"][teacher_mask].detach().to(
            dtype=batch.batch["rollout_log_probs"].dtype
        )

    def _prepare_opd_q_mixture_batch(self, batch: DataProto, metrics: dict) -> None:
        if not self._opd_q_mixture_enabled():
            return

        required_keys = {
            "old_log_probs",
            "response_mask",
            OPD_Q_MIXTURE_ALPHA_KEY,
            OPD_Q_MIXTURE_IS_TEACHER_KEY,
        }
        missing_keys = sorted(required_keys.difference(batch.batch.keys()))
        if missing_keys:
            raise ValueError(f"Q-mixture OPD batch is missing required tensors: {missing_keys}")

        old_log_probs = batch.batch["old_log_probs"]
        teacher_log_probs = batch.batch.get(OPD_Q_MIXTURE_TEACHER_LOG_PROBS_KEY)
        if teacher_log_probs is None:
            teacher_log_probs = batch.batch.get("teacher_on_student_log_probs")
        if teacher_log_probs is None:
            raise ValueError(
                "Q-mixture OPD batch requires sampled-token teacher log-probs in "
                f"{OPD_Q_MIXTURE_TEACHER_LOG_PROBS_KEY!r}."
            )
        rollout_config = self.config.actor_rollout_ref.rollout
        terminal_q_remap_enable = (
            self._config_bool(rollout_config.get("opd_terminal_aware_enable", False))
            and str(rollout_config.get("opd_terminal_objective_mode", "anchor_kl"))
            .strip()
            .lower()
            .replace("-", "_")
            == "teacher_remap_only"
            and self._config_bool(rollout_config.get("opd_terminal_teacher_remap_enable", False))
        )
        if terminal_q_remap_enable:
            remap_required_keys = {
                "responses",
                "student_top_k_ids",
                "teacher_on_student_log_probs",
                OPD_Q_MIXTURE_TEACHER_LOG_PROBS_KEY,
                OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY,
            }
            missing_remap_keys = sorted(remap_required_keys.difference(batch.batch.keys()))
            if missing_remap_keys:
                raise ValueError(
                    "Q-mixture teacher EOS remapping is missing required tensors: "
                    f"{missing_remap_keys}"
                )
            eos_token_id = self.tokenizer.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None:
                raise ValueError("Q-mixture teacher EOS remapping requires a student EOS token id.")
            secondary_token_id = rollout_config.get("opd_terminal_secondary_token_id", None)
            if secondary_token_id is None:
                raise ValueError("Q-mixture teacher EOS remapping requires a secondary EOS token id.")
            teacher_remap_floor = float(
                rollout_config.get("opd_terminal_teacher_remap_floor", 1e-18)
            )
            secondary_log_probs = batch.batch[OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY]
            candidate_remap = prepare_terminal_teacher_remap_only_objective(
                teacher_log_probs=batch.batch["teacher_on_student_log_probs"],
                candidate_ids=batch.batch["student_top_k_ids"],
                teacher_secondary_log_probs=secondary_log_probs,
                responses=batch.batch["responses"],
                response_mask=batch.batch["response_mask"],
                eos_token_id=int(eos_token_id),
                secondary_token_id=int(secondary_token_id),
                teacher_remap_floor=teacher_remap_floor,
            )
            trajectory_remap = remap_terminal_teacher_trajectory_log_probs(
                teacher_log_probs=batch.batch[OPD_Q_MIXTURE_TEACHER_LOG_PROBS_KEY],
                teacher_secondary_log_probs=secondary_log_probs,
                responses=batch.batch["responses"],
                response_mask=batch.batch["response_mask"],
                eos_token_id=int(eos_token_id),
            )
            batch.batch["teacher_on_student_log_probs"] = (
                candidate_remap.remapped_teacher_log_probs
            )
            batch.batch[OPD_Q_MIXTURE_TEACHER_LOG_PROBS_KEY] = (
                trajectory_remap.remapped_teacher_log_probs
            )
            teacher_log_probs = trajectory_remap.remapped_teacher_log_probs
            terminal_count = trajectory_remap.terminal_mask.sum()
            terminal_denom = terminal_count.clamp_min(1)
            metrics.update(
                {
                    "opd_q_terminal_remap/enabled": 1.0,
                    "opd_q_terminal_remap/natural_terminal_count": terminal_count.item(),
                    "opd_q_terminal_remap/trajectory_remap_count": terminal_count.item(),
                    "opd_q_terminal_remap/candidate_eos_count": (
                        candidate_remap.eos_candidate_mask.sum().item()
                    ),
                    "opd_q_terminal_remap/candidate_secondary_count": (
                        candidate_remap.secondary_candidate_mask.sum().item()
                    ),
                    "opd_q_terminal_remap/teacher_secondary_probability_mean": (
                        trajectory_remap.teacher_secondary_probability.sum() / terminal_denom
                    ).item(),
                    "opd_q_terminal_remap/teacher_stop_probability_mean": (
                        trajectory_remap.teacher_stop_probability.sum() / terminal_denom
                    ).item(),
                }
            )
        else:
            metrics["opd_q_terminal_remap/enabled"] = 0.0
        response_mask = batch.batch["response_mask"]
        row_alpha = batch.batch[OPD_Q_MIXTURE_ALPHA_KEY]
        source_is_teacher = batch.batch[OPD_Q_MIXTURE_IS_TEACHER_KEY].to(dtype=torch.bool)
        if source_is_teacher.shape != (len(batch),):
            raise ValueError(
                f"{OPD_Q_MIXTURE_IS_TEACHER_KEY} must have shape ({len(batch)},), "
                f"got {tuple(source_is_teacher.shape)}."
            )

        with torch.no_grad():
            proposal = compute_trajectory_mixture_proposal(
                old_log_probs=old_log_probs,
                teacher_log_probs=teacher_log_probs,
                response_mask=response_mask,
                mixture_alpha=row_alpha,
            )
            response_mask_float = response_mask.float()
            student_token_mask = response_mask_float * (~source_is_teacher).float().unsqueeze(-1)
            teacher_token_mask = response_mask_float * source_is_teacher.float().unsqueeze(-1)
            q_log_probs = proposal.conditional_log_probs.to(dtype=old_log_probs.dtype)

            batch.batch["opd_proposal_log_probs"] = q_log_probs
            batch.batch["opd_proximal_mask"] = student_token_mask.to(dtype=response_mask.dtype)
            batch.batch[OPD_Q_MIXTURE_PRIOR_ALPHA_KEY] = row_alpha.detach().clone()
            batch.meta_info["opd_q_mixture_enable"] = True
            batch.meta_info["opd_q_mixture_teacher_advantage_mode"] = (
                self._opd_q_mixture_teacher_advantage_mode()
            )
            batch.meta_info["opd_q_mixture_prefix_reference_mode"] = (
                self._opd_q_mixture_prefix_reference_mode()
            )
            source_normalize_enable = self._opd_q_mixture_source_normalize_enabled()
            batch.meta_info["opd_q_mixture_source_normalize_enable"] = source_normalize_enable
            batch.meta_info["opd_q_mixture_diagnostics_enable"] = self._opd_q_mixture_diagnostics_enabled()
            if source_normalize_enable:
                source_normalization = compute_q_mixture_source_normalization_weights(
                    response_mask=response_mask_float,
                    student_token_mask=student_token_mask,
                    mixture_alpha=row_alpha,
                    teacher_loss_lambda=self._opd_q_mixture_teacher_loss_lambda(),
                    samplek_loss_coefficient=self._opd_q_mixture_samplek_loss_coef(),
                )
                batch.batch[OPD_Q_MIXTURE_SOURCE_WEIGHTS_KEY] = source_normalization.weights
                metrics["opd_q_source_normalize/enabled"] = 1.0
                for name, value in source_normalization.metrics.items():
                    metrics[f"opd_q_source_normalize/{name}"] = value
            else:
                metrics["opd_q_source_normalize/enabled"] = 0.0

            def add_masked_stats(name: str, values: torch.Tensor, mask: torch.Tensor) -> None:
                valid = values.float()[mask > 0.5]
                if valid.numel() == 0:
                    metrics[f"opd_q_mixture/{name}_mean"] = 0.0
                    metrics[f"opd_q_mixture/{name}_abs_mean"] = 0.0
                    metrics[f"opd_q_mixture/{name}_abs_p95"] = 0.0
                    metrics[f"opd_q_mixture/{name}_abs_max"] = 0.0
                    return
                metrics[f"opd_q_mixture/{name}_mean"] = valid.mean().item()
                metrics[f"opd_q_mixture/{name}_abs_mean"] = valid.abs().mean().item()
                metrics[f"opd_q_mixture/{name}_abs_p95"] = torch.quantile(valid.abs(), 0.95).item()
                metrics[f"opd_q_mixture/{name}_abs_max"] = valid.abs().max().item()

            q_minus_old = proposal.conditional_log_probs - old_log_probs.float()
            q_teacher_advantage = proposal.conditional_log_probs - teacher_log_probs.float()
            for source_name, source_mask in (
                ("all", response_mask_float),
                ("student", student_token_mask),
                ("teacher", teacher_token_mask),
            ):
                add_masked_stats(f"{source_name}/q_minus_old", q_minus_old, source_mask)
                add_masked_stats(
                    f"{source_name}/q_teacher_advantage",
                    q_teacher_advantage,
                    source_mask,
                )
                add_masked_stats(
                    f"{source_name}/old_component_posterior",
                    proposal.old_component_posterior,
                    source_mask,
                )

            valid_tokens = response_mask_float.sum().clamp_min(1.0)
            metrics["opd_q_mixture/teacher_token_fraction"] = (
                teacher_token_mask.sum() / valid_tokens
            ).item()
            metrics["opd_q_mixture/student_token_fraction"] = (
                student_token_mask.sum() / valid_tokens
            ).item()
            metrics["opd_q_mixture/alpha_token_mean"] = (
                row_alpha.float().unsqueeze(-1) * response_mask_float
            ).sum().div(valid_tokens).item()
            metrics["opd_q_mixture/prefix_reference_mode_source_behavior"] = float(
                self._opd_q_mixture_source_behavior_prefix_enabled()
            )
            reconstructed_prefix = (proposal.conditional_log_probs * response_mask_float).cumsum(dim=-1)
            reconstruction_error = (reconstructed_prefix - proposal.prefix_log_probs).abs() * response_mask_float
            metrics["opd_q_mixture/prefix_reconstruction_abs_max"] = reconstruction_error.max().item()
            metrics["opd_q_mixture/prefix_only_samplek"] = float(self._opd_q_prefix_samplek_enabled())

        batch.batch.pop(OPD_Q_MIXTURE_ALPHA_KEY)
        batch.batch.pop(OPD_Q_MIXTURE_IS_TEACHER_KEY)

    def _batch_num_repeat(self, batch: DataProto) -> int:
        return int(batch.meta_info.get("num_repeat", self.config.actor_rollout_ref.rollout.n))

    def _offpolicy_replay_skip_hit_budget(self) -> bool:
        return self._config_bool(
            self.config.actor_rollout_ref.rollout.get("offpolicy_replay_skip_hit_budget", False)
        )

    def _offpolicy_replay_sample_prompt_batch_size(self) -> int:
        return int(self.config.actor_rollout_ref.rollout.get("offpolicy_replay_sample_prompt_batch_size", 8))

    def _offpolicy_replay_current_prompt_batch_size(self) -> int:
        rollout_config = self.config.actor_rollout_ref.rollout
        configured_current_batch_size = rollout_config.get("offpolicy_replay_current_prompt_batch_size", None)
        if configured_current_batch_size is not None:
            return int(configured_current_batch_size)

        configured_gen_batch_size = self.config.data.get("gen_batch_size", None)
        if configured_gen_batch_size is not None:
            return int(configured_gen_batch_size)

        return int(self.config.data.train_batch_size) - self._offpolicy_replay_sample_prompt_batch_size()

    def _offpolicy_replay_target_current_prompt_batch_size(self) -> int:
        target_prompt_batch_size = int(self.config.data.train_batch_size)
        replay_prompt_count = min(
            self._offpolicy_replay_sample_prompt_batch_size(), len(self._offpolicy_replay_buffer)
        )
        current_prompt_count = target_prompt_batch_size - replay_prompt_count
        if current_prompt_count <= 0:
            raise ValueError(
                f"offpolicy_replay current prompt count must be positive; "
                f"target={target_prompt_batch_size}, replay={replay_prompt_count}"
            )
        return current_prompt_count

    def _take_current_prompt_batch_for_offpolicy_replay(self, batch: DataProto, train_iter) -> DataProto:
        if not self._offpolicy_replay_enabled():
            return batch

        target_prompt_count = self._offpolicy_replay_target_current_prompt_batch_size()
        if self._offpolicy_pending_prompt_batch is not None:
            batch = DataProto.concat([self._offpolicy_pending_prompt_batch, batch])
            self._offpolicy_pending_prompt_batch = None

        while len(batch) < target_prompt_count:
            try:
                extra_batch = DataProto.from_single_dict(next(train_iter))
            except StopIteration:
                break
            batch = DataProto.concat([batch, extra_batch])

        if len(batch) <= target_prompt_count:
            return batch

        current_batch = batch[:target_prompt_count]
        self._offpolicy_pending_prompt_batch = batch[target_prompt_count:]
        return current_batch

    def _clone_replay_group_for_storage(self, group: DataProto) -> DataProto:
        stored_group = deepcopy(group)
        stored_group.meta_info = {}
        if stored_group.batch is not None:
            for key in list(stored_group.batch.keys()):
                stored_group.batch[key] = stored_group.batch[key].detach().to("cpu")
        return stored_group

    def _mark_offpolicy_replay_rows(
        self, batch: DataProto, *, is_replay: bool, policy_step: Optional[int] = None
    ) -> None:
        row_count = len(batch)
        batch.non_tensor_batch["offpolicy_replay_is_replay"] = np.full(row_count, is_replay, dtype=bool)
        if policy_step is not None:
            batch.non_tensor_batch["offpolicy_replay_policy_step"] = np.full(row_count, policy_step, dtype=np.int64)
        elif "offpolicy_replay_policy_step" not in batch.non_tensor_batch:
            batch.non_tensor_batch["offpolicy_replay_policy_step"] = np.full(row_count, -1, dtype=np.int64)

    def _offpolicy_replay_prompt_group_hits_budget(
        self, current_batch: DataProto, start: int, end: int
    ) -> bool:
        if not self._offpolicy_replay_skip_hit_budget():
            return False
        if "response_mask" not in current_batch.batch:
            return False

        response_mask = current_batch.batch["response_mask"][start:end]
        if response_mask.numel() == 0:
            return False

        max_response_length = response_mask.shape[-1]
        response_lengths = response_mask.sum(dim=-1)
        return bool(torch.any(response_lengths >= max_response_length).item())

    def _append_current_batch_to_offpolicy_replay_buffer(self, current_batch: DataProto) -> tuple[int, int]:
        if self._offpolicy_replay_buffer.maxlen == 0:
            return 0, 0

        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        if rollout_n <= 0:
            raise ValueError(f"actor_rollout_ref.rollout.n must be positive, got {rollout_n}")
        if len(current_batch) % rollout_n != 0:
            raise ValueError(
                f"Cannot split current rollout batch of {len(current_batch)} rows into prompt groups of n={rollout_n}"
            )

        prompt_group_count = len(current_batch) // rollout_n
        added_prompt_count = 0
        skipped_hit_budget_prompt_count = 0
        for prompt_idx in range(prompt_group_count):
            start = prompt_idx * rollout_n
            end = start + rollout_n
            if self._offpolicy_replay_prompt_group_hits_budget(current_batch, start, end):
                skipped_hit_budget_prompt_count += 1
                continue
            self._offpolicy_replay_buffer.append(self._clone_replay_group_for_storage(current_batch[start:end]))
            added_prompt_count += 1
        return added_prompt_count, skipped_hit_budget_prompt_count

    def _sample_offpolicy_replay_batch(self, prompt_count: int, device: torch.device) -> DataProto | None:
        if prompt_count <= 0 or len(self._offpolicy_replay_buffer) == 0:
            return None

        sample_count = min(prompt_count, len(self._offpolicy_replay_buffer))
        indices = self._offpolicy_replay_rng.choice(
            len(self._offpolicy_replay_buffer), size=sample_count, replace=False
        )
        replay_groups = [deepcopy(self._offpolicy_replay_buffer[int(index)]) for index in indices]
        replay_batch = DataProto.concat(replay_groups)
        replay_batch.to(device)
        self._mark_offpolicy_replay_rows(replay_batch, is_replay=True)
        return replay_batch

    def _mix_offpolicy_replay_batch(self, current_batch: DataProto, metrics: dict) -> DataProto:
        if not self._offpolicy_replay_enabled():
            return current_batch

        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        if len(current_batch) % rollout_n != 0:
            raise ValueError(
                f"Cannot use prompt-level off-policy replay with batch rows={len(current_batch)} and n={rollout_n}"
            )

        current_prompt_count = len(current_batch) // rollout_n
        requested_replay_prompts = self._offpolicy_replay_sample_prompt_batch_size()
        available_replay_prompts = len(self._offpolicy_replay_buffer)
        first_tensor = next(iter(current_batch.batch.values()))
        current_device = first_tensor.device

        self._mark_offpolicy_replay_rows(current_batch, is_replay=False, policy_step=self.global_steps)
        current_meta_info = current_batch.meta_info
        current_batch.meta_info = {}

        replay_batch = self._sample_offpolicy_replay_batch(requested_replay_prompts, current_device)
        replay_prompt_count = 0 if replay_batch is None else len(replay_batch) // rollout_n
        mixed_batch = current_batch if replay_batch is None else DataProto.concat([current_batch, replay_batch])
        mixed_batch.meta_info = current_meta_info

        added_prompt_count, skipped_hit_budget_prompt_count = self._append_current_batch_to_offpolicy_replay_buffer(
            current_batch
        )

        total_prompt_count = current_prompt_count + replay_prompt_count
        metrics.update(
            {
                "offpolicy_replay/current_prompts": current_prompt_count,
                "offpolicy_replay/replay_prompts": replay_prompt_count,
                "offpolicy_replay/replay_rows": replay_prompt_count * rollout_n,
                "offpolicy_replay/requested_replay_prompts": requested_replay_prompts,
                "offpolicy_replay/available_prompts_before_sample": available_replay_prompts,
                "offpolicy_replay/added_prompts": added_prompt_count,
                "offpolicy_replay/skipped_hit_budget_prompts": skipped_hit_budget_prompt_count,
                "offpolicy_replay/skipped_hit_budget_rows": skipped_hit_budget_prompt_count * rollout_n,
                "offpolicy_replay/buffer_prompts": len(self._offpolicy_replay_buffer),
                "offpolicy_replay/replay_prompt_fraction": (
                    replay_prompt_count / total_prompt_count if total_prompt_count > 0 else 0.0
                ),
            }
        )

        return mixed_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    @staticmethod
    def _replace_batch_tensors(batch: DataProto, updates: DataProto, keys: list[str] | None = None) -> None:
        if keys is None:
            keys = list(updates.batch.keys())
        for key in keys:
            if key in updates.batch.keys():
                batch.batch[key] = updates.batch[key]

    @staticmethod
    def _masked_mean_abs_log_ratio(
        current_log_probs: torch.Tensor,
        rollout_old_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        current_log_probs = current_log_probs.detach()
        rollout_old_log_probs = rollout_old_log_probs.to(
            device=current_log_probs.device,
            dtype=current_log_probs.dtype,
        )
        response_mask = response_mask.to(device=current_log_probs.device, dtype=current_log_probs.dtype)
        denom = response_mask.sum().clamp_min(1.0)
        return ((current_log_probs - rollout_old_log_probs).abs() * response_mask).sum() / denom

    @staticmethod
    def _adaptive_ppo_rkl_mask(batch: DataProto) -> torch.Tensor:
        response_mask = batch.batch["response_mask"]
        if "opd_proximal_mask" in batch.batch.keys():
            response_mask = batch.batch["opd_proximal_mask"].to(
                device=response_mask.device,
                dtype=response_mask.dtype,
            )
        if "format_mask" in batch.batch.keys():
            response_mask = response_mask * batch.batch["format_mask"].to(
                device=response_mask.device,
                dtype=response_mask.dtype,
            ).unsqueeze(-1)
        return response_mask

    @staticmethod
    def _adaptive_ppo_threshold_reason(
        *,
        updates_used: int,
        min_updates: int,
        mean_abs_log_ratio: float,
        abs_log_ratio_threshold: float | None,
        sampled_token_rkl_k3: float,
        sampled_token_rkl_k3_threshold: float | None,
    ) -> str | None:
        if updates_used < min_updates:
            return None
        reasons = []
        if abs_log_ratio_threshold is not None and mean_abs_log_ratio > abs_log_ratio_threshold:
            reasons.append("abs_log_ratio")
        if (
            sampled_token_rkl_k3_threshold is not None
            and sampled_token_rkl_k3 > sampled_token_rkl_k3_threshold
        ):
            reasons.append("sampled_token_rkl_k3")
        return "+".join(reasons) or None

    def _update_prefix_drift_for_current_log_probs(
        self,
        batch: DataProto,
        metrics: dict,
        update_idx: int,
    ) -> None:
        if not self._prefix_drift_enabled():
            if PREFIX_DRIFT_WEIGHTS_KEY in batch.batch.keys():
                batch.batch.pop(PREFIX_DRIFT_WEIGHTS_KEY)
            if PREFIX_DRIFT_RAW_WEIGHTS_KEY in batch.batch.keys():
                batch.batch.pop(PREFIX_DRIFT_RAW_WEIGHTS_KEY)
            return

        if PREFIX_DRIFT_REFERENCE_LOG_PROBS_KEY not in batch.batch.keys():
            raise ValueError(
                "prefix_drift_enable=True requires prefix drift reference log-probs. "
                "The trainer should clone old_log_probs before repeated actor updates."
            )
        if "old_log_probs" not in batch.batch.keys():
            raise ValueError("prefix_drift_enable=True requires current old_log_probs in the batch.")

        rollout_config = self.config.actor_rollout_ref.rollout
        method = str(rollout_config.get("prefix_drift_method", "diagnostic"))
        ess_target_fraction = self._config_optional_float(
            rollout_config.get("prefix_drift_ess_target_fraction", 0.5),
            "prefix_drift_ess_target_fraction",
        )
        ess_bisection_steps = self._config_optional_int(
            rollout_config.get("prefix_drift_ess_bisection_steps", 8),
            "prefix_drift_ess_bisection_steps",
        )
        log_clip = self._config_optional_float(
            rollout_config.get("prefix_drift_log_clip", 3.0),
            "prefix_drift_log_clip",
        )
        log_clip_mode = str(rollout_config.get("prefix_drift_log_clip_mode", "symmetric"))
        ctpo_log_clip_base = self._config_optional_float(
            rollout_config.get("prefix_drift_ctpo_log_clip_base", None),
            "prefix_drift_ctpo_log_clip_base",
        )
        ctpo_log_clip_lower_base = self._config_optional_float(
            rollout_config.get("prefix_drift_ctpo_log_clip_lower_base", None),
            "prefix_drift_ctpo_log_clip_lower_base",
        )
        ctpo_log_clip_upper_base = self._config_optional_float(
            rollout_config.get("prefix_drift_ctpo_log_clip_upper_base", None),
            "prefix_drift_ctpo_log_clip_upper_base",
        )
        ctpo_log_clip_power = self._config_optional_float(
            rollout_config.get("prefix_drift_ctpo_log_clip_power", 0.5),
            "prefix_drift_ctpo_log_clip_power",
        )
        normalize = str(rollout_config.get("prefix_drift_normalize", "none"))
        position_beta = self._config_optional_float(
            rollout_config.get("prefix_drift_position_beta", 0.0),
            "prefix_drift_position_beta",
        )
        position_hmax = self._config_optional_int(
            rollout_config.get("prefix_drift_position_hmax", None),
            "prefix_drift_position_hmax",
        )
        if position_hmax is None:
            position_hmax = self._config_optional_int(
                rollout_config.get("response_length", None),
                "actor_rollout_ref.rollout.response_length",
            )
        detach = self._config_bool(rollout_config.get("prefix_drift_detach", True))

        if self._opd_q_prefix_samplek_enabled() and self._opd_q_mixture_source_behavior_prefix_enabled():
            required_keys = {
                OPD_ROLLOUT_REFERENCE_LOG_PROBS_KEY,
                OPD_Q_MIXTURE_TEACHER_LOG_PROBS_KEY,
                "opd_proximal_mask",
            }
            missing_keys = sorted(required_keys.difference(batch.batch.keys()))
            if missing_keys:
                raise ValueError(
                    "source-behavior Q-prefix drift is missing required tensors: "
                    f"{missing_keys}"
                )
            result = compute_source_specific_prefix_drift(
                current_log_probs=batch.batch["old_log_probs"],
                student_reference_log_probs=batch.batch[OPD_ROLLOUT_REFERENCE_LOG_PROBS_KEY],
                teacher_reference_log_probs=batch.batch[OPD_Q_MIXTURE_TEACHER_LOG_PROBS_KEY],
                response_mask=batch.batch["response_mask"],
                student_token_mask=batch.batch["opd_proximal_mask"],
                student_method=method,
                teacher_method=self._opd_q_mixture_teacher_prefix_method(),
                log_clip=log_clip,
                log_clip_mode=log_clip_mode,
                teacher_log_clip=self._opd_q_mixture_teacher_prefix_log_clip(),
                teacher_log_clip_mode=self._opd_q_mixture_teacher_prefix_log_clip_mode(),
                teacher_min_weight=self._opd_q_mixture_teacher_prefix_min_weight(),
                ctpo_log_clip_base=ctpo_log_clip_base,
                ctpo_log_clip_lower_base=ctpo_log_clip_lower_base,
                ctpo_log_clip_upper_base=ctpo_log_clip_upper_base,
                ctpo_log_clip_power=0.5 if ctpo_log_clip_power is None else ctpo_log_clip_power,
                normalize=normalize,
                position_beta=position_beta or 0.0,
                position_hmax=position_hmax,
                detach=detach,
                metric_prefix="prefix_drift",
                ess_target_fraction=0.5 if ess_target_fraction is None else ess_target_fraction,
                ess_bisection_steps=8 if ess_bisection_steps is None else ess_bisection_steps,
            )
            metrics["opd_q_mixture/prefix_reference_is_source_behavior"] = 1.0
        else:
            result = compute_prefix_drift(
                current_log_probs=batch.batch["old_log_probs"],
                reference_log_probs=batch.batch[PREFIX_DRIFT_REFERENCE_LOG_PROBS_KEY],
                response_mask=batch.batch["response_mask"],
                method=method,
                log_clip=log_clip,
                log_clip_mode=log_clip_mode,
                ctpo_log_clip_base=ctpo_log_clip_base,
                ctpo_log_clip_lower_base=ctpo_log_clip_lower_base,
                ctpo_log_clip_upper_base=ctpo_log_clip_upper_base,
                ctpo_log_clip_power=0.5 if ctpo_log_clip_power is None else ctpo_log_clip_power,
                normalize=normalize,
                position_beta=position_beta or 0.0,
                position_hmax=position_hmax,
                detach=detach,
                metric_prefix="prefix_drift",
                ess_target_fraction=0.5 if ess_target_fraction is None else ess_target_fraction,
                ess_bisection_steps=8 if ess_bisection_steps is None else ess_bisection_steps,
            )
            metrics["opd_q_mixture/prefix_reference_is_source_behavior"] = 0.0
        metrics.update(result.metrics)
        for key, value in result.metrics.items():
            if key.startswith("prefix_drift/"):
                metrics[f"prefix_drift/update_{update_idx}/{key[len('prefix_drift/') :]}"] = value

        if result.weights is None:
            if PREFIX_DRIFT_WEIGHTS_KEY in batch.batch.keys():
                batch.batch.pop(PREFIX_DRIFT_WEIGHTS_KEY)
            if PREFIX_DRIFT_RAW_WEIGHTS_KEY in batch.batch.keys():
                batch.batch.pop(PREFIX_DRIFT_RAW_WEIGHTS_KEY)
        else:
            batch.batch[PREFIX_DRIFT_WEIGHTS_KEY] = result.weights
            if self._opd_q_mixture_diagnostics_enabled() and result.raw_weights is not None:
                batch.batch[PREFIX_DRIFT_RAW_WEIGHTS_KEY] = result.raw_weights
            elif PREFIX_DRIFT_RAW_WEIGHTS_KEY in batch.batch.keys():
                batch.batch.pop(PREFIX_DRIFT_RAW_WEIGHTS_KEY)

    @staticmethod
    def _masked_samplek_normalized_ess(
        current_log_probs: torch.Tensor,
        sample_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_log_probs = current_log_probs.detach().float()
        sample_log_probs = sample_log_probs.to(device=current_log_probs.device).detach().float()
        response_mask = response_mask.to(device=current_log_probs.device, dtype=current_log_probs.dtype)

        log_weights = (current_log_probs - sample_log_probs).clamp(min=-20.0, max=20.0)
        weights = torch.exp(log_weights)
        weights_sum = weights.sum(dim=-1)
        weights_sq_sum = weights.square().sum(dim=-1).clamp_min(1e-12)
        normalized_ess = (weights_sum.square() / (weights_sq_sum * weights.size(-1))).clamp(min=0.0, max=1.0)

        valid_mask = response_mask > 0.5
        denom = response_mask.sum().clamp_min(1.0)
        mean_ess = (normalized_ess * response_mask).sum() / denom
        if valid_mask.any():
            valid_ess = normalized_ess[valid_mask]
            min_ess = valid_ess.min()
        else:
            min_ess = mean_ess
        return mean_ess, min_ess, normalized_ess

    def _recompute_reward_and_advantage_for_adaptive_ppo(self, batch: DataProto, metrics: dict) -> None:
        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
        self._attach_reward_extra_batch_tensors(batch, reward_extra_infos_dict, metrics)

        batch.batch["token_level_scores"] = reward_tensor
        if "true_reward_score" in reward_extra_infos_dict:
            true_reward_val = reward_extra_infos_dict["true_reward_score"]
            if isinstance(true_reward_val, torch.Tensor):
                batch.batch["true_reward_score"] = true_reward_val
            else:
                batch.batch["true_reward_score"] = torch.as_tensor(
                    true_reward_val,
                    device=reward_tensor.device,
                    dtype=reward_tensor.dtype,
                )
        else:
            batch.batch["true_reward_score"] = reward_tensor

        if reward_extra_infos_dict:
            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

        if self.config.algorithm.use_kl_in_reward:
            batch, kl_metrics = apply_kl_penalty(
                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
            metrics.update(kl_metrics)
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
        compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self._batch_num_repeat(batch),
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=self.config.algorithm,
        )

    @staticmethod
    def _record_adaptive_samplek_probe_metrics(
        *,
        batch: DataProto,
        current_candidate_log_probs: torch.Tensor,
        metrics: dict,
        update_idx: int,
    ) -> float:
        probe_metrics = compute_samplek_probe_diagnostics(
            current_log_probs=current_candidate_log_probs,
            reference_log_probs=batch.batch[ADAPTIVE_PPO_PROBE_REFERENCE_LOG_PROBS_KEY],
            response_mask=batch.batch["response_mask"],
            teacher_log_probs=batch.batch[ADAPTIVE_PPO_PROBE_TEACHER_LOG_PROBS_KEY],
        )
        for key, value in probe_metrics.items():
            metrics[f"adaptive_ppo/{key}"] = value
            metrics[f"adaptive_ppo/update_{update_idx}/{key}"] = value
        return probe_metrics["candidate_probe/teacher_rkl_is_proxy"]

    def _refresh_batch_for_adaptive_ppo_update(
        self,
        batch: DataProto,
        metrics: dict,
        timing_raw: dict,
        update_idx: int,
        min_updates: int,
        abs_log_ratio_threshold: float | None = None,
        sampled_token_rkl_k3_threshold: float | None = None,
    ) -> tuple[float, float, str | None]:
        with marked_timer("adaptive_ppo_compute_log_prob", timing_raw, color="blue"):
            current_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            self._replace_batch_tensors(batch, current_log_prob)

        mean_abs_log_ratio = self._masked_mean_abs_log_ratio(
            current_log_probs=batch.batch["old_log_probs"],
            rollout_old_log_probs=batch.batch["adaptive_ppo_rollout_old_log_probs"],
            response_mask=batch.batch["response_mask"],
        )
        mean_abs_log_ratio_value = mean_abs_log_ratio.detach().item()
        metrics["adaptive_ppo/mean_abs_log_ratio"] = mean_abs_log_ratio_value
        metrics["adaptive_ppo/batch_mean_abs_log_ratio"] = mean_abs_log_ratio_value
        metrics[f"adaptive_ppo/update_{update_idx}/mean_abs_log_ratio"] = mean_abs_log_ratio_value
        metrics[f"adaptive_ppo/update_{update_idx}/batch_mean_abs_log_ratio"] = mean_abs_log_ratio_value

        rkl_diagnostics = compute_sampled_token_rkl_diagnostics(
            current_log_probs=batch.batch["old_log_probs"],
            reference_log_probs=batch.batch["adaptive_ppo_rollout_old_log_probs"],
            response_mask=self._adaptive_ppo_rkl_mask(batch),
        )
        sampled_token_rkl_k3_value = rkl_diagnostics.k3_mean.detach().item()
        metrics["adaptive_ppo/sampled_token_rkl_k3_mean"] = sampled_token_rkl_k3_value
        metrics["adaptive_ppo/batch_sampled_token_rkl_k3_mean"] = sampled_token_rkl_k3_value
        metrics[f"adaptive_ppo/update_{update_idx}/sampled_token_rkl_k3_mean"] = (
            sampled_token_rkl_k3_value
        )
        metrics[f"adaptive_ppo/update_{update_idx}/batch_sampled_token_rkl_k3_mean"] = (
            sampled_token_rkl_k3_value
        )
        threshold_eligible = float(update_idx >= min_updates)
        metrics["adaptive_ppo/threshold_eligible"] = threshold_eligible
        metrics[f"adaptive_ppo/update_{update_idx}/threshold_eligible"] = threshold_eligible
        if sampled_token_rkl_k3_threshold is not None:
            k3_threshold_margin = sampled_token_rkl_k3_value - sampled_token_rkl_k3_threshold
            metrics["adaptive_ppo/sampled_token_rkl_k3_threshold_margin"] = k3_threshold_margin
            metrics[f"adaptive_ppo/update_{update_idx}/sampled_token_rkl_k3_threshold_margin"] = (
                k3_threshold_margin
            )
            if sampled_token_rkl_k3_threshold > 0.0:
                k3_to_threshold = sampled_token_rkl_k3_value / sampled_token_rkl_k3_threshold
                metrics["adaptive_ppo/sampled_token_rkl_k3_to_threshold"] = k3_to_threshold
                metrics[f"adaptive_ppo/update_{update_idx}/sampled_token_rkl_k3_to_threshold"] = (
                    k3_to_threshold
                )
        self._update_prefix_drift_for_current_log_probs(
            batch=batch,
            metrics=metrics,
            update_idx=update_idx,
        )

        threshold_reason = self._adaptive_ppo_threshold_reason(
            updates_used=update_idx,
            min_updates=min_updates,
            mean_abs_log_ratio=mean_abs_log_ratio_value,
            abs_log_ratio_threshold=abs_log_ratio_threshold,
            sampled_token_rkl_k3=sampled_token_rkl_k3_value,
            sampled_token_rkl_k3_threshold=sampled_token_rkl_k3_threshold,
        )
        if threshold_reason is not None:
            return mean_abs_log_ratio_value, sampled_token_rkl_k3_value, threshold_reason

        batch.meta_info["global_steps"] = self.global_steps
        batch.meta_info["is_plot"] = False
        batch.meta_info["return_teacher_topk_metrics"] = False
        with marked_timer("adaptive_ppo_compute_rm_score", timing_raw, color="magenta"):
            teacher_data = self.rm_wg.compute_rm_score(batch)
            self._replace_batch_tensors(batch, teacher_data)

        top_k = int(
            batch.meta_info.get(
                "log_prob_top_k",
                self.config.actor_rollout_ref.rollout.get("log_prob_top_k", 0),
            )
        )
        if top_k > 0:
            with marked_timer("adaptive_ppo_compute_distillation_reward", timing_raw, color="orange"):
                distillation_output = self.actor_rollout_wg.compute_distillation_reward(batch)
                self._replace_batch_tensors(batch, distillation_output)

        self._recompute_reward_and_advantage_for_adaptive_ppo(batch, metrics)
        return mean_abs_log_ratio_value, sampled_token_rkl_k3_value, None

    def _run_adaptive_ppo_actor_updates(self, batch: DataProto, metrics: dict, timing_raw: dict) -> None:
        rollout_config = self.config.actor_rollout_ref.rollout
        actor_config = self.config.actor_rollout_ref.actor

        actor_ppo_epochs = int(actor_config.get("ppo_epochs", 1))
        if actor_ppo_epochs != 1:
            raise ValueError(
                "adaptive_ppo_update_enable=True expects actor_rollout_ref.actor.ppo_epochs=1. "
                "Use adaptive_ppo_update_max_updates to control repeated updates."
            )
        if not self.use_rm:
            raise ValueError("adaptive_ppo_update_enable=True currently requires reward_model.enable=True.")

        candidate_mode = str(
            batch.meta_info.get(
                "log_prob_candidate_mode",
                rollout_config.get("log_prob_candidate_mode", "topk"),
            )
        ).strip().lower().replace("-", "_")
        candidate_mode_aliases = {
            "sample": "sample_stu",
            "sample_student": "sample_stu",
            "student_sample": "sample_stu",
            "sample_k": "sample_stu",
        }
        candidate_mode = candidate_mode_aliases.get(candidate_mode, candidate_mode)
        top_k = int(batch.meta_info.get("log_prob_top_k", rollout_config.get("log_prob_top_k", 0)))
        if top_k <= 0 or candidate_mode not in {"sample_stu", "adaptive_head_tail", "topk"}:
            raise ValueError(
                "adaptive_ppo_update_enable=True is currently scoped to fixed candidate OPD "
                "(log_prob_candidate_mode=topk|sample_stu|adaptive_head_tail and log_prob_top_k>0)."
            )

        max_updates = int(rollout_config.get("adaptive_ppo_update_max_updates", 1))
        if max_updates < 1:
            raise ValueError("adaptive_ppo_update_max_updates must be >= 1.")
        min_updates = int(rollout_config.get("adaptive_ppo_update_min_updates", 1))
        if min_updates < 1 or min_updates > max_updates:
            raise ValueError(
                "adaptive_ppo_update_min_updates must be between 1 and "
                "adaptive_ppo_update_max_updates."
            )
        abs_log_ratio_threshold = self._config_optional_float(
            rollout_config.get("adaptive_ppo_update_abs_log_ratio_threshold", None),
            "adaptive_ppo_update_abs_log_ratio_threshold",
        )
        sampled_token_rkl_k3_threshold = self._config_optional_float(
            rollout_config.get("adaptive_ppo_update_sampled_token_rkl_k3_threshold", None),
            "adaptive_ppo_update_sampled_token_rkl_k3_threshold",
        )
        for name, value in (
            ("adaptive_ppo_update_abs_log_ratio_threshold", abs_log_ratio_threshold),
            ("adaptive_ppo_update_sampled_token_rkl_k3_threshold", sampled_token_rkl_k3_threshold),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be null or a finite nonnegative value.")
        no_candidate_is = self._config_bool(rollout_config.get("adaptive_ppo_update_no_candidate_is", True))
        loo_variance_ratio_diagnostics_enable = self._config_bool(
            rollout_config.get("opd_samplek_loo_variance_ratio_diagnostics_enable", False)
        )
        loo_variance_histogram_diagnostics_enable = self._config_bool(
            rollout_config.get("opd_samplek_loo_variance_histogram_diagnostics_enable", False)
        )
        loo_variance_filter_threshold = self._config_optional_float(
            rollout_config.get("opd_samplek_loo_variance_filter_threshold", None),
            "opd_samplek_loo_variance_filter_threshold",
        )
        loo_variance_filter_threshold_mode = normalize_samplek_loo_variance_threshold_mode(
            rollout_config.get("opd_samplek_loo_variance_filter_threshold_mode", "fixed")
        )
        loo_variance_filter_quantile = float(
            rollout_config.get("opd_samplek_loo_variance_filter_quantile", 0.7)
        )
        if loo_variance_filter_threshold_mode == "update0_quantile":
            if loo_variance_filter_threshold is not None:
                raise ValueError(
                    "update0_quantile threshold mode requires "
                    "opd_samplek_loo_variance_filter_threshold=null."
                )
            if not math.isfinite(loo_variance_filter_quantile) or not (
                0.0 < loo_variance_filter_quantile < 1.0
            ):
                raise ValueError(
                    "opd_samplek_loo_variance_filter_quantile must be finite and in (0, 1)."
                )
        loo_advantage_centering = batch.meta_info.get(
            "opd_samplek_advantage_centering",
            rollout_config.get("opd_samplek_advantage_centering", "none"),
        )
        if loo_variance_ratio_diagnostics_enable:
            if (
                loo_variance_filter_threshold_mode == "fixed"
                and loo_variance_filter_threshold is None
            ):
                raise ValueError(
                    "LOO variance ratio diagnostics require "
                    "opd_samplek_loo_variance_filter_threshold."
                )
            if str(loo_advantage_centering).strip().lower().replace("-", "_") not in {
                "loo",
                "leave_one_out",
                "leave_one_out_baseline",
            }:
                raise ValueError("LOO variance ratio diagnostics require leave_one_out centering.")
        candidate_reuse_enabled = self._samplek_candidate_reuse_enabled()
        candidate_refresh_interval = rollout_config.get("samplek_candidate_refresh_interval", 5)
        diagnostic_forced_eos_enable = self._config_bool(
            rollout_config.get("opd_diagnostic_forced_eos_enable", False)
        )
        terminal_aware_enable = self._config_bool(
            rollout_config.get("opd_terminal_aware_enable", False)
        )
        terminal_objective_mode = rollout_config.get("opd_terminal_objective_mode", "anchor_kl")
        terminal_anchor_mode = rollout_config.get("opd_terminal_anchor_mode", "behavior")
        terminal_gate_coef = float(rollout_config.get("opd_terminal_gate_coef", 1.0))
        terminal_secondary_token_id = self._config_optional_int(
            rollout_config.get("opd_terminal_secondary_token_id", None),
            "opd_terminal_secondary_token_id",
        )
        terminal_teacher_remap_enable = self._config_bool(
            rollout_config.get("opd_terminal_teacher_remap_enable", False)
        )
        terminal_kl_baseline_mode = rollout_config.get(
            "opd_terminal_kl_baseline_mode", "mc_loo"
        )
        terminal_teacher_remap_floor = float(
            rollout_config.get("opd_terminal_teacher_remap_floor", 1e-18)
        )
        terminal_topm = int(rollout_config.get("opd_terminal_topm", 0))
        eos_future_enable = self._config_bool(rollout_config.get("opd_eos_future_enable", False))
        eos_future_mode = (
            normalize_eos_future_mode(rollout_config.get("opd_eos_future_mode", "fixed_signed"))
            if eos_future_enable
            else EOS_FUTURE_MODE_FIXED_SIGNED
        )
        eos_future_horizon = int(rollout_config.get("opd_eos_future_horizon", 32))
        eos_future_coef = float(rollout_config.get("opd_eos_future_coef", 1.0))
        validate_samplek_candidate_reuse_configuration(
            enabled=candidate_reuse_enabled,
            refresh_interval=candidate_refresh_interval,
            adaptive_update_enable=self._adaptive_ppo_update_enabled(),
            candidate_mode=candidate_mode,
            candidate_count=top_k,
            no_candidate_is=no_candidate_is,
            abs_log_ratio_threshold=abs_log_ratio_threshold,
            sampled_token_rkl_k3_threshold=sampled_token_rkl_k3_threshold,
            q_mixture_enable=self._opd_q_mixture_enabled(),
            forced_eos_diagnostic_enable=diagnostic_forced_eos_enable,
            eos_future_enable=eos_future_enable,
        )
        candidate_refresh_interval = (
            int(candidate_refresh_interval) if candidate_reuse_enabled else 5
        )
        validate_forced_eos_diagnostic_configuration(
            enabled=diagnostic_forced_eos_enable,
            candidate_mode=candidate_mode,
            sample_replacement=self._config_bool(rollout_config.get("sample_k_replacement", True)),
            top_k=top_k,
            top_k_strategy=batch.meta_info.get(
                "top_k_strategy", rollout_config.get("top_k_strategy", "only_stu")
            ),
            opd_loss_type=batch.meta_info.get(
                "opd_loss_type", rollout_config.get("opd_loss_type", "sample_k_reverse_kl")
            ),
            advantage_mode=batch.meta_info.get(
                "opd_advantage_mode", rollout_config.get("opd_advantage_mode", "fixed")
            ),
            candidate_aggregation=batch.meta_info.get(
                "opd_samplek_candidate_aggregation",
                rollout_config.get("opd_samplek_candidate_aggregation", "sum"),
            ),
            advantage_centering=batch.meta_info.get(
                "opd_samplek_advantage_centering",
                rollout_config.get("opd_samplek_advantage_centering", "none"),
            ),
            adaptive_update_enable=self._adaptive_ppo_update_enabled(),
            no_candidate_is=no_candidate_is,
            q_source_normalize_enable=self._opd_q_mixture_source_normalize_enabled(),
            generic_candidate_weights_present="candidate_estimator_weights" in batch.batch.keys(),
        )
        validate_terminal_aware_configuration(
            enabled=terminal_aware_enable,
            objective_mode=terminal_objective_mode,
            anchor_mode=terminal_anchor_mode,
            gate_coef=terminal_gate_coef,
            candidate_mode=candidate_mode,
            sample_replacement=self._config_bool(rollout_config.get("sample_k_replacement", True)),
            top_k=top_k,
            top_k_strategy=batch.meta_info.get(
                "top_k_strategy", rollout_config.get("top_k_strategy", "only_stu")
            ),
            opd_loss_type=batch.meta_info.get(
                "opd_loss_type", rollout_config.get("opd_loss_type", "sample_k_reverse_kl")
            ),
            advantage_mode=batch.meta_info.get(
                "opd_advantage_mode", rollout_config.get("opd_advantage_mode", "fixed")
            ),
            candidate_aggregation=batch.meta_info.get(
                "opd_samplek_candidate_aggregation",
                rollout_config.get("opd_samplek_candidate_aggregation", "sum"),
            ),
            advantage_centering=batch.meta_info.get(
                "opd_samplek_advantage_centering",
                rollout_config.get("opd_samplek_advantage_centering", "none"),
            ),
            adaptive_update_enable=self._adaptive_ppo_update_enabled(),
            no_candidate_is=no_candidate_is,
            forced_eos_diagnostic_enable=diagnostic_forced_eos_enable,
            q_mixture_enable=self._opd_q_mixture_enabled(),
            q_source_normalize_enable=self._opd_q_mixture_source_normalize_enabled(),
            samplek_entropy_coef=float(rollout_config.get("opd_samplek_entropy_coef", 0.0)),
            influence_clip=self._config_optional_float(
                rollout_config.get("opd_samplek_influence_clip", None),
                "opd_samplek_influence_clip",
            ),
            raw_advantage_clip=self._config_optional_float(
                rollout_config.get("opd_raw_advantage_clip", None),
                "opd_raw_advantage_clip",
            ),
            generic_candidate_weights_present="candidate_estimator_weights" in batch.batch.keys(),
            secondary_token_id=terminal_secondary_token_id,
            teacher_remap_enable=terminal_teacher_remap_enable,
            kl_baseline_mode=terminal_kl_baseline_mode,
            teacher_remap_floor=terminal_teacher_remap_floor,
            terminal_topm=terminal_topm,
            subtract_score_baseline=self._config_bool(
                batch.meta_info.get(
                    "sample_k_kl_plus_one",
                    rollout_config.get("sample_k_kl_plus_one", True),
                )
            ),
            candidate_reuse_enabled=candidate_reuse_enabled,
        )
        validate_eos_future_correction_configuration(
            enabled=eos_future_enable,
            mode=eos_future_mode,
            horizon=eos_future_horizon,
            coefficient=eos_future_coef,
            candidate_mode=candidate_mode,
            sample_replacement=self._config_bool(rollout_config.get("sample_k_replacement", True)),
            top_k=top_k,
            opd_loss_type=batch.meta_info.get(
                "opd_loss_type", rollout_config.get("opd_loss_type", "sample_k_reverse_kl")
            ),
            advantage_mode=batch.meta_info.get(
                "opd_advantage_mode", rollout_config.get("opd_advantage_mode", "fixed")
            ),
            candidate_aggregation=batch.meta_info.get(
                "opd_samplek_candidate_aggregation",
                rollout_config.get("opd_samplek_candidate_aggregation", "sum"),
            ),
            advantage_centering=batch.meta_info.get(
                "opd_samplek_advantage_centering",
                rollout_config.get("opd_samplek_advantage_centering", "none"),
            ),
            adaptive_update_enable=self._adaptive_ppo_update_enabled(),
            no_candidate_is=no_candidate_is,
            sample_k_kl_plus_one=self._config_bool(
                batch.meta_info.get(
                    "sample_k_kl_plus_one",
                    rollout_config.get("sample_k_kl_plus_one", True),
                )
            ),
            prefix_drift_enable=self._prefix_drift_enabled(),
            forced_eos_diagnostic_enable=diagnostic_forced_eos_enable,
            terminal_aware_enable=terminal_aware_enable,
            raw_advantage_clip=self._config_optional_float(
                rollout_config.get("opd_raw_advantage_clip", None),
                "opd_raw_advantage_clip",
            ),
            influence_clip=self._config_optional_float(
                rollout_config.get("opd_samplek_influence_clip", None),
                "opd_samplek_influence_clip",
            ),
            terminal_objective_mode=terminal_objective_mode,
            terminal_teacher_remap_enable=terminal_teacher_remap_enable,
        )
        if diagnostic_forced_eos_enable and FORCED_EOS_ESTIMATOR_WEIGHTS_KEY not in batch.batch.keys():
            raise ValueError("forced-EOS diagnostic is missing its dedicated estimator weights.")
        samplek_probe_enabled = self._config_bool(
            rollout_config.get("adaptive_ppo_update_samplek_probe_enable", False)
        )
        if samplek_probe_enabled and candidate_mode != "sample_stu":
            raise ValueError(
                "adaptive_ppo_update_samplek_probe_enable=True currently requires "
                "log_prob_candidate_mode=sample_stu."
            )

        if "old_log_probs" not in batch.batch.keys():
            raise ValueError("adaptive PPO update requires old_log_probs in the rollout batch.")
        if terminal_aware_enable:
            for required_key in ("student_top_k_ids", "student_top_k_log_probs"):
                if required_key not in batch.batch.keys():
                    raise ValueError(f"terminal-aware OPD is missing {required_key}.")
            eos_token_id = self.tokenizer.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None:
                raise ValueError("terminal-aware OPD requires an EOS token id.")
            terminal_mask = batch.batch["response_mask"].bool() & batch.batch["responses"].eq(
                int(eos_token_id)
            )
            candidate_ids = batch.batch["student_top_k_ids"]
            normalized_terminal_objective_mode = str(terminal_objective_mode).strip().lower().replace(
                "-", "_"
            )
            if terminal_mask.any() and normalized_terminal_objective_mode != "teacher_remap_only":
                if not candidate_ids[..., 0][terminal_mask].eq(int(eos_token_id)).all():
                    raise ValueError("terminal-aware OPD requires exact EOS in candidate slot zero.")
                conditional_start = 1
                if (
                    terminal_secondary_token_id is not None
                    and normalized_terminal_objective_mode != "conservative_kl"
                ):
                    if not candidate_ids[..., 1][terminal_mask].eq(terminal_secondary_token_id).all():
                        raise ValueError(
                            "dual-token terminal-aware OPD requires the secondary token in candidate slot one."
                        )
                    conditional_start = 2
                conditional_ids = candidate_ids[..., conditional_start:][terminal_mask]
                if conditional_ids.eq(int(eos_token_id)).any():
                    raise ValueError("terminal-aware OPD conditional candidates must exclude EOS.")
                if (
                    terminal_secondary_token_id is not None
                    and normalized_terminal_objective_mode != "conservative_kl"
                    and conditional_ids.eq(terminal_secondary_token_id).any()
                ):
                    raise ValueError(
                        "dual-token terminal-aware OPD conditional candidates must exclude the secondary token."
                    )
            if (
                normalized_terminal_objective_mode == "conservative_kl"
                and str(terminal_kl_baseline_mode).strip().lower().replace("-", "_") == "topm_coarse"
            ):
                for required_key in (
                    OPD_TERMINAL_STUDENT_TOPM_IDS_KEY,
                    OPD_TERMINAL_STUDENT_TOPM_LOG_PROBS_KEY,
                ):
                    if required_key not in batch.batch:
                        raise ValueError(f"topm_coarse is missing {required_key}.")
            if str(terminal_objective_mode).strip().lower().replace("-", "_") in {
                "anchor",
                "anchor_kl",
                "kl",
            }:
                student_top_k_log_probs = batch.batch["student_top_k_log_probs"]
                batch.batch[OPD_TERMINAL_BEHAVIOR_EOS_LOG_PROBS_KEY] = (
                    student_top_k_log_probs[..., 0].detach().clone()
                )
                if terminal_secondary_token_id is not None:
                    batch.batch[OPD_TERMINAL_BEHAVIOR_SECONDARY_LOG_PROBS_KEY] = (
                        student_top_k_log_probs[..., 1].detach().clone()
                    )
            metrics["opd_terminal_aware/enabled"] = 1.0
            metrics["opd_terminal_aware/natural_terminal_count"] = terminal_mask.sum().item()
        eos_future_eos_token_id = None
        if eos_future_enable:
            eos_future_eos_token_id = self.tokenizer.eos_token_id
            if isinstance(eos_future_eos_token_id, list):
                eos_future_eos_token_id = (
                    eos_future_eos_token_id[0] if eos_future_eos_token_id else None
                )
            if eos_future_eos_token_id is None:
                raise ValueError("EOS future correction requires an EOS token id.")
            batch.meta_info["eos_token_id"] = int(eos_future_eos_token_id)
        batch.meta_info["opd_eos_future_enable"] = eos_future_enable
        batch.meta_info["opd_eos_future_mode"] = eos_future_mode
        batch.meta_info["opd_eos_future_horizon"] = eos_future_horizon
        batch.meta_info["opd_eos_future_coef"] = eos_future_coef
        if eos_future_enable and eos_future_mode == EOS_FUTURE_MODE_FIXED_SIGNED:
            required_keys = ("student_top_k_ids", "student_top_k_log_probs", "teacher_on_student_log_probs")
            missing_keys = [key for key in required_keys if key not in batch.batch]
            if missing_keys:
                raise ValueError(f"EOS future correction is missing batch keys: {missing_keys}.")
            eos_token_id = int(eos_future_eos_token_id)
            future_teacher_log_probs = batch.batch["teacher_on_student_log_probs"]
            normalized_terminal_objective_mode = str(terminal_objective_mode).strip().lower().replace(
                "-", "_"
            )
            if terminal_aware_enable and normalized_terminal_objective_mode == "teacher_remap_only":
                if terminal_secondary_token_id is None:
                    raise ValueError("Global teacher EOS remapping requires a secondary EOS token id.")
                if OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY not in batch.batch:
                    raise ValueError(
                        "Global teacher EOS remapping is missing teacher secondary-token log-probabilities."
                    )
                remap_output = prepare_terminal_teacher_remap_only_objective(
                    teacher_log_probs=future_teacher_log_probs,
                    candidate_ids=batch.batch["student_top_k_ids"],
                    teacher_secondary_log_probs=batch.batch[
                        OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY
                    ],
                    responses=batch.batch["responses"],
                    response_mask=batch.batch["response_mask"],
                    eos_token_id=int(eos_token_id),
                    secondary_token_id=terminal_secondary_token_id,
                    teacher_remap_floor=terminal_teacher_remap_floor,
                )
                future_teacher_log_probs = remap_output.remapped_teacher_log_probs
                metrics["opd_eos_future/global_teacher_eos_remap"] = 1.0
            fixed_output = build_fixed_eos_future_values(
                student_candidate_log_probs=batch.batch["student_top_k_log_probs"],
                teacher_candidate_log_probs=future_teacher_log_probs,
                responses=batch.batch["responses"],
                response_mask=batch.batch["response_mask"],
                eos_token_id=int(eos_token_id),
                horizon=eos_future_horizon,
            )
            batch.batch[OPD_EOS_FUTURE_VALUE_KEY] = fixed_output.future_value
            batch.batch[OPD_EOS_FUTURE_CORRECTION_MASK_KEY] = fixed_output.correction_mask
            valid_mask = batch.batch["response_mask"].bool()
            correction_mask = fixed_output.correction_mask
            valid_denom = valid_mask.sum().clamp_min(1)
            correction_denom = correction_mask.sum().clamp_min(1)
            metrics["opd_eos_future/enabled"] = 1.0
            metrics["opd_eos_future/horizon"] = float(eos_future_horizon)
            metrics["opd_eos_future/coef"] = eos_future_coef
            metrics["opd_eos_future/semantic_eos_count"] = fixed_output.semantic_eos_mask.sum().item()
            metrics["opd_eos_future/correction_position_count"] = correction_mask.sum().item()
            metrics["opd_eos_future/correction_position_fraction"] = (
                correction_mask.sum() / valid_denom
            ).item()
            metrics["opd_eos_future/local_kl_mean"] = (
                fixed_output.local_kl.masked_fill(~valid_mask, 0.0).sum() / valid_denom
            ).item()
            metrics["opd_eos_future/local_kl_negative_fraction"] = (
                ((fixed_output.local_kl < 0.0) & valid_mask).sum() / valid_denom
            ).item()
            metrics["opd_eos_future/fixed_value_mean"] = (
                fixed_output.future_value.masked_fill(~correction_mask, 0.0).sum()
                / correction_denom
            ).item()
            metrics["opd_eos_future/fixed_value_abs_mean"] = (
                fixed_output.future_value.abs().masked_fill(~correction_mask, 0.0).sum()
                / correction_denom
            ).item()
            metrics["opd_eos_future/fixed_value_negative_fraction"] = (
                ((fixed_output.future_value < 0.0) & correction_mask).sum() / correction_denom
            ).item()
        elif eos_future_enable and eos_future_mode in {
            EOS_FUTURE_MODE_DYNAMIC_H1_OFFPOLICY_CLIPPED,
            EOS_FUTURE_MODE_DYNAMIC_H1_REMAINING_HORIZON_RELU,
            EOS_FUTURE_MODE_DYNAMIC_H1_SQRT_REMAINING_HORIZON_RELU,
        }:
            metrics["opd_eos_future/enabled"] = 1.0
            metrics[f"opd_eos_future/{eos_future_mode}"] = 1.0
            metrics["opd_eos_future/horizon"] = 1.0
            metrics["opd_eos_future/coef"] = eos_future_coef
        batch.batch["adaptive_ppo_rollout_old_log_probs"] = batch.batch["old_log_probs"].detach().clone()
        batch.batch[OPD_ROLLOUT_REFERENCE_LOG_PROBS_KEY] = batch.batch["old_log_probs"].detach().clone()
        if self._prefix_drift_enabled():
            if self._opd_q_prefix_samplek_enabled():
                if self._opd_q_mixture_source_behavior_prefix_enabled():
                    batch.batch[PREFIX_DRIFT_REFERENCE_LOG_PROBS_KEY] = (
                        batch.batch[OPD_ROLLOUT_REFERENCE_LOG_PROBS_KEY].detach().clone()
                    )
                    metrics["opd_q_mixture/prefix_reference_is_q"] = 0.0
                    metrics["opd_q_mixture/prefix_reference_is_source_behavior"] = 1.0
                else:
                    if "opd_proposal_log_probs" not in batch.batch.keys():
                        raise ValueError("Q-prefix sample-k requires precomputed opd_proposal_log_probs.")
                    batch.batch[PREFIX_DRIFT_REFERENCE_LOG_PROBS_KEY] = (
                        batch.batch["opd_proposal_log_probs"].detach().clone()
                    )
                    metrics["opd_q_mixture/prefix_reference_is_q"] = 1.0
                    metrics["opd_q_mixture/prefix_reference_is_source_behavior"] = 0.0
            else:
                batch.batch[PREFIX_DRIFT_REFERENCE_LOG_PROBS_KEY] = (
                    batch.batch["old_log_probs"].detach().clone()
                )
                metrics["opd_q_mixture/prefix_reference_is_q"] = 0.0
                metrics["opd_q_mixture/prefix_reference_is_source_behavior"] = 0.0
            self._update_prefix_drift_for_current_log_probs(
                batch=batch,
                metrics=metrics,
                update_idx=0,
            )
        batch.meta_info["opd_current_samplek_no_candidate_is"] = no_candidate_is
        batch.meta_info["opd_current_samplek_force_candidate_is"] = not no_candidate_is

        if loo_variance_filter_threshold_mode == "update0_quantile":
            if str(loo_advantage_centering).strip().lower().replace("-", "_") not in {
                "loo",
                "leave_one_out",
                "leave_one_out_baseline",
            }:
                raise ValueError("update0_quantile threshold mode requires leave_one_out centering.")
            normalized_terminal_objective_mode = str(terminal_objective_mode).strip().lower().replace(
                "-", "_"
            )
            if terminal_aware_enable and normalized_terminal_objective_mode != "teacher_remap_only":
                raise ValueError(
                    "update0_quantile threshold mode currently supports terminal-aware OPD only "
                    "with objective_mode=teacher_remap_only."
                )
            if self._config_bool(
                rollout_config.get("adaptive_head_tail_negative_elu_enable", False)
            ) or self._config_bool(
                rollout_config.get("opd_samplek_eos_negative_relu_enable", False)
            ):
                raise ValueError(
                    "update0_quantile threshold mode does not support advantage transforms before LOO centering."
                )

            required_quantile_keys = (
                "student_top_k_log_probs",
                "teacher_on_student_log_probs",
            )
            missing_quantile_keys = [
                key for key in required_quantile_keys if key not in batch.batch
            ]
            if missing_quantile_keys:
                raise ValueError(
                    "update0_quantile is missing batch keys: "
                    f"{missing_quantile_keys}."
                )
            quantile_teacher_log_probs = batch.batch["teacher_on_student_log_probs"]
            if terminal_aware_enable:
                required_quantile_keys = (
                    "student_top_k_ids",
                    OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY,
                )
                missing_quantile_keys = [
                    key for key in required_quantile_keys if key not in batch.batch
                ]
                if missing_quantile_keys:
                    raise ValueError(
                        "update0_quantile teacher remapping is missing batch keys: "
                        f"{missing_quantile_keys}."
                    )
                quantile_remap_output = prepare_terminal_teacher_remap_only_objective(
                    teacher_log_probs=batch.batch["teacher_on_student_log_probs"],
                    candidate_ids=batch.batch["student_top_k_ids"],
                    teacher_secondary_log_probs=batch.batch[
                        OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY
                    ],
                    responses=batch.batch["responses"],
                    response_mask=batch.batch["response_mask"],
                    eos_token_id=int(eos_token_id),
                    secondary_token_id=int(terminal_secondary_token_id),
                    teacher_remap_floor=terminal_teacher_remap_floor,
                )
                quantile_teacher_log_probs = quantile_remap_output.remapped_teacher_log_probs
            loo_variance_filter_threshold = (
                compute_samplek_update0_loo_variance_quantile_threshold(
                    teacher_log_probs=quantile_teacher_log_probs,
                    student_log_probs=batch.batch["student_top_k_log_probs"],
                    response_mask=batch.batch["response_mask"],
                    quantile=loo_variance_filter_quantile,
                )
            )
            batch.meta_info["opd_samplek_loo_variance_filter_threshold"] = (
                loo_variance_filter_threshold
            )
            metrics["opd_samplek/loo_variance_filter/update0_quantile"] = (
                loo_variance_filter_quantile
            )
            metrics["opd_samplek/loo_variance_filter/frozen_threshold"] = (
                loo_variance_filter_threshold
            )
            metrics["opd_samplek/loo_variance_filter/threshold_mode_update0_quantile"] = 1.0

        actor_metric_accum = defaultdict(float)
        updates_used = 0
        stopped_by_threshold = False
        threshold_reason = None
        last_mean_abs_log_ratio = 0.0
        last_sampled_token_rkl_k3 = 0.0
        initial_teacher_proxy = None
        previous_teacher_proxy = None
        update0_loo_variance = None
        candidate_refresh_count = 1
        candidate_reuse_count = 0

        metrics["adaptive_ppo/candidate_probe/enabled"] = float(samplek_probe_enabled)
        metrics["adaptive_ppo/update_0/sampled_token_rkl_k3_mean"] = 0.0
        metrics["adaptive_ppo/update_0/batch_sampled_token_rkl_k3_mean"] = 0.0
        metrics["adaptive_ppo/update_0/threshold_eligible"] = 0.0
        if samplek_probe_enabled:
            required_probe_keys = [
                "student_top_k_ids",
                "student_top_k_log_probs",
                "teacher_on_student_log_probs",
            ]
            missing_probe_keys = [key for key in required_probe_keys if key not in batch.batch.keys()]
            if missing_probe_keys:
                raise ValueError(f"adaptive sample-k probe is missing batch keys: {missing_probe_keys}.")
            batch.batch[ADAPTIVE_PPO_PROBE_IDS_KEY] = batch.batch["student_top_k_ids"].detach().clone()
            batch.batch[ADAPTIVE_PPO_PROBE_REFERENCE_LOG_PROBS_KEY] = (
                batch.batch["student_top_k_log_probs"].detach().clone()
            )
            batch.batch[ADAPTIVE_PPO_PROBE_TEACHER_LOG_PROBS_KEY] = (
                batch.batch["teacher_on_student_log_probs"].detach().clone()
            )
            initial_teacher_proxy = self._record_adaptive_samplek_probe_metrics(
                batch=batch,
                current_candidate_log_probs=batch.batch[ADAPTIVE_PPO_PROBE_REFERENCE_LOG_PROBS_KEY],
                metrics=metrics,
                update_idx=0,
            )
            previous_teacher_proxy = initial_teacher_proxy
            metrics["adaptive_ppo/update_0/candidate_probe/teacher_proxy_gain_from_previous"] = 0.0
            metrics["adaptive_ppo/update_0/candidate_probe/teacher_proxy_gain_from_initial"] = 0.0

        for update_idx in range(max_updates):
            if update_idx > 0:
                refresh_candidates = (
                    not candidate_reuse_enabled
                    or is_samplek_candidate_refresh_update(
                        update_idx=update_idx,
                        refresh_interval=candidate_refresh_interval,
                    )
                )
                if refresh_candidates:
                    refresh_timer = (
                        marked_timer(
                            "samplek_candidate_reuse_refresh",
                            timing_raw,
                            color="blue",
                        )
                        if candidate_reuse_enabled
                        else nullcontext()
                    )
                    with refresh_timer:
                        (
                            last_mean_abs_log_ratio,
                            last_sampled_token_rkl_k3,
                            threshold_reason,
                        ) = self._refresh_batch_for_adaptive_ppo_update(
                            batch=batch,
                            metrics=metrics,
                            timing_raw=timing_raw,
                            update_idx=update_idx,
                            min_updates=min_updates,
                            abs_log_ratio_threshold=abs_log_ratio_threshold,
                            sampled_token_rkl_k3_threshold=sampled_token_rkl_k3_threshold,
                        )
                    if candidate_reuse_enabled:
                        candidate_refresh_count += 1
                    if threshold_reason is not None:
                        stopped_by_threshold = True
                        break
                else:
                    candidate_reuse_count += 1
                if samplek_probe_enabled and refresh_candidates:
                    current_probe_log_probs = self._compute_current_log_probs_on_ids(
                        batch=batch,
                        target_ids=batch.batch[ADAPTIVE_PPO_PROBE_IDS_KEY],
                        timing_raw=timing_raw,
                        timer_name="adaptive_ppo_samplek_probe_compute_log_probs",
                    )
                    current_teacher_proxy = self._record_adaptive_samplek_probe_metrics(
                        batch=batch,
                        current_candidate_log_probs=current_probe_log_probs,
                        metrics=metrics,
                        update_idx=update_idx,
                    )
                    gain_from_previous = previous_teacher_proxy - current_teacher_proxy
                    gain_from_initial = initial_teacher_proxy - current_teacher_proxy
                    metrics["adaptive_ppo/candidate_probe/teacher_proxy_gain_from_previous"] = gain_from_previous
                    metrics["adaptive_ppo/candidate_probe/teacher_proxy_gain_from_initial"] = gain_from_initial
                    metrics[
                        f"adaptive_ppo/update_{update_idx}/candidate_probe/teacher_proxy_gain_from_previous"
                    ] = gain_from_previous
                    metrics[
                        f"adaptive_ppo/update_{update_idx}/candidate_probe/teacher_proxy_gain_from_initial"
                    ] = gain_from_initial
                    previous_teacher_proxy = current_teacher_proxy
            if candidate_reuse_enabled:
                refreshed = float(
                    is_samplek_candidate_refresh_update(
                        update_idx=update_idx,
                        refresh_interval=candidate_refresh_interval,
                    )
                )
                metrics[f"samplek_candidate_reuse/update_{update_idx}/refreshed"] = refreshed
                metrics[f"samplek_candidate_reuse/update_{update_idx}/reused"] = 1.0 - refreshed
            if loo_variance_ratio_diagnostics_enable:
                diagnostic_advantages = batch.batch["advantages"]
                if terminal_aware_enable and str(terminal_objective_mode).strip().lower().replace(
                    "-", "_"
                ) == "teacher_remap_only":
                    required_diagnostic_keys = (
                        "student_top_k_ids",
                        "student_top_k_log_probs",
                        "teacher_on_student_log_probs",
                        OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY,
                    )
                    missing_diagnostic_keys = [
                        key for key in required_diagnostic_keys if key not in batch.batch
                    ]
                    if missing_diagnostic_keys:
                        raise ValueError(
                            "Teacher-remapped LOO variance diagnostics are missing batch keys: "
                            f"{missing_diagnostic_keys}."
                        )
                    diagnostic_remap_output = prepare_terminal_teacher_remap_only_objective(
                        teacher_log_probs=batch.batch["teacher_on_student_log_probs"],
                        candidate_ids=batch.batch["student_top_k_ids"],
                        teacher_secondary_log_probs=batch.batch[
                            OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY
                        ],
                        responses=batch.batch["responses"],
                        response_mask=batch.batch["response_mask"],
                        eos_token_id=int(eos_token_id),
                        secondary_token_id=int(terminal_secondary_token_id),
                        teacher_remap_floor=terminal_teacher_remap_floor,
                    )
                    diagnostic_advantages = (
                        diagnostic_remap_output.remapped_teacher_log_probs
                        - batch.batch["student_top_k_log_probs"].detach()
                    )
                    metrics[
                        "opd_samplek/loo_variance_ratio_diagnostics/teacher_remap_applied"
                    ] = 1.0
                centered_advantages = apply_samplek_advantage_centering(
                    diagnostic_advantages,
                    mode=loo_advantage_centering,
                )
                current_loo_variance = compute_samplek_loo_variance(centered_advantages)
                if update0_loo_variance is None:
                    update0_loo_variance = current_loo_variance.detach().clone()
                diagnostic_output = compute_samplek_loo_variance_ratio_diagnostics(
                    current_variance=current_loo_variance,
                    update0_variance=update0_loo_variance,
                    response_mask=batch.batch["response_mask"],
                    absolute_threshold=loo_variance_filter_threshold,
                )
                for name, value in diagnostic_output.metrics.items():
                    key = f"opd_samplek/loo_variance_ratio_diagnostics/{name}"
                    metrics[key] = value
                    metrics[f"adaptive_ppo/update_{update_idx}/{key}"] = value
            if loo_variance_histogram_diagnostics_enable and update_idx in (0, max_updates - 1):
                histogram_advantages = batch.batch["advantages"]
                histogram_teacher_remap_applied = 0.0
                if terminal_aware_enable and str(terminal_objective_mode).strip().lower().replace(
                    "-", "_"
                ) == "teacher_remap_only":
                    required_histogram_keys = (
                        "student_top_k_ids",
                        "student_top_k_log_probs",
                        "teacher_on_student_log_probs",
                        OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY,
                    )
                    missing_histogram_keys = [
                        key for key in required_histogram_keys if key not in batch.batch
                    ]
                    if missing_histogram_keys:
                        raise ValueError(
                            "Teacher-remapped LOO variance histogram diagnostics are missing batch keys: "
                            f"{missing_histogram_keys}."
                        )
                    histogram_remap_output = prepare_terminal_teacher_remap_only_objective(
                        teacher_log_probs=batch.batch["teacher_on_student_log_probs"],
                        candidate_ids=batch.batch["student_top_k_ids"],
                        teacher_secondary_log_probs=batch.batch[
                            OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY
                        ],
                        responses=batch.batch["responses"],
                        response_mask=batch.batch["response_mask"],
                        eos_token_id=int(eos_token_id),
                        secondary_token_id=int(terminal_secondary_token_id),
                        teacher_remap_floor=terminal_teacher_remap_floor,
                    )
                    histogram_advantages = (
                        histogram_remap_output.remapped_teacher_log_probs
                        - batch.batch["student_top_k_log_probs"].detach()
                    )
                    histogram_teacher_remap_applied = 1.0
                histogram_centered_advantages = apply_samplek_advantage_centering(
                    histogram_advantages,
                    mode="leave_one_out",
                )
                histogram_variance = compute_samplek_loo_variance(
                    histogram_centered_advantages
                )
                histogram_output = compute_samplek_loo_variance_histogram_diagnostics(
                    variance=histogram_variance,
                    response_mask=batch.batch["response_mask"],
                )
                histogram_metrics = {
                    **histogram_output.metrics,
                    "teacher_remap_applied": histogram_teacher_remap_applied,
                }
                for name, value in histogram_metrics.items():
                    key = f"opd_samplek/loo_variance_histogram/{name}"
                    metrics[f"adaptive_ppo/update_{update_idx}/{key}"] = value
            with marked_timer("update_actor", timing_raw, color="red"):
                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            for key, value in actor_output_metrics.items():
                actor_metric_accum[key] += float(value)
                is_forced_eos_metric = key.startswith("opd_diagnostic/forced_eos/")
                is_terminal_aware_metric = key.startswith("opd_terminal_aware/")
                is_terminal_safe_continue_metric = key.startswith("opd_terminal_safe_continue/")
                is_terminal_conservative_metric = key.startswith("opd_terminal_conservative_kl/")
                is_terminal_teacher_remap_only_metric = key.startswith(
                    "opd_terminal_teacher_remap_only/"
                )
                is_eos_future_metric = key.startswith("opd_eos_future/")
                is_teacher_deficit_residual_metric = key.startswith("opd_teacher_deficit_residual/")
                is_candidate_reuse_metric = key.startswith("samplek_candidate_reuse/")
                is_prefix_drift_metric = key.startswith("prefix_drift/")
                if (
                    key.startswith(("opd_samplek/", "opd/samplek_", "opd_q_samplek/"))
                    or is_forced_eos_metric
                    or is_terminal_aware_metric
                    or is_terminal_safe_continue_metric
                    or is_terminal_conservative_metric
                    or is_terminal_teacher_remap_only_metric
                    or is_eos_future_metric
                    or is_teacher_deficit_residual_metric
                    or is_candidate_reuse_metric
                    or is_prefix_drift_metric
                ):
                    metrics[f"adaptive_ppo/update_{update_idx}/{key}"] = float(value)
            if candidate_reuse_enabled:
                last_mean_abs_log_ratio = float(
                    actor_output_metrics.get(
                        "samplek_candidate_reuse/response_log_ratio_abs_mean",
                        last_mean_abs_log_ratio,
                    )
                )
                last_sampled_token_rkl_k3 = float(
                    actor_output_metrics.get(
                        "samplek_candidate_reuse/response_rkl_k3_mean",
                        last_sampled_token_rkl_k3,
                    )
                )
            updates_used += 1

        if updates_used > 0:
            metrics.update({key: value / updates_used for key, value in actor_metric_accum.items()})
        metrics["adaptive_ppo/updates_used"] = updates_used
        metrics["adaptive_ppo/update_fraction"] = updates_used / max_updates
        metrics["adaptive_ppo/min_updates"] = min_updates
        metrics["adaptive_ppo/max_updates"] = max_updates
        metrics["adaptive_ppo/reached_max_updates"] = float(updates_used == max_updates)
        metrics["adaptive_ppo/stopped_at_update"] = updates_used if stopped_by_threshold else -1
        metrics["adaptive_ppo/stopped_by_threshold"] = float(stopped_by_threshold)
        metrics["adaptive_ppo/stopped_by_abs_log_ratio_threshold"] = float(
            threshold_reason is not None and "abs_log_ratio" in threshold_reason
        )
        metrics["adaptive_ppo/stopped_by_sampled_token_rkl_k3_threshold"] = float(
            threshold_reason is not None and "sampled_token_rkl_k3" in threshold_reason
        )
        metrics["adaptive_ppo/no_candidate_is"] = float(no_candidate_is)
        metrics["adaptive_ppo/force_candidate_is"] = float(not no_candidate_is)
        if candidate_reuse_enabled:
            metrics["samplek_candidate_reuse/enabled"] = 1.0
            metrics["samplek_candidate_reuse/refresh_interval"] = candidate_refresh_interval
            metrics["samplek_candidate_reuse/refresh_count"] = candidate_refresh_count
            metrics["samplek_candidate_reuse/reuse_count"] = candidate_reuse_count
            metrics["samplek_candidate_reuse/refresh_rate"] = (
                candidate_refresh_count / max(updates_used, 1)
            )
        metrics["adaptive_ppo/last_mean_abs_log_ratio"] = last_mean_abs_log_ratio
        metrics["adaptive_ppo/last_batch_mean_abs_log_ratio"] = last_mean_abs_log_ratio
        metrics["adaptive_ppo/last_sampled_token_rkl_k3_mean"] = last_sampled_token_rkl_k3
        metrics["adaptive_ppo/last_batch_sampled_token_rkl_k3_mean"] = last_sampled_token_rkl_k3
        if abs_log_ratio_threshold is not None:
            metrics["adaptive_ppo/abs_log_ratio_threshold"] = abs_log_ratio_threshold
        if sampled_token_rkl_k3_threshold is not None:
            metrics["adaptive_ppo/sampled_token_rkl_k3_threshold"] = sampled_token_rkl_k3_threshold

    def _compute_current_log_probs_on_ids(
        self,
        *,
        batch: DataProto,
        target_ids: torch.Tensor,
        timing_raw: dict,
        timer_name: str,
    ) -> torch.Tensor:
        had_target_ids = "target_ids" in batch.batch.keys()
        previous_target_ids = batch.batch["target_ids"] if had_target_ids else None
        batch.batch["target_ids"] = target_ids
        try:
            with marked_timer(timer_name, timing_raw, color="blue"):
                current_on_sample_ids = self.actor_rollout_wg.compute_log_probs_for_ids(batch)
        finally:
            if had_target_ids:
                batch.batch["target_ids"] = previous_target_ids
            elif "target_ids" in batch.batch.keys():
                batch.batch.pop("target_ids")

        if "student_log_probs_on_teacher_ids" not in current_on_sample_ids.batch.keys():
            raise ValueError("compute_log_probs_for_ids did not return student_log_probs_on_teacher_ids.")
        return current_on_sample_ids.batch["student_log_probs_on_teacher_ids"]

    def _compute_current_log_probs_on_samplek_ids(self, batch: DataProto, timing_raw: dict) -> torch.Tensor:
        if "student_top_k_ids" not in batch.batch.keys():
            raise ValueError("ESS sample-k resample requires student_top_k_ids in the batch.")
        return self._compute_current_log_probs_on_ids(
            batch=batch,
            target_ids=batch.batch["student_top_k_ids"],
            timing_raw=timing_raw,
            timer_name="ess_samplek_compute_log_probs_for_ids",
        )

    def _resample_samplek_candidates_for_ess_update(
        self,
        batch: DataProto,
        metrics: dict,
        timing_raw: dict,
        update_idx: int,
    ) -> None:
        with marked_timer("ess_samplek_resample_compute_log_prob", timing_raw, color="blue"):
            current_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            self._replace_batch_tensors(batch, current_log_prob)

        batch.meta_info["global_steps"] = self.global_steps
        batch.meta_info["is_plot"] = False
        batch.meta_info["return_teacher_topk_metrics"] = False
        with marked_timer("ess_samplek_resample_compute_rm_score", timing_raw, color="magenta"):
            teacher_data = self.rm_wg.compute_rm_score(batch)
            self._replace_batch_tensors(batch, teacher_data)

        top_k = int(
            batch.meta_info.get(
                "log_prob_top_k",
                self.config.actor_rollout_ref.rollout.get("log_prob_top_k", 0),
            )
        )
        if top_k > 0:
            with marked_timer("ess_samplek_resample_compute_distillation_reward", timing_raw, color="orange"):
                distillation_output = self.actor_rollout_wg.compute_distillation_reward(batch)
                self._replace_batch_tensors(batch, distillation_output)

        self._recompute_reward_and_advantage_for_adaptive_ppo(batch, metrics)
        metrics[f"ess_samplek/update_{update_idx}/resampled"] = 1.0

    def _run_ess_samplek_resample_actor_updates(self, batch: DataProto, metrics: dict, timing_raw: dict) -> None:
        rollout_config = self.config.actor_rollout_ref.rollout
        actor_config = self.config.actor_rollout_ref.actor

        actor_ppo_epochs = int(actor_config.get("ppo_epochs", 1))
        if actor_ppo_epochs != 1:
            raise ValueError(
                "ess_samplek_resample_enable=True expects actor_rollout_ref.actor.ppo_epochs=1. "
                "Use ess_samplek_resample_max_updates for the fixed update count."
            )
        if not self.use_rm:
            raise ValueError("ess_samplek_resample_enable=True currently requires reward_model.enable=True.")

        candidate_mode = str(
            batch.meta_info.get(
                "log_prob_candidate_mode",
                rollout_config.get("log_prob_candidate_mode", "topk"),
            )
        ).strip().lower().replace("-", "_")
        candidate_mode = {
            "sample": "sample_stu",
            "sample_student": "sample_stu",
            "student_sample": "sample_stu",
            "sample_k": "sample_stu",
        }.get(candidate_mode, candidate_mode)
        top_k = int(batch.meta_info.get("log_prob_top_k", rollout_config.get("log_prob_top_k", 0)))
        if top_k <= 0 or candidate_mode != "sample_stu":
            raise ValueError(
                "ess_samplek_resample_enable=True is scoped to sample-k OPD "
                "(log_prob_candidate_mode=sample_stu and log_prob_top_k>0)."
            )

        max_updates = int(rollout_config.get("ess_samplek_resample_max_updates", 4))
        if max_updates < 1:
            raise ValueError("ess_samplek_resample_max_updates must be >= 1.")
        mean_threshold = float(rollout_config.get("ess_samplek_resample_mean_threshold", 0.5))

        for required_key in ["student_top_k_ids", "student_top_k_log_probs", "teacher_on_student_log_probs"]:
            if required_key not in batch.batch.keys():
                raise ValueError(f"ESS sample-k resample requires {required_key} in the batch.")

        if "old_log_probs" not in batch.batch.keys():
            raise ValueError("ESS sample-k updates require old_log_probs in the rollout batch.")
        batch.batch[OPD_ROLLOUT_REFERENCE_LOG_PROBS_KEY] = batch.batch["old_log_probs"].detach().clone()

        if self._prefix_drift_enabled():
            if "old_log_probs" not in batch.batch.keys():
                raise ValueError("prefix_drift_enable=True requires old_log_probs in the rollout batch.")
            batch.batch[PREFIX_DRIFT_REFERENCE_LOG_PROBS_KEY] = batch.batch["old_log_probs"].detach().clone()
            self._update_prefix_drift_for_current_log_probs(
                batch=batch,
                metrics=metrics,
                update_idx=0,
            )

        batch.meta_info["opd_current_samplek_no_candidate_is"] = False
        batch.meta_info["opd_current_samplek_force_candidate_is"] = True

        actor_metric_accum = defaultdict(float)
        updates_used = 0
        ess_checks = 0
        resample_count = 0
        reuse_count = 0
        last_mean_ess = 1.0
        last_min_ess = 1.0

        for update_idx in range(max_updates):
            if update_idx > 0:
                if self._prefix_drift_enabled():
                    with marked_timer("ess_samplek_prefix_drift_compute_log_prob", timing_raw, color="blue"):
                        current_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        self._replace_batch_tensors(batch, current_log_prob)
                    self._update_prefix_drift_for_current_log_probs(
                        batch=batch,
                        metrics=metrics,
                        update_idx=update_idx,
                    )

                current_candidate_log_probs = self._compute_current_log_probs_on_samplek_ids(
                    batch=batch,
                    timing_raw=timing_raw,
                )
                mean_ess, min_ess, _ = self._masked_samplek_normalized_ess(
                    current_log_probs=current_candidate_log_probs,
                    sample_log_probs=batch.batch["student_top_k_log_probs"],
                    response_mask=batch.batch["response_mask"],
                )
                last_mean_ess = mean_ess.detach().item()
                last_min_ess = min_ess.detach().item()
                ess_checks += 1

                metrics["ess_samplek/mean_ess"] = last_mean_ess
                metrics["ess_samplek/min_ess"] = last_min_ess
                metrics[f"ess_samplek/update_{update_idx}/mean_ess"] = last_mean_ess
                metrics[f"ess_samplek/update_{update_idx}/min_ess"] = last_min_ess

                if last_mean_ess < mean_threshold:
                    resample_count += 1
                    self._resample_samplek_candidates_for_ess_update(
                        batch=batch,
                        metrics=metrics,
                        timing_raw=timing_raw,
                        update_idx=update_idx,
                    )
                else:
                    reuse_count += 1
                    metrics[f"ess_samplek/update_{update_idx}/resampled"] = 0.0

            with marked_timer("update_actor", timing_raw, color="red"):
                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            for key, value in actor_output_metrics.items():
                actor_metric_accum[key] += float(value)
            updates_used += 1

        if updates_used > 0:
            metrics.update({key: value / updates_used for key, value in actor_metric_accum.items()})
        metrics["ess_samplek/updates_used"] = updates_used
        metrics["ess_samplek/max_updates"] = max_updates
        metrics["ess_samplek/ess_checks"] = ess_checks
        metrics["ess_samplek/resample_count"] = resample_count
        metrics["ess_samplek/reuse_count"] = reuse_count
        metrics["ess_samplek/resample_rate"] = resample_count / max(ess_checks, 1)
        metrics["ess_samplek/last_mean_ess"] = last_mean_ess
        metrics["ess_samplek/last_min_ess"] = last_min_ess
        metrics["ess_samplek/mean_threshold"] = mean_threshold

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            train_iter = iter(self.train_dataloader)
            while True:
                try:
                    batch_dict = next(train_iter)
                except StopIteration:
                    break
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch = self._take_current_prompt_batch_for_offpolicy_replay(batch, train_iter)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch.meta_info["global_steps"] = self.global_steps
                                batch.meta_info["is_plot"] = self.config.trainer.get("is_plot", False)
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    batch = self._mix_teacher_rollout_batch(batch, metrics)
                    batch = self._mix_offpolicy_replay_batch(batch, metrics)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            with marked_timer("compute_log_prob", timing_raw, color="blue"):
                                print("First forward, get student top k ids and log probs")
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)

                                batch = batch.union(old_log_prob)
                                self._patch_teacher_rollout_mix_rollout_log_probs(batch)
                                eos_token_id = self.tokenizer.eos_token_id
                                if isinstance(eos_token_id, list):
                                    eos_token_id = eos_token_id[0] if eos_token_id else None
                                if eos_token_id is not None:
                                    record_nonterminal_eos_probability_metric(
                                        batch_tensors=batch.batch,
                                        metrics=metrics,
                                        eos_token_id=int(eos_token_id),
                                    )

                            top_k = self.config.actor_rollout_ref.rollout.get("log_prob_top_k", 0)
                            strategy = self.config.actor_rollout_ref.rollout.get("top_k_strategy", "only_stu")
                            kl_estimator = self.config.actor_rollout_ref.rollout.get("kl_estimator", "k1")
                            reward_weight_mode = self.config.actor_rollout_ref.rollout.get("reward_weight_mode", "student_p")
                            reward_weight_normalize = self.config.actor_rollout_ref.rollout.get("reward_weight_normalize", None)
                            candidate_mode = self.config.actor_rollout_ref.rollout.get("log_prob_candidate_mode", "topk")
                            adaptive_head_tail_gamma = self.config.actor_rollout_ref.rollout.get("adaptive_head_tail_gamma", 0.5)
                            adaptive_head_tail_k2_min = self.config.actor_rollout_ref.rollout.get("adaptive_head_tail_k2_min", 1)
                            adaptive_head_tail_negative_elu_enable = self.config.actor_rollout_ref.rollout.get("adaptive_head_tail_negative_elu_enable", False)
                            adaptive_head_tail_negative_elu_threshold = self.config.actor_rollout_ref.rollout.get("adaptive_head_tail_negative_elu_threshold", -1.0)
                            adaptive_head_tail_negative_elu_tau = self.config.actor_rollout_ref.rollout.get("adaptive_head_tail_negative_elu_tau", 1.0)
                            sample_k_kl_plus_one = self.config.actor_rollout_ref.rollout.get("sample_k_kl_plus_one", True)
                            sample_k_replacement = self.config.actor_rollout_ref.rollout.get(
                                "sample_k_replacement", True
                            )
                            opd_loss_type = self.config.actor_rollout_ref.rollout.get("opd_loss_type", "sample_k_reverse_kl")
                            opd_advantage_mode = self.config.actor_rollout_ref.rollout.get("opd_advantage_mode", "fixed")
                            opd_decomposed_prefix_is_mode = self.config.actor_rollout_ref.rollout.get(
                                "opd_decomposed_prefix_is_mode", "cumulative_cap"
                            )
                            opd_decomposed_prefix_is_min_weight = self.config.actor_rollout_ref.rollout.get(
                                "opd_decomposed_prefix_is_min_weight", 0.25
                            )
                            opd_decomposed_prefix_is_max_weight = self.config.actor_rollout_ref.rollout.get(
                                "opd_decomposed_prefix_is_max_weight", 4.0
                            )
                            opd_decomposed_proximal_coef = self.config.actor_rollout_ref.rollout.get(
                                "opd_decomposed_proximal_coef", 1.0
                            )
                            opd_q_mixture_enable = self._opd_q_mixture_enabled()
                            opd_q_mixture_teacher_advantage_mode = (
                                self._opd_q_mixture_teacher_advantage_mode()
                            )
                            opd_q_mixture_source_normalize_enable = (
                                self._opd_q_mixture_source_normalize_enabled()
                            )
                            opd_samplek_candidate_aggregation = self.config.actor_rollout_ref.rollout.get(
                                "opd_samplek_candidate_aggregation", "sum"
                            )
                            opd_samplek_advantage_centering = self.config.actor_rollout_ref.rollout.get(
                                "opd_samplek_advantage_centering", "none"
                            )
                            opd_samplek_loo_variance_filter_threshold = self.config.actor_rollout_ref.rollout.get(
                                "opd_samplek_loo_variance_filter_threshold", None
                            )
                            opd_samplek_loo_variance_filter_threshold_mode = (
                                self.config.actor_rollout_ref.rollout.get(
                                    "opd_samplek_loo_variance_filter_threshold_mode", "fixed"
                                )
                            )
                            opd_samplek_loo_variance_filter_quantile = (
                                self.config.actor_rollout_ref.rollout.get(
                                    "opd_samplek_loo_variance_filter_quantile", 0.7
                                )
                            )
                            opd_samplek_loo_variance_filter_selection = (
                                self.config.actor_rollout_ref.rollout.get(
                                    "opd_samplek_loo_variance_filter_selection", "high"
                                )
                            )
                            opd_samplek_loo_variance_filter_mode = self.config.actor_rollout_ref.rollout.get(
                                "opd_samplek_loo_variance_filter_mode", "hard"
                            )
                            opd_samplek_loo_variance_filter_soft_base_weight = (
                                self.config.actor_rollout_ref.rollout.get(
                                    "opd_samplek_loo_variance_filter_soft_base_weight", 0.5
                                )
                            )
                            opd_samplek_loo_variance_filter_soft_active_bonus = (
                                self.config.actor_rollout_ref.rollout.get(
                                    "opd_samplek_loo_variance_filter_soft_active_bonus", 1.0
                                )
                            )
                            opd_samplek_loo_variance_filter_expectile_tau = (
                                self.config.actor_rollout_ref.rollout.get(
                                    "opd_samplek_loo_variance_filter_expectile_tau", 0.75
                                )
                            )
                            opd_samplek_influence_clip = self.config.actor_rollout_ref.rollout.get(
                                "opd_samplek_influence_clip", None
                            )
                            teacher_deficit_residual_enable = self._config_bool(
                                self.config.actor_rollout_ref.rollout.get(
                                    "opd_teacher_deficit_residual_enable", False
                                )
                            )
                            teacher_deficit_residual_k = int(
                                self.config.actor_rollout_ref.rollout.get(
                                    "opd_teacher_deficit_residual_k", 8
                                )
                            )
                            teacher_deficit_residual_coef = float(
                                self.config.actor_rollout_ref.rollout.get(
                                    "opd_teacher_deficit_residual_coef", 0.0
                                )
                            )
                            validate_teacher_deficit_residual_configuration(
                                enabled=teacher_deficit_residual_enable,
                                teacher_sample_count=teacher_deficit_residual_k,
                                coefficient=teacher_deficit_residual_coef,
                                candidate_mode=candidate_mode,
                                top_k=top_k,
                                top_k_strategy=strategy,
                                advantage_mode=opd_advantage_mode,
                                sample_replacement=self._config_bool(sample_k_replacement),
                                opd_loss_type=opd_loss_type,
                            )
                            diagnostic_forced_eos_enable = self.config.actor_rollout_ref.rollout.get(
                                "opd_diagnostic_forced_eos_enable", False
                            )
                            terminal_aware_enable = self.config.actor_rollout_ref.rollout.get(
                                "opd_terminal_aware_enable", False
                            )
                            terminal_objective_mode = self.config.actor_rollout_ref.rollout.get(
                                "opd_terminal_objective_mode", "anchor_kl"
                            )
                            terminal_anchor_mode = self.config.actor_rollout_ref.rollout.get(
                                "opd_terminal_anchor_mode", "behavior"
                            )
                            terminal_gate_coef = self.config.actor_rollout_ref.rollout.get(
                                "opd_terminal_gate_coef", 1.0
                            )
                            terminal_secondary_token_id = self._config_optional_int(
                                self.config.actor_rollout_ref.rollout.get(
                                    "opd_terminal_secondary_token_id", None
                                ),
                                "opd_terminal_secondary_token_id",
                            )
                            terminal_teacher_remap_enable = self.config.actor_rollout_ref.rollout.get(
                                "opd_terminal_teacher_remap_enable", False
                            )
                            terminal_kl_baseline_mode = self.config.actor_rollout_ref.rollout.get(
                                "opd_terminal_kl_baseline_mode", "mc_loo"
                            )
                            terminal_teacher_remap_floor = self.config.actor_rollout_ref.rollout.get(
                                "opd_terminal_teacher_remap_floor", 1e-18
                            )
                            terminal_topm = self.config.actor_rollout_ref.rollout.get(
                                "opd_terminal_topm", 0
                            )
                            opd_samplek_eos_negative_relu_enable = self._config_bool(
                                self.config.actor_rollout_ref.rollout.get(
                                    "opd_samplek_eos_negative_relu_enable", False
                                )
                            )
                            (
                                sampled_token_eos_alignment_enable,
                                sampled_token_student_eos_token_id,
                                sampled_token_teacher_eos_token_id,
                            ) = self._sampled_token_eos_alignment_config()
                            opd_raw_advantage_clip = self.config.actor_rollout_ref.rollout.get(
                                "opd_raw_advantage_clip", None
                            )
                            opd_samplek_entropy_coef = self.config.actor_rollout_ref.rollout.get(
                                "opd_samplek_entropy_coef", 0.0
                            )
                            opd_sampled_token_proximal_coef = self.config.actor_rollout_ref.rollout.get(
                                "opd_sampled_token_proximal_coef", 0.0
                            )
                            opd_sampled_token_proximal_mode = self.config.actor_rollout_ref.rollout.get(
                                "opd_sampled_token_proximal_mode", "reverse_kl"
                            )
                            opd_sampled_token_max_entropy_coef = self.config.actor_rollout_ref.rollout.get(
                                "opd_sampled_token_max_entropy_coef", 0.0
                            )
                            opd_sampled_token_proximal_prefix_weight_enable = (
                                self.config.actor_rollout_ref.rollout.get(
                                    "opd_sampled_token_proximal_prefix_weight_enable", False
                                )
                            )
                            opd_samplek_total_grad_norm = self.config.actor_rollout_ref.rollout.get(
                                "opd_samplek_total_grad_norm", None
                            )
                            self._set_dapo_answer_format_penalty_meta(batch, metrics)
                            chi_square_baseline = self.config.actor_rollout_ref.rollout.get("chi_square_baseline", "mean")
                            on_logprob_mse_clip = self.config.actor_rollout_ref.rollout.get("on_logprob_mse_clip", None)
                            on_logprob_mse_center = self.config.actor_rollout_ref.rollout.get("on_logprob_mse_center", False)
                            on_logprob_mse_normalize = self.config.actor_rollout_ref.rollout.get("on_logprob_mse_normalize", False)

                            batch.meta_info["global_steps"] = self.global_steps
                            batch.meta_info["is_plot"] = self.config.trainer.get("is_plot", False)
                            teacher_temperature = self._current_teacher_temperature()
                            metrics.update(
                                {
                                    "teacher_temperature/current": teacher_temperature,
                                    "teacher_temperature/initial": float(
                                        self.config.actor_rollout_ref.rollout.get("teacher_temperature", 1.0)
                                    ),
                                    "teacher_temperature/minimum": float(
                                        self.config.actor_rollout_ref.rollout.get("teacher_temperature_min", 1.0)
                                    ),
                                    "teacher_temperature/anneal_progress": (
                                        self._teacher_temperature_anneal_progress()
                                    ),
                                    "teacher_temperature/anneal_enabled": float(
                                        self._teacher_temperature_anneal_enabled()
                                    ),
                                    "teacher_temperature/cosine_schedule": float(
                                        self._teacher_temperature_anneal_schedule() == "cosine"
                                    ),
                                }
                            )

                            batch.meta_info["log_prob_top_k"] = top_k
                            batch.meta_info["top_k_strategy"] = strategy
                            batch.meta_info["kl_estimator"] = kl_estimator
                            batch.meta_info["reward_weight_mode"] = reward_weight_mode
                            batch.meta_info["reward_weight_normalize"] = reward_weight_normalize
                            batch.meta_info["log_prob_candidate_mode"] = candidate_mode
                            batch.meta_info["adaptive_head_tail_gamma"] = adaptive_head_tail_gamma
                            batch.meta_info["adaptive_head_tail_k2_min"] = adaptive_head_tail_k2_min
                            batch.meta_info["adaptive_head_tail_negative_elu_enable"] = (
                                adaptive_head_tail_negative_elu_enable
                            )
                            batch.meta_info["adaptive_head_tail_negative_elu_threshold"] = (
                                adaptive_head_tail_negative_elu_threshold
                            )
                            batch.meta_info["adaptive_head_tail_negative_elu_tau"] = (
                                adaptive_head_tail_negative_elu_tau
                            )
                            batch.meta_info["sample_k_kl_plus_one"] = sample_k_kl_plus_one
                            batch.meta_info["sample_k_replacement"] = sample_k_replacement
                            batch.meta_info["adaptive_ppo_update_enable"] = self._adaptive_ppo_update_enabled()
                            batch.meta_info["samplek_candidate_reuse_is_enable"] = (
                                self._samplek_candidate_reuse_enabled()
                            )
                            batch.meta_info["prefix_drift_enable"] = self._prefix_drift_enabled()
                            batch.meta_info["prefix_drift_method"] = self.config.actor_rollout_ref.rollout.get(
                                "prefix_drift_method", "diagnostic"
                            )
                            batch.meta_info["prefix_drift_log_clip"] = self.config.actor_rollout_ref.rollout.get(
                                "prefix_drift_log_clip", 3.0
                            )
                            batch.meta_info["prefix_drift_log_clip_mode"] = (
                                self.config.actor_rollout_ref.rollout.get(
                                    "prefix_drift_log_clip_mode", "symmetric"
                                )
                            )
                            batch.meta_info["prefix_drift_normalize"] = self.config.actor_rollout_ref.rollout.get(
                                "prefix_drift_normalize", "none"
                            )
                            batch.meta_info["prefix_drift_position_beta"] = (
                                self.config.actor_rollout_ref.rollout.get(
                                    "prefix_drift_position_beta", 0.0
                                )
                            )
                            batch.meta_info["prefix_drift_position_hmax"] = (
                                self.config.actor_rollout_ref.rollout.get(
                                    "prefix_drift_position_hmax", None
                                )
                            )
                            batch.meta_info["prefix_drift_detach"] = self.config.actor_rollout_ref.rollout.get(
                                "prefix_drift_detach", True
                            )
                            batch.meta_info["opd_loss_type"] = opd_loss_type
                            batch.meta_info["opd_advantage_mode"] = opd_advantage_mode
                            batch.meta_info["opd_decomposed_prefix_is_mode"] = (
                                opd_decomposed_prefix_is_mode
                            )
                            batch.meta_info["opd_decomposed_prefix_is_min_weight"] = (
                                opd_decomposed_prefix_is_min_weight
                            )
                            batch.meta_info["opd_decomposed_prefix_is_max_weight"] = (
                                opd_decomposed_prefix_is_max_weight
                            )
                            batch.meta_info["opd_decomposed_proximal_coef"] = opd_decomposed_proximal_coef
                            batch.meta_info["opd_q_mixture_enable"] = opd_q_mixture_enable
                            batch.meta_info["opd_q_mixture_teacher_advantage_mode"] = (
                                opd_q_mixture_teacher_advantage_mode
                            )
                            batch.meta_info["opd_q_mixture_source_normalize_enable"] = (
                                opd_q_mixture_source_normalize_enable
                            )
                            batch.meta_info["opd_samplek_candidate_aggregation"] = (
                                opd_samplek_candidate_aggregation
                            )
                            batch.meta_info["opd_samplek_advantage_centering"] = (
                                opd_samplek_advantage_centering
                            )
                            batch.meta_info["opd_samplek_loo_variance_filter_threshold"] = (
                                opd_samplek_loo_variance_filter_threshold
                            )
                            batch.meta_info["opd_samplek_loo_variance_filter_threshold_mode"] = (
                                opd_samplek_loo_variance_filter_threshold_mode
                            )
                            batch.meta_info["opd_samplek_loo_variance_filter_quantile"] = (
                                opd_samplek_loo_variance_filter_quantile
                            )
                            batch.meta_info["opd_samplek_loo_variance_filter_selection"] = (
                                opd_samplek_loo_variance_filter_selection
                            )
                            batch.meta_info["opd_samplek_loo_variance_filter_mode"] = (
                                opd_samplek_loo_variance_filter_mode
                            )
                            batch.meta_info["opd_samplek_loo_variance_filter_soft_base_weight"] = (
                                opd_samplek_loo_variance_filter_soft_base_weight
                            )
                            batch.meta_info["opd_samplek_loo_variance_filter_soft_active_bonus"] = (
                                opd_samplek_loo_variance_filter_soft_active_bonus
                            )
                            batch.meta_info["opd_samplek_loo_variance_filter_expectile_tau"] = (
                                opd_samplek_loo_variance_filter_expectile_tau
                            )
                            batch.meta_info["opd_samplek_influence_clip"] = opd_samplek_influence_clip
                            batch.meta_info["opd_teacher_deficit_residual_enable"] = (
                                teacher_deficit_residual_enable
                            )
                            batch.meta_info["opd_teacher_deficit_residual_k"] = (
                                teacher_deficit_residual_k
                            )
                            batch.meta_info["opd_teacher_deficit_residual_coef"] = (
                                teacher_deficit_residual_coef
                            )
                            metrics["opd_teacher_deficit_residual/enabled"] = float(
                                teacher_deficit_residual_enable
                            )
                            metrics["opd_teacher_deficit_residual/sample_count"] = float(
                                teacher_deficit_residual_k
                            )
                            metrics["opd_teacher_deficit_residual/coef"] = (
                                teacher_deficit_residual_coef
                            )
                            batch.meta_info["opd_diagnostic_forced_eos_enable"] = (
                                diagnostic_forced_eos_enable
                            )
                            batch.meta_info["opd_terminal_aware_enable"] = terminal_aware_enable
                            batch.meta_info["opd_terminal_objective_mode"] = terminal_objective_mode
                            batch.meta_info["opd_terminal_anchor_mode"] = terminal_anchor_mode
                            batch.meta_info["opd_terminal_gate_coef"] = terminal_gate_coef
                            batch.meta_info["opd_terminal_secondary_token_id"] = (
                                terminal_secondary_token_id
                            )
                            batch.meta_info["opd_terminal_teacher_remap_enable"] = (
                                terminal_teacher_remap_enable
                            )
                            batch.meta_info["opd_terminal_kl_baseline_mode"] = (
                                terminal_kl_baseline_mode
                            )
                            batch.meta_info["opd_terminal_teacher_remap_floor"] = (
                                terminal_teacher_remap_floor
                            )
                            batch.meta_info["opd_terminal_topm"] = terminal_topm
                            batch.meta_info["opd_samplek_eos_negative_relu_enable"] = (
                                opd_samplek_eos_negative_relu_enable
                            )
                            batch.meta_info["opd_sampled_token_eos_alignment_enable"] = (
                                sampled_token_eos_alignment_enable
                            )
                            batch.meta_info["opd_sampled_token_teacher_eos_token_id"] = (
                                sampled_token_teacher_eos_token_id
                            )
                            if sampled_token_eos_alignment_enable:
                                batch.meta_info["eos_token_id"] = sampled_token_student_eos_token_id
                            if self._config_bool(
                                terminal_aware_enable
                            ) or opd_samplek_eos_negative_relu_enable:
                                terminal_eos_token_id = self.tokenizer.eos_token_id
                                if isinstance(terminal_eos_token_id, list):
                                    terminal_eos_token_id = (
                                        terminal_eos_token_id[0] if terminal_eos_token_id else None
                                    )
                                if terminal_eos_token_id is None:
                                    raise ValueError(
                                        "terminal-aware or EOS-ReLU OPD requires an EOS token id."
                                    )
                                batch.meta_info["eos_token_id"] = int(terminal_eos_token_id)
                            batch.meta_info["opd_raw_advantage_clip"] = opd_raw_advantage_clip
                            batch.meta_info["opd_samplek_entropy_coef"] = opd_samplek_entropy_coef
                            batch.meta_info["opd_sampled_token_proximal_coef"] = (
                                opd_sampled_token_proximal_coef
                            )
                            batch.meta_info["opd_sampled_token_proximal_mode"] = (
                                opd_sampled_token_proximal_mode
                            )
                            batch.meta_info["opd_sampled_token_max_entropy_coef"] = (
                                opd_sampled_token_max_entropy_coef
                            )
                            batch.meta_info["opd_sampled_token_proximal_prefix_weight_enable"] = (
                                opd_sampled_token_proximal_prefix_weight_enable
                            )
                            batch.meta_info["opd_samplek_total_grad_norm"] = opd_samplek_total_grad_norm
                            batch.meta_info["chi_square_baseline"] = chi_square_baseline
                            batch.meta_info["on_logprob_mse_clip"] = on_logprob_mse_clip
                            batch.meta_info["on_logprob_mse_center"] = on_logprob_mse_center
                            batch.meta_info["on_logprob_mse_normalize"] = on_logprob_mse_normalize
                            batch.meta_info["teacher_temperature"] = teacher_temperature
                            
                            with marked_timer("compute_rm_score", timing_raw, color="magenta"):
                                teacher_data = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(teacher_data)
                            if sampled_token_eos_alignment_enable:
                                teacher_primary_log_probs = batch.batch.pop(
                                    OPD_SAMPLED_TOKEN_EOS_TEACHER_PRIMARY_LOG_PROBS_KEY
                                )
                                teacher_secondary_log_probs = batch.batch.pop(
                                    OPD_SAMPLED_TOKEN_EOS_TEACHER_SECONDARY_LOG_PROBS_KEY
                                )
                                metrics.update(
                                    compute_sampled_token_eos_alignment_metrics(
                                        responses=batch.batch["responses"],
                                        response_mask=batch.batch["response_mask"],
                                        student_eos_token_id=sampled_token_student_eos_token_id,
                                        teacher_primary_log_probs=teacher_primary_log_probs,
                                        teacher_secondary_log_probs=teacher_secondary_log_probs,
                                    )
                                )

                            self._prepare_opd_q_mixture_batch(batch, metrics)

                            if top_k > 0:
                                with marked_timer("compute_distillation_reward", timing_raw, color="orange"):
                                    distillation_output = self.actor_rollout_wg.compute_distillation_reward(batch)
                                    batch = batch.union(distillation_output)
                        
                        if (self.global_steps == 1 or self.global_steps % 10 == 0) and "student_valid_counts" in batch.batch.keys():
                            try:
                                import matplotlib.pyplot as plt
                                import swanlab

                                response_mask = batch.batch["response_mask"]
                                valid_denom = response_mask.sum(dim=0) + 1e-6

                                plot_data = {}
                                
                                if "student_valid_counts" in batch.batch.keys():
                                    student_counts = batch.batch["student_valid_counts"].float()
                                    avg_student_counts = (student_counts * response_mask).sum(dim=0) / valid_denom
                                    plot_data["Student"] = avg_student_counts.detach().cpu().numpy()
                                
                                if "teacher_valid_counts" in batch.batch.keys():
                                    teacher_counts = batch.batch["teacher_valid_counts"].float()
                                    avg_teacher_counts = (teacher_counts * response_mask).sum(dim=0) / valid_denom
                                    plot_data["Teacher"] = avg_teacher_counts.detach().cpu().numpy()
                                    
                                if "overlap_mask" in batch.batch.keys():
                                    overlap_mask = batch.batch["overlap_mask"].float()
                                    overlap_counts = overlap_mask.sum(dim=-1)
                                    avg_overlap_counts = (overlap_counts * response_mask).sum(dim=0) / valid_denom
                                    plot_data["Overlap"] = avg_overlap_counts.detach().cpu().numpy()
                                
                                plt.figure(figsize=(10, 6))
                                for label, data in plot_data.items():
                                    mean_val = data.mean()
                                    plt.plot(data, label=f"Avg {label} (mean: {mean_val:.2f})")
                                
                                plt.title(f"Avg Candidate Tokens per Position (Step {self.global_steps})")
                                plt.xlabel("Position")
                                plt.ylabel("Avg Candidate Count")
                                plt.legend()
                                plt.grid(True)
                                plt.tight_layout()
                                
                                count_plot = swanlab.Image(plt, caption=f"Candidate Counts (Step {self.global_steps})")
                                plt.close()
                                
                                log_payload = {"viz/candidate_counts": count_plot}
                                
                                if "Overlap" in plot_data and "Student" in plot_data and "Teacher" in plot_data:
                                    ratio_student = plot_data["Overlap"] / (plot_data["Student"] + 1e-6)
                                    ratio_teacher = plot_data["Overlap"] / (plot_data["Teacher"] + 1e-6)
                                    
                                    plt.figure(figsize=(10, 6))
                                    plt.plot(ratio_student, label=f"Overlap / Student (mean: {ratio_student.mean():.2f})", color='tab:blue')
                                    plt.title(f"Overlap / Student Ratio (Step {self.global_steps})")
                                    plt.xlabel("Position")
                                    plt.ylabel("Ratio")
                                    plt.ylim(-0.05, 1.05)
                                    plt.legend()
                                    plt.grid(True)
                                    plt.tight_layout()
                                    
                                    ratio_student_plot = swanlab.Image(plt, caption=f"Overlap / Student Ratio (Step {self.global_steps})")
                                    plt.close()
                                    log_payload["viz/overlap_ratio_student"] = ratio_student_plot

                                    plt.figure(figsize=(10, 6))
                                    plt.plot(ratio_teacher, label=f"Overlap / Teacher (mean: {ratio_teacher.mean():.2f})", color='tab:orange')
                                    plt.title(f"Overlap / Teacher Ratio (Step {self.global_steps})")
                                    plt.xlabel("Position")
                                    plt.ylabel("Ratio")
                                    plt.ylim(-0.05, 1.05)
                                    plt.legend()
                                    plt.grid(True)
                                    plt.tight_layout()
                                    
                                    ratio_teacher_plot = swanlab.Image(plt, caption=f"Overlap / Teacher Ratio (Step {self.global_steps})")
                                    plt.close()
                                    log_payload["viz/overlap_ratio_teacher"] = ratio_teacher_plot

                                logger.log(log_payload, step=self.global_steps)
                                print(f"Logged candidate plots to SwanLab at step {self.global_steps}")
                                
                            except Exception as e:
                                print(f"Error plotting candidate counts: {e}")
                        
                        
                        if "student_valid_counts" in batch.batch.keys():
                             batch.batch.pop("student_valid_counts")
                        if "teacher_valid_counts" in batch.batch.keys():
                             batch.batch.pop("teacher_valid_counts")
                        if "overlap_counts" in batch.batch.keys():
                             batch.batch.pop("overlap_counts")
                        if "adaptive_head_tail_head_counts" in batch.batch.keys():
                            response_mask_float = batch.batch["response_mask"].float()
                            adaptive_denom = response_mask_float.sum().clamp_min(1.0)
                            adaptive_head_counts = batch.batch["adaptive_head_tail_head_counts"].float()
                            adaptive_head_mass = batch.batch["adaptive_head_tail_head_mass"].float()
                            metrics.update({
                                "adaptive_head_tail/head_count_mean": (
                                    adaptive_head_counts * response_mask_float
                                ).sum().detach().item() / adaptive_denom.detach().item(),
                                "adaptive_head_tail/head_count_max": adaptive_head_counts.max().detach().item(),
                                "adaptive_head_tail/head_mass_mean": (
                                    adaptive_head_mass * response_mask_float
                                ).sum().detach().item() / adaptive_denom.detach().item(),
                                "adaptive_head_tail/tail_mass_mean": (
                                    (1.0 - adaptive_head_mass).clamp_min(0.0) * response_mask_float
                                ).sum().detach().item() / adaptive_denom.detach().item(),
                            })
                        candidate_weight_needed_for_actor = (
                            str(batch.meta_info.get("opd_advantage_mode", "fixed")).strip().lower().replace("-", "_")
                            in {"current_kl_is", "current_kl"}
                        )
                        for key in [
                            "adaptive_head_tail_head_counts",
                            "adaptive_head_tail_head_mass",
                        ]:
                            if key in batch.batch.keys():
                                batch.batch.pop(key)
                        if (
                            not candidate_weight_needed_for_actor
                            and "candidate_estimator_weights" in batch.batch.keys()
                        ):
                            batch.batch.pop("candidate_estimator_weights")


                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            self._attach_reward_extra_batch_tensors(batch, reward_extra_infos_dict, metrics)
                    
                    from verl.trainer.ppo.rollout_corr_helper import (
                        compute_rollout_correction_and_add_to_batch,
                        maybe_apply_rollout_correction,
                    )

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    need_recomputation = maybe_apply_rollout_correction(
                        batch=batch,
                        rollout_corr_config=rollout_corr_config,
                        policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                    )
                    if need_recomputation:
                        entropys = None
                        if "old_log_probs" in batch.batch.keys() and "entropys" in batch.batch.keys():
                             entropys = batch.batch["entropys"]
                             print("We don't need to re-merge old_log_probs, it's already there.")

                        else:
                             with marked_timer("old_log_prob", timing_raw, color="blue"):
                                 old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                 entropys = old_log_prob.batch["entropys"]
                                 batch = batch.union(old_log_prob)
                                 self._patch_teacher_rollout_mix_rollout_log_probs(batch)
                                 
                                 for key in ["student_top_k_ids", "student_top_k_log_probs"]:
                                     if key in batch.batch.keys() and key in old_log_prob.batch.keys():
                                         pass

                        if entropys is not None:
                            response_masks = batch.batch["response_mask"]
                            if "format_mask" in batch.batch.keys():
                                response_masks = response_masks * batch.batch["format_mask"].unsqueeze(-1)
                            
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            metrics.update({"actor/entropy": entropy_agg.detach().item()})

                            if "teacher_entropy" in batch.batch.keys():
                                teacher_entropy = batch.batch["teacher_entropy"]
                                teacher_entropy_agg = agg_loss(
                                    loss_mat=teacher_entropy, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                                )
                                metrics.update({"teacher/entropy": teacher_entropy_agg.detach().item()})

                            if "entropys" in batch.batch.keys():
                                batch.batch.pop("entropys")


                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            self._attach_reward_extra_batch_tensors(batch, reward_extra_infos_dict, metrics)
                        batch.batch["token_level_scores"] = reward_tensor

                        if "true_reward_score" in reward_extra_infos_dict:
                            true_reward_val = reward_extra_infos_dict["true_reward_score"]
                            if isinstance(true_reward_val, torch.Tensor):
                                batch.batch["true_reward_score"] = true_reward_val
                            else:
                                batch.batch["true_reward_score"] = torch.as_tensor(
                                    true_reward_val,
                                    device=reward_tensor.device,
                                    dtype=reward_tensor.dtype,
                                )
                        else:
                            batch.batch["true_reward_score"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction weights centrally (once per batch)
                        # This corrects for off-policy issues (policy mismatch, model staleness, etc.)
                        # Also computes off-policy diagnostic metrics (KL, PPL, etc.)
                        if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self._batch_num_repeat(batch),
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
 

                        if "overlap_mask" in batch.batch.keys() and "advantages" in batch.batch.keys():
                            try:
                                overlap_mask = batch.batch["overlap_mask"].float()
                                advantages = batch.batch["advantages"]
                                
                                response_mask = batch.batch["response_mask"]
                                max_len = response_mask.shape[-1]
                                top_k = batch.meta_info.get("log_prob_top_k", 0)
                                strategy = batch.meta_info.get("top_k_strategy", "only_stu")
                                
                                teacher_in_student_mask = batch.batch.get("teacher_in_student_mask", None)
                                
                                student_log_probs = batch.batch.get("student_top_k_log_probs", None)
                                teacher_on_stu_log_probs = batch.batch.get("teacher_on_student_log_probs", None)
                                teacher_log_probs = batch.batch.get("teacher_top_k_log_probs", None)
                                student_on_tch_log_probs = batch.batch.get("student_log_probs_on_teacher_ids", None)

                                if top_k > 0 and advantages.dim() == 3:
                                    adv_k = advantages.shape[-1]
                                    is_union = (strategy == "union" or strategy == "union-intersection") and (adv_k == 2 * top_k)
                                    
                                    global_valid_mask_float = response_mask.unsqueeze(-1).expand(advantages.shape[0], advantages.shape[1], adv_k).float()
                                    global_valid_mask_bool = global_valid_mask_float > 0.5
                                    
                                    if is_union:
                                        
                                        student_overlap = overlap_mask
                                        teacher_overlap = teacher_in_student_mask if teacher_in_student_mask is not None else torch.zeros_like(overlap_mask)
                                        
                                        
                                        student_adv = advantages[:, :, :top_k]
                                        teacher_adv = advantages[:, :, top_k:]
                                        
                                        student_valid = response_mask.unsqueeze(-1).expand_as(student_overlap).bool()
                                        teacher_valid = response_mask.unsqueeze(-1).expand_as(teacher_overlap).bool()
                                        
                                        total_valid_k = student_valid.float().sum()
                                        total_overlap_k = (student_overlap * student_valid.float()).sum()
                                        
                                        if total_valid_k > 0:
                                            metrics["val-topk/overlap_ratio"] = (total_overlap_k / total_valid_k).item()
                                        
                                        mask_inter = (student_overlap > 0.5) & student_valid
                                        if mask_inter.any():
                                            avg_adv_inter = student_adv[mask_inter].mean()
                                            metrics["val-topk/adv_intersection"] = avg_adv_inter.item()
                                            
                                            if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                student_p = torch.exp(student_log_probs)
                                                teacher_p = torch.exp(teacher_on_stu_log_probs)
                                                inter_positions = mask_inter.any(dim=-1)
                                                
                                                student_p_masked = torch.where(mask_inter, student_p, torch.zeros_like(student_p))
                                                teacher_p_masked = torch.where(mask_inter, teacher_p, torch.zeros_like(teacher_p))
                                                student_p_sum = student_p_masked.sum(dim=-1)
                                                teacher_p_sum = teacher_p_masked.sum(dim=-1)
                                                metrics["val-topk/student_p_sum_intersection"] = student_p_sum[inter_positions].mean().item()
                                                metrics["val-topk/teacher_p_sum_intersection"] = teacher_p_sum[inter_positions].mean().item()
                                                
                                                student_p_for_max = torch.where(mask_inter, student_p, torch.full_like(student_p, float('-inf')))
                                                teacher_p_for_max = torch.where(mask_inter, teacher_p, torch.full_like(teacher_p, float('-inf')))
                                                max_stu_idx = student_p_for_max.argmax(dim=-1)
                                                max_tch_idx = teacher_p_for_max.argmax(dim=-1)
                                                
                                                max_stu_p = student_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_max_stu = teacher_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                adv_at_max_stu = student_adv.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                max_tch_p = teacher_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_max_tch = student_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                adv_at_max_tch = student_adv.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                
                                                metrics["val-topk/max_student_p_intersection"] = max_stu_p[inter_positions].mean().item()
                                                metrics["val-topk/teacher_p_at_max_student_intersection"] = tch_p_at_max_stu[inter_positions].mean().item()
                                                metrics["val-topk/adv_at_max_student_intersection"] = adv_at_max_stu[inter_positions].mean().item()
                                                metrics["val-topk/max_teacher_p_intersection"] = max_tch_p[inter_positions].mean().item()
                                                metrics["val-topk/student_p_at_max_teacher_intersection"] = stu_p_at_max_tch[inter_positions].mean().item()
                                                metrics["val-topk/adv_at_max_teacher_intersection"] = adv_at_max_tch[inter_positions].mean().item()
                                                
                                                adv_for_max = torch.where(mask_inter, student_adv, torch.full_like(student_adv, float('-inf')))
                                                adv_for_min = torch.where(mask_inter, student_adv, torch.full_like(student_adv, float('inf')))
                                                max_adv_idx = adv_for_max.argmax(dim=-1)
                                                min_adv_idx = adv_for_min.argmin(dim=-1)
                                                
                                                max_adv = student_adv.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_max_adv = student_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_max_adv = teacher_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                min_adv = student_adv.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_min_adv = student_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_min_adv = teacher_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                
                                                metrics["val-extrema/max_adv_intersection"] = max_adv[inter_positions].mean().item()
                                                metrics["val-extrema/student_p_at_max_adv_intersection"] = stu_p_at_max_adv[inter_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_max_adv_intersection"] = tch_p_at_max_adv[inter_positions].mean().item()
                                                metrics["val-extrema/min_adv_intersection"] = min_adv[inter_positions].mean().item()
                                                metrics["val-extrema/student_p_at_min_adv_intersection"] = stu_p_at_min_adv[inter_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_min_adv_intersection"] = tch_p_at_min_adv[inter_positions].mean().item()
                                        
                                        mask_only_stu = (student_overlap < 0.5) & student_valid
                                        if mask_only_stu.any():
                                            avg_adv_only_stu = student_adv[mask_only_stu].mean()
                                            metrics["val-topk/adv_only_stu"] = avg_adv_only_stu.item()
                                            
                                            if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                only_stu_positions = mask_only_stu.any(dim=-1)
                                                adv_for_max = torch.where(mask_only_stu, student_adv, torch.full_like(student_adv, float('-inf')))
                                                adv_for_min = torch.where(mask_only_stu, student_adv, torch.full_like(student_adv, float('inf')))
                                                max_adv_idx = adv_for_max.argmax(dim=-1)
                                                min_adv_idx = adv_for_min.argmin(dim=-1)
                                                
                                                max_adv = student_adv.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_max_adv = student_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_max_adv = teacher_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                min_adv = student_adv.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_min_adv = student_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_min_adv = teacher_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                
                                                metrics["val-extrema/max_adv_only_stu"] = max_adv[only_stu_positions].mean().item()
                                                metrics["val-extrema/student_p_at_max_adv_only_stu"] = stu_p_at_max_adv[only_stu_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_max_adv_only_stu"] = tch_p_at_max_adv[only_stu_positions].mean().item()
                                                metrics["val-extrema/min_adv_only_stu"] = min_adv[only_stu_positions].mean().item()
                                                metrics["val-extrema/student_p_at_min_adv_only_stu"] = stu_p_at_min_adv[only_stu_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_min_adv_only_stu"] = tch_p_at_min_adv[only_stu_positions].mean().item()
                                        
                                        mask_only_tch = (teacher_overlap < 0.5) & teacher_valid
                                        if mask_only_tch.any():
                                            avg_adv_only_tch = teacher_adv[mask_only_tch].mean()
                                            metrics["val-topk/adv_only_tch"] = avg_adv_only_tch.item()
                                            
                                            if teacher_log_probs is not None and student_on_tch_log_probs is not None:
                                                only_tch_positions = mask_only_tch.any(dim=-1)
                                                teacher_p_tch = torch.exp(teacher_log_probs)
                                                student_p_tch = torch.exp(student_on_tch_log_probs)
                                                
                                                adv_for_max = torch.where(mask_only_tch, teacher_adv, torch.full_like(teacher_adv, float('-inf')))
                                                adv_for_min = torch.where(mask_only_tch, teacher_adv, torch.full_like(teacher_adv, float('inf')))
                                                max_adv_idx = adv_for_max.argmax(dim=-1)
                                                min_adv_idx = adv_for_min.argmin(dim=-1)
                                                
                                                max_adv = teacher_adv.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_max_adv = student_p_tch.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_max_adv = teacher_p_tch.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                min_adv = teacher_adv.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_min_adv = student_p_tch.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_min_adv = teacher_p_tch.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                
                                                metrics["val-extrema/max_adv_only_tch"] = max_adv[only_tch_positions].mean().item()
                                                metrics["val-extrema/student_p_at_max_adv_only_tch"] = stu_p_at_max_adv[only_tch_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_max_adv_only_tch"] = tch_p_at_max_adv[only_tch_positions].mean().item()
                                                metrics["val-extrema/min_adv_only_tch"] = min_adv[only_tch_positions].mean().item()
                                                metrics["val-extrema/student_p_at_min_adv_only_tch"] = stu_p_at_min_adv[only_tch_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_min_adv_only_tch"] = tch_p_at_min_adv[only_tch_positions].mean().item()
                                        
                                        chunk_size = 1024
                                        for start_idx in range(0, max_len, chunk_size):
                                            end_idx = min(start_idx + chunk_size, max_len)
                                            chunk_key = f"{start_idx}_{end_idx}"
                                            
                                            chunk_response_mask = response_mask[:, start_idx:end_idx].bool()
                                            chunk_student_overlap = student_overlap[:, start_idx:end_idx]
                                            chunk_teacher_overlap = teacher_overlap[:, start_idx:end_idx]
                                            chunk_student_adv = student_adv[:, start_idx:end_idx]
                                            chunk_teacher_adv = teacher_adv[:, start_idx:end_idx]
                                            
                                            if not chunk_response_mask.any():
                                                continue
                                            
                                            chunk_student_valid = chunk_response_mask.unsqueeze(-1).expand_as(chunk_student_overlap)
                                            chunk_teacher_valid = chunk_response_mask.unsqueeze(-1).expand_as(chunk_teacher_overlap)
                                            
                                            total_valid = chunk_student_valid.float().sum()
                                            total_overlap = (chunk_student_overlap * chunk_student_valid.float()).sum()
                                            if total_valid > 0:
                                                metrics[f"val-topk/overlap_ratio_chunk_{chunk_key}"] = (total_overlap / total_valid).item()
                                            
                                            mask_inter_c = (chunk_student_overlap > 0.5) & chunk_student_valid
                                            if mask_inter_c.any():
                                                metrics[f"val-topk/adv_intersection_chunk_{chunk_key}"] = chunk_student_adv[mask_inter_c].mean().item()
                                                
                                                if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                    chunk_student_lp = student_log_probs[:, start_idx:end_idx]
                                                    chunk_teacher_lp = teacher_on_stu_log_probs[:, start_idx:end_idx]
                                                    student_p_c = torch.exp(chunk_student_lp)
                                                    teacher_p_c = torch.exp(chunk_teacher_lp)
                                                    inter_pos_c = mask_inter_c.any(dim=-1)
                                                    
                                                    student_p_masked_c = torch.where(mask_inter_c, student_p_c, torch.zeros_like(student_p_c))
                                                    teacher_p_masked_c = torch.where(mask_inter_c, teacher_p_c, torch.zeros_like(teacher_p_c))
                                                    metrics[f"val-topk/student_p_sum_intersection_chunk_{chunk_key}"] = student_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/teacher_p_sum_intersection_chunk_{chunk_key}"] = teacher_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                    
                                                    student_p_for_max_c = torch.where(mask_inter_c, student_p_c, torch.full_like(student_p_c, float('-inf')))
                                                    teacher_p_for_max_c = torch.where(mask_inter_c, teacher_p_c, torch.full_like(teacher_p_c, float('-inf')))
                                                    max_stu_idx_c = student_p_for_max_c.argmax(dim=-1)
                                                    max_tch_idx_c = teacher_p_for_max_c.argmax(dim=-1)
                                                    
                                                    max_stu_p_c = student_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_stu_c = teacher_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_stu_c = chunk_student_adv.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    max_tch_p_c = teacher_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_tch_c = student_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_tch_c = chunk_student_adv.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics[f"val-topk/max_student_p_intersection_chunk_{chunk_key}"] = max_stu_p_c[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/teacher_p_at_max_student_intersection_chunk_{chunk_key}"] = tch_p_at_max_stu_c[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/adv_at_max_student_intersection_chunk_{chunk_key}"] = adv_at_max_stu_c[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/max_teacher_p_intersection_chunk_{chunk_key}"] = max_tch_p_c[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/student_p_at_max_teacher_intersection_chunk_{chunk_key}"] = stu_p_at_max_tch_c[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/adv_at_max_teacher_intersection_chunk_{chunk_key}"] = adv_at_max_tch_c[inter_pos_c].mean().item()
                                                    
                                                    adv_for_max_c = torch.where(mask_inter_c, chunk_student_adv, torch.full_like(chunk_student_adv, float('-inf')))
                                                    adv_for_min_c = torch.where(mask_inter_c, chunk_student_adv, torch.full_like(chunk_student_adv, float('inf')))
                                                    max_adv_idx_c = adv_for_max_c.argmax(dim=-1)
                                                    min_adv_idx_c = adv_for_min_c.argmin(dim=-1)
                                                    
                                                    max_adv_c = chunk_student_adv.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_adv_c = student_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_adv_c = teacher_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    min_adv_c = chunk_student_adv.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_min_adv_c = student_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_min_adv_c = teacher_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics[f"val-extrema/max_adv_intersection_chunk_{chunk_key}"] = max_adv_c[inter_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_max_adv_intersection_chunk_{chunk_key}"] = stu_p_at_max_adv_c[inter_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_max_adv_intersection_chunk_{chunk_key}"] = tch_p_at_max_adv_c[inter_pos_c].mean().item()
                                                    metrics[f"val-extrema/min_adv_intersection_chunk_{chunk_key}"] = min_adv_c[inter_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_min_adv_intersection_chunk_{chunk_key}"] = stu_p_at_min_adv_c[inter_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_min_adv_intersection_chunk_{chunk_key}"] = tch_p_at_min_adv_c[inter_pos_c].mean().item()
                                            
                                            mask_only_stu_c = (chunk_student_overlap < 0.5) & chunk_student_valid
                                            if mask_only_stu_c.any():
                                                metrics[f"val-topk/adv_only_stu_chunk_{chunk_key}"] = chunk_student_adv[mask_only_stu_c].mean().item()
                                                
                                                if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                    only_stu_pos_c = mask_only_stu_c.any(dim=-1)
                                                    adv_for_max_c = torch.where(mask_only_stu_c, chunk_student_adv, torch.full_like(chunk_student_adv, float('-inf')))
                                                    adv_for_min_c = torch.where(mask_only_stu_c, chunk_student_adv, torch.full_like(chunk_student_adv, float('inf')))
                                                    max_adv_idx_c = adv_for_max_c.argmax(dim=-1)
                                                    min_adv_idx_c = adv_for_min_c.argmin(dim=-1)
                                                    
                                                    max_adv_c = chunk_student_adv.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_adv_c = student_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_adv_c = teacher_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    min_adv_c = chunk_student_adv.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_min_adv_c = student_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_min_adv_c = teacher_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics[f"val-extrema/max_adv_only_stu_chunk_{chunk_key}"] = max_adv_c[only_stu_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_max_adv_only_stu_chunk_{chunk_key}"] = stu_p_at_max_adv_c[only_stu_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_max_adv_only_stu_chunk_{chunk_key}"] = tch_p_at_max_adv_c[only_stu_pos_c].mean().item()
                                                    metrics[f"val-extrema/min_adv_only_stu_chunk_{chunk_key}"] = min_adv_c[only_stu_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_min_adv_only_stu_chunk_{chunk_key}"] = stu_p_at_min_adv_c[only_stu_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_min_adv_only_stu_chunk_{chunk_key}"] = tch_p_at_min_adv_c[only_stu_pos_c].mean().item()
                                            
                                            mask_only_tch_c = (chunk_teacher_overlap < 0.5) & chunk_teacher_valid
                                            if mask_only_tch_c.any():
                                                metrics[f"val-topk/adv_only_tch_chunk_{chunk_key}"] = chunk_teacher_adv[mask_only_tch_c].mean().item()
                                                
                                                if teacher_log_probs is not None and student_on_tch_log_probs is not None:
                                                    only_tch_pos_c = mask_only_tch_c.any(dim=-1)
                                                    chunk_teacher_lp = teacher_log_probs[:, start_idx:end_idx]
                                                    chunk_stu_on_tch_lp = student_on_tch_log_probs[:, start_idx:end_idx]
                                                    teacher_p_tch_c = torch.exp(chunk_teacher_lp)
                                                    student_p_tch_c = torch.exp(chunk_stu_on_tch_lp)
                                                    
                                                    adv_for_max_c = torch.where(mask_only_tch_c, chunk_teacher_adv, torch.full_like(chunk_teacher_adv, float('-inf')))
                                                    adv_for_min_c = torch.where(mask_only_tch_c, chunk_teacher_adv, torch.full_like(chunk_teacher_adv, float('inf')))
                                                    max_adv_idx_c = adv_for_max_c.argmax(dim=-1)
                                                    min_adv_idx_c = adv_for_min_c.argmin(dim=-1)
                                                    
                                                    max_adv_c = chunk_teacher_adv.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_adv_c = student_p_tch_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_adv_c = teacher_p_tch_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    min_adv_c = chunk_teacher_adv.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_min_adv_c = student_p_tch_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_min_adv_c = teacher_p_tch_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics[f"val-extrema/max_adv_only_tch_chunk_{chunk_key}"] = max_adv_c[only_tch_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_max_adv_only_tch_chunk_{chunk_key}"] = stu_p_at_max_adv_c[only_tch_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_max_adv_only_tch_chunk_{chunk_key}"] = tch_p_at_max_adv_c[only_tch_pos_c].mean().item()
                                                    metrics[f"val-extrema/min_adv_only_tch_chunk_{chunk_key}"] = min_adv_c[only_tch_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_min_adv_only_tch_chunk_{chunk_key}"] = stu_p_at_min_adv_c[only_tch_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_min_adv_only_tch_chunk_{chunk_key}"] = tch_p_at_min_adv_c[only_tch_pos_c].mean().item()
                                    
                                    else:
                                        if strategy == "only_tch" and "teacher_in_student_mask" in batch.batch:
                                            tch_in_stu_mask = batch.batch["teacher_in_student_mask"]
                                            global_valid_mask_float_k = response_mask.unsqueeze(-1).expand_as(tch_in_stu_mask).float()
                                            global_valid_mask_bool_k = global_valid_mask_float_k > 0.5
                                            
                                            global_total_valid_k = global_valid_mask_float_k.sum()
                                            global_total_overlap_k = (tch_in_stu_mask * global_valid_mask_float_k).sum()
                                            
                                            if global_total_valid_k > 0:
                                                metrics["val-topk/overlap_ratio"] = (global_total_overlap_k / global_total_valid_k).item()
                                            
                                            global_mask_inter = (tch_in_stu_mask > 0.5) & global_valid_mask_bool_k
                                            if global_mask_inter.any():
                                                global_avg_adv_inter = advantages[global_mask_inter].mean()
                                                metrics["val-topk/adv_intersection"] = global_avg_adv_inter.item()
                                                
                                                student_on_tch_log_probs = batch.batch.get("student_log_probs_on_teacher_ids", None)
                                                teacher_top_k_lp = batch.batch.get("teacher_top_k_log_probs", None)
                                                if student_on_tch_log_probs is not None and teacher_top_k_lp is not None:
                                                    student_p = torch.exp(student_on_tch_log_probs)
                                                    teacher_p = torch.exp(teacher_top_k_lp)
                                                    inter_positions = global_mask_inter.any(dim=-1)
                                                    
                                                    student_p_masked = torch.where(global_mask_inter, student_p, torch.zeros_like(student_p))
                                                    teacher_p_masked = torch.where(global_mask_inter, teacher_p, torch.zeros_like(teacher_p))
                                                    metrics["val-topk/student_p_sum_intersection"] = student_p_masked.sum(dim=-1)[inter_positions].mean().item()
                                                    metrics["val-topk/teacher_p_sum_intersection"] = teacher_p_masked.sum(dim=-1)[inter_positions].mean().item()
                                                    
                                                    student_p_for_max = torch.where(global_mask_inter, student_p, torch.full_like(student_p, float('-inf')))
                                                    teacher_p_for_max = torch.where(global_mask_inter, teacher_p, torch.full_like(teacher_p, float('-inf')))
                                                    max_stu_idx = student_p_for_max.argmax(dim=-1)
                                                    max_tch_idx = teacher_p_for_max.argmax(dim=-1)
                                                    
                                                    max_stu_p = student_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_stu = teacher_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_stu = advantages.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    max_tch_p = teacher_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_tch = student_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_tch = advantages.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics["val-topk/max_student_p_intersection"] = max_stu_p[inter_positions].mean().item()
                                                    metrics["val-topk/teacher_p_at_max_student_intersection"] = tch_p_at_max_stu[inter_positions].mean().item()
                                                    metrics["val-topk/adv_at_max_student_intersection"] = adv_at_max_stu[inter_positions].mean().item()
                                                    metrics["val-topk/max_teacher_p_intersection"] = max_tch_p[inter_positions].mean().item()
                                                    metrics["val-topk/student_p_at_max_teacher_intersection"] = stu_p_at_max_tch[inter_positions].mean().item()
                                                    metrics["val-topk/adv_at_max_teacher_intersection"] = adv_at_max_tch[inter_positions].mean().item()
                                                    
                                                    adv_for_max = torch.where(global_mask_inter, advantages, torch.full_like(advantages, float('-inf')))
                                                    adv_for_min = torch.where(global_mask_inter, advantages, torch.full_like(advantages, float('inf')))
                                                    max_adv_idx = adv_for_max.argmax(dim=-1)
                                                    min_adv_idx = adv_for_min.argmin(dim=-1)
                                                    
                                                    max_adv = advantages.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_adv = student_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_adv = teacher_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    min_adv = advantages.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_min_adv = student_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_min_adv = teacher_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics["val-extrema/max_adv_intersection"] = max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/student_p_at_max_adv_intersection"] = stu_p_at_max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/teacher_p_at_max_adv_intersection"] = tch_p_at_max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/min_adv_intersection"] = min_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/student_p_at_min_adv_intersection"] = stu_p_at_min_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/teacher_p_at_min_adv_intersection"] = tch_p_at_min_adv[inter_positions].mean().item()
                                                
                                            global_mask_only_tch = (tch_in_stu_mask < 0.5) & global_valid_mask_bool_k
                                            if global_mask_only_tch.any():
                                                global_avg_adv_only_tch = advantages[global_mask_only_tch].mean()
                                                metrics["val-topk/adv_only_tch"] = global_avg_adv_only_tch.item()

                                            chunk_size = 1024
                                            for start_idx in range(0, max_len, chunk_size):
                                                end_idx = min(start_idx + chunk_size, max_len)
                                                chunk_key = f"{start_idx}_{end_idx}"
                                                
                                                chunk_response_mask = response_mask[:, start_idx:end_idx].bool()
                                                chunk_tch_in_stu = tch_in_stu_mask[:, start_idx:end_idx]
                                                chunk_adv = advantages[:, start_idx:end_idx]
                                                
                                                if not chunk_response_mask.any():
                                                    continue
                                                
                                                chunk_valid_mask = chunk_response_mask.unsqueeze(-1).expand_as(chunk_tch_in_stu)
                                                
                                                total_valid_k = chunk_valid_mask.sum()
                                                total_overlap_k = (chunk_tch_in_stu * chunk_valid_mask.float()).sum()
                                                if total_valid_k > 0:
                                                    metrics[f"val-topk/overlap_ratio_chunk_{chunk_key}"] = (total_overlap_k / total_valid_k).item()
                                                
                                                mask_inter = (chunk_tch_in_stu > 0.5) & chunk_valid_mask
                                                if mask_inter.any():
                                                    metrics[f"val-topk/adv_intersection_chunk_{chunk_key}"] = chunk_adv[mask_inter].mean().item()
                                                    
                                                    if student_on_tch_log_probs is not None and teacher_top_k_lp is not None:
                                                        chunk_stu_lp = student_on_tch_log_probs[:, start_idx:end_idx]
                                                        chunk_tch_lp = teacher_top_k_lp[:, start_idx:end_idx]
                                                        student_p_c = torch.exp(chunk_stu_lp)
                                                        teacher_p_c = torch.exp(chunk_tch_lp)
                                                        inter_pos_c = mask_inter.any(dim=-1)
                                                        
                                                        student_p_masked_c = torch.where(mask_inter, student_p_c, torch.zeros_like(student_p_c))
                                                        teacher_p_masked_c = torch.where(mask_inter, teacher_p_c, torch.zeros_like(teacher_p_c))
                                                        metrics[f"val-topk/student_p_sum_intersection_chunk_{chunk_key}"] = student_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/teacher_p_sum_intersection_chunk_{chunk_key}"] = teacher_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                        
                                                        student_p_for_max_c = torch.where(mask_inter, student_p_c, torch.full_like(student_p_c, float('-inf')))
                                                        teacher_p_for_max_c = torch.where(mask_inter, teacher_p_c, torch.full_like(teacher_p_c, float('-inf')))
                                                        max_stu_idx_c = student_p_for_max_c.argmax(dim=-1)
                                                        max_tch_idx_c = teacher_p_for_max_c.argmax(dim=-1)
                                                        
                                                        max_stu_p_c = student_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_max_stu_c = teacher_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        adv_at_max_stu_c = chunk_adv.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        max_tch_p_c = teacher_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_max_tch_c = student_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        adv_at_max_tch_c = chunk_adv.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        
                                                        metrics[f"val-topk/max_student_p_intersection_chunk_{chunk_key}"] = max_stu_p_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/teacher_p_at_max_student_intersection_chunk_{chunk_key}"] = tch_p_at_max_stu_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/adv_at_max_student_intersection_chunk_{chunk_key}"] = adv_at_max_stu_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/max_teacher_p_intersection_chunk_{chunk_key}"] = max_tch_p_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/student_p_at_max_teacher_intersection_chunk_{chunk_key}"] = stu_p_at_max_tch_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/adv_at_max_teacher_intersection_chunk_{chunk_key}"] = adv_at_max_tch_c[inter_pos_c].mean().item()
                                                        
                                                        adv_for_max_c = torch.where(mask_inter, chunk_adv, torch.full_like(chunk_adv, float('-inf')))
                                                        adv_for_min_c = torch.where(mask_inter, chunk_adv, torch.full_like(chunk_adv, float('inf')))
                                                        max_adv_idx_c = adv_for_max_c.argmax(dim=-1)
                                                        min_adv_idx_c = adv_for_min_c.argmin(dim=-1)
                                                        
                                                        max_adv_c = chunk_adv.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_max_adv_c = student_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_max_adv_c = teacher_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        min_adv_c = chunk_adv.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_min_adv_c = student_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_min_adv_c = teacher_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        
                                                        metrics[f"val-extrema/max_adv_intersection_chunk_{chunk_key}"] = max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/student_p_at_max_adv_intersection_chunk_{chunk_key}"] = stu_p_at_max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/teacher_p_at_max_adv_intersection_chunk_{chunk_key}"] = tch_p_at_max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/min_adv_intersection_chunk_{chunk_key}"] = min_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/student_p_at_min_adv_intersection_chunk_{chunk_key}"] = stu_p_at_min_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/teacher_p_at_min_adv_intersection_chunk_{chunk_key}"] = tch_p_at_min_adv_c[inter_pos_c].mean().item()
                                                
                                                mask_only_tch = (chunk_tch_in_stu < 0.5) & chunk_valid_mask
                                                if mask_only_tch.any():
                                                    metrics[f"val-topk/adv_only_tch_chunk_{chunk_key}"] = chunk_adv[mask_only_tch].mean().item()
                                        else:
                                            global_valid_mask_float_k = response_mask.unsqueeze(-1).expand_as(overlap_mask).float()
                                            global_valid_mask_bool_k = global_valid_mask_float_k > 0.5
                                            
                                            global_total_valid_k = global_valid_mask_float_k.sum()
                                            global_total_overlap_k = (overlap_mask * global_valid_mask_float_k).sum()
                                            
                                            if global_total_valid_k > 0:
                                                metrics["val-topk/overlap_ratio"] = (global_total_overlap_k / global_total_valid_k).item()
                                            
                                            global_mask_inter = (overlap_mask > 0.5) & global_valid_mask_bool_k
                                            if global_mask_inter.any():
                                                global_avg_adv_inter = advantages[global_mask_inter].mean()
                                                metrics["val-topk/adv_intersection"] = global_avg_adv_inter.item()
                                                
                                                if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                    student_p = torch.exp(student_log_probs)
                                                    teacher_p = torch.exp(teacher_on_stu_log_probs)
                                                    inter_positions = global_mask_inter.any(dim=-1)
                                                    
                                                    student_p_masked = torch.where(global_mask_inter, student_p, torch.zeros_like(student_p))
                                                    teacher_p_masked = torch.where(global_mask_inter, teacher_p, torch.zeros_like(teacher_p))
                                                    metrics["val-topk/student_p_sum_intersection"] = student_p_masked.sum(dim=-1)[inter_positions].mean().item()
                                                    metrics["val-topk/teacher_p_sum_intersection"] = teacher_p_masked.sum(dim=-1)[inter_positions].mean().item()
                                                    
                                                    student_p_for_max = torch.where(global_mask_inter, student_p, torch.full_like(student_p, float('-inf')))
                                                    teacher_p_for_max = torch.where(global_mask_inter, teacher_p, torch.full_like(teacher_p, float('-inf')))
                                                    max_stu_idx = student_p_for_max.argmax(dim=-1)
                                                    max_tch_idx = teacher_p_for_max.argmax(dim=-1)
                                                    
                                                    max_stu_p = student_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_stu = teacher_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_stu = advantages.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    max_tch_p = teacher_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_tch = student_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_tch = advantages.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics["val-topk/max_student_p_intersection"] = max_stu_p[inter_positions].mean().item()
                                                    metrics["val-topk/teacher_p_at_max_student_intersection"] = tch_p_at_max_stu[inter_positions].mean().item()
                                                    metrics["val-topk/adv_at_max_student_intersection"] = adv_at_max_stu[inter_positions].mean().item()
                                                    metrics["val-topk/max_teacher_p_intersection"] = max_tch_p[inter_positions].mean().item()
                                                    metrics["val-topk/student_p_at_max_teacher_intersection"] = stu_p_at_max_tch[inter_positions].mean().item()
                                                    metrics["val-topk/adv_at_max_teacher_intersection"] = adv_at_max_tch[inter_positions].mean().item()
                                                    
                                                    adv_for_max = torch.where(global_mask_inter, advantages, torch.full_like(advantages, float('-inf')))
                                                    adv_for_min = torch.where(global_mask_inter, advantages, torch.full_like(advantages, float('inf')))
                                                    max_adv_idx = adv_for_max.argmax(dim=-1)
                                                    min_adv_idx = adv_for_min.argmin(dim=-1)
                                                    
                                                    max_adv = advantages.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_adv = student_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_adv = teacher_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    min_adv = advantages.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_min_adv = student_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_min_adv = teacher_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics["val-extrema/max_adv_intersection"] = max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/student_p_at_max_adv_intersection"] = stu_p_at_max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/teacher_p_at_max_adv_intersection"] = tch_p_at_max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/min_adv_intersection"] = min_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/student_p_at_min_adv_intersection"] = stu_p_at_min_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/teacher_p_at_min_adv_intersection"] = tch_p_at_min_adv[inter_positions].mean().item()
                                                
                                            global_mask_only_stu = (overlap_mask < 0.5) & global_valid_mask_bool_k
                                            if global_mask_only_stu.any():
                                                global_avg_adv_only_stu = advantages[global_mask_only_stu].mean()
                                                metrics["val-topk/adv_only_stu"] = global_avg_adv_only_stu.item()

                                            chunk_size = 1024
                                            
                                            for start_idx in range(0, max_len, chunk_size):
                                                end_idx = min(start_idx + chunk_size, max_len)
                                                chunk_key = f"{start_idx}_{end_idx}"
                                                
                                                chunk_response_mask = response_mask[:, start_idx:end_idx].bool()
                                                chunk_overlap_mask = overlap_mask[:, start_idx:end_idx]
                                                chunk_adv = advantages[:, start_idx:end_idx]
                                                
                                                if not chunk_response_mask.any():
                                                    continue
                                                
                                                chunk_valid_mask = chunk_response_mask.unsqueeze(-1).expand_as(chunk_overlap_mask)
                                                
                                                total_valid_k = chunk_valid_mask.sum()
                                                total_overlap_k = (chunk_overlap_mask * chunk_valid_mask.float()).sum()
                                                
                                                if total_valid_k > 0:
                                                    metrics[f"val-topk/overlap_ratio_chunk_{chunk_key}"] = (total_overlap_k / total_valid_k).item()
                                                
                                                mask_inter = (chunk_overlap_mask > 0.5) & chunk_valid_mask
                                                if mask_inter.any():
                                                    avg_adv_inter = chunk_adv[mask_inter].mean()
                                                    metrics[f"val-topk/adv_intersection_chunk_{chunk_key}"] = avg_adv_inter.item()
                                                    
                                                    if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                        chunk_student_lp = student_log_probs[:, start_idx:end_idx]
                                                        chunk_teacher_lp = teacher_on_stu_log_probs[:, start_idx:end_idx]
                                                        student_p_c = torch.exp(chunk_student_lp)
                                                        teacher_p_c = torch.exp(chunk_teacher_lp)
                                                        inter_pos_c = mask_inter.any(dim=-1)
                                                        
                                                        student_p_masked_c = torch.where(mask_inter, student_p_c, torch.zeros_like(student_p_c))
                                                        teacher_p_masked_c = torch.where(mask_inter, teacher_p_c, torch.zeros_like(teacher_p_c))
                                                        metrics[f"val-topk/student_p_sum_intersection_chunk_{chunk_key}"] = student_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/teacher_p_sum_intersection_chunk_{chunk_key}"] = teacher_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                        
                                                        student_p_for_max_c = torch.where(mask_inter, student_p_c, torch.full_like(student_p_c, float('-inf')))
                                                        teacher_p_for_max_c = torch.where(mask_inter, teacher_p_c, torch.full_like(teacher_p_c, float('-inf')))
                                                        max_stu_idx_c = student_p_for_max_c.argmax(dim=-1)
                                                        max_tch_idx_c = teacher_p_for_max_c.argmax(dim=-1)
                                                        
                                                        max_stu_p_c = student_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_max_stu_c = teacher_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        adv_at_max_stu_c = chunk_adv.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        max_tch_p_c = teacher_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_max_tch_c = student_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        adv_at_max_tch_c = chunk_adv.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        
                                                        metrics[f"val-topk/max_student_p_intersection_chunk_{chunk_key}"] = max_stu_p_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/teacher_p_at_max_student_intersection_chunk_{chunk_key}"] = tch_p_at_max_stu_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/adv_at_max_student_intersection_chunk_{chunk_key}"] = adv_at_max_stu_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/max_teacher_p_intersection_chunk_{chunk_key}"] = max_tch_p_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/student_p_at_max_teacher_intersection_chunk_{chunk_key}"] = stu_p_at_max_tch_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/adv_at_max_teacher_intersection_chunk_{chunk_key}"] = adv_at_max_tch_c[inter_pos_c].mean().item()
                                                        
                                                        adv_for_max_c = torch.where(mask_inter, chunk_adv, torch.full_like(chunk_adv, float('-inf')))
                                                        adv_for_min_c = torch.where(mask_inter, chunk_adv, torch.full_like(chunk_adv, float('inf')))
                                                        max_adv_idx_c = adv_for_max_c.argmax(dim=-1)
                                                        min_adv_idx_c = adv_for_min_c.argmin(dim=-1)
                                                        
                                                        max_adv_c = chunk_adv.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_max_adv_c = student_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_max_adv_c = teacher_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        min_adv_c = chunk_adv.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_min_adv_c = student_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_min_adv_c = teacher_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        
                                                        metrics[f"val-extrema/max_adv_intersection_chunk_{chunk_key}"] = max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/student_p_at_max_adv_intersection_chunk_{chunk_key}"] = stu_p_at_max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/teacher_p_at_max_adv_intersection_chunk_{chunk_key}"] = tch_p_at_max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/min_adv_intersection_chunk_{chunk_key}"] = min_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/student_p_at_min_adv_intersection_chunk_{chunk_key}"] = stu_p_at_min_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/teacher_p_at_min_adv_intersection_chunk_{chunk_key}"] = tch_p_at_min_adv_c[inter_pos_c].mean().item()
                                                    
                                                mask_only_stu = (chunk_overlap_mask < 0.5) & chunk_valid_mask
                                                if mask_only_stu.any():
                                                    avg_adv_only_stu = chunk_adv[mask_only_stu].mean()
                                                    metrics[f"val-topk/adv_only_stu_chunk_{chunk_key}"] = avg_adv_only_stu.item()
                                            
                            except Exception as e:
                                print(f"Error computing Top-K metrics: {e}")
                                import traceback
                                traceback.print_exc()
                    
                    if self.config.trainer.get("is_plot", False) and (self.global_steps == 1 or self.global_steps % 10 == 0):
                        try:
                            import matplotlib.pyplot as plt
                            import swanlab
                            
                            if "teacher_entropy" in batch.batch.keys():
                                teacher_entropy = batch.batch["teacher_entropy"]
                                
                                if "token_level_advantage_direct" in batch.batch.keys():
                                    adv = batch.batch["token_level_advantage_direct"]
                                else:
                                    adv = batch.batch["advantages"]

                                if adv.dim() == 3:
                                    adv = adv.sum(dim=-1)
                                
                                response_mask = batch.batch["response_mask"]
                                
                                teacher_entropy_cpu = teacher_entropy.detach().cpu()
                                adv_cpu = adv.detach().cpu()
                                mask_cpu = response_mask.detach().cpu().bool()
                                
                                batch_size, seq_len = teacher_entropy_cpu.shape
                                positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
                                
                                valid_indices = mask_cpu
                                valid_positions = positions[valid_indices].numpy()
                                valid_entropy = teacher_entropy_cpu[valid_indices].numpy()
                                valid_adv = adv_cpu[valid_indices].numpy()
                                
                                plt.figure(figsize=(10, 6))
                                plt.scatter(valid_positions, valid_entropy, alpha=0.05, s=1)
                                plt.title(f"Teacher Entropy vs Position (Step {self.global_steps})")
                                plt.xlabel("Position")
                                plt.ylabel("Teacher Entropy")
                                plt.tight_layout()
                                entropy_plot = swanlab.Image(plt, caption=f"Teacher Entropy vs Position (Step {self.global_steps})")
                                plt.close()
                                
                                plt.figure(figsize=(10, 6))
                                plt.scatter(valid_positions, valid_adv, alpha=0.05, s=1)
                                plt.title(f"Advantage vs Position (Step {self.global_steps})")
                                plt.xlabel("Position")
                                plt.ylabel("Advantage")
                                plt.tight_layout()
                                adv_plot = swanlab.Image(plt, caption=f"Advantage vs Position (Step {self.global_steps})")
                                plt.close()

                                mask_float = mask_cpu.float()
                                
                                sum_entropy = (teacher_entropy_cpu * mask_float).sum(dim=0)
                                sum_adv = (adv_cpu * mask_float).sum(dim=0)
                                count_per_pos = mask_float.sum(dim=0)
                                
                                valid_pos_mask = count_per_pos > 0
                                avg_entropy = torch.zeros_like(sum_entropy)
                                avg_adv = torch.zeros_like(sum_adv)
                                
                                avg_entropy[valid_pos_mask] = sum_entropy[valid_pos_mask] / count_per_pos[valid_pos_mask]
                                avg_adv[valid_pos_mask] = sum_adv[valid_pos_mask] / count_per_pos[valid_pos_mask]

                                avg_adv_inter = None
                                avg_adv_only_stu = None
                                overlap_mask_cpu = None
                                
                                if "overlap_mask" in batch.batch.keys():
                                    overlap_mask_cpu = batch.batch["overlap_mask"].detach().cpu()

                                if overlap_mask_cpu is not None and adv_cpu.dim() == 3:
                                    
                                    mask_cpu_k = mask_cpu.unsqueeze(-1).expand_as(overlap_mask_cpu)
                                    
                                    mask_inter = (overlap_mask_cpu > 0.5) & mask_cpu_k
                                    
                                    sum_adv_inter = (adv_cpu * mask_inter.float()).sum(dim=(0, 2))
                                    count_inter = mask_inter.float().sum(dim=(0, 2))
                                    
                                    avg_adv_inter = torch.zeros(seq_len)
                                    valid_inter = count_inter > 0
                                    avg_adv_inter[valid_inter] = sum_adv_inter[valid_inter] / count_inter[valid_inter]
                                    
                                    mask_only_stu = (overlap_mask_cpu < 0.5) & mask_cpu_k
                                    
                                    sum_adv_only_stu = (adv_cpu * mask_only_stu.float()).sum(dim=(0, 2))
                                    count_only_stu = mask_only_stu.float().sum(dim=(0, 2))
                                    
                                    avg_adv_only_stu = torch.zeros(seq_len)
                                    valid_only_stu = count_only_stu > 0
                                    avg_adv_only_stu[valid_only_stu] = sum_adv_only_stu[valid_only_stu] / count_only_stu[valid_only_stu]
                                
                                if valid_pos_mask.any():
                                    max_valid_pos = torch.where(valid_pos_mask)[0].max().item()
                                    plot_positions = torch.arange(max_valid_pos + 1).numpy()
                                    plot_avg_entropy = avg_entropy[:max_valid_pos + 1].numpy()
                                    plot_avg_adv = avg_adv[:max_valid_pos + 1].numpy()
                                    
                                    plot_avg_adv_inter = avg_adv_inter[:max_valid_pos + 1].numpy() if avg_adv_inter is not None else None
                                    plot_avg_adv_only_stu = avg_adv_only_stu[:max_valid_pos + 1].numpy() if avg_adv_only_stu is not None else None
                                else:
                                    plot_positions = np.array([])
                                    plot_avg_entropy = np.array([])
                                    plot_avg_adv = np.array([])
                                    plot_avg_adv_inter = None
                                    plot_avg_adv_only_stu = None

                                plt.figure(figsize=(10, 6))
                                plt.plot(plot_positions, plot_avg_entropy)
                                plt.title(f"Avg Teacher Entropy vs Position (Step {self.global_steps})")
                                plt.xlabel("Position")
                                plt.ylabel("Avg Teacher Entropy")
                                plt.grid(True)
                                plt.tight_layout()
                                avg_entropy_plot = swanlab.Image(plt, caption=f"Avg Teacher Entropy vs Position (Step {self.global_steps})")
                                plt.close()

                                plt.figure(figsize=(10, 6))
                                plt.plot(plot_positions, plot_avg_adv, label="Total")
                                if plot_avg_adv_inter is not None:
                                    plt.plot(plot_positions, plot_avg_adv_inter, label="Intersection")
                                if plot_avg_adv_only_stu is not None:
                                    plt.plot(plot_positions, plot_avg_adv_only_stu, label="Only Stu")
                                    
                                plt.title(f"Avg Advantage vs Position (Step {self.global_steps})")
                                plt.xlabel("Position")
                                plt.ylabel("Avg Advantage")
                                plt.legend()
                                plt.grid(True)
                                plt.tight_layout()
                                avg_adv_plot = swanlab.Image(plt, caption=f"Avg Advantage vs Position (Step {self.global_steps})")
                                plt.close()
                                
                                swanlab.log({
                                    "viz/teacher_entropy_scatter": entropy_plot,
                                    "viz/advantage_scatter": adv_plot,
                                    "viz/avg_teacher_entropy_line": avg_entropy_plot,
                                    "viz/avg_advantage_line": avg_adv_plot
                                }, step=self.global_steps)
                                
                                print(f"Logged 4 plots to SwanLab at step {self.global_steps}.")
                                
                                del teacher_entropy_cpu, adv_cpu, mask_cpu, mask_float
                                del valid_positions, valid_entropy, valid_adv, positions
                                del sum_entropy, sum_adv, count_per_pos, avg_entropy, avg_adv
                                del plot_positions, plot_avg_entropy, plot_avg_adv
                                del entropy_plot, adv_plot, avg_entropy_plot, avg_adv_plot
                            else:
                                print("teacher_entropy not found in batch. Skipping plot.")
                                
                        except Exception as e:
                            print(f"Error plotting/logging: {e}")
                            import traceback
                            traceback.print_exc()

                    opd_advantage_mode = str(batch.meta_info.get("opd_advantage_mode", "fixed")).strip().lower()
                    keys_to_pop = [
                        "teacher_top_k_ids",
                        "teacher_top_k_log_probs",
                        "teacher_entropy",
                        "overlap_mask",
                        "teacher_in_student_mask",
                        "student_log_probs_on_teacher_ids",
                    ]
                    if not requires_teacher_on_student_log_probs(opd_advantage_mode):
                        keys_to_pop.append("teacher_on_student_log_probs")
                    for key in keys_to_pop:
                        if key in batch.batch.keys():
                            batch.batch.pop(key)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        if is_decomposed_pi_old_mode(opd_advantage_mode):
                            if self._adaptive_ppo_update_enabled() or self._ess_samplek_resample_enabled():
                                raise ValueError(
                                    "decomposed_pi_old requires fixed rollout old_log_probs inside actor PPO epochs; "
                                    "adaptive_ppo_update_enable and ess_samplek_resample_enable must both be false."
                                )
                            if self._prefix_drift_enabled():
                                raise ValueError(
                                    "decomposed_pi_old computes its own prefix IS; prefix_drift_enable must be false."
                                )
                        if self._adaptive_ppo_update_enabled() and self._ess_samplek_resample_enabled():
                            raise ValueError(
                                "adaptive_ppo_update_enable and ess_samplek_resample_enable cannot both be true."
                            )
                        if (
                            self._prefix_drift_enabled()
                            and not self._adaptive_ppo_update_enabled()
                            and not self._ess_samplek_resample_enabled()
                        ):
                            raise ValueError(
                                "prefix_drift_enable=True currently requires trainer-controlled repeated updates "
                                "(adaptive_ppo_update_enable=True or ess_samplek_resample_enable=True)."
                            )
                        if self._adaptive_ppo_update_enabled():
                            self._run_adaptive_ppo_actor_updates(batch, metrics, timing_raw)
                        elif self._ess_samplek_resample_enabled():
                            self._run_ess_samplek_resample_actor_updates(batch, metrics, timing_raw)
                        else:
                            with marked_timer("update_actor", timing_raw, color="red"):
                                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    print(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
