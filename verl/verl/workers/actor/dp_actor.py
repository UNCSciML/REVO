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
Single Process Actor
"""

import logging
import math
import os
import time

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.adaptive_update import compute_samplek_probe_diagnostics
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.trainer.ppo.eos_future_correction import (
    EOS_FUTURE_MODE_DYNAMIC_H1_OFFPOLICY_CLIPPED,
    EOS_FUTURE_MODE_DYNAMIC_H1_REMAINING_HORIZON_RELU,
    EOS_FUTURE_MODE_DYNAMIC_H1_SQRT_REMAINING_HORIZON_RELU,
    EOS_FUTURE_MODE_FIXED_SIGNED,
    OPD_EOS_FUTURE_CORRECTION_MASK_KEY,
    OPD_EOS_FUTURE_VALUE_KEY,
    normalize_eos_future_mode,
    prepare_dynamic_h1_offpolicy_eos_future_loss,
    prepare_eos_future_correction_loss,
)
from verl.trainer.ppo.forced_eos_diagnostic import (
    FORCED_EOS_ESTIMATOR_WEIGHTS_KEY,
    apply_forced_eos_diagnostic_influence,
    sample_forced_eos_candidates,
    validate_forced_eos_diagnostic_configuration,
)
from verl.trainer.ppo.opd_decomposed import (
    OPD_Q_MIXTURE_PRIOR_ALPHA_KEY,
    OPD_Q_MIXTURE_SOURCE_WEIGHTS_KEY,
    OPD_ROLLOUT_REFERENCE_LOG_PROBS_KEY,
    add_samplek_entropy_advantages,
    apply_adaptive_head_tail_negative_elu,
    apply_raw_opd_advantage_clip,
    apply_samplek_advantage_centering,
    apply_samplek_influence_clip,
    apply_samplek_loo_variance_filter,
    compute_q_mixture_samplek_teacher_advantages,
    compute_decomposed_local_opd_loss,
    compute_sampled_token_rkl_diagnostics,
    compute_sampled_token_proximal_rkl_loss,
    is_decomposed_pi_old_mode,
    normalize_opd_mode,
    normalize_q_mixture_teacher_advantage_mode,
    normalize_samplek_advantage_centering,
    normalize_samplek_loo_variance_threshold_mode,
    validate_adaptive_head_tail_negative_elu_configuration,
    validate_samplek_influence_clip_configuration,
    validate_samplek_loo_variance_filter_configuration,
)
from verl.trainer.ppo.prefix_drift import (
    PREFIX_DRIFT_RAW_WEIGHTS_KEY,
    PREFIX_DRIFT_WEIGHTS_KEY,
    compute_prefix_drift,
)
from verl.trainer.ppo.student_eos_diagnostics import eos_log_probs_from_logits
from verl.trainer.ppo.teacher_deficit_residual import (
    TEACHER_DEFICIT_RESIDUAL_IDS_KEY,
    TEACHER_DEFICIT_RESIDUAL_LOG_PROBS_KEY,
    prepare_teacher_deficit_residual_loss,
    validate_teacher_deficit_residual_configuration,
)
from verl.trainer.ppo.terminal_aware_opd import (
    OPD_TERMINAL_BEHAVIOR_EOS_LOG_PROBS_KEY,
    OPD_TERMINAL_BEHAVIOR_SECONDARY_LOG_PROBS_KEY,
    OPD_TERMINAL_STUDENT_TOPM_IDS_KEY,
    OPD_TERMINAL_STUDENT_TOPM_LOG_PROBS_KEY,
    OPD_TERMINAL_STUDENT_SECONDARY_LOG_PROBS_KEY,
    OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY,
    OPD_TERMINAL_TEACHER_TOPM_LOG_PROBS_KEY,
    apply_samplek_eos_negative_relu,
    normalize_terminal_objective_mode,
    prepare_terminal_conservative_kl_objective,
    prepare_terminal_safe_continue_objective,
    prepare_terminal_teacher_remap_only_objective,
    prepare_terminal_aware_objective,
    sample_terminal_objective_candidates,
    select_terminal_topk_candidate_estimator_weights,
    validate_terminal_aware_configuration,
)
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    slice_input_tensor,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DAPO_ANSWER_FORMAT_MASK_KEY = "dapo_answer_format_mask"


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    @staticmethod
    def _parse_bool(value, default: bool, name: str) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        value = str(value).strip().lower()
        if value in ("", "auto", "none", "null"):
            return default
        if value in ("1", "true", "yes", "y", "on"):
            return True
        if value in ("0", "false", "no", "n", "off"):
            return False
        raise ValueError(f"Unknown {name} value: {value}")

    @staticmethod
    def _terminal_candidate_reference(
        *,
        current_log_probs: torch.Tensor,
        proposal_log_probs: torch.Tensor,
        candidate_reuse_enabled: bool,
    ) -> tuple[torch.Tensor, bool]:
        if candidate_reuse_enabled:
            return proposal_log_probs, True
        return current_log_probs.detach(), False

    @staticmethod
    def _compute_candidate_reuse_corrections(
        *,
        sampled_log_probs: torch.Tensor,
        rollout_reference_log_probs: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        candidate_proposal_log_probs: torch.Tensor,
        teacher_candidate_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        prefix_method: str,
        prefix_log_clip: float | None,
        prefix_log_clip_mode: str,
        prefix_normalize: str,
        prefix_position_beta: float,
        prefix_position_hmax: int | None,
        prefix_detach: bool,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        prefix_output = compute_prefix_drift(
            current_log_probs=sampled_log_probs,
            reference_log_probs=rollout_reference_log_probs,
            response_mask=response_mask,
            method=prefix_method,
            log_clip=prefix_log_clip,
            log_clip_mode=prefix_log_clip_mode,
            normalize=prefix_normalize,
            position_beta=prefix_position_beta,
            position_hmax=prefix_position_hmax,
            detach=prefix_detach,
        )
        candidate_metrics = compute_samplek_probe_diagnostics(
            current_log_probs=candidate_log_probs,
            reference_log_probs=candidate_proposal_log_probs,
            response_mask=response_mask,
            teacher_log_probs=teacher_candidate_log_probs,
        )
        response_diagnostics = compute_sampled_token_rkl_diagnostics(
            current_log_probs=sampled_log_probs,
            reference_log_probs=rollout_reference_log_probs,
            response_mask=response_mask,
        )
        mask = response_mask.to(device=sampled_log_probs.device, dtype=torch.float32)
        metrics = dict(prefix_output.metrics)
        for key, value in candidate_metrics.items():
            suffix = key.removeprefix("candidate_probe/")
            metrics[f"samplek_candidate_reuse/candidate_{suffix}"] = value
        metrics["samplek_candidate_reuse/response_rkl_k3_mean"] = (
            response_diagnostics.k3_mean.detach().item()
        )
        metrics["samplek_candidate_reuse/response_log_ratio_abs_mean"] = (
            (response_diagnostics.log_ratio.abs() * mask).sum() / mask.sum().clamp_min(1.0)
        ).detach().item()
        return prefix_output.weights, metrics

    @staticmethod
    def _canonical_candidate_mode(value) -> str:
        mode = str(value or "topk").strip().lower().replace("-", "_")
        aliases = {
            "top_k": "topk",
            "student_topk": "topk",
            "only_stu": "topk",
            "topk_theoretical": "topk_theory",
            "topk_no_renorm": "topk_theory",
            "topk_no_renorm_plus_one": "topk_theory",
            "topk_unbiased": "topk_theory",
            "sample": "sample_stu",
            "sample_student": "sample_stu",
            "student_sample": "sample_stu",
            "sample_k": "sample_stu",
            "uniform": "sample_uniform",
            "uniform_sample": "sample_uniform",
            "sample_uniform_vocab": "sample_uniform",
            "uniform_vocab": "sample_uniform",
            "adaptive_ht": "adaptive_head_tail",
            "head_tail": "adaptive_head_tail",
            "adaptive_headtail": "adaptive_head_tail",
            "sample_head_tail": "adaptive_head_tail",
            "full": "full_vocab",
            "fullvocab": "full_vocab",
            "all_vocab": "full_vocab",
        }
        mode = aliases.get(mode, mode)
        if mode not in (
            "topk",
            "topk_theory",
            "sample_stu",
            "sample_uniform",
            "adaptive_head_tail",
            "full_vocab",
        ):
            raise ValueError(f"Unknown log_prob_candidate_mode: {value}")
        return mode

    @staticmethod
    def _masked_scalar_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=values.device, dtype=values.dtype)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _masked_normalized_ess(weights: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        valid = weights.float()[mask.to(device=weights.device) > 0.5]
        if valid.numel() == 0:
            return weights.new_tensor(1.0, dtype=torch.float32)
        sum_w = valid.sum()
        sum_w2 = valid.square().sum().clamp_min(eps)
        return (sum_w.square() / (sum_w2 * valid.numel())).clamp(min=0.0, max=1.0)

    @classmethod
    def _add_q_mixture_offpolicy_diagnostics(
        cls,
        *,
        metrics: dict[str, float],
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        student_mask: torch.Tensor,
        prefix_weights: torch.Tensor,
        raw_prefix_weights: torch.Tensor | None,
        source_weights: torch.Tensor | None,
        metric_prefix: str = "opd_q_offpolicy_diag",
    ) -> None:

        mask = response_mask.to(device=advantages.device, dtype=advantages.dtype)
        student = student_mask.to(device=advantages.device, dtype=advantages.dtype) * mask
        teacher = (mask - student).clamp_min(0.0)
        prefix = prefix_weights.to(device=advantages.device, dtype=advantages.dtype)
        raw_prefix = prefix if raw_prefix_weights is None else raw_prefix_weights.to(
            device=advantages.device,
            dtype=advantages.dtype,
        )
        if prefix.shape != mask.shape:
            raise ValueError(
                "Q-mixture diagnostics prefix weights must match response_mask, "
                f"got weights={prefix.shape}, mask={mask.shape}."
            )
        if raw_prefix.shape != mask.shape:
            raise ValueError(
                "Q-mixture diagnostics raw prefix weights must match response_mask, "
                f"got weights={raw_prefix.shape}, mask={mask.shape}."
            )
        if student.shape != mask.shape:
            raise ValueError(
                "Q-mixture diagnostics student mask must match response_mask, "
                f"got student={student.shape}, mask={mask.shape}."
            )

        if advantages.dim() == 3:
            token_advantage = advantages.sum(dim=-1)
        elif advantages.dim() == 2:
            token_advantage = advantages
        else:
            raise ValueError(f"Q-mixture diagnostics expect 2D or 3D advantages, got {advantages.shape}.")
        token_advantage = token_advantage.to(dtype=prefix.dtype)
        weighted_advantage = prefix * token_advantage
        raw_weighted_advantage = raw_prefix * token_advantage
        clip_bias = (raw_prefix - prefix) * token_advantage

        source_weights_for_metrics = None
        if source_weights is not None:
            source_weights_for_metrics = source_weights.to(device=advantages.device, dtype=advantages.dtype)
            if source_weights_for_metrics.shape != mask.shape:
                raise ValueError(
                    "Q-mixture diagnostics source weights must match response_mask, "
                    f"got source_weights={source_weights_for_metrics.shape}, mask={mask.shape}."
                )

        total_tokens = mask.sum().clamp_min(1.0)
        total_abs_mass = (weighted_advantage.abs() * mask).sum().clamp_min(1e-12)
        total_signed_mass_abs = (weighted_advantage * mask).sum().abs().clamp_min(1e-12)

        metrics[f"{metric_prefix}/enabled"] = 1.0
        metrics[f"{metric_prefix}/raw_prefix_available"] = float(raw_prefix_weights is not None)
        metrics[f"{metric_prefix}/total_abs_mass"] = total_abs_mass.detach().item()
        metrics[f"{metric_prefix}/total_signed_mass"] = (weighted_advantage * mask).sum().detach().item()

        def add_source(name: str, source_mask: torch.Tensor) -> None:
            source_mask = source_mask.to(dtype=prefix.dtype)
            source_tokens = source_mask.sum().clamp_min(1.0)
            signed_mass = (weighted_advantage * source_mask).sum()
            abs_mass = (weighted_advantage.abs() * source_mask).sum()
            raw_signed_mass = (raw_weighted_advantage * source_mask).sum()
            metrics[f"{metric_prefix}/{name}/token_fraction"] = (
                source_mask.sum() / total_tokens
            ).detach().item()
            metrics[f"{metric_prefix}/{name}/adv_mean"] = cls._masked_scalar_mean(
                token_advantage, source_mask
            ).detach().item()
            metrics[f"{metric_prefix}/{name}/adv_abs_mean"] = cls._masked_scalar_mean(
                token_advantage.abs(), source_mask
            ).detach().item()
            metrics[f"{metric_prefix}/{name}/prefix_weight_mean"] = cls._masked_scalar_mean(
                prefix, source_mask
            ).detach().item()
            metrics[f"{metric_prefix}/{name}/prefix_weight_ess"] = cls._masked_normalized_ess(
                prefix, source_mask
            ).detach().item()
            metrics[f"{metric_prefix}/{name}/raw_prefix_weight_mean"] = cls._masked_scalar_mean(
                raw_prefix, source_mask
            ).detach().item()
            metrics[f"{metric_prefix}/{name}/raw_prefix_weight_ess"] = cls._masked_normalized_ess(
                raw_prefix, source_mask
            ).detach().item()
            metrics[f"{metric_prefix}/{name}/weighted_adv_mean"] = (
                signed_mass / source_tokens
            ).detach().item()
            metrics[f"{metric_prefix}/{name}/weighted_adv_abs_mean"] = (
                abs_mass / source_tokens
            ).detach().item()
            metrics[f"{metric_prefix}/{name}/effective_abs_mass_fraction"] = (
                abs_mass / total_abs_mass
            ).detach().item()
            metrics[f"{metric_prefix}/{name}/signed_mass_fraction"] = (
                signed_mass / total_signed_mass_abs
            ).detach().item()
            metrics[f"{metric_prefix}/{name}/raw_weighted_adv_mean"] = (
                raw_signed_mass / source_tokens
            ).detach().item()
            metrics[f"{metric_prefix}/{name}/clip_bias_mean"] = cls._masked_scalar_mean(
                clip_bias, source_mask
            ).detach().item()
            metrics[f"{metric_prefix}/{name}/clip_bias_abs_mean"] = cls._masked_scalar_mean(
                clip_bias.abs(), source_mask
            ).detach().item()
            if source_weights_for_metrics is not None:
                metrics[f"{metric_prefix}/{name}/source_norm_weight_mean"] = cls._masked_scalar_mean(
                    source_weights_for_metrics, source_mask
                ).detach().item()

        add_source("all", mask)
        add_source("student", student)
        add_source("teacher", teacher)

        response_length = mask.shape[-1]
        for bucket_idx in range(4):
            start = response_length * bucket_idx // 4
            end = response_length * (bucket_idx + 1) // 4
            bucket = torch.zeros_like(mask)
            bucket[:, start:end] = mask[:, start:end]
            bucket_name = f"position_q{bucket_idx + 1}"
            add_source(f"{bucket_name}/student", student * bucket)
            add_source(f"{bucket_name}/teacher", teacher * bucket)

    @staticmethod
    def _canonical_opd_loss_type(value) -> str:
        mode = str(value or "sample_k_reverse_kl").strip().lower().replace("-", "_")
        aliases = {
            "kl": "sample_k_reverse_kl",
            "rkl": "sample_k_reverse_kl",
            "reverse_kl": "sample_k_reverse_kl",
            "sample_k_rkl": "sample_k_reverse_kl",
            "sample_k_kl": "sample_k_reverse_kl",
            "on_logprob_mse": "sample_k_on_logprob_mse",
            "logprob_mse": "sample_k_on_logprob_mse",
            "sample_k_mse": "sample_k_on_logprob_mse",
            "mse_pathwise": "sample_k_logprob_mse_pathwise",
            "logprob_mse_pathwise": "sample_k_logprob_mse_pathwise",
            "pathwise_mse": "sample_k_logprob_mse_pathwise",
            "chi2": "sample_k_chi_square",
            "chisquare": "sample_k_chi_square",
            "chi_square": "sample_k_chi_square",
            "sample_k_chi2": "sample_k_chi_square",
            "sample_k_chisquare": "sample_k_chi_square",
        }
        mode = aliases.get(mode, mode)
        if mode not in (
            "sample_k_reverse_kl",
            "sample_k_on_logprob_mse",
            "sample_k_logprob_mse_pathwise",
            "sample_k_chi_square",
        ):
            raise ValueError(f"Unknown opd_loss_type: {value}")
        return mode

    @staticmethod
    def _canonical_chi_square_baseline(value) -> str:
        mode = str(value or "mean").strip().lower().replace("-", "_")
        aliases = {
            "0": "none",
            "false": "none",
            "zero": "none",
            "no": "none",
            "sample_mean": "mean",
            "center": "mean",
            "loo": "loo_mean",
            "loo_mean": "loo_mean",
            "leave_one_out": "loo_mean",
            "leave_one_out_mean": "loo_mean",
        }
        mode = aliases.get(mode, mode)
        if mode not in ("none", "mean", "loo_mean"):
            raise ValueError(f"Unknown chi_square_baseline: {value}")
        return mode

    @staticmethod
    def _parse_optional_float(value, default=None, name: str = "value"):
        if value is None:
            return default
        if isinstance(value, (float, int)):
            return float(value)
        value = str(value).strip().lower()
        if value in ("", "auto", "none", "null", "false"):
            return default
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Unknown {name} value: {value}") from exc

    @staticmethod
    def _parse_optional_int(value, default=None, name: str = "value"):
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip().lower()
            if value in ("", "auto", "none", "null"):
                return default
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown {name} value: {value}") from exc

    @staticmethod
    def _parse_positive_int(value, default: int, name: str) -> int:
        if value is None:
            return default
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got bool.")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
        if parsed <= 0:
            raise ValueError(f"{name} must be positive, got {parsed}.")
        return parsed

    @staticmethod
    def _make_tail_mask(response_mask: torch.Tensor, tail_tokens: int) -> torch.Tensor:
        if response_mask.dim() != 2:
            raise ValueError(f"response_mask must be 2D, got {response_mask.shape}.")
        positions = torch.arange(response_mask.size(1), device=response_mask.device).unsqueeze(0)
        lengths = response_mask.float().sum(dim=-1, keepdim=True)
        tail_start = (lengths - float(tail_tokens)).clamp_min(0.0)
        return response_mask * (positions >= tail_start).to(dtype=response_mask.dtype)

    @staticmethod
    def _compute_opd_logprob_weight(
        student_logp: torch.Tensor,
        teacher_logp: torch.Tensor,
        opd_loss_type: str,
        sample_k_kl_plus_one: bool = False,
        valid_mask: torch.Tensor | None = None,
        center: bool = False,
        normalize: bool = False,
        clip: float | None = None,
        chi_square_baseline: str = "mean",
    ) -> torch.Tensor:
        f = student_logp.float() - teacher_logp.float()
        if opd_loss_type == "sample_k_reverse_kl":
            weight = f
            if sample_k_kl_plus_one:
                weight = weight + 1.0
            return weight

        if opd_loss_type == "sample_k_on_logprob_mse":
            weight = f.square() + 2.0 * f
        elif opd_loss_type == "sample_k_logprob_mse_pathwise":
            weight = 2.0 * f
        elif opd_loss_type == "sample_k_chi_square":
            log_ratio = f.clamp(min=-20.0, max=20.0)
            weight = 2.0 * torch.exp(log_ratio)
            baseline_mode = DataParallelPPOActor._canonical_chi_square_baseline(chi_square_baseline)
            if baseline_mode != "none":
                if valid_mask is None:
                    stat_mask = torch.ones_like(weight, dtype=torch.bool)
                    denom = torch.full_like(weight[..., :1], weight.size(-1), dtype=weight.dtype)
                else:
                    stat_mask = valid_mask.bool()
                    denom = stat_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
                masked_weight = torch.where(stat_mask, weight, torch.zeros_like(weight))
                if baseline_mode == "mean":
                    baseline = masked_weight.sum(dim=-1, keepdim=True) / denom
                elif baseline_mode == "loo_mean":
                    denom_minus_one = (denom - 1.0).clamp_min(1.0)
                    baseline = (masked_weight.sum(dim=-1, keepdim=True) - masked_weight) / denom_minus_one
                    baseline = torch.where(denom > 1.0, baseline, torch.zeros_like(baseline))
                else:
                    raise ValueError(f"Unknown chi_square_baseline: {chi_square_baseline}")
                weight = torch.where(stat_mask, weight - baseline, torch.zeros_like(weight))
        else:
            raise ValueError(f"Unknown opd_loss_type: {opd_loss_type}")

        if valid_mask is not None:
            valid_mask = valid_mask.bool()
            weight = torch.where(valid_mask, weight, torch.zeros_like(weight))

        if center or normalize:
            if valid_mask is None:
                denom = torch.full_like(weight[..., :1], weight.size(-1), dtype=weight.dtype)
                stat_mask = torch.ones_like(weight, dtype=torch.bool)
            else:
                stat_mask = valid_mask
                denom = stat_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)

            mean = torch.where(stat_mask, weight, torch.zeros_like(weight)).sum(dim=-1, keepdim=True) / denom
            if center:
                weight = torch.where(stat_mask, weight - mean, torch.zeros_like(weight))

            if normalize:
                stat_weight = torch.where(stat_mask, weight, torch.zeros_like(weight))
                stat_mean = stat_weight.sum(dim=-1, keepdim=True) / denom
                var = torch.where(stat_mask, (stat_weight - stat_mean).square(), torch.zeros_like(weight)).sum(
                    dim=-1, keepdim=True
                ) / denom
                weight = torch.where(denom > 1.0, weight / (var.sqrt() + 1e-6), weight)

        if clip is not None and clip > 0:
            weight = weight.clamp(min=-clip, max=clip)

        return weight

    @staticmethod
    def _sample_candidate_ids(log_probs_all: torch.Tensor, num_samples: int, replacement: bool) -> torch.Tensor:
        probs = torch.exp(log_probs_all)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            return torch.multinomial(probs, num_samples=num_samples, replacement=replacement)
        except RuntimeError:
            return torch.multinomial(probs.float(), num_samples=num_samples, replacement=replacement)

    @staticmethod
    def _sample_adaptive_head_tail_candidate_ids(
        log_probs_all: torch.Tensor,
        num_samples: int,
        gamma: float = 0.5,
        k2_min: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_samples <= 0:
            raise ValueError(f"adaptive_head_tail requires num_samples > 0, got {num_samples}")
        k2_min = int(k2_min)
        if k2_min < 1 or k2_min > num_samples:
            raise ValueError(
                f"adaptive_head_tail_k2_min must be in [1, {num_samples}], got {k2_min}"
            )

        original_shape = log_probs_all.shape[:-1]
        vocab_size = log_probs_all.size(-1)
        flat_log_probs = log_probs_all.reshape(-1, vocab_size).float()
        num_rows = flat_log_probs.size(0)
        max_head = num_samples - k2_min

        if max_head > 0:
            top_logp, top_ids = torch.topk(flat_log_probs, k=max_head, dim=-1)
            top_p = torch.exp(top_logp)
            z_before = torch.cumsum(top_p, dim=-1) - top_p
            slot_idx = torch.arange(max_head, device=flat_log_probs.device, dtype=top_p.dtype)
            remaining_budget = (float(num_samples) - slot_idx).clamp_min(1.0)
            threshold = float(gamma) * (1.0 - z_before).clamp_min(0.0) / remaining_budget
            accept = top_p >= threshold
            head_mask = accept.to(torch.int64).cumprod(dim=-1).bool()
            head_counts = head_mask.sum(dim=-1)
            head_mass = (top_p * head_mask.to(top_p.dtype)).sum(dim=-1)
        else:
            top_ids = torch.empty(num_rows, 0, dtype=torch.long, device=flat_log_probs.device)
            top_p = torch.empty(num_rows, 0, dtype=flat_log_probs.dtype, device=flat_log_probs.device)
            head_mask = torch.empty(num_rows, 0, dtype=torch.bool, device=flat_log_probs.device)
            head_counts = torch.zeros(num_rows, dtype=torch.long, device=flat_log_probs.device)
            head_mass = torch.zeros(num_rows, dtype=flat_log_probs.dtype, device=flat_log_probs.device)

        tail_log_probs = flat_log_probs.clone()
        if max_head > 0:
            current_top_tail_logp = tail_log_probs.gather(dim=-1, index=top_ids)
            neg_inf = torch.full_like(current_top_tail_logp, -float("inf"))
            tail_head_src = torch.where(head_mask, neg_inf, current_top_tail_logp)
            tail_log_probs.scatter_(dim=-1, index=top_ids, src=tail_head_src)

        tail_log_probs = tail_log_probs - tail_log_probs.max(dim=-1, keepdim=True).values
        tail_weights = torch.exp(tail_log_probs)
        tail_weights = torch.nan_to_num(tail_weights, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            tail_ids = torch.multinomial(tail_weights, num_samples=num_samples, replacement=True)
        except RuntimeError:
            tail_ids = torch.multinomial(tail_weights.float(), num_samples=num_samples, replacement=True)

        slot_ids = torch.arange(num_samples, device=flat_log_probs.device).unsqueeze(0).expand(num_rows, -1)
        head_slot_mask = slot_ids < head_counts.unsqueeze(-1)
        tail_slot_offsets = (slot_ids - head_counts.unsqueeze(-1)).clamp_min(0)
        selected_tail_ids = tail_ids.gather(dim=-1, index=tail_slot_offsets)

        head_ids_padded = torch.zeros(num_rows, num_samples, dtype=torch.long, device=flat_log_probs.device)
        head_weights_padded = torch.zeros(num_rows, num_samples, dtype=flat_log_probs.dtype, device=flat_log_probs.device)
        if max_head > 0:
            head_ids_padded[:, :max_head] = top_ids
            head_weights_padded[:, :max_head] = top_p

        candidate_ids = torch.where(head_slot_mask, head_ids_padded, selected_tail_ids)
        k2 = (num_samples - head_counts).to(flat_log_probs.dtype).clamp_min(1.0)
        tail_estimator_weight = (1.0 - head_mass).clamp_min(0.0).unsqueeze(-1) / k2.unsqueeze(-1)
        candidate_weights = torch.where(
            head_slot_mask, head_weights_padded, tail_estimator_weight.expand_as(head_weights_padded)
        )

        candidate_ids = candidate_ids.view(*original_shape, num_samples)
        candidate_weights = candidate_weights.view(*original_shape, num_samples)
        head_counts = head_counts.view(*original_shape)
        head_mass = head_mass.view(*original_shape)
        return candidate_ids, candidate_weights, head_counts, head_mass

    @staticmethod
    def _sample_uniform_candidate_ids(log_probs_all: torch.Tensor, num_samples: int) -> torch.Tensor:
        vocab_size = log_probs_all.size(-1)
        if num_samples > vocab_size:
            raise ValueError(f"num_samples={num_samples} must be <= vocab_size={vocab_size}")
        sample_shape = (*log_probs_all.shape[:-1], num_samples)
        topk_ids = torch.randint(vocab_size, sample_shape, device=log_probs_all.device)
        if num_samples <= 1:
            return topk_ids

        for _ in range(max(10, num_samples // 2)):
            sorted_ids, order = topk_ids.sort(dim=-1)
            duplicate_sorted = torch.zeros_like(sorted_ids, dtype=torch.bool)
            duplicate_sorted[..., 1:] = sorted_ids[..., 1:] == sorted_ids[..., :-1]
            duplicate_mask = torch.zeros_like(duplicate_sorted).scatter(dim=-1, index=order, src=duplicate_sorted)
            duplicate_count = int(duplicate_mask.sum().item())
            if duplicate_count == 0:
                return topk_ids
            topk_ids[duplicate_mask] = torch.randint(
                vocab_size, (duplicate_count,), device=log_probs_all.device, dtype=topk_ids.dtype
            )

        flat_ids = topk_ids.reshape(-1, num_samples)
        duplicate_rows = torch.zeros(flat_ids.size(0), dtype=torch.bool, device=flat_ids.device)
        sorted_flat, _ = flat_ids.sort(dim=-1)
        duplicate_rows |= (sorted_flat[:, 1:] == sorted_flat[:, :-1]).any(dim=-1)
        for row_idx in duplicate_rows.nonzero(as_tuple=False).flatten().tolist():
            flat_ids[row_idx] = torch.randperm(vocab_size, device=flat_ids.device)[:num_samples]
        return topk_ids

    @staticmethod
    def _select_topm_excluding_token(
        log_probs_all: torch.Tensor,
        *,
        num_tokens: int,
        excluded_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = int(num_tokens)
        if num_tokens <= 0 or num_tokens >= log_probs_all.size(-1):
            raise ValueError(
                "terminal Top-M must be positive and smaller than the vocabulary size, "
                f"got M={num_tokens} and vocab={log_probs_all.size(-1)}."
            )
        selection_log_probs = log_probs_all.detach().float().clone()
        selection_log_probs[..., int(excluded_token_id)] = -torch.inf
        _, token_ids = torch.topk(selection_log_probs, k=num_tokens, dim=-1)
        return token_ids, log_probs_all.gather(dim=-1, index=token_ids)

    def _get_actor_vocab_size(self) -> int | None:
        module = self.actor_module
        for attr in ("config",):
            config = getattr(module, attr, None)
            vocab_size = getattr(config, "vocab_size", None)
            if vocab_size is not None:
                return int(vocab_size)
        wrapped = getattr(module, "module", None)
        if wrapped is not None:
            config = getattr(wrapped, "config", None)
            vocab_size = getattr(config, "vocab_size", None)
            if vocab_size is not None:
                return int(vocab_size)
        get_embeddings = getattr(module, "get_input_embeddings", None)
        if get_embeddings is not None:
            embeddings = get_embeddings()
            weight = getattr(embeddings, "weight", None)
            if weight is not None:
                return int(weight.shape[0])
        return None

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        top_k=0,
        student_top_k_ids=None,
        candidate_mode="topk",
        sample_replacement=True,
        adaptive_head_tail_gamma=0.5,
        adaptive_head_tail_k2_min=1,
        eos_token_id: int | None = None,
        diagnostic_forced_eos_enable: bool = False,
        terminal_aware_enable: bool = False,
        terminal_secondary_token_id: int | None = None,
        terminal_objective_mode: str = "anchor_kl",
        terminal_topm: int = 0,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            topk_ids: # (bs, response_len, k)
            topk_log_probs: # (bs, response_len, k)
            candidate_weights: # (bs, response_len, k) for adaptive_head_tail
            adaptive_head_counts: # (bs, response_len) for adaptive_head_tail
            adaptive_head_mass: # (bs, response_len) for adaptive_head_tail
            eos_log_probs: # (bs, response_len)
            forced_eos_estimator_weights: # (bs, response_len, k) for the forced-EOS diagnostic
            terminal_topm_ids: # (bs, response_len, M) for coarse terminal KL
            terminal_topm_log_probs: # (bs, response_len, M) for coarse terminal KL
            terminal_secondary_log_probs: # (bs, response_len) diagnostic for student e2
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            topk_ids = None
            topk_log_probs = None
            candidate_weights = None
            adaptive_head_counts = None
            adaptive_head_mass = None
            eos_log_probs = None
            forced_eos_estimator_weights = None
            terminal_topm_ids = None
            terminal_topm_log_probs = None
            terminal_secondary_log_probs = None
            candidate_mode = self._canonical_candidate_mode(candidate_mode)
            return_full_vocab = candidate_mode == "full_vocab"
            sample_replacement = self._parse_bool(sample_replacement, default=True, name="sample_k_replacement")
            response_terminal_mask = None
            if terminal_aware_enable:
                if eos_token_id is None:
                    raise ValueError("terminal-aware OPD requires an EOS token id.")
                if "response_mask" not in micro_batch:
                    raise ValueError("terminal-aware OPD requires response_mask during candidate sampling.")
                response_terminal_mask = micro_batch["response_mask"].bool() & micro_batch["responses"].eq(
                    int(eos_token_id)
                )
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                terminal_mask_rmpad = None
                if response_terminal_mask is not None:
                    full_terminal_mask = torch.zeros(
                        (batch_size, seqlen),
                        dtype=torch.bool,
                        device=response_terminal_mask.device,
                    )
                    full_terminal_mask[:, -response_length - 1 : -1] = response_terminal_mask
                    terminal_mask_rmpad = index_first_axis(
                        rearrange(full_terminal_mask.unsqueeze(-1), "b s ... -> (b s) ..."),
                        indices,
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    if terminal_mask_rmpad is not None:
                        terminal_mask_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                            terminal_mask_rmpad,
                            position_ids_rmpad=None,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
                if terminal_mask_rmpad is not None:
                    terminal_mask_rmpad = terminal_mask_rmpad.squeeze(0).bool()

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                
                need_logits = top_k > 0 or return_full_vocab or eos_token_id is not None
                eos_log_probs_rmpad = None
                terminal_secondary_log_probs_rmpad = None

                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    
                    need_topk = top_k > 0
                    if need_topk or return_full_vocab:
                        log_probs_all = torch.log_softmax(logits_rmpad, dim=-1)
                        log_probs = log_probs_all.gather(
                            dim=-1, index=input_ids_rmpad_rolled.unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                    if eos_token_id is not None:
                        if need_topk or return_full_vocab:
                            eos_log_probs_rmpad = log_probs_all[..., eos_token_id]
                        else:
                            eos_log_probs_rmpad = eos_log_probs_from_logits(logits_rmpad, eos_token_id)
                    if (
                        terminal_objective_mode in {"conservative_kl", "teacher_remap_only"}
                        and terminal_secondary_token_id is not None
                    ):
                        terminal_secondary_log_probs_rmpad = log_probs_all[
                            ..., int(terminal_secondary_token_id)
                        ]

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             topk_ids = student_top_k_ids
                             if student_top_k_ids.ndim == 3:
                                 
                                 
                                 
                                 if student_top_k_ids.shape[1] != seqlen:
                                     full_student_top_k_ids = torch.zeros((batch_size, seqlen, top_k), 
                                                                         dtype=student_top_k_ids.dtype, 
                                                                         device=student_top_k_ids.device)
                                     full_student_top_k_ids[:, -response_length-1:-1, :] = student_top_k_ids
                                     student_top_k_ids = full_student_top_k_ids

                                 flat_ids = student_top_k_ids.view(-1, top_k)
                                 
                                 topk_ids_rmpad = flat_ids[indices]
                                 if self.use_ulysses_sp:
                                     topk_ids_rmpad = slice_input_tensor(
                                         topk_ids_rmpad,
                                         dim=0,
                                         padding=True,
                                     )
                                 
                                 topk_ids = topk_ids_rmpad
                                 
                             else:
                                 pass

                        else:
                             if candidate_mode in ("topk", "topk_theory"):
                                 _, topk_ids = torch.topk(logits_rmpad, k=top_k, dim=-1)
                             elif candidate_mode == "sample_stu":
                                 if terminal_aware_enable:
                                     topk_ids = sample_terminal_objective_candidates(
                                         log_probs_all,
                                         terminal_mask=terminal_mask_rmpad,
                                         num_candidates=top_k,
                                         eos_token_id=eos_token_id,
                                         secondary_token_id=terminal_secondary_token_id,
                                         objective_mode=terminal_objective_mode,
                                     )
                                 elif diagnostic_forced_eos_enable:
                                     if eos_token_id is None:
                                         raise ValueError("forced-EOS diagnostic requires an EOS token id.")
                                     forced_eos_sample = sample_forced_eos_candidates(
                                         log_probs_all,
                                         num_candidates=top_k,
                                         eos_token_id=eos_token_id,
                                     )
                                     topk_ids = forced_eos_sample.candidate_ids
                                     forced_eos_estimator_weights = forced_eos_sample.estimator_weights
                                 else:
                                     topk_ids = self._sample_candidate_ids(
                                         log_probs_all, num_samples=top_k, replacement=sample_replacement
                                     )
                             elif candidate_mode == "sample_uniform":
                                 topk_ids = self._sample_uniform_candidate_ids(log_probs_all, num_samples=top_k)
                             elif candidate_mode == "adaptive_head_tail":
                                 if not sample_replacement:
                                     raise ValueError("adaptive_head_tail currently requires sample_k_replacement=True")
                                 (
                                     topk_ids,
                                     candidate_weights,
                                     adaptive_head_counts,
                                     adaptive_head_mass,
                                 ) = self._sample_adaptive_head_tail_candidate_ids(
                                     log_probs_all,
                                     num_samples=top_k,
                                     gamma=adaptive_head_tail_gamma,
                                     k2_min=adaptive_head_tail_k2_min,
                                 )

                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)
                    if terminal_topm > 0:
                        terminal_topm_ids, terminal_topm_log_probs = self._select_topm_excluding_token(
                            log_probs_all,
                            num_tokens=terminal_topm,
                            excluded_token_id=eos_token_id,
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if return_full_vocab:
                         log_probs_all = gather_outputs_and_unpad(
                            log_probs_all,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                    if eos_log_probs_rmpad is not None:
                        eos_log_probs_rmpad = gather_outputs_and_unpad(
                            eos_log_probs_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if terminal_secondary_log_probs_rmpad is not None:
                        terminal_secondary_log_probs_rmpad = gather_outputs_and_unpad(
                            terminal_secondary_log_probs_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if top_k > 0:
                         topk_ids = gather_outputs_and_unpad(
                            topk_ids,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                         topk_log_probs = gather_outputs_and_unpad(
                            topk_log_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                         if candidate_weights is not None:
                            candidate_weights = gather_outputs_and_unpad(
                                candidate_weights,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                            adaptive_head_counts = gather_outputs_and_unpad(
                                adaptive_head_counts.unsqueeze(-1),
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            ).squeeze(-1)
                            adaptive_head_mass = gather_outputs_and_unpad(
                                adaptive_head_mass.unsqueeze(-1),
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            ).squeeze(-1)
                         if forced_eos_estimator_weights is not None:
                            forced_eos_estimator_weights = gather_outputs_and_unpad(
                                forced_eos_estimator_weights,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                    if terminal_topm_ids is not None:
                        terminal_topm_ids = gather_outputs_and_unpad(
                            terminal_topm_ids,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        terminal_topm_log_probs = gather_outputs_and_unpad(
                            terminal_topm_log_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if eos_token_id is not None:
                    full_eos_log_probs = pad_input(
                        hidden_states=eos_log_probs_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if terminal_secondary_log_probs_rmpad is not None:
                    full_terminal_secondary_log_probs = pad_input(
                        hidden_states=terminal_secondary_log_probs_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                
                if top_k > 0:
                    full_topk_ids = pad_input(
                        hidden_states=topk_ids,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if candidate_weights is not None:
                        full_candidate_weights = pad_input(
                            hidden_states=candidate_weights,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        full_adaptive_head_counts = pad_input(
                            hidden_states=adaptive_head_counts.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        ).squeeze(-1)
                        full_adaptive_head_mass = pad_input(
                            hidden_states=adaptive_head_mass.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        ).squeeze(-1)
                    if forced_eos_estimator_weights is not None:
                        full_forced_eos_estimator_weights = pad_input(
                            hidden_states=forced_eos_estimator_weights,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                if terminal_topm_ids is not None:
                    full_terminal_topm_ids = pad_input(
                        hidden_states=terminal_topm_ids,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_terminal_topm_log_probs = pad_input(
                        hidden_states=terminal_topm_log_probs,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if return_full_vocab:
                    full_vocab_log_probs = pad_input(
                        hidden_states=log_probs_all,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if eos_token_id is not None:
                    eos_log_probs = full_eos_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
                if terminal_secondary_log_probs_rmpad is not None:
                    terminal_secondary_log_probs = full_terminal_secondary_log_probs.squeeze(-1)[
                        :, -response_length - 1 : -1
                    ]
                
                if top_k > 0:
                    topk_ids = full_topk_ids[:, -response_length - 1 : -1, :]
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]
                    if candidate_weights is not None:
                        candidate_weights = full_candidate_weights[:, -response_length - 1 : -1, :]
                        adaptive_head_counts = full_adaptive_head_counts[:, -response_length - 1 : -1]
                        adaptive_head_mass = full_adaptive_head_mass[:, -response_length - 1 : -1]
                    if forced_eos_estimator_weights is not None:
                        forced_eos_estimator_weights = full_forced_eos_estimator_weights[
                            :, -response_length - 1 : -1, :
                        ]
                if return_full_vocab:
                    topk_log_probs = full_vocab_log_probs[:, -response_length - 1 : -1, :]
                if terminal_topm_ids is not None:
                    terminal_topm_ids = full_terminal_topm_ids[:, -response_length - 1 : -1, :]
                    terminal_topm_log_probs = full_terminal_topm_log_probs[
                        :, -response_length - 1 : -1, :
                    ]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                
                need_logits = top_k > 0 or return_full_vocab or eos_token_id is not None
                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    
                    need_topk = top_k > 0
                    if need_topk or return_full_vocab:
                        log_probs_all = torch.log_softmax(logits, dim=-1)
                        log_probs = log_probs_all.gather(
                            dim=-1, index=micro_batch["responses"].unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])

                    if eos_token_id is not None:
                        if need_topk or return_full_vocab:
                            eos_log_probs = log_probs_all[..., eos_token_id]
                        else:
                            eos_log_probs = eos_log_probs_from_logits(logits, eos_token_id)
                    if (
                        terminal_objective_mode in {"conservative_kl", "teacher_remap_only"}
                        and terminal_secondary_token_id is not None
                    ):
                        terminal_secondary_log_probs = log_probs_all[
                            ..., int(terminal_secondary_token_id)
                        ]
                    
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             topk_ids = student_top_k_ids
                        else:
                             if candidate_mode in ("topk", "topk_theory"):
                                 _, topk_ids = torch.topk(logits, k=top_k, dim=-1)
                             elif candidate_mode == "sample_stu":
                                 if terminal_aware_enable:
                                     topk_ids = sample_terminal_objective_candidates(
                                         log_probs_all,
                                         terminal_mask=response_terminal_mask,
                                         num_candidates=top_k,
                                         eos_token_id=eos_token_id,
                                         secondary_token_id=terminal_secondary_token_id,
                                         objective_mode=terminal_objective_mode,
                                     )
                                 elif diagnostic_forced_eos_enable:
                                     if eos_token_id is None:
                                         raise ValueError("forced-EOS diagnostic requires an EOS token id.")
                                     forced_eos_sample = sample_forced_eos_candidates(
                                         log_probs_all,
                                         num_candidates=top_k,
                                         eos_token_id=eos_token_id,
                                     )
                                     topk_ids = forced_eos_sample.candidate_ids
                                     forced_eos_estimator_weights = forced_eos_sample.estimator_weights
                                 else:
                                     topk_ids = self._sample_candidate_ids(
                                         log_probs_all.reshape(-1, log_probs_all.size(-1)),
                                         num_samples=top_k,
                                         replacement=sample_replacement,
                                     ).view(*log_probs_all.shape[:-1], top_k)
                             elif candidate_mode == "sample_uniform":
                                 topk_ids = self._sample_uniform_candidate_ids(log_probs_all, num_samples=top_k)
                             elif candidate_mode == "adaptive_head_tail":
                                 if not sample_replacement:
                                     raise ValueError("adaptive_head_tail currently requires sample_k_replacement=True")
                                 (
                                     topk_ids,
                                     candidate_weights,
                                     adaptive_head_counts,
                                     adaptive_head_mass,
                                 ) = self._sample_adaptive_head_tail_candidate_ids(
                                     log_probs_all,
                                     num_samples=top_k,
                                     gamma=adaptive_head_tail_gamma,
                                     k2_min=adaptive_head_tail_k2_min,
                                 )
                        
                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)
                    if terminal_topm > 0:
                        terminal_topm_ids, terminal_topm_log_probs = self._select_topm_excluding_token(
                            log_probs_all,
                            num_tokens=terminal_topm,
                            excluded_token_id=eos_token_id,
                        )
                    if return_full_vocab:
                        topk_log_probs = log_probs_all

            return (
                entropy,
                log_probs,
                topk_ids,
                topk_log_probs,
                candidate_weights,
                adaptive_head_counts,
                adaptive_head_mass,
                eos_log_probs,
                forced_eos_estimator_weights,
                terminal_topm_ids,
                terminal_topm_log_probs,
                terminal_secondary_log_probs,
            )

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_probs_for_ids(self, data: DataProto) -> torch.Tensor:
        self.actor_module.eval()

        target_ids = data.batch["target_ids"]
        
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "target_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        topk_log_probs_lst = []
        top_k = target_ids.shape[-1]

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            mb_target_ids = model_inputs["target_ids"]
            with torch.no_grad():
                _, _, _, topk_log_probs, *_ = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=False, 
                    top_k=top_k, student_top_k_ids=mb_target_ids
                )
            topk_log_probs_lst.append(topk_log_probs)

        topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_distillation_reward(self, data: DataProto) -> DataProto:
        self.actor_module.eval()

        top_k = data.meta_info.get("log_prob_top_k", 0)
        strategy = data.meta_info.get("top_k_strategy", "only_stu")
        kl_estimator = data.meta_info.get("kl_estimator", "k1")
        reward_weight_mode = data.meta_info.get("reward_weight_mode", "student_p")
        reward_weight_normalize = data.meta_info.get("reward_weight_normalize", None)
        candidate_mode = self._canonical_candidate_mode(data.meta_info.get("log_prob_candidate_mode", "topk"))
        diagnostic_forced_eos_enable = self._parse_bool(
            data.meta_info.get("opd_diagnostic_forced_eos_enable", False),
            default=False,
            name="opd_diagnostic_forced_eos_enable",
        )
        sample_k_kl_plus_one = self._parse_bool(
            data.meta_info.get("sample_k_kl_plus_one", True),
            default=True,
            name="sample_k_kl_plus_one",
        )
        opd_loss_type = self._canonical_opd_loss_type(data.meta_info.get("opd_loss_type", "sample_k_reverse_kl"))
        chi_square_baseline = self._canonical_chi_square_baseline(
            data.meta_info.get("chi_square_baseline", "mean")
        )
        on_logprob_mse_clip = self._parse_optional_float(
            data.meta_info.get("on_logprob_mse_clip", None),
            default=None,
            name="on_logprob_mse_clip",
        )
        on_logprob_mse_center = self._parse_bool(
            data.meta_info.get("on_logprob_mse_center", False),
            default=False,
            name="on_logprob_mse_center",
        )
        on_logprob_mse_normalize = self._parse_bool(
            data.meta_info.get("on_logprob_mse_normalize", False),
            default=False,
            name="on_logprob_mse_normalize",
        )
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        if candidate_mode in ("sample_stu", "sample_uniform", "topk_theory", "adaptive_head_tail") and strategy != "only_stu":
            raise ValueError(
                f"log_prob_candidate_mode={candidate_mode} currently supports only top_k_strategy=only_stu"
            )
        if opd_loss_type != "sample_k_reverse_kl":
            if strategy != "only_stu" or candidate_mode != "sample_stu":
                raise ValueError(
                    f"opd_loss_type={opd_loss_type} currently supports only "
                    "log_prob_candidate_mode=sample_stu and top_k_strategy=only_stu"
                )
        if candidate_mode == "adaptive_head_tail" and opd_loss_type != "sample_k_reverse_kl":
            raise ValueError("adaptive_head_tail currently supports only opd_loss_type=sample_k_reverse_kl")

        def parse_optional_bool(value):
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            value = str(value).strip().lower()
            if value in ("", "auto", "none", "null"):
                return None
            if value in ("1", "true", "yes", "y", "on"):
                return True
            if value in ("0", "false", "no", "n", "off"):
                return False
            raise ValueError(f"Unknown reward_weight_normalize value: {value}")

        normalize_override = parse_optional_bool(reward_weight_normalize)
        if candidate_mode == "topk_theory" and reward_weight_mode != "student_p":
            raise ValueError("log_prob_candidate_mode=topk_theory requires reward_weight_mode=student_p")
        if candidate_mode in ("sample_uniform", "topk_theory", "adaptive_head_tail") and normalize_override is True:
            raise ValueError(
                f"log_prob_candidate_mode={candidate_mode} requires reward_weight_normalize=False/auto"
            )
        default_normalize = (
            True if candidate_mode == "sample_stu"
            else False if candidate_mode in ("sample_uniform", "topk_theory", "adaptive_head_tail")
            else strategy != "union-intersection"
        )
        normalize_weights = default_normalize if normalize_override is None else normalize_override
        effective_reward_weight_mode = reward_weight_mode
        if candidate_mode == "sample_stu" and reward_weight_mode == "student_p":
            effective_reward_weight_mode = "none"
        if candidate_mode == "sample_uniform":
            effective_reward_weight_mode = "student_p"

        S_on_T = None
        if strategy in ["only_tch", "intersection", "union", "union-intersection"]:
            target_ids = data.batch["teacher_top_k_ids"]
            
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
            
            mb_data = data.select(batch_keys=select_keys + ["teacher_top_k_ids"], 
                                 non_tensor_batch_keys=non_tensor_select_keys)
            
            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, batch_idx_list = prepare_dynamic_batch(mb_data, max_token_len=max_token_len)
            else:
                micro_batches = mb_data.split(micro_batch_size)

            S_on_T_lst = []
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                mb_target_ids = model_inputs["teacher_top_k_ids"]
                with torch.no_grad():
                    _, _, _, topk_log_probs, *_ = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=False, 
                        top_k=top_k, student_top_k_ids=mb_target_ids
                    )
                S_on_T_lst.append(topk_log_probs)

            S_on_T = torch.concat(S_on_T_lst, dim=0)
            if use_dynamic_bsz:
                S_on_T = restore_dynamic_batch(S_on_T, batch_idx_list)
        
        device = get_device_id()
        S_ids = data.batch["student_top_k_ids"].to(device)
        S_logp = data.batch["student_top_k_log_probs"].to(device)
        T_on_S = data.batch["teacher_on_student_log_probs"].to(device)
        
        T_ids = data.batch.get("teacher_top_k_ids", None)
        if T_ids is not None: T_ids = T_ids.to(device)
        T_logp = data.batch.get("teacher_top_k_log_probs", None)
        if T_logp is not None: T_logp = T_logp.to(device)
        overlap_mask = data.batch.get("overlap_mask", None)
        if overlap_mask is not None: overlap_mask = overlap_mask.to(device)

        def compute_reward_weights(S_logp, T_logp, valid_mask, weight_mode, normalize=True):
            if weight_mode == "student_p":
                log_probs = S_logp
            elif weight_mode == "teacher_p":
                log_probs = T_logp
            elif weight_mode == "none":
                log_probs = torch.zeros_like(S_logp)
            else:
                raise ValueError(f"Unknown reward_weight_mode: {weight_mode}")
            
            log_probs = torch.where(valid_mask, log_probs, torch.full_like(log_probs, -float('inf')))
            
            if normalize:
                norm_log_weights = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
                weights = torch.exp(norm_log_weights)
            else:
                weights = torch.exp(log_probs)
            
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            
            return weights

        res_tensors = {}
        
        if strategy == "only_stu":
            valid_mask = torch.ones_like(S_logp, dtype=torch.bool)
            sample_k_plus_one = candidate_mode in (
                "sample_stu",
                "sample_uniform",
                "topk_theory",
                "adaptive_head_tail",
            ) and sample_k_kl_plus_one
            loss_weight = self._compute_opd_logprob_weight(
                S_logp,
                T_on_S,
                opd_loss_type=opd_loss_type,
                sample_k_kl_plus_one=sample_k_plus_one,
                valid_mask=valid_mask,
                center=on_logprob_mse_center,
                normalize=on_logprob_mse_normalize,
                clip=on_logprob_mse_clip,
                chi_square_baseline=chi_square_baseline,
            )
            if candidate_mode == "adaptive_head_tail":
                if "candidate_estimator_weights" not in data.batch.keys():
                    raise ValueError("adaptive_head_tail requires candidate_estimator_weights in batch")
                norm_weights = data.batch["candidate_estimator_weights"].to(device=device, dtype=loss_weight.dtype)
            else:
                norm_weights = compute_reward_weights(
                    S_logp, T_on_S, valid_mask, effective_reward_weight_mode, normalize=normalize_weights
                )
                if candidate_mode == "sample_uniform":
                    vocab_size = data.meta_info.get("candidate_vocab_size", None) or self._get_actor_vocab_size()
                    if vocab_size is None:
                        raise ValueError("sample_uniform requires candidate_vocab_size or actor config vocab_size")
                    norm_weights = norm_weights * (float(vocab_size) / float(top_k))
                if diagnostic_forced_eos_enable:
                    if FORCED_EOS_ESTIMATOR_WEIGHTS_KEY not in data.batch.keys():
                        raise ValueError(
                            "forced-EOS diagnostic requires its dedicated estimator weights in the batch."
                        )
                    forced_eos_weights = data.batch[FORCED_EOS_ESTIMATOR_WEIGHTS_KEY].to(
                        device=device,
                        dtype=loss_weight.dtype,
                    )
                    if forced_eos_weights.shape != norm_weights.shape:
                        raise ValueError(
                            "forced-EOS estimator weights must match reward weights, "
                            f"got {forced_eos_weights.shape} and {norm_weights.shape}."
                        )
                    norm_weights = norm_weights * forced_eos_weights
            rm_scores = -loss_weight * norm_weights
            actor_candidate_weights = select_terminal_topk_candidate_estimator_weights(
                candidate_weights=norm_weights,
                terminal_aware_enable=self._parse_bool(
                    data.meta_info.get("opd_terminal_aware_enable", False),
                    default=False,
                    name="opd_terminal_aware_enable",
                ),
                objective_mode=data.meta_info.get("opd_terminal_objective_mode", "anchor_kl"),
                candidate_mode=candidate_mode,
                teacher_remap_enable=self._parse_bool(
                    data.meta_info.get("opd_terminal_teacher_remap_enable", False),
                    default=False,
                    name="opd_terminal_teacher_remap_enable",
                ),
            )
            if actor_candidate_weights is not None:
                res_tensors["candidate_estimator_weights"] = actor_candidate_weights
            
        elif strategy == "only_tch":
            kl_val = S_on_T - T_logp
            valid_mask = torch.ones_like(S_on_T, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_on_T, T_logp, valid_mask, reward_weight_mode, normalize=normalize_weights)
            rm_scores = -kl_val * norm_weights
            res_tensors["union_top_k_ids"] = T_ids
            
        elif strategy == "intersection":
            valid_mask = overlap_mask.bool()
            kl_val = S_logp - T_on_S
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode, normalize=normalize_weights)
            rm_scores = -kl_val * norm_weights
            
        elif strategy == "union":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)
            
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                torch.ones_like(S_ids, dtype=torch.bool),
                ~T_in_S
            ], dim=-1)
            
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode, normalize=normalize_weights)
            rm_scores = -kl_val * norm_weights
            
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T
        
        elif strategy == "union-intersection":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)

            S_in_T = overlap_mask.bool().to(device)
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                ~S_in_T,
                ~T_in_S
            ], dim=-1)
                
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode, normalize=normalize_weights)
            rm_scores = -kl_val * norm_weights
            
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T
            
        res_tensors["rm_scores"] = rm_scores
        return DataProto.from_dict(tensors=res_tensors)

    def _clip_grad_norm(self, max_norm: float):
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=max_norm)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=max_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=max_norm)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        return grad_norm

    def _optimizer_step(self, total_grad_norm_target: float | None = None):
        assert self.config.grad_clip is not None

        normalization_metrics = {"opd_samplek/total_grad_norm_enabled": float(total_grad_norm_target is not None)}
        if total_grad_norm_target is None:
            grad_norm = self._clip_grad_norm(self.config.grad_clip)
        else:
            if (
                not math.isfinite(total_grad_norm_target)
                or total_grad_norm_target <= 0.0
                or total_grad_norm_target > self.config.grad_clip
            ):
                raise ValueError(
                    "opd_samplek_total_grad_norm must be finite, positive, and no larger than "
                    f"actor.grad_clip={self.config.grad_clip}, got {total_grad_norm_target}."
                )
            grad_norm = self._clip_grad_norm(math.inf)
            grad_norm_value = grad_norm.detach().item()
            normalization_scale = (
                total_grad_norm_target / (grad_norm_value + 1e-6) if grad_norm_value > 0.0 else 1.0
            )
            if math.isfinite(grad_norm_value):
                for parameter in self.actor_module.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(normalization_scale)
            normalization_metrics.update(
                {
                    "opd_samplek/total_grad_norm_target": total_grad_norm_target,
                    "opd_samplek/total_grad_norm_pre": grad_norm_value,
                    "opd_samplek/total_grad_norm_scale": normalization_scale,
                    "opd_samplek/total_grad_norm_post": grad_norm_value * normalization_scale,
                }
            )

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm, normalization_metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        terminal_aware_enable = self._parse_bool(
            data.meta_info.get("opd_terminal_aware_enable", False),
            default=False,
            name="opd_terminal_aware_enable",
        )
        terminal_secondary_token_id = self._parse_optional_int(
            data.meta_info.get("opd_terminal_secondary_token_id", None),
            name="opd_terminal_secondary_token_id",
        )
        terminal_objective_mode = (
            normalize_terminal_objective_mode(
                data.meta_info.get("opd_terminal_objective_mode", "anchor_kl")
            )
            if terminal_aware_enable
            else "anchor_kl"
        )
        terminal_kl_baseline_mode = str(
            data.meta_info.get("opd_terminal_kl_baseline_mode", "mc_loo")
        ).strip().lower().replace("-", "_")
        terminal_topm = (
            int(data.meta_info.get("opd_terminal_topm", 0))
            if terminal_aware_enable
            and terminal_objective_mode == "conservative_kl"
            and terminal_kl_baseline_mode == "topm_coarse"
            else 0
        )
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if terminal_aware_enable:
            select_keys.append("response_mask")
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        top_k = data.meta_info.get("top_k", 0)
        candidate_mode = data.meta_info.get("log_prob_candidate_mode", "topk")
        sample_replacement = data.meta_info.get("sample_k_replacement", True)
        adaptive_head_tail_gamma = float(data.meta_info.get("adaptive_head_tail_gamma", 0.5))
        adaptive_head_tail_k2_min = int(data.meta_info.get("adaptive_head_tail_k2_min", 1))
        eos_token_id = data.meta_info.get("eos_token_id")
        diagnostic_forced_eos_enable = self._parse_bool(
            data.meta_info.get("opd_diagnostic_forced_eos_enable", False),
            default=False,
            name="opd_diagnostic_forced_eos_enable",
        )
        print(f"In compute_log_prob, top_k: {top_k}, candidate_mode: {candidate_mode}")
        canonical_candidate_mode = self._canonical_candidate_mode(candidate_mode)
        forward_candidate_mode = "topk" if canonical_candidate_mode == "full_vocab" else canonical_candidate_mode
        log_probs_lst = []
        entropy_lst = []
        topk_ids_lst = []
        topk_log_probs_lst = []
        candidate_weights_lst = []
        adaptive_head_counts_lst = []
        adaptive_head_mass_lst = []
        eos_log_probs_lst = []
        forced_eos_estimator_weights_lst = []
        terminal_topm_ids_lst = []
        terminal_topm_log_probs_lst = []
        terminal_secondary_log_probs_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                (
                    entropy,
                    log_probs,
                    topk_ids,
                    topk_log_probs,
                    candidate_weights,
                    adaptive_head_counts,
                    adaptive_head_mass,
                    eos_log_probs,
                    forced_eos_estimator_weights,
                    terminal_topm_ids,
                    terminal_topm_log_probs,
                    terminal_secondary_log_probs,
                ) = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    top_k=top_k,
                    candidate_mode=forward_candidate_mode,
                    sample_replacement=sample_replacement,
                    adaptive_head_tail_gamma=adaptive_head_tail_gamma,
                    adaptive_head_tail_k2_min=adaptive_head_tail_k2_min,
                    eos_token_id=eos_token_id,
                    diagnostic_forced_eos_enable=diagnostic_forced_eos_enable,
                    terminal_aware_enable=terminal_aware_enable,
                    terminal_secondary_token_id=terminal_secondary_token_id,
                    terminal_objective_mode=terminal_objective_mode,
                    terminal_topm=terminal_topm,
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if eos_log_probs is not None:
                eos_log_probs_lst.append(eos_log_probs)
            if top_k > 0:
                topk_ids_lst.append(topk_ids)
                topk_log_probs_lst.append(topk_log_probs)
                if candidate_weights is not None:
                    candidate_weights_lst.append(candidate_weights)
                    adaptive_head_counts_lst.append(adaptive_head_counts)
                    adaptive_head_mass_lst.append(adaptive_head_mass)
                if forced_eos_estimator_weights is not None:
                    forced_eos_estimator_weights_lst.append(forced_eos_estimator_weights)
            if terminal_topm_ids is not None:
                terminal_topm_ids_lst.append(terminal_topm_ids)
                terminal_topm_log_probs_lst.append(terminal_topm_log_probs)
            if terminal_secondary_log_probs is not None:
                terminal_secondary_log_probs_lst.append(terminal_secondary_log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        
        topk_ids_tensor = None
        topk_log_probs_tensor = None
        candidate_weights_tensor = None
        adaptive_head_counts_tensor = None
        adaptive_head_mass_tensor = None
        eos_log_probs_tensor = None
        forced_eos_estimator_weights_tensor = None
        terminal_topm_ids_tensor = None
        terminal_topm_log_probs_tensor = None
        terminal_secondary_log_probs_tensor = None
        if top_k > 0:
            topk_ids_tensor = torch.concat(topk_ids_lst, dim=0)
            topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)
            if len(candidate_weights_lst) > 0:
                candidate_weights_tensor = torch.concat(candidate_weights_lst, dim=0)
                adaptive_head_counts_tensor = torch.concat(adaptive_head_counts_lst, dim=0)
                adaptive_head_mass_tensor = torch.concat(adaptive_head_mass_lst, dim=0)
            if forced_eos_estimator_weights_lst:
                forced_eos_estimator_weights_tensor = torch.concat(forced_eos_estimator_weights_lst, dim=0)
        if eos_log_probs_lst:
            eos_log_probs_tensor = torch.concat(eos_log_probs_lst, dim=0)
        if terminal_topm_ids_lst:
            terminal_topm_ids_tensor = torch.concat(terminal_topm_ids_lst, dim=0)
            terminal_topm_log_probs_tensor = torch.concat(terminal_topm_log_probs_lst, dim=0)
        if terminal_secondary_log_probs_lst:
            terminal_secondary_log_probs_tensor = torch.concat(
                terminal_secondary_log_probs_lst, dim=0
            )

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if eos_log_probs_tensor is not None:
                eos_log_probs_tensor = restore_dynamic_batch(eos_log_probs_tensor, batch_idx_list)
            if terminal_topm_ids_tensor is not None:
                terminal_topm_ids_tensor = restore_dynamic_batch(
                    terminal_topm_ids_tensor, batch_idx_list
                )
                terminal_topm_log_probs_tensor = restore_dynamic_batch(
                    terminal_topm_log_probs_tensor, batch_idx_list
                )
            if terminal_secondary_log_probs_tensor is not None:
                terminal_secondary_log_probs_tensor = restore_dynamic_batch(
                    terminal_secondary_log_probs_tensor, batch_idx_list
                )
            if top_k > 0:
                topk_ids_tensor = restore_dynamic_batch(topk_ids_tensor, batch_idx_list)
                topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)
                if candidate_weights_tensor is not None:
                    candidate_weights_tensor = restore_dynamic_batch(candidate_weights_tensor, batch_idx_list)
                    adaptive_head_counts_tensor = restore_dynamic_batch(adaptive_head_counts_tensor, batch_idx_list)
                    adaptive_head_mass_tensor = restore_dynamic_batch(adaptive_head_mass_tensor, batch_idx_list)
                if forced_eos_estimator_weights_tensor is not None:
                    forced_eos_estimator_weights_tensor = restore_dynamic_batch(
                        forced_eos_estimator_weights_tensor, batch_idx_list
                    )

        return (
            log_probs,
            entropys,
            topk_ids_tensor,
            topk_log_probs_tensor,
            candidate_weights_tensor,
            adaptive_head_counts_tensor,
            adaptive_head_mass_tensor,
            eos_log_probs_tensor,
            forced_eos_estimator_weights_tensor,
            terminal_topm_ids_tensor,
            terminal_topm_log_probs_tensor,
            terminal_secondary_log_probs_tensor,
        )

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        full_vocab_reward_weight_mode = data.meta_info.get("reward_weight_mode", "student_p")
        opd_advantage_mode = normalize_opd_mode(data.meta_info.get("opd_advantage_mode", "fixed"))
        use_decomposed_pi_old = is_decomposed_pi_old_mode(opd_advantage_mode)
        opd_decomposed_prefix_is_mode = data.meta_info.get(
            "opd_decomposed_prefix_is_mode", "cumulative_cap"
        )
        opd_decomposed_prefix_is_min_weight = float(
            data.meta_info.get("opd_decomposed_prefix_is_min_weight", 0.25)
        )
        opd_decomposed_prefix_is_max_weight = float(
            data.meta_info.get("opd_decomposed_prefix_is_max_weight", 4.0)
        )
        opd_decomposed_proximal_coef = float(data.meta_info.get("opd_decomposed_proximal_coef", 1.0))
        opd_q_mixture_enable = self._parse_bool(
            data.meta_info.get("opd_q_mixture_enable", False),
            default=False,
            name="opd_q_mixture_enable",
        )
        opd_q_mixture_teacher_advantage_mode = normalize_q_mixture_teacher_advantage_mode(
            data.meta_info.get("opd_q_mixture_teacher_advantage_mode", "proposal")
        )
        opd_q_prefix_samplek = (
            opd_q_mixture_enable and opd_q_mixture_teacher_advantage_mode != "proposal"
        )
        opd_q_mixture_source_normalize_enable = self._parse_bool(
            data.meta_info.get("opd_q_mixture_source_normalize_enable", False),
            default=False,
            name="opd_q_mixture_source_normalize_enable",
        )
        if opd_q_mixture_source_normalize_enable and not opd_q_prefix_samplek:
            raise ValueError(
                "opd_q_mixture_source_normalize_enable=True requires Q-prefix sample-k."
            )
        opd_samplek_candidate_aggregation = str(
            data.meta_info.get("opd_samplek_candidate_aggregation", "sum") or "sum"
        ).strip().lower().replace("-", "_")
        opd_samplek_candidate_aggregation = {
            "avg": "mean",
            "average": "mean",
        }.get(opd_samplek_candidate_aggregation, opd_samplek_candidate_aggregation)
        if opd_samplek_candidate_aggregation not in {"sum", "mean"}:
            raise ValueError(
                "opd_samplek_candidate_aggregation must be sum or mean, got "
                f"{opd_samplek_candidate_aggregation!r}."
            )
        opd_samplek_advantage_centering = normalize_samplek_advantage_centering(
            data.meta_info.get("opd_samplek_advantage_centering", "none")
        )
        opd_samplek_loo_variance_filter_threshold = self._parse_optional_float(
            data.meta_info.get("opd_samplek_loo_variance_filter_threshold", None),
            default=None,
            name="opd_samplek_loo_variance_filter_threshold",
        )
        if opd_samplek_loo_variance_filter_threshold is not None and (
            not math.isfinite(opd_samplek_loo_variance_filter_threshold)
            or opd_samplek_loo_variance_filter_threshold <= 0.0
        ):
            raise ValueError(
                "opd_samplek_loo_variance_filter_threshold must be null or a finite positive value."
            )
        opd_samplek_loo_variance_filter_threshold_mode = (
            normalize_samplek_loo_variance_threshold_mode(
                data.meta_info.get("opd_samplek_loo_variance_filter_threshold_mode", "fixed")
            )
        )
        if (
            opd_samplek_loo_variance_filter_threshold_mode == "update0_quantile"
            and opd_samplek_loo_variance_filter_threshold is None
        ):
            raise ValueError(
                "update0_quantile threshold mode requires an adaptive trainer-resolved threshold."
            )
        opd_samplek_loo_variance_filter_selection = str(
            data.meta_info.get("opd_samplek_loo_variance_filter_selection", "high") or "high"
        ).strip().lower().replace("-", "_")
        opd_samplek_loo_variance_filter_mode = str(
            data.meta_info.get("opd_samplek_loo_variance_filter_mode", "hard") or "hard"
        ).strip().lower().replace("-", "_")
        opd_samplek_loo_variance_filter_soft_base_weight = float(
            data.meta_info.get("opd_samplek_loo_variance_filter_soft_base_weight", 0.5)
        )
        opd_samplek_loo_variance_filter_soft_active_bonus = float(
            data.meta_info.get("opd_samplek_loo_variance_filter_soft_active_bonus", 1.0)
        )
        opd_samplek_loo_variance_filter_expectile_tau = float(
            data.meta_info.get("opd_samplek_loo_variance_filter_expectile_tau", 0.75)
        )
        opd_samplek_influence_clip = self._parse_optional_float(
            data.meta_info.get("opd_samplek_influence_clip", None),
            default=None,
            name="opd_samplek_influence_clip",
        )
        if opd_samplek_influence_clip is not None and (
            not math.isfinite(opd_samplek_influence_clip) or opd_samplek_influence_clip <= 0.0
        ):
            raise ValueError("opd_samplek_influence_clip must be null or a finite positive value.")
        teacher_deficit_residual_enable = self._parse_bool(
            data.meta_info.get("opd_teacher_deficit_residual_enable", False),
            default=False,
            name="opd_teacher_deficit_residual_enable",
        )
        teacher_deficit_residual_k = int(
            data.meta_info.get("opd_teacher_deficit_residual_k", 8)
        )
        teacher_deficit_residual_coef = float(
            data.meta_info.get("opd_teacher_deficit_residual_coef", 0.0)
        )
        diagnostic_forced_eos_enable = self._parse_bool(
            data.meta_info.get("opd_diagnostic_forced_eos_enable", False),
            default=False,
            name="opd_diagnostic_forced_eos_enable",
        )
        terminal_aware_enable = self._parse_bool(
            data.meta_info.get("opd_terminal_aware_enable", False),
            default=False,
            name="opd_terminal_aware_enable",
        )
        terminal_objective_mode = (
            normalize_terminal_objective_mode(
                data.meta_info.get("opd_terminal_objective_mode", "anchor_kl")
            )
            if terminal_aware_enable else "anchor_kl"
        )
        terminal_anchor_mode = data.meta_info.get("opd_terminal_anchor_mode", "behavior")
        terminal_gate_coef = float(data.meta_info.get("opd_terminal_gate_coef", 1.0))
        terminal_secondary_token_id = self._parse_optional_int(
            data.meta_info.get("opd_terminal_secondary_token_id", None),
            name="opd_terminal_secondary_token_id",
        )
        terminal_teacher_remap_enable = self._parse_bool(
            data.meta_info.get("opd_terminal_teacher_remap_enable", False),
            default=False,
            name="opd_terminal_teacher_remap_enable",
        )
        terminal_kl_baseline_mode = str(
            data.meta_info.get("opd_terminal_kl_baseline_mode", "mc_loo") or "mc_loo"
        ).strip().lower().replace("-", "_")
        terminal_teacher_remap_floor = float(
            data.meta_info.get("opd_terminal_teacher_remap_floor", 1e-18)
        )
        terminal_topm = int(data.meta_info.get("opd_terminal_topm", 0))
        opd_samplek_eos_negative_relu_enable = self._parse_bool(
            data.meta_info.get("opd_samplek_eos_negative_relu_enable", False),
            default=False,
            name="opd_samplek_eos_negative_relu_enable",
        )
        if opd_samplek_eos_negative_relu_enable and opd_advantage_mode not in {
            "current_kl_is",
            "current_kl",
        }:
            raise ValueError(
                "opd_samplek_eos_negative_relu_enable requires "
                "opd_advantage_mode=current_kl_is."
            )
        eos_future_enable = self._parse_bool(
            data.meta_info.get("opd_eos_future_enable", False),
            default=False,
            name="opd_eos_future_enable",
        )
        eos_future_mode = (
            normalize_eos_future_mode(
                data.meta_info.get("opd_eos_future_mode", EOS_FUTURE_MODE_FIXED_SIGNED)
            )
            if eos_future_enable
            else EOS_FUTURE_MODE_FIXED_SIGNED
        )
        eos_future_horizon = int(data.meta_info.get("opd_eos_future_horizon", 32))
        eos_future_coef = float(data.meta_info.get("opd_eos_future_coef", 1.0))
        if eos_future_enable:
            if eos_future_horizon < 1:
                raise ValueError("opd_eos_future_horizon must be positive.")
            if not math.isfinite(eos_future_coef) or eos_future_coef < 0.0:
                raise ValueError("opd_eos_future_coef must be finite and nonnegative.")
            if PREFIX_DRIFT_WEIGHTS_KEY not in data.batch:
                raise ValueError("EOS future correction requires prefix drift weights.")
            if eos_future_mode == EOS_FUTURE_MODE_FIXED_SIGNED:
                for required_key in (
                    OPD_EOS_FUTURE_VALUE_KEY,
                    OPD_EOS_FUTURE_CORRECTION_MASK_KEY,
                ):
                    if required_key not in data.batch:
                        raise ValueError(f"EOS future correction is missing {required_key}.")
            else:
                if eos_future_horizon != 1:
                    raise ValueError(
                        f"{eos_future_mode} requires opd_eos_future_horizon=1."
                    )
                if OPD_ROLLOUT_REFERENCE_LOG_PROBS_KEY not in data.batch:
                    raise ValueError(
                        "dynamic H=1 EOS correction requires fixed rollout-reference log-probabilities."
                    )
        opd_raw_advantage_clip = self._parse_optional_float(
            data.meta_info.get("opd_raw_advantage_clip", None),
            default=None,
            name="opd_raw_advantage_clip",
        )
        if opd_raw_advantage_clip is not None and (
            not math.isfinite(opd_raw_advantage_clip) or opd_raw_advantage_clip <= 0.0
        ):
            raise ValueError("opd_raw_advantage_clip must be null or a finite positive value.")
        if opd_raw_advantage_clip is not None and opd_q_prefix_samplek:
            raise ValueError(
                "raw advantage clipping does not support Q-prefix transformed teacher advantages."
            )
        if opd_samplek_advantage_centering != "none" and opd_advantage_mode not in {
            "current_kl_is",
            "current_kl",
        }:
            raise ValueError(
                "opd_samplek_advantage_centering requires opd_advantage_mode=current_kl_is."
            )
        opd_samplek_entropy_coef = float(data.meta_info.get("opd_samplek_entropy_coef", 0.0))
        if not math.isfinite(opd_samplek_entropy_coef) or opd_samplek_entropy_coef < 0.0:
            raise ValueError(
                "opd_samplek_entropy_coef must be finite and nonnegative, got "
                f"{opd_samplek_entropy_coef}."
            )
        if opd_samplek_entropy_coef > 0.0 and opd_advantage_mode not in {
            "current_kl_is",
            "current_kl",
        }:
            raise ValueError(
                "opd_samplek_entropy_coef requires opd_advantage_mode=current_kl_is."
            )
        opd_sampled_token_proximal_coef = float(
            data.meta_info.get("opd_sampled_token_proximal_coef", 0.0)
        )
        opd_sampled_token_proximal_mode = str(
            data.meta_info.get("opd_sampled_token_proximal_mode", "reverse_kl")
        )
        if not math.isfinite(opd_sampled_token_proximal_coef) or opd_sampled_token_proximal_coef < 0.0:
            raise ValueError(
                "opd_sampled_token_proximal_coef must be finite and nonnegative, got "
                f"{opd_sampled_token_proximal_coef}."
            )
        opd_sampled_token_max_entropy_coef = float(
            data.meta_info.get("opd_sampled_token_max_entropy_coef", 0.0)
        )
        if (
            not math.isfinite(opd_sampled_token_max_entropy_coef)
            or opd_sampled_token_max_entropy_coef < 0.0
        ):
            raise ValueError(
                "opd_sampled_token_max_entropy_coef must be finite and nonnegative, got "
                f"{opd_sampled_token_max_entropy_coef}."
            )
        if opd_samplek_entropy_coef > 0.0 and opd_sampled_token_max_entropy_coef > 0.0:
            raise ValueError(
                "sample-k entropy and sampled-token max entropy are mutually exclusive; "
                "enable only one estimator."
            )
        opd_sampled_token_proximal_prefix_weight_enable = self._parse_bool(
            data.meta_info.get("opd_sampled_token_proximal_prefix_weight_enable", False),
            default=False,
            name="opd_sampled_token_proximal_prefix_weight_enable",
        )
        opd_samplek_total_grad_norm = self._parse_optional_float(
            data.meta_info.get("opd_samplek_total_grad_norm", None),
            default=None,
            name="opd_samplek_total_grad_norm",
        )
        if opd_samplek_total_grad_norm is not None:
            candidate_mode = self._canonical_candidate_mode(
                data.meta_info.get("log_prob_candidate_mode", "topk")
            )
            opd_loss_type = self._canonical_opd_loss_type(
                data.meta_info.get("opd_loss_type", "sample_k_reverse_kl")
            )
            if (
                not math.isfinite(opd_samplek_total_grad_norm)
                or opd_samplek_total_grad_norm <= 0.0
            ):
                raise ValueError("opd_samplek_total_grad_norm must be finite and positive when enabled.")
            if (
                candidate_mode != "sample_stu"
                or opd_loss_type != "sample_k_reverse_kl"
                or opd_advantage_mode not in {"current_kl_is", "current_kl"}
                or (
                    opd_sampled_token_proximal_coef <= 0.0
                    and opd_sampled_token_max_entropy_coef <= 0.0
                    and opd_samplek_entropy_coef <= 0.0
                )
            ):
                raise ValueError(
                    "opd_samplek_total_grad_norm is restricted to sample_stu reverse-KL with "
                    "current_kl(_is) advantages and a sampled-token RKL or entropy regularizer."
                )
            if self.config.entropy_coeff != 0 or self.config.use_kl_loss:
                raise ValueError(
                    "opd_samplek_total_grad_norm requires entropy_coeff=0 and use_kl_loss=False "
                    "so it normalizes only the sample-k teacher plus proximal RKL objective."
                )
        opd_dapo_format_penalty_enable = self._parse_bool(
            data.meta_info.get("opd_dapo_format_penalty_enable", False),
            default=False,
            name="opd_dapo_format_penalty_enable",
        )
        opd_dapo_format_penalty_coef = self._parse_optional_float(
            data.meta_info.get("opd_dapo_format_penalty_coef", None),
            default=0.0,
            name="opd_dapo_format_penalty_coef",
        )
        if opd_dapo_format_penalty_coef < 0.0 or not math.isfinite(opd_dapo_format_penalty_coef):
            raise ValueError("opd_dapo_format_penalty_coef must be finite and nonnegative.")
        opd_dapo_format_penalty_enable = (
            opd_dapo_format_penalty_enable or opd_dapo_format_penalty_coef > 0.0
        )
        opd_dapo_format_penalty_tail_tokens = self._parse_positive_int(
            data.meta_info.get("opd_dapo_format_penalty_tail_tokens", 512),
            default=512,
            name="opd_dapo_format_penalty_tail_tokens",
        )
        sample_k_kl_plus_one = self._parse_bool(
            data.meta_info.get("sample_k_kl_plus_one", True),
            default=True,
            name="sample_k_kl_plus_one",
        )
        adaptive_head_tail_negative_elu_enable = self._parse_bool(
            data.meta_info.get("adaptive_head_tail_negative_elu_enable", False),
            default=False,
            name="adaptive_head_tail_negative_elu_enable",
        )
        adaptive_head_tail_negative_elu_threshold = float(
            data.meta_info.get("adaptive_head_tail_negative_elu_threshold", -1.0)
        )
        adaptive_head_tail_negative_elu_tau = float(
            data.meta_info.get("adaptive_head_tail_negative_elu_tau", 1.0)
        )
        opd_current_samplek_no_candidate_is = self._parse_bool(
            data.meta_info.get(
                "opd_current_samplek_no_candidate_is",
                self.config.get("opd_current_samplek_no_candidate_is", False),
            ),
            default=False,
            name="opd_current_samplek_no_candidate_is",
        )
        opd_current_samplek_force_candidate_is = self._parse_bool(
            data.meta_info.get("opd_current_samplek_force_candidate_is", False),
            default=False,
            name="opd_current_samplek_force_candidate_is",
        )
        samplek_candidate_reuse_enabled = self._parse_bool(
            data.meta_info.get("samplek_candidate_reuse_is_enable", False),
            default=False,
            name="samplek_candidate_reuse_is_enable",
        )
        prefix_drift_enable = self._parse_bool(
            data.meta_info.get("prefix_drift_enable", False),
            default=False,
            name="prefix_drift_enable",
        )
        prefix_drift_method = str(data.meta_info.get("prefix_drift_method", "diagnostic"))
        prefix_drift_log_clip = self._parse_optional_float(
            data.meta_info.get("prefix_drift_log_clip", 3.0),
            default=3.0,
            name="prefix_drift_log_clip",
        )
        prefix_drift_log_clip_mode = str(
            data.meta_info.get("prefix_drift_log_clip_mode", "symmetric")
        )
        prefix_drift_normalize = str(data.meta_info.get("prefix_drift_normalize", "none"))
        prefix_drift_position_beta = self._parse_optional_float(
            data.meta_info.get("prefix_drift_position_beta", 0.0),
            default=0.0,
            name="prefix_drift_position_beta",
        )
        prefix_drift_position_hmax = self._parse_optional_int(
            data.meta_info.get("prefix_drift_position_hmax", None),
            default=None,
            name="prefix_drift_position_hmax",
        )
        prefix_drift_detach = self._parse_bool(
            data.meta_info.get("prefix_drift_detach", True),
            default=True,
            name="prefix_drift_detach",
        )
        opd_q_mixture_diagnostics_enable = self._parse_bool(
            data.meta_info.get("opd_q_mixture_diagnostics_enable", False),
            default=False,
            name="opd_q_mixture_diagnostics_enable",
        )
        diagnostic_candidate_mode = self._canonical_candidate_mode(
            data.meta_info.get("log_prob_candidate_mode", "topk")
        )
        validate_teacher_deficit_residual_configuration(
            enabled=teacher_deficit_residual_enable,
            teacher_sample_count=teacher_deficit_residual_k,
            coefficient=teacher_deficit_residual_coef,
            candidate_mode=diagnostic_candidate_mode,
            top_k=int(data.meta_info.get("log_prob_top_k", 0)),
            top_k_strategy=data.meta_info.get("top_k_strategy", "only_stu"),
            advantage_mode=opd_advantage_mode,
            sample_replacement=self._parse_bool(
                data.meta_info.get("sample_k_replacement", True),
                default=True,
                name="sample_k_replacement",
            ),
            opd_loss_type=data.meta_info.get("opd_loss_type", "sample_k_reverse_kl"),
        )
        if teacher_deficit_residual_enable:
            for required_key in (
                TEACHER_DEFICIT_RESIDUAL_IDS_KEY,
                TEACHER_DEFICIT_RESIDUAL_LOG_PROBS_KEY,
            ):
                if required_key not in data.batch:
                    raise ValueError(f"teacher deficit residual is missing {required_key}.")
        validate_forced_eos_diagnostic_configuration(
            enabled=diagnostic_forced_eos_enable,
            candidate_mode=diagnostic_candidate_mode,
            sample_replacement=self._parse_bool(
                data.meta_info.get("sample_k_replacement", True),
                default=True,
                name="sample_k_replacement",
            ),
            top_k=int(data.meta_info.get("log_prob_top_k", 0)),
            top_k_strategy=data.meta_info.get("top_k_strategy", "only_stu"),
            opd_loss_type=data.meta_info.get("opd_loss_type", "sample_k_reverse_kl"),
            advantage_mode=opd_advantage_mode,
            candidate_aggregation=opd_samplek_candidate_aggregation,
            advantage_centering=opd_samplek_advantage_centering,
            adaptive_update_enable=self._parse_bool(
                data.meta_info.get("adaptive_ppo_update_enable", False),
                default=False,
                name="adaptive_ppo_update_enable",
            ),
            no_candidate_is=opd_current_samplek_no_candidate_is,
            q_source_normalize_enable=opd_q_mixture_source_normalize_enable,
            generic_candidate_weights_present="candidate_estimator_weights" in data.batch.keys(),
        )
        validate_terminal_aware_configuration(
            enabled=terminal_aware_enable,
            objective_mode=terminal_objective_mode,
            anchor_mode=terminal_anchor_mode,
            gate_coef=terminal_gate_coef,
            candidate_mode=diagnostic_candidate_mode,
            sample_replacement=self._parse_bool(
                data.meta_info.get("sample_k_replacement", True),
                default=True,
                name="sample_k_replacement",
            ),
            top_k=int(data.meta_info.get("log_prob_top_k", 0)),
            top_k_strategy=data.meta_info.get("top_k_strategy", "only_stu"),
            opd_loss_type=data.meta_info.get("opd_loss_type", "sample_k_reverse_kl"),
            advantage_mode=opd_advantage_mode,
            candidate_aggregation=opd_samplek_candidate_aggregation,
            advantage_centering=opd_samplek_advantage_centering,
            adaptive_update_enable=self._parse_bool(
                data.meta_info.get("adaptive_ppo_update_enable", False),
                default=False,
                name="adaptive_ppo_update_enable",
            ),
            no_candidate_is=opd_current_samplek_no_candidate_is,
            forced_eos_diagnostic_enable=diagnostic_forced_eos_enable,
            q_mixture_enable=opd_q_mixture_enable,
            q_source_normalize_enable=opd_q_mixture_source_normalize_enable,
            samplek_entropy_coef=opd_samplek_entropy_coef,
            influence_clip=opd_samplek_influence_clip,
            raw_advantage_clip=opd_raw_advantage_clip,
            generic_candidate_weights_present="candidate_estimator_weights" in data.batch.keys(),
            secondary_token_id=terminal_secondary_token_id,
            teacher_remap_enable=terminal_teacher_remap_enable,
            kl_baseline_mode=terminal_kl_baseline_mode,
            teacher_remap_floor=terminal_teacher_remap_floor,
            terminal_topm=terminal_topm,
            subtract_score_baseline=sample_k_kl_plus_one,
            candidate_reuse_enabled=samplek_candidate_reuse_enabled,
        )

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        if "format_mask" in data.batch.keys():
            select_keys.append("format_mask")
        if DAPO_ANSWER_FORMAT_MASK_KEY in data.batch.keys():
            select_keys.append(DAPO_ANSWER_FORMAT_MASK_KEY)

        if "teacher_full_log_probs" in data.batch.keys():
            select_keys.append("teacher_full_log_probs")

        if "teacher_on_student_log_probs" in data.batch.keys():
            select_keys.append("teacher_on_student_log_probs")
        
        if "student_top_k_log_probs" in data.batch.keys():
            select_keys.append("student_top_k_log_probs")

        if "student_top_k_ids" in data.batch.keys():
            select_keys.append("student_top_k_ids")
        if teacher_deficit_residual_enable:
            select_keys.extend(
                [
                    TEACHER_DEFICIT_RESIDUAL_IDS_KEY,
                    TEACHER_DEFICIT_RESIDUAL_LOG_PROBS_KEY,
                ]
            )

        if "candidate_estimator_weights" in data.batch.keys():
            select_keys.append("candidate_estimator_weights")
        if FORCED_EOS_ESTIMATOR_WEIGHTS_KEY in data.batch.keys():
            select_keys.append(FORCED_EOS_ESTIMATOR_WEIGHTS_KEY)
        if OPD_TERMINAL_BEHAVIOR_EOS_LOG_PROBS_KEY in data.batch.keys():
            select_keys.append(OPD_TERMINAL_BEHAVIOR_EOS_LOG_PROBS_KEY)
        if OPD_TERMINAL_BEHAVIOR_SECONDARY_LOG_PROBS_KEY in data.batch.keys():
            select_keys.append(OPD_TERMINAL_BEHAVIOR_SECONDARY_LOG_PROBS_KEY)
        for key in (
            OPD_EOS_FUTURE_VALUE_KEY,
            OPD_EOS_FUTURE_CORRECTION_MASK_KEY,
            OPD_TERMINAL_STUDENT_TOPM_IDS_KEY,
            OPD_TERMINAL_STUDENT_TOPM_LOG_PROBS_KEY,
            OPD_TERMINAL_STUDENT_SECONDARY_LOG_PROBS_KEY,
            OPD_TERMINAL_TEACHER_TOPM_LOG_PROBS_KEY,
            OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY,
        ):
            if key in data.batch.keys():
                select_keys.append(key)

        if PREFIX_DRIFT_WEIGHTS_KEY in data.batch.keys():
            select_keys.append(PREFIX_DRIFT_WEIGHTS_KEY)
        if PREFIX_DRIFT_RAW_WEIGHTS_KEY in data.batch.keys():
            select_keys.append(PREFIX_DRIFT_RAW_WEIGHTS_KEY)

        if OPD_ROLLOUT_REFERENCE_LOG_PROBS_KEY in data.batch.keys():
            select_keys.append(OPD_ROLLOUT_REFERENCE_LOG_PROBS_KEY)

        for optional_key in [
            "opd_proposal_log_probs",
            "opd_proximal_log_probs",
            "opd_teacher_mask",
            "opd_proximal_mask",
            OPD_Q_MIXTURE_PRIOR_ALPHA_KEY,
            OPD_Q_MIXTURE_SOURCE_WEIGHTS_KEY,
        ]:
            if optional_key in data.batch.keys():
                select_keys.append(optional_key)

        if "union_top_k_ids" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_ids")
            select_keys.append("union_top_k_ids")
            if "student_top_k_ids" in select_keys:
                select_keys.remove("student_top_k_ids")

        if "union_top_k_log_probs" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_log_probs")
            select_keys.append("union_top_k_log_probs")
            if "student_top_k_log_probs" in select_keys:
                select_keys.remove("student_top_k_log_probs")   

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for ppo_epoch_idx in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    terminal_aware_output = None
                    opd_terminal_aware_gate_loss = None
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    current_eos_log_probs = None
                    eos_future_teacher_log_probs = None
                    teacher_residual_current_log_probs = None

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    format_mask = None
                    if "format_mask" in model_inputs.keys():
                        format_mask = model_inputs["format_mask"]
                    dapo_answer_format_mask = None
                    if DAPO_ANSWER_FORMAT_MASK_KEY in model_inputs.keys():
                        dapo_answer_format_mask = model_inputs[DAPO_ANSWER_FORMAT_MASK_KEY]

                    if "teacher_full_log_probs" in model_inputs:
                        entropy, sampled_log_prob, _, student_full_log_probs, *_ = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            top_k=0,
                            candidate_mode="full_vocab",
                        )
                        teacher_full_log_probs = model_inputs["teacher_full_log_probs"].to(
                            device=student_full_log_probs.device,
                            dtype=student_full_log_probs.dtype,
                        )

                        with torch.no_grad():
                            student_full_log_probs_det = student_full_log_probs.detach()
                            if full_vocab_reward_weight_mode == "student_p":
                                weights = torch.exp(student_full_log_probs_det)
                            elif full_vocab_reward_weight_mode == "teacher_p":
                                weights = torch.exp(teacher_full_log_probs)
                            elif full_vocab_reward_weight_mode == "none":
                                vocab_size = student_full_log_probs_det.size(-1)
                                weights = torch.full_like(student_full_log_probs_det, 1.0 / vocab_size)
                            else:
                                raise ValueError(f"Unknown reward_weight_mode: {full_vocab_reward_weight_mode}")

                            kl_val = student_full_log_probs_det - teacher_full_log_probs
                            advantages_full = -kl_val * weights
                            full_vocab_kl = (kl_val * weights).sum(dim=-1)

                        pg_losses = -(advantages_full * student_full_log_probs).sum(dim=-1)
                        rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                        if rollout_is_weights is not None:
                            pg_losses = pg_losses * rollout_is_weights
                        if format_mask is not None:
                            pg_loss = agg_loss(
                                loss_mat=pg_losses,
                                loss_mask=response_mask * format_mask.unsqueeze(-1),
                                loss_agg_mode=loss_agg_mode,
                            )
                            metric_mask = response_mask * format_mask.unsqueeze(-1)
                        else:
                            pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            metric_mask = response_mask

                        micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["distillation/full_vocab_kl"] = verl_F.masked_mean(
                            full_vocab_kl.detach(), metric_mask
                        ).item()

                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=metric_mask, loss_agg_mode=loss_agg_mode)
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                        else:
                            policy_loss = pg_loss

                        if self.config.use_kl_loss:
                            ref_log_prob = model_inputs["ref_log_prob"]
                            kld = kl_penalty(
                                logprob=sampled_log_prob,
                                ref_logprob=ref_log_prob,
                                kl_penalty=self.config.kl_loss_type,
                            )
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=metric_mask, loss_agg_mode=loss_agg_mode)
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                            micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                        loss = policy_loss * loss_scale_factor
                        loss.backward()
                        append_to_dict(metrics, micro_batch_metrics)
                        continue
                    
                    if advantages.dim() == 3:
                        student_candidate_count = advantages.shape[-1]
                        student_top_k_ids = None
                        if "union_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["union_top_k_ids"]
                        elif "student_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["student_top_k_ids"]

                        forward_candidate_ids = student_top_k_ids
                        if teacher_deficit_residual_enable:
                            teacher_residual_ids = model_inputs[
                                TEACHER_DEFICIT_RESIDUAL_IDS_KEY
                            ]
                            if teacher_residual_ids.shape[:-1] != student_top_k_ids.shape[:-1]:
                                raise ValueError(
                                    "teacher residual candidate token shape must match student Sample-K ids, "
                                    f"got teacher={teacher_residual_ids.shape}, "
                                    f"student={student_top_k_ids.shape}."
                                )
                            if teacher_residual_ids.size(-1) != teacher_deficit_residual_k:
                                raise ValueError(
                                    "teacher residual candidate count does not match configuration, "
                                    f"got ids={teacher_residual_ids.size(-1)}, "
                                    f"configured={teacher_deficit_residual_k}."
                                )
                            forward_candidate_ids = torch.cat(
                                (student_top_k_ids, teacher_residual_ids),
                                dim=-1,
                            )
                        forward_candidate_count = forward_candidate_ids.shape[-1]

                        (
                            entropy,
                            sampled_log_prob,
                            _,
                            topk_log_probs,
                            _,
                            _,
                            _,
                            current_eos_log_probs,
                            *_,
                        ) = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            top_k=forward_candidate_count,
                            student_top_k_ids=forward_candidate_ids,
                            eos_token_id=(
                                int(data.meta_info["eos_token_id"])
                                if eos_future_enable
                                else None
                            ),
                        )
                        sampled_log_prob_for_loss = sampled_log_prob
                        log_prob = sampled_log_prob
                        if teacher_deficit_residual_enable:
                            teacher_residual_current_log_probs = topk_log_probs[
                                ..., student_candidate_count:
                            ]
                        log_prob_for_loss = topk_log_probs[..., :student_candidate_count]
                        
                    else:
                        _, log_prob, *_ = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )
                        sampled_log_prob_for_loss = log_prob
                        log_prob_for_loss = log_prob

                    if samplek_candidate_reuse_enabled:
                        if log_prob_for_loss.dim() != 3:
                            raise ValueError(
                                "Sample-K candidate reuse requires 3D candidate log-probabilities."
                            )
                        for required_key in (
                            "student_top_k_log_probs",
                            "teacher_on_student_log_probs",
                            OPD_ROLLOUT_REFERENCE_LOG_PROBS_KEY,
                        ):
                            if required_key not in model_inputs:
                                raise ValueError(
                                    f"Sample-K candidate reuse is missing {required_key}."
                                )
                        current_prefix_weights, reuse_metrics = (
                            self._compute_candidate_reuse_corrections(
                                sampled_log_probs=sampled_log_prob_for_loss,
                                rollout_reference_log_probs=model_inputs[
                                    OPD_ROLLOUT_REFERENCE_LOG_PROBS_KEY
                                ],
                                candidate_log_probs=log_prob_for_loss,
                                candidate_proposal_log_probs=model_inputs[
                                    "student_top_k_log_probs"
                                ],
                                teacher_candidate_log_probs=model_inputs[
                                    "teacher_on_student_log_probs"
                                ],
                                response_mask=response_mask,
                                prefix_method=(
                                    prefix_drift_method if prefix_drift_enable else "none"
                                ),
                                prefix_log_clip=prefix_drift_log_clip,
                                prefix_log_clip_mode=prefix_drift_log_clip_mode,
                                prefix_normalize=prefix_drift_normalize,
                                prefix_position_beta=prefix_drift_position_beta,
                                prefix_position_hmax=prefix_drift_position_hmax,
                                prefix_detach=prefix_drift_detach,
                            )
                        )
                        if current_prefix_weights is None:
                            model_inputs.pop(PREFIX_DRIFT_WEIGHTS_KEY, None)
                        else:
                            model_inputs[PREFIX_DRIFT_WEIGHTS_KEY] = current_prefix_weights
                        micro_batch_metrics.update(reuse_metrics)

                    # for fully_async_policy recipe
                    candidate_is_applied = advantages.dim() == 3
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if advantages.dim() == 3 and opd_current_samplek_no_candidate_is:
                            old_log_prob = log_prob_for_loss.detach()
                            candidate_is_applied = False
                        elif advantages.dim() == 3 and opd_current_samplek_force_candidate_is:
                            if "union_top_k_log_probs" in model_inputs:
                                old_log_prob = model_inputs["union_top_k_log_probs"]
                            elif "student_top_k_log_probs" in model_inputs:
                                old_log_prob = model_inputs["student_top_k_log_probs"]
                            else:
                                old_log_prob = model_inputs["old_log_probs"]
                        elif on_policy:
                            print("on_policy")
                            old_log_prob = log_prob_for_loss.detach()
                            candidate_is_applied = False
                        else:
                            print("off_policy")
                            if advantages.dim() == 3:
                                if "union_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["union_top_k_log_probs"]
                                elif "student_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["student_top_k_log_probs"]
                                else:
                                    old_log_prob = model_inputs["old_log_probs"]
                            else:
                                old_log_prob = model_inputs["old_log_probs"]

                    if use_decomposed_pi_old:
                        if advantages.dim() != 2 or log_prob_for_loss.dim() != 2:
                            raise ValueError(
                                "opd_advantage_mode=decomposed_pi_old currently supports only "
                                "sampled-token OPD with log_prob_top_k=0."
                            )
                        if "teacher_on_student_log_probs" not in model_inputs:
                            raise ValueError(
                                "opd_advantage_mode=decomposed_pi_old requires "
                                "teacher_on_student_log_probs in the batch."
                            )
                        if PREFIX_DRIFT_WEIGHTS_KEY in model_inputs:
                            raise ValueError(
                                "decomposed_pi_old computes prefix IS inside the objective; "
                                "prefix_drift_weights must not also be enabled."
                            )
                        rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                        if rollout_is_weights is not None:
                            raise ValueError(
                                "decomposed_pi_old already corrects its proposal distribution; "
                                "rollout_is_weights must not also be enabled."
                            )

                        teacher_log_prob_for_loss = model_inputs["teacher_on_student_log_probs"].to(
                            device=log_prob_for_loss.device,
                            dtype=log_prob_for_loss.dtype,
                        )
                        proposal_log_prob_for_loss = model_inputs.get(
                            "opd_proposal_log_probs", model_inputs["old_log_probs"]
                        ).to(device=log_prob_for_loss.device, dtype=log_prob_for_loss.dtype)
                        proximal_log_prob_for_loss = model_inputs.get(
                            "opd_proximal_log_probs", model_inputs["old_log_probs"]
                        ).to(device=log_prob_for_loss.device, dtype=log_prob_for_loss.dtype)
                        teacher_loss_mask = model_inputs.get("opd_teacher_mask", response_mask)
                        proximal_loss_mask = model_inputs.get("opd_proximal_mask", response_mask)
                        if format_mask is not None:
                            formatted_token_mask = format_mask.unsqueeze(-1)
                            teacher_loss_mask = teacher_loss_mask * formatted_token_mask
                            proximal_loss_mask = proximal_loss_mask * formatted_token_mask

                        decomposed_output = compute_decomposed_local_opd_loss(
                            current_log_probs=log_prob_for_loss,
                            proposal_log_probs=proposal_log_prob_for_loss,
                            teacher_log_probs=teacher_log_prob_for_loss,
                            proximal_log_probs=proximal_log_prob_for_loss,
                            response_mask=response_mask,
                            prefix_is_mode=opd_decomposed_prefix_is_mode,
                            prefix_is_min_weight=opd_decomposed_prefix_is_min_weight,
                            prefix_is_max_weight=opd_decomposed_prefix_is_max_weight,
                            proximal_loss_coef=opd_decomposed_proximal_coef,
                            loss_agg_mode=loss_agg_mode,
                            teacher_mask=teacher_loss_mask,
                            proximal_mask=proximal_loss_mask,
                            joint_loss_aggregation=opd_q_mixture_enable,
                        )
                        pg_loss = decomposed_output.loss
                        for key, value in decomposed_output.metrics.items():
                            micro_batch_metrics[f"opd_decomp/{key}"] = value
                        epoch_metric_keys = [
                            "prefix_log_ratio_abs_p95",
                            "prefix_geometric_log_ratio_abs_p95",
                            "prefix_is_weight_mean",
                            "prefix_is_weight_ess",
                            "prefix_is_conditional_weight_ess",
                            "prefix_is_cap_fraction",
                            "prefix_is_keep_fraction",
                            "prefix_is_lower_reject_fraction",
                            "prefix_is_upper_reject_fraction",
                            "teacher_valid_token_fraction",
                            "teacher_coef_abs_mean",
                            "proximal_coef_unscaled_abs_mean",
                            "proximal_coef_abs_mean",
                            "teacher_to_prox_abs_ratio",
                            "teacher_fraction_of_abs_coef",
                        ]
                        if opd_q_mixture_enable:
                            epoch_metric_keys.extend(
                                [
                                    "current_to_proposal_log_ratio_abs_mean",
                                    "current_to_proposal_log_ratio_abs_p95",
                                    "current_to_proposal_ratio_ess",
                                    "current_to_proposal_rkl_k3_mean",
                                    "q_teacher_advantage_abs_mean",
                                    "current_teacher_advantage_abs_mean",
                                    "teacher_advantage_delta_abs_mean",
                                    "teacher_advantage_sign_flip_fraction",
                                    "teacher_advantage_pearson",
                                    "student_current_to_proposal_log_ratio_abs_mean",
                                    "teacher_source_current_to_proposal_log_ratio_abs_mean",
                                    "proximal_rkl_k3_mean",
                                    "student_current_to_old_log_ratio_abs_mean",
                                    "student_current_to_old_log_ratio_abs_p95",
                                    "student_current_to_old_ratio_ess",
                                    "student_prefix_is_weight_ess",
                                    "teacher_source_prefix_is_weight_ess",
                                    "student_prefix_is_cap_fraction",
                                    "teacher_source_prefix_is_cap_fraction",
                                ]
                            )
                        for key in epoch_metric_keys:
                            micro_batch_metrics[f"opd_decomp/epoch_{ppo_epoch_idx + 1}/{key}"] = (
                                decomposed_output.metrics[key]
                            )

                    elif opd_advantage_mode in {"current_kl_is", "current_kl"}:
                        if "teacher_on_student_log_probs" not in model_inputs:
                            raise ValueError(
                                "opd_advantage_mode=current_kl_is requires teacher_on_student_log_probs in the batch"
                            )
                        teacher_log_prob_for_loss = model_inputs["teacher_on_student_log_probs"].to(
                            device=log_prob_for_loss.device,
                            dtype=log_prob_for_loss.dtype,
                        )
                        if teacher_log_prob_for_loss.shape != log_prob_for_loss.shape:
                            raise ValueError(
                                "teacher_on_student_log_probs shape must match current student log probs for "
                                f"opd_advantage_mode=current_kl_is, got teacher={teacher_log_prob_for_loss.shape}, "
                                f"student={log_prob_for_loss.shape}"
                            )
                        eos_future_teacher_log_probs = teacher_log_prob_for_loss
                        if opd_q_prefix_samplek:
                            if advantages.dim() != 3:
                                raise ValueError(
                                    "Q-prefix current teacher advantages require 3D sample-k candidates."
                                )
                            if OPD_Q_MIXTURE_PRIOR_ALPHA_KEY not in model_inputs:
                                raise ValueError(
                                    "Q-prefix sample-k is missing the trajectory-mixture prior alpha."
                                )
                            if sample_k_kl_plus_one:
                                raise ValueError(
                                    "Q-prefix sample-k current teacher advantages require "
                                    "sample_k_kl_plus_one=False."
                                )
                            q_samplek_output = compute_q_mixture_samplek_teacher_advantages(
                                current_log_probs=log_prob_for_loss,
                                teacher_log_probs=teacher_log_prob_for_loss,
                                response_mask=response_mask,
                                mixture_alpha=model_inputs[OPD_Q_MIXTURE_PRIOR_ALPHA_KEY],
                                mode=opd_q_mixture_teacher_advantage_mode,
                            )
                            advantages = q_samplek_output.teacher_advantages
                            for key, value in q_samplek_output.metrics.items():
                                micro_batch_metrics[f"opd_q_samplek/{key}"] = value
                        else:
                            if terminal_aware_enable:
                                if "student_top_k_ids" not in model_inputs:
                                    raise ValueError("terminal-aware OPD is missing student_top_k_ids.")
                                eos_token_id = data.meta_info.get("eos_token_id", None)
                                if eos_token_id is None:
                                    raise ValueError("terminal-aware OPD is missing eos_token_id metadata.")
                                terminal_mask = response_mask.bool() & model_inputs["responses"].eq(
                                    int(eos_token_id)
                                )
                                candidate_ids = model_inputs["student_top_k_ids"]
                                if terminal_mask.any() and terminal_objective_mode != "teacher_remap_only":
                                    if not candidate_ids[..., 0][terminal_mask].eq(int(eos_token_id)).all():
                                        raise ValueError(
                                            "terminal-aware OPD requires exact EOS in candidate slot zero."
                                        )
                                    conditional_start = 1
                                    if (
                                        terminal_secondary_token_id is not None
                                        and terminal_objective_mode != "conservative_kl"
                                    ):
                                        if not candidate_ids[..., 1][terminal_mask].eq(
                                            terminal_secondary_token_id
                                        ).all():
                                            raise ValueError(
                                                "dual-token terminal-aware OPD requires the secondary token "
                                                "in candidate slot one."
                                            )
                                        conditional_start = 2
                                    conditional_ids = candidate_ids[..., conditional_start:][terminal_mask]
                                    if conditional_ids.eq(int(eos_token_id)).any():
                                        raise ValueError(
                                            "terminal-aware OPD conditional candidates must exclude EOS."
                                        )
                                    if (
                                        terminal_secondary_token_id is not None
                                        and terminal_objective_mode != "conservative_kl"
                                        and conditional_ids.eq(terminal_secondary_token_id).any()
                                    ):
                                        raise ValueError(
                                            "dual-token terminal-aware OPD conditional candidates must "
                                            "exclude the secondary token."
                                        )
                                if terminal_objective_mode == "teacher_remap_only":
                                    if OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY not in model_inputs:
                                        raise ValueError(
                                            "teacher-remap-only OPD is missing the teacher secondary-token "
                                            "log-probability."
                                        )
                                    remap_output = prepare_terminal_teacher_remap_only_objective(
                                        teacher_log_probs=teacher_log_prob_for_loss,
                                        candidate_ids=candidate_ids,
                                        teacher_secondary_log_probs=model_inputs[
                                            OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY
                                        ],
                                        responses=model_inputs["responses"],
                                        response_mask=response_mask,
                                        eos_token_id=int(eos_token_id),
                                        secondary_token_id=int(terminal_secondary_token_id),
                                        teacher_remap_floor=terminal_teacher_remap_floor,
                                    )
                                    eos_future_teacher_log_probs = remap_output.remapped_teacher_log_probs
                                    advantages = (
                                        remap_output.remapped_teacher_log_probs
                                        - log_prob_for_loss.detach()
                                    )
                                    terminal_count = remap_output.terminal_mask.sum()
                                    terminal_denom = terminal_count.clamp_min(1)
                                    valid_state_count = response_mask.bool().sum()
                                    eos_candidate_count = remap_output.eos_candidate_mask.sum()
                                    eos_candidate_denom = eos_candidate_count.clamp_min(1)
                                    secondary_candidate_count = remap_output.secondary_candidate_mask.sum()
                                    valid_candidate_denom = (
                                        valid_state_count.clamp_min(1) * float(candidate_ids.size(-1))
                                    )
                                    prefix = "opd_terminal_teacher_remap_only"
                                    micro_batch_metrics[f"{prefix}/enabled"] = 1.0
                                    micro_batch_metrics[f"{prefix}/natural_terminal_count"] = (
                                        terminal_count.detach().item()
                                    )
                                    micro_batch_metrics[f"{prefix}/sampled_eos_candidate_count"] = (
                                        eos_candidate_count.detach().item()
                                    )
                                    micro_batch_metrics[f"{prefix}/sampled_secondary_candidate_count"] = (
                                        secondary_candidate_count.detach().item()
                                    )
                                    micro_batch_metrics[f"{prefix}/sampled_eos_candidate_fraction"] = (
                                        eos_candidate_count / valid_candidate_denom
                                    ).detach().item()
                                    micro_batch_metrics[f"{prefix}/sampled_secondary_candidate_fraction"] = (
                                        secondary_candidate_count / valid_candidate_denom
                                    ).detach().item()
                                    micro_batch_metrics[f"{prefix}/teacher_secondary_probability_mean"] = (
                                        remap_output.teacher_secondary_probability.sum() / terminal_denom
                                    ).detach().item()
                                    micro_batch_metrics[f"{prefix}/teacher_stop_probability_mean"] = (
                                        remap_output.teacher_stop_probability.sum() / eos_candidate_denom
                                    ).detach().item()
                                    remapped_eos_probability = (
                                        remap_output.remapped_teacher_log_probs.exp()
                                        * remap_output.eos_candidate_mask
                                    )
                                    micro_batch_metrics[f"{prefix}/teacher_remapped_eos_probability_mean"] = (
                                        remapped_eos_probability.sum() / eos_candidate_denom
                                    ).detach().item()
                                    current_eos_probability = (
                                        log_prob_for_loss.detach().float().exp()
                                        * remap_output.eos_candidate_mask
                                    )
                                    micro_batch_metrics[f"{prefix}/current_eos_probability_mean"] = (
                                        current_eos_probability.sum() / eos_candidate_denom
                                    ).detach().item()
                                    if OPD_TERMINAL_STUDENT_SECONDARY_LOG_PROBS_KEY in model_inputs:
                                        student_secondary_probability = model_inputs[
                                            OPD_TERMINAL_STUDENT_SECONDARY_LOG_PROBS_KEY
                                        ].to(device=log_prob_for_loss.device).float().exp()
                                        student_secondary_probability = student_secondary_probability.masked_fill(
                                            ~remap_output.terminal_mask, 0.0
                                        )
                                        micro_batch_metrics[
                                            f"{prefix}/current_secondary_probability_mean"
                                        ] = (
                                            student_secondary_probability.sum() / terminal_denom
                                        ).detach().item()
                                        if remap_output.terminal_mask.any():
                                            micro_batch_metrics[
                                                f"{prefix}/current_secondary_probability_max"
                                            ] = student_secondary_probability[
                                                remap_output.terminal_mask
                                            ].max().detach().item()
                                        else:
                                            micro_batch_metrics[
                                                f"{prefix}/current_secondary_probability_max"
                                            ] = 0.0
                                elif terminal_objective_mode == "safe_continue":
                                    safe_output = prepare_terminal_safe_continue_objective(
                                        current_log_probs=log_prob_for_loss,
                                        teacher_log_probs=teacher_log_prob_for_loss,
                                        responses=model_inputs["responses"],
                                        response_mask=response_mask,
                                        eos_token_id=int(eos_token_id),
                                        secondary_token_enabled=terminal_secondary_token_id is not None,
                                        subtract_score_baseline=sample_k_kl_plus_one,
                                    )
                                    log_prob_for_loss = safe_output.policy_log_probs
                                    advantages = safe_output.advantages
                                    opd_terminal_aware_gate_loss = safe_output.gate_loss
                                    terminal_count = safe_output.terminal_mask.sum()
                                    terminal_denom = terminal_count.clamp_min(1)

                                    def safe_terminal_mean(values):
                                        return (
                                            values.masked_fill(~safe_output.terminal_mask, 0.0).sum()
                                            / terminal_denom
                                        ).detach().item()

                                    def safe_terminal_max(values):
                                        if not safe_output.terminal_mask.any():
                                            return 0.0
                                        return values[safe_output.terminal_mask].max().detach().item()

                                    micro_batch_metrics["opd_terminal_safe_continue/enabled"] = 1.0
                                    micro_batch_metrics[
                                        "opd_terminal_safe_continue/natural_terminal_count"
                                    ] = terminal_count.detach().item()
                                    micro_batch_metrics[
                                        "opd_terminal_safe_continue/current_eos_probability_mean"
                                    ] = safe_terminal_mean(safe_output.current_eos_probability)
                                    micro_batch_metrics[
                                        "opd_terminal_safe_continue/current_secondary_probability_mean"
                                    ] = safe_terminal_mean(safe_output.current_secondary_probability)
                                    micro_batch_metrics[
                                        "opd_terminal_safe_continue/current_secondary_probability_max"
                                    ] = safe_terminal_max(safe_output.current_secondary_probability)
                                    micro_batch_metrics[
                                        "opd_terminal_safe_continue/teacher_eos_probability_mean"
                                    ] = safe_terminal_mean(safe_output.teacher_eos_probability)
                                    micro_batch_metrics[
                                        "opd_terminal_safe_continue/teacher_secondary_probability_mean"
                                    ] = safe_terminal_mean(safe_output.teacher_secondary_probability)
                                    micro_batch_metrics[
                                        "opd_terminal_safe_continue/teacher_stop_probability_mean"
                                    ] = safe_terminal_mean(safe_output.teacher_stop_probability)
                                    micro_batch_metrics[
                                        "opd_terminal_safe_continue/current_content_probability_mean"
                                    ] = safe_terminal_mean(safe_output.current_content_probability)
                                    micro_batch_metrics[
                                        "opd_terminal_safe_continue/stop_signal_mean"
                                    ] = safe_terminal_mean(safe_output.stop_signal)
                                    micro_batch_metrics[
                                        "opd_terminal_safe_continue/continuation_signal_mean"
                                    ] = safe_terminal_mean(safe_output.continuation_signal)
                                    micro_batch_metrics[
                                        "opd_terminal_safe_continue/safe_gate_advantage_mean"
                                    ] = safe_terminal_mean(safe_output.safe_gate_advantage)
                                    micro_batch_metrics[
                                        "opd_terminal_safe_continue/gate_active_fraction"
                                    ] = safe_terminal_mean(safe_output.gate_active)
                                    micro_batch_metrics[
                                        "opd_terminal_safe_continue/gate_surrogate_terminal_mean"
                                    ] = safe_terminal_mean(safe_output.gate_loss)
                                elif terminal_objective_mode == "conservative_kl":
                                    if sample_k_kl_plus_one:
                                        raise ValueError(
                                            "conservative terminal KL requires sample_k_kl_plus_one=False."
                                        )
                                    teacher_secondary_log_probs = None
                                    if terminal_teacher_remap_enable:
                                        if OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY not in model_inputs:
                                            raise ValueError(
                                                "teacher EOS remapping is missing the teacher secondary-token "
                                                "log-probability."
                                            )
                                        teacher_secondary_log_probs = model_inputs[
                                            OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY
                                        ]
                                    student_topm_ids = None
                                    student_topm_log_probs = None
                                    teacher_topm_log_probs = None
                                    if terminal_kl_baseline_mode == "topm_coarse":
                                        for required_key in (
                                            OPD_TERMINAL_STUDENT_TOPM_IDS_KEY,
                                            OPD_TERMINAL_STUDENT_TOPM_LOG_PROBS_KEY,
                                            OPD_TERMINAL_TEACHER_TOPM_LOG_PROBS_KEY,
                                        ):
                                            if required_key not in model_inputs:
                                                raise ValueError(f"topm_coarse is missing {required_key}.")
                                        student_topm_ids = model_inputs[OPD_TERMINAL_STUDENT_TOPM_IDS_KEY]
                                        student_topm_log_probs = model_inputs[
                                            OPD_TERMINAL_STUDENT_TOPM_LOG_PROBS_KEY
                                        ]
                                        teacher_topm_log_probs = model_inputs[
                                            OPD_TERMINAL_TEACHER_TOPM_LOG_PROBS_KEY
                                        ]
                                    conservative_output = prepare_terminal_conservative_kl_objective(
                                        current_log_probs=log_prob_for_loss,
                                        teacher_log_probs=teacher_log_prob_for_loss,
                                        candidate_ids=candidate_ids,
                                        teacher_secondary_log_probs=teacher_secondary_log_probs,
                                        responses=model_inputs["responses"],
                                        response_mask=response_mask,
                                        eos_token_id=int(eos_token_id),
                                        secondary_token_id=terminal_secondary_token_id,
                                        teacher_remap_enable=terminal_teacher_remap_enable,
                                        teacher_remap_floor=terminal_teacher_remap_floor,
                                        baseline_mode=terminal_kl_baseline_mode,
                                        apply_eos_relu=True,
                                        student_topm_ids=student_topm_ids,
                                        student_topm_log_probs=student_topm_log_probs,
                                        teacher_topm_log_probs=teacher_topm_log_probs,
                                    )
                                    log_prob_for_loss = conservative_output.policy_log_probs
                                    advantages = conservative_output.advantages
                                    terminal_count = conservative_output.terminal_mask.sum()
                                    terminal_denom = terminal_count.clamp_min(1)

                                    def conservative_terminal_mean(values):
                                        return (
                                            values.masked_fill(
                                                ~conservative_output.terminal_mask, 0.0
                                            ).sum()
                                            / terminal_denom
                                        ).detach().item()

                                    prefix = "opd_terminal_conservative_kl"
                                    micro_batch_metrics[f"{prefix}/enabled"] = 1.0
                                    micro_batch_metrics[f"{prefix}/natural_terminal_count"] = (
                                        terminal_count.detach().item()
                                    )
                                    micro_batch_metrics[f"{prefix}/teacher_remap_enabled"] = float(
                                        terminal_teacher_remap_enable
                                    )
                                    micro_batch_metrics[f"{prefix}/topm_coarse_enabled"] = float(
                                        terminal_kl_baseline_mode == "topm_coarse"
                                    )
                                    for name, values in (
                                        ("current_eos_probability_mean", conservative_output.current_eos_probability),
                                        ("teacher_eos_probability_mean", conservative_output.teacher_eos_probability),
                                        (
                                            "teacher_secondary_probability_mean",
                                            conservative_output.teacher_secondary_probability,
                                        ),
                                        (
                                            "teacher_remapped_eos_probability_mean",
                                            conservative_output.teacher_remapped_eos_probability,
                                        ),
                                        (
                                            "teacher_remapped_secondary_probability_mean",
                                            conservative_output.teacher_remapped_secondary_probability,
                                        ),
                                        ("kl_baseline_mean", conservative_output.kl_baseline),
                                        ("eos_advantage_mean", conservative_output.eos_advantage),
                                        ("eos_gate_active_fraction", conservative_output.eos_gate_active),
                                    ):
                                        micro_batch_metrics[f"{prefix}/{name}"] = (
                                            conservative_terminal_mean(values)
                                        )
                                    if OPD_TERMINAL_STUDENT_SECONDARY_LOG_PROBS_KEY in model_inputs:
                                        student_secondary_probability = model_inputs[
                                            OPD_TERMINAL_STUDENT_SECONDARY_LOG_PROBS_KEY
                                        ].to(device=log_prob_for_loss.device).float().exp()
                                        micro_batch_metrics[
                                            f"{prefix}/current_secondary_probability_mean"
                                        ] = conservative_terminal_mean(student_secondary_probability)
                                        if conservative_output.terminal_mask.any():
                                            micro_batch_metrics[
                                                f"{prefix}/current_secondary_probability_max"
                                            ] = student_secondary_probability[
                                                conservative_output.terminal_mask
                                            ].max().detach().item()
                                        else:
                                            micro_batch_metrics[
                                                f"{prefix}/current_secondary_probability_max"
                                            ] = 0.0
                                else:
                                    if OPD_TERMINAL_BEHAVIOR_EOS_LOG_PROBS_KEY not in model_inputs:
                                        raise ValueError(
                                            "terminal-aware OPD is missing the fixed behavior EOS anchor."
                                        )
                                    behavior_secondary_log_probs = None
                                    if terminal_secondary_token_id is not None:
                                        if OPD_TERMINAL_BEHAVIOR_SECONDARY_LOG_PROBS_KEY not in model_inputs:
                                            raise ValueError(
                                                "dual-token terminal-aware OPD is missing the fixed behavior "
                                                "secondary-token probability."
                                            )
                                        behavior_secondary_log_probs = model_inputs[
                                            OPD_TERMINAL_BEHAVIOR_SECONDARY_LOG_PROBS_KEY
                                        ]
                                    terminal_aware_output = prepare_terminal_aware_objective(
                                        current_log_probs=log_prob_for_loss,
                                        teacher_log_probs=teacher_log_prob_for_loss,
                                        behavior_eos_log_probs=model_inputs[
                                            OPD_TERMINAL_BEHAVIOR_EOS_LOG_PROBS_KEY
                                        ],
                                        behavior_secondary_log_probs=behavior_secondary_log_probs,
                                        responses=model_inputs["responses"],
                                        response_mask=response_mask,
                                        eos_token_id=int(eos_token_id),
                                        anchor_mode=terminal_anchor_mode,
                                        subtract_score_baseline=sample_k_kl_plus_one,
                                    )
                                    log_prob_for_loss = terminal_aware_output.policy_log_probs
                                    advantages = terminal_aware_output.advantages
                                    opd_terminal_aware_gate_loss = terminal_aware_output.gate_loss
                                    terminal_count = terminal_aware_output.terminal_mask.sum()
                                    terminal_denom = terminal_count.clamp_min(1)

                                    def terminal_mean(values):
                                        return (
                                            values.masked_fill(~terminal_aware_output.terminal_mask, 0.0).sum()
                                            / terminal_denom
                                        ).detach().item()

                                    def terminal_max(values):
                                        if not terminal_aware_output.terminal_mask.any():
                                            return 0.0
                                        return values[terminal_aware_output.terminal_mask].max().detach().item()

                                    micro_batch_metrics["opd_terminal_aware/enabled"] = 1.0
                                    micro_batch_metrics["opd_terminal_aware/natural_terminal_count"] = (
                                        terminal_count.detach().item()
                                    )
                                    micro_batch_metrics["opd_terminal_aware/behavior_eos_probability_mean"] = (
                                        terminal_mean(terminal_aware_output.behavior_eos_probability)
                                    )
                                    micro_batch_metrics["opd_terminal_aware/current_eos_probability_mean"] = (
                                        terminal_mean(terminal_aware_output.current_eos_probability)
                                    )
                                    micro_batch_metrics["opd_terminal_aware/teacher_eos_probability_mean"] = (
                                        terminal_mean(terminal_aware_output.teacher_eos_probability)
                                    )
                                    micro_batch_metrics[
                                        "opd_terminal_aware/behavior_secondary_probability_mean"
                                    ] = terminal_mean(terminal_aware_output.behavior_secondary_probability)
                                    micro_batch_metrics[
                                        "opd_terminal_aware/current_secondary_probability_mean"
                                    ] = terminal_mean(terminal_aware_output.current_secondary_probability)
                                    micro_batch_metrics[
                                        "opd_terminal_aware/current_secondary_probability_max"
                                    ] = terminal_max(terminal_aware_output.current_secondary_probability)
                                    micro_batch_metrics[
                                        "opd_terminal_aware/teacher_secondary_probability_mean"
                                    ] = terminal_mean(terminal_aware_output.teacher_secondary_probability)
                                    micro_batch_metrics[
                                        "opd_terminal_aware/teacher_stop_probability_mean"
                                    ] = terminal_mean(terminal_aware_output.teacher_stop_probability)
                                    micro_batch_metrics["opd_terminal_aware/anchor_probability_mean"] = (
                                        terminal_mean(terminal_aware_output.anchor_probability)
                                    )
                                    micro_batch_metrics["opd_terminal_aware/content_mass_mean"] = terminal_mean(
                                        terminal_aware_output.content_mass
                                    )
                                    micro_batch_metrics[
                                        "opd_terminal_aware/current_content_probability_mean"
                                    ] = terminal_mean(terminal_aware_output.current_content_probability)
                                    micro_batch_metrics[
                                        "opd_terminal_aware/current_minus_behavior_eos_mean"
                                    ] = terminal_mean(
                                        terminal_aware_output.current_eos_probability
                                        - terminal_aware_output.behavior_eos_probability
                                    )
                                    micro_batch_metrics[
                                        "opd_terminal_aware/gate_kl_terminal_mean"
                                    ] = terminal_mean(terminal_aware_output.gate_loss)
                                    micro_batch_metrics[
                                        "opd_terminal_aware/current_minus_behavior_secondary_mean"
                                    ] = terminal_mean(
                                        terminal_aware_output.current_secondary_probability
                                        - terminal_aware_output.behavior_secondary_probability
                                    )
                                    micro_batch_metrics[
                                        "opd_terminal_aware/teacher_floor_active_fraction"
                                    ] = terminal_mean(terminal_aware_output.teacher_floor_active)
                                    micro_batch_metrics["opd_terminal_aware/secondary_token_enabled"] = float(
                                        terminal_secondary_token_id is not None
                                    )
                                    if terminal_secondary_token_id is not None:
                                        micro_batch_metrics["opd_terminal_aware/secondary_token_id"] = float(
                                            terminal_secondary_token_id
                                        )
                                old_log_prob, candidate_is_applied = self._terminal_candidate_reference(
                                    current_log_probs=log_prob_for_loss,
                                    proposal_log_probs=model_inputs["student_top_k_log_probs"],
                                    candidate_reuse_enabled=samplek_candidate_reuse_enabled,
                                )
                            else:
                                advantages = teacher_log_prob_for_loss - log_prob_for_loss
                                if opd_raw_advantage_clip is not None:
                                    raw_clip_output = apply_raw_opd_advantage_clip(
                                        advantages=advantages,
                                        response_mask=response_mask,
                                        clip=opd_raw_advantage_clip,
                                    )
                                    advantages = raw_clip_output.advantages
                                    for key, value in raw_clip_output.metrics.items():
                                        micro_batch_metrics[f"opd/raw_advantage_clip/{key}"] = value
                                if advantages.dim() == 3 and sample_k_kl_plus_one:
                                    advantages = advantages - 1.0
                        if opd_samplek_eos_negative_relu_enable:
                            if advantages.dim() != 3:
                                raise ValueError(
                                    "Sample-K EOS negative ReLU requires 3D candidate advantages."
                                )
                            if "student_top_k_ids" not in model_inputs:
                                raise ValueError(
                                    "Sample-K EOS negative ReLU is missing student_top_k_ids."
                                )
                            eos_token_id = data.meta_info.get("eos_token_id", None)
                            if eos_token_id is None:
                                raise ValueError(
                                    "Sample-K EOS negative ReLU is missing eos_token_id metadata."
                                )
                            eos_relu_output = apply_samplek_eos_negative_relu(
                                advantages=advantages,
                                candidate_ids=model_inputs["student_top_k_ids"],
                                response_mask=response_mask,
                                eos_token_id=int(eos_token_id),
                            )
                            advantages = eos_relu_output.advantages
                            micro_batch_metrics["opd_eos_negative_relu/enabled"] = 1.0
                            for key, value in eos_relu_output.metrics.items():
                                micro_batch_metrics[f"opd_eos_negative_relu/{key}"] = value
                        if adaptive_head_tail_negative_elu_enable:
                            validate_adaptive_head_tail_negative_elu_configuration(
                                candidate_mode=diagnostic_candidate_mode,
                                advantage_mode=opd_advantage_mode,
                                advantages=advantages,
                                candidate_estimator_weights_present=(
                                    "candidate_estimator_weights" in model_inputs
                                ),
                                sample_k_kl_plus_one=sample_k_kl_plus_one,
                                raw_advantage_clip=opd_raw_advantage_clip,
                            )
                            negative_elu_output = apply_adaptive_head_tail_negative_elu(
                                advantages=advantages,
                                response_mask=response_mask,
                                threshold=adaptive_head_tail_negative_elu_threshold,
                                tau=adaptive_head_tail_negative_elu_tau,
                            )
                            advantages = negative_elu_output.advantages
                            for key, value in negative_elu_output.metrics.items():
                                micro_batch_metrics[f"adaptive_head_tail/negative_elu/{key}"] = value
                        if opd_samplek_entropy_coef > 0.0:
                            if advantages.dim() != 3:
                                raise ValueError(
                                    "opd_samplek_entropy_coef requires 3D sample-k candidate advantages."
                                )
                            samplek_entropy_output = add_samplek_entropy_advantages(
                                teacher_advantages=advantages,
                                current_log_probs=log_prob_for_loss,
                                response_mask=response_mask,
                                coefficient=opd_samplek_entropy_coef,
                            )
                            advantages = samplek_entropy_output.advantages
                            for key, value in samplek_entropy_output.metrics.items():
                                micro_batch_metrics[f"opd_samplek_entropy/{key}"] = value
                        if advantages.dim() == 3:
                            raw_candidate_means = advantages.mean(dim=-1)
                            raw_candidate_abs_means = advantages.abs().mean(dim=-1)
                            micro_batch_metrics["opd/samplek_advantage_raw_mean"] = (
                                verl_F.masked_mean(raw_candidate_means, response_mask).detach().item()
                            )
                            micro_batch_metrics["opd/samplek_advantage_raw_abs_mean"] = (
                                verl_F.masked_mean(raw_candidate_abs_means, response_mask).detach().item()
                            )
                        if opd_samplek_advantage_centering != "none":
                            if "candidate_estimator_weights" in model_inputs:
                                raise ValueError(
                                    "leave_one_out sample-k advantage centering currently requires equally "
                                    "weighted candidates; disable candidate estimator weights."
                                )
                            advantages = apply_samplek_advantage_centering(
                                advantages,
                                mode=opd_samplek_advantage_centering,
                            )
                        if opd_samplek_loo_variance_filter_threshold is not None:
                            validate_samplek_loo_variance_filter_configuration(
                                threshold=opd_samplek_loo_variance_filter_threshold,
                                advantage_centering=opd_samplek_advantage_centering,
                                advantages=advantages,
                                candidate_estimator_weights_present=(
                                    "candidate_estimator_weights" in model_inputs
                                ),
                                entropy_coefficient=opd_samplek_entropy_coef,
                                raw_advantage_clip=opd_raw_advantage_clip,
                            )
                            variance_filter_output = apply_samplek_loo_variance_filter(
                                centered_advantages=advantages,
                                response_mask=response_mask,
                                threshold=opd_samplek_loo_variance_filter_threshold,
                                selection=opd_samplek_loo_variance_filter_selection,
                                mode=opd_samplek_loo_variance_filter_mode,
                                soft_base_weight=opd_samplek_loo_variance_filter_soft_base_weight,
                                soft_active_bonus=opd_samplek_loo_variance_filter_soft_active_bonus,
                                expectile_tau=opd_samplek_loo_variance_filter_expectile_tau,
                            )
                            advantages = variance_filter_output.advantages
                            for key, value in variance_filter_output.metrics.items():
                                micro_batch_metrics[f"opd_samplek/loo_variance_filter/{key}"] = value
                            micro_batch_metrics[
                                "opd_samplek/loo_variance_filter/threshold_mode_update0_quantile"
                            ] = float(
                                opd_samplek_loo_variance_filter_threshold_mode == "update0_quantile"
                            )
                        if advantages.dim() == 3:
                            candidate_means = advantages.mean(dim=-1)
                            candidate_abs_means = advantages.abs().mean(dim=-1)
                            micro_batch_metrics["opd/samplek_advantage_centered_mean"] = (
                                verl_F.masked_mean(candidate_means, response_mask).detach().item()
                            )
                            micro_batch_metrics["opd/samplek_advantage_centered_abs_mean"] = (
                                verl_F.masked_mean(candidate_abs_means, response_mask).detach().item()
                            )
                            micro_batch_metrics["opd/samplek_advantage_centered_mean_abs_max"] = (
                                (candidate_means.abs() * response_mask).max().detach().item()
                            )
                        micro_batch_metrics["opd/samplek_advantage_centering_leave_one_out"] = float(
                            opd_samplek_advantage_centering == "leave_one_out"
                        )
                        micro_batch_metrics["opd/samplek_advantage_centering_scale"] = float(
                            advantages.size(-1) / (advantages.size(-1) - 1)
                            if advantages.dim() == 3
                            and opd_samplek_advantage_centering == "leave_one_out"
                            else 1.0
                        )
                        if advantages.dim() == 3 and "candidate_estimator_weights" in model_inputs:
                            candidate_estimator_weights = model_inputs["candidate_estimator_weights"].to(
                                device=advantages.device,
                                dtype=advantages.dtype,
                            )
                            if candidate_estimator_weights.shape != advantages.shape:
                                raise ValueError(
                                    "candidate_estimator_weights shape must match 3D advantages for "
                                    f"opd_advantage_mode=current_kl_is, got weights={candidate_estimator_weights.shape}, "
                                    f"advantages={advantages.shape}"
                                )
                            advantages = advantages * candidate_estimator_weights
                            micro_batch_metrics["opd/current_kl_candidate_weighted"] = 1.0
                        else:
                            micro_batch_metrics["opd/current_kl_candidate_weighted"] = 0.0
                        if (
                            advantages.dim() == 3
                            and opd_samplek_candidate_aggregation == "mean"
                            and opd_samplek_influence_clip is None
                            and not diagnostic_forced_eos_enable
                        ):
                            advantages = advantages / advantages.size(-1)
                        advantages = advantages.detach()
                        micro_batch_metrics["opd/samplek_candidate_aggregation_mean"] = float(
                            advantages.dim() == 3 and opd_samplek_candidate_aggregation == "mean"
                        )
                        micro_batch_metrics["opd/samplek_candidate_count"] = float(
                            advantages.size(-1) if advantages.dim() == 3 else 1
                        )
                        micro_batch_metrics["opd/current_kl_plus_one"] = float(
                            advantages.dim() == 3 and sample_k_kl_plus_one
                        )
                        metric_advantages = advantages
                        if (
                            opd_samplek_influence_clip is not None
                            and metric_advantages.dim() == 3
                            and opd_samplek_candidate_aggregation == "mean"
                        ):
                            metric_advantages = metric_advantages / metric_advantages.size(-1)
                        micro_batch_metrics["opd/current_kl_adv_mean"] = verl_F.masked_mean(
                            metric_advantages.sum(dim=-1)
                            if metric_advantages.dim() == 3
                            else metric_advantages,
                            response_mask,
                        ).detach().item() * loss_scale_factor

                    samplek_influence_consumed_prefix = False
                    if opd_samplek_influence_clip is not None:
                        validate_samplek_influence_clip_configuration(
                            advantage_mode=opd_advantage_mode,
                            advantages=advantages,
                            no_candidate_is=not candidate_is_applied,
                            force_candidate_is=opd_current_samplek_force_candidate_is,
                            candidate_estimator_weights_present=(
                                "candidate_estimator_weights" in model_inputs
                            ),
                            source_normalize_enabled=opd_q_mixture_source_normalize_enable,
                        )

                    if (
                        opd_q_mixture_diagnostics_enable
                        and opd_q_prefix_samplek
                        and "opd_proximal_mask" in model_inputs
                    ):
                        if PREFIX_DRIFT_WEIGHTS_KEY not in model_inputs:
                            raise ValueError(
                                "Q-mixture diagnostics require prefix_drift_weights. "
                                "Enable prefix_drift for Q-prefix sample-k runs."
                            )
                        diagnostic_mask = response_mask
                        diagnostic_student_mask = model_inputs["opd_proximal_mask"].to(
                            device=response_mask.device,
                            dtype=response_mask.dtype,
                        )
                        if format_mask is not None:
                            formatted_mask = format_mask.unsqueeze(-1)
                            diagnostic_mask = diagnostic_mask * formatted_mask
                            diagnostic_student_mask = diagnostic_student_mask * formatted_mask
                        diagnostic_advantages = advantages
                        if (
                            opd_samplek_influence_clip is not None
                            and diagnostic_advantages.dim() == 3
                            and opd_samplek_candidate_aggregation == "mean"
                        ):
                            diagnostic_advantages = (
                                diagnostic_advantages / diagnostic_advantages.size(-1)
                            )
                        self._add_q_mixture_offpolicy_diagnostics(
                            metrics=micro_batch_metrics,
                            advantages=diagnostic_advantages,
                            response_mask=diagnostic_mask,
                            student_mask=diagnostic_student_mask,
                            prefix_weights=model_inputs[PREFIX_DRIFT_WEIGHTS_KEY],
                            raw_prefix_weights=model_inputs.get(PREFIX_DRIFT_RAW_WEIGHTS_KEY, None),
                            source_weights=model_inputs.get(OPD_Q_MIXTURE_SOURCE_WEIGHTS_KEY, None),
                        )

                    if diagnostic_forced_eos_enable:
                        if FORCED_EOS_ESTIMATOR_WEIGHTS_KEY not in model_inputs:
                            raise ValueError(
                                "forced-EOS diagnostic is missing its dedicated estimator weights."
                            )
                        if "student_top_k_ids" not in model_inputs:
                            raise ValueError("forced-EOS diagnostic is missing student_top_k_ids.")
                        eos_token_id = data.meta_info.get("eos_token_id", None)
                        if eos_token_id is None:
                            raise ValueError("forced-EOS diagnostic is missing eos_token_id metadata.")
                        influence_prefix_weights = model_inputs.get(PREFIX_DRIFT_WEIGHTS_KEY, None)
                        influence_mask = response_mask
                        if format_mask is not None:
                            influence_mask = influence_mask * format_mask.unsqueeze(-1)
                        forced_eos_output = apply_forced_eos_diagnostic_influence(
                            advantages=advantages,
                            response_mask=influence_mask,
                            prefix_weights=influence_prefix_weights,
                            estimator_weights=model_inputs[FORCED_EOS_ESTIMATOR_WEIGHTS_KEY],
                            candidate_ids=model_inputs["student_top_k_ids"],
                            eos_token_id=int(eos_token_id),
                            clip=opd_samplek_influence_clip,
                        )
                        advantages = forced_eos_output.advantages / advantages.size(-1)
                        samplek_influence_consumed_prefix = forced_eos_output.consumed_prefix
                        for key, value in forced_eos_output.metrics.items():
                            micro_batch_metrics[f"opd_diagnostic/forced_eos/{key}"] = value
                        for key, value in forced_eos_output.clip_metrics.items():
                            micro_batch_metrics[f"opd_samplek/influence_clip/{key}"] = value
                        if samplek_influence_consumed_prefix:
                            micro_batch_metrics["prefix_drift/applied_in_actor"] = 1.0
                    elif opd_samplek_influence_clip is not None:
                        influence_prefix_weights = model_inputs.get(PREFIX_DRIFT_WEIGHTS_KEY, None)
                        influence_mask = response_mask
                        if format_mask is not None:
                            influence_mask = influence_mask * format_mask.unsqueeze(-1)
                        influence_output = apply_samplek_influence_clip(
                            advantages=advantages,
                            response_mask=influence_mask,
                            prefix_weights=influence_prefix_weights,
                            clip=opd_samplek_influence_clip,
                        )
                        advantages = influence_output.advantages
                        samplek_influence_consumed_prefix = influence_prefix_weights is not None
                        if opd_samplek_candidate_aggregation == "mean":
                            advantages = advantages / advantages.size(-1)
                        for key, value in influence_output.metrics.items():
                            micro_batch_metrics[f"opd_samplek/influence_clip/{key}"] = value
                        if samplek_influence_consumed_prefix:
                            micro_batch_metrics["prefix_drift/applied_in_actor"] = 1.0

                    if (
                        not use_decomposed_pi_old
                        and PREFIX_DRIFT_WEIGHTS_KEY in model_inputs
                        and not samplek_influence_consumed_prefix
                    ):
                        prefix_drift_weights = model_inputs[PREFIX_DRIFT_WEIGHTS_KEY].to(
                            device=advantages.device,
                            dtype=advantages.dtype,
                        )
                        if prefix_drift_weights.shape != response_mask.shape:
                            raise ValueError(
                                "prefix_drift_weights shape must match response_mask, "
                                f"got weights={prefix_drift_weights.shape}, mask={response_mask.shape}"
                            )
                        if advantages.dim() == 3:
                            prefix_drift_weights = prefix_drift_weights.unsqueeze(-1)
                        elif advantages.dim() != 2:
                            raise ValueError(
                                "prefix_drift_weights supports 2D or 3D advantages, "
                                f"got advantages={advantages.shape}"
                            )
                        advantages = advantages * prefix_drift_weights
                        micro_batch_metrics["prefix_drift/applied_in_actor"] = 1.0

                    if opd_q_mixture_source_normalize_enable:
                        if OPD_Q_MIXTURE_SOURCE_WEIGHTS_KEY not in model_inputs:
                            raise ValueError(
                                "Q-mixture source normalization is missing its precomputed token weights."
                            )
                        source_weights = model_inputs[OPD_Q_MIXTURE_SOURCE_WEIGHTS_KEY].to(
                            device=advantages.device,
                            dtype=advantages.dtype,
                        )
                        if source_weights.shape != response_mask.shape:
                            raise ValueError(
                                "Q-mixture source weights must match response_mask, "
                                f"got weights={source_weights.shape}, mask={response_mask.shape}."
                            )
                        advantages = advantages * (
                            source_weights.unsqueeze(-1) if advantages.dim() == 3 else source_weights
                        )
                        valid_source_weights = source_weights[response_mask > 0.5].float()
                        micro_batch_metrics["opd_q_source_normalize/applied_in_actor"] = 1.0
                        micro_batch_metrics["opd_q_source_normalize/actor_weight_mean"] = (
                            valid_source_weights.mean().item()
                        )
                        micro_batch_metrics["opd_q_source_normalize/actor_weight_min"] = (
                            valid_source_weights.min().item()
                        )
                        micro_batch_metrics["opd_q_source_normalize/actor_weight_max"] = (
                            valid_source_weights.max().item()
                        )

                    if not use_decomposed_pi_old:
                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                        rollout_is_weights = model_inputs.get("rollout_is_weights", None)


                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob_for_loss,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                            format_mask=format_mask,
                        )
                        micro_batch_metrics.update(pg_metrics)
                        if opd_terminal_aware_gate_loss is not None:
                            gate_loss_mat = opd_terminal_aware_gate_loss * terminal_gate_coef
                            if PREFIX_DRIFT_WEIGHTS_KEY in model_inputs:
                                gate_prefix_weights = model_inputs[PREFIX_DRIFT_WEIGHTS_KEY].to(
                                    device=gate_loss_mat.device,
                                    dtype=gate_loss_mat.dtype,
                                )
                                gate_loss_mat = gate_loss_mat * gate_prefix_weights
                                micro_batch_metrics["opd_terminal_aware/gate_prefix_weighted"] = 1.0
                            else:
                                micro_batch_metrics["opd_terminal_aware/gate_prefix_weighted"] = 0.0
                            gate_loss_mask = response_mask
                            if format_mask is not None:
                                gate_loss_mask = gate_loss_mask * format_mask.unsqueeze(-1)
                            terminal_gate_loss = agg_loss(
                                loss_mat=gate_loss_mat,
                                loss_mask=gate_loss_mask,
                                loss_agg_mode=loss_agg_mode,
                            )
                            pg_loss = pg_loss + terminal_gate_loss
                            micro_batch_metrics["opd_terminal_aware/gate_coef"] = terminal_gate_coef
                            micro_batch_metrics["opd_terminal_aware/gate_loss"] = (
                                terminal_gate_loss.detach().item()
                            )
                        if eos_future_enable:
                            if current_eos_log_probs is None:
                                raise ValueError(
                                    "EOS future correction requires differentiable current EOS log-probabilities."
                                )
                            eos_future_loss_mask = response_mask
                            if format_mask is not None:
                                eos_future_loss_mask = eos_future_loss_mask * format_mask.unsqueeze(-1)
                            if eos_future_mode == EOS_FUTURE_MODE_FIXED_SIGNED:
                                eos_future_output = prepare_eos_future_correction_loss(
                                    current_eos_log_probs=current_eos_log_probs,
                                    fixed_future_value=model_inputs[OPD_EOS_FUTURE_VALUE_KEY],
                                    correction_mask=model_inputs[OPD_EOS_FUTURE_CORRECTION_MASK_KEY],
                                    prefix_weights=model_inputs[PREFIX_DRIFT_WEIGHTS_KEY],
                                )
                                eos_future_loss = agg_loss(
                                    loss_mat=eos_future_output.loss_matrix,
                                    loss_mask=eos_future_loss_mask,
                                    loss_agg_mode=loss_agg_mode,
                                )
                                correction_mask = (
                                    eos_future_output.correction_mask
                                    & eos_future_loss_mask.bool()
                                )
                                correction_denom = correction_mask.sum().clamp_min(1)

                                def eos_future_mean(values):
                                    return (
                                        values.float().masked_fill(~correction_mask, 0.0).sum()
                                        / correction_denom
                                    ).detach().item()

                                micro_batch_metrics["opd_eos_future/fixed_value_mean"] = (
                                    eos_future_mean(eos_future_output.fixed_future_value)
                                )
                                micro_batch_metrics["opd_eos_future/weighted_value_mean"] = (
                                    eos_future_mean(eos_future_output.weighted_future_value)
                                )
                                micro_batch_metrics["opd_eos_future/weighted_value_abs_mean"] = (
                                    eos_future_mean(eos_future_output.weighted_future_value.abs())
                                )
                                micro_batch_metrics["opd_eos_future/fixed_value_negative_fraction"] = (
                                    eos_future_mean(
                                        (eos_future_output.fixed_future_value < 0.0).float()
                                    )
                                )
                            else:
                                if eos_future_teacher_log_probs is None:
                                    raise ValueError(
                                        "dynamic H=1 EOS correction requires current teacher candidate log-probabilities."
                                    )
                                eos_future_cuda_start = None
                                eos_future_cuda_end = None
                                if current_eos_log_probs.is_cuda:
                                    eos_future_cuda_start = torch.cuda.Event(enable_timing=True)
                                    eos_future_cuda_end = torch.cuda.Event(enable_timing=True)
                                    eos_future_cuda_start.record()
                                eos_future_prepare_start = time.perf_counter()
                                dynamic_eos_future_output = (
                                    prepare_dynamic_h1_offpolicy_eos_future_loss(
                                        current_eos_log_probs=current_eos_log_probs,
                                        current_action_log_probs=sampled_log_prob_for_loss,
                                        behavior_action_log_probs=model_inputs[
                                            OPD_ROLLOUT_REFERENCE_LOG_PROBS_KEY
                                        ],
                                        current_candidate_log_probs=log_prob_for_loss,
                                        teacher_candidate_log_probs=eos_future_teacher_log_probs,
                                        responses=model_inputs["responses"],
                                        response_mask=response_mask,
                                        eos_token_id=int(data.meta_info["eos_token_id"]),
                                        prefix_weights=model_inputs[PREFIX_DRIFT_WEIGHTS_KEY],
                                        remaining_horizon_relu=(
                                            eos_future_mode
                                            == EOS_FUTURE_MODE_DYNAMIC_H1_REMAINING_HORIZON_RELU
                                        ),
                                        sqrt_remaining_horizon_relu=(
                                            eos_future_mode
                                            == EOS_FUTURE_MODE_DYNAMIC_H1_SQRT_REMAINING_HORIZON_RELU
                                        ),
                                    )
                                )
                                eos_future_prepare_dispatch_ms = (
                                    time.perf_counter() - eos_future_prepare_start
                                ) * 1000.0
                                if eos_future_cuda_end is not None:
                                    eos_future_cuda_end.record()
                                    eos_future_cuda_end.synchronize()
                                    eos_future_prepare_cuda_ms = eos_future_cuda_start.elapsed_time(
                                        eos_future_cuda_end
                                    )
                                else:
                                    eos_future_prepare_cuda_ms = eos_future_prepare_dispatch_ms
                                eos_future_loss = agg_loss(
                                    loss_mat=dynamic_eos_future_output.loss_matrix,
                                    loss_mask=eos_future_loss_mask,
                                    loss_agg_mode=loss_agg_mode,
                                )
                                correction_mask = (
                                    dynamic_eos_future_output.correction_mask
                                    & eos_future_loss_mask.bool()
                                )
                                correction_count = correction_mask.sum()
                                correction_denom = correction_count.clamp_min(1)

                                def eos_future_mean(values):
                                    return (
                                        values.float().masked_fill(~correction_mask, 0.0).sum()
                                        / correction_denom
                                    ).detach().item()

                                def eos_future_max(values):
                                    if not correction_mask.any():
                                        return 0.0
                                    return values.float()[correction_mask].max().detach().item()

                                action_ratio = dynamic_eos_future_output.action_ratio
                                action_ratio_sum = action_ratio.masked_fill(~correction_mask, 0.0).sum()
                                action_ratio_square_sum = action_ratio.square().masked_fill(
                                    ~correction_mask, 0.0
                                ).sum()
                                action_ratio_ess_fraction = (
                                    action_ratio_sum.square()
                                    / (
                                        action_ratio_square_sum
                                        * correction_count.clamp_min(1).to(action_ratio.dtype)
                                    ).clamp_min(torch.finfo(action_ratio.dtype).tiny)
                                )
                                if correction_mask.any():
                                    action_ratio_p95 = torch.quantile(
                                        action_ratio[correction_mask].float(), 0.95
                                    ).detach().item()
                                else:
                                    action_ratio_p95 = 0.0
                                raw_weight = dynamic_eos_future_output.raw_weighted_value
                                remaining_horizon = dynamic_eos_future_output.remaining_horizon
                                horizon_scaled_weight = (
                                    dynamic_eos_future_output.horizon_scaled_weighted_value
                                )
                                clipped_weight = dynamic_eos_future_output.clipped_weighted_value
                                clip_low_mask = (horizon_scaled_weight < 0.0) & correction_mask
                                weight_above_one_mask = (horizon_scaled_weight > 1.0) & correction_mask
                                if eos_future_mode == EOS_FUTURE_MODE_DYNAMIC_H1_OFFPOLICY_CLIPPED:
                                    clip_high_mask = weight_above_one_mask
                                    micro_batch_metrics[
                                        "opd_eos_future/dynamic_h1_offpolicy_clipped"
                                    ] = 1.0
                                elif (
                                    eos_future_mode
                                    == EOS_FUTURE_MODE_DYNAMIC_H1_SQRT_REMAINING_HORIZON_RELU
                                ):
                                    clip_high_mask = torch.zeros_like(correction_mask)
                                    micro_batch_metrics[
                                        "opd_eos_future/dynamic_h1_sqrt_remaining_horizon_relu"
                                    ] = 1.0
                                else:
                                    clip_high_mask = torch.zeros_like(correction_mask)
                                    micro_batch_metrics[
                                        "opd_eos_future/dynamic_h1_remaining_horizon_relu"
                                    ] = 1.0
                                micro_batch_metrics["opd_eos_future/prepare_dispatch_ms"] = (
                                    eos_future_prepare_dispatch_ms
                                )
                                micro_batch_metrics["opd_eos_future/prepare_cuda_ms"] = (
                                    eos_future_prepare_cuda_ms
                                )
                                micro_batch_metrics["opd_eos_future/local_kl_mean"] = (
                                    eos_future_mean(dynamic_eos_future_output.local_kl)
                                )
                                micro_batch_metrics["opd_eos_future/future_value_mean"] = (
                                    eos_future_mean(dynamic_eos_future_output.future_value)
                                )
                                micro_batch_metrics["opd_eos_future/future_value_negative_fraction"] = (
                                    eos_future_mean(
                                        (dynamic_eos_future_output.future_value < 0.0).float()
                                    )
                                )
                                micro_batch_metrics["opd_eos_future/action_log_ratio_abs_mean"] = (
                                    eos_future_mean(dynamic_eos_future_output.action_log_ratio.abs())
                                )
                                micro_batch_metrics["opd_eos_future/action_ratio_mean"] = (
                                    eos_future_mean(action_ratio)
                                )
                                micro_batch_metrics["opd_eos_future/action_ratio_p95"] = action_ratio_p95
                                micro_batch_metrics["opd_eos_future/action_ratio_max"] = (
                                    eos_future_max(action_ratio)
                                )
                                micro_batch_metrics["opd_eos_future/action_ratio_ess_fraction"] = (
                                    action_ratio_ess_fraction.detach().item()
                                )
                                micro_batch_metrics["opd_eos_future/raw_weight_mean"] = (
                                    eos_future_mean(raw_weight)
                                )
                                micro_batch_metrics["opd_eos_future/raw_weight_abs_mean"] = (
                                    eos_future_mean(raw_weight.abs())
                                )
                                micro_batch_metrics["opd_eos_future/raw_weight_max"] = (
                                    eos_future_max(raw_weight)
                                )
                                micro_batch_metrics["opd_eos_future/remaining_horizon_mean"] = (
                                    eos_future_mean(remaining_horizon)
                                )
                                micro_batch_metrics["opd_eos_future/remaining_horizon_max"] = (
                                    eos_future_max(remaining_horizon)
                                )
                                micro_batch_metrics[
                                    "opd_eos_future/horizon_scaled_weight_mean"
                                ] = eos_future_mean(horizon_scaled_weight)
                                micro_batch_metrics[
                                    "opd_eos_future/horizon_scaled_weight_abs_mean"
                                ] = eos_future_mean(horizon_scaled_weight.abs())
                                micro_batch_metrics[
                                    "opd_eos_future/horizon_scaled_weight_max"
                                ] = eos_future_max(horizon_scaled_weight)
                                micro_batch_metrics["opd_eos_future/clip_low_count"] = (
                                    clip_low_mask.sum().detach().item()
                                )
                                micro_batch_metrics["opd_eos_future/clip_low_fraction"] = (
                                    clip_low_mask.sum() / correction_denom
                                ).detach().item()
                                micro_batch_metrics["opd_eos_future/clip_high_count"] = (
                                    clip_high_mask.sum().detach().item()
                                )
                                micro_batch_metrics["opd_eos_future/clip_high_fraction"] = (
                                    clip_high_mask.sum() / correction_denom
                                ).detach().item()
                                micro_batch_metrics["opd_eos_future/weight_above_one_count"] = (
                                    weight_above_one_mask.sum().detach().item()
                                )
                                micro_batch_metrics["opd_eos_future/weight_above_one_fraction"] = (
                                    weight_above_one_mask.sum() / correction_denom
                                ).detach().item()
                                micro_batch_metrics["opd_eos_future/clipped_weight_mean"] = (
                                    eos_future_mean(clipped_weight)
                                )
                                micro_batch_metrics["opd_eos_future/clipped_weight_max"] = (
                                    eos_future_max(clipped_weight)
                                )
                                if correction_mask.any():
                                    effective_weights = clipped_weight[correction_mask].float()
                                    micro_batch_metrics["opd_eos_future/effective_weight_p95"] = (
                                        torch.quantile(effective_weights, 0.95).detach().item()
                                    )
                                    micro_batch_metrics["opd_eos_future/effective_weight_p99"] = (
                                        torch.quantile(effective_weights, 0.99).detach().item()
                                    )
                                else:
                                    micro_batch_metrics["opd_eos_future/effective_weight_p95"] = 0.0
                                    micro_batch_metrics["opd_eos_future/effective_weight_p99"] = 0.0
                                eos_gradient_proxy = (
                                    clipped_weight * current_eos_log_probs.float().exp()
                                )
                                micro_batch_metrics[
                                    "opd_eos_future/eos_gradient_proxy_mean"
                                ] = eos_future_mean(eos_gradient_proxy)
                                micro_batch_metrics[
                                    "opd_eos_future/eos_gradient_proxy_max"
                                ] = eos_future_max(eos_gradient_proxy)

                            pg_loss = pg_loss + eos_future_coef * eos_future_loss
                            micro_batch_metrics["opd_eos_future/enabled"] = 1.0
                            micro_batch_metrics["opd_eos_future/horizon"] = float(
                                eos_future_horizon
                            )
                            micro_batch_metrics["opd_eos_future/coef"] = eos_future_coef
                            micro_batch_metrics["opd_eos_future/correction_position_count"] = (
                                correction_mask.sum().detach().item()
                            )
                            micro_batch_metrics["opd_eos_future/current_eos_probability_mean"] = (
                                eos_future_mean(current_eos_log_probs.float().exp())
                            )
                            micro_batch_metrics["opd_eos_future/loss"] = (
                                eos_future_loss.detach().item()
                            )
                            micro_batch_metrics["opd_eos_future/scaled_loss"] = (
                                (eos_future_coef * eos_future_loss).detach().item()
                            )
                        if teacher_deficit_residual_enable:
                            if teacher_residual_current_log_probs is None:
                                raise ValueError(
                                    "teacher deficit residual requires differentiable student log-probabilities "
                                    "on teacher-sampled candidates."
                                )
                            residual_prefix_weights = model_inputs.get(
                                PREFIX_DRIFT_WEIGHTS_KEY,
                                None,
                            )
                            residual_prepare_start = time.perf_counter()
                            teacher_residual_output = prepare_teacher_deficit_residual_loss(
                                current_log_probs=teacher_residual_current_log_probs,
                                teacher_log_probs=model_inputs[
                                    TEACHER_DEFICIT_RESIDUAL_LOG_PROBS_KEY
                                ],
                                prefix_weights=residual_prefix_weights,
                            )
                            residual_prepare_dispatch_ms = (
                                time.perf_counter() - residual_prepare_start
                            ) * 1000.0
                            residual_loss_mask = response_mask
                            if format_mask is not None:
                                residual_loss_mask = residual_loss_mask * format_mask.unsqueeze(-1)
                            teacher_residual_loss = agg_loss(
                                loss_mat=teacher_residual_output.loss_matrix,
                                loss_mask=residual_loss_mask,
                                loss_agg_mode=loss_agg_mode,
                            )
                            pg_loss = pg_loss + teacher_deficit_residual_coef * teacher_residual_loss

                            residual_candidate_mask = residual_loss_mask.bool().unsqueeze(-1).expand_as(
                                teacher_residual_output.weights
                            )
                            residual_candidate_count = residual_candidate_mask.sum().clamp_min(1)
                            residual_weight_sum = teacher_residual_output.weights.masked_fill(
                                ~residual_candidate_mask,
                                0.0,
                            ).sum()
                            residual_active_count = (
                                teacher_residual_output.active_mask & residual_candidate_mask
                            ).sum()
                            micro_batch_metrics["opd_teacher_deficit_residual/enabled"] = 1.0
                            micro_batch_metrics["opd_teacher_deficit_residual/coef"] = (
                                teacher_deficit_residual_coef
                            )
                            micro_batch_metrics["opd_teacher_deficit_residual/sample_count"] = float(
                                teacher_deficit_residual_k
                            )
                            micro_batch_metrics["opd_teacher_deficit_residual/prefix_weighted"] = float(
                                residual_prefix_weights is not None
                            )
                            micro_batch_metrics["opd_teacher_deficit_residual/prepare_dispatch_ms"] = (
                                residual_prepare_dispatch_ms
                            )
                            micro_batch_metrics["opd_teacher_deficit_residual/loss"] = (
                                teacher_residual_loss.detach().item()
                            )
                            micro_batch_metrics["opd_teacher_deficit_residual/scaled_loss"] = (
                                teacher_deficit_residual_coef * teacher_residual_loss
                            ).detach().item()
                            micro_batch_metrics["opd_teacher_deficit_residual/weight_mean"] = (
                                residual_weight_sum / residual_candidate_count
                            ).detach().item()
                            micro_batch_metrics["opd_teacher_deficit_residual/active_fraction"] = (
                                residual_active_count / residual_candidate_count
                            ).detach().item()
                        teacher_pg_loss = pg_loss

                        if (
                            opd_sampled_token_proximal_coef > 0.0
                            or opd_sampled_token_max_entropy_coef > 0.0
                        ):
                            if opd_advantage_mode not in {"current_kl_is", "current_kl"} or advantages.dim() != 3:
                                raise ValueError(
                                    "sampled-token RKL/max entropy currently requires "
                                    "sample-k current_kl_is with 3D candidate advantages."
                                )
                            if OPD_ROLLOUT_REFERENCE_LOG_PROBS_KEY not in model_inputs:
                                raise ValueError(
                                    "sampled-token RKL/max entropy requires the fixed rollout-policy "
                                    f"log-probs in {OPD_ROLLOUT_REFERENCE_LOG_PROBS_KEY!r}."
                                )
                            proximal_prefix_weights = None
                            if opd_sampled_token_proximal_prefix_weight_enable:
                                if PREFIX_DRIFT_WEIGHTS_KEY not in model_inputs:
                                    raise ValueError(
                                        "geometric sampled-token proximal RKL requires prefix_drift_weights. "
                                        "Enable prefix_drift and select the intended prefix weighting method."
                                    )
                                proximal_prefix_weights = model_inputs[PREFIX_DRIFT_WEIGHTS_KEY]
                            proximal_mask = response_mask
                            if opd_q_prefix_samplek and "opd_proximal_mask" in model_inputs:
                                proximal_mask = model_inputs["opd_proximal_mask"].to(
                                    device=response_mask.device,
                                    dtype=response_mask.dtype,
                                )
                            if format_mask is not None:
                                proximal_mask = proximal_mask * format_mask.unsqueeze(-1)
                            proximal_output = compute_sampled_token_proximal_rkl_loss(
                                current_log_probs=sampled_log_prob_for_loss,
                                reference_log_probs=model_inputs[OPD_ROLLOUT_REFERENCE_LOG_PROBS_KEY],
                                response_mask=proximal_mask,
                                coefficient=opd_sampled_token_proximal_coef,
                                loss_agg_mode=loss_agg_mode,
                                prefix_weights=proximal_prefix_weights,
                                max_entropy_coefficient=opd_sampled_token_max_entropy_coef,
                                mode=opd_sampled_token_proximal_mode,
                            )
                            pg_loss = teacher_pg_loss + proximal_output.loss
                            for key, value in proximal_output.metrics.items():
                                micro_batch_metrics[f"opd_samplek/proximal_{key}"] = value
                            teacher_signal_abs_mean = verl_F.masked_mean(
                                advantages.abs().sum(dim=-1),
                                proximal_mask,
                            ).detach()
                            proximal_signal_abs_mean = sampled_log_prob_for_loss.new_tensor(
                                proximal_output.metrics["coefficient_abs_mean"]
                            )
                            entropy_regularizer_signal_abs_mean = sampled_log_prob_for_loss.new_tensor(
                                proximal_output.metrics["entropy_regularizer_coefficient_abs_mean"]
                            )
                            micro_batch_metrics["opd_samplek/teacher_loss"] = teacher_pg_loss.detach().item()
                            micro_batch_metrics["opd_samplek/total_loss"] = pg_loss.detach().item()
                            micro_batch_metrics["opd_samplek/teacher_signal_abs_mean"] = (
                                teacher_signal_abs_mean.item()
                            )
                            micro_batch_metrics["opd_samplek/proximal_signal_abs_mean"] = (
                                proximal_signal_abs_mean.item()
                            )
                            micro_batch_metrics["opd_samplek/entropy_regularizer_signal_abs_mean"] = (
                                entropy_regularizer_signal_abs_mean.item()
                            )
                            micro_batch_metrics["opd_samplek/teacher_to_prox_signal_abs_ratio"] = (
                                teacher_signal_abs_mean / proximal_signal_abs_mean.clamp_min(1e-6)
                            ).item()
                            micro_batch_metrics["opd_samplek/teacher_to_entropy_regularizer_signal_abs_ratio"] = (
                                teacher_signal_abs_mean
                                / entropy_regularizer_signal_abs_mean.clamp_min(1e-6)
                            ).item()
                            micro_batch_metrics["opd_samplek/current_token_candidate_is_applied"] = float(
                                not opd_current_samplek_no_candidate_is
                            )

                    if opd_dapo_format_penalty_enable:
                        if dapo_answer_format_mask is None:
                            raise ValueError(
                                "opd_dapo_format_penalty_enable=True requires "
                                f"{DAPO_ANSWER_FORMAT_MASK_KEY!r} in the batch. "
                                "Make sure dapo_answer_format_check_enable is set in batch.meta_info."
                            )
                        if sampled_log_prob_for_loss.dim() != 2:
                            raise ValueError(
                                "DAPO answer format penalty expects sampled response log-probs with shape "
                                f"(batch, seq), got {sampled_log_prob_for_loss.shape}."
                            )
                        dapo_answer_format_mask = dapo_answer_format_mask.to(
                            device=response_mask.device,
                            dtype=response_mask.dtype,
                        )
                        if dapo_answer_format_mask.dim() != 1 or dapo_answer_format_mask.shape[0] != response_mask.shape[0]:
                            raise ValueError(
                                "dapo_answer_format_mask must have shape (batch,), got "
                                f"{dapo_answer_format_mask.shape} for response_mask={response_mask.shape}."
                            )
                        valid_lengths = response_mask.float().sum(dim=-1)
                        invalid_response_mask = (1.0 - dapo_answer_format_mask).clamp(min=0.0, max=1.0)
                        tail_mask = self._make_tail_mask(
                            response_mask=response_mask,
                            tail_tokens=opd_dapo_format_penalty_tail_tokens,
                        )
                        penalty_token_mask = tail_mask * invalid_response_mask.unsqueeze(-1)
                        dapo_format_penalty_loss = opd_dapo_format_penalty_coef * agg_loss(
                            loss_mat=sampled_log_prob_for_loss,
                            loss_mask=penalty_token_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        pg_loss = pg_loss + dapo_format_penalty_loss
                        response_count = dapo_answer_format_mask.numel()
                        response_token_denom = response_mask.float().sum().clamp_min(1.0)
                        invalid_count = invalid_response_mask.sum()
                        valid_count = dapo_answer_format_mask.sum()
                        micro_batch_metrics["opd_dapo_format_penalty/enabled"] = 1.0
                        micro_batch_metrics["opd_dapo_format_penalty/coef"] = opd_dapo_format_penalty_coef
                        micro_batch_metrics["opd_dapo_format_penalty/tail_tokens"] = float(
                            opd_dapo_format_penalty_tail_tokens
                        )
                        micro_batch_metrics["opd_dapo_format_penalty/dapo_answer_ok_rate"] = (
                            valid_count / max(response_count, 1)
                        ).detach().item()
                        micro_batch_metrics["opd_dapo_format_penalty/dapo_answer_missing_rate"] = (
                            invalid_count / max(response_count, 1)
                        ).detach().item()
                        micro_batch_metrics["opd_dapo_format_penalty/token_fraction"] = (
                            penalty_token_mask.float().sum() / response_token_denom
                        ).detach().item()
                        micro_batch_metrics["opd_dapo_format_penalty/loss"] = (
                            dapo_format_penalty_loss.detach().item() * loss_scale_factor
                        )
                        micro_batch_metrics["opd_dapo_format_penalty/valid_response_length_mean"] = (
                            (valid_lengths * dapo_answer_format_mask).sum() / valid_count.clamp_min(1.0)
                        ).detach().item()
                        micro_batch_metrics["opd_dapo_format_penalty/invalid_response_length_mean"] = (
                            (valid_lengths * invalid_response_mask).sum() / invalid_count.clamp_min(1.0)
                        ).detach().item()
                    else:
                        micro_batch_metrics["opd_dapo_format_penalty/enabled"] = 0.0

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm, grad_norm_metrics = self._optimizer_step(opd_samplek_total_grad_norm)
                mini_batch_metrics = {
                    "actor/grad_norm": grad_norm.detach().item(),
                    **grad_norm_metrics,
                }
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
