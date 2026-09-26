from __future__ import annotations

import math
from dataclasses import dataclass

import torch


OPD_EOS_FUTURE_VALUE_KEY = "opd_eos_future_value"
OPD_EOS_FUTURE_CORRECTION_MASK_KEY = "opd_eos_future_correction_mask"
EOS_FUTURE_MODE_FIXED_SIGNED = "fixed_signed"
EOS_FUTURE_MODE_DYNAMIC_H1_OFFPOLICY_CLIPPED = "dynamic_h1_offpolicy_clipped"
EOS_FUTURE_MODE_DYNAMIC_H1_REMAINING_HORIZON_RELU = "dynamic_h1_remaining_horizon_relu"
EOS_FUTURE_MODE_DYNAMIC_H1_SQRT_REMAINING_HORIZON_RELU = (
    "dynamic_h1_sqrt_remaining_horizon_relu"
)


def normalize_eos_future_mode(mode: object) -> str:
    normalized = str(mode or EOS_FUTURE_MODE_FIXED_SIGNED).strip().lower().replace("-", "_")
    supported = {
        EOS_FUTURE_MODE_FIXED_SIGNED,
        EOS_FUTURE_MODE_DYNAMIC_H1_OFFPOLICY_CLIPPED,
        EOS_FUTURE_MODE_DYNAMIC_H1_REMAINING_HORIZON_RELU,
        EOS_FUTURE_MODE_DYNAMIC_H1_SQRT_REMAINING_HORIZON_RELU,
    }
    if normalized not in supported:
        raise ValueError(
            f"Unknown opd_eos_future_mode={mode!r}; expected one of {sorted(supported)}."
        )
    return normalized


@dataclass(frozen=True)
class FixedEOSFutureValues:
    local_kl: torch.Tensor
    future_value: torch.Tensor
    correction_mask: torch.Tensor
    semantic_eos_mask: torch.Tensor


@dataclass(frozen=True)
class EOSFutureCorrectionLoss:
    loss_matrix: torch.Tensor
    log_continue: torch.Tensor
    fixed_future_value: torch.Tensor
    weighted_future_value: torch.Tensor
    correction_mask: torch.Tensor


@dataclass(frozen=True)
class DynamicH1EOSFutureCorrectionLoss:
    loss_matrix: torch.Tensor
    log_continue: torch.Tensor
    local_kl: torch.Tensor
    future_value: torch.Tensor
    action_log_ratio: torch.Tensor
    action_ratio: torch.Tensor
    raw_weighted_value: torch.Tensor
    remaining_horizon: torch.Tensor
    horizon_scaled_weighted_value: torch.Tensor
    clipped_weighted_value: torch.Tensor
    correction_mask: torch.Tensor


def build_fixed_eos_future_values(
    *,
    student_candidate_log_probs: torch.Tensor,
    teacher_candidate_log_probs: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    eos_token_id: int,
    horizon: int,
) -> FixedEOSFutureValues:

    if student_candidate_log_probs.ndim != 3:
        raise ValueError(
            "student_candidate_log_probs must have shape [batch, sequence, K], "
            f"got {student_candidate_log_probs.shape}."
        )
    if teacher_candidate_log_probs.shape != student_candidate_log_probs.shape:
        raise ValueError(
            "teacher_candidate_log_probs must match student candidates, "
            f"got teacher={teacher_candidate_log_probs.shape} and "
            f"student={student_candidate_log_probs.shape}."
        )
    state_shape = student_candidate_log_probs.shape[:2]
    for name, value in (("responses", responses), ("response_mask", response_mask)):
        if value.shape != state_shape:
            raise ValueError(f"{name} must have shape {state_shape}, got {value.shape}.")
    horizon_float = float(horizon)
    if not math.isfinite(horizon_float) or not horizon_float.is_integer() or horizon_float < 1:
        raise ValueError(f"opd_eos_future_horizon must be a positive integer, got {horizon!r}.")
    horizon = int(horizon_float)

    with torch.no_grad():
        student = student_candidate_log_probs.detach().float()
        teacher = teacher_candidate_log_probs.to(device=student.device).detach().float()
        valid_mask = response_mask.to(device=student.device).bool()
        semantic_eos_mask = valid_mask & responses.to(device=student.device).eq(int(eos_token_id))
        local_kl = (student - teacher).mean(dim=-1).masked_fill(~valid_mask, 0.0)
        future_value = torch.zeros_like(local_kl)
        full_window_observed = torch.ones_like(valid_mask)
        semantic_eos_in_window = torch.zeros_like(valid_mask)
        active_before_step = torch.ones_like(valid_mask)
        sequence_length = local_kl.size(-1)

        for offset in range(1, horizon + 1):
            shifted_kl = torch.zeros_like(local_kl)
            shifted_valid = torch.zeros_like(valid_mask)
            shifted_eos = torch.zeros_like(semantic_eos_mask)
            if offset < sequence_length:
                shifted_kl[:, :-offset] = local_kl[:, offset:]
                shifted_valid[:, :-offset] = valid_mask[:, offset:]
                shifted_eos[:, :-offset] = semantic_eos_mask[:, offset:]
            future_value += shifted_kl * (active_before_step & shifted_valid)
            full_window_observed &= shifted_valid
            semantic_eos_in_window |= shifted_eos
            active_before_step &= ~shifted_eos

        continued_mask = valid_mask & ~semantic_eos_mask
        correction_mask = continued_mask & (full_window_observed | semantic_eos_in_window)

    return FixedEOSFutureValues(
        local_kl=local_kl,
        future_value=future_value,
        correction_mask=correction_mask,
        semantic_eos_mask=semantic_eos_mask,
    )


def _log_one_minus_exp(log_probability: torch.Tensor) -> torch.Tensor:
    log_probability = log_probability.float()
    eps = torch.finfo(log_probability.dtype).eps
    log_probability = log_probability.clamp_max(-eps)
    cutoff = -math.log(2.0)
    return torch.where(
        log_probability < cutoff,
        torch.log1p(-log_probability.exp()),
        torch.log(-torch.expm1(log_probability)),
    )


def prepare_eos_future_correction_loss(
    *,
    current_eos_log_probs: torch.Tensor,
    fixed_future_value: torch.Tensor,
    correction_mask: torch.Tensor,
    prefix_weights: torch.Tensor | None,
) -> EOSFutureCorrectionLoss:

    state_shape = current_eos_log_probs.shape
    if current_eos_log_probs.ndim != 2:
        raise ValueError(
            f"current_eos_log_probs must have shape [batch, sequence], got {state_shape}."
        )
    for name, value in (
        ("fixed_future_value", fixed_future_value),
        ("correction_mask", correction_mask),
    ):
        if value.shape != state_shape:
            raise ValueError(f"{name} must have shape {state_shape}, got {value.shape}.")
    if prefix_weights is not None and prefix_weights.shape != state_shape:
        raise ValueError(f"prefix_weights must have shape {state_shape}, got {prefix_weights.shape}.")

    fixed_future_value = fixed_future_value.to(device=current_eos_log_probs.device).detach().float()
    correction_mask = correction_mask.to(device=current_eos_log_probs.device).detach().bool()
    if prefix_weights is None:
        detached_prefix_weights = torch.ones_like(fixed_future_value)
    else:
        detached_prefix_weights = prefix_weights.to(
            device=current_eos_log_probs.device
        ).detach().float()
    weighted_future_value = fixed_future_value * detached_prefix_weights
    log_continue = _log_one_minus_exp(current_eos_log_probs)
    loss_matrix = weighted_future_value * log_continue * correction_mask.to(log_continue.dtype)

    return EOSFutureCorrectionLoss(
        loss_matrix=loss_matrix,
        log_continue=log_continue,
        fixed_future_value=fixed_future_value,
        weighted_future_value=weighted_future_value,
        correction_mask=correction_mask,
    )


def prepare_dynamic_h1_offpolicy_eos_future_loss(
    *,
    current_eos_log_probs: torch.Tensor,
    current_action_log_probs: torch.Tensor,
    behavior_action_log_probs: torch.Tensor,
    current_candidate_log_probs: torch.Tensor,
    teacher_candidate_log_probs: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    eos_token_id: int,
    prefix_weights: torch.Tensor,
    remaining_horizon_relu: bool = False,
    sqrt_remaining_horizon_relu: bool = False,
) -> DynamicH1EOSFutureCorrectionLoss:

    state_shape = current_eos_log_probs.shape
    if current_eos_log_probs.ndim != 2:
        raise ValueError(
            f"current_eos_log_probs must have shape [batch, sequence], got {state_shape}."
        )
    for name, value in (
        ("current_action_log_probs", current_action_log_probs),
        ("behavior_action_log_probs", behavior_action_log_probs),
        ("responses", responses),
        ("response_mask", response_mask),
        ("prefix_weights", prefix_weights),
    ):
        if value.shape != state_shape:
            raise ValueError(f"{name} must have shape {state_shape}, got {value.shape}.")
    if current_candidate_log_probs.ndim != 3 or current_candidate_log_probs.shape[:2] != state_shape:
        raise ValueError(
            "current_candidate_log_probs must have shape [batch, sequence, K] matching "
            f"{state_shape}, got {current_candidate_log_probs.shape}."
        )
    if teacher_candidate_log_probs.shape != current_candidate_log_probs.shape:
        raise ValueError(
            "teacher_candidate_log_probs must match current candidates, "
            f"got teacher={teacher_candidate_log_probs.shape} and "
            f"student={current_candidate_log_probs.shape}."
        )

    if remaining_horizon_relu and sqrt_remaining_horizon_relu:
        raise ValueError("Linear and sqrt remaining-horizon scaling are mutually exclusive.")

    device = current_eos_log_probs.device
    with torch.no_grad():
        valid_mask = response_mask.to(device=device).bool()
        responses = responses.to(device=device)
        semantic_eos_mask = valid_mask & responses.eq(int(eos_token_id))
        current_candidates = current_candidate_log_probs.detach().to(device=device).float()
        teacher_candidates = teacher_candidate_log_probs.detach().to(device=device).float()
        local_kl = (current_candidates - teacher_candidates).mean(dim=-1)
        local_kl = local_kl.masked_fill(~valid_mask, 0.0)

        future_value = torch.zeros_like(local_kl)
        next_valid = torch.zeros_like(valid_mask)
        if local_kl.size(-1) > 1:
            future_value[:, :-1] = local_kl[:, 1:]
            next_valid[:, :-1] = valid_mask[:, 1:]
        correction_mask = valid_mask & ~semantic_eos_mask & next_valid

        action_log_ratio = (
            current_action_log_probs.detach().to(device=device).float()
            - behavior_action_log_probs.detach().to(device=device).float()
        )
        action_ratio = action_log_ratio.clamp(min=-80.0, max=80.0).exp()
        detached_prefix_weights = prefix_weights.detach().to(device=device).float()
        raw_weighted_value = future_value * action_ratio * detached_prefix_weights
        raw_weighted_value = raw_weighted_value.masked_fill(~correction_mask, 0.0)
        remaining_horizon = torch.ones_like(raw_weighted_value)
        horizon_scaled_weighted_value = raw_weighted_value
        if remaining_horizon_relu or sqrt_remaining_horizon_relu:
            remaining_horizon = valid_mask.flip(-1).cumsum(-1).flip(-1).float()
            remaining_horizon = (remaining_horizon - valid_mask.float()).clamp_min(0.0)
            horizon_multiplier = (
                remaining_horizon.sqrt()
                if sqrt_remaining_horizon_relu
                else remaining_horizon
            )
            horizon_scaled_weighted_value = raw_weighted_value * horizon_multiplier
            clipped_weighted_value = horizon_scaled_weighted_value.clamp_min(0.0)
        else:
            clipped_weighted_value = raw_weighted_value.clamp(min=0.0, max=1.0)

    log_continue = _log_one_minus_exp(current_eos_log_probs)
    loss_matrix = clipped_weighted_value * log_continue

    return DynamicH1EOSFutureCorrectionLoss(
        loss_matrix=loss_matrix,
        log_continue=log_continue,
        local_kl=local_kl,
        future_value=future_value,
        action_log_ratio=action_log_ratio,
        action_ratio=action_ratio,
        raw_weighted_value=raw_weighted_value,
        remaining_horizon=remaining_horizon,
        horizon_scaled_weighted_value=horizon_scaled_weighted_value,
        clipped_weighted_value=clipped_weighted_value,
        correction_mask=correction_mask,
    )


def validate_eos_future_correction_configuration(
    *,
    enabled: bool,
    mode: object,
    horizon: int,
    coefficient: float,
    candidate_mode: object,
    sample_replacement: bool,
    top_k: int,
    opd_loss_type: object,
    advantage_mode: object,
    candidate_aggregation: object,
    advantage_centering: object,
    adaptive_update_enable: bool,
    no_candidate_is: bool,
    sample_k_kl_plus_one: bool,
    prefix_drift_enable: bool,
    forced_eos_diagnostic_enable: bool,
    terminal_aware_enable: bool,
    raw_advantage_clip: float | None,
    influence_clip: float | None,
    terminal_objective_mode: object = "anchor_kl",
    terminal_teacher_remap_enable: bool = False,
) -> None:

    if not enabled:
        return

    normalized_mode = normalize_eos_future_mode(mode)
    horizon_float = float(horizon)
    if not math.isfinite(horizon_float) or not horizon_float.is_integer() or horizon_float < 1:
        raise ValueError("opd_eos_future_horizon must be a positive integer.")
    if normalized_mode in {
        EOS_FUTURE_MODE_DYNAMIC_H1_OFFPOLICY_CLIPPED,
        EOS_FUTURE_MODE_DYNAMIC_H1_REMAINING_HORIZON_RELU,
        EOS_FUTURE_MODE_DYNAMIC_H1_SQRT_REMAINING_HORIZON_RELU,
    } and int(horizon_float) != 1:
        raise ValueError(f"{normalized_mode} requires opd_eos_future_horizon=1.")
    coefficient = float(coefficient)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("opd_eos_future_coef must be finite and nonnegative.")

    def normalized(value: object) -> str:
        return str(value).strip().lower().replace("-", "_")

    if normalized(candidate_mode) != "sample_stu":
        raise ValueError("EOS future correction requires candidate_mode=sample_stu.")
    if not sample_replacement:
        raise ValueError("EOS future correction requires sampling with replacement.")
    if int(top_k) < 2:
        raise ValueError("EOS future correction requires at least 2 Sample-K candidates.")
    if normalized(opd_loss_type) != "sample_k_reverse_kl":
        raise ValueError("EOS future correction requires the sample-k reverse_kl loss.")
    if normalized(advantage_mode) != "current_kl_is":
        raise ValueError("EOS future correction requires advantage_mode=current_kl_is.")
    if normalized(candidate_aggregation) != "mean":
        raise ValueError("EOS future correction requires candidate aggregation=mean.")
    if normalized(advantage_centering) != "none":
        raise ValueError("EOS future correction requires advantage centering=none.")
    if not adaptive_update_enable:
        raise ValueError("EOS future correction requires adaptive resampling updates.")
    if not no_candidate_is:
        raise ValueError("EOS future correction requires candidate IS to be disabled.")
    if sample_k_kl_plus_one:
        raise ValueError("EOS future correction requires sample_k_kl_plus_one=False.")
    if not prefix_drift_enable:
        raise ValueError("EOS future correction requires prefix drift weights.")
    if forced_eos_diagnostic_enable:
        raise ValueError("EOS future correction is mutually exclusive with forced-EOS sampling.")
    if terminal_aware_enable:
        if normalized(terminal_objective_mode) != "teacher_remap_only":
            raise ValueError(
                "EOS future correction only supports the terminal-aware teacher_remap_only objective."
            )
        if not terminal_teacher_remap_enable:
            raise ValueError("EOS future correction with teacher_remap_only requires teacher EOS remapping.")
    if raw_advantage_clip is not None:
        raise ValueError("EOS future correction does not support raw advantage clip.")
    if influence_clip is not None:
        raise ValueError("EOS future correction does not support influence clip.")
