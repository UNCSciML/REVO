from __future__ import annotations

import math
from dataclasses import dataclass

import torch


OPD_TERMINAL_BEHAVIOR_EOS_LOG_PROBS_KEY = "opd_terminal_behavior_eos_log_probs"
OPD_TERMINAL_BEHAVIOR_SECONDARY_LOG_PROBS_KEY = "opd_terminal_behavior_secondary_log_probs"
OPD_TERMINAL_STUDENT_TOPM_IDS_KEY = "opd_terminal_student_topm_ids"
OPD_TERMINAL_STUDENT_TOPM_LOG_PROBS_KEY = "opd_terminal_student_topm_log_probs"
OPD_TERMINAL_STUDENT_SECONDARY_LOG_PROBS_KEY = "opd_terminal_student_secondary_log_probs"
OPD_TERMINAL_TEACHER_TOPM_LOG_PROBS_KEY = "opd_terminal_teacher_topm_log_probs"
OPD_TERMINAL_TEACHER_SECONDARY_LOG_PROBS_KEY = "opd_terminal_teacher_secondary_log_probs"


@dataclass(frozen=True)
class TerminalAwareObjective:
    policy_log_probs: torch.Tensor
    advantages: torch.Tensor
    gate_loss: torch.Tensor
    terminal_mask: torch.Tensor
    behavior_eos_probability: torch.Tensor
    current_eos_probability: torch.Tensor
    teacher_eos_probability: torch.Tensor
    behavior_secondary_probability: torch.Tensor
    current_secondary_probability: torch.Tensor
    teacher_secondary_probability: torch.Tensor
    teacher_stop_probability: torch.Tensor
    anchor_probability: torch.Tensor
    content_mass: torch.Tensor
    current_content_probability: torch.Tensor
    teacher_floor_active: torch.Tensor


@dataclass(frozen=True)
class TerminalSafeContinueObjective:
    policy_log_probs: torch.Tensor
    advantages: torch.Tensor
    gate_loss: torch.Tensor
    terminal_mask: torch.Tensor
    current_eos_probability: torch.Tensor
    current_secondary_probability: torch.Tensor
    teacher_eos_probability: torch.Tensor
    teacher_secondary_probability: torch.Tensor
    teacher_stop_probability: torch.Tensor
    current_content_probability: torch.Tensor
    stop_signal: torch.Tensor
    continuation_signal: torch.Tensor
    safe_gate_advantage: torch.Tensor
    gate_active: torch.Tensor


@dataclass(frozen=True)
class TerminalConservativeKLObjective:
    policy_log_probs: torch.Tensor
    advantages: torch.Tensor
    remapped_teacher_log_probs: torch.Tensor
    terminal_mask: torch.Tensor
    kl_baseline: torch.Tensor
    eos_advantage: torch.Tensor
    eos_gate_active: torch.Tensor
    current_eos_probability: torch.Tensor
    teacher_eos_probability: torch.Tensor
    teacher_secondary_probability: torch.Tensor
    teacher_remapped_eos_probability: torch.Tensor
    teacher_remapped_secondary_probability: torch.Tensor


@dataclass(frozen=True)
class TerminalTeacherRemapOnlyObjective:
    remapped_teacher_log_probs: torch.Tensor
    terminal_mask: torch.Tensor
    eos_candidate_mask: torch.Tensor
    secondary_candidate_mask: torch.Tensor
    teacher_secondary_probability: torch.Tensor
    teacher_stop_probability: torch.Tensor


@dataclass(frozen=True)
class TerminalTeacherTrajectoryRemap:
    remapped_teacher_log_probs: torch.Tensor
    terminal_mask: torch.Tensor
    teacher_secondary_probability: torch.Tensor
    teacher_stop_probability: torch.Tensor


@dataclass(frozen=True)
class SampleKEOSNegativeReLUOutput:
    advantages: torch.Tensor
    eos_candidate_mask: torch.Tensor
    clipped_mask: torch.Tensor
    metrics: dict[str, float]


def apply_samplek_eos_negative_relu(
    *,
    advantages: torch.Tensor,
    candidate_ids: torch.Tensor,
    response_mask: torch.Tensor,
    eos_token_id: int,
) -> SampleKEOSNegativeReLUOutput:

    if advantages.ndim != 3:
        raise ValueError(
            "Sample-K EOS negative ReLU requires advantages with shape [batch, sequence, K], "
            f"got {tuple(advantages.shape)}."
        )
    if candidate_ids.shape != advantages.shape:
        raise ValueError(
            "Sample-K EOS negative ReLU candidate_ids must match advantages, "
            f"got ids={tuple(candidate_ids.shape)} and advantages={tuple(advantages.shape)}."
        )
    if response_mask.shape != advantages.shape[:2]:
        raise ValueError(
            "Sample-K EOS negative ReLU response_mask must match the first two advantage dimensions, "
            f"got mask={tuple(response_mask.shape)} and advantages={tuple(advantages.shape)}."
        )

    valid_candidate_mask = response_mask.to(device=advantages.device).bool().unsqueeze(-1)
    eos_candidate_mask = valid_candidate_mask & candidate_ids.to(device=advantages.device).eq(
        int(eos_token_id)
    )
    clipped_mask = eos_candidate_mask & advantages.lt(0.0)
    transformed = torch.where(clipped_mask, torch.zeros_like(advantages), advantages)

    valid_candidate_count = valid_candidate_mask.expand_as(advantages).sum()
    eos_candidate_count = eos_candidate_mask.sum()
    negative_eos_candidate_count = clipped_mask.sum()
    removed_abs_sum = advantages.detach().float().abs().masked_fill(~clipped_mask, 0.0).sum()
    metrics = {
        "sampled_eos_candidate_count": eos_candidate_count.detach().item(),
        "sampled_eos_candidate_fraction": (
            eos_candidate_count / valid_candidate_count.clamp_min(1)
        ).detach().item(),
        "negative_eos_candidate_count": negative_eos_candidate_count.detach().item(),
        "negative_fraction_of_eos": (
            negative_eos_candidate_count / eos_candidate_count.clamp_min(1)
        ).detach().item(),
        "removed_negative_advantage_abs_sum": removed_abs_sum.detach().item(),
        "removed_negative_advantage_abs_mean": (
            removed_abs_sum / negative_eos_candidate_count.clamp_min(1)
        ).detach().item(),
    }
    return SampleKEOSNegativeReLUOutput(
        advantages=transformed,
        eos_candidate_mask=eos_candidate_mask,
        clipped_mask=clipped_mask,
        metrics=metrics,
    )


def normalize_terminal_objective_mode(mode: object) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "anchor": "anchor_kl",
        "kl": "anchor_kl",
        "safe": "safe_continue",
        "continue": "safe_continue",
        "conservative": "conservative_kl",
        "remapped_kl": "conservative_kl",
        "remap_only": "teacher_remap_only",
        "teacher_remap": "teacher_remap_only",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"anchor_kl", "safe_continue", "conservative_kl", "teacher_remap_only"}:
        raise ValueError(
            "opd_terminal_objective_mode must be anchor_kl, safe_continue, conservative_kl, "
            "or teacher_remap_only, "
            f"got {mode!r}."
        )
    return normalized


def select_terminal_topk_candidate_estimator_weights(
    *,
    candidate_weights: torch.Tensor,
    terminal_aware_enable: bool,
    objective_mode: object,
    candidate_mode: object,
    teacher_remap_enable: bool,
) -> torch.Tensor | None:

    if not terminal_aware_enable or not teacher_remap_enable:
        return None
    if normalize_terminal_objective_mode(objective_mode) != "teacher_remap_only":
        return None
    normalized_candidate_mode = str(candidate_mode).strip().lower().replace("-", "_")
    if normalized_candidate_mode != "topk":
        return None
    return candidate_weights.detach()


def _normalize_anchor_mode(mode: object) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "po": "behavior",
        "p_o": "behavior",
        "max": "teacher_floor",
        "max_teacher_behavior": "teacher_floor",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"behavior", "teacher_floor"}:
        raise ValueError(
            "opd_terminal_anchor_mode must be behavior or teacher_floor, "
            f"got {mode!r}."
        )
    return normalized


def _log_one_minus_probability(log_probability: torch.Tensor) -> torch.Tensor:
    probability = log_probability.exp()
    eps = torch.finfo(probability.dtype).eps
    return torch.log1p(-probability.clamp(min=0.0, max=1.0 - eps))


def _log_one_minus_probability_sum(*log_probabilities: torch.Tensor) -> torch.Tensor:
    probability = sum(log_probability.exp() for log_probability in log_probabilities)
    eps = torch.finfo(probability.dtype).eps
    return torch.log1p(-probability.clamp(min=0.0, max=1.0 - eps))


def _normalize_terminal_kl_baseline_mode(mode: object) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "mc": "mc_loo",
        "loo": "mc_loo",
        "topm": "topm_coarse",
        "coarse": "topm_coarse",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"mc_loo", "topm_coarse"}:
        raise ValueError(
            "opd_terminal_kl_baseline_mode must be mc_loo or topm_coarse, "
            f"got {mode!r}."
        )
    return normalized


def _xlogratio(probability: torch.Tensor, reference_probability: torch.Tensor) -> torch.Tensor:
    return torch.xlogy(probability, probability) - torch.xlogy(probability, reference_probability)


def sample_terminal_aware_candidates(
    log_probs_all: torch.Tensor,
    *,
    terminal_mask: torch.Tensor,
    num_candidates: int,
    eos_token_id: int,
    secondary_token_id: int | None = None,
    reserve_secondary_token: bool = True,
) -> torch.Tensor:

    if log_probs_all.ndim < 2:
        raise ValueError(f"log_probs_all must include row and vocabulary dimensions, got {log_probs_all.shape}.")
    if terminal_mask.shape != log_probs_all.shape[:-1]:
        raise ValueError(
            "terminal_mask must match the non-vocabulary dimensions of log_probs_all, "
            f"got mask={terminal_mask.shape} and log_probs={log_probs_all.shape}."
        )
    num_candidates = int(num_candidates)
    reserve_secondary_token = bool(reserve_secondary_token)
    exact_token_count = 2 if secondary_token_id is not None and reserve_secondary_token else 1
    minimum_candidates = exact_token_count + 1
    if num_candidates < minimum_candidates:
        raise ValueError(
            f"terminal-aware Sample-K requires at least {minimum_candidates} candidates, got {num_candidates}."
        )
    vocab_size = log_probs_all.size(-1)
    eos_token_id = int(eos_token_id)
    if eos_token_id < 0 or eos_token_id >= vocab_size:
        raise ValueError(f"EOS token id {eos_token_id} is outside vocabulary size {vocab_size}.")
    if secondary_token_id is not None:
        secondary_token_id = int(secondary_token_id)
        if secondary_token_id < 0 or secondary_token_id >= vocab_size:
            raise ValueError(
                f"Secondary token id {secondary_token_id} is outside vocabulary size {vocab_size}."
            )
        if secondary_token_id == eos_token_id:
            raise ValueError("Secondary token id must differ from the student EOS token id.")

    detached = log_probs_all.detach().float().reshape(-1, vocab_size)
    probabilities = detached.exp()
    try:
        candidate_ids = torch.multinomial(probabilities, num_candidates, replacement=True)
    except RuntimeError:
        candidate_ids = torch.multinomial(probabilities.float(), num_candidates, replacement=True)

    flat_terminal_mask = terminal_mask.reshape(-1).bool().to(device=candidate_ids.device)
    if flat_terminal_mask.any():
        conditional_logits = detached[flat_terminal_mask].clone()
        conditional_logits[:, eos_token_id] = -torch.inf
        if secondary_token_id is not None and reserve_secondary_token:
            conditional_logits[:, secondary_token_id] = -torch.inf
        if not torch.isfinite(conditional_logits).any(dim=-1).all():
            raise ValueError("terminal-aware Sample-K requires nonzero non-EOS support at every terminal row.")
        conditional_probabilities = torch.softmax(conditional_logits, dim=-1)
        non_eos_ids = torch.multinomial(
            conditional_probabilities,
            num_samples=num_candidates - exact_token_count,
            replacement=True,
        )
        exact_ids = [torch.full_like(non_eos_ids[:, :1], eos_token_id)]
        if secondary_token_id is not None and reserve_secondary_token:
            exact_ids.append(torch.full_like(non_eos_ids[:, :1], secondary_token_id))
        terminal_ids = torch.cat(
            (*exact_ids, non_eos_ids),
            dim=-1,
        )
        candidate_ids[flat_terminal_mask] = terminal_ids

    return candidate_ids.view(*log_probs_all.shape[:-1], num_candidates)


def sample_terminal_objective_candidates(
    log_probs_all: torch.Tensor,
    *,
    terminal_mask: torch.Tensor,
    num_candidates: int,
    eos_token_id: int,
    objective_mode: object,
    secondary_token_id: int | None = None,
) -> torch.Tensor:

    objective_mode = normalize_terminal_objective_mode(objective_mode)
    if objective_mode != "teacher_remap_only":
        return sample_terminal_aware_candidates(
            log_probs_all,
            terminal_mask=terminal_mask,
            num_candidates=num_candidates,
            eos_token_id=eos_token_id,
            secondary_token_id=secondary_token_id,
            reserve_secondary_token=objective_mode != "conservative_kl",
        )

    if log_probs_all.ndim < 2:
        raise ValueError(f"log_probs_all must include row and vocabulary dimensions, got {log_probs_all.shape}.")
    if terminal_mask.shape != log_probs_all.shape[:-1]:
        raise ValueError(
            "terminal_mask must match the non-vocabulary dimensions of log_probs_all, "
            f"got mask={terminal_mask.shape} and log_probs={log_probs_all.shape}."
        )
    num_candidates = int(num_candidates)
    if num_candidates < 1:
        raise ValueError(f"teacher-remap-only Sample-K requires at least one candidate, got {num_candidates}.")
    probabilities = log_probs_all.detach().float().exp().reshape(-1, log_probs_all.size(-1))
    candidate_ids = torch.multinomial(probabilities, num_candidates, replacement=True)
    return candidate_ids.view(*log_probs_all.shape[:-1], num_candidates)


def prepare_terminal_teacher_remap_only_objective(
    *,
    teacher_log_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    teacher_secondary_log_probs: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    eos_token_id: int,
    secondary_token_id: int,
    teacher_remap_floor: float,
) -> TerminalTeacherRemapOnlyObjective:

    if teacher_log_probs.ndim != 3:
        raise ValueError(
            "teacher-remap-only teacher_log_probs must have shape [batch, sequence, K], "
            f"got {teacher_log_probs.shape}."
        )
    if candidate_ids.shape != teacher_log_probs.shape:
        raise ValueError(
            "candidate_ids must match teacher_log_probs, "
            f"got ids={candidate_ids.shape} and teacher={teacher_log_probs.shape}."
        )
    state_shape = teacher_log_probs.shape[:2]
    for name, value in (
        ("teacher_secondary_log_probs", teacher_secondary_log_probs),
        ("responses", responses),
        ("response_mask", response_mask),
    ):
        if value.shape != state_shape:
            raise ValueError(f"{name} must have shape {state_shape}, got {value.shape}.")

    eos_token_id = int(eos_token_id)
    secondary_token_id = int(secondary_token_id)
    if eos_token_id == secondary_token_id:
        raise ValueError("Secondary token id must differ from the student EOS token id.")
    floor = float(teacher_remap_floor)
    if not math.isfinite(floor) or floor <= 0.0 or floor >= 1.0:
        raise ValueError("opd_terminal_teacher_remap_floor must be finite and in (0, 1).")

    with torch.no_grad():
        teacher_log_probs = teacher_log_probs.detach().float()
        candidate_ids = candidate_ids.to(device=teacher_log_probs.device)
        teacher_secondary_log_probs = teacher_secondary_log_probs.to(
            device=teacher_log_probs.device
        ).detach().float()
        teacher_secondary_probability = teacher_secondary_log_probs.exp()
        valid_state_mask = response_mask.to(device=teacher_log_probs.device).bool()
        terminal_mask = valid_state_mask & responses.to(device=teacher_log_probs.device).eq(eos_token_id)
        eos_candidate_mask = valid_state_mask.unsqueeze(-1) & candidate_ids.eq(eos_token_id)
        secondary_candidate_mask = valid_state_mask.unsqueeze(-1) & candidate_ids.eq(secondary_token_id)
        teacher_stop_log_probability = torch.logaddexp(
            teacher_log_probs,
            teacher_secondary_log_probs.unsqueeze(-1),
        )
        teacher_stop_probability = teacher_stop_log_probability.exp()

        remapped_teacher_log_probs = teacher_log_probs.clone()
        remapped_teacher_log_probs = torch.where(
            eos_candidate_mask,
            teacher_stop_log_probability,
            remapped_teacher_log_probs,
        )
        remapped_teacher_log_probs = torch.where(
            secondary_candidate_mask,
            torch.full_like(remapped_teacher_log_probs, math.log(floor)),
            remapped_teacher_log_probs,
        )

    return TerminalTeacherRemapOnlyObjective(
        remapped_teacher_log_probs=remapped_teacher_log_probs,
        terminal_mask=terminal_mask,
        eos_candidate_mask=eos_candidate_mask,
        secondary_candidate_mask=secondary_candidate_mask,
        teacher_secondary_probability=teacher_secondary_probability.masked_fill(~terminal_mask, 0.0),
        teacher_stop_probability=teacher_stop_probability.masked_fill(~eos_candidate_mask, 0.0),
    )


def remap_terminal_teacher_trajectory_log_probs(
    *,
    teacher_log_probs: torch.Tensor,
    teacher_secondary_log_probs: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    eos_token_id: int,
) -> TerminalTeacherTrajectoryRemap:

    if teacher_log_probs.ndim != 2:
        raise ValueError(
            "teacher trajectory log-probs must have shape [batch, sequence], "
            f"got {teacher_log_probs.shape}."
        )
    expected_shape = teacher_log_probs.shape
    for name, value in (
        ("teacher_secondary_log_probs", teacher_secondary_log_probs),
        ("responses", responses),
        ("response_mask", response_mask),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}.")

    with torch.no_grad():
        teacher = teacher_log_probs.detach().float()
        secondary = teacher_secondary_log_probs.to(device=teacher.device).detach().float()
        terminal_mask = (
            response_mask.to(device=teacher.device).bool()
            & responses.to(device=teacher.device).eq(int(eos_token_id))
        )
        stop_log_probs = torch.logaddexp(teacher, secondary)
        remapped = torch.where(terminal_mask, stop_log_probs, teacher)

    return TerminalTeacherTrajectoryRemap(
        remapped_teacher_log_probs=remapped,
        terminal_mask=terminal_mask,
        teacher_secondary_probability=secondary.exp().masked_fill(~terminal_mask, 0.0),
        teacher_stop_probability=stop_log_probs.exp().masked_fill(~terminal_mask, 0.0),
    )


def prepare_terminal_aware_objective(
    *,
    current_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    behavior_eos_log_probs: torch.Tensor,
    behavior_secondary_log_probs: torch.Tensor | None = None,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    eos_token_id: int,
    anchor_mode: object,
    subtract_score_baseline: bool,
) -> TerminalAwareObjective:

    exact_token_count = 2 if behavior_secondary_log_probs is not None else 1
    minimum_candidates = exact_token_count + 1
    if current_log_probs.ndim != 3 or current_log_probs.size(-1) < minimum_candidates:
        raise ValueError(
            f"terminal-aware current_log_probs must have shape [batch, sequence, K>={minimum_candidates}], "
            f"got {current_log_probs.shape}."
        )
    if teacher_log_probs.shape != current_log_probs.shape:
        raise ValueError(
            "teacher_log_probs must match current_log_probs, "
            f"got teacher={teacher_log_probs.shape} and current={current_log_probs.shape}."
        )
    state_shape = current_log_probs.shape[:2]
    for name, value in (
        ("behavior_eos_log_probs", behavior_eos_log_probs),
        ("responses", responses),
        ("response_mask", response_mask),
    ):
        if value.shape != state_shape:
            raise ValueError(f"{name} must have shape {state_shape}, got {value.shape}.")
    if behavior_secondary_log_probs is not None and behavior_secondary_log_probs.shape != state_shape:
        raise ValueError(
            f"behavior_secondary_log_probs must have shape {state_shape}, "
            f"got {behavior_secondary_log_probs.shape}."
        )

    anchor_mode = _normalize_anchor_mode(anchor_mode)
    current_log_probs = current_log_probs.float()
    teacher_log_probs = teacher_log_probs.to(device=current_log_probs.device).float()
    terminal_mask = response_mask.bool() & responses.eq(int(eos_token_id))
    current_eos_log_probs = current_log_probs[..., 0]
    teacher_eos_log_probs = teacher_log_probs[..., 0]
    detached_behavior_log_probs = behavior_eos_log_probs.to(
        device=current_log_probs.device,
    ).detach().float()
    detached_teacher_log_probs = teacher_eos_log_probs.detach()

    behavior_probability = detached_behavior_log_probs.exp()
    current_probability = current_eos_log_probs.exp()
    teacher_probability = detached_teacher_log_probs.exp()
    zero_probability = torch.zeros_like(behavior_probability)
    if behavior_secondary_log_probs is None:
        behavior_secondary_probability = zero_probability
        current_secondary_probability = zero_probability
        teacher_secondary_probability = zero_probability
        teacher_stop_probability = teacher_probability
        current_log_content_mass = _log_one_minus_probability(current_eos_log_probs)
        teacher_log_content_mass = _log_one_minus_probability(teacher_eos_log_probs)
    else:
        behavior_secondary_log_probs = behavior_secondary_log_probs.to(
            device=current_log_probs.device,
        ).detach().float()
        current_secondary_log_probs = current_log_probs[..., 1]
        teacher_secondary_log_probs = teacher_log_probs[..., 1]
        behavior_secondary_probability = behavior_secondary_log_probs.exp()
        current_secondary_probability = current_secondary_log_probs.exp()
        teacher_secondary_probability = teacher_secondary_log_probs.detach().exp()
        teacher_stop_probability = (teacher_probability + teacher_secondary_probability).clamp(
            min=0.0,
            max=1.0,
        )
        current_log_content_mass = _log_one_minus_probability_sum(
            current_eos_log_probs,
            current_secondary_log_probs,
        )
        teacher_log_content_mass = _log_one_minus_probability_sum(
            teacher_eos_log_probs,
            teacher_secondary_log_probs,
        )
    if anchor_mode == "behavior":
        anchor_probability = behavior_probability
    else:
        anchor_probability = torch.maximum(behavior_probability, teacher_stop_probability)

    eps = torch.finfo(current_probability.dtype).eps
    stable_current_probability = current_probability.clamp(min=eps, max=1.0 - eps)
    stable_anchor_probability = anchor_probability.clamp(min=eps, max=1.0 - eps)
    gate_loss = stable_current_probability * (
        stable_current_probability.log() - stable_anchor_probability.log()
    ) + (1.0 - stable_current_probability) * (
        torch.log1p(-stable_current_probability) - torch.log1p(-stable_anchor_probability)
    )
    gate_loss = gate_loss.masked_fill(~terminal_mask, 0.0)

    current_conditional = current_log_probs[..., exact_token_count:] - current_log_content_mass.unsqueeze(-1)
    teacher_conditional = teacher_log_probs[..., exact_token_count:] - teacher_log_content_mass.unsqueeze(-1)

    score_baseline = 1.0 if subtract_score_baseline else 0.0
    ordinary_advantages = teacher_log_probs - current_log_probs - score_baseline
    conditional_advantages = teacher_conditional - current_conditional - score_baseline
    content_mass = (1.0 - behavior_probability - behavior_secondary_probability).clamp(
        min=0.0,
        max=1.0,
    ).detach()
    current_content_probability = (
        1.0 - current_probability - current_secondary_probability
    ).clamp(min=0.0, max=1.0)
    candidate_count = current_log_probs.size(-1)
    conditional_scale = (
        content_mass.unsqueeze(-1)
        * float(candidate_count)
        / float(candidate_count - exact_token_count)
    )
    terminal_advantages = torch.cat(
        (
            torch.zeros_like(current_log_probs[..., :exact_token_count]),
            conditional_advantages * conditional_scale,
        ),
        dim=-1,
    )
    terminal_policy_log_probs = torch.cat(
        (current_log_probs[..., :exact_token_count], current_conditional),
        dim=-1,
    )
    expanded_terminal_mask = terminal_mask.unsqueeze(-1)

    masked = lambda value: value.masked_fill(~terminal_mask, 0.0)
    return TerminalAwareObjective(
        policy_log_probs=torch.where(expanded_terminal_mask, terminal_policy_log_probs, current_log_probs),
        advantages=torch.where(expanded_terminal_mask, terminal_advantages, ordinary_advantages),
        gate_loss=gate_loss,
        terminal_mask=terminal_mask,
        behavior_eos_probability=masked(behavior_probability),
        current_eos_probability=masked(current_probability),
        teacher_eos_probability=masked(teacher_probability),
        behavior_secondary_probability=masked(behavior_secondary_probability),
        current_secondary_probability=masked(current_secondary_probability),
        teacher_secondary_probability=masked(teacher_secondary_probability),
        teacher_stop_probability=masked(teacher_stop_probability),
        anchor_probability=masked(anchor_probability),
        content_mass=masked(content_mass),
        current_content_probability=masked(current_content_probability),
        teacher_floor_active=masked(
            (teacher_stop_probability > behavior_probability).to(current_probability.dtype)
            if anchor_mode == "teacher_floor"
            else zero_probability
        ),
    )


def prepare_terminal_safe_continue_objective(
    *,
    current_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    eos_token_id: int,
    secondary_token_enabled: bool,
    subtract_score_baseline: bool,
) -> TerminalSafeContinueObjective:

    exact_token_count = 2 if secondary_token_enabled else 1
    minimum_candidates = exact_token_count + 1
    if current_log_probs.ndim != 3 or current_log_probs.size(-1) < minimum_candidates:
        raise ValueError(
            f"safe-continue current_log_probs must have shape [batch, sequence, K>={minimum_candidates}], "
            f"got {current_log_probs.shape}."
        )
    if teacher_log_probs.shape != current_log_probs.shape:
        raise ValueError(
            "teacher_log_probs must match current_log_probs, "
            f"got teacher={teacher_log_probs.shape} and current={current_log_probs.shape}."
        )
    state_shape = current_log_probs.shape[:2]
    for name, value in (("responses", responses), ("response_mask", response_mask)):
        if value.shape != state_shape:
            raise ValueError(f"{name} must have shape {state_shape}, got {value.shape}.")

    current_log_probs = current_log_probs.float()
    teacher_log_probs = teacher_log_probs.to(device=current_log_probs.device).float()
    terminal_mask = response_mask.bool() & responses.eq(int(eos_token_id))

    current_eos_log_probs = current_log_probs[..., 0]
    teacher_eos_log_probs = teacher_log_probs[..., 0]
    current_eos_probability = current_eos_log_probs.exp()
    teacher_eos_probability = teacher_eos_log_probs.detach().exp()
    if secondary_token_enabled:
        current_secondary_log_probs = current_log_probs[..., 1]
        teacher_secondary_log_probs = teacher_log_probs[..., 1]
        current_secondary_probability = current_secondary_log_probs.exp()
        teacher_secondary_probability = teacher_secondary_log_probs.detach().exp()
        teacher_stop_log_probs = torch.logaddexp(teacher_eos_log_probs, teacher_secondary_log_probs)
        current_log_content_mass = _log_one_minus_probability_sum(
            current_eos_log_probs,
            current_secondary_log_probs,
        )
        teacher_log_content_mass = _log_one_minus_probability_sum(
            teacher_eos_log_probs,
            teacher_secondary_log_probs,
        )
    else:
        current_secondary_probability = torch.zeros_like(current_eos_probability)
        teacher_secondary_probability = torch.zeros_like(teacher_eos_probability)
        teacher_stop_log_probs = teacher_eos_log_probs
        current_log_content_mass = _log_one_minus_probability(current_eos_log_probs)
        teacher_log_content_mass = _log_one_minus_probability(teacher_eos_log_probs)

    current_conditional = current_log_probs[..., exact_token_count:] - current_log_content_mass.unsqueeze(-1)
    teacher_conditional = teacher_log_probs[..., exact_token_count:] - teacher_log_content_mass.unsqueeze(-1)
    with torch.no_grad():
        stop_signal = teacher_stop_log_probs.detach() - current_eos_log_probs.detach()
        continuation_signal = (
            teacher_log_probs[..., exact_token_count:].detach()
            - current_log_probs[..., exact_token_count:].detach()
        ).mean(dim=-1)
        safe_gate_advantage = torch.relu(stop_signal - continuation_signal)
        gate_active = (stop_signal > continuation_signal).to(current_log_probs.dtype)
        current_content_probability = (
            1.0 - current_eos_probability.detach() - current_secondary_probability.detach()
        ).clamp(min=0.0, max=1.0)
        teacher_stop_probability = teacher_stop_log_probs.detach().exp().clamp(min=0.0, max=1.0)

    gate_loss = -safe_gate_advantage * current_eos_probability
    gate_loss = gate_loss.masked_fill(~terminal_mask, 0.0)

    score_baseline = 1.0 if subtract_score_baseline else 0.0
    ordinary_advantages = teacher_log_probs - current_log_probs - score_baseline
    conditional_advantages = teacher_conditional - current_conditional - score_baseline
    candidate_count = current_log_probs.size(-1)
    conditional_scale = (
        current_content_probability.unsqueeze(-1)
        * float(candidate_count)
        / float(candidate_count - exact_token_count)
    )
    terminal_advantages = torch.cat(
        (
            torch.zeros_like(current_log_probs[..., :exact_token_count]),
            conditional_advantages * conditional_scale,
        ),
        dim=-1,
    )
    terminal_policy_log_probs = torch.cat(
        (current_log_probs[..., :exact_token_count], current_conditional),
        dim=-1,
    )
    expanded_terminal_mask = terminal_mask.unsqueeze(-1)
    masked = lambda value: value.masked_fill(~terminal_mask, 0.0)

    return TerminalSafeContinueObjective(
        policy_log_probs=torch.where(expanded_terminal_mask, terminal_policy_log_probs, current_log_probs),
        advantages=torch.where(expanded_terminal_mask, terminal_advantages, ordinary_advantages),
        gate_loss=gate_loss,
        terminal_mask=terminal_mask,
        current_eos_probability=masked(current_eos_probability),
        current_secondary_probability=masked(current_secondary_probability),
        teacher_eos_probability=masked(teacher_eos_probability),
        teacher_secondary_probability=masked(teacher_secondary_probability),
        teacher_stop_probability=masked(teacher_stop_probability),
        current_content_probability=masked(current_content_probability),
        stop_signal=masked(stop_signal),
        continuation_signal=masked(continuation_signal),
        safe_gate_advantage=masked(safe_gate_advantage),
        gate_active=masked(gate_active),
    )


def prepare_terminal_conservative_kl_objective(
    *,
    current_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    candidate_ids: torch.Tensor,
    teacher_secondary_log_probs: torch.Tensor | None,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    eos_token_id: int,
    secondary_token_id: int | None,
    teacher_remap_enable: bool,
    teacher_remap_floor: float,
    baseline_mode: object,
    apply_eos_relu: bool,
    student_topm_ids: torch.Tensor | None = None,
    student_topm_log_probs: torch.Tensor | None = None,
    teacher_topm_log_probs: torch.Tensor | None = None,
) -> TerminalConservativeKLObjective:

    if current_log_probs.ndim != 3 or current_log_probs.size(-1) < 2:
        raise ValueError(
            "conservative terminal current_log_probs must have shape [batch, sequence, K>=2], "
            f"got {current_log_probs.shape}."
        )
    if teacher_log_probs.shape != current_log_probs.shape:
        raise ValueError(
            "teacher_log_probs must match current_log_probs, "
            f"got teacher={teacher_log_probs.shape} and current={current_log_probs.shape}."
        )
    if candidate_ids.shape != current_log_probs.shape:
        raise ValueError(
            "candidate_ids must match current_log_probs, "
            f"got ids={candidate_ids.shape} and current={current_log_probs.shape}."
        )
    state_shape = current_log_probs.shape[:2]
    for name, value in (("responses", responses), ("response_mask", response_mask)):
        if value.shape != state_shape:
            raise ValueError(f"{name} must have shape {state_shape}, got {value.shape}.")

    baseline_mode = _normalize_terminal_kl_baseline_mode(baseline_mode)
    candidate_count = current_log_probs.size(-1)
    conditional_count = candidate_count - 1
    if baseline_mode == "mc_loo" and conditional_count < 2:
        raise ValueError("conservative MC-LOO requires at least 3 Sample-K candidates.")

    current_log_probs = current_log_probs.float()
    teacher_log_probs = teacher_log_probs.to(device=current_log_probs.device).detach().float()
    candidate_ids = candidate_ids.to(device=current_log_probs.device)
    terminal_mask = response_mask.bool().to(current_log_probs.device) & responses.to(
        current_log_probs.device
    ).eq(int(eos_token_id))
    if terminal_mask.any() and not candidate_ids[..., 0][terminal_mask].eq(int(eos_token_id)).all():
        raise ValueError("conservative terminal OPD requires exact EOS in candidate slot zero.")
    if terminal_mask.any() and candidate_ids[..., 1:][terminal_mask].eq(int(eos_token_id)).any():
        raise ValueError("conservative terminal conditional candidates must exclude EOS.")

    floor = float(teacher_remap_floor)
    if not math.isfinite(floor) or floor <= 0.0 or floor >= 1.0:
        raise ValueError("opd_terminal_teacher_remap_floor must be finite and in (0, 1).")
    if teacher_remap_enable:
        if secondary_token_id is None or teacher_secondary_log_probs is None:
            raise ValueError("teacher EOS remapping requires a secondary teacher token and its log-probability.")
        if teacher_secondary_log_probs.shape != state_shape:
            raise ValueError(
                f"teacher_secondary_log_probs must have shape {state_shape}, "
                f"got {teacher_secondary_log_probs.shape}."
            )
        secondary_token_id = int(secondary_token_id)

    with torch.no_grad():
        remapped_teacher_log_probs = teacher_log_probs.clone()
        raw_teacher_eos_probability = teacher_log_probs[..., 0].exp()
        zero_probability = torch.zeros_like(raw_teacher_eos_probability)
        teacher_secondary_probability = zero_probability
        remapped_eos_probability = raw_teacher_eos_probability
        remapped_secondary_probability = zero_probability
        if teacher_remap_enable:
            teacher_secondary_log_probs = teacher_secondary_log_probs.to(
                device=current_log_probs.device
            ).detach().float()
            teacher_secondary_probability = teacher_secondary_log_probs.exp()
            stop_probability = raw_teacher_eos_probability + teacher_secondary_probability
            if terminal_mask.any() and (stop_probability[terminal_mask] <= floor).any():
                raise ValueError("teacher STOP probability must exceed the configured remap floor.")
            remapped_eos_probability = torch.where(
                terminal_mask,
                stop_probability - floor,
                raw_teacher_eos_probability,
            )
            remapped_secondary_probability = torch.where(
                terminal_mask,
                torch.full_like(stop_probability, floor),
                teacher_secondary_probability,
            )
            remapped_teacher_log_probs[..., 0] = torch.where(
                terminal_mask,
                remapped_eos_probability.clamp_min(torch.finfo(torch.float32).tiny).log(),
                remapped_teacher_log_probs[..., 0],
            )
            sampled_secondary = terminal_mask.unsqueeze(-1) & candidate_ids.eq(secondary_token_id)
            remapped_teacher_log_probs = torch.where(
                sampled_secondary,
                torch.full_like(remapped_teacher_log_probs, math.log(floor)),
                remapped_teacher_log_probs,
            )

        rewards = remapped_teacher_log_probs - current_log_probs.detach()
        current_eos_probability = current_log_probs[..., 0].detach().exp()
        eos_reward = rewards[..., 0]
        conditional_rewards = rewards[..., 1:]

        if baseline_mode == "mc_loo":
            conditional_reward_mean = conditional_rewards.mean(dim=-1)
            kl_baseline = (
                -current_eos_probability * eos_reward
                - (1.0 - current_eos_probability) * conditional_reward_mean
            )
            conditional_reward_loo = (
                float(conditional_count) * conditional_reward_mean.unsqueeze(-1)
                - conditional_rewards
            ) / float(conditional_count - 1)
            conditional_baseline = (
                -current_eos_probability.unsqueeze(-1) * eos_reward.unsqueeze(-1)
                - (1.0 - current_eos_probability).unsqueeze(-1) * conditional_reward_loo
            )
        else:
            topm_values = (student_topm_ids, student_topm_log_probs, teacher_topm_log_probs)
            if any(value is None for value in topm_values):
                raise ValueError(
                    "topm_coarse requires student_topm_ids, student_topm_log_probs, "
                    "and teacher_topm_log_probs."
                )
            if not (
                student_topm_ids.shape
                == student_topm_log_probs.shape
                == teacher_topm_log_probs.shape
            ):
                raise ValueError("all Top-M tensors must have the same shape.")
            if student_topm_ids.shape[:2] != state_shape:
                raise ValueError(f"Top-M tensors must start with shape {state_shape}.")
            student_topm_ids = student_topm_ids.to(device=current_log_probs.device)
            if terminal_mask.any() and student_topm_ids[terminal_mask].eq(int(eos_token_id)).any():
                raise ValueError("student Top-M support must exclude EOS.")
            sorted_topm_ids = student_topm_ids.sort(dim=-1).values
            if sorted_topm_ids.size(-1) > 1 and (
                sorted_topm_ids[..., 1:] == sorted_topm_ids[..., :-1]
            )[terminal_mask].any():
                raise ValueError("student Top-M support must contain unique token ids.")

            student_topm_probability = student_topm_log_probs.to(
                device=current_log_probs.device
            ).detach().float().exp()
            remapped_teacher_topm_log_probs = teacher_topm_log_probs.to(
                device=current_log_probs.device
            ).detach().float()
            if teacher_remap_enable:
                topm_secondary = terminal_mask.unsqueeze(-1) & student_topm_ids.eq(secondary_token_id)
                remapped_teacher_topm_log_probs = torch.where(
                    topm_secondary,
                    torch.full_like(remapped_teacher_topm_log_probs, math.log(floor)),
                    remapped_teacher_topm_log_probs,
                )
            teacher_topm_probability = remapped_teacher_topm_log_probs.exp()
            student_support_probability = torch.cat(
                (current_eos_probability.unsqueeze(-1), student_topm_probability), dim=-1
            )
            teacher_support_probability = torch.cat(
                (remapped_eos_probability.unsqueeze(-1), teacher_topm_probability), dim=-1
            )
            student_rest = 1.0 - student_support_probability.sum(dim=-1)
            teacher_rest = 1.0 - teacher_support_probability.sum(dim=-1)
            tolerance = 32.0 * torch.finfo(torch.float32).eps
            if terminal_mask.any() and (
                (student_rest[terminal_mask] < -tolerance).any()
                or (teacher_rest[terminal_mask] < -tolerance).any()
            ):
                raise ValueError("Top-M support probability exceeds one.")
            student_rest = student_rest.clamp_min(0.0)
            teacher_rest = teacher_rest.clamp_min(0.0)
            kl_baseline = (
                _xlogratio(student_support_probability, teacher_support_probability).sum(dim=-1)
                + _xlogratio(student_rest, teacher_rest)
            ).clamp_min(0.0)
            conditional_baseline = kl_baseline.unsqueeze(-1)

        unclipped_eos_advantage = eos_reward + kl_baseline
        eos_advantage = torch.relu(unclipped_eos_advantage) if apply_eos_relu else unclipped_eos_advantage
        conditional_advantages = conditional_rewards + conditional_baseline
        terminal_advantages = torch.cat(
            (
                float(candidate_count)
                * current_eos_probability.unsqueeze(-1)
                * eos_advantage.unsqueeze(-1),
                float(candidate_count)
                * (1.0 - current_eos_probability).unsqueeze(-1)
                * conditional_advantages
                / float(conditional_count),
            ),
            dim=-1,
        )
        ordinary_advantages = teacher_log_probs - current_log_probs.detach()
        advantages = torch.where(
            terminal_mask.unsqueeze(-1), terminal_advantages, ordinary_advantages
        ).detach()
        eos_gate_active = (unclipped_eos_advantage > 0.0).to(current_log_probs.dtype)

        masked = lambda value: value.masked_fill(~terminal_mask, 0.0)

    return TerminalConservativeKLObjective(
        policy_log_probs=current_log_probs,
        advantages=advantages,
        remapped_teacher_log_probs=remapped_teacher_log_probs,
        terminal_mask=terminal_mask,
        kl_baseline=masked(kl_baseline),
        eos_advantage=masked(eos_advantage),
        eos_gate_active=masked(eos_gate_active),
        current_eos_probability=masked(current_eos_probability),
        teacher_eos_probability=masked(raw_teacher_eos_probability),
        teacher_secondary_probability=masked(teacher_secondary_probability),
        teacher_remapped_eos_probability=masked(remapped_eos_probability),
        teacher_remapped_secondary_probability=masked(remapped_secondary_probability),
    )


def validate_terminal_aware_configuration(
    *,
    enabled: bool,
    anchor_mode: object,
    gate_coef: float,
    candidate_mode: object,
    sample_replacement: bool,
    top_k: int,
    top_k_strategy: object,
    opd_loss_type: object,
    advantage_mode: object,
    candidate_aggregation: object,
    advantage_centering: object,
    adaptive_update_enable: bool,
    no_candidate_is: bool,
    forced_eos_diagnostic_enable: bool,
    q_mixture_enable: bool,
    q_source_normalize_enable: bool,
    samplek_entropy_coef: float,
    influence_clip: float | None,
    raw_advantage_clip: float | None,
    generic_candidate_weights_present: bool,
    secondary_token_id: int | None = None,
    objective_mode: object = "anchor_kl",
    teacher_remap_enable: bool = False,
    kl_baseline_mode: object = "mc_loo",
    teacher_remap_floor: float = 1e-18,
    terminal_topm: int = 0,
    subtract_score_baseline: bool = False,
    candidate_reuse_enabled: bool = False,
) -> None:

    if not enabled:
        return

    objective_mode = normalize_terminal_objective_mode(objective_mode)
    anchor_mode = _normalize_anchor_mode(anchor_mode)
    gate_coef = float(gate_coef)
    if not math.isfinite(gate_coef) or gate_coef < 0.0:
        raise ValueError("opd_terminal_gate_coef must be finite and nonnegative.")
    remap_objective_modes = {"conservative_kl", "teacher_remap_only"}
    if teacher_remap_enable and objective_mode not in remap_objective_modes:
        raise ValueError(
            "teacher EOS remapping requires objective_mode=conservative_kl or teacher_remap_only."
        )
    if objective_mode in remap_objective_modes:
        teacher_remap_floor = float(teacher_remap_floor)
        if (
            not math.isfinite(teacher_remap_floor)
            or teacher_remap_floor <= 0.0
            or teacher_remap_floor >= 1.0
        ):
            raise ValueError("opd_terminal_teacher_remap_floor must be finite and in (0, 1).")
        if teacher_remap_enable and secondary_token_id is None:
            raise ValueError("teacher EOS remapping requires opd_terminal_secondary_token_id.")
    if objective_mode == "teacher_remap_only":
        if not teacher_remap_enable:
            raise ValueError("teacher_remap_only requires teacher EOS remapping to be enabled.")
        if subtract_score_baseline:
            raise ValueError("teacher_remap_only requires sample_k_kl_plus_one=False.")
    if objective_mode == "conservative_kl":
        kl_baseline_mode = _normalize_terminal_kl_baseline_mode(kl_baseline_mode)
        if kl_baseline_mode == "mc_loo" and int(top_k) < 3:
            raise ValueError("conservative MC-LOO requires at least 3 Sample-K candidates.")
        if kl_baseline_mode == "topm_coarse" and int(terminal_topm) <= 0:
            raise ValueError("topm_coarse requires opd_terminal_topm to be positive.")
        if subtract_score_baseline:
            raise ValueError("conservative terminal KL requires sample_k_kl_plus_one=False.")
    if secondary_token_id is not None:
        secondary_token_id = int(secondary_token_id)
        if secondary_token_id < 0:
            raise ValueError("opd_terminal_secondary_token_id must be nonnegative or null.")
        if objective_mode == "anchor_kl" and anchor_mode != "teacher_floor":
            raise ValueError(
                "opd_terminal_secondary_token_id requires opd_terminal_anchor_mode=teacher_floor."
            )
        if objective_mode in {"anchor_kl", "safe_continue"} and int(top_k) < 3:
            raise ValueError("dual-token terminal-aware Sample-K requires at least 3 candidates.")

    def normalized(value: object) -> str:
        return str(value).strip().lower().replace("-", "_")

    normalized_candidate_mode = normalized(candidate_mode)
    adaptive_teacher_remap = (
        objective_mode == "teacher_remap_only"
        and normalized_candidate_mode == "adaptive_head_tail"
    )
    topk_teacher_remap = (
        objective_mode == "teacher_remap_only"
        and normalized_candidate_mode == "topk"
    )
    weighted_teacher_remap = adaptive_teacher_remap or topk_teacher_remap
    if normalized_candidate_mode != "sample_stu" and not weighted_teacher_remap:
        raise ValueError(
            "terminal-aware OPD requires candidate_mode=sample_stu, except teacher_remap_only "
            "may use adaptive_head_tail or topk."
        )
    if not topk_teacher_remap and not sample_replacement:
        raise ValueError("terminal-aware OPD requires sampling with replacement.")
    if int(top_k) < 2:
        raise ValueError("terminal-aware OPD requires at least 2 Sample-K candidates.")
    if normalized(top_k_strategy) != "only_stu":
        raise ValueError("terminal-aware OPD requires top_k_strategy=only_stu.")
    if normalized(opd_loss_type) != "sample_k_reverse_kl":
        raise ValueError("terminal-aware OPD requires the sample-k reverse_kl loss.")
    if normalized(advantage_mode) != "current_kl_is":
        raise ValueError("terminal-aware OPD requires advantage_mode=current_kl_is.")
    normalized_candidate_aggregation = normalized(candidate_aggregation)
    if weighted_teacher_remap:
        if normalized_candidate_aggregation != "sum":
            raise ValueError(
                "teacher_remap_only with weighted candidates requires candidate aggregation=sum."
            )
        if not generic_candidate_weights_present:
            raise ValueError(
                "teacher_remap_only with weighted candidates requires generic candidate weights."
            )
    elif normalized_candidate_aggregation != "mean":
        raise ValueError("terminal-aware OPD requires candidate aggregation=mean.")
    normalized_advantage_centering = normalized(advantage_centering)
    remap_only_samplek_loo = (
        objective_mode == "teacher_remap_only"
        and normalized_candidate_mode == "sample_stu"
        and normalized_advantage_centering == "leave_one_out"
    )
    if normalized_advantage_centering != "none" and not remap_only_samplek_loo:
        raise ValueError("terminal-aware OPD requires advantage centering=none; LOO is unsupported.")
    if topk_teacher_remap and adaptive_update_enable:
        raise ValueError("teacher_remap_only with topk initially supports non-adaptive updates only.")
    if not topk_teacher_remap and not adaptive_update_enable:
        raise ValueError("terminal-aware OPD requires adaptive resampling updates.")
    if topk_teacher_remap:
        if no_candidate_is:
            raise ValueError("teacher_remap_only with topk requires candidate IS to be enabled.")
        if candidate_reuse_enabled:
            raise ValueError("teacher_remap_only with topk does not use Sample-K candidate reuse.")
    else:
        if candidate_reuse_enabled and no_candidate_is:
            raise ValueError("terminal-aware candidate reuse requires candidate IS to be enabled.")
        if not no_candidate_is and not candidate_reuse_enabled:
            raise ValueError("terminal-aware OPD requires candidate IS to be disabled because candidates are resampled.")
    if forced_eos_diagnostic_enable:
        raise ValueError("terminal-aware OPD is mutually exclusive with the forced-EOS diagnostic.")
    if q_mixture_enable and objective_mode != "teacher_remap_only":
        raise ValueError("Q-mixture terminal-aware OPD requires objective_mode=teacher_remap_only.")
    if q_source_normalize_enable and not q_mixture_enable:
        raise ValueError("terminal-aware Q-mixture source normalization requires Q-mixture OPD.")
    if float(samplek_entropy_coef) != 0.0:
        raise ValueError("terminal-aware OPD does not initially support Sample-K entropy advantages.")
    if influence_clip is not None:
        raise ValueError("terminal-aware OPD does not initially support Sample-K influence clip.")
    if raw_advantage_clip is not None:
        raise ValueError("terminal-aware OPD does not initially support raw advantage clip.")
    if generic_candidate_weights_present and not weighted_teacher_remap:
        raise ValueError("terminal-aware OPD does not support generic candidate weights.")
