from __future__ import annotations

import math

import torch


def is_samplek_candidate_refresh_update(*, update_idx: int, refresh_interval: int) -> bool:

    if isinstance(update_idx, bool) or int(update_idx) != update_idx or int(update_idx) < 0:
        raise ValueError("Sample-K candidate refresh update index must be a nonnegative integer.")
    if (
        isinstance(refresh_interval, bool)
        or not math.isfinite(float(refresh_interval))
        or int(refresh_interval) != refresh_interval
        or int(refresh_interval) < 1
    ):
        raise ValueError("Sample-K candidate refresh interval must be a positive integer.")
    return int(update_idx) % int(refresh_interval) == 0


def validate_samplek_candidate_reuse_configuration(
    *,
    enabled: bool,
    refresh_interval: int,
    adaptive_update_enable: bool,
    candidate_mode: object,
    candidate_count: int,
    no_candidate_is: bool,
    abs_log_ratio_threshold: float | None,
    sampled_token_rkl_k3_threshold: float | None,
    q_mixture_enable: bool,
    forced_eos_diagnostic_enable: bool,
    eos_future_enable: bool,
) -> None:

    if not enabled:
        return
    is_samplek_candidate_refresh_update(update_idx=0, refresh_interval=refresh_interval)
    if not adaptive_update_enable:
        raise ValueError("Sample-K candidate reuse requires adaptive PPO updates.")
    mode = str(candidate_mode or "").strip().lower().replace("-", "_")
    mode = {
        "sample": "sample_stu",
        "sample_student": "sample_stu",
        "student_sample": "sample_stu",
        "sample_k": "sample_stu",
    }.get(mode, mode)
    if mode != "sample_stu":
        raise ValueError("Sample-K candidate reuse requires candidate_mode=sample_stu.")
    if isinstance(candidate_count, bool) or int(candidate_count) != candidate_count or int(candidate_count) < 1:
        raise ValueError("Sample-K candidate reuse requires a positive candidate count.")
    if no_candidate_is:
        raise ValueError("Sample-K candidate reuse requires candidate IS to be enabled.")
    if abs_log_ratio_threshold is not None or sampled_token_rkl_k3_threshold is not None:
        raise ValueError("Sample-K candidate reuse does not yet support adaptive stop thresholds.")
    if q_mixture_enable:
        raise ValueError("Sample-K candidate reuse does not yet support Q-mixture OPD.")
    if forced_eos_diagnostic_enable:
        raise ValueError("Sample-K candidate reuse does not yet support forced-EOS diagnostics.")
    if eos_future_enable:
        raise ValueError("Sample-K candidate reuse does not yet support EOS future correction.")


def _quantile(values: torch.Tensor, q: float, default: float) -> float:
    if values.numel() == 0:
        return default
    return torch.quantile(values.float(), q).item()


def _sequence_means(values: torch.Tensor, state_mask: torch.Tensor) -> torch.Tensor:
    denom = state_mask.sum(dim=-1).clamp_min(1.0)
    means = (values * state_mask).sum(dim=-1) / denom
    return means[state_mask.sum(dim=-1) > 0]


def compute_samplek_probe_diagnostics(
    *,
    current_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_log_probs: torch.Tensor | None = None,
    log_ratio_clip: float = 20.0,
) -> dict[str, float]:

    current = current_log_probs.detach().float()
    reference = reference_log_probs.to(device=current.device).detach().float()
    token_mask = response_mask.to(device=current.device).detach().float()

    if current.ndim != 3:
        raise ValueError(f"sample-k probe log-probs must be 3D, got shape {current.shape}.")
    if current.shape != reference.shape:
        raise ValueError(
            f"current sample-k shape {current.shape} does not match reference shape {reference.shape}."
        )
    if current.shape[:2] != token_mask.shape:
        raise ValueError(
            f"sample-k prefix shape {current.shape[:2]} does not match response mask {token_mask.shape}."
        )
    if teacher_log_probs is not None and teacher_log_probs.shape != current.shape:
        raise ValueError(
            f"teacher sample-k shape {teacher_log_probs.shape} does not match current shape {current.shape}."
        )

    candidate_mask = (
        (token_mask > 0.5).unsqueeze(-1)
        & torch.isfinite(current)
        & torch.isfinite(reference)
        & (current > -1e6)
        & (reference > -1e6)
    )
    candidate_mask_f = candidate_mask.to(dtype=current.dtype)
    valid_count = candidate_mask_f.sum(dim=-1)
    state_mask = ((token_mask > 0.5) & (valid_count > 0)).to(dtype=current.dtype)
    valid_count = valid_count.clamp_min(1.0)

    raw_log_ratio = torch.where(candidate_mask, current - reference, torch.zeros_like(current))
    clip = abs(float(log_ratio_clip))
    log_ratio = raw_log_ratio.clamp(min=-clip, max=clip)
    clipped = (raw_log_ratio.abs() > clip) & candidate_mask

    k3 = (torch.expm1(log_ratio) - log_ratio) * candidate_mask_f
    state_k3 = k3.sum(dim=-1) / valid_count

    masked_log_ratio = torch.where(candidate_mask, log_ratio, torch.full_like(log_ratio, -torch.inf))
    max_log_ratio = masked_log_ratio.max(dim=-1, keepdim=True).values
    max_log_ratio = torch.where(torch.isfinite(max_log_ratio), max_log_ratio, torch.zeros_like(max_log_ratio))
    stable_weights = torch.exp(log_ratio - max_log_ratio) * candidate_mask_f
    weight_sum = stable_weights.sum(dim=-1)
    weight_sq_sum = stable_weights.square().sum(dim=-1).clamp_min(1e-12)
    state_ess = (weight_sum.square() / (weight_sq_sum * valid_count)).clamp(min=0.0, max=1.0)

    state_abs_log_ratio = (log_ratio.abs() * candidate_mask_f).sum(dim=-1) / valid_count
    valid_state_k3 = state_k3[state_mask > 0.5]
    valid_state_ess = state_ess[state_mask > 0.5]
    valid_state_abs_log_ratio = state_abs_log_ratio[state_mask > 0.5]
    sequence_k3 = _sequence_means(state_k3, state_mask)
    sequence_ess = _sequence_means(state_ess, state_mask)
    sequence_abs_log_ratio = _sequence_means(state_abs_log_ratio, state_mask)

    state_denom = state_mask.sum().clamp_min(1.0)
    metrics = {
        "candidate_probe/k3_kl_mean": ((state_k3 * state_mask).sum() / state_denom).item(),
        "candidate_probe/k3_kl_state_p95": _quantile(valid_state_k3, 0.95, 0.0),
        "candidate_probe/k3_kl_sequence_p50": _quantile(sequence_k3, 0.50, 0.0),
        "candidate_probe/k3_kl_sequence_p90": _quantile(sequence_k3, 0.90, 0.0),
        "candidate_probe/ess_mean": ((state_ess * state_mask).sum() / state_denom).item(),
        "candidate_probe/ess_state_p10": _quantile(valid_state_ess, 0.10, 1.0),
        "candidate_probe/ess_sequence_p10": _quantile(sequence_ess, 0.10, 1.0),
        "candidate_probe/log_ratio_abs_mean": (
            (state_abs_log_ratio * state_mask).sum() / state_denom
        ).item(),
        "candidate_probe/log_ratio_abs_state_p95": _quantile(valid_state_abs_log_ratio, 0.95, 0.0),
        "candidate_probe/log_ratio_abs_sequence_p90": _quantile(sequence_abs_log_ratio, 0.90, 0.0),
        "candidate_probe/log_ratio_clip_fraction": (
            clipped.to(dtype=current.dtype).sum() / candidate_mask_f.sum().clamp_min(1.0)
        ).item(),
        "candidate_probe/valid_candidate_fraction": (
            candidate_mask_f.sum() / ((token_mask > 0.5).sum().clamp_min(1) * current.size(-1))
        ).item(),
    }

    if teacher_log_probs is not None:
        teacher = teacher_log_probs.to(device=current.device).detach().float()
        teacher_mask = candidate_mask & torch.isfinite(teacher) & (teacher > -1e6)
        teacher_mask_f = teacher_mask.to(dtype=current.dtype)
        teacher_count = teacher_mask_f.sum(dim=-1).clamp_min(1.0)
        teacher_state_mask = ((token_mask > 0.5) & (teacher_mask_f.sum(dim=-1) > 0)).to(dtype=current.dtype)

        ratio = torch.exp(log_ratio) * teacher_mask_f
        teacher_gap = torch.where(teacher_mask, current - teacher, torch.zeros_like(current))
        rkl_terms = ratio * teacher_gap.clamp(min=-1e3, max=1e3)
        state_rkl = rkl_terms.sum(dim=-1) / teacher_count
        sequence_rkl = _sequence_means(state_rkl, teacher_state_mask)
        teacher_state_denom = teacher_state_mask.sum().clamp_min(1.0)
        teacher_gap_abs = (teacher_gap.abs() * teacher_mask_f).sum(dim=-1) / teacher_count
        metrics.update(
            {
                "candidate_probe/teacher_rkl_is_proxy": (
                    (state_rkl * teacher_state_mask).sum() / teacher_state_denom
                ).item(),
                "candidate_probe/teacher_rkl_is_proxy_sequence_p50": _quantile(sequence_rkl, 0.50, 0.0),
                "candidate_probe/teacher_logprob_gap_abs_mean": (
                    (teacher_gap_abs * teacher_state_mask).sum() / teacher_state_denom
                ).item(),
            }
        )

    return metrics
