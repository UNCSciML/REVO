from __future__ import annotations

from dataclasses import dataclass

import torch

from verl.trainer.ppo.opd_decomposed import apply_samplek_influence_clip


FORCED_EOS_ESTIMATOR_WEIGHTS_KEY = "opd_diagnostic_forced_eos_estimator_weights"


@dataclass(frozen=True)
class ForcedEOSCandidateSample:
    candidate_ids: torch.Tensor
    estimator_weights: torch.Tensor
    eos_probabilities: torch.Tensor


@dataclass(frozen=True)
class ForcedEOSInfluenceOutput:
    advantages: torch.Tensor
    clip_metrics: dict[str, float]
    metrics: dict[str, float]
    consumed_prefix: bool


def _masked_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask.bool()]
    if selected.numel() == 0:
        return values.new_zeros(1, dtype=torch.float32)
    return selected.float()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    return _masked_values(values, mask).mean().item()


def _masked_quantile(values: torch.Tensor, mask: torch.Tensor, quantile: float) -> float:
    return torch.quantile(_masked_values(values, mask), quantile).item()


def sample_forced_eos_candidates(
    log_probs_all: torch.Tensor,
    *,
    num_candidates: int,
    eos_token_id: int,
) -> ForcedEOSCandidateSample:

    if log_probs_all.ndim < 1:
        raise ValueError(f"log_probs_all must have a vocabulary dimension, got {log_probs_all.shape}.")
    num_candidates = int(num_candidates)
    if num_candidates < 2:
        raise ValueError(f"forced-EOS diagnostic requires at least 2 candidates, got {num_candidates}.")
    vocab_size = log_probs_all.size(-1)
    if vocab_size < 2:
        raise ValueError(f"forced-EOS diagnostic requires vocabulary size at least 2, got {vocab_size}.")
    eos_token_id = int(eos_token_id)
    if eos_token_id < 0 or eos_token_id >= vocab_size:
        raise ValueError(f"EOS token id {eos_token_id} is outside vocabulary size {vocab_size}.")

    detached_log_probs = log_probs_all.detach().float()
    eos_probabilities = detached_log_probs[..., eos_token_id].exp()
    if not torch.isfinite(eos_probabilities).all():
        raise ValueError("forced-EOS diagnostic requires finite EOS probabilities.")

    conditional_logits = detached_log_probs.clone()
    conditional_logits[..., eos_token_id] = -torch.inf
    has_non_eos_support = torch.isfinite(conditional_logits).any(dim=-1)
    if not has_non_eos_support.all():
        raise ValueError("forced-EOS diagnostic requires nonzero non-EOS probability mass in every row.")
    conditional_probs = torch.softmax(conditional_logits, dim=-1)
    if not torch.isfinite(conditional_probs).all():
        raise ValueError("forced-EOS conditional probabilities must be finite.")

    flat_probs = conditional_probs.reshape(-1, vocab_size)
    try:
        non_eos_ids = torch.multinomial(flat_probs, num_samples=num_candidates - 1, replacement=True)
    except RuntimeError:
        non_eos_ids = torch.multinomial(flat_probs.float(), num_samples=num_candidates - 1, replacement=True)
    non_eos_ids = non_eos_ids.view(*log_probs_all.shape[:-1], num_candidates - 1)
    eos_ids = torch.full_like(non_eos_ids[..., :1], eos_token_id)
    candidate_ids = torch.cat((eos_ids, non_eos_ids), dim=-1)

    eos_weights = float(num_candidates) * eos_probabilities.unsqueeze(-1)
    non_eos_weights = (
        float(num_candidates)
        * (1.0 - eos_probabilities).clamp_min(0.0).unsqueeze(-1)
        / float(num_candidates - 1)
    )
    estimator_weights = torch.cat((eos_weights, non_eos_weights.expand_as(non_eos_ids)), dim=-1)
    return ForcedEOSCandidateSample(
        candidate_ids=candidate_ids,
        estimator_weights=estimator_weights.detach(),
        eos_probabilities=eos_probabilities.detach(),
    )


def apply_forced_eos_diagnostic_influence(
    *,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    prefix_weights: torch.Tensor | None,
    estimator_weights: torch.Tensor,
    candidate_ids: torch.Tensor,
    eos_token_id: int,
    clip: float | None,
) -> ForcedEOSInfluenceOutput:

    if advantages.ndim != 3:
        raise ValueError(f"forced-EOS advantages must have shape [batch, sequence, K], got {advantages.shape}.")
    if response_mask.shape != advantages.shape[:2]:
        raise ValueError(
            "forced-EOS response_mask must match candidate batch and sequence dimensions, "
            f"got mask={response_mask.shape}, candidates={advantages.shape}."
        )
    if estimator_weights.shape != advantages.shape or candidate_ids.shape != advantages.shape:
        raise ValueError(
            "forced-EOS estimator weights and candidate ids must match advantages, "
            f"got weights={estimator_weights.shape}, ids={candidate_ids.shape}, advantages={advantages.shape}."
        )
    if prefix_weights is not None and prefix_weights.shape != response_mask.shape:
        raise ValueError(
            f"forced-EOS prefix weights must match response_mask, got {prefix_weights.shape} and {response_mask.shape}."
        )

    if clip is not None:
        clip_output = apply_samplek_influence_clip(
            advantages=advantages,
            response_mask=response_mask,
            prefix_weights=prefix_weights,
            clip=clip,
        )
        unweighted = clip_output.advantages.detach().float()
        clip_metrics = clip_output.metrics
    else:
        unweighted = advantages.detach().float()
        if prefix_weights is not None:
            prefix = prefix_weights.to(device=unweighted.device).detach().float()
            unweighted = unweighted * prefix.unsqueeze(-1)
        clip_metrics = {
            "enabled": 0.0,
            "clip": 0.0,
            "prefix_weight_enabled": float(prefix_weights is not None),
        }

    weights = estimator_weights.to(device=unweighted.device).detach().float()
    ids = candidate_ids.to(device=unweighted.device)
    weighted = unweighted * weights
    state_mask = response_mask.to(device=unweighted.device).bool()
    candidate_mask = state_mask.unsqueeze(-1).expand_as(unweighted)
    eos_mask = ids.eq(int(eos_token_id)) & candidate_mask
    non_eos_mask = ids.ne(int(eos_token_id)) & candidate_mask
    candidate_count = advantages.size(-1)

    eos_counts = ids.eq(int(eos_token_id)).sum(dim=-1)
    weight_sums = weights.sum(dim=-1)
    eos_probabilities = (weights * ids.eq(int(eos_token_id)).to(weights.dtype)).sum(dim=-1) / float(
        candidate_count
    )
    natural_inclusion = 1.0 - (1.0 - eos_probabilities).clamp(min=0.0, max=1.0).pow(candidate_count)
    absolute_weighted = weighted.abs()
    eos_absolute_sum = absolute_weighted.masked_fill(~eos_mask, 0.0).sum()
    total_absolute_sum = absolute_weighted.masked_fill(~candidate_mask, 0.0).sum().clamp_min(1e-12)

    metrics = {
        "enabled": 1.0,
        "candidate_count": float(candidate_count),
        "eos_probability_mean": _masked_mean(eos_probabilities, state_mask),
        "eos_probability_min": _masked_values(eos_probabilities, state_mask).min().item(),
        "eos_probability_p50": _masked_quantile(eos_probabilities, state_mask, 0.50),
        "eos_probability_p95": _masked_quantile(eos_probabilities, state_mask, 0.95),
        "eos_probability_p99": _masked_quantile(eos_probabilities, state_mask, 0.99),
        "eos_probability_max": _masked_values(eos_probabilities, state_mask).max().item(),
        "natural_eos_inclusion_probability_mean": _masked_mean(natural_inclusion, state_mask),
        "estimator_weight_sum_max_error": _masked_values(
            (weight_sums - float(candidate_count)).abs(), state_mask
        ).max().item(),
        "exactly_one_eos_fraction": _masked_mean(eos_counts.eq(1).float(), state_mask),
        "eos_unweighted_influence_mean": _masked_mean(unweighted, eos_mask),
        "non_eos_unweighted_influence_mean": _masked_mean(unweighted, non_eos_mask),
        "eos_weighted_influence_abs_mean": _masked_mean(absolute_weighted, eos_mask),
        "non_eos_weighted_influence_abs_mean": _masked_mean(absolute_weighted, non_eos_mask),
        "eos_absolute_signal_fraction": (eos_absolute_sum / total_absolute_sum).item(),
    }
    return ForcedEOSInfluenceOutput(
        advantages=weighted.to(dtype=advantages.dtype),
        clip_metrics=clip_metrics,
        metrics=metrics,
        consumed_prefix=prefix_weights is not None,
    )


def validate_forced_eos_diagnostic_configuration(
    *,
    enabled: bool,
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
    q_source_normalize_enable: bool,
    generic_candidate_weights_present: bool,
) -> None:

    if not enabled:
        return

    def normalized(value: object) -> str:
        return str(value).strip().lower().replace("-", "_")

    if normalized(candidate_mode) != "sample_stu":
        raise ValueError("forced-EOS diagnostic requires candidate_mode=sample_stu.")
    if not sample_replacement:
        raise ValueError("forced-EOS diagnostic requires sampling with replacement.")
    if int(top_k) < 2:
        raise ValueError("forced-EOS diagnostic requires at least 2 Sample-K candidates.")
    if normalized(top_k_strategy) != "only_stu":
        raise ValueError("forced-EOS diagnostic requires top_k_strategy=only_stu.")
    if normalized(opd_loss_type) != "sample_k_reverse_kl":
        raise ValueError("forced-EOS diagnostic requires the sample-k reverse_kl loss.")
    if normalized(advantage_mode) != "current_kl_is":
        raise ValueError("forced-EOS diagnostic requires advantage_mode=current_kl_is.")
    if normalized(candidate_aggregation) != "mean":
        raise ValueError("forced-EOS diagnostic requires candidate aggregation=mean.")
    if normalized(advantage_centering) != "none":
        raise ValueError("forced-EOS diagnostic requires advantage centering=none; LOO is unsupported.")
    if not adaptive_update_enable:
        raise ValueError("forced-EOS diagnostic requires adaptive PPO updates.")
    if not no_candidate_is:
        raise ValueError("forced-EOS diagnostic requires candidate IS to be disabled.")
    if q_source_normalize_enable:
        raise ValueError("forced-EOS diagnostic does not support Q-mixture source normalization.")
    if generic_candidate_weights_present:
        raise ValueError("forced-EOS diagnostic cannot compose with generic candidate estimator weights.")
