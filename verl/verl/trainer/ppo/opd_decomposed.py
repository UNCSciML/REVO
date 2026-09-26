from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from verl.trainer.ppo.core_algos import agg_loss

DECOMPOSED_PI_OLD_MODES = frozenset(
    {
        "decomposed_pi_old",
        "pi_old_decomposed",
        "prefix_is_pi_old",
    }
)
CURRENT_TEACHER_LOG_PROB_MODES = frozenset({"current_kl_is", "current_kl"})
PREFIX_IS_MODES = frozenset(
    {
        "cumulative_cap",
        "hard_window",
        "prefix_geometric",
    }
)
PREFIX_IS_MODE_IDS = {
    "cumulative_cap": 0.0,
    "hard_window": 1.0,
    "prefix_geometric": 2.0,
}

_NUMERICAL_LOG_RATIO_BOUND = 20.0
OPD_ROLLOUT_REFERENCE_LOG_PROBS_KEY = "opd_rollout_reference_log_probs"
OPD_Q_MIXTURE_ALPHA_KEY = "opd_q_mixture_alpha"
OPD_Q_MIXTURE_IS_TEACHER_KEY = "opd_q_mixture_is_teacher"
OPD_Q_MIXTURE_PRIOR_ALPHA_KEY = "opd_q_mixture_prior_alpha"
OPD_Q_MIXTURE_SOURCE_WEIGHTS_KEY = "opd_q_mixture_source_weights"
OPD_Q_MIXTURE_TEACHER_LOG_PROBS_KEY = "opd_q_mixture_teacher_log_probs"
Q_MIXTURE_TEACHER_ADVANTAGE_MODE_IDS = {
    "proposal": 0.0,
    "current": 1.0,
    "fixed_alpha_current": 2.0,
}
SAMPLEK_ADVANTAGE_CENTERING_MODES = frozenset({"none", "leave_one_out"})
SAMPLEK_LOO_VARIANCE_FILTER_MODES = frozenset(
    {"hard", "soft", "expectile", "sqrt_clip", "fisher_posterior"}
)
SAMPLEK_LOO_VARIANCE_THRESHOLD_MODES = frozenset({"fixed", "update0_quantile"})
SAMPLEK_LOO_VARIANCE_SQRT_EPSILON = 1.0e-8
SAMPLEK_LOO_VARIANCE_QUANTILE_MIN_THRESHOLD = 1.0e-12
SAMPLED_TOKEN_PROXIMAL_MODES = frozenset({"reverse_kl", "quadratic_log_ratio"})
SAMPLED_TOKEN_PROXIMAL_MODE_IDS = {
    "reverse_kl": 0.0,
    "quadratic_log_ratio": 1.0,
}


@dataclass(frozen=True)
class DecomposedOPDLossOutput:
    loss: torch.Tensor
    teacher_loss: torch.Tensor
    proximal_loss: torch.Tensor
    prefix_is_weights: torch.Tensor
    teacher_coefficients: torch.Tensor
    proximal_coefficients: torch.Tensor
    teacher_effective_mask: torch.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class SampledTokenProximalLossOutput:
    loss: torch.Tensor
    coefficients: torch.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class SampledTokenRKLDiagnosticsOutput:
    log_ratio: torch.Tensor
    ratio: torch.Tensor
    k3: torch.Tensor
    numerical_clip_mask: torch.Tensor
    k3_mean: torch.Tensor


@dataclass(frozen=True)
class SampleKEntropyAdvantageOutput:
    advantages: torch.Tensor
    entropy_advantages: torch.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class RawOPDAdvantageClipOutput:
    advantages: torch.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class AdaptiveHeadTailNegativeELUOutput:
    advantages: torch.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class SampleKInfluenceClipOutput:
    advantages: torch.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class SampleKLOOVarianceFilterOutput:
    advantages: torch.Tensor
    variance: torch.Tensor
    active_mask: torch.Tensor
    normalization_weights: torch.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class SampleKLOOVarianceRatioDiagnosticsOutput:
    ratio: torch.Tensor
    log_ratio: torch.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class SampleKLOOVarianceHistogramDiagnosticsOutput:
    bin_edges: torch.Tensor
    bin_counts: torch.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class TrajectoryMixtureProposalOutput:
    conditional_log_probs: torch.Tensor
    prefix_log_probs: torch.Tensor
    old_component_posterior: torch.Tensor


@dataclass(frozen=True)
class QMixtureSampleKAdvantageOutput:
    teacher_advantages: torch.Tensor
    transformed_log_ratio: torch.Tensor
    current_teacher_log_ratio: torch.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class QMixtureSourceNormalizationOutput:
    weights: torch.Tensor
    metrics: dict[str, float]


def normalize_opd_mode(value: object) -> str:
    return str(value or "fixed").strip().lower().replace("-", "_")


def normalize_samplek_advantage_centering(value: object) -> str:
    mode = str(value or "none").strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "disabled": "none",
        "loo": "leave_one_out",
        "leave_one_out_baseline": "leave_one_out",
    }
    mode = aliases.get(mode, mode)
    if mode not in SAMPLEK_ADVANTAGE_CENTERING_MODES:
        raise ValueError(
            "sample-k advantage centering must be none or leave_one_out, "
            f"got {value!r}."
        )
    return mode


def normalize_samplek_loo_variance_threshold_mode(value: object) -> str:
    mode = str(value or "fixed").strip().lower().replace("-", "_")
    if mode not in SAMPLEK_LOO_VARIANCE_THRESHOLD_MODES:
        raise ValueError(
            "sample-k LOO variance threshold mode must be fixed or update0_quantile, "
            f"got {value!r}."
        )
    return mode


def apply_samplek_advantage_centering(
    advantages: torch.Tensor,
    mode: object = "none",
) -> torch.Tensor:

    mode = normalize_samplek_advantage_centering(mode)
    if mode == "none":
        return advantages
    if advantages.dim() != 3:
        raise ValueError(
            "leave_one_out sample-k advantage centering requires a [batch, sequence, K] tensor, "
            f"got shape {tuple(advantages.shape)}."
        )
    candidate_count = advantages.size(-1)
    if candidate_count < 2:
        raise ValueError("leave_one_out sample-k advantage centering requires K >= 2.")

    centered = advantages - advantages.mean(dim=-1, keepdim=True)
    return centered * (candidate_count / (candidate_count - 1))


def validate_samplek_loo_variance_filter_configuration(
    *,
    threshold: float | None,
    advantage_centering: object,
    advantages: torch.Tensor,
    candidate_estimator_weights_present: bool,
    entropy_coefficient: float,
    raw_advantage_clip: float | None,
) -> None:
    if threshold is None:
        return
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(
            "Sample-K LOO variance filter threshold must be null or a finite positive value."
        )
    if normalize_samplek_advantage_centering(advantage_centering) != "leave_one_out":
        raise ValueError("Sample-K LOO variance filtering requires leave_one_out advantage centering.")
    if advantages.dim() != 3:
        raise ValueError("Sample-K LOO variance filtering requires 3D candidate advantages.")
    if candidate_estimator_weights_present:
        raise ValueError("Sample-K LOO variance filtering requires equally weighted candidates.")
    if float(entropy_coefficient) != 0.0:
        raise ValueError("Sample-K LOO variance filtering does not support sample-k entropy advantages.")
    if raw_advantage_clip is not None:
        raise ValueError("Sample-K LOO variance filtering does not support raw advantage clipping.")


def apply_samplek_loo_variance_filter(
    *,
    centered_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    threshold: float,
    selection: str = "high",
    mode: str = "hard",
    soft_base_weight: float = 0.5,
    soft_active_bonus: float = 1.0,
    expectile_tau: float = 0.75,
) -> SampleKLOOVarianceFilterOutput:

    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("Sample-K LOO variance filter threshold must be finite and positive.")
    selection = str(selection or "high").strip().lower().replace("-", "_")
    if selection not in {"high", "low"}:
        raise ValueError(
            "Sample-K LOO variance filter selection must be high or low, "
            f"got {selection!r}."
        )
    mode = str(mode or "hard").strip().lower().replace("-", "_")
    if mode not in SAMPLEK_LOO_VARIANCE_FILTER_MODES:
        raise ValueError(
            "Sample-K LOO variance filter mode must be hard, soft, expectile, sqrt_clip, "
            "or fisher_posterior, "
            f"got {mode!r}."
        )
    soft_base_weight = float(soft_base_weight)
    soft_active_bonus = float(soft_active_bonus)
    if not math.isfinite(soft_base_weight) or soft_base_weight <= 0.0:
        raise ValueError("Sample-K LOO soft base weight must be finite and positive.")
    if not math.isfinite(soft_active_bonus) or soft_active_bonus < 0.0:
        raise ValueError("Sample-K LOO soft active bonus must be finite and nonnegative.")
    expectile_tau = float(expectile_tau)
    if not math.isfinite(expectile_tau) or not 0.5 <= expectile_tau < 1.0:
        raise ValueError("Sample-K LOO expectile tau must be finite and in [0.5, 1.0).")
    if centered_advantages.dim() != 3:
        raise ValueError(
            "Sample-K LOO variance filtering requires centered advantages with shape "
            f"[batch, sequence, K], got {tuple(centered_advantages.shape)}."
        )
    if response_mask.shape != centered_advantages.shape[:2]:
        raise ValueError(
            "response_mask shape must match the first two centered advantage dimensions, "
            f"got mask={tuple(response_mask.shape)}, advantages={tuple(centered_advantages.shape)}."
        )
    candidate_count = centered_advantages.size(-1)
    if candidate_count < 2:
        raise ValueError("Sample-K LOO variance filtering requires K >= 2.")

    with torch.no_grad():
        centered = centered_advantages.detach().float()
        mask = response_mask.to(device=centered.device, dtype=torch.float32)
        variance = compute_samplek_loo_variance(centered)
        if selection == "low":
            active_mask = (variance <= threshold).to(mask.dtype) * mask
        else:
            active_mask = (variance > threshold).to(mask.dtype) * mask
        filtered_mask = (mask - active_mask).clamp_min(0.0)
        valid_count = mask.sum()
        active_count = active_mask.sum()
        filtered_count = filtered_mask.sum()
        sqrt_clip_floor_mask = mask.new_zeros(mask.shape)
        sqrt_clip_cap_mask = mask.new_zeros(mask.shape)
        fisher_posterior_probability = mask.new_zeros(mask.shape)
        if mode == "fisher_posterior":
            fisher_posterior_degrees_of_freedom = float(candidate_count - 1)
            fisher_posterior_probability = torch.special.gammainc(
                variance.new_tensor(fisher_posterior_degrees_of_freedom / 2.0),
                fisher_posterior_degrees_of_freedom * variance / (2.0 * threshold),
            ).clamp_(0.0, 1.0)
            selected_probability = (
                1.0 - fisher_posterior_probability
                if selection == "low"
                else fisher_posterior_probability
            )
            raw_weights = (
                soft_base_weight + soft_active_bonus * selected_probability
            ) * mask
        elif mode == "sqrt_clip":
            sqrt_ratio = torch.sqrt(
                (variance + SAMPLEK_LOO_VARIANCE_SQRT_EPSILON)
                / (threshold + SAMPLEK_LOO_VARIANCE_SQRT_EPSILON)
            )
            sqrt_clip_lower_weight = soft_base_weight
            sqrt_clip_upper_weight = soft_base_weight + soft_active_bonus
            sqrt_clip_floor_mask = (sqrt_ratio <= sqrt_clip_lower_weight).to(mask.dtype) * mask
            sqrt_clip_cap_mask = (sqrt_ratio >= sqrt_clip_upper_weight).to(mask.dtype) * mask
            raw_weights = torch.clamp(
                sqrt_ratio,
                min=sqrt_clip_lower_weight,
                max=sqrt_clip_upper_weight,
            ) * mask
        elif mode == "expectile":
            raw_weights = (1.0 - expectile_tau) * mask + (2.0 * expectile_tau - 1.0) * active_mask
        elif mode == "soft":
            raw_weights = soft_base_weight * mask + soft_active_bonus * active_mask
        else:
            raw_weights = active_mask
        raw_weight_sum = raw_weights.sum()
        renormalization_scale = torch.where(
            raw_weight_sum > 0.0,
            valid_count / raw_weight_sum.clamp_min(1.0e-12),
            valid_count.new_zeros(()),
        )
        normalization_weights = raw_weights * renormalization_scale
        filtered_advantages = centered * normalization_weights.unsqueeze(-1)
        normalized_weight_sum = normalization_weights.sum()
        normalized_weight_square_sum = normalization_weights.square().sum()
        effective_sample_fraction = torch.where(
            normalized_weight_square_sum > 0.0,
            normalized_weight_sum.square()
            / (
                valid_count.clamp_min(1.0)
                * normalized_weight_square_sum.clamp_min(1.0e-12)
            ),
            valid_count.new_zeros(()),
        )
        valid_variance = _masked_values(variance, mask).float()
        if valid_variance.numel() == 0:
            variance_quantiles = variance.new_zeros(4)
        else:
            variance_quantiles = torch.quantile(
                valid_variance,
                variance.new_tensor([0.50, 0.75, 0.90, 0.95]),
            )
        valid_fisher_posterior_probability = _masked_values(
            fisher_posterior_probability, mask
        ).float()
        if valid_fisher_posterior_probability.numel() == 0:
            fisher_posterior_probability_quantiles = variance.new_zeros(3)
            fisher_posterior_transition_fraction = variance.new_zeros(())
        else:
            fisher_posterior_probability_quantiles = torch.quantile(
                valid_fisher_posterior_probability,
                variance.new_tensor([0.10, 0.50, 0.90]),
            )
            fisher_posterior_transition_fraction = (
                (valid_fisher_posterior_probability > 0.10)
                & (valid_fisher_posterior_probability < 0.90)
            ).float().mean()
        metrics = {
            "enabled": 1.0,
            "mode_soft": float(mode == "soft"),
            "mode_expectile": float(mode == "expectile"),
            "mode_sqrt_clip": float(mode == "sqrt_clip"),
            "mode_fisher_posterior": float(mode == "fisher_posterior"),
            "selection_low": float(selection == "low"),
            "threshold": threshold,
            "soft_base_weight": soft_base_weight,
            "soft_active_bonus": soft_active_bonus,
            "expectile_tau": expectile_tau,
            "sqrt_clip_epsilon": SAMPLEK_LOO_VARIANCE_SQRT_EPSILON,
            "sqrt_clip_lower_weight": soft_base_weight,
            "sqrt_clip_upper_weight": soft_base_weight + soft_active_bonus,
            "sqrt_clip_floor_fraction": (
                sqrt_clip_floor_mask.sum() / valid_count.clamp_min(1.0)
            ).item(),
            "sqrt_clip_cap_fraction": (
                sqrt_clip_cap_mask.sum() / valid_count.clamp_min(1.0)
            ).item(),
            "fisher_posterior_degrees_of_freedom": float(candidate_count - 1),
            "fisher_posterior_probability_mean": _masked_mean(
                fisher_posterior_probability, mask
            ).item(),
            "fisher_posterior_probability_p10": fisher_posterior_probability_quantiles[0].item(),
            "fisher_posterior_probability_p50": fisher_posterior_probability_quantiles[1].item(),
            "fisher_posterior_probability_p90": fisher_posterior_probability_quantiles[2].item(),
            "fisher_posterior_transition_fraction": fisher_posterior_transition_fraction.item(),
            "valid_count": valid_count.item(),
            "active_count": active_count.item(),
            "filtered_count": filtered_count.item(),
            "active_fraction": (active_count / valid_count.clamp_min(1.0)).item(),
            "filtered_fraction": (filtered_count / valid_count.clamp_min(1.0)).item(),
            "empty_active_set": float(active_count.item() == 0.0),
            "renormalization_scale": renormalization_scale.item(),
            "raw_weight_mean": _masked_mean(raw_weights, mask).item(),
            "normalized_weight_mean": _masked_mean(normalization_weights, mask).item(),
            "normalized_active_weight_mean": _masked_mean(
                normalization_weights, active_mask
            ).item(),
            "normalized_inactive_weight_mean": _masked_mean(
                normalization_weights, filtered_mask
            ).item(),
            "effective_sample_fraction": effective_sample_fraction.item(),
            "variance_mean": _masked_mean(variance, mask).item(),
            "variance_p50": variance_quantiles[0].item(),
            "variance_p75": variance_quantiles[1].item(),
            "variance_p90": variance_quantiles[2].item(),
            "variance_p95": variance_quantiles[3].item(),
            "variance_active_mean": _masked_mean(variance, active_mask).item(),
            "variance_filtered_mean": _masked_mean(variance, filtered_mask).item(),
        }

    return SampleKLOOVarianceFilterOutput(
        advantages=filtered_advantages.to(dtype=centered_advantages.dtype),
        variance=variance,
        active_mask=active_mask,
        normalization_weights=normalization_weights,
        metrics=metrics,
    )


def compute_samplek_loo_variance(centered_advantages: torch.Tensor) -> torch.Tensor:

    if centered_advantages.dim() != 3:
        raise ValueError("LOO variance requires centered advantages with shape [batch, sequence, K].")
    candidate_count = centered_advantages.size(-1)
    if candidate_count < 2:
        raise ValueError("LOO variance requires K >= 2.")
    centered = centered_advantages.detach().float()
    return ((candidate_count - 1) / candidate_count) * centered.square().mean(dim=-1)


def compute_samplek_loo_variance_histogram_diagnostics(
    *,
    variance: torch.Tensor,
    response_mask: torch.Tensor,
    log10_min: float = -8.0,
    log10_max: float = 2.0,
    num_bins: int = 100,
) -> SampleKLOOVarianceHistogramDiagnosticsOutput:

    if variance.dim() != 2:
        raise ValueError("LOO variance histogram requires variance with shape [batch, sequence].")
    if response_mask.shape != variance.shape:
        raise ValueError("response_mask shape must match the LOO variance tensor.")
    log10_min = float(log10_min)
    log10_max = float(log10_max)
    num_bins = int(num_bins)
    if not math.isfinite(log10_min) or not math.isfinite(log10_max) or log10_min >= log10_max:
        raise ValueError("LOO variance histogram log bounds must be finite and increasing.")
    if num_bins < 1:
        raise ValueError("LOO variance histogram num_bins must be positive.")

    with torch.no_grad():
        valid = variance.detach().float()[response_mask.to(device=variance.device).bool()]
        if valid.numel() == 0:
            raise ValueError("LOO variance histogram requires at least one valid response token.")
        if not torch.isfinite(valid).all():
            raise ValueError("LOO variance histogram requires finite variance values.")
        if (valid < 0.0).any():
            raise ValueError("LOO variance histogram requires nonnegative variance values.")

        bin_edges = torch.logspace(
            log10_min,
            log10_max,
            num_bins + 1,
            device=valid.device,
            dtype=torch.float32,
        )
        zero_mask = valid == 0.0
        positive = valid[~zero_mask]
        underflow_count = (positive < bin_edges[0]).sum()
        overflow_count = (positive > bin_edges[-1]).sum()
        in_range = positive[(positive >= bin_edges[0]) & (positive <= bin_edges[-1])]
        if in_range.numel() == 0:
            bin_counts = torch.zeros(num_bins, device=valid.device, dtype=torch.float32)
        else:
            bin_indices = torch.searchsorted(bin_edges, in_range, right=True) - 1
            bin_indices = bin_indices.clamp(min=0, max=num_bins - 1)
            bin_counts = torch.bincount(bin_indices, minlength=num_bins).float()

        valid_count = float(valid.numel())
        zero_count = zero_mask.sum().float()
        positive_count = (~zero_mask).sum().float()
        metrics = {
            "enabled": 1.0,
            "valid_count": valid_count,
            "zero_count": zero_count.item(),
            "zero_fraction": (zero_count / valid_count).item(),
            "positive_count": positive_count.item(),
            "positive_fraction": (positive_count / valid_count).item(),
            "underflow_count": underflow_count.float().item(),
            "underflow_fraction": (underflow_count.float() / valid_count).item(),
            "overflow_count": overflow_count.float().item(),
            "overflow_fraction": (overflow_count.float() / valid_count).item(),
            "in_range_count": bin_counts.sum().item(),
            "in_range_fraction": (bin_counts.sum() / valid_count).item(),
            "log10_min": log10_min,
            "log10_max": log10_max,
            "num_bins": float(num_bins),
        }
        for index, count in enumerate(bin_counts):
            metrics[f"bin_{index:03d}_count"] = count.item()
            metrics[f"bin_{index:03d}_fraction"] = (count / valid_count).item()

    return SampleKLOOVarianceHistogramDiagnosticsOutput(
        bin_edges=bin_edges,
        bin_counts=bin_counts,
        metrics=metrics,
    )


def compute_samplek_loo_variance_quantile_threshold(
    *,
    centered_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    quantile: float,
) -> float:

    quantile = float(quantile)
    if not math.isfinite(quantile) or not 0.0 < quantile < 1.0:
        raise ValueError("Sample-K LOO variance quantile must be finite and in (0, 1).")
    variance = compute_samplek_loo_variance(centered_advantages)
    if response_mask.shape != variance.shape:
        raise ValueError(
            "response_mask shape must match the LOO variance shape, "
            f"got mask={tuple(response_mask.shape)}, variance={tuple(variance.shape)}."
        )
    valid_variance = variance[response_mask.to(device=variance.device).bool()]
    if valid_variance.numel() == 0:
        raise ValueError("Sample-K LOO variance quantile requires at least one valid response token.")
    threshold = torch.quantile(valid_variance.float(), quantile).item()
    return max(float(threshold), SAMPLEK_LOO_VARIANCE_QUANTILE_MIN_THRESHOLD)


def compute_samplek_update0_loo_variance_quantile_threshold(
    *,
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    quantile: float,
) -> float:

    if teacher_log_probs.shape != student_log_probs.shape:
        raise ValueError(
            "update0 quantile teacher and student log-probability shapes must match, "
            f"got teacher={tuple(teacher_log_probs.shape)}, student={tuple(student_log_probs.shape)}."
        )
    centered_advantages = apply_samplek_advantage_centering(
        teacher_log_probs - student_log_probs.detach(),
        mode="leave_one_out",
    )
    return compute_samplek_loo_variance_quantile_threshold(
        centered_advantages=centered_advantages,
        response_mask=response_mask,
        quantile=quantile,
    )


def compute_samplek_loo_variance_ratio_diagnostics(
    *,
    current_variance: torch.Tensor,
    update0_variance: torch.Tensor,
    response_mask: torch.Tensor,
    absolute_threshold: float,
    epsilon: float = 1e-12,
) -> SampleKLOOVarianceRatioDiagnosticsOutput:

    if current_variance.dim() != 2 or update0_variance.shape != current_variance.shape:
        raise ValueError("current and update0 LOO variance must have the same [batch, sequence] shape.")
    if response_mask.shape != current_variance.shape:
        raise ValueError("response_mask shape must match the LOO variance tensors.")
    absolute_threshold = float(absolute_threshold)
    epsilon = float(epsilon)
    if not math.isfinite(absolute_threshold) or absolute_threshold <= 0.0:
        raise ValueError("absolute_threshold must be finite and positive.")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive.")

    with torch.no_grad():
        current = current_variance.detach().float()
        update0 = update0_variance.detach().float()
        mask = response_mask.to(device=current.device, dtype=torch.float32)
        ratio = (current + epsilon) / (update0 + epsilon)
        log_ratio = ratio.clamp_min(epsilon).log()

        update0_active = (update0 > absolute_threshold).to(mask.dtype) * mask
        current_active = (current > absolute_threshold).to(mask.dtype) * mask
        retained = update0_active * current_active
        entered = (mask - update0_active).clamp_min(0.0) * current_active
        exited = update0_active * (mask - current_active).clamp_min(0.0)
        active_union = ((update0_active + current_active) > 0.0).to(mask.dtype) * mask

        valid_count = mask.sum()
        update0_active_count = update0_active.sum()
        current_active_count = current_active.sum()
        retained_count = retained.sum()
        union_count = active_union.sum()
        stable_ratio = current / update0.clamp_min(absolute_threshold)
        metrics = {
            "enabled": 1.0,
            "absolute_threshold": absolute_threshold,
            "valid_count": valid_count.item(),
            "update0_variance_mean": _masked_mean(update0, mask).item(),
            "current_variance_mean": _masked_mean(current, mask).item(),
            "variance_pearson": _masked_pearson(current, update0, mask).item(),
            "variance_pearson_baseline_active": _masked_pearson(
                current, update0, update0_active
            ).item(),
            "ratio_mean_baseline_active": _masked_mean(ratio, update0_active).item(),
            "ratio_p10_baseline_active": _masked_quantile(ratio, update0_active, 0.10).item(),
            "ratio_p25_baseline_active": _masked_quantile(ratio, update0_active, 0.25).item(),
            "ratio_p50_baseline_active": _masked_quantile(ratio, update0_active, 0.50).item(),
            "ratio_p75_baseline_active": _masked_quantile(ratio, update0_active, 0.75).item(),
            "ratio_p90_baseline_active": _masked_quantile(ratio, update0_active, 0.90).item(),
            "log_ratio_mean_baseline_active": _masked_mean(log_ratio, update0_active).item(),
            "log_ratio_p10_baseline_active": _masked_quantile(
                log_ratio, update0_active, 0.10
            ).item(),
            "log_ratio_p50_baseline_active": _masked_quantile(
                log_ratio, update0_active, 0.50
            ).item(),
            "log_ratio_p90_baseline_active": _masked_quantile(
                log_ratio, update0_active, 0.90
            ).item(),
            "stable_ratio_mean_all_valid": _masked_mean(stable_ratio, mask).item(),
            "update0_active_fraction": (update0_active_count / valid_count.clamp_min(1.0)).item(),
            "current_active_fraction": (current_active_count / valid_count.clamp_min(1.0)).item(),
            "retained_fraction_of_update0_active": (
                retained_count / update0_active_count.clamp_min(1.0)
            ).item(),
            "entered_fraction": (entered.sum() / valid_count.clamp_min(1.0)).item(),
            "exited_fraction": (exited.sum() / valid_count.clamp_min(1.0)).item(),
            "active_jaccard": torch.where(
                union_count > 0.0,
                retained_count / union_count.clamp_min(1.0),
                union_count.new_ones(()),
            ).item(),
        }
        for candidate_threshold, label in ((0.9, "0p9"), (0.8, "0p8"), (0.7, "0p7"), (0.5, "0p5")):
            kept = (ratio >= candidate_threshold).to(mask.dtype) * update0_active
            metrics[f"ratio_ge_{label}_fraction_baseline_active"] = (
                kept.sum() / update0_active_count.clamp_min(1.0)
            ).item()

    return SampleKLOOVarianceRatioDiagnosticsOutput(
        ratio=ratio,
        log_ratio=log_ratio,
        metrics=metrics,
    )


def apply_raw_opd_advantage_clip(
    *,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip: float,
) -> RawOPDAdvantageClipOutput:

    if advantages.dim() not in {2, 3}:
        raise ValueError(
            "raw OPD advantage clipping requires a 2D sampled-token or 3D Sample-K tensor, "
            f"got shape {tuple(advantages.shape)}."
        )
    if response_mask.dim() != 2 or response_mask.shape != advantages.shape[:2]:
        raise ValueError(
            "raw OPD advantage clipping response_mask must match the advantage batch and sequence dimensions, "
            f"got mask={tuple(response_mask.shape)}, advantages={tuple(advantages.shape)}."
        )
    clip = float(clip)
    if not math.isfinite(clip) or clip <= 0.0:
        raise ValueError(f"raw OPD advantage clip must be finite and positive, got {clip}.")

    with torch.no_grad():
        raw = advantages.detach().float()
        mask = response_mask.to(device=raw.device, dtype=torch.float32)
        if raw.dim() == 3:
            mask = mask.unsqueeze(-1).expand_as(raw)
        clipped = raw.clamp(min=-clip, max=clip)
        clipped_mask = raw.abs() > clip
        active_count = mask.sum()
        clip_count = (clipped_mask.to(mask.dtype) * mask).sum()
        metrics = {
            "enabled": 1.0,
            "clip": clip,
            "active_count": active_count.item(),
            "clip_count": clip_count.item(),
            "clip_fraction": _masked_mean(clipped_mask.to(mask.dtype), mask).item(),
            "pre_mean": _masked_mean(raw, mask).item(),
            "pre_abs_mean": _masked_mean(raw.abs(), mask).item(),
            "pre_abs_max": _masked_abs_max(raw, mask).item(),
            "post_mean": _masked_mean(clipped, mask).item(),
            "post_abs_mean": _masked_mean(clipped.abs(), mask).item(),
            "post_abs_max": _masked_abs_max(clipped, mask).item(),
        }

    return RawOPDAdvantageClipOutput(
        advantages=clipped.to(dtype=advantages.dtype),
        metrics=metrics,
    )


def validate_adaptive_head_tail_negative_elu_configuration(
    *,
    candidate_mode: str,
    advantage_mode: str,
    advantages: torch.Tensor,
    candidate_estimator_weights_present: bool,
    sample_k_kl_plus_one: bool,
    raw_advantage_clip: float | None,
) -> None:
    if candidate_mode != "adaptive_head_tail":
        raise ValueError(
            "adaptive head-tail negative ELU requires log_prob_candidate_mode=adaptive_head_tail."
        )
    if advantage_mode not in CURRENT_TEACHER_LOG_PROB_MODES:
        raise ValueError(
            "adaptive head-tail negative ELU requires opd_advantage_mode=current_kl(_is)."
        )
    if advantages.dim() != 3:
        raise ValueError("adaptive head-tail negative ELU requires 3D Sample-K advantages.")
    if not candidate_estimator_weights_present:
        raise ValueError("adaptive head-tail negative ELU requires adaptive estimator weights.")
    if sample_k_kl_plus_one:
        raise ValueError("adaptive head-tail negative ELU requires sample-K plus-one to be disabled.")
    if raw_advantage_clip is not None:
        raise ValueError("adaptive head-tail negative ELU cannot be combined with raw advantage clipping.")


def apply_adaptive_head_tail_negative_elu(
    *,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    threshold: float,
    tau: float,
) -> AdaptiveHeadTailNegativeELUOutput:

    if advantages.dim() != 3:
        raise ValueError(
            "adaptive head-tail negative ELU requires advantages with shape [batch, sequence, K], "
            f"got {tuple(advantages.shape)}."
        )
    if response_mask.dim() != 2 or response_mask.shape != advantages.shape[:2]:
        raise ValueError(
            "adaptive head-tail negative ELU response_mask must match the advantage batch and sequence "
            f"dimensions, got mask={tuple(response_mask.shape)}, advantages={tuple(advantages.shape)}."
        )
    threshold = float(threshold)
    tau = float(tau)
    if not math.isfinite(threshold):
        raise ValueError(f"adaptive head-tail negative ELU threshold must be finite, got {threshold}.")
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"adaptive head-tail negative ELU tau must be finite and positive, got {tau}.")

    raw = advantages.float()
    scaled_delta = ((raw - threshold) / tau).clamp_max(0.0)
    softened_tail = threshold + tau * torch.expm1(scaled_delta)
    transformed = torch.where(raw >= threshold, raw, softened_tail)

    with torch.no_grad():
        raw_metrics = raw.detach()
        transformed_metrics = transformed.detach()
        mask = response_mask.to(device=raw.device, dtype=torch.float32).unsqueeze(-1).expand_as(raw)
        transformed_mask = raw_metrics < threshold
        valid_pre = _masked_values(raw_metrics, mask)
        valid_post = _masked_values(transformed_metrics, mask)
        active_count = mask.sum()
        transformed_count = (transformed_mask.to(mask.dtype) * mask).sum()
        metrics = {
            "enabled": 1.0,
            "threshold": threshold,
            "tau": tau,
            "lower_bound": threshold - tau,
            "active_count": active_count.item(),
            "transformed_count": transformed_count.item(),
            "transformed_fraction": _masked_mean(
                transformed_mask.to(mask.dtype), mask
            ).item(),
            "pre_min": valid_pre.min().item() if valid_pre.numel() else 0.0,
            "post_min": valid_post.min().item() if valid_post.numel() else 0.0,
            "pre_negative_abs_mean": _masked_mean(
                (-raw_metrics).clamp_min(0.0), mask
            ).item(),
            "post_negative_abs_mean": _masked_mean(
                (-transformed_metrics).clamp_min(0.0), mask
            ).item(),
        }

    return AdaptiveHeadTailNegativeELUOutput(
        advantages=transformed.to(dtype=advantages.dtype),
        metrics=metrics,
    )


def apply_samplek_influence_clip(
    *,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    prefix_weights: torch.Tensor | None,
    clip: float,
) -> SampleKInfluenceClipOutput:

    if advantages.dim() != 3:
        raise ValueError(
            "sample-k influence clipping requires [batch, sequence, K] advantages, "
            f"got shape {tuple(advantages.shape)}."
        )
    if response_mask.shape != advantages.shape[:2]:
        raise ValueError(
            "sample-k influence clipping response_mask must match the candidate batch and sequence dimensions, "
            f"got mask={tuple(response_mask.shape)}, candidates={tuple(advantages.shape)}."
        )
    if prefix_weights is not None and prefix_weights.shape != response_mask.shape:
        raise ValueError(
            "sample-k influence clipping prefix_weights must match response_mask, "
            f"got weights={tuple(prefix_weights.shape)}, mask={tuple(response_mask.shape)}."
        )
    clip = float(clip)
    if not math.isfinite(clip) or clip <= 0.0:
        raise ValueError(f"sample-k influence clip must be finite and positive, got {clip}.")

    with torch.no_grad():
        weighted = advantages.detach().float()
        if prefix_weights is not None:
            prefix = prefix_weights.to(device=weighted.device).detach().float()
            weighted = weighted * prefix.unsqueeze(-1)
        mask = response_mask.to(device=weighted.device, dtype=torch.float32).unsqueeze(-1).expand_as(weighted)
        clipped = weighted.clamp(min=-clip, max=clip)
        clipped_mask = weighted.abs() > clip
        active_count = mask.sum()
        clip_count = (clipped_mask.to(mask.dtype) * mask).sum()
        metrics = {
            "enabled": 1.0,
            "clip": clip,
            "prefix_weight_enabled": float(prefix_weights is not None),
            "active_count": active_count.item(),
            "clip_count": clip_count.item(),
            "clip_fraction": _masked_mean(clipped_mask.to(mask.dtype), mask).item(),
            "pre_mean": _masked_mean(weighted, mask).item(),
            "pre_abs_mean": _masked_mean(weighted.abs(), mask).item(),
            "pre_abs_p95": _masked_quantile(weighted.abs(), mask, 0.95).item(),
            "pre_abs_p99": _masked_quantile(weighted.abs(), mask, 0.99).item(),
            "pre_abs_max": _masked_abs_max(weighted, mask).item(),
            "post_mean": _masked_mean(clipped, mask).item(),
            "post_abs_mean": _masked_mean(clipped.abs(), mask).item(),
            "post_abs_p95": _masked_quantile(clipped.abs(), mask, 0.95).item(),
            "post_abs_p99": _masked_quantile(clipped.abs(), mask, 0.99).item(),
            "post_abs_max": _masked_abs_max(clipped, mask).item(),
        }

    return SampleKInfluenceClipOutput(
        advantages=clipped.to(dtype=advantages.dtype),
        metrics=metrics,
    )


def validate_samplek_influence_clip_configuration(
    *,
    advantage_mode: object,
    advantages: torch.Tensor,
    no_candidate_is: bool,
    force_candidate_is: bool,
    candidate_estimator_weights_present: bool,
    source_normalize_enabled: bool,
) -> None:

    mode = normalize_opd_mode(advantage_mode)
    if mode not in CURRENT_TEACHER_LOG_PROB_MODES:
        raise ValueError(
            "sample-k influence clipping requires opd_advantage_mode=current_kl or current_kl_is."
        )
    if advantages.dim() != 3:
        raise ValueError(
            "sample-k influence clipping requires 3D candidate advantages, "
            f"got shape {tuple(advantages.shape)}."
        )
    if not no_candidate_is or force_candidate_is:
        raise ValueError(
            "sample-k influence clipping requires current-policy candidates with no candidate importance ratio."
        )
    if candidate_estimator_weights_present:
        raise ValueError("sample-k influence clipping requires equally weighted candidates.")
    if source_normalize_enabled:
        raise ValueError("sample-k influence clipping does not support Q-mixture source normalization.")


def is_decomposed_pi_old_mode(value: object) -> bool:
    return normalize_opd_mode(value) in DECOMPOSED_PI_OLD_MODES


def normalize_q_mixture_teacher_advantage_mode(value: object) -> str:
    mode = str(value or "proposal").strip().lower().replace("-", "_")
    aliases = {
        "q": "proposal",
        "q_teacher": "proposal",
        "q_proposal": "proposal",
        "direct": "current",
        "current_teacher": "current",
        "current_kl": "current",
        "fixed_alpha": "fixed_alpha_current",
        "fixed_alpha_mixture": "fixed_alpha_current",
        "smooth": "fixed_alpha_current",
        "smooth_current": "fixed_alpha_current",
    }
    mode = aliases.get(mode, mode)
    if mode not in Q_MIXTURE_TEACHER_ADVANTAGE_MODE_IDS:
        raise ValueError(
            "Q-mixture teacher advantage mode must be proposal, current, or "
            f"fixed_alpha_current, got {value!r}."
        )
    return mode


def normalize_prefix_is_mode(value: object) -> str:
    mode = str(value or "cumulative_cap").strip().lower().replace("-", "_")
    aliases = {
        "cap": "cumulative_cap",
        "upper_cap": "cumulative_cap",
        "cumulative": "cumulative_cap",
        "hard_clip": "hard_window",
        "mask": "hard_window",
        "geometric": "prefix_geometric",
        "geometric_mean": "prefix_geometric",
        "gspo": "prefix_geometric",
    }
    mode = aliases.get(mode, mode)
    if mode not in PREFIX_IS_MODES:
        raise ValueError(f"prefix_is_mode must be one of {sorted(PREFIX_IS_MODES)}, got {value!r}.")
    return mode


def requires_teacher_on_student_log_probs(value: object) -> bool:
    mode = normalize_opd_mode(value)
    return mode in CURRENT_TEACHER_LOG_PROB_MODES or mode in DECOMPOSED_PI_OLD_MODES


def normalize_teacher_temperature_anneal_schedule(value: object) -> str:
    schedule = str(value or "linear").strip().lower().replace("-", "_")
    schedule = {"cos": "cosine", "half_cosine": "cosine"}.get(schedule, schedule)
    if schedule not in {"linear", "cosine"}:
        raise ValueError(
            "teacher temperature anneal schedule must be linear or cosine, "
            f"got {value!r}."
        )
    return schedule


def compute_teacher_temperature_anneal_progress(*, step: int, anneal_steps: int) -> float:

    if isinstance(step, bool) or int(step) != step or int(step) < 0:
        raise ValueError("teacher temperature schedule step must be a nonnegative integer.")
    if isinstance(anneal_steps, bool) or int(anneal_steps) != anneal_steps or int(anneal_steps) <= 0:
        raise ValueError("teacher temperature anneal_steps must be a positive integer.")
    step = int(step)
    anneal_steps = int(anneal_steps)
    if anneal_steps == 1:
        return float(step >= 1)
    return min(max(step - 1, 0), anneal_steps - 1) / (anneal_steps - 1)


def compute_annealed_teacher_temperature(
    *,
    step: int,
    initial_temperature: float,
    minimum_temperature: float,
    anneal_steps: int,
    schedule: str = "linear",
) -> float:

    initial_temperature = float(initial_temperature)
    minimum_temperature = float(minimum_temperature)
    if not math.isfinite(initial_temperature) or initial_temperature <= 0.0:
        raise ValueError("initial teacher temperature must be finite and positive.")
    if not math.isfinite(minimum_temperature) or minimum_temperature <= 0.0:
        raise ValueError("minimum teacher temperature must be finite and positive.")
    if initial_temperature < minimum_temperature:
        raise ValueError("initial teacher temperature must be greater than or equal to the minimum.")

    progress = compute_teacher_temperature_anneal_progress(step=step, anneal_steps=anneal_steps)
    schedule = normalize_teacher_temperature_anneal_schedule(schedule)
    if schedule == "cosine":
        remaining_fraction = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        remaining_fraction = 1.0 - progress
    return minimum_temperature + (initial_temperature - minimum_temperature) * remaining_fraction


def validate_teacher_temperature_configuration(
    *,
    teacher_temperature: float,
    anneal_enable: bool,
    minimum_temperature: float,
    anneal_steps: int,
    teacher_rollout_mix_enable: bool,
    schedule: str = "linear",
) -> None:

    teacher_temperature = float(teacher_temperature)
    if not math.isfinite(teacher_temperature) or teacher_temperature <= 0.0:
        raise ValueError("teacher_temperature must be finite and positive.")

    if teacher_rollout_mix_enable and (anneal_enable or not math.isclose(teacher_temperature, 1.0)):
        raise ValueError(
            "Teacher rollout mixing cannot be combined with a modified or annealed teacher temperature. "
            "This combination requires a new branch with separate behavior-policy and target-supervision "
            "teacher log-probabilities."
        )

    if anneal_enable:
        compute_annealed_teacher_temperature(
            step=0,
            initial_temperature=teacher_temperature,
            minimum_temperature=minimum_temperature,
            anneal_steps=anneal_steps,
            schedule=schedule,
        )


def _check_shape(name: str, value: torch.Tensor, expected: torch.Size) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} shape must be {tuple(expected)}, got {tuple(value.shape)}.")


def _masked_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask > 0.5]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_abs_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = _masked_values(values.abs(), mask)
    if valid.numel() == 0:
        return values.new_tensor(0.0)
    return valid.max()


def _masked_quantile(values: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    valid = _masked_values(values, mask)
    if valid.numel() == 0:
        return values.new_tensor(0.0)
    return torch.quantile(valid.float(), q)


def _normalized_ess(weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = _masked_values(weights.float(), mask)
    if valid.numel() == 0:
        return weights.new_tensor(1.0, dtype=torch.float32)
    return (valid.sum().square() / (valid.square().sum().clamp_min(1e-12) * valid.numel())).clamp(min=0.0, max=1.0)


def _masked_pearson(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_mask = mask > 0.5
    first_valid = first.float()[valid_mask]
    second_valid = second.float()[valid_mask]
    if first_valid.numel() < 2:
        return first.new_tensor(0.0, dtype=torch.float32)
    first_centered = first_valid - first_valid.mean()
    second_centered = second_valid - second_valid.mean()
    denominator = torch.sqrt(first_centered.square().sum() * second_centered.square().sum())
    if denominator <= 1e-12:
        return first.new_tensor(0.0, dtype=torch.float32)
    return (first_centered * second_centered).sum() / denominator


def add_samplek_entropy_advantages(
    *,
    teacher_advantages: torch.Tensor,
    current_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    coefficient: float,
) -> SampleKEntropyAdvantageOutput:

    if teacher_advantages.dim() != 3:
        raise ValueError(
            "sample-k entropy expects teacher_advantages with shape "
            f"(batch, response_length, K), got {tuple(teacher_advantages.shape)}."
        )
    _check_shape("current_log_probs", current_log_probs, teacher_advantages.shape)
    if response_mask.shape != teacher_advantages.shape[:2]:
        raise ValueError(
            "sample-k entropy response_mask must match the candidate batch and response dimensions, "
            f"got mask={tuple(response_mask.shape)}, candidates={tuple(teacher_advantages.shape)}."
        )
    coefficient = float(coefficient)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError(f"sample-k entropy coefficient must be finite and nonnegative, got {coefficient}.")

    with torch.no_grad():
        teacher = teacher_advantages.detach().float()
        current = current_log_probs.detach().float()
        entropy_advantages = -coefficient * current
        combined = teacher + entropy_advantages
        mask = response_mask.to(device=current.device, dtype=torch.float32).unsqueeze(-1).expand_as(current)
        teacher_abs_mean = _masked_mean(teacher.abs(), mask)
        entropy_abs_mean = _masked_mean(entropy_advantages.abs(), mask)
        metrics = {
            "coefficient": coefficient,
            "entropy_estimate": _masked_mean(-current, mask).item(),
            "entropy_advantage_mean": _masked_mean(entropy_advantages, mask).item(),
            "entropy_advantage_abs_mean": entropy_abs_mean.item(),
            "teacher_advantage_abs_mean": teacher_abs_mean.item(),
            "combined_advantage_abs_mean": _masked_mean(combined.abs(), mask).item(),
            "entropy_to_teacher_abs_ratio": (
                entropy_abs_mean / teacher_abs_mean.clamp_min(1e-12)
            ).item(),
        }

    return SampleKEntropyAdvantageOutput(
        advantages=combined.to(dtype=teacher_advantages.dtype),
        entropy_advantages=entropy_advantages.to(dtype=teacher_advantages.dtype),
        metrics=metrics,
    )


def compute_sampled_token_rkl_diagnostics(
    *,
    current_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> SampledTokenRKLDiagnosticsOutput:

    if current_log_probs.dim() != 2:
        raise ValueError(
            "sampled-token RKL diagnostics expect current_log_probs with shape "
            f"(batch, response_length), got {tuple(current_log_probs.shape)}."
        )
    expected_shape = current_log_probs.shape
    _check_shape("reference_log_probs", reference_log_probs, expected_shape)
    _check_shape("response_mask", response_mask, expected_shape)

    mask = response_mask.to(device=current_log_probs.device, dtype=torch.float32)
    with torch.no_grad():
        current = current_log_probs.detach().float()
        reference = reference_log_probs.to(device=current.device).detach().float()
        log_ratio = (current - reference) * mask
        numerical_clip_mask = (log_ratio.abs() > _NUMERICAL_LOG_RATIO_BOUND).to(mask.dtype)
        ratio = torch.exp(log_ratio.clamp(min=-_NUMERICAL_LOG_RATIO_BOUND, max=_NUMERICAL_LOG_RATIO_BOUND))
        k3 = (ratio * log_ratio - (ratio - 1.0)) * mask
        k3_mean = _masked_mean(k3, mask)

    return SampledTokenRKLDiagnosticsOutput(
        log_ratio=log_ratio,
        ratio=ratio,
        k3=k3,
        numerical_clip_mask=numerical_clip_mask,
        k3_mean=k3_mean,
    )


def compute_q_mixture_source_normalization_weights(
    *,
    response_mask: torch.Tensor,
    student_token_mask: torch.Tensor,
    mixture_alpha: float | torch.Tensor,
    teacher_loss_lambda: float | None = None,
    samplek_loss_coefficient: float = 1.0,
) -> QMixtureSourceNormalizationOutput:

    if response_mask.dim() != 2:
        raise ValueError(
            "Q-mixture source normalization expects response_mask with shape "
            f"(batch, response_length), got {tuple(response_mask.shape)}."
        )
    _check_shape("student_token_mask", student_token_mask, response_mask.shape)

    device = response_mask.device
    mask = response_mask.to(device=device, dtype=torch.float32)
    student_mask = student_token_mask.to(device=device, dtype=torch.float32)
    if not torch.isfinite(mask).all() or not torch.isfinite(student_mask).all():
        raise ValueError("Q-mixture source-normalization masks must be finite.")
    if (mask < 0.0).any() or (student_mask < 0.0).any() or (student_mask > mask + 1e-6).any():
        raise ValueError("student_token_mask must be nonnegative and contained in response_mask.")
    teacher_mask = (mask - student_mask).clamp_min(0.0)

    alpha = torch.as_tensor(mixture_alpha, device=device, dtype=torch.float32)
    if alpha.ndim == 0:
        alpha = alpha.reshape(1).expand(mask.shape[0])
    elif alpha.ndim == 2 and alpha.shape == (mask.shape[0], 1):
        alpha = alpha.squeeze(-1)
    elif alpha.ndim != 1 or alpha.shape[0] != mask.shape[0]:
        raise ValueError(
            "mixture_alpha must be scalar, (batch,), or (batch, 1), got "
            f"{tuple(alpha.shape)} for batch size {mask.shape[0]}."
        )
    if not torch.isfinite(alpha).all() or (alpha < 0.0).any() or (alpha > 1.0).any():
        raise ValueError("mixture_alpha values must be finite and within [0, 1].")

    valid_rows = mask.sum(dim=-1) > 0.0
    if not valid_rows.any():
        raise ValueError("Q-mixture source normalization requires at least one response token.")
    valid_alpha = alpha[valid_rows]
    if (valid_alpha.max() - valid_alpha.min()).item() > 1e-6:
        raise ValueError(
            "Q-mixture source normalization currently requires one constant mixture_alpha "
            "across all nonempty trajectories."
        )
    source_alpha = valid_alpha.mean()
    if teacher_loss_lambda is None:
        teacher_coefficient = 1.0 - source_alpha
        teacher_relative_lambda = teacher_coefficient / source_alpha.clamp_min(1e-12)
    else:
        teacher_loss_lambda = float(teacher_loss_lambda)
        if not math.isfinite(teacher_loss_lambda) or teacher_loss_lambda < 0.0:
            raise ValueError("teacher_loss_lambda must be finite and nonnegative.")
        if source_alpha.item() <= 0.0:
            raise ValueError("teacher_loss_lambda requires a positive mixture_alpha.")
        teacher_relative_lambda = source_alpha.new_tensor(teacher_loss_lambda)
        teacher_coefficient = source_alpha * teacher_relative_lambda

    samplek_loss_coefficient = float(samplek_loss_coefficient)
    if not math.isfinite(samplek_loss_coefficient) or samplek_loss_coefficient < 0.0:
        raise ValueError("samplek_loss_coefficient must be finite and nonnegative.")
    student_coefficient = source_alpha * samplek_loss_coefficient
    teacher_coefficient = teacher_coefficient * samplek_loss_coefficient

    student_tokens = student_mask.sum()
    teacher_tokens = teacher_mask.sum()
    if student_tokens.item() <= 0.0 or teacher_tokens.item() <= 0.0:
        raise ValueError("Q-mixture source normalization requires both student and teacher response tokens.")
    total_tokens = student_tokens + teacher_tokens
    student_weight = student_coefficient * total_tokens / student_tokens
    teacher_weight = teacher_coefficient * total_tokens / teacher_tokens
    weights = student_mask * student_weight + teacher_mask * teacher_weight
    weighted_mass = (weights * mask).sum().clamp_min(1e-12)

    metrics = {
        "alpha": source_alpha.item(),
        "strict_mixture_coefficients": float(teacher_loss_lambda is None),
        "samplek_loss_coefficient": samplek_loss_coefficient,
        "student_loss_coefficient": student_coefficient.item(),
        "teacher_loss_coefficient": teacher_coefficient.item(),
        "teacher_relative_lambda": teacher_relative_lambda.item(),
        "total_loss_coefficient": (student_coefficient + teacher_coefficient).item(),
        "student_token_count": student_tokens.item(),
        "teacher_token_count": teacher_tokens.item(),
        "student_token_fraction": (student_tokens / total_tokens).item(),
        "teacher_token_fraction": (teacher_tokens / total_tokens).item(),
        "student_per_token_weight": student_weight.item(),
        "teacher_per_token_weight": teacher_weight.item(),
        "student_weighted_mass": ((weights * student_mask).sum() / total_tokens).item(),
        "teacher_weighted_mass": ((weights * teacher_mask).sum() / total_tokens).item(),
        "student_weighted_mass_fraction": ((weights * student_mask).sum() / weighted_mass).item(),
        "teacher_weighted_mass_fraction": ((weights * teacher_mask).sum() / weighted_mass).item(),
        "weight_mean": _masked_mean(weights, mask).item(),
    }
    return QMixtureSourceNormalizationOutput(weights=weights, metrics=metrics)


def _position_bucket_metrics(
    *,
    prefix_log_ratio: torch.Tensor,
    prefix_is_weights: torch.Tensor,
    cap_mask: torch.Tensor,
    keep_mask: torch.Tensor,
    lower_reject_mask: torch.Tensor,
    upper_reject_mask: torch.Tensor,
    response_mask: torch.Tensor,
    num_buckets: int = 4,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    response_length = response_mask.shape[-1]
    for bucket_idx in range(num_buckets):
        start = response_length * bucket_idx // num_buckets
        end = response_length * (bucket_idx + 1) // num_buckets
        bucket_mask = torch.zeros_like(response_mask)
        bucket_mask[:, start:end] = response_mask[:, start:end]
        bucket_name = f"position_q{bucket_idx + 1}"
        metrics[f"{bucket_name}/weight_mean"] = _masked_mean(prefix_is_weights, bucket_mask).item()
        metrics[f"{bucket_name}/weight_ess"] = _normalized_ess(prefix_is_weights, bucket_mask).item()
        metrics[f"{bucket_name}/cap_fraction"] = _masked_mean(cap_mask, bucket_mask).item()
        metrics[f"{bucket_name}/keep_fraction"] = _masked_mean(keep_mask, bucket_mask).item()
        metrics[f"{bucket_name}/lower_reject_fraction"] = _masked_mean(lower_reject_mask, bucket_mask).item()
        metrics[f"{bucket_name}/upper_reject_fraction"] = _masked_mean(upper_reject_mask, bucket_mask).item()
        metrics[f"{bucket_name}/prefix_log_ratio_abs_mean"] = _masked_mean(prefix_log_ratio.abs(), bucket_mask).item()
    return metrics


def compute_trajectory_mixture_proposal(
    *,
    old_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    mixture_alpha: float | torch.Tensor,
) -> TrajectoryMixtureProposalOutput:

    if old_log_probs.dim() != 2:
        raise ValueError(
            "trajectory-mixture proposal expects old_log_probs with shape "
            f"(batch, response_length), got {tuple(old_log_probs.shape)}."
        )
    expected_shape = old_log_probs.shape
    _check_shape("teacher_log_probs", teacher_log_probs, expected_shape)
    _check_shape("response_mask", response_mask, expected_shape)

    device = old_log_probs.device
    mask = response_mask.to(device=device, dtype=torch.float32)
    alpha = torch.as_tensor(mixture_alpha, device=device, dtype=torch.float32)
    if alpha.ndim == 0:
        alpha = alpha.reshape(1, 1).expand(expected_shape[0], 1)
    elif alpha.ndim == 1 and alpha.shape[0] == expected_shape[0]:
        alpha = alpha.unsqueeze(-1)
    elif alpha.ndim == 2 and alpha.shape == (expected_shape[0], 1):
        pass
    else:
        raise ValueError(
            "mixture_alpha must be scalar, (batch,), or (batch, 1), got "
            f"{tuple(alpha.shape)} for batch size {expected_shape[0]}."
        )
    if not torch.isfinite(alpha).all() or (alpha < 0.0).any() or (alpha > 1.0).any():
        raise ValueError("mixture_alpha values must be finite and within [0, 1].")

    old = old_log_probs.detach().to(device=device, dtype=torch.float32) * mask
    teacher = teacher_log_probs.detach().to(device=device, dtype=torch.float32) * mask
    old_prefix = old.cumsum(dim=-1)
    teacher_prefix = teacher.cumsum(dim=-1)
    zeros = torch.zeros((expected_shape[0], 1), device=device, dtype=torch.float32)
    old_prefix_before = torch.cat([zeros, old_prefix[:, :-1]], dim=-1)
    teacher_prefix_before = torch.cat([zeros, teacher_prefix[:, :-1]], dim=-1)

    negative_inf = torch.full_like(alpha, -torch.inf)
    log_alpha = torch.where(alpha > 0.0, alpha.log(), negative_inf)
    log_one_minus_alpha = torch.where(alpha < 1.0, torch.log1p(-alpha), negative_inf)

    prefix_log_probs = torch.logaddexp(
        log_alpha + old_prefix,
        log_one_minus_alpha + teacher_prefix,
    )
    prefix_log_probs_before = torch.logaddexp(
        log_alpha + old_prefix_before,
        log_one_minus_alpha + teacher_prefix_before,
    )
    conditional_log_probs = (prefix_log_probs - prefix_log_probs_before) * mask

    old_component_posterior = torch.exp(
        (log_alpha + old_prefix_before - prefix_log_probs_before).clamp(max=0.0)
    )
    old_component_posterior = torch.where(alpha <= 0.0, 0.0, old_component_posterior)
    old_component_posterior = torch.where(alpha >= 1.0, 1.0, old_component_posterior)
    old_component_posterior = old_component_posterior * mask

    return TrajectoryMixtureProposalOutput(
        conditional_log_probs=conditional_log_probs,
        prefix_log_probs=prefix_log_probs * mask,
        old_component_posterior=old_component_posterior,
    )


def compute_q_mixture_samplek_teacher_advantages(
    *,
    current_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    mixture_alpha: float | torch.Tensor,
    mode: str,
) -> QMixtureSampleKAdvantageOutput:

    mode = normalize_q_mixture_teacher_advantage_mode(mode)
    if mode == "proposal":
        raise ValueError("proposal teacher advantage is handled by the sampled-token decomposed objective.")
    if current_log_probs.dim() != 3:
        raise ValueError(
            "Q-prefix sample-k advantages expect current_log_probs with shape "
            f"(batch, response_length, k), got {tuple(current_log_probs.shape)}."
        )
    _check_shape("teacher_log_probs", teacher_log_probs, current_log_probs.shape)
    if response_mask.shape != current_log_probs.shape[:2]:
        raise ValueError(
            "response_mask shape must match the sample-k batch and response dimensions, "
            f"got mask={tuple(response_mask.shape)}, candidates={tuple(current_log_probs.shape)}."
        )

    current = current_log_probs.detach().float()
    teacher = teacher_log_probs.to(device=current.device).detach().float()
    mask = response_mask.to(device=current.device, dtype=torch.float32).unsqueeze(-1).expand_as(current)
    alpha = torch.as_tensor(mixture_alpha, device=current.device, dtype=torch.float32)
    if alpha.ndim == 0:
        alpha = alpha.reshape(1).expand(current.shape[0])
    elif alpha.ndim == 2 and alpha.shape == (current.shape[0], 1):
        alpha = alpha.squeeze(-1)
    elif alpha.ndim != 1 or alpha.shape[0] != current.shape[0]:
        raise ValueError(
            "mixture_alpha must be scalar, (batch,), or (batch, 1), got "
            f"{tuple(alpha.shape)} for batch size {current.shape[0]}."
        )
    if not torch.isfinite(alpha).all() or (alpha < 0.0).any() or (alpha > 1.0).any():
        raise ValueError("mixture_alpha values must be finite and within [0, 1].")

    current_teacher_log_ratio = current - teacher
    if mode == "current":
        transformed_log_ratio = current_teacher_log_ratio
    else:
        alpha_3d = alpha.view(-1, 1, 1)
        negative_inf = torch.full_like(alpha_3d, -torch.inf)
        log_alpha = torch.where(alpha_3d > 0.0, alpha_3d.log(), negative_inf)
        log_one_minus_alpha = torch.where(
            alpha_3d < 1.0,
            torch.log1p(-alpha_3d),
            negative_inf,
        )
        transformed_log_ratio = torch.logaddexp(
            log_one_minus_alpha,
            log_alpha + current_teacher_log_ratio,
        )

    direct_abs_mean = _masked_mean(current_teacher_log_ratio.abs(), mask)
    transformed_abs_mean = _masked_mean(transformed_log_ratio.abs(), mask)
    sign_flip = ((current_teacher_log_ratio * transformed_log_ratio) < 0.0).to(mask.dtype)
    alpha_tokens = alpha.view(-1, 1, 1).expand_as(current)
    metrics = {
        "teacher_advantage_mode_id": Q_MIXTURE_TEACHER_ADVANTAGE_MODE_IDS[mode],
        "prior_alpha_mean": _masked_mean(alpha_tokens, mask).item(),
        "current_teacher_log_ratio_mean": _masked_mean(current_teacher_log_ratio, mask).item(),
        "current_teacher_log_ratio_abs_mean": direct_abs_mean.item(),
        "transformed_teacher_log_ratio_mean": _masked_mean(transformed_log_ratio, mask).item(),
        "transformed_teacher_log_ratio_abs_mean": transformed_abs_mean.item(),
        "advantage_abs_compression_ratio": (transformed_abs_mean / direct_abs_mean.clamp_min(1e-12)).item(),
        "advantage_sign_flip_fraction": _masked_mean(sign_flip, mask).item(),
    }
    return QMixtureSampleKAdvantageOutput(
        teacher_advantages=(-transformed_log_ratio).to(dtype=current_log_probs.dtype),
        transformed_log_ratio=transformed_log_ratio,
        current_teacher_log_ratio=current_teacher_log_ratio,
        metrics=metrics,
    )


def compute_sampled_token_proximal_rkl_loss(
    *,
    current_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    coefficient: float,
    loss_agg_mode: str,
    prefix_weights: torch.Tensor | None = None,
    max_entropy_coefficient: float = 0.0,
    mode: str = "reverse_kl",
) -> SampledTokenProximalLossOutput:

    if current_log_probs.dim() != 2:
        raise ValueError(
            "sampled-token proximal RKL expects current_log_probs with shape "
            f"(batch, response_length), got {tuple(current_log_probs.shape)}."
        )
    expected_shape = current_log_probs.shape
    _check_shape("reference_log_probs", reference_log_probs, expected_shape)
    _check_shape("response_mask", response_mask, expected_shape)
    if prefix_weights is not None:
        _check_shape("prefix_weights", prefix_weights, expected_shape)

    mode = str(mode or "reverse_kl").strip().lower().replace("-", "_")
    if mode not in SAMPLED_TOKEN_PROXIMAL_MODES:
        raise ValueError(
            f"Unknown sampled-token proximal mode {mode!r}; expected reverse_kl|quadratic_log_ratio."
        )

    coefficient = float(coefficient)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError(f"sampled-token proximal coefficient must be finite and nonnegative, got {coefficient}.")
    max_entropy_coefficient = float(max_entropy_coefficient)
    if not math.isfinite(max_entropy_coefficient) or max_entropy_coefficient < 0.0:
        raise ValueError(
            "sampled-token max-entropy coefficient must be finite and nonnegative, "
            f"got {max_entropy_coefficient}."
        )

    mask = response_mask.to(device=current_log_probs.device, dtype=torch.float32)
    diagnostics = compute_sampled_token_rkl_diagnostics(
        current_log_probs=current_log_probs,
        reference_log_probs=reference_log_probs,
        response_mask=mask,
    )
    with torch.no_grad():
        log_ratio = diagnostics.log_ratio
        ratio = diagnostics.ratio
        current = current_log_probs.detach().float()
        if mode == "reverse_kl":
            rkl_unweighted_coefficients = coefficient * ratio * log_ratio
        else:
            rkl_unweighted_coefficients = coefficient * log_ratio
        max_entropy_unweighted_coefficients = max_entropy_coefficient * ratio * current * mask
        unweighted_coefficients = rkl_unweighted_coefficients + max_entropy_unweighted_coefficients
        if prefix_weights is None:
            effective_prefix_weights = torch.ones_like(unweighted_coefficients)
        else:
            effective_prefix_weights = prefix_weights.to(device=current_log_probs.device).detach().float()
        rkl_coefficients = effective_prefix_weights * rkl_unweighted_coefficients
        max_entropy_coefficients = effective_prefix_weights * max_entropy_unweighted_coefficients
        coefficients = rkl_coefficients + max_entropy_coefficients
        if mode == "reverse_kl":
            anchor_objective = coefficient * effective_prefix_weights * diagnostics.k3
        else:
            anchor_objective = 0.5 * coefficient * effective_prefix_weights * log_ratio.square()
        metrics = {
            "coefficient": coefficient,
            "anchor_mode_id": SAMPLED_TOKEN_PROXIMAL_MODE_IDS[mode],
            "quadratic_log_ratio_enabled": float(mode == "quadratic_log_ratio"),
            "anchor_objective_mean": _masked_mean(anchor_objective, mask).item(),
            "entropy_regularizer_coefficient": max_entropy_coefficient,
            "prefix_weight_enabled": float(prefix_weights is not None),
            "prefix_weight_mean": _masked_mean(effective_prefix_weights, mask).item(),
            "prefix_weight_max": _masked_abs_max(effective_prefix_weights, mask).item(),
            "prefix_weight_ess": _normalized_ess(effective_prefix_weights, mask).item(),
            "log_ratio_mean": _masked_mean(log_ratio, mask).item(),
            "log_ratio_abs_mean": _masked_mean(log_ratio.abs(), mask).item(),
            "log_ratio_abs_p95": _masked_quantile(log_ratio.abs(), mask, 0.95).item(),
            "log_ratio_abs_max": _masked_abs_max(log_ratio, mask).item(),
            "ratio_mean": _masked_mean(ratio, mask).item(),
            "ratio_max": _masked_abs_max(ratio, mask).item(),
            "ratio_ess": _normalized_ess(ratio, mask).item(),
            "rkl_k3_mean": diagnostics.k3_mean.item(),
            "rkl_coefficient_abs_mean": _masked_mean(rkl_coefficients.abs(), mask).item(),
            "anchor_gradient_coefficient_mean": _masked_mean(rkl_coefficients, mask).item(),
            "anchor_gradient_coefficient_abs_mean": _masked_mean(rkl_coefficients.abs(), mask).item(),
            "anchor_gradient_coefficient_abs_max": _masked_abs_max(rkl_coefficients, mask).item(),
            "entropy_regularizer_coefficient_mean": _masked_mean(max_entropy_coefficients, mask).item(),
            "entropy_regularizer_coefficient_abs_mean": _masked_mean(
                max_entropy_coefficients.abs(), mask
            ).item(),
            "entropy_regularizer_coefficient_abs_max": _masked_abs_max(
                max_entropy_coefficients, mask
            ).item(),
            "is_entropy_estimate": -_masked_mean(ratio * current, mask).item(),
            "prefix_weighted_is_entropy_estimate": -_masked_mean(
                effective_prefix_weights * ratio * current, mask
            ).item(),
            "unweighted_coefficient_abs_mean": _masked_mean(unweighted_coefficients.abs(), mask).item(),
            "coefficient_mean": _masked_mean(coefficients, mask).item(),
            "coefficient_abs_mean": _masked_mean(coefficients.abs(), mask).item(),
            "coefficient_abs_max": _masked_abs_max(coefficients, mask).item(),
            "numerical_clip_fraction": _masked_mean(diagnostics.numerical_clip_mask, mask).item(),
        }

    if mode == "reverse_kl":
        loss_mat = coefficients.to(dtype=current_log_probs.dtype) * current_log_probs
    else:
        live_log_ratio = (
            current_log_probs.float()
            - reference_log_probs.to(device=current_log_probs.device).detach().float()
        ) * mask
        quadratic_loss_mat = 0.5 * coefficient * effective_prefix_weights * live_log_ratio.square()
        entropy_loss_mat = max_entropy_coefficients * current_log_probs.float()
        loss_mat = quadratic_loss_mat + entropy_loss_mat
    loss = agg_loss(loss_mat=loss_mat, loss_mask=mask, loss_agg_mode=loss_agg_mode)
    metrics["loss"] = loss.detach().item()
    return SampledTokenProximalLossOutput(loss=loss, coefficients=coefficients, metrics=metrics)


def compute_decomposed_local_opd_loss(
    *,
    current_log_probs: torch.Tensor,
    proposal_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    proximal_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    prefix_is_max_weight: float,
    prefix_is_min_weight: float,
    prefix_is_mode: str,
    proximal_loss_coef: float,
    loss_agg_mode: str,
    teacher_mask: torch.Tensor | None = None,
    proximal_mask: torch.Tensor | None = None,
    joint_loss_aggregation: bool = False,
) -> DecomposedOPDLossOutput:

    if current_log_probs.dim() != 2:
        raise ValueError(
            "decomposed local OPD currently supports sampled-token tensors with shape "
            f"(batch, response_length), got {tuple(current_log_probs.shape)}."
        )
    expected_shape = current_log_probs.shape
    _check_shape("proposal_log_probs", proposal_log_probs, expected_shape)
    _check_shape("teacher_log_probs", teacher_log_probs, expected_shape)
    _check_shape("proximal_log_probs", proximal_log_probs, expected_shape)
    _check_shape("response_mask", response_mask, expected_shape)

    prefix_is_max_weight = float(prefix_is_max_weight)
    if not math.isfinite(prefix_is_max_weight) or prefix_is_max_weight < 1.0:
        raise ValueError(f"prefix_is_max_weight must be finite and at least 1.0, got {prefix_is_max_weight}.")
    prefix_is_min_weight = float(prefix_is_min_weight)
    if (
        not math.isfinite(prefix_is_min_weight)
        or prefix_is_min_weight <= 0.0
        or prefix_is_min_weight > prefix_is_max_weight
    ):
        raise ValueError(
            "prefix_is_min_weight must be finite, positive, and no greater than "
            f"prefix_is_max_weight, got {prefix_is_min_weight}."
        )
    prefix_is_mode = normalize_prefix_is_mode(prefix_is_mode)
    proximal_loss_coef = float(proximal_loss_coef)
    if not math.isfinite(proximal_loss_coef) or proximal_loss_coef < 0.0:
        raise ValueError(f"proximal_loss_coef must be finite and nonnegative, got {proximal_loss_coef}.")

    response_mask_float = response_mask.to(device=current_log_probs.device, dtype=torch.float32)
    if teacher_mask is None:
        teacher_mask_float = response_mask_float
    else:
        _check_shape("teacher_mask", teacher_mask, expected_shape)
        teacher_mask_float = teacher_mask.to(device=current_log_probs.device, dtype=torch.float32)
        teacher_mask_float = teacher_mask_float * response_mask_float
    if proximal_mask is None:
        proximal_mask_float = response_mask_float
    else:
        _check_shape("proximal_mask", proximal_mask, expected_shape)
        proximal_mask_float = proximal_mask.to(device=current_log_probs.device, dtype=torch.float32)
        proximal_mask_float = proximal_mask_float * response_mask_float

    with torch.no_grad():
        current = current_log_probs.detach().float()
        proposal = proposal_log_probs.to(device=current.device).detach().float()
        teacher = teacher_log_probs.to(device=current.device).detach().float()
        proximal = proximal_log_probs.to(device=current.device).detach().float()

        token_proposal_log_ratio = (current - proposal) * response_mask_float
        prefix_log_ratio = token_proposal_log_ratio.cumsum(dim=-1) * response_mask_float
        prefix_token_count = response_mask_float.cumsum(dim=-1).clamp_min(1.0)
        prefix_geometric_log_ratio = (prefix_log_ratio / prefix_token_count) * response_mask_float

        max_log_weight = math.log(prefix_is_max_weight)
        keep_mask = response_mask_float
        lower_reject_mask = torch.zeros_like(response_mask_float)
        upper_reject_mask = torch.zeros_like(response_mask_float)
        if prefix_is_mode == "cumulative_cap":
            cap_mask = (prefix_log_ratio > max_log_weight).to(response_mask_float.dtype)
            effective_prefix_log_ratio = torch.clamp_max(prefix_log_ratio, max_log_weight)
            prefix_is_weights = torch.exp(effective_prefix_log_ratio)
        elif prefix_is_mode == "hard_window":
            min_log_weight = math.log(prefix_is_min_weight)
            lower_reject_mask = (prefix_log_ratio < min_log_weight).to(response_mask_float.dtype)
            upper_reject_mask = (prefix_log_ratio > max_log_weight).to(response_mask_float.dtype)
            keep_mask = (
                (prefix_log_ratio >= min_log_weight) & (prefix_log_ratio <= max_log_weight)
            ).to(response_mask_float.dtype)
            keep_mask = keep_mask * response_mask_float
            cap_mask = torch.zeros_like(response_mask_float)
            prefix_is_weights = torch.exp(
                prefix_log_ratio.clamp(
                    min=-_NUMERICAL_LOG_RATIO_BOUND,
                    max=_NUMERICAL_LOG_RATIO_BOUND,
                )
            )
            prefix_is_weights = prefix_is_weights * keep_mask
        else:
            cap_mask = (prefix_geometric_log_ratio > max_log_weight).to(response_mask_float.dtype)
            effective_prefix_log_ratio = torch.clamp_max(prefix_geometric_log_ratio, max_log_weight)
            prefix_is_weights = torch.exp(effective_prefix_log_ratio)

        cap_mask = cap_mask * response_mask_float
        lower_reject_mask = lower_reject_mask * response_mask_float
        upper_reject_mask = upper_reject_mask * response_mask_float
        prefix_is_weights = torch.where(
            response_mask_float > 0.5, prefix_is_weights, torch.ones_like(prefix_is_weights)
        )
        teacher_effective_mask = teacher_mask_float * keep_mask

        teacher_log_ratio = proposal - teacher
        teacher_coefficients = prefix_is_weights * teacher_log_ratio

        proximal_log_ratio = current - proximal
        proximal_numerical_clip_mask = (proximal_log_ratio.abs() > _NUMERICAL_LOG_RATIO_BOUND).to(
            response_mask_float.dtype
        )
        proximal_ratio = torch.exp(
            proximal_log_ratio.clamp(
                min=-_NUMERICAL_LOG_RATIO_BOUND,
                max=_NUMERICAL_LOG_RATIO_BOUND,
            )
        )
        proximal_coefficients_unscaled = proximal_ratio * proximal_log_ratio
        proximal_coefficients = proximal_loss_coef * proximal_coefficients_unscaled

        raw_prefix_is_weights = torch.exp(
            prefix_log_ratio.clamp(
                min=-_NUMERICAL_LOG_RATIO_BOUND,
                max=_NUMERICAL_LOG_RATIO_BOUND,
            )
        )
        raw_prefix_geometric_weights = torch.exp(
            prefix_geometric_log_ratio.clamp(
                min=-_NUMERICAL_LOG_RATIO_BOUND,
                max=_NUMERICAL_LOG_RATIO_BOUND,
            )
        )
        teacher_abs_mean = _masked_mean(teacher_coefficients.abs(), teacher_effective_mask)
        proximal_unscaled_abs_mean = _masked_mean(proximal_coefficients_unscaled.abs(), proximal_mask_float)
        proximal_abs_mean = _masked_mean(proximal_coefficients.abs(), proximal_mask_float)

        metrics = {
            "prefix_is_mode_id": PREFIX_IS_MODE_IDS[prefix_is_mode],
            "prefix_is_max_weight": prefix_is_max_weight,
            "prefix_is_min_weight": prefix_is_min_weight,
            "proximal_loss_coef": proximal_loss_coef,
            "prefix_log_ratio_mean": _masked_mean(prefix_log_ratio, response_mask_float).item(),
            "prefix_log_ratio_abs_mean": _masked_mean(prefix_log_ratio.abs(), response_mask_float).item(),
            "prefix_log_ratio_abs_p95": _masked_quantile(prefix_log_ratio.abs(), response_mask_float, 0.95).item(),
            "prefix_log_ratio_abs_max": _masked_abs_max(prefix_log_ratio, response_mask_float).item(),
            "prefix_geometric_log_ratio_abs_mean": _masked_mean(
                prefix_geometric_log_ratio.abs(), response_mask_float
            ).item(),
            "prefix_geometric_log_ratio_abs_p95": _masked_quantile(
                prefix_geometric_log_ratio.abs(), response_mask_float, 0.95
            ).item(),
            "prefix_geometric_weight_mean": _masked_mean(
                raw_prefix_geometric_weights, response_mask_float
            ).item(),
            "prefix_geometric_weight_ess": _normalized_ess(
                raw_prefix_geometric_weights, response_mask_float
            ).item(),
            "prefix_is_raw_weight_mean": _masked_mean(raw_prefix_is_weights, response_mask_float).item(),
            "prefix_is_raw_weight_max": _masked_abs_max(raw_prefix_is_weights, response_mask_float).item(),
            "prefix_is_weight_mean": _masked_mean(prefix_is_weights, response_mask_float).item(),
            "prefix_is_weight_max": _masked_abs_max(prefix_is_weights, response_mask_float).item(),
            "prefix_is_weight_ess": _normalized_ess(prefix_is_weights, response_mask_float).item(),
            "prefix_is_conditional_weight_mean": _masked_mean(prefix_is_weights, keep_mask).item(),
            "prefix_is_conditional_weight_ess": _normalized_ess(prefix_is_weights, keep_mask).item(),
            "prefix_is_cap_fraction": _masked_mean(cap_mask, response_mask_float).item(),
            "prefix_is_keep_fraction": _masked_mean(keep_mask, response_mask_float).item(),
            "prefix_is_lower_reject_fraction": _masked_mean(lower_reject_mask, response_mask_float).item(),
            "prefix_is_upper_reject_fraction": _masked_mean(upper_reject_mask, response_mask_float).item(),
            "teacher_valid_token_fraction": _masked_mean(keep_mask, teacher_mask_float).item(),
            "teacher_coef_mean": _masked_mean(teacher_coefficients, teacher_effective_mask).item(),
            "teacher_coef_abs_mean": teacher_abs_mean.item(),
            "proximal_coef_unscaled_mean": _masked_mean(
                proximal_coefficients_unscaled, proximal_mask_float
            ).item(),
            "proximal_coef_unscaled_abs_mean": proximal_unscaled_abs_mean.item(),
            "proximal_coef_mean": _masked_mean(proximal_coefficients, proximal_mask_float).item(),
            "proximal_coef_abs_mean": proximal_abs_mean.item(),
            "teacher_to_prox_abs_ratio": (teacher_abs_mean / proximal_abs_mean.clamp_min(1e-6)).item(),
            "teacher_fraction_of_abs_coef": (
                teacher_abs_mean / (teacher_abs_mean + proximal_abs_mean).clamp_min(1e-12)
            ).item(),
            "proximal_numerical_clip_fraction": _masked_mean(proximal_numerical_clip_mask, proximal_mask_float).item(),
        }
        if joint_loss_aggregation:
            current_teacher_log_ratio = current - teacher
            teacher_advantage_sign_flip = (
                (teacher_log_ratio * current_teacher_log_ratio) < 0.0
            ).to(response_mask_float.dtype)
            conditional_proposal_ratio = torch.exp(
                token_proposal_log_ratio.clamp(
                    min=-_NUMERICAL_LOG_RATIO_BOUND,
                    max=_NUMERICAL_LOG_RATIO_BOUND,
                )
            )
            proposal_rkl_k3 = (
                conditional_proposal_ratio * token_proposal_log_ratio
                - (conditional_proposal_ratio - 1.0)
            )
            proximal_rkl_k3 = proximal_ratio * proximal_log_ratio - (proximal_ratio - 1.0)
            teacher_source_mask = (teacher_mask_float - proximal_mask_float).clamp_min(0.0)
            metrics.update(
                {
                    "joint_loss_aggregation": 1.0,
                    "current_to_proposal_log_ratio_mean": _masked_mean(
                        token_proposal_log_ratio, response_mask_float
                    ).item(),
                    "current_to_proposal_log_ratio_abs_mean": _masked_mean(
                        token_proposal_log_ratio.abs(), response_mask_float
                    ).item(),
                    "current_to_proposal_log_ratio_abs_p95": _masked_quantile(
                        token_proposal_log_ratio.abs(), response_mask_float, 0.95
                    ).item(),
                    "current_to_proposal_log_ratio_abs_max": _masked_abs_max(
                        token_proposal_log_ratio, response_mask_float
                    ).item(),
                    "current_to_proposal_ratio_ess": _normalized_ess(
                        conditional_proposal_ratio, response_mask_float
                    ).item(),
                    "student_current_to_proposal_ratio_ess": _normalized_ess(
                        conditional_proposal_ratio, proximal_mask_float
                    ).item(),
                    "teacher_source_current_to_proposal_ratio_ess": _normalized_ess(
                        conditional_proposal_ratio, teacher_source_mask
                    ).item(),
                    "current_to_proposal_rkl_k3_mean": _masked_mean(
                        proposal_rkl_k3, response_mask_float
                    ).item(),
                    "q_teacher_advantage_mean": _masked_mean(
                        teacher_log_ratio, teacher_mask_float
                    ).item(),
                    "q_teacher_advantage_abs_mean": _masked_mean(
                        teacher_log_ratio.abs(), teacher_mask_float
                    ).item(),
                    "q_teacher_advantage_abs_p95": _masked_quantile(
                        teacher_log_ratio.abs(), teacher_mask_float, 0.95
                    ).item(),
                    "current_teacher_advantage_mean": _masked_mean(
                        current_teacher_log_ratio, teacher_mask_float
                    ).item(),
                    "current_teacher_advantage_abs_mean": _masked_mean(
                        current_teacher_log_ratio.abs(), teacher_mask_float
                    ).item(),
                    "current_teacher_advantage_abs_p95": _masked_quantile(
                        current_teacher_log_ratio.abs(), teacher_mask_float, 0.95
                    ).item(),
                    "teacher_advantage_delta_abs_mean": _masked_mean(
                        token_proposal_log_ratio.abs(), teacher_mask_float
                    ).item(),
                    "teacher_advantage_sign_flip_fraction": _masked_mean(
                        teacher_advantage_sign_flip, teacher_mask_float
                    ).item(),
                    "teacher_advantage_pearson": _masked_pearson(
                        teacher_log_ratio, current_teacher_log_ratio, teacher_mask_float
                    ).item(),
                    "student_current_to_proposal_log_ratio_abs_mean": _masked_mean(
                        token_proposal_log_ratio.abs(), proximal_mask_float
                    ).item(),
                    "teacher_source_current_to_proposal_log_ratio_abs_mean": _masked_mean(
                        token_proposal_log_ratio.abs(), teacher_source_mask
                    ).item(),
                    "proximal_rkl_k3_mean": _masked_mean(
                        proximal_rkl_k3, proximal_mask_float
                    ).item(),
                    "student_current_to_old_log_ratio_mean": _masked_mean(
                        proximal_log_ratio, proximal_mask_float
                    ).item(),
                    "student_current_to_old_log_ratio_abs_mean": _masked_mean(
                        proximal_log_ratio.abs(), proximal_mask_float
                    ).item(),
                    "student_current_to_old_log_ratio_abs_p95": _masked_quantile(
                        proximal_log_ratio.abs(), proximal_mask_float, 0.95
                    ).item(),
                    "student_current_to_old_ratio_ess": _normalized_ess(
                        proximal_ratio, proximal_mask_float
                    ).item(),
                    "student_prefix_is_weight_ess": _normalized_ess(
                        prefix_is_weights, proximal_mask_float
                    ).item(),
                    "teacher_source_prefix_is_weight_ess": _normalized_ess(
                        prefix_is_weights, teacher_source_mask
                    ).item(),
                    "student_prefix_is_cap_fraction": _masked_mean(
                        cap_mask, proximal_mask_float
                    ).item(),
                    "teacher_source_prefix_is_cap_fraction": _masked_mean(
                        cap_mask, teacher_source_mask
                    ).item(),
                }
            )
        metrics.update(
            _position_bucket_metrics(
                prefix_log_ratio=prefix_log_ratio,
                prefix_is_weights=prefix_is_weights,
                cap_mask=cap_mask,
                keep_mask=keep_mask,
                lower_reject_mask=lower_reject_mask,
                upper_reject_mask=upper_reject_mask,
                response_mask=response_mask_float,
            )
        )

    teacher_loss_mat = teacher_coefficients.to(dtype=current_log_probs.dtype) * current_log_probs
    proximal_loss_mat = proximal_coefficients.to(dtype=current_log_probs.dtype) * current_log_probs
    if joint_loss_aggregation:
        teacher_loss = agg_loss(
            loss_mat=teacher_loss_mat * teacher_effective_mask,
            loss_mask=teacher_mask_float,
            loss_agg_mode=loss_agg_mode,
        )
        proximal_loss = agg_loss(
            loss_mat=proximal_loss_mat * proximal_mask_float,
            loss_mask=teacher_mask_float,
            loss_agg_mode=loss_agg_mode,
        )
    else:
        teacher_loss = agg_loss(
            loss_mat=teacher_loss_mat,
            loss_mask=teacher_effective_mask,
            loss_agg_mode=loss_agg_mode,
        )
        proximal_loss = agg_loss(
            loss_mat=proximal_loss_mat,
            loss_mask=proximal_mask_float,
            loss_agg_mode=loss_agg_mode,
        )
    loss = teacher_loss + proximal_loss
    metrics.update(
        {
            "teacher_loss": teacher_loss.detach().item(),
            "proximal_loss": proximal_loss.detach().item(),
            "total_loss": loss.detach().item(),
        }
    )

    return DecomposedOPDLossOutput(
        loss=loss,
        teacher_loss=teacher_loss,
        proximal_loss=proximal_loss,
        prefix_is_weights=prefix_is_weights,
        teacher_coefficients=teacher_coefficients,
        proximal_coefficients=proximal_coefficients,
        teacher_effective_mask=teacher_effective_mask,
        metrics=metrics,
    )
