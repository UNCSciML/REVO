from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


PREFIX_DRIFT_REFERENCE_LOG_PROBS_KEY = "prefix_drift_reference_log_probs"
PREFIX_DRIFT_WEIGHTS_KEY = "prefix_drift_weights"
PREFIX_DRIFT_RAW_WEIGHTS_KEY = "prefix_drift_raw_weights"


@dataclass(frozen=True)
class PrefixDriftResult:
    weights: Optional[torch.Tensor]
    raw_weights: Optional[torch.Tensor]
    metrics: dict[str, float]


def _normalize_method(method: str | None) -> str:
    method = (method or "none").strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "false": "none",
        "0": "none",
        "metrics": "diagnostic",
        "log_only": "diagnostic",
        "diagnostics": "diagnostic",
        "token_is": "token",
        "prefix_is": "prefix",
        "prefix_exclusive": "prefix",
        "prefix_exclusive_is": "prefix",
        "prefix_inclusive_is": "prefix_inclusive",
        "geometric": "prefix_geometric",
        "geometric_prefix": "prefix_geometric",
        "prefix_geom": "prefix_geometric",
        "gspo_prefix": "prefix_geometric",
        "ctpo": "prefix_ctpo",
        "ctpo_prefix": "prefix_ctpo",
        "prefix_ctpo_is": "prefix_ctpo",
        "seq": "sequence",
        "seq_is": "sequence",
        "sequence_is": "sequence",
    }
    return aliases.get(method, method)


def _masked_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask > 0.5]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def _masked_std(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = _masked_values(values, mask)
    if valid.numel() <= 1:
        return values.new_tensor(0.0)
    return valid.float().std(unbiased=False)


def _masked_abs_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = _masked_values(values.abs(), mask)
    if valid.numel() == 0:
        return values.new_tensor(0.0)
    return valid.max()


def _masked_quantile_abs(values: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    valid = _masked_values(values.abs(), mask)
    if valid.numel() == 0:
        return values.new_tensor(0.0)
    return torch.quantile(valid.float(), q)


def _masked_row_quantile_abs(values: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    row_quantiles = []
    for row_values, row_mask in zip(values, mask, strict=True):
        valid = row_values[row_mask > 0.5].abs()
        if valid.numel() > 0:
            row_quantiles.append(torch.quantile(valid.float(), q))
    if not row_quantiles:
        return values.new_empty((0,), dtype=torch.float32)
    return torch.stack(row_quantiles)


def _normalized_ess(weights: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    valid = _masked_values(weights.float(), mask)
    if valid.numel() == 0:
        return weights.new_tensor(1.0, dtype=torch.float32)
    sum_w = valid.sum()
    sum_w2 = valid.square().sum().clamp_min(eps)
    return (sum_w.square() / (sum_w2 * valid.numel())).clamp(min=0.0, max=1.0)


def _position_normalized_ess_from_log_weights(
    log_weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    active = mask > 0.5
    counts = active.sum(dim=0)
    neg_inf = torch.full_like(log_weights, -torch.inf)
    log_sum_w = torch.logsumexp(torch.where(active, log_weights, neg_inf), dim=0)
    log_sum_w2 = torch.logsumexp(torch.where(active, 2.0 * log_weights, neg_inf), dim=0)
    log_fraction = 2.0 * log_sum_w - log_sum_w2 - counts.clamp_min(1).to(log_weights.dtype).log()
    fractions = log_fraction.exp().clamp(min=0.0, max=1.0)
    return torch.where(counts > 0, fractions, torch.ones_like(fractions))


def _position_log_mean_exp(log_weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask > 0.5
    counts = active.sum(dim=0)
    neg_inf = torch.full_like(log_weights, -torch.inf)
    log_sum = torch.logsumexp(torch.where(active, log_weights, neg_inf), dim=0)
    log_mean = log_sum - counts.clamp_min(1).to(log_weights.dtype).log()
    return torch.where(counts > 0, log_mean, torch.zeros_like(log_mean))


def _solve_prefix_ess_beta(
    prefix_log_ratio: torch.Tensor,
    mask: torch.Tensor,
    target_fraction: float,
    bisection_steps: int,
) -> torch.Tensor:
    active_counts = (mask > 0.5).sum(dim=0)
    low = torch.zeros(prefix_log_ratio.shape[1], device=prefix_log_ratio.device, dtype=prefix_log_ratio.dtype)
    high = torch.ones_like(low)
    exact_ess = _position_normalized_ess_from_log_weights(prefix_log_ratio, mask)
    for _ in range(bisection_steps):
        mid = 0.5 * (low + high)
        mid_ess = _position_normalized_ess_from_log_weights(prefix_log_ratio * mid.unsqueeze(0), mask)
        feasible = mid_ess >= target_fraction
        low = torch.where(feasible, mid, low)
        high = torch.where(feasible, high, mid)
    beta = torch.where(exact_ess >= target_fraction, torch.ones_like(low), low)
    return torch.where(active_counts <= 1, torch.ones_like(beta), beta)


def _solve_prefix_ess_beta_cpu_numpy(
    prefix_log_ratio: torch.Tensor,
    mask: torch.Tensor,
    target_fraction: float,
    bisection_steps: int,
) -> torch.Tensor:

    log_weights = prefix_log_ratio.detach().float().numpy()
    active = mask.detach().numpy() > 0.5
    active_counts = active.sum(axis=0)

    def normalized_ess(values: np.ndarray) -> np.ndarray:
        masked_values = np.where(active, values, -np.inf)
        maxima = masked_values.max(axis=0)
        centered = values - maxima[None, :]
        weights = np.zeros_like(values)
        with np.errstate(over="ignore", invalid="ignore"):
            np.exp(centered, out=weights, where=active)
        sum_weights = weights.sum(axis=0)
        sum_squared_weights = np.square(weights).sum(axis=0)
        denominator = sum_squared_weights * np.maximum(active_counts, 1)
        fractions = np.ones_like(sum_weights)
        np.divide(
            np.square(sum_weights),
            denominator,
            out=fractions,
            where=active_counts > 0,
        )
        return np.clip(fractions, 0.0, 1.0)

    low = np.zeros(log_weights.shape[1], dtype=log_weights.dtype)
    high = np.ones_like(low)
    exact_ess = normalized_ess(log_weights)
    for _ in range(bisection_steps):
        mid = 0.5 * (low + high)
        feasible = normalized_ess(log_weights * mid[None, :]) >= target_fraction
        low = np.where(feasible, mid, low)
        high = np.where(feasible, high, mid)
    beta = np.where(exact_ess >= target_fraction, 1.0, low)
    beta = np.where(active_counts <= 1, 1.0, beta).astype(log_weights.dtype, copy=False)
    return torch.from_numpy(beta).to(dtype=prefix_log_ratio.dtype)


def _position_mean_normalize_log_weights(log_weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask > 0.5
    counts = active.sum(dim=0)
    log_mean = _position_log_mean_exp(log_weights, mask)
    normalized = torch.exp(log_weights - log_mean.unsqueeze(0))
    normalized = torch.where(active, normalized, torch.ones_like(normalized))
    return torch.where((counts > 0).unsqueeze(0), normalized, torch.ones_like(normalized))


def _active_position_values(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    default: float,
) -> torch.Tensor:
    active_values = values[(mask > 0.5).sum(dim=0) > 0]
    if active_values.numel() > 0:
        return active_values
    return values.new_tensor([default])


def _masked_fraction(numerator_mask: torch.Tensor, denominator_mask: torch.Tensor) -> torch.Tensor:
    denom = denominator_mask.sum().clamp_min(1.0)
    return (numerator_mask.to(dtype=denominator_mask.dtype) * denominator_mask).sum() / denom


def _add_weight_metrics(
    *,
    metrics: dict[str, float],
    metric_prefix: str,
    weights: torch.Tensor,
    raw_weights: torch.Tensor | None,
    mask: torch.Tensor,
) -> None:
    metrics.update(
        {
            f"{metric_prefix}/weight_mean": _masked_mean(weights.float(), mask).item(),
            f"{metric_prefix}/weight_std": _masked_std(weights.float(), mask).item(),
            f"{metric_prefix}/weight_abs_max": _masked_abs_max(weights.float(), mask).item(),
            f"{metric_prefix}/weight_ess": _normalized_ess(weights.float(), mask).item(),
        }
    )
    if raw_weights is not None:
        metrics.update(
            {
                f"{metric_prefix}/raw_weight_mean": _masked_mean(raw_weights.float(), mask).item(),
                f"{metric_prefix}/raw_weight_std": _masked_std(raw_weights.float(), mask).item(),
                f"{metric_prefix}/raw_weight_abs_max": _masked_abs_max(raw_weights.float(), mask).item(),
                f"{metric_prefix}/raw_weight_ess": _normalized_ess(raw_weights.float(), mask).item(),
            }
        )


def _normalize_weights(weights: torch.Tensor, mask: torch.Tensor, normalize: str) -> torch.Tensor:
    normalize = (normalize or "none").strip().lower().replace("-", "_")
    if normalize in {"none", "false", "0", "off"}:
        return weights
    if normalize in {"batch", "global"}:
        mean_w = _masked_mean(weights, mask).clamp_min(1e-12)
        return weights / mean_w
    if normalize in {"sequence", "seq", "per_sequence"}:
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_w = ((weights * mask).sum(dim=-1, keepdim=True) / denom).clamp_min(1e-12)
        return weights / mean_w
    raise ValueError(f"Unknown prefix drift weight normalization: {normalize!r}")


def _position_coefficients(
    *,
    mask: torch.Tensor,
    beta: float,
    hmax: int | None,
) -> torch.Tensor:
    if not -1.0 <= beta <= 1.0:
        raise ValueError(f"prefix drift position beta must be in [-1, 1], got {beta}.")

    response_length = mask.shape[-1]
    if hmax is None:
        hmax = response_length
    hmax = int(hmax)
    if hmax < 2:
        raise ValueError(f"prefix drift position hmax must be at least 2, got {hmax}.")
    if hmax < response_length:
        raise ValueError(
            "prefix drift position hmax should be the max generation budget and must be "
            f">= response length, got hmax={hmax}, response_length={response_length}."
        )

    positions = torch.arange(response_length, device=mask.device, dtype=mask.dtype)
    coefficients = 1.0 + beta * torch.cos(math.pi * positions / float(hmax - 1))
    return coefficients.view(1, response_length)


def compute_prefix_drift(
    *,
    current_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    method: str,
    log_clip: float | None = None,
    log_clip_mode: str = "symmetric",
    ctpo_log_clip_base: float | None = None,
    ctpo_log_clip_lower_base: float | None = None,
    ctpo_log_clip_upper_base: float | None = None,
    ctpo_log_clip_power: float = 0.5,
    normalize: str = "none",
    position_beta: float = 0.0,
    position_hmax: int | None = None,
    detach: bool = True,
    metric_prefix: str = "prefix_drift",
    ess_target_fraction: float = 0.5,
    ess_bisection_steps: int = 8,
) -> PrefixDriftResult:

    method = _normalize_method(method)
    if method not in {
        "none",
        "diagnostic",
        "token",
        "prefix",
        "prefix_geometric",
        "prefix_ess",
        "prefix_ctpo",
        "prefix_inclusive",
        "sequence",
    }:
        raise ValueError(
            "Unknown prefix drift method "
            f"{method!r}; expected none|diagnostic|token|prefix|prefix_geometric|prefix_ess|prefix_ctpo|"
            "prefix_inclusive|sequence."
        )
    log_clip_mode = str(log_clip_mode or "symmetric").strip().lower().replace("-", "_")
    if log_clip_mode not in {"symmetric", "upper"}:
        raise ValueError(f"Unknown prefix drift log clip mode {log_clip_mode!r}; expected symmetric|upper.")
    if method == "prefix_ess":
        ess_target_fraction = float(ess_target_fraction)
        if not math.isfinite(ess_target_fraction) or not 0.0 < ess_target_fraction <= 1.0:
            raise ValueError(
                "prefix drift ESS target fraction must be finite and in (0, 1], "
                f"got {ess_target_fraction}."
            )
        ess_bisection_steps_float = float(ess_bisection_steps)
        if not math.isfinite(ess_bisection_steps_float) or not ess_bisection_steps_float.is_integer():
            raise ValueError(
                "prefix drift ESS bisection steps must be a positive integer, "
                f"got {ess_bisection_steps_float}."
            )
        ess_bisection_steps = int(ess_bisection_steps_float)
        if ess_bisection_steps < 1:
            raise ValueError(
                "prefix drift ESS bisection steps must be a positive integer, "
                f"got {ess_bisection_steps_float}."
            )
        if log_clip is not None:
            raise ValueError("prefix_ess requires prefix_drift_log_clip=None.")
        normalized_mode = str(normalize or "none").strip().lower().replace("-", "_")
        if normalized_mode not in {"none", "false", "0", "off"}:
            raise ValueError("prefix_ess performs position normalization internally and requires normalize=none.")
        if abs(float(position_beta or 0.0)) > 1e-12:
            raise ValueError("prefix_ess requires position_beta=0.")

    current = current_log_probs.detach().float()
    reference = reference_log_probs.to(device=current.device).detach().float()
    mask = response_mask.to(device=current.device, dtype=current.dtype)
    if current.shape != reference.shape:
        raise ValueError(
            f"current_log_probs shape {current.shape} does not match reference_log_probs shape {reference.shape}."
        )
    if current.shape != mask.shape:
        raise ValueError(f"log_prob shape {current.shape} does not match response_mask shape {mask.shape}.")

    token_log_ratio = (current - reference) * mask
    prefix_inclusive_log_ratio = token_log_ratio.cumsum(dim=-1) * mask
    prefix_log_ratio = (prefix_inclusive_log_ratio - token_log_ratio) * mask
    prefix_token_count = (mask.cumsum(dim=-1) - mask).clamp_min(1.0)
    prefix_geometric_log_ratio = (prefix_log_ratio / prefix_token_count) * mask
    sequence_log_ratio = token_log_ratio.sum(dim=-1, keepdim=True).expand_as(token_log_ratio) * mask
    prefix_sequence_p95 = _masked_row_quantile_abs(prefix_log_ratio, mask, 0.95)

    metrics = {
        f"{metric_prefix}/enabled": 1.0,
        f"{metric_prefix}/token_log_ratio_mean": _masked_mean(token_log_ratio, mask).item(),
        f"{metric_prefix}/token_log_ratio_abs_mean": _masked_mean(token_log_ratio.abs(), mask).item(),
        f"{metric_prefix}/prefix_log_ratio_mean": _masked_mean(prefix_log_ratio, mask).item(),
        f"{metric_prefix}/prefix_log_ratio_std": _masked_std(prefix_log_ratio, mask).item(),
        f"{metric_prefix}/prefix_log_ratio_abs_mean": _masked_mean(prefix_log_ratio.abs(), mask).item(),
        f"{metric_prefix}/prefix_log_ratio_abs_max": _masked_abs_max(prefix_log_ratio, mask).item(),
        f"{metric_prefix}/prefix_log_ratio_abs_p50": _masked_quantile_abs(prefix_log_ratio, mask, 0.50).item(),
        f"{metric_prefix}/prefix_log_ratio_abs_p90": _masked_quantile_abs(prefix_log_ratio, mask, 0.90).item(),
        f"{metric_prefix}/prefix_log_ratio_abs_p95": _masked_quantile_abs(prefix_log_ratio, mask, 0.95).item(),
        f"{metric_prefix}/prefix_log_ratio_abs_p99": _masked_quantile_abs(prefix_log_ratio, mask, 0.99).item(),
        f"{metric_prefix}/prefix_log_ratio_abs_sequence_p95_p50": (
            torch.quantile(prefix_sequence_p95, 0.50).item() if prefix_sequence_p95.numel() else 0.0
        ),
        f"{metric_prefix}/prefix_log_ratio_abs_sequence_p95_p90": (
            torch.quantile(prefix_sequence_p95, 0.90).item() if prefix_sequence_p95.numel() else 0.0
        ),
        f"{metric_prefix}/prefix_geometric_log_ratio_abs_mean": _masked_mean(
            prefix_geometric_log_ratio.abs(), mask
        ).item(),
        f"{metric_prefix}/prefix_geometric_log_ratio_abs_p95": _masked_quantile_abs(
            prefix_geometric_log_ratio, mask, 0.95
        ).item(),
        f"{metric_prefix}/sequence_log_ratio_mean": _masked_mean(sequence_log_ratio, mask).item(),
        f"{metric_prefix}/sequence_log_ratio_std": _masked_std(sequence_log_ratio, mask).item(),
        f"{metric_prefix}/sequence_log_ratio_abs_mean": _masked_mean(sequence_log_ratio.abs(), mask).item(),
        f"{metric_prefix}/sequence_log_ratio_abs_max": _masked_abs_max(sequence_log_ratio, mask).item(),
    }

    if method in {"none", "diagnostic"}:
        metrics[f"{metric_prefix}/applied"] = 0.0
        metrics[f"{metric_prefix}/position_applied"] = 0.0
        return PrefixDriftResult(weights=None, raw_weights=None, metrics=metrics)

    if method == "prefix_ess":
        if prefix_log_ratio.is_cuda:
            solve_start = torch.cuda.Event(enable_timing=True)
            solve_end = torch.cuda.Event(enable_timing=True)
            solve_start.record()
            beta = _solve_prefix_ess_beta(
                prefix_log_ratio=prefix_log_ratio,
                mask=mask,
                target_fraction=ess_target_fraction,
                bisection_steps=ess_bisection_steps,
            )
            solve_end.record()
            solve_end.synchronize()
            beta_solve_ms = solve_start.elapsed_time(solve_end)
            beta_solve_cpu_numpy = 0.0
        else:
            solve_start_time = time.perf_counter()
            beta = _solve_prefix_ess_beta_cpu_numpy(
                prefix_log_ratio=prefix_log_ratio,
                mask=mask,
                target_fraction=ess_target_fraction,
                bisection_steps=ess_bisection_steps,
            )
            beta_solve_ms = (time.perf_counter() - solve_start_time) * 1000.0
            beta_solve_cpu_numpy = 1.0
        tempered_log_weights = prefix_log_ratio * beta.unsqueeze(0)
        weights = _position_mean_normalize_log_weights(tempered_log_weights, mask)
        achieved_ess = _position_normalized_ess_from_log_weights(tempered_log_weights, mask)
        exact_ess = _position_normalized_ess_from_log_weights(prefix_log_ratio, mask)
        exact_log_mean_weight = _position_log_mean_exp(prefix_log_ratio, mask)
        active_counts = (mask > 0.5).sum(dim=0).to(torch.float32)
        beta_values = _active_position_values(beta, mask, default=1.0)
        achieved_ess_values = _active_position_values(achieved_ess, mask, default=1.0)
        exact_ess_values = _active_position_values(exact_ess, mask, default=1.0)
        exact_log_mean_weight_values = _active_position_values(exact_log_mean_weight, mask, default=0.0)
        active_count_values = _active_position_values(active_counts, mask, default=0.0)

        raw_weights = torch.exp(prefix_log_ratio)
        raw_weights = torch.nan_to_num(
            raw_weights,
            nan=0.0,
            posinf=torch.finfo(raw_weights.dtype).max,
            neginf=0.0,
        )
        raw_weights = torch.where(mask > 0.5, raw_weights, torch.ones_like(raw_weights))
        weights = torch.where(mask > 0.5, weights, torch.ones_like(weights))
        if detach:
            weights = weights.detach()
            raw_weights = raw_weights.detach()

        metrics.update(
            {
                f"{metric_prefix}/applied": 1.0,
                f"{metric_prefix}/ctpo_enabled": 0.0,
                f"{metric_prefix}/log_clip_fraction": 0.0,
                f"{metric_prefix}/log_clip_upper_only": float(log_clip_mode == "upper"),
                f"{metric_prefix}/position_beta": 0.0,
                f"{metric_prefix}/position_applied": 0.0,
                f"{metric_prefix}/ess_enabled": 1.0,
                f"{metric_prefix}/ess_target_fraction": ess_target_fraction,
                f"{metric_prefix}/ess_bisection_steps": float(ess_bisection_steps),
                f"{metric_prefix}/ess_beta_solve_ms": float(beta_solve_ms),
                f"{metric_prefix}/ess_beta_solve_cpu_numpy": beta_solve_cpu_numpy,
                f"{metric_prefix}/ess_beta_mean": beta_values.mean().item(),
                f"{metric_prefix}/ess_beta_min": beta_values.min().item(),
                f"{metric_prefix}/ess_beta_p50": torch.quantile(beta_values, 0.50).item(),
                f"{metric_prefix}/ess_beta_p95": torch.quantile(beta_values, 0.95).item(),
                f"{metric_prefix}/ess_beta_max": beta_values.max().item(),
                f"{metric_prefix}/ess_fraction_mean": achieved_ess_values.mean().item(),
                f"{metric_prefix}/ess_fraction_min": achieved_ess_values.min().item(),
                f"{metric_prefix}/ess_exact_fraction_mean": exact_ess_values.mean().item(),
                f"{metric_prefix}/ess_exact_fraction_min": exact_ess_values.min().item(),
                f"{metric_prefix}/ess_active_count_mean": active_count_values.mean().item(),
                f"{metric_prefix}/ess_active_count_min": active_count_values.min().item(),
                f"{metric_prefix}/ess_exact_log_mean_weight_mean": exact_log_mean_weight_values.mean().item(),
                f"{metric_prefix}/ess_exact_log_mean_weight_min": exact_log_mean_weight_values.min().item(),
                f"{metric_prefix}/weight_mean": _masked_mean(weights.float(), mask).item(),
                f"{metric_prefix}/weight_std": _masked_std(weights.float(), mask).item(),
                f"{metric_prefix}/weight_abs_max": _masked_abs_max(weights.float(), mask).item(),
                f"{metric_prefix}/weight_ess": _normalized_ess(weights.float(), mask).item(),
            }
        )
        return PrefixDriftResult(
            weights=weights.to(dtype=current_log_probs.dtype),
            raw_weights=raw_weights.to(dtype=current_log_probs.dtype),
            metrics=metrics,
        )

    if method == "token":
        selected_log_ratio = token_log_ratio
    elif method == "prefix":
        selected_log_ratio = prefix_log_ratio
    elif method == "prefix_geometric":
        selected_log_ratio = prefix_geometric_log_ratio
    elif method == "prefix_ctpo":
        selected_log_ratio = prefix_log_ratio
    elif method == "prefix_inclusive":
        selected_log_ratio = prefix_inclusive_log_ratio
    else:
        selected_log_ratio = sequence_log_ratio

    selected_log_ratio = selected_log_ratio * mask
    raw_selected_log_ratio = selected_log_ratio.detach().clone()
    if method == "prefix_ctpo":
        if not math.isfinite(float(ctpo_log_clip_power)) or float(ctpo_log_clip_power) < 0.0:
            raise ValueError(f"prefix drift CTPO clip power must be finite and nonnegative, got {ctpo_log_clip_power}.")
        base = ctpo_log_clip_base
        if base is None:
            base = log_clip
        if base is None:
            base = 0.02
        upper_base = ctpo_log_clip_upper_base if ctpo_log_clip_upper_base is not None else base
        lower_base = ctpo_log_clip_lower_base if ctpo_log_clip_lower_base is not None else base
        upper_base = abs(float(upper_base))
        lower_base = abs(float(lower_base))
        power = float(ctpo_log_clip_power)
        position_scale = prefix_token_count.pow(power) * mask
        upper_bound = upper_base * position_scale
        lower_bound = lower_base * position_scale
        clipped_upper_mask = selected_log_ratio > upper_bound
        if log_clip_mode == "symmetric":
            clipped_lower_mask = selected_log_ratio < -lower_bound
            selected_log_ratio = torch.minimum(torch.maximum(selected_log_ratio, -lower_bound), upper_bound)
        else:
            clipped_lower_mask = torch.zeros_like(clipped_upper_mask)
            selected_log_ratio = torch.minimum(selected_log_ratio, upper_bound)
        clipped_mask = clipped_upper_mask | clipped_lower_mask
        metrics.update(
            {
                f"{metric_prefix}/log_clip": float(base),
                f"{metric_prefix}/log_clip_fraction": _masked_mean(clipped_mask.to(mask.dtype), mask).item(),
                f"{metric_prefix}/log_clip_upper_fraction": _masked_mean(
                    clipped_upper_mask.to(mask.dtype), mask
                ).item(),
                f"{metric_prefix}/log_clip_lower_fraction": _masked_mean(
                    clipped_lower_mask.to(mask.dtype), mask
                ).item(),
                f"{metric_prefix}/ctpo_enabled": 1.0,
                f"{metric_prefix}/ctpo_log_clip_upper_base": upper_base,
                f"{metric_prefix}/ctpo_log_clip_lower_base": lower_base,
                f"{metric_prefix}/ctpo_log_clip_power": power,
                f"{metric_prefix}/ctpo_upper_bound_mean": _masked_mean(upper_bound, mask).item(),
                f"{metric_prefix}/ctpo_upper_bound_p95": _masked_quantile_abs(upper_bound, mask, 0.95).item(),
                f"{metric_prefix}/ctpo_lower_bound_mean": _masked_mean(lower_bound, mask).item(),
                f"{metric_prefix}/ctpo_lower_bound_p95": _masked_quantile_abs(lower_bound, mask, 0.95).item(),
            }
        )
    elif log_clip is not None:
        clip = abs(float(log_clip))
        if log_clip_mode == "symmetric":
            clipped_mask = selected_log_ratio.abs() > clip
            selected_log_ratio = selected_log_ratio.clamp(min=-clip, max=clip)
        else:
            clipped_mask = selected_log_ratio > clip
            selected_log_ratio = selected_log_ratio.clamp_max(clip)
        clipped_fraction = _masked_mean(clipped_mask.to(mask.dtype), mask)
        metrics[f"{metric_prefix}/log_clip"] = clip
        metrics[f"{metric_prefix}/log_clip_fraction"] = clipped_fraction.item()
    else:
        metrics[f"{metric_prefix}/log_clip_fraction"] = 0.0
        metrics[f"{metric_prefix}/ctpo_enabled"] = 0.0
    metrics.setdefault(f"{metric_prefix}/ctpo_enabled", 0.0)
    metrics[f"{metric_prefix}/log_clip_upper_only"] = float(log_clip_mode == "upper")

    raw_weights = torch.exp(raw_selected_log_ratio)
    raw_weights = torch.nan_to_num(
        raw_weights,
        nan=0.0,
        posinf=torch.finfo(raw_weights.dtype).max,
        neginf=0.0,
    )
    raw_weights = torch.where(mask > 0.5, raw_weights, torch.ones_like(raw_weights))

    weights = torch.exp(selected_log_ratio)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=torch.finfo(weights.dtype).max, neginf=0.0)

    position_beta = float(position_beta or 0.0)
    metrics[f"{metric_prefix}/position_beta"] = position_beta
    metrics[f"{metric_prefix}/position_applied"] = 0.0
    if abs(position_beta) > 1e-12:
        metrics.update(
            {
                f"{metric_prefix}/weight_pre_position_mean": _masked_mean(weights.float(), mask).item(),
                f"{metric_prefix}/weight_pre_position_std": _masked_std(weights.float(), mask).item(),
                f"{metric_prefix}/weight_pre_position_abs_max": _masked_abs_max(weights.float(), mask).item(),
                f"{metric_prefix}/weight_pre_position_ess": _normalized_ess(weights.float(), mask).item(),
            }
        )
        position_coefficients = _position_coefficients(mask=mask, beta=position_beta, hmax=position_hmax)
        position_coefficients_for_metrics = position_coefficients.expand_as(mask)
        weights = weights * position_coefficients
        metrics.update(
            {
                f"{metric_prefix}/position_applied": 1.0,
                f"{metric_prefix}/position_hmax": float(position_hmax or mask.shape[-1]),
                f"{metric_prefix}/position_coeff_first": position_coefficients[0, 0].item(),
                f"{metric_prefix}/position_coeff_last": position_coefficients[0, -1].item(),
                f"{metric_prefix}/position_coeff_masked_mean": _masked_mean(
                    position_coefficients_for_metrics, mask
                ).item(),
                f"{metric_prefix}/position_coeff_masked_std": _masked_std(
                    position_coefficients_for_metrics, mask
                ).item(),
                f"{metric_prefix}/weight_pre_normalize_mean": _masked_mean(weights.float(), mask).item(),
                f"{metric_prefix}/weight_pre_normalize_std": _masked_std(weights.float(), mask).item(),
                f"{metric_prefix}/weight_pre_normalize_abs_max": _masked_abs_max(weights.float(), mask).item(),
                f"{metric_prefix}/weight_pre_normalize_ess": _normalized_ess(weights.float(), mask).item(),
            }
        )

    weights = _normalize_weights(weights, mask, normalize)
    weights = torch.where(mask > 0.5, weights, torch.ones_like(weights))
    if detach:
        weights = weights.detach()
        raw_weights = raw_weights.detach()

    metrics.update(
        {
            f"{metric_prefix}/applied": 1.0,
            f"{metric_prefix}/weight_mean": _masked_mean(weights.float(), mask).item(),
            f"{metric_prefix}/weight_std": _masked_std(weights.float(), mask).item(),
            f"{metric_prefix}/weight_abs_max": _masked_abs_max(weights.float(), mask).item(),
            f"{metric_prefix}/weight_ess": _normalized_ess(weights.float(), mask).item(),
        }
    )
    return PrefixDriftResult(
        weights=weights.to(dtype=current_log_probs.dtype),
        raw_weights=raw_weights.to(dtype=current_log_probs.dtype),
        metrics=metrics,
    )


def compute_source_specific_prefix_drift(
    *,
    current_log_probs: torch.Tensor,
    student_reference_log_probs: torch.Tensor,
    teacher_reference_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    student_token_mask: torch.Tensor,
    student_method: str,
    teacher_method: str = "prefix",
    log_clip: float | None = None,
    log_clip_mode: str = "symmetric",
    teacher_log_clip: float | None = None,
    teacher_log_clip_mode: str | None = None,
    teacher_min_weight: float | None = None,
    ctpo_log_clip_base: float | None = None,
    ctpo_log_clip_lower_base: float | None = None,
    ctpo_log_clip_upper_base: float | None = None,
    ctpo_log_clip_power: float = 0.5,
    normalize: str = "none",
    position_beta: float = 0.0,
    position_hmax: int | None = None,
    detach: bool = True,
    metric_prefix: str = "prefix_drift",
    ess_target_fraction: float = 0.5,
    ess_bisection_steps: int = 8,
) -> PrefixDriftResult:

    if current_log_probs.dim() != 2:
        raise ValueError(
            "source-specific prefix drift expects current_log_probs with shape "
            f"(batch, response_length), got {tuple(current_log_probs.shape)}."
        )
    expected_shape = current_log_probs.shape
    for name, tensor in (
        ("student_reference_log_probs", student_reference_log_probs),
        ("teacher_reference_log_probs", teacher_reference_log_probs),
        ("response_mask", response_mask),
        ("student_token_mask", student_token_mask),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match "
                f"current_log_probs shape {expected_shape}."
            )

    device = current_log_probs.device
    mask = response_mask.to(device=device, dtype=torch.float32)
    student_mask = student_token_mask.to(device=device, dtype=torch.float32) * mask
    teacher_mask = (mask - student_mask).clamp(min=0.0, max=1.0)

    student_result = compute_prefix_drift(
        current_log_probs=current_log_probs,
        reference_log_probs=student_reference_log_probs,
        response_mask=student_mask,
        method=student_method,
        log_clip=log_clip,
        log_clip_mode=log_clip_mode,
        ctpo_log_clip_base=ctpo_log_clip_base,
        ctpo_log_clip_lower_base=ctpo_log_clip_lower_base,
        ctpo_log_clip_upper_base=ctpo_log_clip_upper_base,
        ctpo_log_clip_power=ctpo_log_clip_power,
        normalize=normalize,
        position_beta=position_beta,
        position_hmax=position_hmax,
        detach=detach,
        metric_prefix=f"{metric_prefix}/student",
        ess_target_fraction=ess_target_fraction,
        ess_bisection_steps=ess_bisection_steps,
    )
    teacher_result = compute_prefix_drift(
        current_log_probs=current_log_probs,
        reference_log_probs=teacher_reference_log_probs,
        response_mask=teacher_mask,
        method=teacher_method,
        log_clip=log_clip if teacher_log_clip is None else teacher_log_clip,
        log_clip_mode=log_clip_mode if teacher_log_clip_mode is None else teacher_log_clip_mode,
        normalize="none",
        position_beta=0.0,
        position_hmax=position_hmax,
        detach=detach,
        metric_prefix=f"{metric_prefix}/teacher",
        ess_target_fraction=ess_target_fraction,
        ess_bisection_steps=ess_bisection_steps,
    )
    if student_result.weights is None or student_result.raw_weights is None:
        raise ValueError("source-specific prefix drift requires the student method to return weights.")
    if teacher_result.weights is None or teacher_result.raw_weights is None:
        raise ValueError("source-specific prefix drift requires the teacher method to return weights.")

    teacher_weights = teacher_result.weights
    teacher_raw_weights = teacher_result.raw_weights
    threshold_enabled = teacher_min_weight is not None
    if threshold_enabled:
        min_weight = float(teacher_min_weight)
        if not math.isfinite(min_weight) or min_weight < 0.0:
            raise ValueError(
                "source-specific teacher prefix min weight must be null or a finite "
                f"nonnegative value, got {teacher_min_weight}."
            )
        teacher_valid = ((teacher_raw_weights.float() >= min_weight) | (teacher_mask <= 0.5)).to(
            dtype=teacher_weights.dtype
        )
        teacher_weights = teacher_weights * teacher_valid
    else:
        min_weight = 0.0
        teacher_valid = torch.ones_like(teacher_weights)

    teacher_bool = teacher_mask > 0.5
    weights = torch.where(teacher_bool, teacher_weights, student_result.weights)
    raw_weights = torch.where(teacher_bool, teacher_raw_weights, student_result.raw_weights)
    weights = torch.where(mask > 0.5, weights, torch.ones_like(weights))
    raw_weights = torch.where(mask > 0.5, raw_weights, torch.ones_like(raw_weights))

    if detach:
        weights = weights.detach()
        raw_weights = raw_weights.detach()

    total_tokens = mask.sum().clamp_min(1.0)
    student_tokens = student_mask.sum()
    teacher_tokens = teacher_mask.sum()
    student_fraction = (student_tokens / total_tokens).item()
    teacher_fraction = (teacher_tokens / total_tokens).item()
    teacher_valid_fraction = (
        _masked_fraction(teacher_valid, teacher_mask).item()
        if teacher_tokens.item() > 0.0
        else 1.0
    )
    metrics: dict[str, float] = {
        f"{metric_prefix}/enabled": 1.0,
        f"{metric_prefix}/applied": 1.0,
        f"{metric_prefix}/source_specific_enabled": 1.0,
        f"{metric_prefix}/source_specific_student_token_fraction": student_fraction,
        f"{metric_prefix}/source_specific_teacher_token_fraction": teacher_fraction,
        f"{metric_prefix}/teacher_min_weight": min_weight,
        f"{metric_prefix}/teacher_min_weight_enabled": float(threshold_enabled),
        f"{metric_prefix}/teacher_valid_weight_fraction": teacher_valid_fraction,
        f"{metric_prefix}/teacher_masked_weight_fraction": 1.0 - teacher_valid_fraction,
    }
    metrics.update(student_result.metrics)
    metrics.update(teacher_result.metrics)
    _add_weight_metrics(
        metrics=metrics,
        metric_prefix=metric_prefix,
        weights=weights,
        raw_weights=raw_weights,
        mask=mask,
    )

    for suffix in ("log_clip_fraction", "log_clip_upper_fraction", "log_clip_lower_fraction"):
        student_key = f"{metric_prefix}/student/{suffix}"
        teacher_key = f"{metric_prefix}/teacher/{suffix}"
        if student_key in metrics or teacher_key in metrics:
            student_value = metrics.get(student_key, 0.0)
            teacher_value = metrics.get(teacher_key, 0.0)
            metrics[f"{metric_prefix}/{suffix}"] = (
                student_value * student_fraction + teacher_value * teacher_fraction
            )

    return PrefixDriftResult(
        weights=weights.to(dtype=current_log_probs.dtype),
        raw_weights=raw_weights.to(dtype=current_log_probs.dtype),
        metrics=metrics,
    )
