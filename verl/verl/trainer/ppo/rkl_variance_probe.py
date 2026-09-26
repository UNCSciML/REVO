from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EstimatorMoments:
    scalar_variance: float
    gradient_trace_variance: float
    optimal_baseline: float
    optimal_gradient_trace_variance: float
    target_gradient_norm_squared: float
    head_count: int
    head_mass: float
    tail_samples: int


def remap_teacher_stop_log_probs(
    teacher_log_probs: np.ndarray,
    *,
    primary_eos_token_id: int,
    secondary_eos_token_id: int,
    secondary_floor: float,
) -> np.ndarray:

    log_probs = np.asarray(teacher_log_probs, dtype=np.float64)
    if log_probs.ndim != 1 or not np.isfinite(log_probs).all():
        raise ValueError("teacher_log_probs must be a finite one-dimensional array.")
    primary = int(primary_eos_token_id)
    secondary = int(secondary_eos_token_id)
    if primary == secondary or min(primary, secondary) < 0 or max(primary, secondary) >= log_probs.size:
        raise ValueError("primary and secondary EOS ids must be distinct in-vocabulary ids.")
    floor = float(secondary_floor)
    if not np.isfinite(floor) or floor <= 0.0 or floor >= 1.0:
        raise ValueError("secondary_floor must be finite and lie in (0, 1).")

    remapped = log_probs.copy()
    remapped[primary] = np.logaddexp(log_probs[primary], log_probs[secondary])
    remapped[secondary] = np.log(floor)
    return remapped


def _normalized_probabilities(probabilities: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {values.shape}.")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError(f"{name} must contain finite, nonnegative values.")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass.")
    return values / total


def adaptive_head_indices(
    probabilities: np.ndarray,
    *,
    candidate_budget: int,
    gamma: float,
    tail_samples_min: int,
) -> np.ndarray:

    p = _normalized_probabilities(probabilities, name="probabilities")
    candidate_budget = int(candidate_budget)
    tail_samples_min = int(tail_samples_min)
    gamma = float(gamma)
    if candidate_budget <= 0:
        raise ValueError("candidate_budget must be positive.")
    if not 1 <= tail_samples_min <= candidate_budget:
        raise ValueError("tail_samples_min must lie in [1, candidate_budget].")
    if not np.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and nonnegative.")

    max_head = candidate_budget - tail_samples_min
    order = np.argsort(-p, kind="stable")[:max_head]
    accepted: list[int] = []
    head_mass = 0.0
    for slot, token_id in enumerate(order):
        remaining_budget = max(float(candidate_budget - slot), 1.0)
        threshold = gamma * max(1.0 - head_mass, 0.0) / remaining_budget
        if float(p[token_id]) < threshold:
            break
        accepted.append(int(token_id))
        head_mass += float(p[token_id])
    return np.asarray(accepted, dtype=np.int64)


def union_head_indices(
    student_probabilities: np.ndarray,
    teacher_probabilities: np.ndarray,
    *,
    head_budget: int,
    student_slots: int,
    teacher_slots: int,
) -> np.ndarray:

    student = _normalized_probabilities(student_probabilities, name="student_probabilities")
    teacher = _normalized_probabilities(teacher_probabilities, name="teacher_probabilities")
    if student.shape != teacher.shape:
        raise ValueError("student and teacher probabilities must have matching shapes.")
    head_budget = int(head_budget)
    student_slots = int(student_slots)
    teacher_slots = int(teacher_slots)
    if not 0 <= student_slots <= head_budget or not 0 <= teacher_slots <= head_budget:
        raise ValueError("student_slots and teacher_slots must lie in [0, head_budget].")
    if head_budget < 0 or head_budget > student.size:
        raise ValueError("head_budget must lie in [0, vocab_size].")

    student_order = np.argsort(-student, kind="stable")
    teacher_order = np.argsort(-teacher, kind="stable")
    selected: list[int] = []
    seen: set[int] = set()

    def add(token_ids: np.ndarray) -> None:
        for raw_token_id in token_ids:
            token_id = int(raw_token_id)
            if token_id not in seen and len(selected) < head_budget:
                seen.add(token_id)
                selected.append(token_id)

    add(student_order[:student_slots])
    add(teacher_order[:teacher_slots])
    add(student_order)
    return np.asarray(selected, dtype=np.int64)


def estimator_moments(
    student_probabilities: np.ndarray,
    log_ratio: np.ndarray,
    *,
    head_indices: np.ndarray,
    tail_samples: int,
) -> EstimatorMoments:

    p = _normalized_probabilities(student_probabilities, name="student_probabilities")
    values = np.asarray(log_ratio, dtype=np.float64)
    if values.shape != p.shape or not np.isfinite(values).all():
        raise ValueError("log_ratio must be finite and match student_probabilities.")
    tail_samples = int(tail_samples)
    if tail_samples <= 0:
        raise ValueError("tail_samples must be positive.")

    head = np.asarray(head_indices, dtype=np.int64)
    if head.ndim != 1:
        raise ValueError("head_indices must be one-dimensional.")
    if head.size and (head.min() < 0 or head.max() >= p.size):
        raise ValueError("head_indices contain an out-of-range token id.")
    if np.unique(head).size != head.size:
        raise ValueError("head_indices must not contain duplicates.")

    tail_mask = np.ones(p.size, dtype=bool)
    tail_mask[head] = False
    tail_ids = np.flatnonzero(tail_mask)
    tail_mass = float(p[tail_ids].sum())
    head_mass = max(0.0, 1.0 - tail_mass)

    full_mean = float(np.dot(p, values))
    target_gradient = p * (values - full_mean)
    target_gradient_norm_squared = float(np.dot(target_gradient, target_gradient))
    if tail_mass <= np.finfo(np.float64).tiny:
        return EstimatorMoments(
            scalar_variance=0.0,
            gradient_trace_variance=0.0,
            optimal_baseline=0.0,
            optimal_gradient_trace_variance=0.0,
            target_gradient_norm_squared=target_gradient_norm_squared,
            head_count=int(head.size),
            head_mass=head_mass,
            tail_samples=tail_samples,
        )

    conditional = p[tail_ids] / tail_mass
    tail_values = values[tail_ids]
    tail_mean = float(np.dot(conditional, tail_values))
    scalar_conditional_variance = float(
        np.dot(conditional, np.square(tail_values - tail_mean))
    )
    scalar_variance = tail_mass * tail_mass * scalar_conditional_variance / tail_samples

    p_squared_sum = float(np.dot(p, p))
    score_norm_squared = 1.0 - 2.0 * p[tail_ids] + p_squared_sum

    def conditional_gradient_trace_variance(baseline: float) -> float:
        centered_values = tail_values - float(baseline)
        centered_mean = float(np.dot(conditional, centered_values))
        mean_gradient = -p * centered_mean
        mean_gradient[tail_ids] += conditional * centered_values
        second_moment = float(
            np.dot(conditional, np.square(centered_values) * score_norm_squared)
        )
        return max(second_moment - float(np.dot(mean_gradient, mean_gradient)), 0.0)

    gradient_conditional_variance = conditional_gradient_trace_variance(0.0)
    gradient_trace_variance = (
        tail_mass * tail_mass * gradient_conditional_variance / tail_samples
    )

    score_mean = -p.copy()
    score_mean[tail_ids] += conditional
    value_score_mean = -p * tail_mean
    value_score_mean[tail_ids] += conditional * tail_values
    baseline_quadratic = float(
        np.dot(conditional, score_norm_squared) - np.dot(score_mean, score_mean)
    )
    baseline_linear = float(
        np.dot(conditional, tail_values * score_norm_squared)
        - np.dot(value_score_mean, score_mean)
    )
    optimal_baseline = baseline_linear / baseline_quadratic if baseline_quadratic > 1e-18 else 0.0
    optimal_gradient_conditional_variance = conditional_gradient_trace_variance(optimal_baseline)
    optimal_gradient_trace_variance = (
        tail_mass * tail_mass * optimal_gradient_conditional_variance / tail_samples
    )

    return EstimatorMoments(
        scalar_variance=max(scalar_variance, 0.0),
        gradient_trace_variance=max(gradient_trace_variance, 0.0),
        optimal_baseline=optimal_baseline,
        optimal_gradient_trace_variance=max(optimal_gradient_trace_variance, 0.0),
        target_gradient_norm_squared=target_gradient_norm_squared,
        head_count=int(head.size),
        head_mass=head_mass,
        tail_samples=tail_samples,
    )
