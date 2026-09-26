import math
from dataclasses import dataclass

import torch


TEACHER_DEFICIT_RESIDUAL_IDS_KEY = "teacher_deficit_residual_ids"
TEACHER_DEFICIT_RESIDUAL_LOG_PROBS_KEY = "teacher_deficit_residual_log_probs"


@dataclass(frozen=True)
class TeacherDeficitResidualLossOutput:
    loss_matrix: torch.Tensor
    weights: torch.Tensor
    active_mask: torch.Tensor


@dataclass(frozen=True)
class TeacherResidualRemapOutput:
    candidate_ids: torch.Tensor
    teacher_log_probs: torch.Tensor
    terminal_mask: torch.Tensor
    secondary_samples_remapped: torch.Tensor


@dataclass(frozen=True)
class TeacherResidualSampleOutput:
    candidate_ids: torch.Tensor
    teacher_log_probs: torch.Tensor


def sample_teacher_residual_candidates(
    logits: torch.Tensor,
    num_samples: int,
    *,
    generator: torch.Generator | None = None,
    log_normalizer: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
    selected_rows: torch.Tensor | None = None,
) -> TeacherResidualSampleOutput:
    if logits.ndim != 2:
        raise ValueError(f"teacher residual logits must have shape (tokens, vocab), got {logits.shape}.")
    if int(num_samples) < 1:
        raise ValueError("teacher residual num_samples must be positive.")
    if not logits.is_floating_point():
        raise ValueError("teacher residual logits must use a floating-point dtype.")

    if sample_mask is not None and selected_rows is not None:
        raise ValueError("teacher residual sampling accepts either sample_mask or selected_rows, not both.")
    if sample_mask is None and selected_rows is None:
        source_logits = logits.detach().clone()
    else:
        if selected_rows is None:
            if sample_mask.shape != logits.shape[:-1]:
                raise ValueError(
                    "teacher residual sample mask must match the token dimension, "
                    f"got mask={sample_mask.shape}, logits={logits.shape}."
                )
            selected_rows = sample_mask.to(device=logits.device, dtype=torch.bool).nonzero(
                as_tuple=False
            ).squeeze(-1)
        else:
            if selected_rows.ndim != 1:
                raise ValueError(
                    "teacher residual selected rows must be one-dimensional, "
                    f"got {selected_rows.shape}."
                )
            selected_rows = selected_rows.to(device=logits.device, dtype=torch.long)
        source_logits = logits.detach().index_select(0, selected_rows)

    if source_logits.size(0) == 0:
        empty_ids = torch.empty(
            (0, int(num_samples)),
            dtype=torch.long,
            device=logits.device,
        )
        return TeacherResidualSampleOutput(
            candidate_ids=empty_ids,
            teacher_log_probs=logits.new_empty(empty_ids.shape),
        )
    if log_normalizer is None:
        log_normalizer = torch.logsumexp(source_logits, dim=-1, keepdim=True)
    elif log_normalizer.shape != source_logits.shape[:-1] + (1,):
        raise ValueError(
            "teacher residual log normalizer must have shape (tokens, 1), "
            f"got normalizer={log_normalizer.shape}, selected_logits={source_logits.shape}."
        )

    sampling_weights = source_logits.detach()
    sampling_weights.sub_(sampling_weights.max(dim=-1, keepdim=True).values)
    sampling_weights.exp_()
    candidate_ids = torch.multinomial(
        sampling_weights,
        num_samples=int(num_samples),
        replacement=True,
        generator=generator,
    )
    if selected_rows is None:
        sampled_logits = logits.gather(dim=-1, index=candidate_ids)
    else:
        sampled_logits = logits[selected_rows.unsqueeze(-1), candidate_ids]
    teacher_log_probs = sampled_logits - log_normalizer
    return TeacherResidualSampleOutput(
        candidate_ids=candidate_ids,
        teacher_log_probs=teacher_log_probs,
    )


def prepare_teacher_deficit_residual_loss(
    current_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    prefix_weights: torch.Tensor | None = None,
) -> TeacherDeficitResidualLossOutput:
    if current_log_probs.shape != teacher_log_probs.shape:
        raise ValueError(
            "current and teacher residual log-probabilities must have the same shape, "
            f"got current={current_log_probs.shape}, teacher={teacher_log_probs.shape}."
        )
    if current_log_probs.ndim != 3:
        raise ValueError(
            "teacher residual log-probabilities must have shape (batch, sequence, samples), "
            f"got {current_log_probs.shape}."
        )

    work_dtype = (
        torch.float64
        if current_log_probs.dtype == torch.float64 or teacher_log_probs.dtype == torch.float64
        else torch.float32
    )
    detached_log_ratio = (
        current_log_probs.detach().to(work_dtype) - teacher_log_probs.detach().to(work_dtype)
    ).clamp(max=0.0)
    weights = (-torch.expm1(detached_log_ratio)).detach()
    loss_matrix = -(weights * current_log_probs.to(work_dtype)).mean(dim=-1)

    if prefix_weights is not None:
        if prefix_weights.shape != current_log_probs.shape[:-1]:
            raise ValueError(
                "teacher residual prefix weights must match (batch, sequence), "
                f"got weights={prefix_weights.shape}, log_probs={current_log_probs.shape}."
            )
        loss_matrix = loss_matrix * prefix_weights.to(
            device=loss_matrix.device,
            dtype=loss_matrix.dtype,
        )

    return TeacherDeficitResidualLossOutput(
        loss_matrix=loss_matrix,
        weights=weights,
        active_mask=detached_log_ratio < 0.0,
    )


def remap_teacher_residual_samples(
    *,
    candidate_ids: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    teacher_primary_log_probs: torch.Tensor,
    teacher_secondary_log_probs: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    primary_token_id: int,
    secondary_token_id: int,
) -> TeacherResidualRemapOutput:
    if candidate_ids.shape != teacher_log_probs.shape or candidate_ids.ndim != 3:
        raise ValueError(
            "teacher residual ids and log-probabilities must share shape (batch, sequence, samples), "
            f"got ids={candidate_ids.shape}, log_probs={teacher_log_probs.shape}."
        )
    expected_token_shape = candidate_ids.shape[:-1]
    for name, value in (
        ("teacher_primary_log_probs", teacher_primary_log_probs),
        ("teacher_secondary_log_probs", teacher_secondary_log_probs),
        ("responses", responses),
        ("response_mask", response_mask),
    ):
        if value.shape != expected_token_shape:
            raise ValueError(
                f"{name} must match teacher residual token shape {expected_token_shape}, got {value.shape}."
            )

    terminal_mask = response_mask.bool() & responses.eq(int(primary_token_id))
    terminal_candidates = terminal_mask.unsqueeze(-1)
    original_primary = candidate_ids.eq(int(primary_token_id))
    original_secondary = candidate_ids.eq(int(secondary_token_id))
    secondary_samples = terminal_candidates & original_secondary
    remapped_stop_samples = terminal_candidates & (original_primary | original_secondary)

    remapped_ids = candidate_ids.clone()
    remapped_ids.masked_fill_(secondary_samples, int(primary_token_id))

    combined_stop_log_probs = torch.logaddexp(
        teacher_primary_log_probs,
        teacher_secondary_log_probs,
    ).unsqueeze(-1)
    remapped_log_probs = torch.where(
        remapped_stop_samples,
        combined_stop_log_probs,
        teacher_log_probs,
    )

    return TeacherResidualRemapOutput(
        candidate_ids=remapped_ids,
        teacher_log_probs=remapped_log_probs,
        terminal_mask=terminal_mask,
        secondary_samples_remapped=secondary_samples.sum(),
    )


def validate_teacher_deficit_residual_configuration(
    *,
    enabled: bool,
    teacher_sample_count: int,
    coefficient: float,
    candidate_mode: object,
    top_k: int,
    top_k_strategy: object,
    advantage_mode: object,
    sample_replacement: bool,
    opd_loss_type: object,
) -> None:
    if not enabled:
        return

    if int(teacher_sample_count) < 1:
        raise ValueError("teacher deficit residual sample count must be positive.")
    if not math.isfinite(float(coefficient)) or float(coefficient) <= 0.0:
        raise ValueError("teacher deficit residual coefficient must be finite and positive.")

    normalized_candidate_mode = str(candidate_mode).strip().lower().replace("-", "_")
    normalized_candidate_mode = {
        "sample": "sample_stu",
        "sample_k": "sample_stu",
        "sample_student": "sample_stu",
        "student_sample": "sample_stu",
    }.get(normalized_candidate_mode, normalized_candidate_mode)
    if normalized_candidate_mode != "sample_stu":
        raise ValueError("teacher deficit residual requires candidate_mode=sample_stu.")
    if int(top_k) < 1:
        raise ValueError("teacher deficit residual requires log_prob_top_k > 0.")
    if str(top_k_strategy).strip().lower().replace("-", "_") != "only_stu":
        raise ValueError("teacher deficit residual requires top_k_strategy=only_stu.")

    normalized_advantage_mode = str(advantage_mode).strip().lower().replace("-", "_")
    if normalized_advantage_mode not in {"current_kl", "current_kl_is"}:
        raise ValueError("teacher deficit residual requires current_kl(_is) advantages.")
    if not sample_replacement:
        raise ValueError("teacher deficit residual requires student Sample-K sampling with replacement.")
    normalized_loss_type = str(opd_loss_type).strip().lower().replace("-", "_")
    normalized_loss_type = {
        "kl": "sample_k_reverse_kl",
        "reverse_kl": "sample_k_reverse_kl",
        "rkl": "sample_k_reverse_kl",
        "sample_k_rkl": "sample_k_reverse_kl",
        "sample_k_kl": "sample_k_reverse_kl",
    }.get(normalized_loss_type, normalized_loss_type)
    if normalized_loss_type != "sample_k_reverse_kl":
        raise ValueError("teacher deficit residual requires opd_loss_type=sample_k_reverse_kl.")
