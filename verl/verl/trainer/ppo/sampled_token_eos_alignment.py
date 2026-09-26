from __future__ import annotations

from dataclasses import dataclass

import torch


OPD_SAMPLED_TOKEN_EOS_TEACHER_PRIMARY_LOG_PROBS_KEY = (
    "opd_sampled_token_eos_teacher_primary_log_probs"
)
OPD_SAMPLED_TOKEN_EOS_TEACHER_SECONDARY_LOG_PROBS_KEY = (
    "opd_sampled_token_eos_teacher_secondary_log_probs"
)


@dataclass(frozen=True)
class SampledTokenEOSAlignmentOutput:
    aligned_log_probs: torch.Tensor
    primary_log_probs: torch.Tensor
    secondary_log_probs: torch.Tensor
    combined_log_probs: torch.Tensor
    eos_mask: torch.Tensor


def validate_sampled_token_eos_alignment_configuration(
    *,
    enable: bool,
    log_prob_top_k: int,
    student_eos_token_id: int | None,
    teacher_eos_token_id: int | None,
    use_reward_model: bool,
) -> None:

    if not enable:
        return
    if int(log_prob_top_k) != 0:
        raise ValueError("sampled-token EOS alignment requires log_prob_top_k=0.")
    if student_eos_token_id is None:
        raise ValueError("sampled-token EOS alignment requires a student EOS token id.")
    if teacher_eos_token_id is None:
        raise ValueError("sampled-token EOS alignment requires a teacher EOS token id.")
    if int(student_eos_token_id) < 0 or int(teacher_eos_token_id) < 0:
        raise ValueError("sampled-token EOS alignment token ids must be nonnegative.")
    if int(student_eos_token_id) == int(teacher_eos_token_id):
        raise ValueError("Student and teacher EOS token ids must differ.")
    if not use_reward_model:
        raise ValueError("sampled-token EOS alignment requires the teacher reward model worker.")


def compute_sampled_token_eos_alignment_metrics(
    *,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    student_eos_token_id: int,
    teacher_primary_log_probs: torch.Tensor,
    teacher_secondary_log_probs: torch.Tensor,
) -> dict[str, float]:

    expected_shape = responses.shape
    for name, tensor in (
        ("response_mask", response_mask),
        ("teacher_primary_log_probs", teacher_primary_log_probs),
        ("teacher_secondary_log_probs", teacher_secondary_log_probs),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} must match responses, got {tensor.shape} and {expected_shape}."
            )

    prefix = "opd_sampled_token_eos_alignment/"
    eos_mask = responses.eq(int(student_eos_token_id)) & response_mask.bool()
    eos_count = int(eos_mask.sum().item())
    metrics = {
        f"{prefix}enabled": 1.0,
        f"{prefix}sampled_eos_count": float(eos_count),
        f"{prefix}teacher_primary_probability_mean": 0.0,
        f"{prefix}teacher_secondary_probability_mean": 0.0,
        f"{prefix}teacher_combined_probability_mean": 0.0,
        f"{prefix}log_prob_correction_mean": 0.0,
        f"{prefix}log_prob_correction_max": 0.0,
    }
    if eos_count == 0:
        return metrics

    primary = teacher_primary_log_probs.detach().float()[eos_mask]
    secondary = teacher_secondary_log_probs.detach().float()[eos_mask]
    combined = torch.logaddexp(primary, secondary)
    correction = combined - primary
    metrics.update(
        {
            f"{prefix}teacher_primary_probability_mean": primary.exp().mean().item(),
            f"{prefix}teacher_secondary_probability_mean": secondary.exp().mean().item(),
            f"{prefix}teacher_combined_probability_mean": combined.exp().mean().item(),
            f"{prefix}log_prob_correction_mean": correction.mean().item(),
            f"{prefix}log_prob_correction_max": correction.max().item(),
        }
    )
    return metrics


def align_sampled_token_teacher_log_probs(
    *,
    logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    base_log_probs: torch.Tensor,
    student_eos_token_id: int,
    teacher_eos_token_id: int,
) -> SampledTokenEOSAlignmentOutput:

    if logits.ndim < 2:
        raise ValueError(f"logits must include token and vocabulary dimensions, got {logits.shape}.")
    expected_shape = logits.shape[:-1]
    if sampled_token_ids.shape != expected_shape or base_log_probs.shape != expected_shape:
        raise ValueError(
            "sampled_token_ids and base_log_probs must match logits non-vocabulary dimensions, "
            f"got logits={logits.shape}, sampled={sampled_token_ids.shape}, base={base_log_probs.shape}."
        )

    vocab_size = logits.size(-1)
    student_eos_token_id = int(student_eos_token_id)
    teacher_eos_token_id = int(teacher_eos_token_id)
    for name, token_id in (
        ("Student EOS", student_eos_token_id),
        ("Teacher EOS", teacher_eos_token_id),
    ):
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"{name} token id {token_id} is outside vocabulary size {vocab_size}.")
    if student_eos_token_id == teacher_eos_token_id:
        raise ValueError("Student and teacher EOS token ids must differ.")

    sampled_logits = logits.gather(-1, sampled_token_ids.unsqueeze(-1)).squeeze(-1)
    primary_log_probs = base_log_probs + logits[..., student_eos_token_id] - sampled_logits
    secondary_log_probs = base_log_probs + logits[..., teacher_eos_token_id] - sampled_logits
    combined_log_probs = torch.logaddexp(primary_log_probs, secondary_log_probs)
    eos_mask = sampled_token_ids.eq(student_eos_token_id)
    aligned_log_probs = torch.where(eos_mask, combined_log_probs, base_log_probs)

    return SampledTokenEOSAlignmentOutput(
        aligned_log_probs=aligned_log_probs,
        primary_log_probs=primary_log_probs,
        secondary_log_probs=secondary_log_probs,
        combined_log_probs=combined_log_probs,
        eos_mask=eos_mask,
    )
