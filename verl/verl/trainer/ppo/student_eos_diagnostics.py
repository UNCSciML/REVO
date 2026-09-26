import torch


STUDENT_EOS_LOG_PROBS_KEY = "student_eos_log_probs"
STUDENT_EOS_PROB_METRIC = "student/eos_prob_nonterminal_mean"


def eos_log_probs_from_logits(logits: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    if logits.ndim < 1:
        raise ValueError(f"logits must have a vocabulary dimension, got shape {logits.shape}.")
    vocab_size = logits.shape[-1]
    if eos_token_id < 0 or eos_token_id >= vocab_size:
        raise ValueError(f"eos_token_id={eos_token_id} is outside vocabulary size {vocab_size}.")
    return logits[..., eos_token_id] - torch.logsumexp(logits, dim=-1)


def nonterminal_eos_probability_mean(
    *,
    eos_log_probs: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    if eos_log_probs.shape != responses.shape or responses.shape != response_mask.shape:
        raise ValueError(
            "eos_log_probs, responses, and response_mask must have identical shapes, "
            f"got {eos_log_probs.shape}, {responses.shape}, and {response_mask.shape}."
        )
    nonterminal_mask = response_mask.bool() & responses.ne(eos_token_id)
    probabilities = eos_log_probs.float().exp()
    return probabilities.masked_fill(~nonterminal_mask, 0.0).sum() / nonterminal_mask.sum().clamp_min(1)


def record_nonterminal_eos_probability_metric(
    *,
    batch_tensors,
    metrics: dict,
    eos_token_id: int,
) -> None:
    if STUDENT_EOS_LOG_PROBS_KEY not in batch_tensors:
        return
    eos_log_probs = batch_tensors.pop(STUDENT_EOS_LOG_PROBS_KEY)
    value = nonterminal_eos_probability_mean(
        eos_log_probs=eos_log_probs,
        responses=batch_tensors["responses"],
        response_mask=batch_tensors["response_mask"],
        eos_token_id=eos_token_id,
    )
    metrics[STUDENT_EOS_PROB_METRIC] = value.detach().item()
