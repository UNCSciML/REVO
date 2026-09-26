from __future__ import annotations

from verl.utils.reward_score import ttrl_math


def reward_func(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    extra_info = extra_info or {}
    prefix_text = str(extra_info.get("prefix_text", "") or "")
    scored_solution = prefix_text + str(solution_str)
    result = ttrl_math.compute_score(scored_solution, str(ground_truth))
    if isinstance(result, dict):
        result = dict(result)
        result["suffix_only"] = solution_str
        result["scored_solution"] = scored_solution[-500:]
        result["prefix_char_len"] = len(prefix_text)
    return result
