from __future__ import annotations

from verl.utils.reward_score import math_dapo


def reward_func(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    result = math_dapo.compute_score(str(solution_str), str(ground_truth))
    if isinstance(result, dict):
        result = dict(result)
        result["scored_solution"] = str(solution_str)[-500:]
    return result
