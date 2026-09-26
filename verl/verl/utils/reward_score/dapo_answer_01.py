from __future__ import annotations

from verl.utils.reward_score import math_dapo


def reward_func(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    solution = str(solution_str)
    result = dict(math_dapo.compute_score(solution, str(ground_truth)))
    result["score"] = 1.0 if result["acc"] else 0.0
    result["scored_solution"] = solution[-500:]
    return result
