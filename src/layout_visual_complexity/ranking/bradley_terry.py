"""Bradley–Terry scores by maximum likelihood, with the MM algorithm (Hunter 2004).

A weak prior keeps the estimate finite on any graph, even for an image that always won: each image also plays
`prior` virtual matches against a virtual image of strength 1, half won and half lost.
"""

import math

import numpy as np


def log_strengths(
    winners: np.ndarray,
    losers: np.ndarray,
    items: int,
    prior: float = 1.0,
    tolerance: float = 1e-10,
    max_iterations: int = 100_000,
) -> np.ndarray:
    """Each item's log-strength, centred on zero: P(i beats j) = 1 / (1 + exp(s_j - s_i))."""
    if prior <= 0:
        raise ValueError("the prior must be positive, so that the estimate exists on any graph")
    winners, losers = np.asarray(winners, dtype=np.intp), np.asarray(losers, dtype=np.intp)
    first, second = np.minimum(winners, losers), np.maximum(winners, losers)
    pairs, games = np.unique(np.stack([first, second], axis=1), axis=0, return_counts=True)
    i, j = pairs[:, 0], pairs[:, 1]
    wins = np.bincount(winners, minlength=items) + prior / 2
    strength = np.ones(items)
    for _ in range(max_iterations):
        rate = games / (strength[i] + strength[j])
        denominator = np.bincount(i, rate, items) + np.bincount(j, rate, items) + prior / (strength + 1.0)
        updated = wins / denominator
        change = np.max(np.abs(np.log(updated) - np.log(strength)))
        strength = updated
        if change < tolerance:
            break
    logs = np.log(strength)
    return logs - logs.mean()


def on_elo_scale(logs: np.ndarray, centre: float = 1500.0, scale: float = 400.0) -> np.ndarray:
    """Log-strengths on Elo's scale: `scale / ln 10` points per unit, centred on `centre`."""
    logs = np.asarray(logs, dtype=float)
    return centre + scale / math.log(10.0) * (logs - logs.mean())
