"""Elo scores over random permutations of the judgments' order.

Each judgment is a match the chosen image wins. Elo's updates depend on the order of the matches, so the scores are
computed over M permutations: permutation m is drawn from `numpy.random.default_rng([seed, m])`, and the scores of
a block of permutations are computed together, one judgment per step across the block. Blocks are a fixed size
whatever the number of workers, so the results don't depend on it.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

BLOCK = 25


@dataclass(frozen=True)
class EloSettings:
    """Elo's parameters, as in the article's equation 1.

    Args:
        k: How far one judgment moves the scores.
        start: Every image's score before the first judgment.
        scale: The score difference at which the higher one is expected to win ten times out of eleven.
    """

    k: float = 100.0
    start: float = 1500.0
    scale: float = 400.0


def expected(score: float | np.ndarray, other: float | np.ndarray, scale: float = 400.0) -> float | np.ndarray:
    """The probability that an image with `score` beats one with `other`."""
    return 1.0 / (1.0 + 10.0 ** ((other - score) / scale))


def play(
    winners: np.ndarray,
    losers: np.ndarray,
    orders: np.ndarray,
    items: int,
    settings: EloSettings | None = None,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Final scores for each order of the judgments: an array of shape (orders, items).

    `winners` and `losers` hold item indices, one per judgment; `orders` holds one order of the judgments per row.
    """
    settings = settings or EloSettings()
    orders = np.atleast_2d(np.asarray(orders))
    count = orders.shape[0]
    scores = np.empty((count, items), dtype=float)
    scores[:] = settings.start if initial is None else np.asarray(initial, dtype=float)
    rows = np.arange(count)
    for step in range(orders.shape[1]):
        judged = orders[:, step]
        winner, loser = winners[judged], losers[judged]
        won, lost = scores[rows, winner], scores[rows, loser]
        change = settings.k / (1.0 + 10.0 ** ((won - lost) / settings.scale))
        scores[rows, winner] = won + change
        scores[rows, loser] = lost - change
    return scores


def permutation(judgments: int, seed: int, index: int) -> np.ndarray:
    """The order of the judgments in permutation `index`."""
    return np.random.default_rng([seed, index]).permutation(judgments)


def _play_block(
    winners: np.ndarray, losers: np.ndarray, items: int, seed: int, indices: range, settings: EloSettings
) -> np.ndarray:
    orders = np.stack([permutation(len(winners), seed, index) for index in indices])
    return play(winners, losers, orders, items, settings)


def workers(jobs: int) -> int:
    """The number of worker processes: `jobs`, or all cores when it is 0."""
    if jobs > 0:
        return jobs
    count = os.process_cpu_count() if hasattr(os, "process_cpu_count") else os.cpu_count()
    return max(1, count or 1)


def permuted_scores(
    winners: np.ndarray,
    losers: np.ndarray,
    items: int,
    permutations: int,
    seed: int,
    settings: EloSettings | None = None,
    jobs: int = 0,
) -> np.ndarray:
    """Final scores in each of `permutations` random orders: an array of shape (permutations, items).

    Runs in a pool of `jobs` processes (0: all cores); the result is the same for any number.
    """
    settings = settings or EloSettings()
    winners, losers = np.asarray(winners, dtype=np.intp), np.asarray(losers, dtype=np.intp)
    blocks = [range(start, min(start + BLOCK, permutations)) for start in range(0, permutations, BLOCK)]
    jobs = min(workers(jobs), len(blocks))
    if jobs <= 1:
        parts = [_play_block(winners, losers, items, seed, block, settings) for block in blocks]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
            futures = [pool.submit(_play_block, winners, losers, items, seed, b, settings) for b in blocks]
            parts = [future.result() for future in futures]
    return np.concatenate(parts) if parts else np.empty((0, items))
