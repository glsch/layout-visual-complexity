"""The split-half reliability of the Elo scores, and the ceiling for SROCC: how well Elo scores from half the annotators
predict those from the other half.

Each repeat splits the persons at random into two halves, computes Elo scores from each half's judgments, and
correlates them (Spearman) over the images judged in both. The Spearman–Brown formula, 2r / (1 + r), corrects the
correlation to what the full set of annotators would reach: its median over repeats is the reliability of the Elo
scores, the correlation expected between two independent runs of the annotation. A method's scores are compared with
one run, not two, and carry none of its noise: the most they can reach is the reliability's square root, the
correlation of the observed scores with the true ones. That is the ceiling for a method's SROCC against the Elo
scores, as consistency is the reference for its accuracy.
"""

import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from layout_visual_complexity.ranking.elo import EloSettings, permuted_scores, workers
from layout_visual_complexity.ranking.reliability import spearman

RELIABILITY = "Spearman-Brown corrected"
CEILING = "ceiling for SROCC"


def ceiling(reliability: float) -> float:
    """The ceiling for a method's SROCC against scores of this reliability: its square root (0 when it is negative)."""
    return math.sqrt(max(reliability, 0.0)) if math.isfinite(reliability) else math.nan


@dataclass(frozen=True)
class SplitHalfSettings:
    """How the split-half ceiling is computed.

    Args:
        repeats: How many random splits of the annotators.
        permutations: Elo permutations per half.
        seed: The seed of the splits and of the halves' permutations.
        balance: `judgments` puts each person, in random order, into the half with fewer judgments so far, so the
            halves hold about as many judgments; `persons` gives each half the same number of persons.
    """

    repeats: int = 100
    permutations: int = 100
    seed: int = 1
    balance: Literal["judgments", "persons"] = "judgments"


def spearman_brown(r: float) -> float:
    """The reliability of the full set from the correlation r between its two halves."""
    return 2 * r / (1 + r)


def split_persons(counts: pd.Series, rng: np.random.Generator, balance: str) -> np.ndarray:
    """Which half (0 or 1) each person goes to; `counts` holds each person's number of judgments."""
    order = rng.permutation(len(counts))
    half = np.zeros(len(counts), dtype=int)
    if balance == "persons":
        half[order[len(order) // 2 :]] = 1
        return half
    if balance != "judgments":
        raise ValueError(f"unknown balance {balance!r}: expected judgments or persons")
    load = [0, 0]
    values = counts.to_numpy()
    for person in order:
        side = int(load[1] < load[0])
        half[person] = side
        load[side] += int(values[person])
    return half


def _one_repeat(
    repeat: int,
    winners: np.ndarray,
    losers: np.ndarray,
    person_of_judgment: np.ndarray,
    counts: pd.Series,
    items: int,
    settings: SplitHalfSettings,
    elo: EloSettings,
) -> dict[str, float]:
    rng = np.random.default_rng([settings.seed, repeat])
    half_of_person = split_persons(counts, rng, settings.balance)
    half = half_of_person[person_of_judgment]
    means, judged = [], []
    for side in (0, 1):
        mask = half == side
        seed = int(np.random.SeedSequence([settings.seed, repeat, side]).generate_state(1)[0])
        scores = permuted_scores(winners[mask], losers[mask], items, settings.permutations, seed, elo, jobs=1)
        means.append(scores.mean(axis=0))
        judged.append(np.bincount(np.concatenate([winners[mask], losers[mask]]), minlength=items) > 0)
    both = judged[0] & judged[1]
    r = spearman(means[0][both], means[1][both])
    return {
        "repeat": repeat,
        "persons_a": int((half_of_person == 0).sum()),
        "persons_b": int((half_of_person == 1).sum()),
        "judgments_a": int((half == 0).sum()),
        "judgments_b": int((half == 1).sum()),
        "images": int(both.sum()),
        "spearman": r,
        "spearman_brown": spearman_brown(r),
    }


def split_half_correlations(
    judgments: pd.DataFrame,
    images: list[str],
    settings: SplitHalfSettings | None = None,
    elo: EloSettings | None = None,
    jobs: int = 0,
) -> pd.DataFrame:
    """One row per repeat: the halves' sizes, the images judged in both, the correlation and its correction.

    Repeats run in a pool of `jobs` processes (0: all cores); the result is the same for any number.
    """
    settings, elo = settings or SplitHalfSettings(), elo or EloSettings()
    index = {image: number for number, image in enumerate(images)}
    winners = judgments["chosen"].map(index).to_numpy(dtype=np.intp)
    other = judgments["right"].where(judgments["chosen"] == judgments["left"], judgments["left"])
    losers = other.map(index).to_numpy(dtype=np.intp)
    persons = list(dict.fromkeys(judgments["annotator"]))
    person_of_judgment = judgments["annotator"].map({p: n for n, p in enumerate(persons)}).to_numpy()
    counts = pd.Series(np.bincount(person_of_judgment, minlength=len(persons)), index=persons)
    arguments = (winners, losers, person_of_judgment, counts, len(images), settings, elo)
    jobs = min(workers(jobs), settings.repeats)
    if jobs <= 1:
        rows = [_one_repeat(repeat, *arguments) for repeat in range(settings.repeats)]
    else:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_one_repeat, repeat, *arguments) for repeat in range(settings.repeats)]
            rows = [future.result() for future in futures]
    return pd.DataFrame(rows)


def summarise(repeats: pd.DataFrame) -> dict[str, float]:
    """The median and the 5–95% interval of the correlation and of its correction."""
    summary: dict[str, float] = {"repeats": len(repeats)}
    for column in ["spearman", "spearman_brown"]:
        values = repeats[column].to_numpy(dtype=float)
        summary[f"{column}_median"] = float(np.median(values))
        summary[f"{column}_q05"] = float(np.quantile(values, 0.05))
        summary[f"{column}_q95"] = float(np.quantile(values, 0.95))
    for column in ["judgments_a", "judgments_b", "images"]:
        summary[f"{column}_median"] = float(repeats[column].median())
    return summary
