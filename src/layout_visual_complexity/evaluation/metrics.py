"""Metrics of page scores against judgments: pairwise accuracy, SROCC, log-loss, calibration, and intervals."""

import math

import numpy as np
import pandas as pd
from scipy import optimize, special

from layout_visual_complexity.ranking.reliability import gap_bin, gap_bins, gap_label, other_image, spearman


def hits(judgments: pd.DataFrame, scores: pd.Series) -> np.ndarray:
    """Per judgment: 1 where the chosen page scores higher, 0.5 on a tie, 0 otherwise."""
    chosen = judgments["chosen"].map(scores).to_numpy(dtype=float)
    other = other_image(judgments).map(scores).to_numpy(dtype=float)
    return (chosen > other) + 0.5 * (chosen == other)


def by_gap(values: np.ndarray, gaps: np.ndarray, edges: tuple[float, ...]) -> dict[str, float]:
    """The mean of `values` in each gap bin, and overall."""
    bins = gap_bin(gaps, edges)
    means = {}
    for number, (low, high) in enumerate(gap_bins(edges)):
        here = bins == number
        means[gap_label(low, high)] = float(values[here].mean()) if here.any() else math.nan
    means["overall"] = float(values.mean()) if len(values) else math.nan
    return means


def counts_by_gap(gaps: np.ndarray, edges: tuple[float, ...]) -> dict[str, int]:
    """How many judgments fall in each gap bin, and overall."""
    bins = gap_bin(gaps, edges)
    counts = {gap_label(low, high): int((bins == n).sum()) for n, (low, high) in enumerate(gap_bins(edges))}
    counts["overall"] = len(gaps)
    return counts


def srocc(scores: pd.Series, reference: pd.Series) -> float:
    """Spearman's correlation between scores and reference scores, over the pages both have."""
    both = pd.concat([scores, reference], axis=1, join="inner").dropna()
    return spearman(both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy())


def sigmoid(values: np.ndarray) -> np.ndarray:
    return special.expit(values)


def fit_scale(differences: np.ndarray, bound: float = 100.0) -> float:
    """The scale s that makes sigmoid(s × difference) the most likely probability that the chosen page wins.

    `differences` holds chosen-minus-other scores; s is searched within ±`bound`.
    """
    differences = np.asarray(differences, dtype=float)
    if not len(differences) or not np.any(differences):
        return 0.0

    def loss(scale: float) -> float:
        return float(np.sum(np.logaddexp(0.0, -scale * differences)))

    return float(optimize.minimize_scalar(loss, bounds=(-bound, bound), method="bounded").x)


def log_loss(chosen_probabilities: np.ndarray) -> float:
    """The mean negative log-probability of the observed choices."""
    p = np.clip(np.asarray(chosen_probabilities, dtype=float), 1e-12, 1.0)
    return float(-np.mean(np.log(p))) if len(p) else math.nan


def reliability_curve(left_probabilities: np.ndarray, left_chosen: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted against observed: per bin of the predicted probability that the left page is chosen, the mean
    prediction and the share of judgments that did choose it."""
    p = np.asarray(left_probabilities, dtype=float)
    observed = np.asarray(left_chosen, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    rows = []
    for number in range(bins):
        here = index == number
        rows.append(
            {
                "low": edges[number],
                "high": edges[number + 1],
                "judgments": int(here.sum()),
                "predicted": float(p[here].mean()) if here.any() else math.nan,
                "observed": float(observed[here].mean()) if here.any() else math.nan,
            }
        )
    return pd.DataFrame(rows)


def manuscript_multiplicities(manuscripts: list[str], resamples: int, seed: int) -> np.ndarray:
    """How often each manuscript is drawn in each bootstrap resample: an array of shape (resamples, manuscripts)."""
    rng = np.random.default_rng(seed)
    count = len(manuscripts)
    draws = rng.integers(0, count, size=(resamples, count))
    return np.stack([np.bincount(row, minlength=count) for row in draws])


def judgment_weights(
    multiplicities: np.ndarray, left_manuscript: np.ndarray, right_manuscript: np.ndarray
) -> np.ndarray:
    """Each judgment's weight in each resample: the product of its two pages' manuscripts' multiplicities, or, for two
    pages of one manuscript, that manuscript's multiplicity. A judgment then counts as often as its manuscripts' copies
    hold it: m² would also count each copy's pages against another copy's, doubling the weight of the judgments within
    a manuscript on average."""
    left = multiplicities[:, left_manuscript]
    right = multiplicities[:, right_manuscript]
    return np.where(np.asarray(left_manuscript) == np.asarray(right_manuscript), left, left * right)


def bootstrap_accuracy(
    hit_values: np.ndarray,
    left_manuscript: np.ndarray,
    right_manuscript: np.ndarray,
    multiplicities: np.ndarray,
) -> np.ndarray:
    """The accuracy in each resample, each judgment weighted as `judgment_weights` says."""
    weights = judgment_weights(multiplicities, left_manuscript, right_manuscript)
    totals = weights.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (weights * np.asarray(hit_values, dtype=float)).sum(axis=1) / totals


def bootstrap_srocc(
    scores: np.ndarray, reference: np.ndarray, page_manuscript: np.ndarray, multiplicities: np.ndarray
) -> np.ndarray:
    """SROCC in each resample, each page repeated as often as its manuscript was drawn."""
    values = []
    for row in multiplicities:
        repeat = row[page_manuscript]
        values.append(spearman(np.repeat(scores, repeat), np.repeat(reference, repeat)))
    return np.array(values)


def interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """The percentile interval of bootstrap values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    tail = (1 - level) / 2
    return float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))
