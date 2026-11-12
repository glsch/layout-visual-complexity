"""The evaluation protocol every method is measured with.

On each fold, the pairwise model is fitted on the training judgments and scores the test pages. Scores are z-scored
per fold, with the mean and spread of the training pages' scores, so folds are comparable; a page's out-of-fold
score is the mean of its z-scores over the folds that tested it. Win probabilities are a sigmoid of the score
difference, with its scale fitted on the training judgments.

The reference for accuracy is the consistency of the same test judgments with the final Elo scores; the reference
for SROCC is the split-half ceiling.
"""

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from layout_visual_complexity.evaluation import metrics
from layout_visual_complexity.evaluation.folds import Fold, judgments_within
from layout_visual_complexity.evaluation.model import fit_pairwise_model
from layout_visual_complexity.ranking.reliability import DEFAULT_GAP_EDGES, judgment_gaps, other_image


@dataclass(frozen=True)
class ProtocolSettings:
    """How methods are evaluated.

    Args:
        gap_edges: The edges of the bins of the final Elo gap.
        regularisation: The logistic regression's C: smaller is stronger.
        bootstrap: Resamples of manuscripts for the intervals.
        reliability_bins: Bins of the predicted probability in the reliability curve.
        seed: The seed of the bootstrap and of the learning curves' subsamples.
    """

    gap_edges: tuple[float, ...] = DEFAULT_GAP_EDGES
    regularisation: float = 1.0
    bootstrap: int = 1000
    reliability_bins: int = 10
    seed: int = 1


@dataclass
class Evaluation:
    """A method's results: per fold, by gap, per tested judgment, the out-of-fold scores, and a summary."""

    folds: pd.DataFrame
    by_gap: pd.DataFrame
    tested: pd.DataFrame
    scores: pd.Series
    reliability: pd.DataFrame
    summary: dict[str, float] = field(default_factory=dict)


def _z(scores: pd.Series, centre: float, spread: float) -> pd.Series:
    return (scores - centre) / (spread if spread > 0 else 1.0)


def _gap_columns(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{label}": value for label, value in values.items() if label != "overall"}


def _summarise_by_gap(
    folds: pd.DataFrame, tested: pd.DataFrame, column: str, values: str, edges: tuple
) -> pd.DataFrame:
    """Per gap bin and overall: test judgments per fold, the mean of the fold values, their SD, the pooled value.

    `column` names the per-fold values in `folds`; `values` the per-judgment values in `tested`.
    """
    pooled_values = metrics.by_gap(tested[values].to_numpy(dtype=float), tested["gap"].to_numpy(), edges)
    counts = metrics.counts_by_gap(tested["gap"].to_numpy(), edges)
    rows = []
    for label, pooled in pooled_values.items():
        name = column if label == "overall" else f"{column}_{label}"
        values = folds[name].to_numpy(dtype=float)
        rows.append(
            {
                "gap": label,
                "judgments_per_fold": counts[label] / max(len(folds), 1),
                "fold_mean": float(np.nanmean(values)) if np.isfinite(values).any() else math.nan,
                "fold_sd": float(np.nanstd(values)) if np.isfinite(values).any() else math.nan,
                "pooled": pooled,
            }
        )
    return pd.DataFrame(rows)


def evaluate(
    features: pd.DataFrame,
    judgments: pd.DataFrame,
    reference: pd.Series,
    folds: list[Fold],
    settings: ProtocolSettings | None = None,
) -> Evaluation:
    """Fit and test the pairwise model on every fold. `features` is pages × features; `reference` the final Elo."""
    settings = settings or ProtocolSettings()
    edges = tuple(settings.gap_edges)
    gaps = judgment_gaps(judgments, reference)
    fold_rows, tested_parts, oof = [], [], []
    for number, fold in enumerate(folds):
        row = {"kind": fold.kind, "repeat": fold.repeat, "fold": fold.label, "train": len(fold.train)}
        row["test"] = len(fold.test)
        if not len(fold.train) or not len(fold.test):
            fold_rows.append(row)
            continue
        train, test = judgments.iloc[fold.train], judgments.iloc[fold.test]
        model = fit_pairwise_model(features.loc[fold.train_pages], train, settings.regularisation)
        train_scores = model.scores(features.loc[fold.train_pages])
        centre, spread = float(train_scores.mean()), float(train_scores.std(ddof=0))
        z_train = _z(train_scores, centre, spread)
        z_test = _z(model.scores(features.loc[fold.test_pages]), centre, spread)
        scale = metrics.fit_scale(z_train[train["chosen"]].to_numpy() - z_train[other_image(train)].to_numpy())
        difference = z_test[test["chosen"]].to_numpy() - z_test[other_image(test)].to_numpy()
        left_difference = z_test[test["left"]].to_numpy() - z_test[test["right"]].to_numpy()
        hit = metrics.hits(test, z_test)
        chosen_p = metrics.sigmoid(scale * difference)
        part = pd.DataFrame(
            {
                "fold_number": number,
                "row": fold.test,
                "gap": gaps[fold.test],
                "hit": hit,
                "chosen_probability": chosen_p,
                "left_probability": metrics.sigmoid(scale * left_difference),
                "left_chosen": (test["chosen"] == test["left"]).to_numpy(),
            }
        )
        tested_parts.append(part)
        oof.append(z_test)
        accuracy = metrics.by_gap(hit, gaps[fold.test], edges)
        row.update(accuracy=accuracy["overall"], log_loss=metrics.log_loss(chosen_p), scale=scale)
        row.update(_gap_columns("accuracy", accuracy))
        fold_rows.append(row)
    folds_table = pd.DataFrame(fold_rows)
    tested = pd.concat(tested_parts, ignore_index=True) if tested_parts else pd.DataFrame()
    scores = pd.concat(oof).groupby(level=0).mean() if oof else pd.Series(dtype=float)
    by_gap = _summarise_by_gap(folds_table, tested, "accuracy", "hit", edges) if len(tested) else pd.DataFrame()
    reliability = metrics.reliability_curve(
        tested["left_probability"].to_numpy(), tested["left_chosen"].to_numpy(), settings.reliability_bins
    )
    summary = {
        "folds": len(folds),
        "accuracy": float(folds_table["accuracy"].mean()),
        "accuracy_sd": float(folds_table["accuracy"].std(ddof=0)),
        "accuracy_pooled": float(tested["hit"].mean()),
        "srocc": metrics.srocc(scores, reference),
        "log_loss": metrics.log_loss(tested["chosen_probability"].to_numpy()),
    }
    return Evaluation(folds_table, by_gap, tested, scores, reliability, summary)


def consistency_reference(
    judgments: pd.DataFrame, reference: pd.Series, folds: list[Fold], edges: tuple[float, ...] = DEFAULT_GAP_EDGES
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The consistency of each fold's test judgments with the final Elo scores: per fold, and by gap.

    The summary by gap gives the mean of the fold values (as the article reports it), their SD and the pooled value.
    """
    gaps = judgment_gaps(judgments, reference)
    agrees = metrics.hits(judgments, reference)
    fold_rows, tested = [], []
    for number, fold in enumerate(folds):
        values = metrics.by_gap(agrees[fold.test], gaps[fold.test], edges)
        row = {"kind": fold.kind, "repeat": fold.repeat, "fold": fold.label, "test": len(fold.test)}
        row["consistency"] = values["overall"]
        row.update(_gap_columns("consistency", values))
        fold_rows.append(row)
        tested.append(pd.DataFrame({"fold_number": number, "gap": gaps[fold.test], "consistency": agrees[fold.test]}))
    per_fold = pd.DataFrame(fold_rows)
    pooled = pd.concat(tested, ignore_index=True)
    return per_fold, _summarise_by_gap(per_fold, pooled, "consistency", "consistency", edges)


def per_judgment_hits(evaluation: Evaluation, count: int) -> np.ndarray:
    """Each judgment's hit, averaged over the folds that tested it; NaN for judgments never tested."""
    tested = evaluation.tested
    sums = np.bincount(tested["row"], tested["hit"], minlength=count)
    times = np.bincount(tested["row"], minlength=count)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(times > 0, sums / times, np.nan)


def bootstrap(
    evaluation: Evaluation,
    judgments: pd.DataFrame,
    reference: pd.Series,
    manuscripts: pd.Series,
    settings: ProtocolSettings | None = None,
) -> dict[str, float]:
    """95% intervals for the pooled accuracy and the SROCC, resampling manuscripts with replacement."""
    settings = settings or ProtocolSettings()
    names = sorted(manuscripts.unique())
    number = {name: index for index, name in enumerate(names)}
    multiplicities = metrics.manuscript_multiplicities(names, settings.bootstrap, settings.seed)
    hit = per_judgment_hits(evaluation, len(judgments))
    tested = np.isfinite(hit)
    left = judgments["left"].map(manuscripts).map(number).to_numpy()[tested]
    right = judgments["right"].map(manuscripts).map(number).to_numpy()[tested]
    accuracy = metrics.bootstrap_accuracy(hit[tested], left, right, multiplicities)
    pages = evaluation.scores.index.intersection(reference.index)
    page_manuscript = manuscripts.reindex(pages).map(number).to_numpy()
    srocc = metrics.bootstrap_srocc(
        evaluation.scores[pages].to_numpy(), reference[pages].to_numpy(), page_manuscript, multiplicities
    )
    (a_low, a_high), (s_low, s_high) = metrics.interval(accuracy), metrics.interval(srocc)
    return {"accuracy_low": a_low, "accuracy_high": a_high, "srocc_low": s_low, "srocc_high": s_high}


def fold_mean_intervals(
    values: pd.DataFrame,
    judgments: pd.DataFrame,
    manuscripts: pd.Series,
    edges: tuple[float, ...] = DEFAULT_GAP_EDGES,
    settings: ProtocolSettings | None = None,
) -> pd.DataFrame:
    """95% intervals of a mean over folds (each fold's share of hits), by gap bin and overall, resampling manuscripts.

    `values` has one row per tested judgment: `fold_number`, `row` (the judgment's position in `judgments`), `gap`
    and `value` (a hit: 1, 0.5 or 0). In each resample a judgment is weighted as `metrics.judgment_weights` says, in
    the same resamples as `bootstrap` draws."""
    settings = settings or ProtocolSettings()
    names = sorted(manuscripts.unique())
    number = {name: index for index, name in enumerate(names)}
    multiplicities = metrics.manuscript_multiplicities(names, settings.bootstrap, settings.seed)
    rows = values["row"].to_numpy()
    left = judgments["left"].iloc[rows].map(manuscripts).map(number).to_numpy()
    right = judgments["right"].iloc[rows].map(manuscripts).map(number).to_numpy()
    weights = metrics.judgment_weights(multiplicities, left, right)
    _, fold = np.unique(values["fold_number"].to_numpy(), return_inverse=True)
    one_hot = np.zeros((len(values), fold.max() + 1 if len(fold) else 0))
    one_hot[np.arange(len(values)), fold] = 1.0
    hit = values["value"].to_numpy(dtype=float)
    bins = metrics.gap_bin(values["gap"].to_numpy(), edges)
    selections = [(metrics.gap_label(low, high), bins == n) for n, (low, high) in enumerate(metrics.gap_bins(edges))]
    found = []
    for label, chosen in [*selections, ("overall", np.ones(len(values), dtype=bool))]:
        weight, folds = weights[:, chosen], one_hot[chosen]
        with np.errstate(invalid="ignore", divide="ignore"):
            per_fold = ((weight * hit[chosen]) @ folds) / (weight @ folds)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(per_fold, axis=1) if per_fold.shape[1] else np.full(len(weight), math.nan)
        low, high = metrics.interval(means)
        found.append({"gap": label, "low": low, "high": high})
    return pd.DataFrame(found)


def paired_comparison(
    first: Evaluation,
    second: Evaluation,
    judgments: pd.DataFrame,
    manuscripts: pd.Series,
    settings: ProtocolSettings | None = None,
) -> dict[str, float]:
    """The first method's accuracy minus the second's, on the same folds: the mean of the per-fold differences, and
    a 95% interval of the pooled difference from the same manuscript resamples for both."""
    settings = settings or ProtocolSettings()
    if len(first.folds) != len(second.folds):
        raise ValueError("the two evaluations ran on different folds")
    per_fold = first.folds["accuracy"].to_numpy(dtype=float) - second.folds["accuracy"].to_numpy(dtype=float)
    names = sorted(manuscripts.unique())
    number = {name: index for index, name in enumerate(names)}
    multiplicities = metrics.manuscript_multiplicities(names, settings.bootstrap, settings.seed)
    a, b = per_judgment_hits(first, len(judgments)), per_judgment_hits(second, len(judgments))
    both = np.isfinite(a) & np.isfinite(b)
    left = judgments["left"].map(manuscripts).map(number).to_numpy()[both]
    right = judgments["right"].map(manuscripts).map(number).to_numpy()[both]
    difference = metrics.bootstrap_accuracy(a[both] - b[both], left, right, multiplicities)
    low, high = metrics.interval(difference)
    return {
        "difference": float(np.nanmean(per_fold)),
        "difference_pooled": float(np.mean(a[both] - b[both])),
        "low": low,
        "high": high,
    }


def learning_curve(
    features: pd.DataFrame,
    judgments: pd.DataFrame,
    reference: pd.Series,
    folds: list[Fold],
    shares: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0),
    by: str = "judgments",
    settings: ProtocolSettings | None = None,
) -> pd.DataFrame:
    """Accuracy on the same test judgments when training on a share of the training judgments or of the training
    images (then on the judgments between the images kept). One row per share: the mean over folds."""
    settings = settings or ProtocolSettings()
    if by not in {"judgments", "images"}:
        raise ValueError(f"unknown learning curve {by!r}: expected judgments or images")
    rows = []
    for share_number, share in enumerate(shares):
        reduced = []
        for number, fold in enumerate(folds):
            rng = np.random.default_rng([settings.seed, share_number, number])
            if by == "judgments":
                keep = max(1, round(share * len(fold.train)))
                train = np.sort(rng.choice(fold.train, size=keep, replace=False)) if len(fold.train) else fold.train
                train_pages = fold.train_pages
            else:
                keep = max(2, round(share * len(fold.train_pages)))
                train_pages = rng.choice(fold.train_pages, size=min(keep, len(fold.train_pages)), replace=False)
                within = judgments_within(judgments, train_pages)
                train = np.intersect1d(within, fold.train)
            reduced.append(
                Fold(fold.kind, fold.repeat, fold.index, fold.label, train_pages, fold.test_pages, train, fold.test)
            )
        result = evaluate(features, judgments, reference, reduced, settings)
        rows.append(
            {
                "by": by,
                "share": share,
                "train_per_fold": float(result.folds["train"].mean()),
                "accuracy": result.summary["accuracy"],
                "accuracy_sd": result.summary["accuracy_sd"],
                "srocc": result.summary["srocc"],
            }
        )
    return pd.DataFrame(rows)
