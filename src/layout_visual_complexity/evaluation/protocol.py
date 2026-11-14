"""The evaluation protocol every method is measured with.

On each fold, a method is trained on the training judgments and scores the training and test pages: the pairwise
model on a feature table, the local ViT head on tokens, or an ensemble of earlier evaluations. Scores are z-scored per
fold, with the mean and spread of the training pages' scores, so folds are comparable; a page's out-of-fold score is
the mean of its z-scores over the folds that tested it. Win probabilities are a sigmoid of the score difference, with
its scale fitted on the training judgments. An ensemble combines its members' z-scores fold by fold, with weights
given or fitted on the fold's training judgments.

The reference for accuracy is the consistency of the same test judgments with the final Elo scores; the reference
for SROCC is its ceiling, the square root of the Elo scores' split-half reliability.
"""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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
    """A method's results: per fold, by gap, per tested judgment, the out-of-fold scores, and a summary; and each
    fold's z-scores of its training and test pages (`fold_scores`: `fold_number`, `page`, `test`, `z`), which an
    ensemble combines."""

    folds: pd.DataFrame
    by_gap: pd.DataFrame
    tested: pd.DataFrame
    scores: pd.Series
    reliability: pd.DataFrame
    summary: dict[str, float] = field(default_factory=dict)
    fold_scores: pd.DataFrame = field(default_factory=pd.DataFrame)


# Each fold's raw scores of its training and test pages, None where the fold has no training or no test judgments.
FoldScores = list[pd.Series | None]


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


def pairwise_fold_scores(
    features: pd.DataFrame, judgments: pd.DataFrame, folds: list[Fold], regularisation: float = 1.0
) -> FoldScores:
    """The pairwise model fitted on each fold's training judgments: its raw scores of the fold's pages."""
    found: FoldScores = []
    for fold in folds:
        if not len(fold.train) or not len(fold.test):
            found.append(None)
            continue
        model = fit_pairwise_model(features.loc[fold.train_pages], judgments.iloc[fold.train], regularisation)
        found.append(model.scores(features.loc[np.concatenate([fold.train_pages, fold.test_pages])]))
    return found


def evaluate(
    features: pd.DataFrame,
    judgments: pd.DataFrame,
    reference: pd.Series,
    folds: list[Fold],
    settings: ProtocolSettings | None = None,
) -> Evaluation:
    """Fit and test the pairwise model on every fold. `features` is pages × features; `reference` the final Elo."""
    settings = settings or ProtocolSettings()
    scores = pairwise_fold_scores(features, judgments, folds, settings.regularisation)
    return evaluate_fold_scores(scores, judgments, reference, folds, settings)


def evaluate_fold_scores(
    fold_scores: FoldScores,
    judgments: pd.DataFrame,
    reference: pd.Series,
    folds: list[Fold],
    settings: ProtocolSettings | None = None,
) -> Evaluation:
    """The protocol's measures of a method from its raw scores on each fold (`fold_scores`, one per fold: the
    training and test pages' scores under the method trained on the fold's training judgments)."""
    settings = settings or ProtocolSettings()
    edges = tuple(settings.gap_edges)
    gaps = judgment_gaps(judgments, reference)
    fold_rows, tested_parts, oof, z_parts = [], [], [], []
    for number, (fold, raw) in enumerate(zip(folds, fold_scores, strict=True)):
        row = {"kind": fold.kind, "repeat": fold.repeat, "fold": fold.label, "train": len(fold.train)}
        row["test"] = len(fold.test)
        if raw is None or not len(fold.train) or not len(fold.test):
            fold_rows.append(row)
            continue
        train, test = judgments.iloc[fold.train], judgments.iloc[fold.test]
        train_scores = raw.loc[fold.train_pages]
        centre, spread = float(train_scores.mean()), float(train_scores.std(ddof=0))
        z_train = _z(train_scores, centre, spread)
        z_test = _z(raw.loc[fold.test_pages], centre, spread)
        z_parts.append(
            pd.DataFrame(
                {
                    "fold_number": number,
                    "page": np.concatenate([fold.train_pages, fold.test_pages]),
                    "test": np.repeat([False, True], [len(fold.train_pages), len(fold.test_pages)]),
                    "z": np.concatenate([z_train.to_numpy(dtype=float), z_test.to_numpy(dtype=float)]),
                }
            )
        )
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
    fold_z = pd.concat(z_parts, ignore_index=True) if z_parts else pd.DataFrame()
    return Evaluation(folds_table, by_gap, tested, scores, reliability, summary, fold_z)


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
    reference: pd.Series | None = None,
) -> dict[str, float]:
    """The first method's accuracy minus the second's, on the same folds: the mean of the per-fold differences, and
    a 95% interval of the pooled difference from the same manuscript resamples for both. With the `reference` scores,
    also the difference of their SROCC and its interval, on the same resamples."""
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
    found = {
        "difference": float(np.nanmean(per_fold)),
        "difference_pooled": float(np.mean(a[both] - b[both])),
        "low": low,
        "high": high,
    }
    if reference is not None:
        pages = first.scores.index.intersection(second.scores.index).intersection(reference.index)
        page_manuscript = manuscripts.reindex(pages).map(number).to_numpy()
        values = [evaluation.scores[pages].to_numpy() for evaluation in (first, second)]
        target = reference[pages].to_numpy()
        resampled = [metrics.bootstrap_srocc(value, target, page_manuscript, multiplicities) for value in values]
        srocc_low, srocc_high = metrics.interval(resampled[0] - resampled[1])
        found |= {
            "srocc_difference": metrics.srocc(first.scores, reference) - metrics.srocc(second.scores, reference),
            "srocc_low": srocc_low,
            "srocc_high": srocc_high,
        }
    return found


def paired_difference(
    first: Evaluation,
    second: Evaluation,
    judgments: pd.DataFrame,
    manuscripts: pd.Series,
    settings: ProtocolSettings,
    reference: pd.Series,
) -> dict[str, float]:
    """The first method's accuracy (the mean over folds) and SROCC minus the second's, on the same folds, each with
    its 95% interval from the same resamples of the manuscripts for both: accuracy's as `fold_mean_intervals` draws
    it (each fold's mean difference of hits, then their mean). The same within manuscripts: the accuracy on the test
    judgments between two pages of one manuscript, and ρ within manuscripts (`within_and_across`)."""
    merged = first.tested.merge(second.tested, on=["fold_number", "row"], suffixes=("_a", "_b"))
    if len(merged) != len(first.tested) or len(merged) != len(second.tested):
        raise ValueError("the two evaluations ran on other folds")
    values = merged.assign(value=merged["hit_a"] - merged["hit_b"], gap=merged["gap_a"])
    interval = fold_mean_intervals(values, judgments, manuscripts, settings.gap_edges, settings).iloc[-1]
    found = paired_comparison(first, second, judgments, manuscripts, settings, reference=reference)
    within = within_manuscripts(judgments, manuscripts)[values["row"].to_numpy()]
    accuracy_within = fold_mean(values[within], judgments, manuscripts, settings)
    multiplicities, number = manuscript_resamples(manuscripts, settings)
    rho = [centred_srocc(evaluation.scores, reference, manuscripts, multiplicities, number) for evaluation in
           (first, second)]  # fmt: skip
    rho_low, rho_high = metrics.interval(rho[0][1] - rho[1][1])
    return {
        "accuracy_difference": found["difference"],
        "accuracy_low": float(interval["low"]),
        "accuracy_high": float(interval["high"]),
        "srocc_difference": found["srocc_difference"],
        "srocc_low": found["srocc_low"],
        "srocc_high": found["srocc_high"],
        "accuracy_within_difference": accuracy_within[0],
        "accuracy_within_low": accuracy_within[1],
        "accuracy_within_high": accuracy_within[2],
        "srocc_within_difference": rho[0][0] - rho[1][0],
        "srocc_within_low": rho_low,
        "srocc_within_high": rho_high,
    }


# Within and across manuscripts.


def within_manuscripts(judgments: pd.DataFrame, manuscripts: pd.Series) -> np.ndarray:
    """Which judgments are between two pages of one manuscript."""
    left = judgments["left"].map(manuscripts).to_numpy()
    return left == judgments["right"].map(manuscripts).to_numpy()


def manuscript_resamples(manuscripts: pd.Series, settings: ProtocolSettings) -> tuple[np.ndarray, dict[str, int]]:
    """The resamples of the manuscripts `bootstrap` draws, and each manuscript's number in them."""
    names = sorted(manuscripts.unique())
    multiplicities = metrics.manuscript_multiplicities(names, settings.bootstrap, settings.seed)
    return multiplicities, {name: index for index, name in enumerate(names)}


def fold_mean(
    values: pd.DataFrame, judgments: pd.DataFrame, manuscripts: pd.Series, settings: ProtocolSettings
) -> tuple[float, float, float]:
    """The mean over folds of each fold's mean `value` (the folds without any left out), with its 95% interval as
    `fold_mean_intervals` draws it."""
    if not len(values):
        return math.nan, math.nan, math.nan
    estimate = float(values.groupby("fold_number")["value"].mean().mean())
    interval = fold_mean_intervals(values, judgments, manuscripts, settings.gap_edges, settings).iloc[-1]
    return estimate, float(interval["low"]), float(interval["high"])


def centred(scores: pd.Series, manuscripts: pd.Series) -> pd.Series:
    """Scores less their manuscript's mean, over the manuscripts with at least two pages scored."""
    groups = manuscripts.reindex(scores.index)
    counts = groups.map(groups.value_counts())
    kept = scores[(counts >= 2).to_numpy()]
    return kept - kept.groupby(groups[kept.index]).transform("mean")


def centred_srocc(
    scores: pd.Series, reference: pd.Series, manuscripts: pd.Series, multiplicities: np.ndarray, number: dict
) -> tuple[float, np.ndarray]:
    """Spearman's ρ within manuscripts: between the scores and the reference scores, both centred on each manuscript's
    mean, over the pages of the manuscripts with at least two pages that have both; and its value in each resample of
    the manuscripts (a page as often as its manuscript is drawn)."""
    pages = scores.dropna().index.intersection(reference.dropna().index)
    found, target = centred(scores[pages], manuscripts), centred(reference[pages], manuscripts)
    if len(found) < 3:
        return math.nan, np.full(len(multiplicities), math.nan)
    page_manuscript = manuscripts.reindex(found.index).map(number).to_numpy()
    values, aims = found.to_numpy(dtype=float), target[found.index].to_numpy(dtype=float)
    rho = metrics.spearman(values, aims)
    return rho, metrics.bootstrap_srocc(values, aims, page_manuscript, multiplicities)


def within_and_across(
    evaluation: Evaluation,
    judgments: pd.DataFrame,
    reference: pd.Series,
    manuscripts: pd.Series,
    settings: ProtocolSettings | None = None,
) -> dict[str, float]:
    """A method's accuracy on the test judgments between two pages of one manuscript and on those across manuscripts,
    apart (each the mean over folds, with its 95% interval), the consistency of the same judgments with the reference
    scores, and its ρ within manuscripts with its 95% interval, all on the same resamples of the manuscripts as
    `bootstrap`; with how many test judgments of each kind a fold holds."""
    settings = settings or ProtocolSettings()
    tested = evaluation.tested
    rows = tested["row"].to_numpy()
    within = within_manuscripts(judgments, manuscripts)[rows]
    agreement = metrics.hits(judgments, reference)[rows]
    folds = tested["fold_number"].nunique()
    found: dict[str, float] = {}
    for kind, chosen in (("within", within), ("across", ~within)):
        for name, value in (("accuracy", tested["hit"].to_numpy()), ("consistency", agreement)):
            part = tested[chosen].assign(value=value[chosen])
            estimate, low, high = fold_mean(part, judgments, manuscripts, settings)
            found |= {f"{name}_{kind}": estimate, f"{name}_{kind}_low": low, f"{name}_{kind}_high": high}
        found[f"judgments_{kind}_per_fold"] = float(chosen.sum()) / max(folds, 1)
    multiplicities, number = manuscript_resamples(manuscripts, settings)
    rho, resampled = centred_srocc(evaluation.scores, reference, manuscripts, multiplicities, number)
    low, high = metrics.interval(resampled)
    return found | {"srocc_within": rho, "srocc_within_low": low, "srocc_within_high": high}


# Folds as a method is trained on them, with the number of the learning curve's share (0 for the evaluation itself),
# to their raw scores.
Scorer = Callable[[list[Fold], int], FoldScores]


def reduced_folds(
    folds: list[Fold], judgments: pd.DataFrame, share: float, by: str, seed: int, share_number: int
) -> list[Fold]:
    """The folds with a share of their training judgments, or of their training images (then the judgments between
    the images kept), drawn from `numpy.random.default_rng([seed, share_number, fold number])`; the test sides
    unchanged."""
    if by not in {"judgments", "images"}:
        raise ValueError(f"unknown learning curve {by!r}: expected judgments or images")
    reduced = []
    for number, fold in enumerate(folds):
        rng = np.random.default_rng([seed, share_number, number])
        if by == "judgments":
            keep = max(1, round(share * len(fold.train)))
            train = np.sort(rng.choice(fold.train, size=keep, replace=False)) if len(fold.train) else fold.train
            train_pages = fold.train_pages
        else:
            keep = max(2, round(share * len(fold.train_pages)))
            train_pages = rng.choice(fold.train_pages, size=min(keep, len(fold.train_pages)), replace=False)
            within = judgments_within(judgments, train_pages)
            train = np.intersect1d(within, fold.train)
        kept = Fold(fold.kind, fold.repeat, fold.index, fold.label, train_pages, fold.test_pages, train, fold.test)
        reduced.append(kept)
    return reduced


def learning_curve_runs(
    method: pd.DataFrame | Scorer,
    judgments: pd.DataFrame,
    reference: pd.Series,
    folds: list[Fold],
    shares: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0),
    by: str = "judgments",
    settings: ProtocolSettings | None = None,
) -> list[tuple[float, list[Fold], Evaluation]]:
    """The method evaluated on the folds reduced to each share: a feature table for the pairwise model, or a scorer
    that trains the method on folds."""
    settings = settings or ProtocolSettings()
    if isinstance(method, pd.DataFrame):
        features = method

        def method(chosen: list[Fold], share_number: int) -> FoldScores:
            return pairwise_fold_scores(features, judgments, chosen, settings.regularisation)

    runs = []
    for share_number, share in enumerate(shares):
        reduced = reduced_folds(folds, judgments, share, by, settings.seed, share_number)
        evaluation = evaluate_fold_scores(method(reduced, share_number + 1), judgments, reference, reduced, settings)
        runs.append((share, reduced, evaluation))
    return runs


def learning_curve(
    method: pd.DataFrame | Scorer,
    judgments: pd.DataFrame,
    reference: pd.Series,
    folds: list[Fold],
    shares: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0),
    by: str = "judgments",
    settings: ProtocolSettings | None = None,
) -> pd.DataFrame:
    """Accuracy on the same test judgments when training on a share of the training judgments or of the training
    images (then on the judgments between the images kept). One row per share: the mean over folds."""
    rows = []
    for share, reduced, result in learning_curve_runs(method, judgments, reference, folds, shares, by, settings):
        rows.append(
            {
                "by": by,
                "share": share,
                "train_per_fold": float(np.mean([len(fold.train) for fold in reduced])),
                "accuracy": result.summary["accuracy"],
                "accuracy_sd": result.summary["accuracy_sd"],
                "srocc": result.summary["srocc"],
            }
        )
    return pd.DataFrame(rows)


# Ensembles.


def fitted_weights(z: pd.DataFrame, judgments: pd.DataFrame, regularisation: float = 1.0) -> np.ndarray:
    """Weights of the members' z-scores (pages × members) that best predict the judgments: an intercept-free
    logistic regression on the differences between the two pages' z-scores, each judgment entered in both orders;
    scaled so that their absolute values sum to one."""
    chosen = z.loc[judgments["chosen"].to_numpy()].to_numpy(dtype=float)
    other = z.loc[other_image(judgments).to_numpy()].to_numpy(dtype=float)
    differences = chosen - other
    design = np.vstack([differences, -differences])
    outcome = np.concatenate([np.ones(len(differences)), np.zeros(len(differences))])
    regression = LogisticRegression(fit_intercept=False, C=regularisation, max_iter=1000)
    weights = regression.fit(design, outcome).coef_[0].astype(float)
    total = np.abs(weights).sum()
    return weights / total if total > 0 else np.full(len(weights), 1 / len(weights))


def ensemble_fold_scores(
    members: dict[str, pd.DataFrame],
    folds: list[Fold],
    judgments: pd.DataFrame,
    weights: dict[str, float] | None = None,
    regularisation: float = 1.0,
) -> tuple[FoldScores, pd.DataFrame]:
    """Each fold's weighted sum of its members' z-scores of the fold's pages. `members` holds each member's fold
    z-scores (`fold_number`, `page`, `test`, `z`) from an evaluation on the same folds. The weights are as given, or,
    when None, fitted on each fold's training judgments alone (`fitted_weights`). Returns the fold scores and each
    fold's weights."""
    names = list(members)
    by_fold = {name: dict(tuple(table.groupby("fold_number"))) for name, table in members.items()}
    found: FoldScores = []
    rows = []
    for number, fold in enumerate(folds):
        if not len(fold.train) or not len(fold.test) or any(number not in by_fold[name] for name in names):
            found.append(None)
            rows.append({"fold_number": number})
            continue
        columns = {}
        for name in names:
            part = by_fold[name][number]
            if set(part.loc[part["test"].astype(bool), "page"]) != set(fold.test_pages):
                raise ValueError(f"the member {name} was evaluated on other folds than the ensemble's")
            columns[name] = part.set_index("page")["z"]
        pages = np.concatenate([fold.train_pages, fold.test_pages])
        z = pd.DataFrame(columns).reindex(pages)
        if z.isna().any().any():
            raise ValueError(f"fold {number}: a member has no score for some of its pages")
        if weights is None:
            chosen = fitted_weights(z.loc[fold.train_pages], judgments.iloc[fold.train], regularisation)
        else:
            chosen = np.array([float(weights[name]) for name in names])
        found.append(pd.Series(z.to_numpy(dtype=float) @ chosen, index=pages))
        rows.append({"fold_number": number, **dict(zip(names, chosen, strict=True))})
    return found, pd.DataFrame(rows)
