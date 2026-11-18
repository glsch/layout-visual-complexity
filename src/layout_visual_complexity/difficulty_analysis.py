"""The difficulty experiment's statistics: how the measures of automatic difficulty go with predicted complexity.

- Rank correlations: Spearman's ρ of each measure with complexity over all pages, and within manuscripts (both centred
  on each manuscript's mean, over the manuscripts with at least two pages), with 95% intervals from resampling the
  manuscripts.
- Regressions of each measure on complexity and the controls, two ways:
  - a mixed model with a random intercept for the manuscript, fitted by REML (`statsmodels`' `MixedLM`), with Wald
    intervals and the marginal R² (Nakagawa and Schielzeth's: the variance of the fixed effects' predictions over
    itself plus the manuscripts' variance and the residual variance);
  - least squares across manuscripts (every page pooled, with an intercept) and within them (each manuscript's means
    removed from the measure and the covariates, so only what varies within a manuscript is left), their intervals
    from the same resamples of the manuscripts, a page counted as often as its manuscript is drawn.

Complexity is the expected Elo score standardised over the pages analysed, so its coefficient is per standard
deviation. The controls: the log of the ground truth's lines, the century as a number, Latin or not, and the log of
the median line height in pixels (the resolution); and, in the held-out comparison, held out and its interaction with
complexity.

A page whose ground-truth texts don't fit their lines (`lvc difficulty`'s `texts_misplaced`) is left out of every
character error rate, and keeps its segmentation measures.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from layout_visual_complexity.difficulty import texts_misplaced
from layout_visual_complexity.evaluation import metrics
from layout_visual_complexity.evaluation.protocol import centred_srocc

# The measures, with their names in tables and figures.
MEASURES = {
    "zone_f1": "Zone F1",
    "line_f1": "Line F1",
    "baseline_distance": "Baseline distance",
    "cer_lines": "CER, ground-truth lines",
    "cer_end_to_end": "CER, end to end",
}
SEGMENTATION = ["zone_f1", "line_f1", "baseline_distance"]
TERMS = ["complexity", "log_lines", "century", "latin", "log_resolution"]
HELD_OUT_TERMS = [*TERMS, "held_out", "complexity:held_out"]
# Added to the CER before its log, so a page read without error stays finite.
LOG_OFFSET = 0.001
# The columns a page's character errors are in, with and without spaces, overall, in main text and in margins.
CHARACTER_COLUMNS = ("cer_", "errors_", "characters")


def standardised(values: pd.Series) -> pd.Series:
    """Values less their mean, over their standard deviation."""
    values = values.astype(float)
    return (values - values.mean()) / values.std(ddof=1)


def character_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if column.startswith(CHARACTER_COLUMNS)]


def prepared(rows: pd.DataFrame) -> pd.DataFrame:
    """Each page's measures with the regressions' covariates: complexity (the expected Elo score standardised over these
    pages), log lines, century, Latin, log resolution, held out, and the manuscript. The character errors of the pages
    whose ground-truth texts don't fit their lines are left out (missing)."""
    data = rows.copy()
    columns = character_columns(data)
    data[columns] = data[columns].astype(float)
    data.loc[texts_misplaced(data), columns] = np.nan
    data["complexity"] = standardised(data["expected_elo"])
    data["log_lines"] = np.log(data["lines"].astype(float).where(data["lines"] > 0))
    data["century"] = data["century"].astype(float)
    data["latin"] = (data["language"].astype(str) == "Latin").astype(float)
    data["log_resolution"] = np.log(data["line_height"].astype(float).where(data["line_height"] > 0))
    data["held_out"] = (data["split"].astype(str) == "test").astype(float)
    data["complexity:held_out"] = data["complexity"] * data["held_out"]
    data["manuscript"] = data["document"].astype(str)
    for measure in ("cer_lines", "cer_end_to_end"):
        if measure in data:
            data[f"log_{measure}"] = np.log(data[measure].astype(float) + LOG_OFFSET)
    return data


@dataclass
class Fit:
    """A regression's coefficients by term (estimate, low, high), the pages and manuscripts it used, and its R²
    (the marginal R² for the mixed model)."""

    coefficients: dict[str, tuple[float, float, float]]
    pages: int
    manuscripts: int
    r2: float
    notes: list[str] = field(default_factory=list)


def complete(data: pd.DataFrame, measure: str, terms: list[str]) -> pd.DataFrame:
    return data.dropna(subset=[measure, *terms, "manuscript"])


def varying(data: pd.DataFrame, terms: list[str], within: bool = False) -> list[str]:
    """The terms that vary over these pages (within manuscripts, when asked)."""
    kept = []
    for term in terms:
        values = data[term] - data.groupby("manuscript")[term].transform("mean") if within else data[term]
        if np.nanstd(values.to_numpy(dtype=float)) > 1e-12:
            kept.append(term)
    return kept


def mixed_model(data: pd.DataFrame, measure: str, terms: list[str] = TERMS) -> Fit:
    """The measure on the terms with a random intercept for the manuscript, by REML, with Wald intervals. Where the
    default optimisers don't converge (a few extreme pages can stop them), it is fitted again by Powell's method."""
    import warnings

    import statsmodels.api as sm

    used = complete(data, measure, terms)
    kept = varying(used, terms)
    exog = sm.add_constant(used[kept].to_numpy(dtype=float), has_constant="add")
    model = sm.MixedLM(used[measure].to_numpy(dtype=float), exog, groups=used["manuscript"].to_numpy())
    notes = []
    for options in ({}, {"method": "powell", "maxiter": 5000}):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.fit(reml=True, **options)
        if result.converged:
            break
    notes += sorted({str(item.message).split("\n")[0] for item in caught})
    if options:
        notes.append("fitted by Powell's method, the default optimisers having failed to converge")
    if not result.converged:
        notes.append("did not converge")
    fixed = np.asarray(result.fe_params, dtype=float)
    interval = np.asarray(result.conf_int(), dtype=float)[: len(fixed)]
    coefficients = {term: (fixed[i + 1], interval[i + 1, 0], interval[i + 1, 1]) for i, term in enumerate(kept)}
    predicted = exog @ fixed
    fixed_variance = float(np.var(predicted))
    group_variance = float(np.asarray(result.cov_re)[0, 0])
    total = fixed_variance + group_variance + float(result.scale)
    return Fit(
        coefficients, len(used), used["manuscript"].nunique(), fixed_variance / total if total else math.nan, notes
    )


def _weighted(design: np.ndarray, outcome: np.ndarray, weights: np.ndarray) -> np.ndarray:
    root = np.sqrt(weights)[:, None]
    solution, *_ = np.linalg.lstsq(design * root, outcome * root[:, 0], rcond=None)
    return solution


def least_squares(
    data: pd.DataFrame,
    measure: str,
    terms: list[str] = TERMS,
    within: bool = False,
    resamples: int = 1000,
    seed: int = 1,
) -> Fit:
    """Least squares across manuscripts (pooled, with an intercept) or within them (each manuscript's means removed),
    with 95% intervals from resampling the manuscripts."""
    used = complete(data, measure, terms)
    kept = varying(used, terms, within)
    outcome = used[measure].to_numpy(dtype=float)
    design = used[kept].to_numpy(dtype=float)
    groups = used["manuscript"]
    if within:
        outcome = outcome - used.groupby("manuscript")[measure].transform("mean").to_numpy(dtype=float)
        design = design - used.groupby("manuscript")[kept].transform("mean").to_numpy(dtype=float)
    else:
        design = np.column_stack([np.ones(len(used)), design])
    offset = 0 if within else 1
    estimate = _weighted(design, outcome, np.ones(len(used)))
    names = sorted(groups.unique())
    number = {name: index for index, name in enumerate(names)}
    page_manuscript = groups.map(number).to_numpy()
    multiplicities = metrics.manuscript_multiplicities(names, resamples, seed)
    drawn = []
    for row in multiplicities:
        weights = row[page_manuscript].astype(float)
        if np.count_nonzero(weights) <= design.shape[1] or np.linalg.matrix_rank(design[weights > 0]) < design.shape[1]:
            continue
        drawn.append(_weighted(design, outcome, weights))
    resampled = np.array(drawn) if drawn else np.full((0, design.shape[1]), np.nan)
    coefficients = {}
    for i, term in enumerate(kept):
        low, high = metrics.interval(resampled[:, i + offset]) if len(resampled) else (math.nan, math.nan)
        coefficients[term] = (float(estimate[i + offset]), low, high)
    residual = outcome - design @ estimate
    spread = outcome - (0.0 if within else outcome.mean())
    r2 = 1 - float(residual @ residual) / float(spread @ spread) if float(spread @ spread) else math.nan
    notes = [] if len(drawn) == resamples else [f"{resamples - len(drawn)} resamples couldn't be fitted"]
    return Fit(coefficients, len(used), len(names), r2, notes)


@dataclass
class Correlation:
    """Spearman's ρ with its 95% interval, over all pages and within manuscripts."""

    overall: tuple[float, float, float]
    within: tuple[float, float, float]
    pages: int
    manuscripts: int


def correlation(data: pd.DataFrame, measure: str, resamples: int = 1000, seed: int = 1) -> Correlation:
    """Spearman's ρ of a measure with complexity, overall and within manuscripts, on the same resamples of the
    manuscripts."""
    used = data.dropna(subset=[measure, "complexity"])
    names = sorted(used["manuscript"].unique())
    number = {name: index for index, name in enumerate(names)}
    multiplicities = metrics.manuscript_multiplicities(names, resamples, seed)
    x, y = used["complexity"].to_numpy(dtype=float), used[measure].to_numpy(dtype=float)
    rho = metrics.spearman(x, y) if len(used) >= 3 else math.nan
    resampled = metrics.bootstrap_srocc(x, y, used["manuscript"].map(number).to_numpy(), multiplicities)
    overall = (rho, *metrics.interval(resampled))
    pages = used.set_index("page")
    within_rho, within_resampled = centred_srocc(
        pages["complexity"], pages[measure], pages["manuscript"], multiplicities, number
    )
    return Correlation(overall, (within_rho, *metrics.interval(within_resampled)), len(used), len(names))


def skewness(values: pd.Series) -> float:
    """The sample skewness, missing values left out."""
    from scipy.stats import skew

    found = values.dropna().to_numpy(dtype=float)
    return float(skew(found)) if len(found) >= 3 else math.nan


def similar_pages(data: pd.DataFrame, measures: list[str], within: float = 0.25) -> pd.DataFrame:
    """The held-out pages beside the seen pages of similar predicted complexity (within `within` SD): for each
    measure, the held-out pages' median, the median over them of their similar seen pages' median, and the median of
    each held-out page's difference from its similar seen pages' median, with how many held-out pages lie above."""
    held = data[data["held_out"] == 1]
    seen = data[data["held_out"] == 0]
    rows = []
    for measure in measures:
        differences, neighbours, values = [], [], []
        for _, page in held.dropna(subset=[measure, "complexity"]).iterrows():
            near = seen[(seen["complexity"] - page["complexity"]).abs() <= within][measure].dropna()
            if not len(near):
                continue
            values.append(page[measure])
            neighbours.append(float(near.median()))
            differences.append(page[measure] - float(near.median()))
        rows.append(
            {
                "measure": measure,
                "held_out_pages": len(values),
                "held_out_median": float(np.median(values)) if values else math.nan,
                "similar_seen_median": float(np.median(neighbours)) if neighbours else math.nan,
                "median_difference": float(np.median(differences)) if differences else math.nan,
                "pages_above": int(sum(value > 0 for value in differences)),
            }
        )
    return pd.DataFrame(rows)


def with_ranker(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """The pages with their complexity taken from one ranker's score, standardised over these pages."""
    found = data.assign(complexity=standardised(data[column]))
    found["complexity:held_out"] = found["complexity"] * found["held_out"]
    return found


# The measures by quarter of complexity, each a median over the quarter's pages.
QUARTER_MEASURES = [
    "line_f1",
    "line_recall",
    "line_precision",
    "cer_end_to_end",
    "cer_lines",
    "zone_f1",
    "baseline_distance",
    "lines",
]


def by_quarter(data: pd.DataFrame, quarters: int = 4) -> pd.DataFrame:
    """The pages in `quarters` groups of equal size by their expected Elo score, the least complex first: each group's
    pages, the range of its scores, the median of each measure over its pages (line recall, matched over matched and
    missed lines, and precision, matched over matched and invented, page by page), and how many pages have a CER."""
    pages = data.assign(
        line_recall=data["lines_matched"] / (data["lines_matched"] + data["lines_missed"]),
        line_precision=data["lines_matched"] / (data["lines_matched"] + data["lines_invented"]),
    )
    group = pd.qcut(pages["expected_elo"].astype(float), quarters, labels=False)
    rows = []
    for number, part in pages.groupby(group, sort=True):
        elo = part["expected_elo"].astype(float)
        row = {"quarter": int(number) + 1, "pages": len(part), "elo_low": elo.min(), "elo_high": elo.max()}
        row |= {measure: float(part[measure].astype(float).median()) for measure in QUARTER_MEASURES}
        row["cer_pages"] = int(part["cer_end_to_end"].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def pooled(data: pd.DataFrame, errors: str, characters: str, resamples: int = 1000, seed: int = 1) -> tuple:
    """A pooled rate (total errors over total characters, or matched over all for F1) with its 95% interval from
    resampling the manuscripts."""
    used = data.dropna(subset=[errors, characters])
    used = used[used[characters] > 0]
    if not len(used):
        return math.nan, math.nan, math.nan
    names = sorted(used["manuscript"].unique())
    number = used["manuscript"].map({name: index for index, name in enumerate(names)}).to_numpy()
    top, bottom = used[errors].to_numpy(dtype=float), used[characters].to_numpy(dtype=float)
    multiplicities = metrics.manuscript_multiplicities(names, resamples, seed)
    weights = multiplicities[:, number]
    with np.errstate(invalid="ignore", divide="ignore"):
        resampled = (weights * top).sum(axis=1) / (weights * bottom).sum(axis=1)
    return float(top.sum() / bottom.sum()), *metrics.interval(resampled)
