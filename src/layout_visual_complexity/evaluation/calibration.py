"""Win probabilities that hold out of sample, and the Elo scale of predicted scores.

A method's scale turns a difference of z-scored page scores into the log-odds that the first page is judged more
complex. Fitted on the judgments a model was trained on, it is overconfident on new pages: the model fits its own
judgments more closely than others. Fitted on the evaluation's out-of-fold scores, each page scored by models that
never saw it, it gives probabilities that hold on new pages.

The expected Elo score of a page is the regression of the Elo scores on the out-of-fold scores, so a predicted score
reads on the Elo scale with the spread a prediction has, which is narrower than the Elo scores'.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from layout_visual_complexity.evaluation import metrics
from layout_visual_complexity.ranking.reliability import other_image


def score_differences(scores: pd.Series, judgments: pd.DataFrame) -> np.ndarray:
    """The chosen page's score minus the other's, one per judgment between pages that have scores."""
    known = judgments["chosen"].isin(scores.index) & other_image(judgments).isin(scores.index)
    kept = judgments[known]
    return scores[kept["chosen"]].to_numpy(dtype=float) - scores[other_image(kept)].to_numpy(dtype=float)


def out_of_fold_scale(scores: pd.Series, judgments: pd.DataFrame) -> float:
    """The scale fitted by maximum likelihood on the out-of-fold scores' differences over every judgment."""
    return metrics.fit_scale(score_differences(scores.dropna(), judgments))


def read_out_of_fold_scores(folder: Path) -> pd.DataFrame:
    """An evaluation's `scores.tsv`: each page's out-of-fold score (`score`), its Elo score and its manuscript."""
    table = pd.read_csv(Path(folder) / "scores.tsv", sep="\t", dtype={"page": str, "manuscript": str})
    return table.set_index("page")


def test_differences(folder: Path, judgments: pd.DataFrame) -> pd.DataFrame:
    """Every tested judgment of an evaluation: its fold, the chosen page's z-score minus the other's under the
    fold's model, and the probability the evaluation gave it with the scale fitted on the fold's training
    judgments."""
    folder = Path(folder)
    hits = pd.read_csv(folder / "hits.tsv", sep="\t", dtype={"judgment": str})
    z = pd.read_csv(folder / "fold_scores.tsv", sep="\t", dtype={"page": str})
    z = z[z["test"].astype(bool)].set_index(["fold_number", "page"])["z"]
    by_id = judgments.assign(other=other_image(judgments)).set_index(judgments["judgment"].astype(str))
    chosen = by_id.loc[hits["judgment"], "chosen"].to_numpy()
    other = by_id.loc[hits["judgment"], "other"].to_numpy()
    folds = hits["fold_number"].to_numpy()
    first = z.reindex(pd.MultiIndex.from_arrays([folds, chosen])).to_numpy(dtype=float)
    second = z.reindex(pd.MultiIndex.from_arrays([folds, other])).to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "fold_number": folds,
            "chosen": chosen,
            "other": other,
            "difference": first - second,
            "training_probability": hits["chosen_probability"].to_numpy(dtype=float),
        }
    )


def calibration_error(chosen_probabilities: np.ndarray, bins: int = 10) -> float:
    """The expected calibration error: the mean gap between the predicted and the observed share, over bins of the
    predicted probability that the left page of a symmetric pair is chosen, weighted by the bins' judgments. Each
    judgment counts in both orders, so the curve is symmetric about one half."""
    p = np.clip(np.asarray(chosen_probabilities, dtype=float), 0.0, 1.0)
    predicted = np.concatenate([p, 1 - p])
    observed = np.concatenate([np.ones(len(p)), np.zeros(len(p))])
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(predicted, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for number in range(bins):
        inside = which == number
        if inside.any():
            error += inside.mean() * abs(predicted[inside].mean() - observed[inside].mean())
    return float(error)


def calibration_curve(chosen_probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Per bin of the predicted probability (each judgment in both orders): judgments, mean predicted, observed."""
    p = np.clip(np.asarray(chosen_probabilities, dtype=float), 0.0, 1.0)
    predicted = np.concatenate([p, 1 - p])
    observed = np.concatenate([np.ones(len(p)), np.zeros(len(p))])
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(predicted, edges[1:-1]), 0, bins - 1)
    rows = []
    for number in range(bins):
        inside = which == number
        rows.append(
            {
                "low": edges[number],
                "high": edges[number + 1],
                "judgments": int(inside.sum()),
                "predicted": float(predicted[inside].mean()) if inside.any() else math.nan,
                "observed": float(observed[inside].mean()) if inside.any() else math.nan,
            }
        )
    return pd.DataFrame(rows)


def held_out_scale_probabilities(scores: pd.Series, judgments: pd.DataFrame, groups: pd.Series) -> np.ndarray:
    """Each judgment's probability under a scale fitted on the out-of-fold scores of the judgments of the other
    groups (manuscripts): the out-of-fold scale checked on judgments it wasn't fitted on. A judgment belongs to
    its chosen page's group and to the other page's; it is held out of both."""
    other = other_image(judgments)
    first = groups.reindex(judgments["chosen"]).to_numpy()
    second = groups.reindex(other).to_numpy()
    differences = scores.reindex(judgments["chosen"]).to_numpy(dtype=float) - scores.reindex(other).to_numpy(
        dtype=float
    )
    probabilities = np.full(len(judgments), np.nan)
    for group in pd.unique(np.concatenate([first, second])):
        held = (first == group) | (second == group)
        todo = held & np.isnan(probabilities)
        if not todo.any():
            continue
        scale = metrics.fit_scale(differences[~held & np.isfinite(differences)])
        probabilities[todo] = metrics.sigmoid(scale * differences[todo])
    return probabilities


@dataclass(frozen=True)
class EloScale:
    """The expected Elo score of a z-scored page score: `intercept + slope × z`, the least-squares regression of the
    Elo scores on the out-of-fold scores. `correlation` is their Pearson correlation, `elo_sd` and `score_sd` their
    spreads (ddof 0), so the slope is `correlation × elo_sd / score_sd`."""

    intercept: float
    slope: float
    correlation: float
    elo_mean: float
    elo_sd: float
    score_sd: float
    pages: int

    def __call__(self, z: np.ndarray | pd.Series | float) -> np.ndarray | pd.Series | float:
        return self.intercept + self.slope * z

    def to_dict(self) -> dict[str, float]:
        return {
            "intercept": self.intercept,
            "slope": self.slope,
            "correlation": self.correlation,
            "elo_mean": self.elo_mean,
            "elo_sd": self.elo_sd,
            "score_sd": self.score_sd,
            "pages": self.pages,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EloScale":
        return cls(**{name: float(data[name]) for name in cls.__dataclass_fields__ if name != "pages"},
                   pages=int(data["pages"]))  # fmt: skip


def fit_elo_scale(scores: pd.Series, elo: pd.Series) -> EloScale:
    """The regression of the Elo scores on the out-of-fold scores, over the pages that have both."""
    both = pd.concat([scores.rename("score"), elo.rename("elo")], axis=1, join="inner").dropna()
    if len(both) < 3:
        raise ValueError("the Elo scale needs at least three pages with both an out-of-fold score and an Elo score")
    x, y = both["score"].to_numpy(dtype=float), both["elo"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    correlation = float(np.corrcoef(x, y)[0, 1])
    return EloScale(float(intercept), float(slope), correlation, float(y.mean()), float(y.std()), float(x.std()),
                    len(both))  # fmt: skip
