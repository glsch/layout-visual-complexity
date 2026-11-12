"""The pairwise logistic model: a page score that is linear in the page's features, fitted on judgments.

Features are standardised on the training pages (missing values filled with the training median). An
intercept-free, L2-regularised logistic regression is fitted on the differences between the two pages' features,
each judgment entered in both orders so the model stays symmetric. A page's score is the linear predictor.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from layout_visual_complexity.ranking.reliability import other_image


@dataclass
class PairwiseModel:
    """A fitted model: how to standardise each feature, and its weight."""

    columns: list[str]
    centre: np.ndarray
    spread: np.ndarray
    fill: np.ndarray
    weights: np.ndarray

    def standardise(self, features: pd.DataFrame) -> np.ndarray:
        values = features.reindex(columns=self.columns).to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, self.fill)
        return (values - self.centre) / self.spread

    def scores(self, features: pd.DataFrame) -> pd.Series:
        """Each page's score: its standardised features times the weights."""
        return pd.Series(self.standardise(features) @ self.weights, index=features.index, name="score")


def difference_matrix(standardised: pd.DataFrame, judgments: pd.DataFrame) -> np.ndarray:
    """Chosen page's features minus the other page's, one row per judgment."""
    chosen = standardised.loc[judgments["chosen"].to_numpy()].to_numpy(dtype=float)
    other = standardised.loc[other_image(judgments).to_numpy()].to_numpy(dtype=float)
    return chosen - other


def fit_pairwise_model(features: pd.DataFrame, judgments: pd.DataFrame, regularisation: float = 1.0) -> PairwiseModel:
    """Fit on `judgments` between the pages of `features` (pages × features, numeric).

    `regularisation` is scikit-learn's C: smaller is stronger.
    """
    if judgments.empty:
        raise ValueError("no training judgments")
    columns = list(features.columns)
    values = features.to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fill = np.nanmedian(values, axis=0) if len(values) else np.zeros(len(columns))
    fill = np.where(np.isfinite(fill), fill, 0.0)
    values = np.where(np.isnan(values), fill, values)
    centre = values.mean(axis=0)
    spread = values.std(axis=0)
    spread = np.where(spread > 0, spread, 1.0)
    standardised = pd.DataFrame((values - centre) / spread, index=features.index, columns=columns)
    differences = difference_matrix(standardised, judgments)
    design = np.vstack([differences, -differences])
    outcome = np.concatenate([np.ones(len(differences)), np.zeros(len(differences))])
    regression = LogisticRegression(fit_intercept=False, C=regularisation, max_iter=1000)
    regression.fit(design, outcome)
    return PairwiseModel(columns, centre, spread, fill, regression.coef_[0].astype(float))


class FeatureMismatch(ValueError):
    """Features computed otherwise than a trained model's: another family's settings, or columns missing."""


@dataclass
class TrainedModel:
    """A pairwise model trained on every judgment, with what its features must be: each family as the feature table
    recorded it (name, class, version, settings and their hash). A page's score is z-scored with the training pages'
    mean and spread; `scale` turns a difference of scores into the log-odds that the first page is chosen."""

    model: PairwiseModel
    families: list[dict]
    score_mean: float
    score_sd: float
    scale: float
    regularisation: float
    trained_on: dict

    def check(self, hashes: dict[str, str], columns: list[str]) -> None:
        """Refuse features whose families' settings differ from the model's, or that lack its columns."""
        for family in self.families:
            found = hashes.get(family["name"])
            if found is None:
                raise FeatureMismatch(f"the features have no family {family['name']}, which the model was trained on")
            if found != family["hash"]:
                raise FeatureMismatch(
                    f"the features' family {family['name']} was computed with other settings than the model's "
                    f"({found}, not {family['hash']})"
                )
        missing = [column for column in self.model.columns if column not in set(columns)]
        if missing:
            raise FeatureMismatch(f"the features lack {len(missing)} of the model's columns: {', '.join(missing[:3])}")

    def scores(self, features: pd.DataFrame, hashes: dict[str, str]) -> pd.Series:
        """Each page's z-scored score, after checking the features."""
        self.check(hashes, list(features.columns))
        raw = self.model.scores(features)
        return (raw - self.score_mean) / (self.score_sd if self.score_sd > 0 else 1.0)

    def to_dict(self) -> dict:
        return {
            "model": "pairwise logistic",
            "columns": self.model.columns,
            "families": self.families,
            "centre": self.model.centre.tolist(),
            "spread": self.model.spread.tolist(),
            "fill": self.model.fill.tolist(),
            "weights": self.model.weights.tolist(),
            "score_mean": self.score_mean,
            "score_sd": self.score_sd,
            "scale": self.scale,
            "regularisation": self.regularisation,
            "trained_on": self.trained_on,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainedModel":
        model = PairwiseModel(
            list(data["columns"]),
            *(np.asarray(data[key], dtype=float) for key in ("centre", "spread", "fill", "weights")),
        )
        return cls(
            model,
            list(data["families"]),
            float(data["score_mean"]),
            float(data["score_sd"]),
            float(data["scale"]),
            float(data["regularisation"]),
            dict(data.get("trained_on", {})),
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "TrainedModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
