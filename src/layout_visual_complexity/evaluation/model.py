"""The pairwise logistic model: a page score that is linear in the page's features, fitted on judgments.

Features are standardised on the training pages (missing values filled with the training median). An
intercept-free, L2-regularised logistic regression is fitted on the differences between the two pages' features,
each judgment entered in both orders so the model stays symmetric. A page's score is the linear predictor.
"""

import warnings
from dataclasses import dataclass

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
