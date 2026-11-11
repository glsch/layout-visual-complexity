"""Pairwise judgments to scores: the published dataset, Elo over permutations, Bradley–Terry, the comparison graph
and the judgments' reliability.

Self-contained: imports nothing from the rest of the package and needs only numpy and pandas.
"""

from layout_visual_complexity.ranking.dataset import Dataset, DatasetError, read_dataset, validate_dataset
from layout_visual_complexity.ranking.elo import EloSettings, permuted_scores, play
from layout_visual_complexity.ranking.graph import GraphSummary, describe
from layout_visual_complexity.ranking.reliability import position_bias, reliability_by_gap
from layout_visual_complexity.ranking.scores import Scores, compute_scores, read_scores, write_scores
from layout_visual_complexity.ranking.split_half import SplitHalfSettings, split_half_correlations

__all__ = [
    "Dataset",
    "DatasetError",
    "EloSettings",
    "GraphSummary",
    "Scores",
    "SplitHalfSettings",
    "compute_scores",
    "describe",
    "permuted_scores",
    "play",
    "position_bias",
    "read_dataset",
    "read_scores",
    "reliability_by_gap",
    "split_half_correlations",
    "validate_dataset",
    "write_scores",
]
