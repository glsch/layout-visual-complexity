"""Pairwise judgments to scores: the dataset, Elo over permutations, Bradley–Terry, the comparison graph and the
judgments' reliability.

Self-contained: imports nothing from the rest of the package and needs only numpy and pandas.
"""

from layout_visual_complexity.ranking.annotators import fold_spelling, pseudonyms, read_merge_table
from layout_visual_complexity.ranking.dataset import Dataset, Rules, build_judgments, read_dataset
from layout_visual_complexity.ranking.elo import EloSettings, permuted_scores, play
from layout_visual_complexity.ranking.export import read_export
from layout_visual_complexity.ranking.graph import GraphSummary, describe
from layout_visual_complexity.ranking.reliability import position_bias, reliability_by_gap
from layout_visual_complexity.ranking.scores import Scores, compute_scores, read_scores, write_scores
from layout_visual_complexity.ranking.split_half import SplitHalfSettings, split_half_correlations

__all__ = [
    "Dataset",
    "EloSettings",
    "GraphSummary",
    "Rules",
    "Scores",
    "SplitHalfSettings",
    "build_judgments",
    "compute_scores",
    "describe",
    "fold_spelling",
    "permuted_scores",
    "play",
    "position_bias",
    "pseudonyms",
    "read_dataset",
    "read_export",
    "read_merge_table",
    "read_scores",
    "reliability_by_gap",
    "split_half_correlations",
    "write_scores",
]
