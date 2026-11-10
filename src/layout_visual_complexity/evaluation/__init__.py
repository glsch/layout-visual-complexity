"""The evaluation protocol: folds, the pairwise logistic model, metrics and their intervals."""

from layout_visual_complexity.evaluation.folds import Fold, annotator_folds, category_folds, manuscript_folds
from layout_visual_complexity.evaluation.model import PairwiseModel, fit_pairwise_model
from layout_visual_complexity.evaluation.protocol import (
    Evaluation,
    ProtocolSettings,
    bootstrap,
    consistency_reference,
    evaluate,
    learning_curve,
    paired_comparison,
)

__all__ = [
    "Evaluation",
    "Fold",
    "PairwiseModel",
    "ProtocolSettings",
    "annotator_folds",
    "bootstrap",
    "category_folds",
    "consistency_reference",
    "evaluate",
    "fit_pairwise_model",
    "learning_curve",
    "manuscript_folds",
    "paired_comparison",
]
