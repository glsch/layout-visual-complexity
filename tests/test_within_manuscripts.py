"""Accuracy within and across manuscripts, and ρ within manuscripts, for any evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from layout_visual_complexity.evaluate import EvaluateSettings, read_elo, read_evaluation, run_evaluate
from layout_visual_complexity.evaluation import metrics
from layout_visual_complexity.evaluation.folds import FoldSettings, manuscript_folds
from layout_visual_complexity.evaluation.protocol import (
    ProtocolSettings,
    centred,
    centred_srocc,
    manuscript_resamples,
    paired_difference,
    within_and_across,
    within_manuscripts,
)
from layout_visual_complexity.ranking.dataset import read_dataset
from test_evaluate_command import setup

FOLDS = FoldSettings(k=2, repeats=3, seed=5)


def test_the_test_judgments_split_within_and_across_manuscripts(tmp_path: Path, dataset_folder: Path) -> None:
    dataset = read_dataset(dataset_folder)
    judgments = dataset.judgments.reset_index(drop=True)
    manuscripts = dataset.images.set_index("image")["manuscript"]
    within = within_manuscripts(judgments, manuscripts)
    by_hand = [left.split("_")[0] == right.split("_")[0] for left, right in zip(judgments["left"], judgments["right"],
                                                                                strict=True)]  # fmt: skip
    assert within.tolist() == by_hand and 0 < within.sum() < len(judgments)
    features, scores = setup(tmp_path, dataset_folder)
    output = tmp_path / "evaluation"
    run_evaluate(EvaluateSettings(dataset_folder, scores, output, features, "signal", folds=FOLDS, bootstrap=200))
    found = pd.read_csv(output / "metrics.tsv", sep="\t").set_index("metric")["value"]
    folds = manuscript_folds(dataset.images["image"].tolist(), manuscripts, judgments, 2, 3, 5)
    tested = np.concatenate([fold.test for fold in folds])
    assert found["judgments_within_per_fold"] == pytest.approx(within[tested].sum() / len(folds))
    assert found["judgments_across_per_fold"] == pytest.approx((~within[tested]).sum() / len(folds))
    # Every judgment within a manuscript is tested once per repeat: its two pages are always in one fold.
    assert within[tested].sum() == 3 * within.sum()


def test_rho_within_manuscripts_removes_each_manuscripts_mean() -> None:
    manuscripts = pd.Series({"a1": "a", "a2": "a", "a3": "a", "b1": "b", "b2": "b", "c1": "c"})
    scores = pd.Series({"a1": 1.0, "a2": 2.0, "a3": 6.0, "b1": 10.0, "b2": 13.0, "c1": 50.0})
    elo = pd.Series({"a1": 1400.0, "a2": 1500.0, "a3": 1450.0, "b1": 1700.0, "b2": 1600.0, "c1": 1000.0})
    centred_scores = centred(scores, manuscripts)
    assert list(centred_scores.index) == ["a1", "a2", "a3", "b1", "b2"]  # c has one page
    assert centred_scores.tolist() == pytest.approx([-2, -1, 3, -1.5, 1.5])
    hand_scores = [1 - 3, 2 - 3, 6 - 3, 10 - 11.5, 13 - 11.5]
    hand_elo = [1400 - 1450, 1500 - 1450, 1450 - 1450, 1700 - 1650, 1600 - 1650]
    multiplicities, number = manuscript_resamples(manuscripts, ProtocolSettings(bootstrap=50, seed=3))
    rho, resampled = centred_srocc(scores, elo, manuscripts, multiplicities, number)
    assert rho == pytest.approx(stats.spearmanr(hand_scores, hand_elo).statistic)
    assert len(resampled) == 50
    # Without the means removed the manuscripts' differences dominate: ρ over all pages differs.
    assert metrics.srocc(scores, elo) != pytest.approx(rho)


def test_the_values_from_saved_hits_and_scores_equal_a_fresh_evaluations(tmp_path: Path, dataset_folder: Path) -> None:
    features, scores = setup(tmp_path, dataset_folder)
    for name, family in (("signal", "signal"), ("other", "other")):
        settings = EvaluateSettings(dataset_folder, scores, tmp_path / name, features, family, folds=FOLDS,
                                    bootstrap=200, seed=4)  # fmt: skip
        run_evaluate(settings)
    fresh = pd.read_csv(tmp_path / "signal" / "metrics.tsv", sep="\t").set_index("metric")
    dataset = read_dataset(dataset_folder)
    judgments = dataset.judgments.reset_index(drop=True)
    manuscripts = dataset.images.set_index("image")["manuscript"]
    elo = read_elo(scores)
    protocol = ProtocolSettings(bootstrap=200, seed=4)
    saved = {name: read_evaluation(tmp_path / name, judgments) for name in ("signal", "other")}
    found = within_and_across(saved["signal"], judgments, elo, manuscripts, protocol)
    for name in ("accuracy_within", "accuracy_across", "consistency_within", "consistency_across", "srocc_within"):
        for suffix, column in (("", "value"), ("_low", "low"), ("_high", "high")):
            assert found[f"{name}{suffix}"] == pytest.approx(fresh.at[name, column], abs=1e-5)
    assert 0 <= found["accuracy_within"] <= 1 and found["srocc_within"] > 0
    difference = paired_difference(saved["signal"], saved["other"], judgments, manuscripts, protocol, elo)
    other = within_and_across(saved["other"], judgments, elo, manuscripts, protocol)
    assert difference["accuracy_within_difference"] == pytest.approx(
        found["accuracy_within"] - other["accuracy_within"]
    )
    assert difference["srocc_within_difference"] == pytest.approx(found["srocc_within"] - other["srocc_within"])
    assert difference["srocc_within_low"] <= difference["srocc_within_difference"] <= difference["srocc_within_high"]
