import numpy as np
import pandas as pd
import pytest

from layout_visual_complexity.evaluation import (
    ProtocolSettings,
    annotator_folds,
    bootstrap,
    category_folds,
    consistency_reference,
    evaluate,
    fit_pairwise_model,
    learning_curve,
    manuscript_folds,
    metrics,
    paired_comparison,
)


def synthetic(pages: int = 60, judgments: int = 3000, seed: int = 0):
    """Pages in 15 manuscripts and 3 categories, two informative features and a noise one, judgments drawn from a
    Bradley–Terry model on the features."""
    rng = np.random.default_rng(seed)
    ids = [f"p{n:03d}" for n in range(pages)]
    features = pd.DataFrame(rng.normal(size=(pages, 3)), index=ids, columns=["strong", "weak", "noise"])
    weights = np.array([1.5, -0.7, 0.0])
    strength = pd.Series(features.to_numpy() @ weights, index=ids)
    manuscripts = pd.Series([f"m{n % 15:02d}" for n in range(pages)], index=ids)
    categories = pd.Series([["A", "B", "C"][n % 3] for n in range(pages)], index=ids)
    first = rng.integers(0, pages, judgments)
    second = (first + rng.integers(1, pages, judgments)) % pages
    p = 1 / (1 + np.exp(strength.iloc[second].to_numpy() - strength.iloc[first].to_numpy()))
    chosen = np.where(rng.random(judgments) < p, first, second)
    table = pd.DataFrame(
        {
            "left": [ids[i] for i in first],
            "right": [ids[i] for i in second],
            "chosen": [ids[i] for i in chosen],
            "annotator": [f"annotator_{n % 12:03d}" for n in range(judgments)],
        }
    )
    return ids, features, weights, strength, manuscripts, categories, table


def test_manuscript_folds_keep_manuscripts_apart_and_test_every_page_once_per_repeat() -> None:
    ids, _, _, _, manuscripts, _, judgments = synthetic()
    folds = manuscript_folds(ids, manuscripts, judgments, k=5, repeats=3, seed=1)
    assert len(folds) == 15
    for repeat in range(3):
        tested = np.concatenate([f.test_pages for f in folds if f.repeat == repeat])
        assert sorted(tested) == ids
    for fold in folds:
        assert not set(manuscripts[fold.train_pages]) & set(manuscripts[fold.test_pages])
        train, test = judgments.iloc[fold.train], judgments.iloc[fold.test]
        assert train["left"].isin(fold.train_pages).all() and train["right"].isin(fold.train_pages).all()
        assert test["left"].isin(fold.test_pages).all() and test["right"].isin(fold.test_pages).all()
        across = len(judgments) - len(fold.train) - len(fold.test)
        assert across == (judgments["left"].isin(fold.test_pages) != judgments["right"].isin(fold.test_pages)).sum()
    # Balanced: 15 manuscripts of 4 pages into 5 folds.
    assert {len(f.test_pages) for f in folds} == {12}
    assert [f.test_pages.tolist() for f in folds[:5]] != [f.test_pages.tolist() for f in folds[5:10]]


def test_annotator_and_category_folds() -> None:
    ids, _, _, _, _, categories, judgments = synthetic()
    folds = annotator_folds(ids, judgments, k=4, repeats=1, seed=1)
    assert sorted(np.concatenate([f.test for f in folds])) == list(range(len(judgments)))
    for fold in folds:
        assert not set(judgments["annotator"].iloc[fold.train]) & set(judgments["annotator"].iloc[fold.test])
    by_category = category_folds(ids, categories, judgments)
    assert [f.label for f in by_category] == ["A", "B", "C"]
    assert set(categories[by_category[0].test_pages]) == {"A"}


def test_the_pairwise_model_recovers_feature_weights() -> None:
    _, features, weights, _, _, _, judgments = synthetic(judgments=20_000)
    model = fit_pairwise_model(features, judgments, regularisation=100.0)
    recovered = model.weights / model.spread
    assert recovered == pytest.approx(weights, abs=0.12)


def test_the_pairwise_model_is_symmetric_and_fills_missing_values() -> None:
    _, features, _, _, _, _, judgments = synthetic(judgments=500)
    features.iloc[0, 0] = np.nan
    model = fit_pairwise_model(features, judgments)
    assert np.isfinite(model.scores(features)).all()
    swapped = judgments.assign(left=judgments["right"], right=judgments["left"])
    assert np.allclose(fit_pairwise_model(features, swapped).weights, model.weights)


def test_accuracy_counts_ties_half() -> None:
    judgments = pd.DataFrame({"left": ["a", "a", "b"], "right": ["b", "c", "c"], "chosen": ["a", "c", "b"]})
    scores = pd.Series({"a": 2.0, "b": 1.0, "c": 1.0})
    assert metrics.hits(judgments, scores).tolist() == [1.0, 0.0, 0.5]


def test_the_scale_of_win_probabilities() -> None:
    rng = np.random.default_rng(0)
    differences = rng.normal(size=20_000)
    chosen_first = rng.random(20_000) < 1 / (1 + np.exp(-2.0 * differences))
    signed = np.where(chosen_first, differences, -differences)
    assert metrics.fit_scale(signed) == pytest.approx(2.0, abs=0.1)
    assert metrics.log_loss(np.array([0.5, 0.5])) == pytest.approx(np.log(2))


def test_the_protocol_learns_the_signal_and_stays_below_consistency() -> None:
    ids, features, _, strength, manuscripts, _, judgments = synthetic()
    folds = manuscript_folds(ids, manuscripts, judgments, k=5, repeats=2, seed=1)
    settings = ProtocolSettings(bootstrap=200, seed=2)
    learned = evaluate(features, judgments, strength, folds, settings)
    chance = evaluate(features[["noise"]], judgments, strength, folds, settings)
    assert learned.summary["accuracy"] > 0.7 and learned.summary["srocc"] > 0.9
    assert abs(chance.summary["accuracy"] - 0.5) < 0.06
    assert len(learned.folds) == 10 and set(learned.scores.index) == set(ids)
    assert learned.summary["log_loss"] < chance.summary["log_loss"]
    curve = learned.reliability
    assert curve["judgments"].sum() == len(learned.tested)
    filled = curve.dropna()
    assert np.abs(filled["predicted"] - filled["observed"]).max() < 0.15

    per_fold, by_gap = consistency_reference(judgments, strength, folds)
    assert len(per_fold) == 10 and by_gap["gap"].iloc[-1] == "overall"
    assert by_gap["pooled"].iloc[-1] >= learned.summary["accuracy_pooled"] - 0.02

    intervals = bootstrap(learned, judgments, strength, manuscripts, settings)
    assert intervals["accuracy_low"] < learned.summary["accuracy_pooled"] < intervals["accuracy_high"]
    assert intervals["srocc_low"] < learned.summary["srocc"] <= intervals["srocc_high"] + 1e-9

    comparison = paired_comparison(learned, chance, judgments, manuscripts, settings)
    assert comparison["difference"] > 0.15 and comparison["low"] > 0
    same = paired_comparison(learned, learned, judgments, manuscripts, settings)
    assert same["difference"] == 0 and same["low"] == same["high"] == 0


def test_learning_curves_rise_with_more_training() -> None:
    ids, features, _, strength, manuscripts, _, judgments = synthetic()
    folds = manuscript_folds(ids, manuscripts, judgments, k=5, repeats=1, seed=1)
    settings = ProtocolSettings(seed=3)
    by_judgments = learning_curve(features, judgments, strength, folds, (0.02, 1.0), "judgments", settings)
    assert by_judgments["train_per_fold"].iloc[0] < by_judgments["train_per_fold"].iloc[1]
    assert by_judgments["accuracy"].iloc[0] < by_judgments["accuracy"].iloc[1]
    by_images = learning_curve(features, judgments, strength, folds, (0.1, 1.0), "images", settings)
    assert by_images["train_per_fold"].iloc[0] < by_images["train_per_fold"].iloc[1]


def test_a_judgment_within_one_manuscript_weighs_its_manuscripts_multiplicity() -> None:
    multiplicities = np.array([[2, 3, 0], [1, 1, 1]])
    left, right = np.array([0, 0, 1, 2]), np.array([0, 1, 1, 0])
    weights = metrics.judgment_weights(multiplicities, left, right)
    assert weights.tolist() == [[2, 6, 3, 0], [1, 1, 1, 1]]
    # Within a manuscript drawn twice, a judgment counts twice, not four times.
    hits = np.array([1.0, 0.0, 1.0, 0.0])
    assert metrics.bootstrap_accuracy(hits, left, right, multiplicities).tolist() == [5 / 11, 0.5]
