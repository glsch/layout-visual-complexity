"""Ensembles of evaluations, the per-fold hits and z-scores `lvc evaluate` keeps, paired comparisons read back from
them, the local ViT head evaluated on tokens, learning curves, and `lvc train`."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import write_token_set
from layout_visual_complexity import cli
from layout_visual_complexity.evaluate import EvaluateSettings, LearningCurveSettings, read_evaluation, run_evaluate
from layout_visual_complexity.evaluation.folds import FoldSettings, manuscript_folds
from layout_visual_complexity.evaluation.protocol import ensemble_fold_scores, paired_comparison
from layout_visual_complexity.evaluation.vit_head import HeadSettings, TrainedHead
from layout_visual_complexity.features.vit import TokenMismatch, read_token_set
from layout_visual_complexity.ranking.dataset import read_dataset
from layout_visual_complexity.train import TrainSettings, run_train
from test_evaluate_command import setup

FOLDS = FoldSettings(k=2, repeats=2, seed=5)
SMALL_HEAD = HeadSettings(hidden=4, epochs=40, learning_rate=0.01, seed=3)


def evaluate(tmp_path: Path, dataset_folder: Path, name: str, **method: object) -> Path:
    features, scores = setup(tmp_path, dataset_folder)
    if "features" in method:
        method["features"] = features
    output = tmp_path / name
    run_evaluate(EvaluateSettings(dataset_folder, scores, output, folds=FOLDS, bootstrap=50, jobs=1, **method))  # type: ignore[arg-type]
    return output


def test_each_folds_scores_are_z_scored_by_its_training_pages(tmp_path: Path, dataset_folder: Path) -> None:
    folder = evaluate(tmp_path, dataset_folder, "signal", features=True, families="signal")
    fold_scores = pd.read_csv(folder / "fold_scores.tsv", sep="\t")
    for _, fold in fold_scores.groupby("fold_number"):
        train = fold.loc[~fold["test"], "z"]
        assert abs(train.mean()) < 1e-5 and abs(train.std(ddof=0) - 1) < 1e-5
        assert fold["test"].any()


def test_the_hits_of_each_fold_are_kept_and_compare_in_pairs(tmp_path: Path, dataset_folder: Path) -> None:
    first = evaluate(tmp_path, dataset_folder, "signal", features=True, families="signal")
    second = evaluate(tmp_path, dataset_folder, "other", features=True, families="other")
    hits = pd.read_csv(first / "hits.tsv", sep="\t")
    folds = pd.read_csv(first / "folds.tsv", sep="\t")
    assert len(hits) == folds["test"].sum() and set(hits["hit"]) <= {0.0, 0.5, 1.0}
    dataset = read_dataset(dataset_folder)
    judgments = dataset.judgments.reset_index(drop=True)
    manuscripts = dataset.images.set_index("image")["manuscript"]
    a, b = read_evaluation(first, judgments), read_evaluation(second, judgments)
    same = paired_comparison(a, a, judgments, manuscripts)
    assert same["difference"] == 0 and same["low"] == same["high"] == 0
    _, scores = setup(tmp_path, dataset_folder)
    elo = pd.read_csv(scores, sep="\t").set_index("image")["elo_mean"]
    found = paired_comparison(a, b, judgments, manuscripts, reference=elo)
    assert found["low"] <= found["difference_pooled"] <= found["high"]
    hit_a = a.tested.groupby("row")["hit"].mean()
    hit_b = b.tested.groupby("row")["hit"].mean()
    assert found["difference_pooled"] == pytest.approx((hit_a - hit_b).mean())
    assert found["srocc_low"] <= found["srocc_difference"] <= found["srocc_high"]
    assert found["difference"] > 0


def members(dataset_folder: Path) -> tuple[dict[str, pd.DataFrame], list, pd.DataFrame]:
    dataset = read_dataset(dataset_folder)
    pages = dataset.images["image"].tolist()
    judgments = dataset.judgments.reset_index(drop=True)
    folds = manuscript_folds(pages, dataset.images.set_index("image")["manuscript"], judgments, 2, 1, 5)
    rng = np.random.default_rng(4)
    strength = pd.Series(np.linspace(-2, 2, len(pages)), index=pages)
    tables: dict[str, list[pd.DataFrame]] = {"good": [], "noise": []}
    for number, fold in enumerate(folds):
        both = np.concatenate([fold.train_pages, fold.test_pages])
        test = np.isin(both, fold.test_pages)
        for name, values in (("good", strength[both].to_numpy()), ("noise", rng.normal(size=len(both)))):
            tables[name].append(pd.DataFrame({"fold_number": number, "page": both, "test": test, "z": values}))
    return {name: pd.concat(parts) for name, parts in tables.items()}, folds, judgments


def test_equal_weights_from_the_configuration_average_the_members(dataset_folder: Path) -> None:
    found, folds, judgments = members(dataset_folder)
    scores, weights = ensemble_fold_scores(found, folds, judgments, {"good": 0.5, "noise": 0.5})
    for number in range(len(folds)):
        tables = {name: table[table["fold_number"] == number].set_index("page")["z"] for name, table in found.items()}
        expected = (tables["good"] + tables["noise"]) / 2
        pd.testing.assert_series_equal(scores[number], expected.reindex(scores[number].index), check_names=False)
    assert weights[["good", "noise"]].to_numpy().tolist() == [[0.5, 0.5]] * len(folds)


def test_fitted_weights_use_the_training_fold_alone(dataset_folder: Path) -> None:
    found, folds, judgments = members(dataset_folder)
    _, weights = ensemble_fold_scores(found, folds, judgments, None)
    assert (weights["good"] > weights["noise"].abs()).all()
    scrambled = {}
    for name, table in found.items():
        table = table.copy()
        table.loc[table["test"], "z"] = np.random.default_rng(9).normal(size=int(table["test"].sum())) * 10
        scrambled[name] = table
    _, same = ensemble_fold_scores(scrambled, folds, judgments, None)
    pd.testing.assert_frame_equal(weights, same)
    changed = {name: table.assign(z=np.where(table["test"], table["z"], -table["z"])) for name, table in found.items()}
    _, other = ensemble_fold_scores(changed, folds, judgments, None)
    assert (other["good"] < 0).all()


def test_an_ensemble_refuses_members_evaluated_on_other_folds(dataset_folder: Path) -> None:
    found, folds, judgments = members(dataset_folder)
    found["noise"] = found["noise"].assign(test=~found["noise"]["test"])
    with pytest.raises(ValueError, match="other folds"):
        ensemble_fold_scores(found, folds, judgments, {"good": 1, "noise": 1})


def strength(dataset_folder: Path) -> dict[str, float]:
    pages = read_dataset(dataset_folder).images["image"].tolist()
    return dict(zip(pages, np.linspace(-2, 2, len(pages)), strict=True))


def test_lvc_evaluate_trains_the_head_per_fold_and_keeps_it(tmp_path: Path, dataset_folder: Path) -> None:
    pytest.importorskip("torch")
    tokens = write_token_set(tmp_path / "tokens", strength(dataset_folder))
    folder = evaluate(tmp_path, dataset_folder, "vit", tokens=tokens, head=SMALL_HEAD)
    metrics = pd.read_csv(folder / "metrics.tsv", sep="\t").set_index("metric")
    assert metrics.at["accuracy", "value"] > 0.6
    assert len(pd.read_csv(folder / "hits.tsv", sep="\t")) == pd.read_csv(folder / "folds.tsv", sep="\t")["test"].sum()
    head = TrainedHead.load(folder / "head.npz")
    token_set = read_token_set(tokens)
    assert head.settings["epochs"] == 40 and head.trained_on["pages"] == 12
    assert head.scores(token_set).corr(pd.Series(strength(dataset_folder))) > 0.8
    head.tokens["long_edge"] = 518
    with pytest.raises(TokenMismatch):
        head.scores(token_set)


def test_an_ensemble_of_two_evaluations_with_learning_curves(tmp_path: Path, dataset_folder: Path) -> None:
    curves = LearningCurveSettings(shares=[0.5], repeats=1)
    signal = evaluate(tmp_path, dataset_folder, "signal", features=True, families="signal", learning_curves=curves)
    other = evaluate(tmp_path, dataset_folder, "other", features=True, families="other", learning_curves=curves)
    ensemble = {"signal": signal, "other": other}
    folder = evaluate(tmp_path, dataset_folder, "ensemble", ensemble=ensemble, learning_curves=curves)
    curve = pd.read_csv(folder / "learning_curve.tsv", sep="\t")
    assert curve[["by", "share"]].values.tolist() == [["images", 0.5], ["images", 1.0], ["judgments", 0.5],
                                                       ["judgments", 1.0]]  # fmt: skip
    assert curve["accuracy"].between(0, 1).all() and (curve["low"] <= curve["high"]).all()
    folds = pd.read_csv(folder / "folds.tsv", sep="\t")
    whole = curve[curve["share"] == 1.0]
    assert (whole["train_per_fold"] == folds.loc[folds["repeat"] == 0, "train"].mean()).all()
    assert pd.read_csv(folder / "learning_curve_fold_scores.tsv", sep="\t")["fold_number"].max() == 1
    model = pd.read_json(folder / "ensemble.json", typ="series")
    assert model["weights"] == {"signal": 0.5, "other": 0.5} and not model["fitted"]
    fitted = evaluate(tmp_path, dataset_folder, "fitted", ensemble=ensemble, fitted_weights=True)
    folds = pd.read_csv(fitted / "folds.tsv", sep="\t")
    assert (folds["weight_signal"] > folds["weight_other"].abs()).all()
    with pytest.raises(ValueError, match="every member"):
        evaluate(tmp_path, dataset_folder, "half", ensemble=ensemble, weights={"signal": 1.0})
    with pytest.raises(ValueError, match="no learning curves"):
        evaluate(tmp_path, dataset_folder, "curves", ensemble={"signal": signal, "fitted": fitted},
                 learning_curves=curves)  # fmt: skip


def test_lvc_train_trains_the_head_on_every_judgment(tmp_path: Path, dataset_folder: Path) -> None:
    pytest.importorskip("torch")
    tokens = write_token_set(tmp_path / "tokens", strength(dataset_folder))
    summary = run_train(TrainSettings(dataset_folder, tmp_path / "trained", tokens=tokens, head=SMALL_HEAD))
    assert summary["method"] == "tokens" and summary["judgments"] == 160
    head = TrainedHead.load(tmp_path / "trained" / "head.npz")
    evaluated = evaluate(tmp_path, dataset_folder, "vit", tokens=tokens, head=SMALL_HEAD)
    np.testing.assert_array_equal(head.weights.first, TrainedHead.load(evaluated / "head.npz").weights.first)
    arguments = ["--quiet", "train", "--dataset", str(dataset_folder), "--output", str(tmp_path / "cli")]
    assert cli.main([*arguments, "--tokens", str(tokens), "--head.epochs", "5"]) == 0
    assert TrainedHead.load(tmp_path / "cli" / "head.npz").settings["epochs"] == 5
