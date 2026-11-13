"""`lvc evaluate` on a tiny synthetic dataset: folds by manuscript, intervals, the trained model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from layout_visual_complexity import cli
from layout_visual_complexity.evaluate import EvaluateSettings, run_evaluate
from layout_visual_complexity.evaluation.folds import FoldSettings, manuscript_folds
from layout_visual_complexity.evaluation.model import FeatureMismatch, TrainedModel
from layout_visual_complexity.features.contract import Family, Group
from layout_visual_complexity.features.extraction import read_feature_table, write_feature_table
from layout_visual_complexity.ranking.dataset import read_dataset


class Signal(Family):
    """A test family of two columns: one that follows the pages' strength, one noise."""

    name = "signal"

    def groups(self) -> list[Group]:
        return [Group("Test", "Signal", "Test__Signal", ("strength", "noise"), "Strength and noise.")]


class Other(Family):
    name = "other"

    def groups(self) -> list[Group]:
        return [Group("Test", "Other", "Test__Other", ("value",), "Another value.")]


def setup(tmp_path: Path, dataset_folder: Path) -> tuple[Path, Path]:
    dataset = read_dataset(dataset_folder)
    pages = dataset.images["image"].tolist()
    rng = np.random.default_rng(1)
    values = pd.DataFrame(
        {
            "Test__Signal__strength": np.linspace(-2, 2, len(pages)),
            "Test__Signal__noise": rng.normal(size=len(pages)),
            "Test__Other__value": rng.normal(size=len(pages)),
        },
        index=pages,
    )
    features = write_feature_table(tmp_path / "features.parquet", values, [Signal(), Other()])
    wins = pd.concat([dataset.judgments["chosen"], dataset.judgments["left"], dataset.judgments["right"]])
    scores = pd.DataFrame({"image": pages, "elo_mean": [1500 + 40 * (wins == page).sum() for page in pages]})
    scores_path = tmp_path / "scores.tsv"
    scores.to_csv(scores_path, sep="\t", index=False)
    return features, scores_path


def test_evaluate_folds_by_manuscript_with_intervals_and_keeps_the_model(tmp_path: Path, dataset_folder: Path) -> None:
    features, scores = setup(tmp_path, dataset_folder)
    output = tmp_path / "evaluation"
    folds = FoldSettings(k=2, repeats=3, seed=5)
    settings = EvaluateSettings(dataset_folder, scores, output, features, "signal", folds=folds, bootstrap=200, seed=2)
    summary = run_evaluate(settings)
    assert summary["columns"] == 2 and summary["folds"] == 6

    dataset = read_dataset(dataset_folder)
    manuscripts = dataset.images.set_index("image")["manuscript"]
    for fold in manuscript_folds(dataset.images["image"].tolist(), manuscripts, dataset.judgments, 2, 3, 5):
        assert not set(fold.train_pages) & set(fold.test_pages)
        assert not set(manuscripts[fold.train_pages]) & set(manuscripts[fold.test_pages])

    metrics = pd.read_csv(output / "metrics.tsv", sep="\t").set_index("metric")
    for metric in ("accuracy", "accuracy_pooled", "consistency", "srocc"):
        low, value, high = metrics.loc[metric, ["low", "value", "high"]]
        assert np.isfinite([low, value, high]).all() and low <= high
    by_gap = pd.read_csv(output / "by_elo_gap.tsv", sep="\t")
    assert by_gap["gap"].tolist()[-1] == "overall"
    assert {"accuracy", "accuracy_low", "accuracy_high", "consistency", "consistency_low"} <= set(by_gap.columns)
    scores_out = pd.read_csv(output / "scores.tsv", sep="\t")
    assert len(scores_out) == 12 and scores_out["score"].notna().all()
    assert (output / "calibration.tsv").is_file() and (output / "report.md").is_file()

    model = TrainedModel.load(output / "model.json")
    assert model.model.columns == ["Test__Signal__strength", "Test__Signal__noise"]
    assert [family["name"] for family in model.families] == ["signal"]
    table = read_feature_table(features)
    found = model.scores(table.values, table.hashes())
    assert found.corr(table.values["Test__Signal__strength"]) > 0.9
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["configuration"]["families"] == "signal"


def test_a_model_refuses_another_familys_settings(tmp_path: Path, dataset_folder: Path) -> None:
    features, scores = setup(tmp_path, dataset_folder)
    folds = FoldSettings(k=2, repeats=1, seed=5)
    run_evaluate(EvaluateSettings(dataset_folder, scores, tmp_path / "out", features, [], folds=folds, bootstrap=20))
    model = TrainedModel.load(tmp_path / "out" / "model.json")
    table = read_feature_table(features)
    with pytest.raises(FeatureMismatch):
        model.scores(table.values, table.hashes() | {"other": "0" * 16})


def test_lvc_evaluate_from_the_command_line(tmp_path: Path, dataset_folder: Path) -> None:
    features, scores = setup(tmp_path, dataset_folder)
    output = tmp_path / "evaluation"
    arguments = ["--quiet", "evaluate", "--features", str(features), "--dataset", str(dataset_folder)]
    arguments += ["--scores", str(scores), "--output", str(output), "--families", "signal,other"]
    small = ["--folds.k", "2", "--folds.repeats", "1", "--bootstrap", "20"]
    assert cli.main([*arguments, *small]) == 0
    assert TrainedModel.load(output / "model.json").model.columns[-1] == "Test__Other__value"


def test_the_ceiling_for_srocc_is_the_square_root_of_the_reliability(tmp_path: Path, dataset_folder: Path) -> None:
    features, scores = setup(tmp_path, dataset_folder)
    split = tmp_path / "split_half_ceiling.tsv"
    rows = [("split-half correlation", 0.68), ("Spearman-Brown corrected", 0.81)]
    pd.DataFrame(rows, columns=["measure", "median"]).to_csv(split, sep="\t", index=False)
    folds = FoldSettings(k=2, repeats=1, seed=5)
    output = tmp_path / "out"
    run_evaluate(
        EvaluateSettings(dataset_folder, scores, output, features, [], folds=folds, bootstrap=20, ceiling=split)
    )
    metrics = pd.read_csv(output / "metrics.tsv", sep="\t").set_index("metric")["value"]
    assert metrics["srocc_ceiling"] == pytest.approx(0.9) and metrics["elo_reliability"] == pytest.approx(0.81)
