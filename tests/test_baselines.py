"""The baselines' evaluations and outputs: the mean token, measures chosen by their columns, the drop rule against
ink coverage, and the baselines' tables and figures from small synthetic evaluations."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from layout_visual_complexity import cli
from layout_visual_complexity.evaluate import EvaluateSettings, RestatingSettings, run_evaluate
from layout_visual_complexity.evaluation.folds import FoldSettings
from layout_visual_complexity.features.contract import Family, FeatureError, Group, aggregate
from layout_visual_complexity.features.extraction import write_feature_table
from layout_visual_complexity.features.generic import Generic, GenericSettings
from layout_visual_complexity.features.mean_token import MeanToken, MeanTokenSettings
from layout_visual_complexity.features.page import Page
from layout_visual_complexity.features.trivial import Trivial
from layout_visual_complexity.features.vit import TokenGrid
from layout_visual_complexity.ranking.dataset import read_dataset
from layout_visual_complexity.reports import ReportSettings, run_report

FOLDS = FoldSettings(k=2, repeats=2, seed=5)


def test_the_mean_token_is_the_mean_of_a_known_grid() -> None:
    tokens = np.arange(12, dtype=np.float16).reshape(4, 3)
    page = Page("ms_1r", 10, 10, tokens=TokenGrid(tokens, 2, 2, (0, 0, 10, 10)))
    family = MeanToken(MeanTokenSettings(dimensions=3))
    assert family.needs == {"tokens"}
    values = aggregate(family, family.measure(page))
    assert list(values) == ["MeanToken__Dimension__d0", "MeanToken__Dimension__d1", "MeanToken__Dimension__d2"]
    assert list(values.values()) == [4.5, 5.5, 6.5]
    with pytest.raises(FeatureError, match="wide"):
        MeanToken().measure(page)


class Coverage(Family):
    """A test family standing for ink coverage."""

    name = "coverage"

    def groups(self) -> list[Group]:
        return [Group("Test", "Coverage", "Test__Coverage", ("share",), "Ink coverage, made up.")]


class Restated(Family):
    """A test family of two measures: one equal to the coverage, one following the pages' strength."""

    name = "restated"

    def groups(self) -> list[Group]:
        return [
            Group("Test", "Copy", "Test__Copy", ("share",), "The coverage again."),
            Group("Test", "Strength", "Test__Strength", ("value", "noise"), "The pages' strength, and noise."),
        ]


def setup(folder: Path, dataset_folder: Path) -> tuple[Path, Path]:
    dataset = read_dataset(dataset_folder)
    pages = dataset.images["image"].tolist()
    rng = np.random.default_rng(2)
    coverage = rng.normal(size=len(pages))
    values = pd.DataFrame(
        {
            "Test__Coverage__share": coverage,
            "Test__Copy__share": coverage,
            "Test__Strength__value": np.linspace(-2, 2, len(pages)),
            "Test__Strength__noise": rng.normal(size=len(pages)),
        },
        index=pages,
    )
    features = write_feature_table(folder / "features.parquet", values, [Coverage(), Restated()])
    wins = pd.concat([dataset.judgments["chosen"], dataset.judgments["left"], dataset.judgments["right"]])
    scores = pd.DataFrame({"image": pages, "elo_mean": [1500 + 40 * (wins == page).sum() for page in pages]})
    scores.to_csv(folder / "scores.tsv", sep="\t", index=False)
    return features, folder / "scores.tsv"


def test_a_measure_equal_to_ink_coverage_is_dropped_from_its_groups_combination(
    tmp_path: Path, dataset_folder: Path
) -> None:
    features, scores = setup(tmp_path, dataset_folder)
    restating = RestatingSettings(reference="Test__Coverage__share", threshold=0.9)
    output = tmp_path / "restated"
    summary = run_evaluate(EvaluateSettings(dataset_folder, scores, output, features, families="restated",
                                            restating=restating, folds=FOLDS, bootstrap=50))  # fmt: skip
    table = pd.read_csv(output / "restating.tsv", sep="\t").set_index("column")
    assert table.at["Test__Copy__share", "rho"] == pytest.approx(1)
    assert bool(table.at["Test__Copy__share", "compared"]) and bool(table.at["Test__Copy__share", "dropped"])
    assert table.at["Test__Copy__share", "accuracy_difference"] == pytest.approx(0)
    assert not table["dropped"].drop("Test__Copy__share").any()
    assert summary["columns"] == 2 and summary["dropped"] == ["Test__Copy__share"]
    assert "Test__Copy__share" not in (output / "model.json").read_text()
    assert "Left out" in (output / "report.md").read_text()


def test_a_measure_is_chosen_by_its_prefix(tmp_path: Path, dataset_folder: Path) -> None:
    features, scores = setup(tmp_path, dataset_folder)
    output = tmp_path / "strength"
    arguments = ["--quiet", "evaluate", "--dataset", str(dataset_folder), "--scores", str(scores), "--output",
                 str(output), "--features", str(features), "--columns", "Test__Strength", "--folds.k", "2",
                 "--folds.repeats", "1", "--bootstrap", "20"]  # fmt: skip
    assert cli.main(arguments) == 0
    summary = pd.read_json(output / "manifest.json", typ="series")["summary"]
    assert summary["selected"] == ["Test__Strength__value", "Test__Strength__noise"]
    assert summary["measures"] == [{"prefix": "Test__Strength", "family": "Test", "feature": "Strength", "columns": 2}]
    with pytest.raises(ValueError, match="no column"):
        run_evaluate(EvaluateSettings(dataset_folder, scores, tmp_path / "none", features, columns="Test__None"))


def baseline_evaluations(folder: Path, dataset_folder: Path) -> tuple[dict[str, Path], Path, Path]:
    """Three evaluations of the made-up measures: the strength alone, the copy alone, and both families together."""
    features, scores = setup(folder, dataset_folder)
    common = {"folds": FOLDS, "bootstrap": 40}
    methods = {
        "strength": {"columns": "Test__Strength"},
        "copy": {"columns": "Test__Copy"},
        "together": {"families": "restated"},
    }
    found = {}
    for name, method in methods.items():
        found[name] = folder / name
        run_evaluate(EvaluateSettings(dataset_folder, scores, found[name], features, **method, **common))  # type: ignore[arg-type]
    return found, features, scores


def test_the_baselines_tables_and_figures(tmp_path: Path, dataset_folder: Path) -> None:
    found, features, scores = baseline_evaluations(tmp_path / "inputs", dataset_folder)
    output = tmp_path / "results"
    labels = {"strength": "Strength alone", "copy": "Copy alone", "together": "Both"}
    settings = ReportSettings(
        output,
        [
            "baselines_by_elo_gap",
            "baselines_by_information",
            "baselines_by_information_each_measure",
            "baseline_differences",
        ],  # fmt: skip
        dataset=dataset_folder,
        scores=scores,
        evaluations=found,
        labels=labels,
        levels={"strength": "Trivial", "copy": "Trivial", "together": "Image only"},
        alone=["strength", "copy"],
        colours={"strength": "trivial", "copy": "trivial", "together": "generic_image"},
        comparisons=[["together", "strength"]],
    )
    summary = run_report(settings)
    table = pd.read_csv(output / "tables" / "baselines_by_elo_gap.tsv", sep="\t")
    assert {"strength", "copy", "together"} <= set(table.columns)
    caption = (output / "tables" / "baselines_by_elo_gap.tex").read_text()
    assert "\\emph{Strength alone}, the pairwise logistic model on strength alone (two columns)" in caption
    assert "\\emph{Both}, the pairwise logistic model on the 3 restated features" in caption
    for stem in ("baselines_by_information", "baselines_by_information_each_measure"):
        for suffix in (".svg", ".png", ".tex"):
            assert (output / "figures" / f"{stem}{suffix}").is_file()
    placed = pd.read_csv(output / "tables" / "baselines_by_information_each_measure.tsv", sep="\t")
    assert placed["alone"].tolist() == [True, True, False]
    assert placed["level"].tolist() == ["Trivial", "Trivial", "Image only"]
    drawn = (output / "figures" / "baselines_by_information.svg").read_text()
    assert ">Reference<" in drawn and "ceiling" in drawn and "Consistency" not in drawn
    figure = (output / "figures" / "baselines_by_information_each_measure.tex").read_text()
    assert "open dots: one measure alone" in figure and "\\emph{Both}" in figure
    assert "\\emph{Copy alone}" not in figure
    differences = pd.read_csv(output / "tables" / "baseline_differences.tsv", sep="\t")
    assert differences[["first", "second"]].values.tolist() == [["together", "strength"]]
    assert "baseline_differences" in summary

    with pytest.raises(ValueError, match="no level of information for copy"):
        run_report(ReportSettings(output, ["baselines_by_information"], evaluations=found,
                                  levels={"strength": "Trivial", "together": "Image only"}))  # fmt: skip


def test_the_measures_table_and_the_generic_measures_correlations(tmp_path: Path, dataset_folder: Path) -> None:
    pages = read_dataset(dataset_folder).images["image"].tolist()
    rng = np.random.default_rng(4)
    families = [Trivial(), Generic(GenericSettings(codebook=str(tmp_path / "codebook.npz")))]
    columns = [column for family in families for column in family.columns()]
    values = pd.DataFrame(rng.normal(size=(len(pages), len(columns))), index=pages, columns=columns)
    values["Generic__Colourfulness__opponent"] = values["Trivial__InkCoverage__share"] * 2
    features = write_feature_table(tmp_path / "features.parquet", values, families)
    output = tmp_path / "results"
    run_report(ReportSettings(output, ["baseline_measures"], features=features, families=["trivial", "generic"]))
    measures = pd.read_csv(output / "tables" / "baseline_measures.tsv", sep="\t")
    assert measures["code_name"].tolist()[:3] == ["Trivial__InkCoverage", "Trivial__TextLines",
                                                  "Generic__EdgeOrientationEntropy"]  # fmt: skip
    assert measures["source"].notna().all() and measures["parameters"].notna().all()
    assert "\\emph{Generic}" in (output / "tables" / "baseline_measures.tex").read_text()

    settings = ReportSettings(output, ["generic_measure_correlations"], dataset=dataset_folder, features=features,
                              families=["generic"], reference_column="Trivial__InkCoverage__share",
                              bootstrap=30)  # fmt: skip
    run_report(settings)
    found = pd.read_csv(output / "tables" / "generic_measure_correlations.tsv", sep="\t").set_index("column")
    assert found.at["Generic__Colourfulness__opponent", "rho_reference"] == pytest.approx(1)
    assert len(found) == 15
    matrix = pd.read_csv(output / "tables" / "generic_measure_correlations_matrix.tsv", sep="\t")
    assert len(matrix) == 16 * 16
    for suffix in (".svg", ".png", ".tex"):
        assert (output / "figures" / f"generic_measure_correlations{suffix}").is_file()
