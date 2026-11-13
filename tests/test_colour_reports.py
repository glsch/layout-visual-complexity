"""`lvc report`: the colour level's tables and figure, and every method within and across manuscripts, from small
synthetic evaluations and pages."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from layout_visual_complexity.evaluate import EvaluateSettings, run_evaluate
from layout_visual_complexity.evaluation.folds import FoldSettings
from layout_visual_complexity.features.colour import Colour
from layout_visual_complexity.features.extraction import write_feature_table
from layout_visual_complexity.reports import ReportSettings, run_report
from test_evaluate_command import setup
from test_features_contract import write_store

FOLDS = FoldSettings(k=2, repeats=2, seed=5)


def evaluations(folder: Path, dataset_folder: Path) -> tuple[dict[str, Path], Path]:
    """Three feature methods standing for geometry, real colour, and geometry with real colour."""
    features, scores = setup(folder, dataset_folder)
    methods = {"geometry": "signal", "colour": "other", "geometry_colour": "signal,other"}
    found = {}
    for name, families in methods.items():
        found[name] = folder / name
        run_evaluate(EvaluateSettings(dataset_folder, scores, found[name], features, families, folds=FOLDS,
                                      bootstrap=50))  # fmt: skip
    return found, scores


def test_the_colour_table_and_the_differences(tmp_path: Path, dataset_folder: Path) -> None:
    found, scores = evaluations(tmp_path / "inputs", dataset_folder)
    output = tmp_path / "results"
    settings = ReportSettings(
        output,
        ["colour_by_elo_gap", "colour_differences", "within_and_across_manuscripts", "line_texture_families"],
        dataset=dataset_folder,
        scores=scores,
        evaluations=found,
        shown=["colour", "geometry_colour"],
        labels={"geometry": "Geometry", "colour": "Real colour", "geometry_colour": "Geometry with real colour"},
        comparisons=[["geometry_colour", "geometry"]],
    )
    run_report(settings)
    table = pd.read_csv(output / "tables" / "colour_by_elo_gap.tsv", sep="\t").set_index("row")
    assert "geometry" not in table.columns and {"reference", "colour", "geometry_colour"} <= set(table.columns)
    assert list(table.index[-6:]) == ["accuracy_within", "srocc_within", "added_accuracy", "added_srocc",
                                      "added_accuracy_within", "added_srocc_within"]  # fmt: skip
    geometry = pd.read_csv(found["geometry"] / "metrics.tsv", sep="\t").set_index("metric")["value"]
    combined = pd.read_csv(found["geometry_colour"] / "metrics.tsv", sep="\t").set_index("metric")["value"]
    assert table.at["added_srocc", "geometry_colour"] == pytest.approx(combined["srocc"] - geometry["srocc"], abs=1e-5)
    assert table.at["accuracy_within", "geometry_colour"] == pytest.approx(combined["accuracy_within"], abs=1e-5)
    assert table.at["accuracy_within", "reference"] == pytest.approx(combined["consistency_within"], abs=1e-5)
    assert np.isnan(table.at["added_accuracy", "colour"])
    latex = (output / "tables" / "colour_by_elo_gap.tex").read_text()
    assert "Score $\\Delta$ & Reference & Real colour & Geometry with real colour" in latex
    assert "\\emph{Geometry}, the pairwise logistic model" in latex and "Added: accuracy within" in latex
    differences = pd.read_csv(output / "tables" / "colour_differences.tsv", sep="\t")
    assert differences[["first", "second"]].values.tolist() == [["geometry_colour", "geometry"]]
    assert differences.at[0, "srocc_within_difference"] == pytest.approx(table.at["added_srocc_within",
                                                                                   "geometry_colour"])  # fmt: skip
    assert "Accuracy within & $\\rho$ within" in (output / "tables" / "colour_differences.tex").read_text()
    split = pd.read_csv(output / "tables" / "within_and_across_manuscripts.tsv", sep="\t").set_index("method")
    assert list(split.index) == ["reference", "colour", "geometry_colour"]
    assert split.at["colour", "srocc"] == pytest.approx(
        pd.read_csv(found["colour"] / "metrics.tsv", sep="\t").set_index("metric").at["srocc", "value"]
    )
    assert np.isnan(split.at["reference", "srocc_within"])
    families = pd.read_csv(output / "tables" / "line_texture_families.tsv", sep="\t")
    assert families["count"].tolist() == [1, 5, 8, 18, 6]
    assert "LineTextureFeature" in (output / "tables" / "line_texture_families.tex").read_text()


def test_the_colour_measures_and_the_ink_colour_figure(tmp_path: Path) -> None:
    store = write_store(tmp_path / "segment", pages=3)
    pages = [f"ms_{number}r" for number in range(1, 4)]
    family = Colour()
    values = pd.DataFrame(np.random.default_rng(1).random((3, len(family.columns()))), index=pages,
                          columns=family.columns())  # fmt: skip
    features = write_feature_table(tmp_path / "features.parquet", values, [family])
    output = tmp_path / "results"
    settings = ReportSettings(output, ["colour_measures", "ink_colour_on_page"], features=features, layout=store,
                              families=["colour"], colour_pages=pages[:2], image_credit="Synthetic pages.")  # fmt: skip
    run_report(settings)
    measures = pd.read_csv(output / "tables" / "colour_measures.tsv", sep="\t")
    assert measures["code_name"].tolist()[0] == "Colour__ChromaticShare" and len(measures) == 7
    assert "stamps is left out" in (output / "tables" / "colour_measures.tex").read_text()
    for suffix in (".svg", ".png", ".tex"):
        assert (output / "figures" / f"ink_colour_on_page{suffix}").is_file()
    caption = (output / "figures" / "ink_colour_on_page.tex").read_text()
    assert "(a) ms\\_1r" in caption and "Synthetic pages." in caption
    shares = pd.read_csv(output / "tables" / "ink_colour_on_page.tsv", sep="\t")
    assert shares["page"].tolist() == pages[:2] and (shares[["red", "gold", "green", "blue"]] >= 0).all().all()
