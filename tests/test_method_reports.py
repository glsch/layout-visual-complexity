"""`lvc report`: the methods' tables and figures, from small synthetic evaluations, and the local complexity maps."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import write_token_set
from layout_visual_complexity.embed import EmbedSettings, run_embed
from layout_visual_complexity.evaluate import EvaluateSettings, LearningCurveSettings, run_evaluate
from layout_visual_complexity.evaluation.folds import FoldSettings
from layout_visual_complexity.evaluation.vit_head import HeadSettings, HeadWeights, TrainedHead
from layout_visual_complexity.features.vit import TokenSettings, read_token_set
from layout_visual_complexity.ranking.dataset import read_dataset
from layout_visual_complexity.reports import ReportSettings, run_report
from test_evaluate_command import setup
from test_features_contract import write_store

FOLDS = FoldSettings(k=2, repeats=2, seed=5)


def evaluations(folder: Path, dataset_folder: Path) -> tuple[dict[str, Path], Path]:
    """Two feature methods and the head on made-up tokens, with learning curves, and two ensembles."""
    features, scores = setup(folder, dataset_folder)
    pages = read_dataset(dataset_folder).images["image"].tolist()
    tokens = write_token_set(folder / "tokens", dict(zip(pages, np.linspace(-2, 2, len(pages)), strict=True)))
    curves = LearningCurveSettings(shares=[0.5])
    common = {"folds": FOLDS, "bootstrap": 50, "jobs": 1, "learning_curves": curves}
    methods = {
        "geometry": {"features": features, "families": "signal"},
        "colour": {"features": features, "families": "other"},
        "vit": {"tokens": tokens, "head": HeadSettings(hidden=4, epochs=30, learning_rate=0.01)},
    }
    found = {}
    for name, method in methods.items():
        found[name] = folder / name
        run_evaluate(EvaluateSettings(dataset_folder, scores, found[name], **method, **common))  # type: ignore[arg-type]
    pair = {"geometry": found["geometry"], "vit": found["vit"]}
    for name, fitted in (("ensemble", False), ("ensemble_fitted", True)):
        found[name] = folder / name
        settings = EvaluateSettings(dataset_folder, scores, found[name], ensemble=pair, fitted_weights=fitted, **common)  # type: ignore[arg-type]
        run_evaluate(settings)
    return found, scores


def test_the_methods_tables_and_figures(tmp_path: Path, dataset_folder: Path) -> None:
    pytest.importorskip("torch")
    found, scores = evaluations(tmp_path / "inputs", dataset_folder)
    output = tmp_path / "results"
    labels = {"geometry": "Geometry", "colour": "Colour", "vit": "ViT", "ensemble": "Ensemble",
              "ensemble_fitted": "Ensemble, fitted"}  # fmt: skip
    main = {key: found[key] for key in ("geometry", "vit", "ensemble")}
    settings = ReportSettings(
        output,
        ["methods_by_elo_gap", "accuracy_by_elo_gap", "paired_differences", "learning_curves_and_calibration"],
        dataset=dataset_folder,
        scores=scores,
        evaluations=main,
        labels=labels,
        comparisons=[["vit", "geometry"], ["ensemble", "geometry"]],
    )
    summary = run_report(settings)
    table = pd.read_csv(output / "tables" / "methods_by_elo_gap.tsv", sep="\t")
    assert table["gap"].tolist()[-2:] == ["overall", "srocc"]
    assert {"consistency", "geometry", "vit_low", "ensemble_high"} <= set(table.columns)
    latex = (output / "tables" / "methods_by_elo_gap.tex").read_text()
    assert latex.count("[") == 1 + 4 + 3  # [htbp], then intervals in Overall (with consistency) and SROCC
    assert "the mean of the z-scored page scores of \\emph{Geometry} and \\emph{ViT}" in latex
    assert "the local scoring head on the patch tokens of tiny" in latex
    assert "the pairwise logistic model on the 2 signal features" in latex
    every = (output / "tables" / "accuracy_by_elo_gap.tex").read_text()
    assert every.count("[") > latex.count("[")
    for suffix in (".svg", ".png", ".tex"):
        assert (output / "figures" / f"accuracy_by_elo_gap{suffix}").is_file()
        assert (output / "figures" / f"learning_curves_and_calibration{suffix}").is_file()
    differences = pd.read_csv(output / "tables" / "paired_differences.tsv", sep="\t")
    assert differences[["first", "second"]].values.tolist() == [["vit", "geometry"], ["ensemble", "geometry"]]
    assert (differences["accuracy_low"] <= differences["accuracy_difference"]).all()
    assert (differences["srocc_low"] <= differences["srocc_high"]).all()
    curves = pd.read_csv(output / "tables" / "learning_curves.tsv", sep="\t")
    assert len(curves) == 3 * 4 and set(curves["method"]) == {"geometry", "vit", "ensemble"}
    assert "learning_curves_and_calibration" in summary

    appendix = ReportSettings(output, ["ensembles_by_elo_gap"], evaluations={key: found[key] for key in
                              ("ensemble", "ensemble_fitted")}, labels=labels)  # fmt: skip
    run_report(appendix)
    caption = (output / "tables" / "ensembles_by_elo_gap.tex").read_text()
    assert "weighted as each fold's training judgments fit" in caption
    assert "\\emph{Geometry}, the pairwise logistic model on the 2 signal features" in caption


def test_the_local_complexity_maps(tmp_path: Path, tiny_vit: Path) -> None:
    store = write_store(tmp_path / "segment", pages=3)
    tokens = TokenSettings(backbone=str(tiny_vit), revision=None, long_edge=56)
    run_embed(EmbedSettings(store, tmp_path / "tokens", tokens, "cpu", tmp_path / "user_cache"))
    token_set = read_token_set(tmp_path / "tokens")
    rng = np.random.default_rng(1)
    weights = HeadWeights(rng.normal(size=(16, 4)), rng.normal(size=4), rng.normal(size=4), 0.0)
    head = TrainedHead(weights, token_set.identity(), asdict(HeadSettings()), 0.5, 0.1, 1.0, {})
    head_path = head.save(tmp_path / "head.npz")
    output = tmp_path / "results"
    settings = ReportSettings(output, ["local_complexity_maps"], layout=store, tokens=tmp_path / "tokens",
                              head=head_path, map_pages=["ms_1r", "ms_2r", "ms_3r"], map_range=[0.1, 0.9],
                              image_credit="Synthetic pages.")  # fmt: skip
    summary = run_report(settings)
    for suffix in (".svg", ".png", ".tex"):
        assert (output / "figures" / f"local_complexity_maps{suffix}").is_file()
    caption = (output / "figures" / "local_complexity_maps.tex").read_text()
    assert "ms\\_2r" in caption and "Synthetic pages." in caption and "from the 10th to the 90th percentile" in caption
    found = summary["local_complexity_maps"]
    assert set(found["pages"]) == {"ms_1r", "ms_2r", "ms_3r"} and found["scale"][0] < found["scale"][1]
    other = TrainedHead(weights, token_set.identity() | {"long_edge": 518}, asdict(HeadSettings()), 0, 1, 1, {})
    settings = ReportSettings(output, ["local_complexity_maps"], layout=store, tokens=tmp_path / "tokens",
                              head=other.save(tmp_path / "other.npz"), map_pages=["ms_1r"])  # fmt: skip
    with pytest.raises(ValueError, match="long_edge"):
        run_report(settings)
