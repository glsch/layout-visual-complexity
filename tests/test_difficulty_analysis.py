"""The difficulty experiment's statistics and outputs, on synthetic pages whose coefficients are known."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from layout_visual_complexity import difficulty_analysis as analysis

pytest.importorskip("statsmodels")

TRUE = {"complexity": -0.08, "log_lines": 0.05, "century": -0.02, "latin": 0.04, "log_resolution": 0.0}


def synthetic(manuscripts: int = 30, pages: int = 12, seed: int = 5, held_out: int = 0) -> pd.DataFrame:
    """Pages of `manuscripts` manuscripts, each measure linear in the covariates with a manuscript's own intercept."""
    rng = np.random.default_rng(seed)
    rows = []
    for number in range(manuscripts):
        century, latin = int(rng.integers(12, 17)), bool(rng.integers(0, 2))
        intercept = rng.normal(0, 0.1)
        for page in range(pages):
            z, lines, height = rng.normal(), float(np.exp(rng.normal(3.5, 0.4))), float(np.exp(rng.normal(4, 0.2)))
            rows.append({"page": f"m{number}_{page}", "document": f"m{number}", "expected_elo": 1500 + 170 * z,
                         "lines": lines, "line_height": height, "century": century,
                         "language": "Latin" if latin else "French", "split": "test" if number < held_out else "train",
                         "z": z, "intercept": intercept, "noise": rng.normal(0, 0.03)})  # fmt: skip
    data = pd.DataFrame(rows)
    standard = (data["expected_elo"] - data["expected_elo"].mean()) / data["expected_elo"].std(ddof=1)
    data["measure"] = (0.8 + TRUE["complexity"] * standard + TRUE["log_lines"] * np.log(data["lines"])
                       + TRUE["century"] * data["century"] + TRUE["latin"] * (data["language"] == "Latin")
                       + data["intercept"] + data["noise"])  # fmt: skip
    return data


def test_the_mixed_model_finds_the_coefficients_with_wald_intervals() -> None:
    data = analysis.prepared(synthetic())
    fit = analysis.mixed_model(data, "measure")
    assert fit.pages == 360 and fit.manuscripts == 30
    for term, value in TRUE.items():
        estimate, low, high = fit.coefficients[term]
        assert low <= value <= high and estimate == pytest.approx(value, abs=0.03)
    assert 0 < fit.r2 < 1
    assert not any("Powell" in note for note in fit.notes)


def test_the_mixed_model_is_fitted_again_by_powells_method_when_it_doesnt_converge(monkeypatch) -> None:
    import statsmodels.api as sm

    fit = sm.MixedLM.fit

    def failing(self, *args, **kwargs):
        result = fit(self, *args, **kwargs)
        if "method" not in kwargs:
            result.converged = False
        return result

    monkeypatch.setattr(sm.MixedLM, "fit", failing)
    found = analysis.mixed_model(analysis.prepared(synthetic()), "measure")
    assert any("Powell" in note for note in found.notes) and "did not converge" not in found.notes
    estimate, low, high = found.coefficients["complexity"]
    assert low <= TRUE["complexity"] <= high


def test_least_squares_across_and_within_find_the_coefficients_and_resample_by_the_seed() -> None:
    data = analysis.prepared(synthetic())
    across = analysis.least_squares(data, "measure", resamples=200, seed=20260629)
    within = analysis.least_squares(data, "measure", within=True, resamples=200, seed=20260629)
    assert across.coefficients["complexity"][0] == pytest.approx(TRUE["complexity"], abs=0.02)
    assert set(within.coefficients) == {"complexity", "log_lines", "log_resolution"}  # the manuscripts' own drop out
    for term in ("complexity", "log_lines"):
        estimate, low, high = within.coefficients[term]
        assert estimate == pytest.approx(TRUE[term], abs=0.01) and low < estimate < high
    again = analysis.least_squares(data, "measure", within=True, resamples=200, seed=20260629)
    assert again.coefficients == within.coefficients
    other = analysis.least_squares(data, "measure", within=True, resamples=200, seed=1)
    assert other.coefficients["complexity"][1] != within.coefficients["complexity"][1]


def test_the_held_out_terms_and_the_correlations() -> None:
    data = analysis.prepared(synthetic(held_out=10))
    fit = analysis.mixed_model(data, "measure", analysis.HELD_OUT_TERMS)
    assert {"held_out", "complexity:held_out"} <= set(fit.coefficients)
    within = analysis.least_squares(data, "measure", analysis.HELD_OUT_TERMS, within=True, resamples=50)
    assert "held_out" not in within.coefficients and "complexity:held_out" in within.coefficients
    found = analysis.correlation(data, "measure", resamples=100, seed=3)
    assert found.overall[0] < 0 and found.overall[1] <= found.overall[0] <= found.overall[2]
    assert found.within[0] < 0 and found.pages == 360 and found.manuscripts == 30
    assert analysis.correlation(data, "measure", resamples=100, seed=3) == found
    similar = analysis.similar_pages(data, ["measure"])
    assert similar.loc[0, "held_out_pages"] == 120


def test_pooled_rates_count_every_character() -> None:
    data = pd.DataFrame({"manuscript": ["a", "a", "b"], "errors": [1, 2, 3], "characters": [10, 10, 20]})
    value, low, high = analysis.pooled(data, "errors", "characters", resamples=100)
    assert value == pytest.approx(6 / 40) and low <= value <= high


def write_run(folder: Path, data: pd.DataFrame, transcribed: bool, misplaced: int = 0, outside: int = 0) -> Path:
    """A run's rows: the first `misplaced` pages with texts that don't fit their lines, the first `outside` with many
    found lines outside every ground-truth zone."""
    folder.mkdir(parents=True)
    rows = data.drop(columns=["intercept", "noise", "century", "language", "split"]).copy()
    rng = np.random.default_rng(1)
    rows["ensemble"] = (rows["expected_elo"] - 1500) / 170
    rows["geometry"] = rows.pop("z") + rng.normal(0, 0.5, len(rows))
    rows["vit"] = rows["ensemble"] + rng.normal(0, 0.8, len(rows))
    rows["status"] = "done"
    rows["lines_found"], rows["lines_matched"], rows["lines_invented"], rows["lines_missed"] = 50, 45, 5, 3
    rows["zones_matched"], rows["zones_invented"], rows["zones_outside"] = 4, 1, 0
    rows["lines_outside"] = [10 if number < outside else 1 for number in range(len(rows))]
    rows["lines_shifted"] = [4 if number < misplaced else 0 for number in range(len(rows))] if transcribed else np.nan
    rows["lines_read_empty"] = 0 if transcribed else np.nan
    rows["texts_misplaced"] = [number < misplaced for number in range(len(rows))]
    for name in ("zone_f1", "line_f1"):
        rows[name] = np.clip(rows["measure"] + rng.normal(0, 0.02, len(rows)), 0, 1)
    rows["baseline_distance"] = 0.2 - rows["measure"] / 10
    rows["lines"] = rows["lines"].round()
    for suffix in ("", "_main", "_margin"):
        rows[f"characters{suffix}"] = 1000.0
        cer = np.clip(1 - rows["measure"], 0.001, 1) ** 2 if transcribed else np.nan
        rows[f"errors_lines{suffix}"] = cer * 1000
        rows[f"errors_end_to_end{suffix}"] = cer * 1500
        rows[f"cer_lines{suffix}"] = cer
        rows[f"cer_end_to_end{suffix}"] = cer * 1.5
    for part in ("main", "margin"):
        rows[f"{part}_lines"] = 10
        rows[f"{part}_lines_matched"], rows[f"{part}_lines_invented"], rows[f"{part}_lines_missed"] = 8, 1, 2
        rows[f"{part}_line_f1"] = rows["line_f1"]
    rows.to_csv(folder / "difficulty.tsv", sep="\t", index=False)
    return folder


def test_the_report_writes_the_figures_and_tables(tmp_path: Path) -> None:
    from layout_visual_complexity.reports import ReportSettings, run_report

    main = synthetic(manuscripts=12, pages=8, seed=2, held_out=2)
    test = synthetic(manuscripts=6, pages=5, seed=3)
    test["document"] = "t" + test["document"]
    test["page"] = "t" + test["page"]
    no_text_lines = test.index[test["century"] >= 14][0]
    test.loc[no_text_lines, "lines"] = 0
    runs = {"main": write_run(tmp_path / "main", main, True), "test": write_run(tmp_path / "test", test, False)}
    facts = pd.concat([main, test]).groupby("document").first().reset_index()
    table = pd.DataFrame({"source": "S", "folder": facts["document"], "shelfmark": "Shelf " + facts["document"],
                          "language": facts["language"], "century": facts["century"],
                          "split": np.where(facts["document"].str.startswith("t"), "test", facts["split"]),
                          "comparison": np.where(facts["century"] >= 14, "yes", "no")})  # fmt: skip
    table.to_csv(tmp_path / "manuscripts.tsv", sep="\t", index=False)
    outputs = ["difficulty_against_complexity", "difficulty_regression", "difficulty_regression_across_and_within",
               "difficulty_held_out", "difficulty_correlations", "difficulty_main_text_and_margins",
               "difficulty_ground_truth_coverage", "difficulty_by_ranker", "difficulty_by_quarter",
               "difficulty_material", "difficulty_pages"]  # fmt: skip
    settings = ReportSettings(output=tmp_path / "results", outputs=outputs, name="difficulty", bootstrap=50,
                              difficulty_runs=runs, difficulty_main="main",
                              manuscript_table=tmp_path / "manuscripts.tsv")  # type: ignore[arg-type]  # fmt: skip
    run_report(settings)
    found = {path.relative_to(tmp_path / "results").as_posix() for path in (tmp_path / "results").rglob("*.*")}
    for stem in ("figures/difficulty_against_complexity", "tables/difficulty_regression",
                 "tables/difficulty_regression_across_and_within", "tables/difficulty_held_out",
                 "tables/difficulty_correlations", "tables/difficulty_main_text_and_margins",
                 "tables/difficulty_ground_truth_coverage", "tables/difficulty_by_ranker",
                 "tables/difficulty_by_quarter", "tables/difficulty_held_out_pages",
                 "tables/difficulty_material"):  # fmt: skip
        assert f"{stem}.tex" in found
    assert {"figures/difficulty_against_complexity.svg", "difficulty/main.tsv", "difficulty/test.tsv",
            "tables/difficulty_held_out_pages.tsv"} <= found  # fmt: skip
    assert not any("log_cer" in name for name in found)
    regression = pd.read_csv(tmp_path / "results" / "tables" / "difficulty_regression.tsv", sep="\t")
    assert set(regression["measure"]) >= {"zone_f1", "line_f1", "baseline_distance", "cer_lines", "cer_end_to_end"}
    material = pd.read_csv(tmp_path / "results" / "tables" / "difficulty_material.tsv", sep="\t")
    assert material["pages"].sum() == len(main) + len(test)
    from layout_visual_complexity.difficulty_reports import DifficultySettings, load

    loaded = load(DifficultySettings(runs, "main", tmp_path / "manuscripts.tsv"))
    assert len(loaded.rows) == len(main) + len(test)
    assert test.loc[no_text_lines, "page"] not in set(loaded.comparison["page"])
    assert len(loaded.comparison) == (test["century"] >= 14).sum() - 1 + (main["century"] >= 14).sum()
    caption = (tmp_path / "results" / "figures" / "difficulty_against_complexity.tex").read_text(encoding="utf-8")
    # The figure shows the main run's pages only: the held-out run's page without a text line isn't counted.
    assert "both on a log axis" in caption and f"(c) all {len(main)}." in caption and "Left out" not in caption
    assert "Vermillion: the 0 pages" not in caption


def report_runs(tmp_path: Path, misplaced: int = 0, outside: int = 0) -> tuple[dict[str, Path], Path]:
    """A main run of 12 manuscripts, the first `misplaced` pages with texts that don't fit their lines and the first
    `outside` with many lines outside every zone, and a held-out run without transcriptions; the table of
    manuscripts."""
    main = synthetic(manuscripts=12, pages=8, seed=2, held_out=2)
    test = synthetic(manuscripts=6, pages=5, seed=3)
    test["document"], test["page"] = "t" + test["document"], "t" + test["page"]
    runs = {"main": write_run(tmp_path / "main", main, True, misplaced, outside),
            "test": write_run(tmp_path / "test", test, False, 0, 2)}  # fmt: skip
    facts = pd.concat([main, test]).groupby("document").first().reset_index()
    table = pd.DataFrame({"source": np.where(facts["document"].str.startswith("t"), "CATMuS", "CREMMA Medieval"),
                          "folder": facts["document"], "shelfmark": "Shelf " + facts["document"],
                          "language": facts["language"], "century": facts["century"],
                          "split": np.where(facts["document"].str.startswith("t"), "test", facts["split"]),
                          "comparison": np.where(facts["century"] >= 14, "yes", "no")})  # fmt: skip
    table.to_csv(tmp_path / "manuscripts.tsv", sep="\t", index=False)
    return runs, tmp_path / "manuscripts.tsv"


def report(tmp_path: Path, outputs: list[str], misplaced: int = 0, outside: int = 0) -> Path:
    from layout_visual_complexity.reports import ReportSettings, run_report

    runs, table = report_runs(tmp_path, misplaced, outside)
    settings = ReportSettings(output=tmp_path / "results", outputs=outputs, name="difficulty", bootstrap=50,
                              difficulty_runs=runs, difficulty_main="main",
                              manuscript_table=table)  # type: ignore[arg-type]  # fmt: skip
    run_report(settings)
    return tmp_path / "results"


def test_pages_whose_texts_dont_fit_are_left_out_of_every_cer_and_counted(tmp_path: Path) -> None:
    from layout_visual_complexity.difficulty_reports import DifficultySettings, load

    folder = report(tmp_path, ["difficulty_regression", "difficulty_against_complexity",
                               "difficulty_main_text_and_margins"], misplaced=3)  # fmt: skip
    runs = {"main": tmp_path / "main", "test": tmp_path / "test"}
    material = load(DifficultySettings(runs, "main", tmp_path / "manuscripts.tsv"))
    flagged = material.main["page"].isin(["m0_0", "m0_1", "m0_2"])
    assert material.misplaced["page"].tolist() == ["m0_0", "m0_1", "m0_2"]
    for column in ("cer_lines", "cer_end_to_end", "cer_lines_main", "errors_end_to_end_margin",
                   "characters", "log_cer_lines", "log_cer_end_to_end"):  # fmt: skip
        assert material.main.loc[flagged, column].isna().all() and material.main.loc[~flagged, column].notna().all()
    assert material.main[["line_f1", "zone_f1", "baseline_distance"]].notna().all().all()
    regression = pd.read_csv(folder / "tables" / "difficulty_regression.tsv", sep="\t")
    pages = regression[regression["term"] == "complexity"].set_index("measure")["pages"]
    assert pages["line_f1"] == pages["zone_f1"] == 96 and pages["cer_lines"] == pages["cer_end_to_end"] == 93
    notes = regression[regression["term"] == "complexity"].set_index("measure")["notes"].fillna("")
    assert "3 pages whose ground-truth texts don't fit" in notes["cer_lines"] and "3 pages" not in notes["line_f1"]
    for name in ("tables/difficulty_regression.tex", "figures/difficulty_against_complexity.tex",
                 "tables/difficulty_main_text_and_margins.tex"):  # fmt: skip
        assert "The CERs leave out 3 pages whose ground-truth texts don't fit their lines" in (folder / name).read_text(
            encoding="utf-8"
        )
    # Without them, the fits on every page are the same as on the remaining pages.
    kept = analysis.prepared(material.main[~flagged].drop(columns=["texts_misplaced"]))
    assert analysis.mixed_model(material.main, "cer_lines").coefficients["complexity"][0] == pytest.approx(
        analysis.mixed_model(
            kept.assign(complexity=material.main.loc[~flagged, "complexity"]), "cer_lines"
        ).coefficients["complexity"][0]
    )


def test_the_coverage_table_with_and_without_the_pages_that_may_leave_text_out(tmp_path: Path) -> None:
    folder = report(tmp_path, ["difficulty_ground_truth_coverage"], outside=10)
    table = pd.read_csv(folder / "tables" / "difficulty_ground_truth_coverage.tsv", sep="\t").set_index("measure")
    assert (table["pages"].loc[["zone_f1", "line_f1"]] == 96).all()
    assert (table["without_pages"].loc[["zone_f1", "line_f1"]] == 86).all()
    assert table.loc["line_f1", "low"] <= table.loc["line_f1", "estimate"] <= table.loc["line_f1", "high"]
    assert table.loc["line_f1", "estimate"] != table.loc["line_f1", "without_estimate"]
    caption = (folder / "tables" / "difficulty_ground_truth_coverage.tex").read_text(encoding="utf-8")
    # Each run's pages flagged, its lines outside and its precision with and without them.
    assert "CREMMA: 10 of 96 pages flagged" in caption and "CATMuS's test split: 2 of 30 pages flagged" in caption
    assert "line precision 0.900, 0.900 without" in caption and "at least 5 of Kraken's lines" in caption
    # With no page flagged, both columns are the same.
    other = report(tmp_path / "none", ["difficulty_ground_truth_coverage"])
    found = pd.read_csv(other / "tables" / "difficulty_ground_truth_coverage.tsv", sep="\t").set_index("measure")
    assert (found["estimate"] == found["without_estimate"]).all() and (found["pages"] == found["without_pages"]).all()


def test_each_ranker_alone_standardised_with_the_ensembles_rows_the_main_analysis(tmp_path: Path) -> None:
    from layout_visual_complexity.difficulty_reports import DifficultySettings, load

    folder = report(tmp_path, ["difficulty_by_ranker", "difficulty_correlations", "difficulty_regression"])
    runs = {"main": tmp_path / "main", "test": tmp_path / "test"}
    material = load(DifficultySettings(runs, "main", tmp_path / "manuscripts.tsv"))
    for ranker in ("geometry", "vit", "ensemble"):
        standard = analysis.with_ranker(material.main, ranker)["complexity"]
        assert standard.mean() == pytest.approx(0, abs=1e-12) and standard.std(ddof=1) == pytest.approx(1)
    by_ranker = pd.read_csv(folder / "tables" / "difficulty_by_ranker.tsv", sep="\t")
    assert set(by_ranker["ranker"]) == {"geometry", "vit", "ensemble"}
    ensemble = by_ranker[by_ranker["ranker"] == "ensemble"].set_index("measure")
    correlations = pd.read_csv(folder / "tables" / "difficulty_correlations.tsv", sep="\t").set_index("measure")
    regression = pd.read_csv(folder / "tables" / "difficulty_regression.tsv", sep="\t")
    regression = regression[regression["term"] == "complexity"].set_index("measure")
    for measure in analysis.MEASURES:
        for column in ("rho", "rho_low", "rho_high", "rho_within", "rho_within_low", "rho_within_high"):
            assert ensemble.loc[measure, column] == pytest.approx(correlations.loc[measure, column])
        for column in ("estimate", "low", "high"):
            assert ensemble.loc[measure, column] == pytest.approx(regression.loc[measure, column])
    again = report(tmp_path / "again", ["difficulty_by_ranker"])
    pd.testing.assert_frame_equal(pd.read_csv(again / "tables" / "difficulty_by_ranker.tsv", sep="\t"), by_ranker)


def test_the_quarters_pages_medians_recall_and_precision() -> None:
    pages = pd.DataFrame({
        "expected_elo": [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800],
        "line_f1": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        "lines_matched": [10, 10, 8, 8, 6, 6, 4, 4], "lines_missed": [0, 0, 2, 2, 4, 4, 6, 6],
        "lines_invented": [0, 10, 0, 8, 0, 6, 0, 4],
        "cer_end_to_end": [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, np.nan, 0.4], "cer_lines": 0.01,
        "zone_f1": 1.0, "baseline_distance": 0.1, "lines": [10, 11, 20, 21, 30, 31, 40, 41],
    })  # fmt: skip
    table = analysis.by_quarter(pages)
    assert table["pages"].tolist() == [2, 2, 2, 2] and table["quarter"].tolist() == [1, 2, 3, 4]
    assert table["elo_low"].tolist() == [1100, 1300, 1500, 1700] and table["elo_high"].tolist() == [
        1200,
        1400,
        1600,
        1800,
    ]
    assert table["line_f1"].tolist() == pytest.approx([0.95, 0.75, 0.55, 0.35])
    assert table["line_recall"].tolist() == pytest.approx([1.0, 0.8, 0.6, 0.4])
    assert table["line_precision"].tolist() == pytest.approx([0.75, 0.75, 0.75, 0.75])
    assert table["cer_end_to_end"].tolist() == pytest.approx([0.02, 0.06, 0.15, 0.4])
    assert table["cer_pages"].tolist() == [2, 2, 2, 1] and table["lines"].tolist() == [10.5, 20.5, 30.5, 40.5]


def test_the_trend_line_is_deterministic_and_fitted_on_the_log_scale_for_log_panels() -> None:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    from layout_visual_complexity.plots.difficulty import trend

    rng = np.random.default_rng(4)
    x = rng.normal(1300, 100, 200)
    y = np.exp(-4 + (x - 1300) / 100 + rng.normal(0, 0.5, 200))
    y[:5] = 0
    first, second = trend(x, y, log=True), trend(x, y, log=True)
    np.testing.assert_array_equal(first, second)
    expected = lowess(np.log(np.clip(y, 0.001, None)), x, frac=0.5, it=3)
    np.testing.assert_allclose(first[:, 1], np.exp(expected[:, 1]))
    np.testing.assert_allclose(trend(x, y)[:, 1], lowess(y, x, frac=0.5, it=3)[:, 1])
    assert np.all(np.diff(first[:, 0]) >= 0)


def test_the_figure_has_its_legend_and_the_score_label(tmp_path: Path) -> None:
    from layout_visual_complexity.plots.difficulty import Panel, draw
    from layout_visual_complexity.plots.manuscripts import SCORE_LABEL

    data = analysis.prepared(synthetic(manuscripts=10, pages=6, held_out=3))
    data["line_f1"] = data["cer_lines"] = data["cer_end_to_end"] = data["measure"].clip(0.001, 1)
    panels = [
        Panel(name, name, data, log=name.startswith("cer")) for name in ("cer_lines", "cer_end_to_end", "line_f1")
    ]
    figure, counts = draw(panels)
    labels = [text.get_text() for text in figure.legends[0].get_texts()]
    assert labels == ["Seen in training", "Held out", "Trend over all pages"]
    assert figure.get_supxlabel() == "P" + SCORE_LABEL[1:]
    assert counts["line_f1"] == 60


def test_the_difficulty_figure_has_three_panels_one_trend_over_all_pages_and_the_held_out_marked() -> None:
    from matplotlib.colors import to_hex

    from layout_visual_complexity.plots.difficulty import HELD_OUT, Panel, draw

    pages = synthetic(manuscripts=10, pages=6, seed=4, held_out=2)
    pages["held_out"] = (pages["split"] == "test").astype(int)
    pages["cer"] = np.exp(pages["measure"] - 4)
    pages["f1"] = 1 / (1 + np.exp(-pages["measure"]))
    panels = [Panel("cer", "CER, ground-truth lines", pages, log=True), Panel("cer", "CER, end to end", pages, True),
              Panel("f1", "Line F1", pages)]  # fmt: skip
    figure, counts = draw(panels)
    assert len(figure.axes) == 3 and counts == {"cer": 60, "f1": 60}
    for axis in figure.axes:
        assert len(axis.lines) == 1 and len(axis.lines[0].get_xdata()) == len(pages)
        held = [
            collection for collection in axis.collections if to_hex(collection.get_facecolor()[0]) == HELD_OUT.lower()
        ]
        assert len(held) == 1 and len(held[0].get_offsets()) == pages["held_out"].sum() == 12
    assert [axis.get_yscale() for axis in figure.axes] == ["log", "log", "linear"]
    assert [axis.get_title(loc="left") for axis in figure.axes] == ["(a)", "(b)", "(c)"]
