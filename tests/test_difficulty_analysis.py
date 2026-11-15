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


def write_run(folder: Path, data: pd.DataFrame, transcribed: bool) -> Path:
    folder.mkdir(parents=True)
    rows = data.drop(columns=["z", "intercept", "noise", "century", "language", "split"]).copy()
    rng = np.random.default_rng(1)
    rows["status"] = "done"
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
    runs = {"main": write_run(tmp_path / "main", main, True), "test": write_run(tmp_path / "test", test, False)}
    facts = pd.concat([main, test]).groupby("document").first().reset_index()
    table = pd.DataFrame({"source": "S", "folder": facts["document"], "shelfmark": "Shelf " + facts["document"],
                          "language": facts["language"], "century": facts["century"],
                          "split": np.where(facts["document"].str.startswith("t"), "test", facts["split"]),
                          "comparison": np.where(facts["century"] >= 14, "yes", "no")})  # fmt: skip
    table.to_csv(tmp_path / "manuscripts.tsv", sep="\t", index=False)
    outputs = ["difficulty_against_complexity", "difficulty_regression", "difficulty_regression_across_and_within",
               "difficulty_held_out", "difficulty_correlations", "difficulty_main_text_and_margins",
               "difficulty_material", "difficulty_pages"]  # fmt: skip
    settings = ReportSettings(output=tmp_path / "results", outputs=outputs, name="difficulty", bootstrap=50,
                              difficulty_runs=runs, difficulty_main="main",
                              manuscript_table=tmp_path / "manuscripts.tsv")  # type: ignore[arg-type]  # fmt: skip
    run_report(settings)
    found = {path.relative_to(tmp_path / "results").as_posix() for path in (tmp_path / "results").rglob("*.*")}
    for stem in ("figures/difficulty_against_complexity", "tables/difficulty_regression",
                 "tables/difficulty_regression_across_and_within", "tables/difficulty_held_out",
                 "tables/difficulty_correlations", "tables/difficulty_main_text_and_margins",
                 "tables/difficulty_material"):  # fmt: skip
        assert f"{stem}.tex" in found
    assert {"figures/difficulty_against_complexity_log_cer.svg", "difficulty/main.tsv", "difficulty/test.tsv",
            "tables/difficulty_held_out_pages.tsv"} <= found  # fmt: skip
    regression = pd.read_csv(tmp_path / "results" / "tables" / "difficulty_regression.tsv", sep="\t")
    assert set(regression["measure"]) >= {"zone_f1", "line_f1", "baseline_distance", "cer_lines", "cer_end_to_end"}
    material = pd.read_csv(tmp_path / "results" / "tables" / "difficulty_material.tsv", sep="\t")
    assert material["pages"].sum() == len(main) + len(test)
