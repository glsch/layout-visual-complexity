"""Manuscripts through their pages' scores: profiles, statistics with intervals, the catalogue's dates, the colour of
pages, the readings, and the housekeeping commands."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from layout_visual_complexity import manuscripts as ms
from layout_visual_complexity.caches import CacheSettings, run_cache
from layout_visual_complexity.model_inventory import KNOWN, inventory, run_models
from layout_visual_complexity.score import ScoreSettings
from layout_visual_complexity.segmentation.regions import RegionSettings


def test_the_rolling_median_takes_nine_pages_and_fewer_at_the_ends() -> None:
    values = np.arange(20, dtype=float)
    values[10] = 100.0
    found = ms.rolling_median(values, 9)
    assert found[10] == 11.0  # one outlier among nine moves the median by one place, not by its size
    assert found[0] == np.median(values[:5]) and found[-1] == np.median(values[-5:])
    assert len(found) == len(values)


def test_the_most_and_least_complex_pages_come_in_the_manuscripts_order() -> None:
    scores = pd.Series([5.0, 9.0, 1.0, 9.0, 0.0, 7.0, 2.0], index=list("abcdefg"))
    most, least = ms.extremes(scores, 3)
    assert most == ["b", "d", "f"] and least == ["c", "e", "g"]
    few = ms.extremes(scores[:4], 3)
    assert few == (["a", "b", "d"], ["c"])


def collection(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for number, (name, spread) in enumerate((("plain", 20.0), ("varied", 150.0), ("middle", 60.0))):
        for page in range(80):
            rows.append({"page": f"{name}_{page}", "document": name, "order": page,
                         "expected_elo": 1400 + 100 * number + spread * rng.normal()})  # fmt: skip
    return pd.DataFrame(rows)


def test_each_manuscripts_mean_and_range_have_intervals_from_resampling_its_pages() -> None:
    scores = collection()
    table, drawn = ms.manuscript_statistics(scores, resamples=400, seed=3)
    assert list(table.index) == ["plain", "varied", "middle"]
    for name, row in table.iterrows():
        values = scores.loc[scores["document"] == name, "expected_elo"].to_numpy()
        assert row["mean"] == pytest.approx(values.mean())
        assert row["iqr"] == pytest.approx(ms.interquartile_range(values))
        assert row["mean_low"] < row["mean"] < row["mean_high"]
        assert row["iqr_low"] < row["iqr"] < row["iqr_high"]
        assert len(drawn[name].means) == 400
    assert table.at["varied", "iqr"] > table.at["middle", "iqr"] > table.at["plain", "iqr"]
    # A manuscript's draws don't depend on the others, or their order.
    reordered, _ = ms.manuscript_statistics(scores.iloc[::-1], resamples=400, seed=3)
    pd.testing.assert_frame_equal(table.loc[["plain"], ["mean_low", "iqr_high"]],
                                  reordered.loc[["plain"], ["mean_low", "iqr_high"]])  # fmt: skip
    places = ms.places(table, drawn, ["varied", "plain"])
    assert [(place["manuscript"], place["place"]) for place in places] == [("varied", 1), ("plain", 3)]
    assert places[0]["low"] == places[0]["high"] == 1


@pytest.mark.parametrize(
    ("text", "short"),
    [
        ("XV2 (parts dated 1463, 1492)", "1463–1492"),
        ("XV in.", "XV in."),
        ("1463", "1463"),
        ("1427–1428", "1427–1428"),
        ("XIV ex. (1377?)", "1377"),
        ("1502 + XV med. (a part dated 1436) + XV ex.", "1436–1502"),
        ("XV in. + XV2", "XV in. + XV2"),
        ("XIV/XV", "XIV/XV"),
    ],
)
def test_dates_are_shortened_to_the_years_the_catalogue_names_else_its_centuries(text: str, short: str) -> None:
    assert ms.short_date(text) == short


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("XV in. + XV2", "early 15th century and second half of the 15th century"),
        (
            "1502 + XV med. (a part dated 1436) + XV ex. (parts dated 1477, 1478, 1479, 1484) + XV in.",
            "1502, mid-15th century (a part dated 1436), late 15th century (parts dated 1477, 1478, 1479 and 1484) "
            "and early 15th century",
        ),
        ("XIV/XV", "turn of the 14th and 15th centuries"),
        ("XIV. ex.", "late 14th century"),
        ("XV1 + XIII", "first half of the 15th century and 13th century"),
        ("XV2 (parts dated 1468, 1470),", "second half of the 15th century (parts dated 1468 and 1470)"),
        ("1427–1428", "1427–1428"),
    ],
)
def test_dates_are_written_out_in_words_in_the_catalogues_order(text: str, words: str) -> None:
    assert ms.date_in_words(text) == words


def test_a_profile_opens_with_the_manuscript_as_catalogued_its_title_where_it_has_one() -> None:
    from layout_visual_complexity.manuscript_reports import described

    catalogue = pd.DataFrame(
        {"shelfmark": ["I D 18", "XI C 1"], "date": ["XV in. + XV2", "1502"], "folios": [310, 444],
         "material": ["paper", ""], "title": ["[Novum testamentum]", ""]},
        index=["id18", "xic1"],
    )  # fmt: skip

    class Catalogued:
        def __init__(self) -> None:
            self.catalogue = catalogue

        def label(self, manuscript: str) -> str:
            return str(catalogue.at[manuscript, "shelfmark"])

    collection = Catalogued()
    assert described(collection, "id18", "Prague, National Library of the Czech Republic") == (  # type: ignore[arg-type]
        "Prague, National Library of the Czech Republic, I D 18, paper, 310 leaves, early 15th century and second half "
        "of the 15th century: [Novum testamentum]"
    )
    assert described(collection, "xic1") == "XI C 1, 444 leaves, 1502"  # type: ignore[arg-type]


def test_the_overview_names_the_ends_of_its_axes_outside_the_plot() -> None:
    from layout_visual_complexity.plots.manuscripts import plot_overview

    table = pd.DataFrame(
        {"mean": [1400.0, 1500, 1600], "iqr": [80.0, 120, 160], "hands": [2.0, 5, 9], "folios": [150.0, 300, 450],
         "label": ["A (2)", "B (5)", "C (9)"]},
    )  # fmt: skip
    for column in ("mean", "iqr"):
        table[f"{column}_low"], table[f"{column}_high"] = table[column] - 20, table[column] + 20
    figure = plot_overview(table)
    axis = figure.axes[0]
    ends = {text.get_text(): text for text in axis.texts if "→" in text.get_text() or "←" in text.get_text()}
    assert set(ends) == {"← plainer", "more complex →", "← more uniform", "more varied →"}
    renderer = figure.canvas.get_renderer()
    frame = axis.get_window_extent(renderer)
    for text in ends.values():
        box = text.get_window_extent(renderer)
        assert box.y1 <= frame.y0 or box.x1 <= frame.x0  # below or left of the plotting area


def test_a_page_has_colour_with_a_rubric_line_or_a_coloured_initial() -> None:
    features = pd.DataFrame(
        {"Colour__Lines__rubric_share": [0.0, 0.2, 0.0, np.nan], "Colour__ColouredInitials__count": [0, 0, 2, np.nan]},
        index=list("abcd"),
    )
    found = ms.colour_pages(features)
    assert found.tolist()[:3] == [0.0, 1.0, 1.0] and np.isnan(found["d"])
    assert ms.colour_pages(pd.DataFrame(index=["a"])).isna().all()


def test_the_readings_measure_hands_spread_composites_and_colour(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    names = [f"m{i}" for i in range(12)]
    rows = []
    for number, name in enumerate(names):
        for page in range(40):
            rows.append({"page": f"{name}_{page}", "document": name, "order": page,
                         "expected_elo": 1500 + (10 + 15 * number) * rng.normal()})  # fmt: skip
    table, drawn = ms.manuscript_statistics(pd.DataFrame(rows), resamples=200, seed=1)
    path = tmp_path / "catalogue.tsv"
    catalogue = pd.DataFrame({"manuscript": names, "shelfmark": names, "hands": range(1, 13),
                              "homogeneous": [1] * 6 + [0] * 5 + ["?"], "colours": range(12)})  # fmt: skip
    path.write_text("# a made-up catalogue\n" + catalogue.to_csv(sep="\t", index=False))
    facts = ms.read_catalogue(path)
    colour = pd.Series(np.linspace(0, 1, 12), index=names)
    found = ms.readings(table, drawn, facts, colour, ["m0", "m11"], 200, 1).set_index("reading")
    hands_range = found.loc["Spearman's rho, hands and interquartile range"]
    assert hands_range["value"] > 0.8 and hands_range["low"] <= hands_range["value"] <= hands_range["high"]
    composite = found.loc["Interquartile range, composite less homogeneous (mean)"]
    assert composite["value"] > 0 and "m11" in composite["note"]
    assert found.loc["Spearman's rho, pages with colour and the catalogue's colours", "value"] == pytest.approx(1.0)
    assert found.loc["Place by interquartile range, m11 (1 the most varied)", "value"] == 1


def test_lvc_cache_lists_its_parts_and_clears_them_by_kind(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    for kind, size in (("images", 3000), ("tokens", 500), ("manifests", 10)):
        (tmp_path / kind).mkdir()
        (tmp_path / kind / "file.bin").write_bytes(b"x" * size)
    found = {name: total for name, _, _, total, _ in run_cache(CacheSettings(tmp_path))}
    assert found == {"images": 3000, "manifests": 10, "tokens": 500}
    found = {name for name, *_ in run_cache(CacheSettings(tmp_path, ["tokens", "images"]))}
    assert found == {"manifests"} and not (tmp_path / "tokens").exists()
    assert "cleared tokens" in capsys.readouterr().out
    with pytest.raises(ValueError, match="no such kind"):
        run_cache(CacheSettings(tmp_path, ["nothing"]))
    run_cache(CacheSettings(tmp_path, ["all"]))
    assert not any(tmp_path.iterdir())


def test_lvc_models_finds_the_shipped_rankers_and_a_regions_model_that_differs(tmp_path: Path) -> None:
    other = tmp_path / "medieval-yolov11x.pt"
    other.write_bytes(b"not the model")
    items = {item.what: item for item in inventory(ScoreSettings(regions=RegionSettings(model=other)))}
    regions = items["regions model (YOLO)"]
    assert regions.present and regions.matches is False and regions.sha256 != KNOWN[other.name]
    assert items["ranker geometry"].matches and items["ranker vit"].matches
    with pytest.raises(RuntimeError, match="missing or differ"):
        run_models(ScoreSettings(regions=RegionSettings(model=tmp_path / "missing.pt")))
    missing = inventory(ScoreSettings(regions=RegionSettings(model=tmp_path / "missing.pt")))[0]
    assert not missing.present and "models/" in missing.note
