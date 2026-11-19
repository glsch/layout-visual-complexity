"""The judgments' figures and tables: the comparison graph in two colourings, the score distributions as violins and
as bands with chosen pages, the position-bias funnel, the reliability table's split-half row, the consistency
reference's intervals and the tables' alignment."""

import io
import re
from pathlib import Path

import httpx2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import published, write_published
from layout_visual_complexity import judgment_reports
from layout_visual_complexity.evaluation import ProtocolSettings, evaluate, manuscript_folds
from layout_visual_complexity.evaluation.folds import FoldSettings
from layout_visual_complexity.evaluation.metrics import hits
from layout_visual_complexity.evaluation.protocol import fold_mean_intervals
from layout_visual_complexity.judgments import (
    ImportSettings,
    RankSettings,
    ReportSettings,
    run_import,
    run_rank,
    run_report,
)
from layout_visual_complexity.outputs import LatexTable
from layout_visual_complexity.plots.comparison_graph import graph_layout, plot_comparison_graph
from layout_visual_complexity.plots.score_distributions import Thumbnail, Trio, plot_score_distributions, score_bands
from layout_visual_complexity.plots.style import HIGHLIGHT
from layout_visual_complexity.ranking import graph
from layout_visual_complexity.ranking.dataset import read_dataset
from layout_visual_complexity.ranking.reliability import chance_bounds, expected_outside, position_bias
from layout_visual_complexity.ranking.scores import read_scores
from layout_visual_complexity.ranking.split_half import SplitHalfSettings, ceiling


@pytest.fixture
def ranked(tmp_path: Path) -> tuple[Path, Path]:
    """The synthetic dataset, imported and ranked, its judgments spread over three days."""
    dataset = published()
    dataset.judgments["date"] = np.repeat(["2025-03-01", "2026-04-14", "2026-06-02", "2026-06-25"], 40)
    source = write_published(tmp_path / "published", dataset)
    run_import(ImportSettings(str(source), tmp_path / "dataset"))
    run_rank(RankSettings(tmp_path / "dataset", tmp_path / "ranking", permutations=50, seed=3, jobs=1))
    return tmp_path / "dataset", tmp_path / "ranking"


class Thumbnails:
    """An image service answering every thumbnail request with a small page of a given width and height."""

    def __init__(self) -> None:
        self.requests: list[str] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(str(request.url))
        picture = io.BytesIO()
        Image.new("RGB", (30, 45), (240, 230, 200)).save(picture, "JPEG")
        return httpx2.Response(200, content=picture.getvalue(), headers={"content-type": "image/jpeg"})


def test_the_comparison_graph_has_two_panels_with_the_same_layout(ranked: tuple[Path, Path]) -> None:
    dataset = read_dataset(ranked[0])
    images = dataset.images["image"].tolist()
    scores = read_scores(ranked[1]).table.set_index("image")["elo_mean"]
    starts = {"Pilot": "2025-03-01", "First": "2026-04-01", "Second": "2026-06-01"}
    batches = graph.batches_by_first_judgment(dataset.judgments, images, starts)
    assert list(batches.cat.categories) == list(starts) and batches.notna().all()
    figure = plot_comparison_graph(dataset.judgments, images, scores, ("msa_1r",), batches, seed=5)
    panels = figure.axes[:2]  # then the colour bar's
    nodes = [
        [collection for collection in axes.collections if collection.get_offsets().shape[0] == len(images)][0]
        for axes in panels
    ]
    assert len(panels) == 2
    np.testing.assert_allclose(nodes[0].get_offsets(), nodes[1].get_offsets())
    _, position = graph_layout(dataset.judgments, images, seed=5)
    np.testing.assert_allclose(nodes[0].get_offsets(), position)
    # Lengthwise: the layout is wider than it is tall.
    assert np.ptp(position[:, 0]) >= np.ptp(position[:, 1])


def test_batches_are_the_last_begun_by_an_images_first_judgment() -> None:
    judgments = pd.DataFrame(
        {"left": ["a", "b", "c"], "right": ["b", "a", "a"], "date": ["2025-03-03", "2026-04-20", "2026-06-01"]}
    )
    starts = {"one": "2025-01-01", "two": "2026-05-28"}
    batches = graph.batches_by_first_judgment(judgments, ["a", "b", "c", "d"], starts)
    assert batches.tolist()[:3] == ["one", "one", "two"] and pd.isna(batches["d"])
    assert graph.batches_by_first_judgment(judgments, ["a"], {}).tolist() == ["Images"]


def test_the_score_figures_draw_and_mark_the_configured_pages(ranked: tuple[Path, Path], tmp_path: Path) -> None:
    server = Thumbnails()
    pages = ["msc_2r", "msa_1v", "msd_1r"]
    settings = ReportSettings(
        dataset=ranked[0], ranking=ranked[1], output=tmp_path / "results", outputs=["elo_score_distributions",
        "elo_score_bands"], thumbnail_pages=pages, cache=tmp_path / "cache", jobs=1,
    )  # fmt: skip
    summary = run_report(settings, httpx2.MockTransport(server))
    scores = read_scores(ranked[1])
    order = np.argsort(scores.distributions.mean(axis=0), kind="stable")
    ranked_pages = scores.table["image"].iloc[order].tolist()
    expected = sorted(pages, key=ranked_pages.index)
    for name in ("elo_score_distributions", "elo_score_bands"):
        assert [shown["image"] for shown in summary[name]["thumbnails"]] == expected
        assert (tmp_path / "results" / "figures" / f"{name}.png").is_file()
    assert len(server.requests) == len(pages)  # fetched once for both figures
    assert "chosen near evenly spaced ranks" in (tmp_path / "results" / "figures" / "elo_score_bands.tex").read_text()

    thumbnails = [Thumbnail(ranked_pages.index(page), np.ones((45, 30, 3)), page) for page in pages]
    for kind in ("violins", "bands"):
        figure = plot_score_distributions(scores.distributions, thumbnails, kind)  # type: ignore[arg-type]
        axes = figure.axes[0]
        marked = [
            collection.get_paths()[0].vertices[:, 0].mean()
            for collection in axes.collections
            if np.allclose(collection.get_facecolor()[0][:3], [int(HIGHLIGHT[i : i + 2], 16) / 255 for i in (1, 3, 5)])
        ]
        assert sorted(round(x) for x in marked) == sorted(ranked_pages.index(page) + 1 for page in pages)
        labels = [text.get_text() for text in figure.texts]
        assert set(pages) <= set(labels)
        assert all(text.get_fontsize() >= 6 for text in figure.texts if text.get_text() in pages)


def test_the_trios_draw_three_pages_at_each_place_the_one_in_front_over_the_two_behind() -> None:
    rng = np.random.default_rng(2)
    distributions = rng.normal(np.linspace(1000, 2000, 12), 40, size=(60, 12))
    ids = [f"page_{rank}" for rank in range(12)]
    pages = [[0, 1, 2], [5, 4, 6], [9, 8, 11]]  # ranks: front, left, right
    trios = [Trio(*(Thumbnail(rank, np.ones((45, 30, 3)), ids[rank]) for rank in trio)) for trio in pages]
    figure = plot_score_distributions(distributions, kind="bands", trios=trios)
    insets = figure.axes[1:]
    assert len(insets) == 3 * len(trios)
    for number, trio in enumerate(insets[i : i + 3] for i in range(0, len(insets), 3)):
        left, right, front = trio
        boxes = [axes.get_position() for axes in trio]
        assert front.get_zorder() > max(left.get_zorder(), right.get_zorder())
        assert boxes[0].x0 < boxes[2].x0 < boxes[1].x0 and boxes[2].y0 < min(boxes[0].y0, boxes[1].y0)
        assert all(boxes[2].overlaps(box) for box in boxes[:2])
        assert all(spine.get_edgecolor()[:3] == (1.0, 1.0, 1.0) for spine in front.spines.values())
        assert [axes.images[0].get_array().shape for axes in trio] == [(45, 30, 3)] * 3, number
    labels = {text.get_text(): text for text in figure.texts}
    assert {ids[rank] for trio in pages for rank in trio} <= set(labels)
    assert {labels[ids[rank]].get_fontsize() for trio in pages for rank in trio} == {4.5}
    marked = [
        collection
        for collection in figure.axes[0].collections
        if np.allclose(collection.get_facecolor()[0][:3], [int(HIGHLIGHT[i : i + 2], 16) / 255 for i in (1, 3, 5)])
    ]
    assert len(marked) == 9
    with pytest.raises(ValueError, match="not both"):
        plot_score_distributions(distributions, [trios[0].front], kind="bands", trios=trios)


def test_the_single_version_draws_as_before() -> None:
    """One thumbnail at each place, laid out by the single version's constants, whatever the trios' settings."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from layout_visual_complexity.plots import score_distributions as drawn
    from layout_visual_complexity.plots.style import CENTIMETRE, FULL_WIDTH

    rng = np.random.default_rng(3)
    distributions = rng.normal(np.linspace(1000, 2000, 12), 40, size=(60, 12))
    thumbnails = [Thumbnail(rank, np.ones((45, 30, 3)), f"page_{rank}") for rank in (0, 4, 8, 11)]
    figure = plot_score_distributions(distributions, thumbnails, "bands")
    again = plot_score_distributions(distributions, thumbnails, "bands", trios=[])
    rendered = []
    for each in (figure, again):
        canvas = FigureCanvasAgg(each)
        canvas.draw()
        rendered.append(np.asarray(canvas.buffer_rgba()).copy())
    assert np.array_equal(*rendered)
    width_cm = FULL_WIDTH / CENTIMETRE
    expected_width = min(drawn.THUMBNAIL_WIDTH, (width_cm - 0.1) / 2 - drawn.THUMBNAIL_GAP)
    expected_height = expected_width / (30 / 45)
    band = expected_height + drawn.LABEL + drawn.JOIN
    height_cm = 0.1 + band + drawn.PLOT + band + 0.55
    assert figure.get_size_inches() * 2.54 == pytest.approx([width_cm, height_cm])
    insets = figure.axes[1:]
    assert len(insets) == len(thumbnails)
    for axes in insets:
        box = axes.get_position()
        assert box.width * width_cm == pytest.approx(expected_height * 30 / 45)
        assert box.height * height_cm == pytest.approx(expected_height)
        assert not axes.axison
    assert {text.get_fontsize() for text in figure.texts if text.get_text().startswith("page_")} == {6}


def test_the_trios_caption_computes_its_span_and_credits_the_images(ranked: tuple[Path, Path], tmp_path: Path) -> None:
    server = Thumbnails()
    trios = [["msa_1r", "msb_1v", "msc_2r"], ["msd_2r", "msa_2r", "msb_1r"]]
    choice = "In front, the page typical of its score; behind it, two others."
    settings = ReportSettings(
        dataset=ranked[0], ranking=ranked[1], output=tmp_path / "results", outputs=["elo_score_bands_trios"],
        thumbnail_trios=trios, trio_choice=choice, image_credit="Page images by a library.",
        cache=tmp_path / "cache", jobs=1,
    )  # fmt: skip
    summary = run_report(settings, httpx2.MockTransport(server))
    scores = read_scores(ranked[1])
    means = pd.Series(scores.distributions.mean(axis=0), index=scores.table["image"])
    span = max(means[trio].max() - means[trio].min() for trio in trios)
    caption = (tmp_path / "results" / "figures" / "elo_score_bands_trios.tex").read_text()
    assert (
        "At each of the two places, chosen near evenly spaced ranks, three pages whose mean scores lie within "
        f"{span:.0f} points of each other are highlighted in vermillion and labelled with their ids." in caption
    )
    assert caption.count(choice) == 1 and caption.rstrip().endswith("\\end{figure}")
    assert "Similar scores can belong to quite different layouts. Page images by a library." in caption
    found = summary["elo_score_bands_trios"]
    assert found["span"] == pytest.approx(span)
    assert [[trio[position]["image"] for position in ("front", "left", "right")] for trio in found["trios"]] == sorted(
        trios, key=lambda trio: means[trio[0]]
    )
    assert len(server.requests) == 6
    assert "elo_score_bands_trios" not in ReportSettings(ranked[0], ranked[1], tmp_path).outputs


def test_a_trio_naming_a_page_not_in_the_dataset_is_refused(ranked: tuple[Path, Path], tmp_path: Path) -> None:
    wrong = [
        ([["msa_1r", "msb_1v", "msz_9r"]], "not in the dataset"),
        ([["msa_1r", "msb_1v"]], "three different pages"),
        ([], "needs --thumbnail_trios"),
    ]
    for trios, message in wrong:
        settings = ReportSettings(
            dataset=ranked[0], ranking=ranked[1], output=tmp_path / "results", outputs=["elo_score_bands_trios"],
            thumbnail_trios=trios, cache=tmp_path / "cache", jobs=1,
        )  # fmt: skip
        with pytest.raises(ValueError, match=message):
            run_report(settings, httpx2.MockTransport(Thumbnails()))


def test_unknown_thumbnail_pages_are_refused() -> None:
    images = pd.DataFrame({"image": ["a", "b"], "image_service": ["https://x/a", ""]})
    assert judgment_reports.thumbnail_choice(images, ["b", "a"], 2, ["a"]) == [1]
    with pytest.raises(ValueError, match="not in the dataset"):
        judgment_reports.thumbnail_choice(images, ["b", "a"], 2, ["c"])
    with pytest.raises(ValueError, match="without an image service"):
        judgment_reports.thumbnail_choice(images, ["b", "a"], 2, ["b"])


def test_bands_from_known_permutation_scores() -> None:
    # Two pages over 101 permutations: the first's scores 0..100, the second's constant.
    ordered = np.stack([np.arange(101.0), np.full(101, 7.0)], axis=1)
    bands = score_bands(ordered)
    assert bands["median"].tolist() == [50.0, 7.0]
    assert bands["q05"].tolist() == [5.0, 7.0] and bands["q95"].tolist() == [95.0, 7.0]
    assert bands["q25"].tolist() == [25.0, 7.0] and bands["q75"].tolist() == [75.0, 7.0]


def test_the_funnels_bounds_for_known_counts() -> None:
    low, high = chance_bounds(np.array([100, 400]))
    np.testing.assert_allclose(high - 0.5, [1.96 * 0.05, 1.96 * 0.025], rtol=1e-3)
    np.testing.assert_allclose(low, 1 - high)
    # A share outside the bounds is one whose Wilson interval leaves out one half: with 100 judgments the bounds are
    # 0.402 and 0.598, so 40 left choices lie outside and 41 inside.
    judgments = pd.DataFrame(
        {"left": ["a"] * 200, "right": ["b"] * 200, "annotator": ["x"] * 100 + ["y"] * 100,
         "chosen": ["a"] * 40 + ["b"] * 60 + ["a"] * 41 + ["b"] * 59}
    )  # fmt: skip
    leaning = judgment_reports.leaning_annotators(position_bias(judgments)).set_index("annotator")
    assert leaning["outside"].to_dict() == {"x": True, "y": False}
    # By chance alone: one judgment can never lie outside; four do when all go one way, 2 / 16 of the time.
    assert expected_outside(np.array([1])) == 0
    assert expected_outside(np.array([4, 4])) == pytest.approx(2 * 2 / 16)


def test_the_funnel_and_the_tables_say_what_outside_chance_means(ranked: tuple[Path, Path], tmp_path: Path) -> None:
    settings = ReportSettings(
        dataset=ranked[0], ranking=ranked[1], output=tmp_path / "results",
        outputs=["position_bias_funnel", "reliability_by_elo_gap"], jobs=1,
        split_half=SplitHalfSettings(repeats=3, permutations=5),
    )  # fmt: skip
    run_report(settings)
    text = (tmp_path / "results" / "tables" / "position_bias.tex").read_text()
    assert "within chance" in text and "by chance alone" in text and "leaves out one half" in text
    # The funnel's caption as its author words it: the overall share and one half given as values.
    overall = position_bias(read_dataset(ranked[0]).judgments).iloc[0]
    caption = (tmp_path / "results" / "figures" / "position_bias_funnel.tex").read_text()
    share = f"{overall['left']:.3f} [{overall['low']:.3f}, {overall['high']:.3f}]"
    assert f"The solid line is the overall share ({share}), the dashed one---one half (0.5)." in caption
    assert "By chance alone, i.e.\\ with no preference" in caption and "whereas chance would expect about" in caption
    assert (tmp_path / "results" / "figures" / "position_bias_funnel.svg").is_file()


def test_the_reliability_tables_last_row_is_the_split_half(ranked: tuple[Path, Path], tmp_path: Path) -> None:
    dataset = read_dataset(ranked[0])
    scores = read_scores(ranked[1])
    repeats = pd.DataFrame({"spearman": [0.5, 0.6, 0.7], "spearman_brown": [0.64, 0.81, 0.36],
                            "judgments_a": 1, "judgments_b": 1, "images": 12})  # fmt: skip
    settings = ReportSettings(dataset=ranked[0], ranking=ranked[1], output=tmp_path).split_half
    judgment_reports.reliability_by_elo_gap(dataset, scores, (100.0, 200.0), 100, repeats, settings, tmp_path)
    table = pd.read_csv(tmp_path / "tables" / "reliability_by_elo_gap.tsv", sep="\t")
    assert table["gap"].tolist()[-2:] == ["overall", "split-half"]
    last = table.iloc[-1]
    assert last["reliability"] == pytest.approx(0.64)
    assert last["reliability_q05"] == pytest.approx(np.quantile([0.64, 0.81, 0.36], 0.05))
    assert last["ceiling"] == pytest.approx(0.8)
    assert last["ceiling_q95"] == pytest.approx(ceiling(last["reliability_q95"]))
    overall = table.iloc[-2]
    judged = zip(dataset.judgments["left"], dataset.judgments["right"], strict=True)
    assert overall["pairs"] == len({tuple(sorted(pair)) for pair in judged})
    assert overall["pairs"] <= overall["annotated"] and table["pairs"].iloc[:-2].sum() == overall["pairs"]
    tex = (tmp_path / "tables" / "reliability_by_elo_gap.tex").read_text()
    assert r"Split-half & \multicolumn{6}{c}{$r$ = 0.640" in tex and r"$\sqrt{r}$ = 0.800" in tex
    assert "all theoretically possible pairs" in tex and "actually judged" in tex


def test_the_consistency_references_intervals_are_the_evaluations(ranked: tuple[Path, Path], tmp_path: Path) -> None:
    dataset = read_dataset(ranked[0])
    scores = read_scores(ranked[1])
    edges = (100.0, 200.0)
    written = judgment_reports.consistency_reference_by_fold(dataset, scores, 2, 3, 9, edges, 200, 11, tmp_path)
    table = pd.read_csv(tmp_path / "tables" / "consistency_reference_by_fold.tsv", sep="\t").set_index("gap")
    assert {"fold_sd", "low", "high"} <= set(table.columns)
    # As the evaluation of a method draws the reference's interval on the same folds and resamples.
    judgments = dataset.judgments.reset_index(drop=True)
    elo = scores.table.set_index("image")["elo_mean"]
    pages = dataset.images["image"].tolist()
    manuscripts = dataset.images.set_index("image")["manuscript"]
    folds = manuscript_folds(pages, manuscripts, judgments, 2, 3, 9)
    features = pd.DataFrame({"x": np.arange(len(pages), dtype=float)}, index=pages)
    protocol = ProtocolSettings(gap_edges=edges, bootstrap=200, seed=11)
    tested = evaluate(features, judgments, elo, folds, protocol).tested
    reference = tested.assign(value=hits(judgments, elo)[tested["row"].to_numpy()])
    expected = fold_mean_intervals(reference, judgments, manuscripts, edges, protocol).set_index("gap")
    np.testing.assert_allclose(table["low"], expected["low"].reindex(table.index), atol=1e-6)
    np.testing.assert_allclose(table["high"], expected["high"].reindex(table.index), atol=1e-6)
    assert written.summary["low"] <= written.summary["fold_mean"] <= written.summary["high"]
    tex = (tmp_path / "tables" / "consistency_reference_by_fold.tex").read_text()
    assert "SD" not in tex and "95\\% interval" in tex


def test_tables_are_centred_but_their_first_column(ranked: tuple[Path, Path], tmp_path: Path) -> None:
    assert LatexTable(["a", "b", "c"], [], "", "tab:x").render().count("{lcc}") == 1
    settings = ReportSettings(
        dataset=ranked[0], ranking=ranked[1], output=tmp_path / "results", thumbnails=0, jobs=1,
        folds=FoldSettings(k=2, repeats=1), split_half=SplitHalfSettings(repeats=3, permutations=5), bootstrap=20,
    )  # fmt: skip
    run_report(settings)
    tables = sorted((tmp_path / "results" / "tables").glob("*.tex"))
    assert {path.stem for path in tables} == {"dataset_categories", "reliability_by_elo_gap", "position_bias",
                                              "consistency_reference_by_fold"}  # fmt: skip
    for path in tables:
        align = re.search(r"\\begin\{tabular\}\{(.*)\}", path.read_text()).group(1)  # type: ignore[union-attr]
        assert re.fullmatch(r"l(c|p\{[^}]*\})+", align), (path.name, align)
    # The split-half table and the graph's statistics are TSVs alone.
    assert not (tmp_path / "results" / "tables" / "split_half_ceiling.tex").exists()
    assert (tmp_path / "results" / "tables" / "comparison_graph.tsv").is_file()
    # The comparison graph's caption: the possible pairs beside the density, the rings and connectedness before (a).
    caption = (tmp_path / "results" / "figures" / "comparison_graph.tex").read_text()
    images = len(read_dataset(ranked[0]).images)
    assert f"of the {images * (images - 1) // 2:,} possible pairs" in caption and "in one layout" not in caption
    assert caption.index("Ringed nodes") < caption.index("The graph is") < caption.index("(a)") < caption.index("(b)")
