"""Tables and figures about a set of judgments: the dataset's categories, reliability by Elo gap with the split-half
reliability and the ceiling for SROCC, position bias, the consistency reference per fold, the comparison graph and the
score distributions."""

import io
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from layout_visual_complexity.download import Cache, HttpClient, cached_get
from layout_visual_complexity.evaluation import ProtocolSettings, consistency_reference, manuscript_folds
from layout_visual_complexity.evaluation.metrics import hits
from layout_visual_complexity.evaluation.protocol import fold_mean_intervals
from layout_visual_complexity.logs import logger
from layout_visual_complexity.outputs import LatexTable, escape, gap_latex, save_figure, write_table
from layout_visual_complexity.plots.comparison_graph import BRIDGE, plot_comparison_graph
from layout_visual_complexity.plots.position_bias import plot_position_bias_funnel
from layout_visual_complexity.plots.score_distributions import Thumbnail, plot_score_distributions, thumbnail_ranks
from layout_visual_complexity.ranking import graph, reliability, split_half
from layout_visual_complexity.ranking.dataset import Dataset
from layout_visual_complexity.ranking.scores import Scores

# The reliability table's last row, the split-half reliability and the ceiling for SROCC.
SPLIT_HALF = "split-half"


@dataclass
class Written:
    """What an output wrote, and the numbers worth keeping in the run's manifest."""

    paths: list[Path] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)


def integer(value: float) -> str:
    return f"{int(value):,}"


def decimal(value: float, digits: int = 3) -> str:
    return "--" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def in_words(value: float) -> str:
    """A whole number up to ten in words, others as figures."""
    return WORDS[int(value)] if float(value).is_integer() and 0 <= value <= 10 else f"{value:,g}"


def with_interval(value: float, low: float, high: float) -> str:
    if not math.isfinite(value):
        return "--"
    if not (math.isfinite(low) and math.isfinite(high)):
        return decimal(value)
    return f"{decimal(value)} [{decimal(low)}, {decimal(high)}]"


def dataset_categories(dataset: Dataset, labels: dict[str, str], folder: Path) -> Written:
    """Manuscripts and images per category, each category's share of the images, and images per manuscript."""
    manuscripts = dataset.manuscripts
    rows = []
    categories = sorted(manuscripts["category"].unique())
    groups = [(category, manuscripts[manuscripts["category"] == category]) for category in categories]
    for category, group in [*groups, ("total", manuscripts)]:
        pages = group["pages"]
        rows.append(
            {
                "category": category,
                "label": labels.get(category, category) if category != "total" else "Total",
                "manuscripts": len(group),
                "images": int(pages.sum()),
                "share": 100 * pages.sum() / manuscripts["pages"].sum(),
                "mean": float(pages.mean()),
                "median": float(pages.median()),
                "min": int(pages.min()),
                "max": int(pages.max()),
            }
        )
    table = pd.DataFrame(rows)
    latex_rows = []
    for row in table.itertuples(index=False):
        if row.category == "total":
            name = "Total"
        elif row.label != row.category:
            name = f"{row.category} --- {escape(row.label)}"
        else:
            name = escape(row.category)
        median = f"{row.median:g} [{row.min}--{row.max}]"
        cells = [name, integer(row.manuscripts), integer(row.images), f"{row.share:.1f}", f"{row.mean:.1f}", median]
        latex_rows.append(cells)
    latex = LatexTable(
        headers=["Category", "Manuscripts", "Images", r"Share (\%)", "Mean", "Median [range]"],
        rows=latex_rows,
        caption="The dataset by text category: the number of manuscripts and of images, each category's share of "
        "the images, and the mean and median number of images per manuscript (range in brackets).",
        label="tab:dataset_categories",
        align="lccccc",
        rules_before=[len(latex_rows) - 1],
    )
    paths = write_table(table, folder / "tables" / "dataset_categories", latex, float_format="%.4f")
    return Written(paths, {"categories": table.to_dict("records")})


def split_half_numbers(repeats: pd.DataFrame) -> dict[str, float]:
    """The reliability r (the halves' correlation with its Spearman–Brown correction) and the ceiling for SROCC, √r,
    each as its median and its 5th and 95th percentiles over the splits."""
    summary = split_half.summarise(repeats)
    numbers = {key: summary[f"spearman_brown_{key}"] for key in ("median", "q05", "q95")}
    # The square root is monotonic: its quantiles are the square roots of the reliability's.
    return {f"reliability_{key}": value for key, value in numbers.items()} | {
        f"ceiling_{key}": split_half.ceiling(value) for key, value in numbers.items()
    }


def reliability_by_elo_gap(
    dataset: Dataset,
    scores: Scores,
    edges: tuple,
    min_judgments: int,
    repeats: pd.DataFrame,
    settings: split_half.SplitHalfSettings,
    folder: Path,
) -> Written:
    """Table 2 of the article, with the split-half reliability and the ceiling for SROCC as its last row; and position
    bias, overall and per annotator."""
    elo = scores.table.set_index("image")["elo_mean"]
    table = reliability.reliability_by_gap(dataset.judgments, elo, edges)
    numbers = split_half_numbers(repeats)
    names = {"median": "", "q05": "_q05", "q95": "_q95"}
    split_row = {"gap": SPLIT_HALF} | {
        f"{measure}{names[key]}": numbers[f"{measure}_{key}"] for measure in ("reliability", "ceiling") for key in names
    }
    table = pd.concat([table, pd.DataFrame([split_row])], ignore_index=True)
    bins = table[table["gap"] != SPLIT_HALF]
    for column in ("possible", "pairs", "annotated", "rejudged"):
        table[column] = table[column].astype("Int64")
    latex_rows = [
        [
            gap_latex(row.gap),
            integer(row.possible),
            integer(row.pairs),
            integer(row.annotated),
            decimal(row.consistency),
            integer(row.rejudged),
            decimal(row.agreement),
        ]
        for row in bins.itertuples(index=False)
    ]
    reliability_text = with_interval(numbers["reliability_median"], numbers["reliability_q05"],
                                     numbers["reliability_q95"])  # fmt: skip
    ceiling_text = with_interval(numbers["ceiling_median"], numbers["ceiling_q05"], numbers["ceiling_q95"])
    latex_rows.append(
        ["Split-half", rf"\multicolumn{{6}}{{c}}{{$r$ = {reliability_text}; ceiling for $\rho$: $\sqrt{{r}}$ = "
                       rf"{ceiling_text}}}"]
    )  # fmt: skip
    balanced = (
        "about as many judgments each" if settings.balance == "judgments" else "the same number of annotators each"
    )
    latex = LatexTable(
        headers=[r"Score $\Delta$", "Possible pairs", "Pairs judged", "Judgments", "Consistency", "Pairs rejudged",
                 "Agreement"],
        rows=latex_rows,
        caption=r"Reliability of the judgments by the difference ($\Delta$) between the two images' final Elo "
        r"scores (means over the permutations). \emph{Possible pairs} are all theoretically possible pairs of the "
        f"{len(elo):,} images whose scores differ by $\\Delta$; \\emph{{pairs judged}}, those actually judged, once "
        r"or more; \emph{judgments}, how often they were judged, a pair judged twice counting twice. "
        r"\emph{Consistency} is the share of judgments that chose the image with the higher final score. "
        r"\emph{Pairs rejudged} are the pairs judged by at least two different annotators; \emph{agreement}, how "
        "often two of their judgments by different annotators chose the same image. \\emph{Split-half}: the "
        f"annotators split at random into two halves with {balanced}, {settings.repeats} times; Elo scores from each "
        f"half's judgments ({settings.permutations} permutations), and Spearman's correlation between the two halves' "
        r"scores over the images judged in both, raised by the Spearman--Brown correction, $2\rho/(1+\rho)$, to the "
        r"full set of annotators. This is the reliability $r$ of the scores: the correlation expected between two "
        "independent runs of the whole annotation (the median over the splits, and their 5th to 95th percentiles in "
        "brackets). A method's scores carry none of the annotation's noise and are compared with one run, not two, "
        r"so the most their $\rho$ with the Elo scores can reach is $\sqrt{r}$, the correlation of the observed "
        r"scores with the true ones: the ceiling for $\rho$.",
        label="tab:reliability_by_elo_gap",
        align="lcccccc",
        rules_before=[len(latex_rows) - 2, len(latex_rows) - 1],
    )  # fmt: skip
    paths = write_table(table, folder / "tables" / "reliability_by_elo_gap", latex)

    bias = reliability.position_bias(dataset.judgments)
    shown = bias[(bias["annotator"] == reliability.OVERALL) | (bias["judgments"] >= min_judgments)]
    bias_rows = [
        [
            "Overall" if row.annotator == reliability.OVERALL else escape(row.annotator),
            integer(row.judgments),
            decimal(row.left),
            f"[{decimal(row.low)}, {decimal(row.high)}]",
        ]
        for row in shown.itertuples(index=False)
    ]
    leaning = leaning_annotators(bias)
    bias_latex = LatexTable(
        headers=["Annotator", "Judgments", "Left chosen", r"95\% interval"],
        rows=bias_rows,
        caption="Position bias: the share of judgments that chose the image shown on the left, with its 95\\% "
        f"(Wilson) interval, overall and for the annotators with at least {min_judgments:,} judgments; the TSV lists "
        f"all {len(bias) - 1:,}. With no preference for either side, a share scatters around one half by chance "
        "alone, the less the more judgments it counts; an interval that leaves out one half marks a lean larger than "
        f"chance would give. {position_bias_findings(bias, leaning)}",
        label="tab:position_bias",
        align="lccc",
        rules_before=[1] if len(bias_rows) > 1 else [],
    )
    paths += write_table(bias, folder / "tables" / "position_bias", bias_latex)
    overall = bias.iloc[0]
    summary = {
        "reliability": table.to_dict("records"),
        "position_bias_overall": {"left": overall["left"], "low": overall["low"], "high": overall["high"]},
        "annotators_with_position_bias": int(leaning["outside"].sum()),
        "annotators_outside_by_chance": reliability.expected_outside(leaning["judgments"].to_numpy()),
    }
    return Written(paths, summary)


def leaning_annotators(bias: pd.DataFrame) -> pd.DataFrame:
    """The annotators' rows of a position-bias table, with `outside`: whether their share lies outside the bounds of
    chance, which is when their interval leaves out one half."""
    annotators = bias[bias["annotator"] != reliability.OVERALL].copy()
    annotators["outside"] = (annotators["low"] > 0.5) | (annotators["high"] < 0.5)
    return annotators


def position_bias_findings(bias: pd.DataFrame, leaning: pd.DataFrame) -> str:
    """The overall share, whether it is within chance, how many annotators lean beyond it against how many would by
    chance alone, and the two most prolific annotators' shares, in plain words."""
    overall = bias[bias["annotator"] == reliability.OVERALL].iloc[0]
    within = "within" if overall["low"] <= 0.5 <= overall["high"] else "outside"
    expected = reliability.expected_outside(leaning["judgments"].to_numpy())
    outside = in_words(int(leaning["outside"].sum()))
    text = (
        f"The overall share, {decimal(overall['left'])} [{decimal(overall['low'])}, {decimal(overall['high'])}], is "
        f"{within} chance. {outside.capitalize()} of the {len(leaning):,} annotators lean beyond chance, where about "
        f"{in_words(round(expected))} would by chance alone"
    )
    prolific = leaning.sort_values("judgments", ascending=False, kind="stable").head(2)
    if len(prolific) == 2 and (prolific["left"] - 0.5).prod() < 0:
        first, second = (row for row in prolific.itertuples(index=False))
        text += (
            "; the two most prolific lean in opposite directions: one chose the left image in "
            f"{100 * first.left:.1f}\\% of {first.judgments:,} judgments, the other in {100 * second.left:.1f}\\% of "
            f"{second.judgments:,}"
        )
    return text + "."


def position_bias_funnel(dataset: Dataset, folder: Path) -> Written:
    """Each annotator's share of left choices against their number of judgments, within the bounds of chance."""
    bias = reliability.position_bias(dataset.judgments)
    leaning = leaning_annotators(bias)
    overall = float(bias.iloc[0]["left"])
    figure = plot_position_bias_funnel(leaning, overall)
    caption = (
        "Position bias by annotator: each annotator's share of judgments that chose the image shown on the left, "
        "against their number of judgments (on a logarithmic scale). With no preference for either side, a share "
        "falls within the grey bounds by chance alone 95\\% of the time: one half $\\pm\\ 1.96\\sqrt{1/4n}$ for $n$ "
        "judgments, narrower as the judgments grow. An annotator outside them (in vermillion) leans to one side more "
        "than chance would give, which is when the 95\\% (Wilson) interval of their share leaves out one half. The "
        f"solid line is the overall share, the dashed one one half. {position_bias_findings(bias, leaning)}"
    )
    paths = save_figure(figure, folder / "figures" / "position_bias_funnel", caption, "fig:position_bias_funnel")
    summary = {
        "annotators": len(leaning),
        "outside": int(leaning["outside"].sum()),
        "outside_by_chance": reliability.expected_outside(leaning["judgments"].to_numpy()),
    }
    return Written(paths, summary)


def consistency_reference_by_fold(
    dataset: Dataset,
    scores: Scores,
    k: int,
    repeats: int,
    seed: int,
    edges: tuple,
    resamples: int,
    resample_seed: int,
    folder: Path,
) -> Written:
    """The consistency of each fold's test judgments with the final Elo scores, by gap: the reference for the
    accuracy of any method evaluated on the same folds, with the 95% interval of its mean over the folds from
    `resamples` resamples of the manuscripts, as the methods' evaluations draw them."""
    elo = scores.table.set_index("image")["elo_mean"]
    pages = dataset.images["image"].tolist()
    manuscripts = dataset.images.set_index("image")["manuscript"]
    judgments = dataset.judgments.reset_index(drop=True)
    folds = manuscript_folds(pages, manuscripts, judgments, k, repeats, seed)
    per_fold, table = consistency_reference(judgments, elo, folds, edges)
    agrees, gaps = hits(judgments, elo), reliability.judgment_gaps(judgments, elo)
    values = pd.concat(
        [pd.DataFrame({"fold_number": number, "row": fold.test, "gap": gaps[fold.test], "value": agrees[fold.test]})
         for number, fold in enumerate(folds)],
        ignore_index=True,
    )  # fmt: skip
    protocol = ProtocolSettings(gap_edges=edges, bootstrap=resamples, seed=resample_seed)
    intervals = fold_mean_intervals(values, judgments, manuscripts, edges, protocol).set_index("gap")
    table["low"] = table["gap"].map(intervals["low"])
    table["high"] = table["gap"].map(intervals["high"])
    latex_rows = [
        [
            gap_latex(row.gap),
            f"{row.judgments_per_fold:.1f}",
            with_interval(row.fold_mean, row.low, row.high),
            decimal(row.pooled),
        ]
        for row in table.itertuples(index=False)
    ]
    latex = LatexTable(
        headers=[r"Score $\Delta$", "Judgments per fold", "Consistency", "Pooled"],
        rows=latex_rows,
        caption=f"The consistency reference: over {k}-fold cross-validation repeated {repeats} times, with each "
        "manuscript's pages in one fold, the share of each fold's test judgments (both images in the fold) that "
        "agree with the final Elo scores. \\emph{Consistency} is the mean over the folds, with its 95\\% interval "
        f"from {resamples:,} resamples of the manuscripts, and \\emph{{Pooled}} the share over all folds' test "
        "judgments together.",
        label="tab:consistency_reference_by_fold",
        align="lccc",
        rules_before=[len(latex_rows) - 1],
    )
    paths = write_table(table, folder / "tables" / "consistency_reference_by_fold", latex)
    overall = table.iloc[-1]
    summary = {
        "folds": len(folds),
        "test_judgments": int(per_fold["test"].sum()),
        "fold_mean": overall["fold_mean"],
        "fold_sd": overall["fold_sd"],
        "low": overall["low"],
        "high": overall["high"],
        "pooled": overall["pooled"],
        "by_gap": table.to_dict("records"),
    }
    return Written(paths, summary)


def split_half_ceiling(repeats: pd.DataFrame, settings: split_half.SplitHalfSettings, folder: Path) -> Written:
    """The split-half correlation of the Elo scores; its Spearman–Brown correction, the scores' reliability; and the
    reliability's square root, the ceiling for SROCC. A TSV alone: the reliability table shows them in its last row."""
    summary = split_half.summarise(repeats)
    rows = []
    for column, name in [("spearman", "split-half correlation"), ("spearman_brown", split_half.RELIABILITY)]:
        rows.append(
            {
                "measure": name,
                "median": summary[f"{column}_median"],
                "q05": summary[f"{column}_q05"],
                "q95": summary[f"{column}_q95"],
            }
        )
    numbers = split_half_numbers(repeats)
    rows.append({"measure": split_half.CEILING} | {key: numbers[f"ceiling_{key}"] for key in ("median", "q05", "q95")})
    table = pd.DataFrame(rows)
    table["repeats"] = settings.repeats
    table["permutations"] = settings.permutations
    table["balance"] = settings.balance
    paths = write_table(table, folder / "tables" / "split_half_ceiling", None)
    return Written(paths, summary)


def comparison_graph(
    dataset: Dataset, scores: Scores, anchor_factor: float, batches: dict[str, str], seed: int, folder: Path
) -> Written:
    """Figure 2 of the article, coloured by batch and by score, and a table describing the graph, its data."""
    images = dataset.images["image"].tolist()
    elo = scores.table.set_index("image")["elo_mean"]
    described = graph.describe(dataset.judgments, images, anchor_factor)
    batch = graph.batches_by_first_judgment(dataset.judgments, images, batches)
    figure = plot_comparison_graph(dataset.judgments, images, elo, described.anchors, batch, seed)
    connected = (
        "The graph is connected."
        if described.components == 1
        else f"The graph has {described.components:,} connected components."
    )
    bridged = dataset.judgments["campaign"].eq(BRIDGE).sum() if "campaign" in dataset.judgments else 0
    batch_text = "by batch, each image in the one in which it was first judged, with" if len(batches) else "with"
    batch_text += (
        f" the pairs judged in the bridge sessions ({bridged:,} judgments, in which both batches were judged) in "
        "vermillion"
    )
    caption = (
        f"The comparison graph, in one layout: each node is one of the {described.images:,} images, sized by its "
        f"number of comparisons ({described.comparisons_min:,} to {described.comparisons_max:,}, median "
        f"{described.comparisons_median:g}), and each edge joins two images judged against each other: "
        f"{described.pairs:,} pairs in {described.judgments:,} judgments, a density of "
        f"{100 * described.density:.1f}\\% of all possible pairs. (a) Coloured {batch_text}; (b) coloured by final "
        f"Elo score. Ringed: the {len(described.anchors)} anchor images, compared more than {in_words(anchor_factor)} "
        f"times the median number of comparisons. {connected}"
    )
    paths = save_figure(figure, folder / "figures" / "comparison_graph", caption, "fig:comparison_graph")
    rows = [
        ("images", described.images),
        ("judgments", described.judgments),
        ("judged pairs", described.pairs),
        ("density", described.density),
        ("connected components", described.components),
        ("largest component", described.largest_component),
        ("comparisons per image, minimum", described.comparisons_min),
        ("comparisons per image, median", described.comparisons_median),
        ("comparisons per image, mean", described.comparisons_mean),
        ("comparisons per image, maximum", described.comparisons_max),
        ("anchors", len(described.anchors)),
    ]
    table = pd.DataFrame({"measure": [m for m, _ in rows], "value": pd.Series([v for _, v in rows], dtype=object)})
    paths += write_table(table, folder / "tables" / "comparison_graph", None)
    summary = {key: value for key, value in rows} | {
        "batches": {str(name): int(count) for name, count in batch.value_counts(sort=False).items()}
    }
    return Written(paths, summary)


def iiif_thumbnail_url(service: str, size: int) -> str:
    """A IIIF Image API request for the image fitted in a `size` × `size` square."""
    return f"{service.rstrip('/')}/full/!{size},{size}/0/default.jpg"


def thumbnail_choice(images: pd.DataFrame, ranked: list[str], count: int, pages: list[str]) -> list[int]:
    """The ranks of the pages to show: `pages` if given, else `count` evenly spaced ranks, a page without an image
    service giving way to the nearest rank that has one."""
    services = images.set_index("image")["image_service"]
    has_service = [isinstance(services.get(image), str) and bool(services.get(image)) for image in ranked]
    if pages:
        rank_of = {image: rank for rank, image in enumerate(ranked)}
        unknown = [page for page in pages if page not in rank_of]
        if unknown:
            raise ValueError(f"thumbnail pages not in the dataset: {', '.join(unknown)}")
        without = [page for page in pages if not has_service[rank_of[page]]]
        if without:
            raise ValueError(f"thumbnail pages without an image service: {', '.join(without)}")
        return sorted(rank_of[page] for page in dict.fromkeys(pages))
    chosen: list[int] = []
    for rank in thumbnail_ranks(len(ranked), count):
        candidates = sorted(range(len(ranked)), key=lambda other: (abs(other - rank), other))
        usable = next((other for other in candidates if has_service[other] and other not in chosen), None)
        if usable is not None:
            chosen.append(usable)
    return sorted(chosen)


def fetch_thumbnails(
    images: pd.DataFrame, ranked: list[str], ranks: list[int], size: int, client: HttpClient, cache: Cache
) -> list[Thumbnail]:
    """The pages at `ranks`, fetched from their image services at `size` pixels at most, labelled by their ids."""
    import matplotlib.image

    services = images.set_index("image")["image_service"]
    found = []
    for rank in ranks:
        data = cached_get(client, cache, iiif_thumbnail_url(services[ranked[rank]], size))
        found.append(Thumbnail(rank, matplotlib.image.imread(io.BytesIO(data), format="jpg"), ranked[rank]))
    return found


def score_figures(
    dataset: Dataset,
    scores: Scores,
    kinds: list[str],
    count: int,
    pages: list[str],
    size: int,
    credit: str,
    client: HttpClient,
    cache: Cache,
    folder: Path,
) -> dict[str, Written]:
    """Figure 3 of the article, each page's Elo scores over the permutations as violins (`elo_score_distributions`)
    or as bands (`elo_score_bands`), with the same thumbnails: `pages` if given, else `count` pages at evenly spaced
    ranks."""
    means = scores.distributions.mean(axis=0)
    ranked = [scores.table["image"].iloc[page] for page in np.argsort(means, kind="stable")]
    ranks = thumbnail_choice(dataset.images, ranked, count, pages) if (count or pages) else []
    thumbnails = fetch_thumbnails(dataset.images, ranked, ranks, size, client, cache)
    logger.info("score distributions: %d thumbnails", len(thumbnails))
    permutations = scores.distributions.shape[0]
    placed = "chosen near evenly spaced ranks" if pages else "at evenly spaced ranks"
    shown = (
        f" The {in_words(len(thumbnails))} pages shown, {placed}, are marked in vermillion and labelled with their ids."
        if thumbnails
        else ""
    )
    drawn = {
        "elo_score_distributions": f"one violin for each of the {len(ranked):,} pages, ordered by mean score from the "
        "lowest to the highest.",
        "elo_score_bands": f"for each of the {len(ranked):,} pages, ordered by mean score from the lowest to the "
        "highest, the median (line), the middle half (25th to 75th percentiles, dark band) and the middle 90\\% (5th "
        "to 95th percentiles, light band).",
    }
    written = {}
    for name in kinds:
        figure = plot_score_distributions(
            scores.distributions, thumbnails, "violins" if name == "elo_score_distributions" else "bands"
        )
        caption = f"Elo scores over {permutations:,} random permutations of the judgments' order: {drawn[name]}{shown}"
        if credit:
            caption += f" {escape(credit)}"
        paths = save_figure(figure, folder / "figures" / name, caption, f"fig:{name}")
        listed = [{"rank": thumbnail.rank + 1, "image": thumbnail.label} for thumbnail in thumbnails]
        written[name] = Written(paths, {"permutations": permutations, "thumbnails": listed})
    return written
