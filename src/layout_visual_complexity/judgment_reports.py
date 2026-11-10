"""Tables and figures about a set of judgments: the dataset's categories, reliability by Elo gap, position bias,
the consistency reference per fold, the split-half ceiling, the comparison graph and the score distributions."""

import io
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from layout_visual_complexity.download import Cache, HttpClient, cached_get
from layout_visual_complexity.evaluation import consistency_reference, manuscript_folds
from layout_visual_complexity.logs import logger
from layout_visual_complexity.outputs import LatexTable, escape, gap_latex, save_figure, write_table
from layout_visual_complexity.plots.comparison_graph import plot_comparison_graph
from layout_visual_complexity.plots.score_distributions import plot_score_distributions, thumbnail_ranks
from layout_visual_complexity.ranking import graph, reliability, split_half
from layout_visual_complexity.ranking.dataset import Dataset
from layout_visual_complexity.ranking.elo import EloSettings
from layout_visual_complexity.ranking.scores import Scores


@dataclass
class Written:
    """What an output wrote, and the numbers worth keeping in the run's manifest."""

    paths: list[Path] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)


def integer(value: float) -> str:
    return f"{int(value):,}"


def decimal(value: float, digits: int = 3) -> str:
    return "--" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def dataset_categories(dataset: Dataset, labels: dict[str, str], folder: Path) -> Written:
    """Manuscripts and images per category, each category's share of the images, and images per manuscript."""
    manuscripts = dataset.manuscripts
    names = manuscripts.groupby("category")["category_name"].agg(lambda values: values.mode().iloc[0])
    rows = []
    groups = [(category, manuscripts[manuscripts["category"] == category]) for category in sorted(names.index)]
    for category, group in [*groups, ("total", manuscripts)]:
        pages = group["pages"]
        rows.append(
            {
                "category": category,
                "label": labels.get(category, names.get(category, "")) if category != "total" else "Total",
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
        name = "Total" if row.category == "total" else f"{row.category} --- {escape(row.label)}"
        median = f"{row.median:g} [{row.min}--{row.max}]"
        cells = [name, integer(row.manuscripts), integer(row.images), f"{row.share:.1f}", f"{row.mean:.1f}", median]
        latex_rows.append(cells)
    latex = LatexTable(
        headers=["Category", "Manuscripts", "Images", r"Share (\%)", "Mean", "Median [range]"],
        rows=latex_rows,
        caption="The dataset by text category: the number of manuscripts and of images, each category's share of "
        "the images, and the mean and median number of images per manuscript (range in brackets).",
        label="tab:dataset_categories",
        align="lrrrrr",
        rules_before=[len(latex_rows) - 1],
    )
    paths = write_table(table, folder / "tables" / "dataset_categories", latex, float_format="%.4f")
    return Written(paths, {"categories": table.to_dict("records")})


def reliability_by_elo_gap(dataset: Dataset, scores: Scores, edges: tuple, min_judgments: int, folder: Path) -> Written:
    """Table 2 of the article, and position bias, overall and per annotator."""
    elo = scores.table.set_index("image")["elo_mean"]
    table = reliability.reliability_by_gap(dataset.judgments, elo, edges)
    latex_rows = [
        [
            gap_latex(row.gap),
            integer(row.possible),
            integer(row.annotated),
            decimal(row.consistency),
            integer(row.rejudged),
            decimal(row.agreement),
        ]
        for row in table.itertuples(index=False)
    ]
    latex = LatexTable(
        headers=[
            r"Score $\Delta$",
            r"$N_{\mathrm{possible}}$",
            r"$N_{\mathrm{annotated}}$",
            "Consistency",
            r"$N_{\mathrm{rejudged}}$",
            "Agreement",
        ],
        rows=latex_rows,
        caption=r"Reliability of the judgments by the difference ($\Delta$) between the two images' final Elo "
        r"scores (means over the permutations). $N_{\mathrm{possible}}$ counts all pairs of the "
        f"{len(elo):,} images in each bin, and "
        r"$N_{\mathrm{annotated}}$ the judgments. \emph{Consistency} is the share of judgments that chose the "
        r"image with the higher final score; $N_{\mathrm{rejudged}}$ counts the image pairs judged by at least two "
        r"different annotators; \emph{Agreement} is the share of pairs of their judgments, by two different "
        "annotators, that chose the same image.",
        label="tab:reliability_by_elo_gap",
        align="crrrrr",
        rules_before=[len(latex_rows) - 1],
    )
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
    excluding_half = bias[(bias["annotator"] != reliability.OVERALL) & ((bias["low"] > 0.5) | (bias["high"] < 0.5))]
    bias_latex = LatexTable(
        headers=["Annotator", "Judgments", "Left chosen", r"95\% interval"],
        rows=bias_rows,
        caption="Position bias: the share of judgments that chose the image shown on the left, with its 95\\% "
        f"(Wilson) interval, overall and for the annotators with at least {min_judgments:,} judgments. The TSV "
        f"lists all {len(bias) - 1:,} annotators; the intervals of {len(excluding_half):,} of them exclude one half.",
        label="tab:position_bias",
        align="lrrr",
        rules_before=[1] if len(bias_rows) > 1 else [],
    )
    paths += write_table(bias, folder / "tables" / "position_bias", bias_latex)
    overall = bias.iloc[0]
    summary = {
        "reliability": table.to_dict("records"),
        "position_bias_overall": {"left": overall["left"], "low": overall["low"], "high": overall["high"]},
        "annotators_with_position_bias": len(excluding_half),
    }
    return Written(paths, summary)


def consistency_reference_by_fold(
    dataset: Dataset, scores: Scores, k: int, repeats: int, seed: int, edges: tuple, folder: Path
) -> Written:
    """The consistency of each fold's test judgments with the final Elo scores, by gap: the reference for the
    accuracy of any method evaluated on the same folds."""
    elo = scores.table.set_index("image")["elo_mean"]
    pages = dataset.images["image"].tolist()
    manuscripts = dataset.images.set_index("image")["manuscript"]
    folds = manuscript_folds(pages, manuscripts, dataset.judgments, k, repeats, seed)
    per_fold, table = consistency_reference(dataset.judgments, elo, folds, edges)
    latex_rows = [
        [
            gap_latex(row.gap),
            f"{row.judgments_per_fold:.1f}",
            decimal(row.fold_mean),
            decimal(row.fold_sd),
            decimal(row.pooled),
        ]
        for row in table.itertuples(index=False)
    ]
    latex = LatexTable(
        headers=[r"Score $\Delta$", "Judgments per fold", "Consistency", "SD", "Pooled"],
        rows=latex_rows,
        caption=f"The consistency reference: over {k}-fold cross-validation repeated {repeats} times, with each "
        "manuscript's pages in one fold, the share of each fold's test judgments (both images in the fold) that "
        "agree with the final Elo scores. \\emph{Consistency} is the mean over the folds, \\emph{SD} its standard "
        "deviation, and \\emph{Pooled} the share over all folds' test judgments together.",
        label="tab:consistency_reference_by_fold",
        align="crrrr",
        rules_before=[len(latex_rows) - 1],
    )
    paths = write_table(table, folder / "tables" / "consistency_reference_by_fold", latex)
    overall = table.iloc[-1]
    summary = {
        "folds": len(folds),
        "test_judgments": int(per_fold["test"].sum()),
        "fold_mean": overall["fold_mean"],
        "fold_sd": overall["fold_sd"],
        "pooled": overall["pooled"],
        "by_gap": table.to_dict("records"),
    }
    return Written(paths, summary)


def split_half_ceiling(
    dataset: Dataset, settings: split_half.SplitHalfSettings, elo: EloSettings, jobs: int, folder: Path
) -> Written:
    """The split-half correlation of the Elo scores, and its Spearman–Brown correction: the ceiling for SROCC."""
    images = dataset.images["image"].tolist()
    repeats = split_half.split_half_correlations(dataset.judgments, images, settings, elo, jobs)
    summary = split_half.summarise(repeats)
    rows = []
    for column, name in [("spearman", "split-half correlation"), ("spearman_brown", "Spearman-Brown corrected")]:
        rows.append(
            {
                "measure": name,
                "median": summary[f"{column}_median"],
                "q05": summary[f"{column}_q05"],
                "q95": summary[f"{column}_q95"],
            }
        )
    table = pd.DataFrame(rows)
    table["repeats"] = settings.repeats
    table["permutations"] = settings.permutations
    table["balance"] = settings.balance
    latex_rows = [
        [escape(row.measure.capitalize()), decimal(row.median), f"[{decimal(row.q05)}, {decimal(row.q95)}]"]
        for row in table.itertuples(index=False)
    ]
    balanced = (
        "about as many judgments in each half"
        if settings.balance == "judgments"
        else "the same number of annotators in each half"
    )
    latex = LatexTable(
        headers=["", r"Spearman's $\rho$", r"5--95\%"],
        rows=latex_rows,
        caption=f"The split-half ceiling: the annotators split at random into two halves ({balanced}), "
        f"{settings.repeats} times; Elo scores from each half's judgments ({settings.permutations} permutations); "
        "and Spearman's correlation between the halves' mean scores over the images judged in both, with its "
        "Spearman--Brown correction to the full set of annotators. The corrected median is the ceiling for a "
        "method's SROCC against the Elo scores.",
        label="tab:split_half_ceiling",
        align="lrr",
    )
    paths = write_table(table, folder / "tables" / "split_half_ceiling", latex)
    return Written(paths, summary)


def comparison_graph(dataset: Dataset, scores: Scores, anchor_factor: float, seed: int, folder: Path) -> Written:
    """Figure 2 of the article, and a table describing the graph."""
    images = dataset.images["image"].tolist()
    elo = scores.table.set_index("image")["elo_mean"]
    described = graph.describe(dataset.judgments, images, anchor_factor)
    figure = plot_comparison_graph(dataset.judgments, images, elo, described.anchors, seed)
    caption = (
        f"The comparison graph: each node is one of the {described.images:,} images, coloured by its final Elo "
        f"score and sized by its number of comparisons; each edge joins two images judged against each other "
        f"({described.pairs:,} pairs in {described.judgments:,} judgments; density {100 * described.density:.1f}\\%). "
        f"Outlined: the {len(described.anchors)} anchor images, compared more than {anchor_factor:g} times the "
        f"median number of comparisons. Connected components: {described.components}."
    )
    paths = save_figure(
        figure, folder / "figures" / "comparison_graph", caption, "fig:comparison_graph", r"0.7\textwidth"
    )
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
    latex_rows = []
    for measure, value in rows:
        text = (
            f"{100 * value:.1f}\\%"
            if measure == "density"
            else (f"{value:,.1f}" if isinstance(value, float) else f"{value:,}")
        )
        latex_rows.append([escape(measure.capitalize()), text])
    latex = LatexTable(
        headers=["", "Value"],
        rows=latex_rows,
        caption="The comparison graph: images are nodes, judged pairs edges. Density is the share of all image pairs "
        f"judged at least once; anchors are the images compared more than {anchor_factor:g} times the median.",
        label="tab:comparison_graph",
        align="lr",
    )
    paths += write_table(table, folder / "tables" / "comparison_graph", latex)
    summary = {key: value for key, value in rows}
    return Written(paths, summary)


def iiif_thumbnail_url(service: str, size: int) -> str:
    """A IIIF Image API request for the image fitted in a `size` × `size` square."""
    return f"{service.rstrip('/')}/full/!{size},{size}/0/default.jpg"


def fetch_thumbnails(
    images: pd.DataFrame, ranked: list[str], count: int, size: int, client: HttpClient, cache: Cache
) -> dict[int, np.ndarray]:
    """Thumbnails of pages at `count` evenly spaced ranks; a page without an image service gives way to the
    nearest rank that has one."""
    import matplotlib.image

    services = images.set_index("image")["image_service"]
    has_service = [isinstance(services.get(image), str) and bool(services.get(image)) for image in ranked]
    chosen: dict[int, np.ndarray] = {}
    for rank in thumbnail_ranks(len(ranked), count):
        candidates = sorted(range(len(ranked)), key=lambda other: (abs(other - rank), other))
        usable = next((other for other in candidates if has_service[other] and other not in chosen), None)
        if usable is None:
            continue
        data = cached_get(client, cache, iiif_thumbnail_url(services[ranked[usable]], size))
        chosen[usable] = matplotlib.image.imread(io.BytesIO(data), format="jpg")
    return chosen


def elo_score_distributions(
    dataset: Dataset,
    scores: Scores,
    count: int,
    size: int,
    credit: str,
    client: HttpClient,
    cache: Cache,
    folder: Path,
) -> Written:
    """Figure 3 of the article: each page's Elo scores over the permutations, with thumbnails at evenly spaced
    ranks."""
    means = scores.distributions.mean(axis=0)
    ranked = [scores.table["image"].iloc[page] for page in np.argsort(means, kind="stable")]
    thumbnails = fetch_thumbnails(dataset.images, ranked, count, size, client, cache) if count else {}
    logger.info("score distributions: %d thumbnails", len(thumbnails))
    figure = plot_score_distributions(scores.distributions, thumbnails)
    permutations = scores.distributions.shape[0]
    caption = (
        f"Elo scores over {permutations:,} random permutations of the judgments' order: one violin for each of the "
        f"{len(ranked):,} pages, ordered by mean score from the lowest to the highest. The pages shown are at evenly "
        "spaced ranks."
    )
    if credit:
        caption += f" {escape(credit)}"
    paths = save_figure(figure, folder / "figures" / "elo_score_distributions", caption, "fig:elo_score_distributions")
    shown = [{"rank": rank + 1, "image": ranked[rank]} for rank in sorted(thumbnails)]
    return Written(paths, {"permutations": permutations, "thumbnails": shown})
