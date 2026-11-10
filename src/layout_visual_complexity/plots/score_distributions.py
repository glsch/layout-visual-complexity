"""The distributions of each page's Elo score over the permutations, as violins ordered by mean, with thumbnails."""

import numpy as np


def thumbnail_ranks(pages: int, count: int) -> list[int]:
    """`count` ranks evenly spaced from the lowest to the highest."""
    if count <= 0 or pages == 0:
        return []
    return sorted({int(rank) for rank in np.linspace(0, pages - 1, min(count, pages)).round()})


def plot_score_distributions(distributions: np.ndarray, thumbnails: dict[int, np.ndarray] | None = None):
    """Violins of each page's scores (columns of `distributions`), ordered by mean from low to high.

    `thumbnails` maps a rank (from 0, in that order) to an image, drawn above or below the plot by turns and
    linked to the page's mean.
    """
    from matplotlib.figure import Figure
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    means = distributions.mean(axis=0)
    order = np.argsort(means, kind="stable")
    data = [distributions[:, page] for page in order]
    pages = len(order)
    figure = Figure(figsize=(13.0, 6.0))
    axes = figure.subplots()
    figure.subplots_adjust(top=0.72, bottom=0.28, left=0.07, right=0.98)
    parts = axes.violinplot(data, positions=np.arange(1, pages + 1), widths=2.0, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#4a7fb5")
        body.set_edgecolor("none")
        body.set_alpha(0.55)
    transform = axes.get_xaxis_transform()
    for turn, (rank, image) in enumerate(sorted((thumbnails or {}).items())):
        above = turn % 2 == 0
        zoom = 0.33 * 200 / max(image.shape[:2])
        axes.add_artist(
            AnnotationBbox(
                OffsetImage(image, zoom=zoom),
                xy=(rank + 1, means[order[rank]]),
                xycoords="data",
                xybox=(rank + 1, 1.04 if above else -0.04),
                boxcoords=transform,
                box_alignment=(0.5, 0.0 if above else 1.0),
                frameon=False,
                pad=0,
                arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.7, "shrinkA": 1, "shrinkB": 2},
                zorder=5,
            )
        )
    low, high = np.quantile(distributions, [0.0005, 0.9995])
    margin = 0.05 * (high - low)
    axes.set_xlim(0, pages + 1)
    axes.set_ylim(low - margin, high + margin)
    axes.set_xticks([])
    axes.set_ylabel("Elo score")
    axes.set_xlabel("Pages, ordered by mean score (low to high)", labelpad=95 if thumbnails else 6)
    axes.grid(axis="y", color="0.9", linewidth=0.7)
    axes.spines[["top", "right"]].set_visible(False)
    return figure
