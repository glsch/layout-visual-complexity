"""The distributions of each page's Elo score over the permutations, ordered by mean, as violins or as bands, with
thumbnails of some pages above and below."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

# The layout, in centimetres: the thumbnails' widest, the gap between two in a row, the label under each, the space
# for the lines joining them to their pages, the plot's height, and its left margin for the axis's ticks and label.
THUMBNAIL_WIDTH = 2.7
THUMBNAIL_GAP = 0.25
LABEL = 0.32
JOIN = 0.45
PLOT = 5.2
LEFT = 1.6
LABEL_SIZE = 6  # points: the page ids under the thumbnails


@dataclass(frozen=True)
class Thumbnail:
    """A page shown above or below the plot: its rank by mean score (from 0), its image and its label."""

    rank: int
    image: np.ndarray
    label: str


def thumbnail_ranks(pages: int, count: int) -> list[int]:
    """`count` ranks evenly spaced from the lowest to the highest."""
    if count <= 0 or pages == 0:
        return []
    return sorted({int(rank) for rank in np.linspace(0, pages - 1, min(count, pages)).round()})


def score_bands(ordered: np.ndarray) -> dict[str, np.ndarray]:
    """Per page (column) of scores over the permutations (rows): the 5th, 25th, 50th, 75th and 95th percentiles."""
    names = ("q05", "q25", "median", "q75", "q95")
    return dict(zip(names, np.quantile(ordered, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0), strict=True))


def spread_in_row(centres: list[float], width: float, low: float, high: float) -> list[float]:
    """Centres as close to `centres` (ascending) as boxes `width` wide allow without overlapping, within
    [`low`, `high`]."""
    placed = [min(max(centre, low + width / 2), high - width / 2) for centre in centres]
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + width)
    if placed:
        placed[-1] = min(placed[-1], high - width / 2)
    for index in range(len(placed) - 2, -1, -1):
        placed[index] = min(placed[index], placed[index + 1] - width)
    return placed


def plot_score_distributions(
    distributions: np.ndarray,
    thumbnails: list[Thumbnail] | None = None,
    kind: Literal["violins", "bands"] = "violins",
):
    """Each page's scores (columns of `distributions`), ordered by mean from low to high: one violin each, or their
    median with the bands holding the middle half and the middle 90%. The pages of `thumbnails` are marked by their
    own violins in a second colour, joined to their thumbnails, which alternate above and below the plot, labelled
    with their page ids. Full width."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from layout_visual_complexity.plots import style

    thumbnails = sorted(thumbnails or [], key=lambda thumbnail: thumbnail.rank)
    means = distributions.mean(axis=0)
    order = np.argsort(means, kind="stable")
    ordered = distributions[:, order]
    pages = len(order)
    rows = [thumbnails[0::2], thumbnails[1::2]]
    per_row = max((len(row) for row in rows), default=0)
    width_cm = style.FULL_WIDTH / style.CENTIMETRE
    usable = width_cm - 0.1
    thumb_width = min(THUMBNAIL_WIDTH, usable / per_row - THUMBNAIL_GAP) if per_row else 0.0
    aspects = [thumbnail.image.shape[1] / thumbnail.image.shape[0] for thumbnail in thumbnails]
    thumb_height = thumb_width / min(aspects) if aspects else 0.0
    band = thumb_height + LABEL + JOIN if thumbnails else 0.0
    below = band if len(rows[1]) else 0.0
    axis_label = 0.55
    height_cm = 0.1 + band + PLOT + below + axis_label
    figure = style.new_figure("full", height_cm, constrained=False)
    bottom = axis_label + below
    axes = figure.add_axes((LEFT / width_cm, bottom / height_cm, (width_cm - LEFT - 0.1) / width_cm, PLOT / height_cm))

    shown = {thumbnail.rank for thumbnail in thumbnails}
    positions = np.arange(1, pages + 1)
    if kind == "violins":
        parts = axes.violinplot(list(ordered.T), positions=positions, widths=2.0, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(style.METHODS["vit"])
            body.set_edgecolor("none")
            body.set_alpha(0.55)
        handles, labels = [], []
    else:
        bands = score_bands(ordered)
        colour = style.METHODS["vit"]
        axes.fill_between(positions, bands["q05"], bands["q95"], color=style.lighter(colour, 0.75), linewidth=0,
                          zorder=1)  # fmt: skip
        axes.fill_between(positions, bands["q25"], bands["q75"], color=style.lighter(colour, 0.45), linewidth=0,
                          zorder=1)  # fmt: skip
        axes.plot(positions, bands["median"], color=colour, linewidth=0.8, zorder=2)
        handles = [
            Line2D([], [], color=colour, linewidth=0.8),
            Patch(facecolor=style.lighter(colour, 0.45), linewidth=0),
            Patch(facecolor=style.lighter(colour, 0.75), linewidth=0),
        ]
        labels = ["Median", "25–75%", "5–95%"]
    if shown:
        marked = axes.violinplot(
            [ordered[:, rank] for rank in sorted(shown)], positions=[rank + 1 for rank in sorted(shown)],
            widths=max(6.0, pages / 70), showextrema=False,
        )  # fmt: skip
        for body in marked["bodies"]:
            body.set_facecolor(style.HIGHLIGHT)
            body.set_edgecolor("none")
            body.set_alpha(0.9)
            body.set_zorder(3)
        handles.append(Patch(facecolor=style.HIGHLIGHT, linewidth=0))
        labels.append("Pages shown")
    if kind == "bands":
        axes.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.04, 1.0), handlelength=1.4)
    low, high = np.quantile(distributions, [0.0005, 0.9995])
    margin = 0.05 * (high - low)
    edge = max(4.0, pages / 100)
    axes.set_xlim(1 - edge, pages + edge)
    axes.set_ylim(low - margin, high + margin)
    axes.set_xticks([])
    axes.set_ylabel("Elo score")
    style.light_grid(axes)
    figure.text(
        (LEFT + (width_cm - LEFT - 0.1) / 2) / width_cm, 0.1 / height_cm, "Pages, ordered by mean score (low to high)",
        ha="center", va="bottom", fontsize=9,
    )  # fmt: skip

    def x_cm(rank: int) -> float:
        start, stop = axes.get_xlim()
        return LEFT + (rank + 1 - start) / (stop - start) * (width_cm - LEFT - 0.1)

    for above, row in zip((True, False), rows, strict=True):
        centres = spread_in_row([x_cm(t.rank) for t in row], thumb_width + THUMBNAIL_GAP, 0.05, width_cm - 0.05)
        for thumbnail, centre in zip(row, centres, strict=True):
            aspect = thumbnail.image.shape[1] / thumbnail.image.shape[0]
            image_height = min(thumb_height, thumb_width / aspect)
            image_width = image_height * aspect
            if above:
                image_bottom = height_cm - 0.1 - thumb_height
                label_top, join_from = image_bottom, image_bottom - LABEL
            else:
                image_bottom = bottom - JOIN - image_height
                label_top, join_from = image_bottom, bottom - JOIN
            inset = figure.add_axes(
                ((centre - image_width / 2) / width_cm, image_bottom / height_cm, image_width / width_cm,
                 image_height / height_cm)
            )  # fmt: skip
            inset.imshow(thumbnail.image, aspect="auto")
            inset.axis("off")
            figure.text(centre / width_cm, label_top / height_cm, thumbnail.label, ha="center", va="top",
                        fontproperties=style.monospace(LABEL_SIZE))  # fmt: skip
            scores = ordered[:, thumbnail.rank]
            end = np.quantile(scores, 0.99 if above else 0.01)  # where its violin still shows
            axes.annotate(
                "", xy=(thumbnail.rank + 1, end), xycoords="data",
                xytext=(centre / width_cm, join_from / height_cm), textcoords="figure fraction",
                arrowprops={"arrowstyle": "-", "color": style.SECONDARY, "lw": 0.5, "shrinkA": 1, "shrinkB": 1},
                annotation_clip=False,
            )  # fmt: skip
    return figure
