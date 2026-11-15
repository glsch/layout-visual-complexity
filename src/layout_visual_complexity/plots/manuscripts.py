"""Manuscripts drawn through their pages' scores: a profile along the pages (in two designs), the strips of a whole
collection, and the overview of a collection's means against their spreads (in two designs).

Scores are coloured on the score scale (viridis) between two values shared by every figure of a collection, so the
figures can be read together. The most complex pages are marked in vermillion, the least complex in blue.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from layout_visual_complexity.plots import style

COMPLEX, PLAIN = style.HIGHLIGHT, style.NEGATIVE
SCORE_LABEL = "predicted score (Elo scale)"


@dataclass
class ShownPage:
    """A page shown as a thumbnail: its place in the manuscript (from 0), id, label, score and image."""

    place: int
    page: str
    label: str
    score: float
    image: np.ndarray | None
    complex: bool


def thousands(value: float) -> str:
    return f"{value:,.0f}"


def folio(label: str) -> str:
    """A canvas label as a folio is written: without leading zeros (`0043 ter r` → `43 ter r`)."""
    import re

    return re.sub(r"^0+(?=\d)", "", " ".join(str(label).split()))


def page_ticks(labels: Sequence[str], count: int = 8) -> tuple[list[int], list[str]]:
    """About `count` evenly spaced places along the pages, labelled with their folios."""
    n = len(labels)
    if n == 0:
        return [], []
    places = sorted({int(round(value)) for value in np.linspace(0, n - 1, min(count, n))})
    return [place + 1 for place in places], [folio(labels[place]) for place in places]


def draw_strip(axis, values: np.ndarray, norm: Normalize, length: int | None = None) -> None:
    """The pages as a strip of colours, page 1 at the left; `length` pages wide (the longest manuscript's)."""
    axis.imshow(values[np.newaxis, :], aspect="auto", cmap=style.score_colormap(), norm=norm,
                extent=(0.5, len(values) + 0.5, 0, 1), interpolation="nearest")  # fmt: skip
    axis.set_xlim(0.5, (length or len(values)) + 0.5)
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    for side in ("left", "right", "top", "bottom"):
        axis.spines[side].set_visible(False)


def draw_thumbnail(axis, shown: ShownPage, number: int | None, design: str) -> None:
    colour = COMPLEX if shown.complex else PLAIN
    if shown.image is not None:
        axis.imshow(shown.image)
    axis.set_xticks([])
    axis.set_yticks([])
    for side in axis.spines.values():
        side.set_visible(True)
        side.set_edgecolor(colour)
        side.set_linewidth(1.0)
    if design == "ours":
        axis.set_xlabel(f"{number}  {folio(shown.label)}\n{thousands(shown.score)}", color=colour, fontsize=7)
    else:
        axis.set_xlabel(f"{folio(shown.label)}\n{thousands(shown.score)}", color=colour, fontsize=7)


def plot_profile(
    values: np.ndarray, labels: Sequence[str], shown: list[ShownPage], norm: Normalize, window: int = 9
) -> Figure:
    """A manuscript's profile: its pages as a strip coloured by score, the scores with their rolling median, and the
    pages shown, numbered and marked on both."""
    from layout_visual_complexity.manuscripts import rolling_median

    figure = style.new_figure("full", 10.5)
    grid = GridSpec(3, 6, figure=figure, height_ratios=[0.45, 3.2, 3.6])
    strip = figure.add_subplot(grid[0, :])
    curve = figure.add_subplot(grid[1, :], sharex=strip)
    n = len(values)
    places = np.arange(1, n + 1)
    draw_strip(strip, values, norm)
    strip.tick_params(axis="x", labelbottom=False, length=0)
    curve.scatter(places, values, c=values, cmap=style.score_colormap(), norm=norm, s=3, linewidths=0, zorder=2)
    curve.plot(places, rolling_median(values, window), color=style.TEXT, linewidth=style.LINE, zorder=3)
    curve.set_ylabel(SCORE_LABEL)
    curve.set_xlabel("pages in the manuscript's order, by folio")
    ticks, names = page_ticks(labels)
    curve.set_xticks(ticks, names)
    style.light_grid(curve)
    near = 0.02 * n  # numbers of pages closer than this are set one above the other
    raised: list[tuple[float, int]] = []
    for number, page in sorted(enumerate(shown, start=1), key=lambda item: item[1].place):
        colour = COMPLEX if page.complex else PLAIN
        x = page.place + 1
        level = 1 + max((other for position, other in raised if abs(position - x) < near), default=-1)
        raised.append((x, level))
        strip.plot([x], [1.0], marker="v", color=colour, markersize=5, clip_on=False, zorder=4)
        strip.annotate(str(number), (x, 1.0), xytext=(0, 5 + 7 * level), textcoords="offset points", ha="center",
                       va="bottom", fontsize=6.5, color=colour, annotation_clip=False)  # fmt: skip
        curve.scatter([x], [page.score], s=28, facecolors="none", edgecolors=colour, linewidths=1.0, zorder=4)
    for index, page in enumerate(shown[:6]):
        draw_thumbnail(figure.add_subplot(grid[2, index]), page, index + 1, "ours")
    return figure


def plot_profile_strip(values: np.ndarray, shown: list[ShownPage], norm: Normalize) -> Figure:
    """A manuscript's profile as the article first drew it: the strip above the three most complex and the three
    least complex pages, labelled so, each with its folio and score, their places boxed on the strip."""
    figure = style.new_figure("full", 6.8)
    grid = GridSpec(2, 6, figure=figure, height_ratios=[0.8, 4.0])
    strip = figure.add_subplot(grid[0, :])
    draw_strip(strip, values, norm)
    strip.set_xticks([])
    for page in shown:
        colour = COMPLEX if page.complex else PLAIN
        strip.add_patch(Rectangle((page.place + 0.5 - 1.5, 0.02), 3.0 + 0.0, 0.96, fill=False, edgecolor=colour,
                                  linewidth=1.2, zorder=4))  # fmt: skip
    axes = [figure.add_subplot(grid[1, index]) for index in range(min(6, len(shown)))]
    for axis, page in zip(axes, shown[:6], strict=False):
        draw_thumbnail(axis, page, None, "article")
    if len(axes) >= 2:
        axes[1].set_title("complex", color=COMPLEX, fontsize=8)
    if len(axes) >= 5:
        axes[4].set_title("plain", color=PLAIN, fontsize=8)
    return figure


def plot_strips(rows: list[tuple[str, np.ndarray]], norm: Normalize) -> Figure:
    """A collection's manuscripts as strips, one row each, page 1 at the left, on one colour scale."""
    longest = max(len(values) for _, values in rows)
    height = 0.42 * len(rows) + 2.2
    figure = style.new_figure("full", height)
    grid = GridSpec(len(rows) + 1, 1, figure=figure, height_ratios=[1.0] * len(rows) + [0.9])
    axes = []
    for index, (label, values) in enumerate(rows):
        axis = figure.add_subplot(grid[index, 0])
        draw_strip(axis, values, norm, longest)
        axis.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7, labelpad=4)
        axis.tick_params(axis="x", labelbottom=False, length=0)
        axes.append(axis)
    axes[-1].tick_params(axis="x", labelbottom=True, length=2)
    axes[-1].set_xlabel("pages in the manuscript's order")
    bar = figure.add_subplot(grid[-1, 0])
    figure.colorbar(_mappable(norm), cax=bar, orientation="horizontal", label=SCORE_LABEL)
    bar.set_box_aspect(0.025)
    return figure


def _mappable(norm: Normalize):
    from matplotlib.cm import ScalarMappable

    return ScalarMappable(norm=norm, cmap=style.score_colormap())


def marker_sizes(values: pd.Series, smallest: float = 18.0, largest: float = 260.0) -> pd.Series:
    """Marker areas (points squared) in proportion to the values, the largest at `largest`."""
    top = float(values.max()) if values.notna().any() and values.max() > 0 else 1.0
    return (smallest + (largest - smallest) * values.fillna(0) / top).clip(lower=smallest)


def place_labels(figure: Figure, axis, xs: np.ndarray, ys: np.ndarray, texts: list[str], sizes: np.ndarray) -> None:
    """Each point's label beside it, where it overlaps no other label, point, legend or text already drawn, if there
    is such a place."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(figure)
    figure.draw_without_rendering()
    figure.set_layout_engine("none")  # the axes stay where the labels were placed against them
    renderer = canvas.get_renderer()
    centres = axis.transData.transform(np.column_stack([xs, ys]))
    radii = np.sqrt(sizes / np.pi) * figure.dpi / 72 + 1.5
    frame = axis.get_window_extent(renderer)
    drawn = [*axis.texts, *([axis.get_legend()] if axis.get_legend() is not None else [])]
    placed: list[tuple[float, float, float, float]] = [
        tuple(item.get_window_extent(renderer).extents) for item in drawn
    ]
    candidates = []
    for angle in np.radians([45, -45, 135, -135, 90, -90, 0, 180, 22.5, -22.5, 157.5, -157.5, 67.5, -67.5, 112.5,
                             -112.5]):  # fmt: skip
        dx, dy = float(np.cos(angle)), float(np.sin(angle))
        ha = "left" if dx > 0.3 else "right" if dx < -0.3 else "center"
        va = "bottom" if dy > 0.3 else "top" if dy < -0.3 else "center"
        candidates.append((dx * 1.4, dy * 1.4, ha, va))

    def overlap(box: tuple[float, float, float, float]) -> float:
        x0, y0, x1, y1 = box
        total = 0.0
        for a0, b0, a1, b1 in placed:
            total += max(0.0, min(x1, a1) - max(x0, a0)) * max(0.0, min(y1, b1) - max(y0, b0))
        for (cx, cy), radius in zip(centres, radii, strict=True):
            total += max(0.0, min(x1, cx + radius) - max(x0, cx - radius)) * max(
                0.0, min(y1, cy + radius) - max(y0, cy - radius)
            )
        outside = max(0.0, frame.x0 - x0) + max(0.0, x1 - frame.x1) + max(0.0, frame.y0 - y0) + max(0.0, y1 - frame.y1)
        return total + 1e3 * outside

    for index in np.argsort(-ys, kind="stable"):
        best = None
        for ring in range(1, 9):
            for dx, dy, ha, va in candidates:
                gap = radii[index] * 0.6 + 1.0 + 9.0 * (ring - 1)
                offset = (dx * gap * 72 / figure.dpi, dy * gap * 72 / figure.dpi)
                leader = {
                    "arrowprops": {
                        "arrowstyle": "-",
                        "color": style.SECONDARY,
                        "linewidth": 0.4,
                        "shrinkA": 0.0,
                        "shrinkB": float(radii[index] * 72 / figure.dpi),
                    }
                }
                text = axis.annotate(texts[index], (xs[index], ys[index]), xytext=offset, textcoords="offset points",
                                     ha=ha, va=va, fontsize=6.5, color=style.TEXT, zorder=5,
                                     **(leader if ring > 1 else {}))  # fmt: skip
                extent = text_extent(text, renderer)
                box = (extent.x0, extent.y0, extent.x1, extent.y1)
                mine = overlap(box) - _own(box, centres[index], radii[index]) + 0.5 * (ring - 1)
                if best is None or mine < best[0]:
                    if best is not None:
                        best[1].remove()
                    best = (mine, text, box)
                else:
                    text.remove()
            if best is not None and best[0] <= 0.5 * (ring - 1):
                break
        assert best is not None
        placed.append(best[2])


def text_extent(annotation, renderer):
    """An annotation's text alone, without its leader line."""
    from matplotlib.text import Text

    annotation.update_positions(renderer)
    return Text.get_window_extent(annotation, renderer)


def _own(box: tuple[float, float, float, float], centre: np.ndarray, radius: float) -> float:
    x0, y0, x1, y1 = box
    cx, cy = centre
    return max(0.0, min(x1, cx + radius) - max(x0, cx - radius)) * max(0.0, min(y1, cy + radius) - max(y0, cy - radius))


def plot_overview(table: pd.DataFrame, design: str = "ours") -> Figure:
    """Each manuscript's mean score against the interquartile range of its pages' scores, coloured by the number of
    hands, sized by folios, labelled; the medians dashed. `table` has `mean`, `iqr`, their intervals (`mean_low`, …),
    `hands`, `folios` and `label`. In our design the intervals are whiskers and nothing else is written inside; in
    the article's, the quadrants are named in the corners and the sizes' key is inside."""
    figure = style.new_figure("full", 11.5)
    axis = figure.add_subplot()
    sizes = marker_sizes(table["folios"])
    hands = table["hands"]
    known = bool(hands.notna().any())
    norm = Normalize(float(hands.min()), float(hands.max())) if known else Normalize(0, 1)
    colours = style.count_colormap() if known else _single(style.METHODS["ensemble"])
    axis.axvline(table["mean"].median(), color=style.SECONDARY, linewidth=0.6, linestyle="--", zorder=1)
    axis.axhline(table["iqr"].median(), color=style.SECONDARY, linewidth=0.6, linestyle="--", zorder=1)
    if design == "ours":
        axis.errorbar(table["mean"], table["iqr"],
                      xerr=[table["mean"] - table["mean_low"], table["mean_high"] - table["mean"]],
                      yerr=[table["iqr"] - table["iqr_low"], table["iqr_high"] - table["iqr"]],
                      fmt="none", ecolor=style.GRID_DARK, elinewidth=0.6, capsize=0, zorder=2)  # fmt: skip
    points = axis.scatter(table["mean"], table["iqr"], s=sizes, c=hands.fillna(0), cmap=colours, norm=norm,
                          edgecolors=style.TEXT, linewidths=0.5, alpha=0.9, zorder=3)  # fmt: skip
    axis.set_xlabel(f"mean {SCORE_LABEL}")
    axis.set_ylabel(f"interquartile range of the {SCORE_LABEL.replace('predicted score', 'predicted scores')}")
    style.light_grid(axis)
    if known:
        bar = figure.colorbar(points, ax=axis, label="hands", fraction=0.04, pad=0.02)
        bar.ax.yaxis.set_major_locator(_integer_locator())
    keys = _size_keys(table["folios"])
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=float(np.sqrt(size)), markerfacecolor="none",
                      markeredgecolor=style.TEXT, markeredgewidth=0.5, label=f"{folios:,} folios")
               for folios, size in keys]  # fmt: skip
    if design == "ours":
        figure.legend(handles=handles, loc="outside lower center", ncols=len(handles), handletextpad=0.3)
    else:
        axis.legend(handles=handles, loc="best", handletextpad=0.3, labelspacing=1.0)
        for x, y, ha, va, text, strong in (
            (0.015, 0.985, "left", "top", "plain but varied", False),
            (0.985, 0.985, "right", "top", "complex and varied", True),
            (0.015, 0.015, "left", "bottom", "plain and uniform", False),
            (0.985, 0.015, "right", "bottom", "complex but uniform", False),
        ):
            colour, weight = (style.TEXT, "bold") if strong else (style.SECONDARY, "normal")
            axis.text(x, y, text, transform=axis.transAxes, ha=ha, va=va, fontsize=7, style="italic", color=colour,
                      fontweight=weight)  # fmt: skip
    _pad(axis, table)
    place_labels(figure, axis, table["mean"].to_numpy(float), table["iqr"].to_numpy(float), list(table["label"]),
                 sizes.to_numpy(float))  # fmt: skip
    return figure


def _pad(axis, table: pd.DataFrame) -> None:
    """Room around the points for their labels."""
    for column, setter in (("mean", axis.set_xlim), ("iqr", axis.set_ylim)):
        low = min(table[column].min(), table.get(f"{column}_low", table[column]).min())
        high = max(table[column].max(), table.get(f"{column}_high", table[column]).max())
        span = high - low or 1.0
        setter(low - 0.08 * span, high + 0.12 * span)


def _size_keys(folios: pd.Series) -> list[tuple[int, float]]:
    """Three round folio counts spanning the manuscripts', with their marker areas."""
    top = float(folios.max()) if folios.notna().any() else 1.0
    step = 100 if top >= 300 else 50
    values = sorted({max(step, int(round(value / step) * step)) for value in np.linspace(folios.min(), top, 3)})
    sizes = marker_sizes(pd.Series([*values, top]))
    return [(value, float(sizes.iloc[index])) for index, value in enumerate(values)]


def _single(colour: str):
    from matplotlib.colors import ListedColormap

    return ListedColormap([colour])


def _integer_locator():
    from matplotlib.ticker import MaxNLocator

    return MaxNLocator(integer=True)
