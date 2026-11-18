"""The difficulty figure: four measures of automatic difficulty against predicted complexity, one dot per page."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from layout_visual_complexity.plots import style
from layout_visual_complexity.plots.manuscripts import SCORE_LABEL

SEEN, HELD_OUT = style.METHODS["trivial"], style.HIGHLIGHT
SEEN_TREND = "#5A5A5A"
LIGHT = 0.6
# The share of a group's pages each local fit of the trend line takes: a third left zone F1's trends wiggling between
# the few values it takes on pages with few zones, two thirds flattened the held-out pages' rise in the end-to-end CER.
FRAC = 1 / 2


@dataclass(frozen=True)
class Panel:
    """A panel: its measure and label, the pages it shows, and which of them its trend lines go over."""

    measure: str
    label: str
    pages: pd.DataFrame
    trend: pd.Series
    log: bool = False


def trend(x: np.ndarray, y: np.ndarray, log: bool = False, floor: float = 0.001, frac: float = FRAC) -> np.ndarray:
    """A robust LOWESS trend of y on x (statsmodels' `lowess`, three robustifying iterations), as (x, fitted) pairs
    in the order of x; on a log scale, fitted to the log of y clipped at `floor`, then transformed back."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    values = np.log(np.clip(y, floor, None)) if log else np.asarray(y, dtype=float)
    fitted = lowess(values, np.asarray(x, dtype=float), frac=frac, it=3, return_sorted=True)
    if log:
        fitted[:, 1] = np.exp(fitted[:, 1])
    return fitted


def draw(panels: list[Panel], floor: float = 0.001, frac: float = FRAC) -> tuple[Figure, dict[str, dict[str, int]]]:
    """The 2 × 2 grid: complexity across, shared within each column; each page a dot, seen grey and held out
    vermillion, the pages outside the trend lines' groups lighter; a robust LOWESS trend for each group. Returns the
    figure and each panel's pages by group of its trend lines."""
    figure = style.new_figure("full", 12.5)
    grid = figure.subplots(2, 2, sharex="col")
    counts: dict[str, dict[str, int]] = {}
    for axis, panel in zip(grid.flat, panels, strict=True):
        pages = panel.pages.dropna(subset=[panel.measure, "expected_elo"])
        values = pages[panel.measure].astype(float)
        if panel.log:
            values = values.clip(lower=floor)
        in_trend = panel.trend.reindex(pages.index).fillna(False).astype(bool)
        for held, colour in ((False, SEEN), (True, HELD_OUT)):
            group = (pages["held_out"] == 1) == held
            for counted, alpha_colour, size in ((False, style.lighter(colour, LIGHT), 5), (True, colour, 7)):
                chosen = group & (in_trend == counted)
                axis.scatter(
                    pages.loc[chosen, "expected_elo"],
                    values[chosen],
                    s=size,
                    color=alpha_colour,
                    linewidths=0,
                    alpha=0.85,
                    zorder=2 if held else 1,
                )
        counts[panel.measure] = {}
        for held, colour, name in ((False, SEEN_TREND, "seen"), (True, HELD_OUT, "held_out")):
            group = pages[in_trend & ((pages["held_out"] == 1) == held)]
            if len(group) >= 5:
                fitted = trend(group["expected_elo"], values[group.index], panel.log, floor, frac)
                axis.plot(fitted[:, 0], fitted[:, 1], color=colour, linewidth=1.2, zorder=3)
                counts[panel.measure][name] = len(group)
        if panel.log:
            axis.set_yscale("log")
        axis.set_ylabel(panel.label)
        style.light_grid(axis)
    for axis in grid[1]:
        axis.set_xlabel(SCORE_LABEL[0].upper() + SCORE_LABEL[1:])
    style.panel_letters(grid.flat)

    def dot(colour: str) -> Line2D:
        return Line2D([], [], marker="o", linestyle="", color=colour, markersize=4)

    handles = [
        dot(SEEN),
        dot(HELD_OUT),
        (dot(style.lighter(SEEN, LIGHT)), dot(style.lighter(HELD_OUT, LIGHT))),
        Line2D([], [], color=SEEN_TREND, linewidth=1.2),
        Line2D([], [], color=HELD_OUT, linewidth=1.2),
    ]
    labels = ["Seen", "Held out", "12th–13th century, not compared", "Trend, seen", "Trend, held out"]
    figure.legend(
        handles=handles,
        labels=labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
        loc="outside lower center",
        ncol=5,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.9,
    )
    return figure, counts
