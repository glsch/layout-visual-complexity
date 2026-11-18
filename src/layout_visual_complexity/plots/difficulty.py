"""The difficulty figure: measures of automatic difficulty against predicted complexity, one dot per page."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from layout_visual_complexity.plots import style
from layout_visual_complexity.plots.manuscripts import SCORE_LABEL

SEEN, HELD_OUT, TREND = style.METHODS["trivial"], style.HIGHLIGHT, style.TEXT
# The share of the pages each local fit of the trend line takes: a third left the trends wiggling where pages are few,
# two thirds flattened the rise of the end-to-end CER.
FRAC = 1 / 2


@dataclass(frozen=True)
class Panel:
    """A panel: its measure, its label, the pages it shows, and whether its axis is logarithmic."""

    measure: str
    label: str
    pages: pd.DataFrame
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


def draw(panels: list[Panel], floor: float = 0.001, frac: float = FRAC) -> tuple[Figure, dict[str, int]]:
    """The panels in a row, complexity across: each page a dot, the held-out pages (`held_out`) vermillion and the
    others grey; one robust LOWESS trend over all the panel's pages. Returns the figure and each panel's pages."""
    figure = style.new_figure("full", 6.2)
    axes = figure.subplots(1, len(panels), sharex=True)
    counts: dict[str, int] = {}
    for axis, panel in zip(axes, panels, strict=True):
        pages = panel.pages.dropna(subset=[panel.measure, "expected_elo"])
        values = pages[panel.measure].astype(float)
        if panel.log:
            values = values.clip(lower=floor)
        held = pages["held_out"] == 1
        for chosen, colour, order in ((~held, SEEN, 1), (held, HELD_OUT, 2)):
            axis.scatter(pages.loc[chosen, "expected_elo"], values[chosen], s=6, color=colour, linewidths=0,
                         alpha=0.85, zorder=order)  # fmt: skip
        fitted = trend(pages["expected_elo"], values, panel.log, floor, frac)
        axis.plot(fitted[:, 0], fitted[:, 1], color=TREND, linewidth=1.2, zorder=3)
        counts[panel.measure] = len(pages)
        if panel.log:
            axis.set_yscale("log")
        axis.set_ylabel(panel.label)
        style.light_grid(axis)
    shown = pd.concat([panel.pages["expected_elo"] for panel in panels]).dropna()
    low, high = float(shown.min()), float(shown.max())
    axes[0].set_xlim(low - 0.03 * (high - low), high + 0.03 * (high - low))  # no tick label past the last panel
    figure.supxlabel(SCORE_LABEL[0].upper() + SCORE_LABEL[1:], fontsize=9)
    style.panel_letters(axes)

    def dot(colour: str) -> Line2D:
        return Line2D([], [], marker="o", linestyle="", color=colour, markersize=4)

    handles = [dot(SEEN), dot(HELD_OUT), Line2D([], [], color=TREND, linewidth=1.2)]
    labels = ["Seen in training", "Held out", "Trend over all pages"]
    figure.legend(handles=handles, labels=labels, loc="outside upper center", ncol=3, frameon=False,
                  handletextpad=0.3, columnspacing=1.2)  # fmt: skip
    return figure, counts
