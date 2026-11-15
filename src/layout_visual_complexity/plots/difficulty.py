"""The difficulty figure: four measures of automatic difficulty against predicted complexity, one dot per page."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from layout_visual_complexity.manuscripts import rolling_median
from layout_visual_complexity.plots import style

SEEN, HELD_OUT = style.METHODS["trivial"], style.HIGHLIGHT
LIGHT = 0.6


@dataclass(frozen=True)
class Panel:
    """A panel: its measure and label, the pages it shows, and which of them its running medians go over."""

    measure: str
    label: str
    pages: pd.DataFrame
    medians: pd.Series
    log: bool = False


def window(count: int) -> int:
    """The running median's window: about an eighth of a group's pages, odd, at least 5."""
    return max(5, round(count / 8)) | 1


def running(pages: pd.DataFrame, measure: str) -> tuple[np.ndarray, np.ndarray, int]:
    ordered = pages.dropna(subset=[measure, "expected_elo"]).sort_values("expected_elo", kind="stable")
    size = window(len(ordered))
    return ordered["expected_elo"].to_numpy(), rolling_median(ordered[measure].to_numpy(), size), size


def draw(panels: list[Panel], floor: float = 0.001) -> tuple[Figure, dict[str, dict[str, int]]]:
    """The 2 × 2 grid: complexity across, shared within each column; each page a dot, seen grey and held out
    vermillion, the pages outside the medians' groups lighter; a running median for each group. Returns the figure and
    each panel's windows by group."""
    figure = style.new_figure("full", 12.5)
    grid = figure.subplots(2, 2, sharex="col")
    windows: dict[str, dict[str, int]] = {}
    for axis, panel in zip(grid.flat, panels, strict=True):
        pages = panel.pages.dropna(subset=[panel.measure, "expected_elo"])
        values = pages[panel.measure].astype(float)
        if panel.log:
            values = values.clip(lower=floor)
        in_medians = panel.medians.reindex(pages.index).fillna(False).astype(bool)
        for held, colour in ((False, SEEN), (True, HELD_OUT)):
            group = (pages["held_out"] == 1) == held
            for counted, alpha_colour, size in ((False, style.lighter(colour, LIGHT), 5), (True, colour, 7)):
                chosen = group & (in_medians == counted)
                axis.scatter(
                    pages.loc[chosen, "expected_elo"],
                    values[chosen],
                    s=size,
                    color=alpha_colour,
                    linewidths=0,
                    alpha=0.85,
                    zorder=2 if held else 1,
                )
        windows[panel.measure] = {}
        for held, colour, name in ((False, SEEN, "seen"), (True, HELD_OUT, "held_out")):
            group = pages[in_medians & ((pages["held_out"] == 1) == held)]
            if len(group) >= 5:
                x, y, size = running(group.assign(**{panel.measure: values[group.index]}), panel.measure)
                darker = "#5A5A5A" if not held else colour
                axis.plot(x, y, color=darker, linewidth=1.2, zorder=3)
                windows[panel.measure][name] = size
        if panel.log:
            axis.set_yscale("log")
        axis.set_ylabel(panel.label)
        style.light_grid(axis)
    for axis in grid[1]:
        axis.set_xlabel("Predicted complexity (Elo scale)")
    style.panel_letters(grid.flat)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=SEEN, markersize=4, label="Seen"),
        Line2D([], [], marker="o", linestyle="", color=HELD_OUT, markersize=4, label="Held out"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=style.lighter(SEEN, LIGHT),
            markersize=4,
            label="12th–13th century",
        ),
        Line2D([], [], color="#5A5A5A", linewidth=1.2, label="Median, seen"),
        Line2D([], [], color=HELD_OUT, linewidth=1.2, label="Median, held out"),
    ]
    figure.legend(
        handles=handles, loc="outside lower center", ncol=5, frameon=False, handletextpad=0.4, columnspacing=1.2
    )
    return figure, windows
