"""Figures of the baselines: what each level of information buys (each method's accuracy with its interval and its ρ,
grouped by the information it needs), and the generic image measures' correlations with each other and with ink
coverage.

Each method keeps its colour (`style.METHODS`); the Reference is black, chance a grey dashed line.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from layout_visual_complexity.plots import style


@dataclass
class Placed:
    """A method in the information figure: its level of information, its label, its colour, its accuracy with its
    interval, its ρ, and whether it is one measure alone (an open dot) or measures combined (a filled one)."""

    level: str
    label: str
    colour: str
    accuracy: float
    low: float
    high: float
    rho: float
    alone: bool = False


def signed(text: str) -> str:
    """A number's text with a minus sign rather than a hyphen."""
    return text.replace("-", "\u2212")


def plot_by_information(
    methods: list[Placed], reference: float, chance: str = "", ceiling: float | None = None
) -> Figure:
    """Each method's accuracy as a dot with its 95% interval, one row each, grouped under its level of information in
    the order given, its ρ to the right under a header giving the `ceiling` for ρ; the Reference for accuracy (the
    consistency of the test judgments) as a black line and chance as a dashed one, a "Chance" row under the level
    named `chance`. Full width."""
    rows: list[tuple[str, Placed | None]] = []
    for level in dict.fromkeys(method.level for method in methods):
        rows.append((level, None))
        if level == chance:
            rows.append(("Chance", Placed(level, "Chance", style.METHODS["chance"], 0.5, np.nan, np.nan, np.nan)))
        rows += [(method.label, method) for method in methods if method.level == level]
    figure = style.new_figure("full", 1.9 + 0.43 * len(rows))
    axes = figure.subplots()
    y = np.arange(len(rows))[::-1].astype(float)
    for position, (_, method) in zip(y, rows, strict=True):
        if method is None:
            continue
        if np.isfinite(method.low) and np.isfinite(method.high):
            axes.plot([method.low, method.high], [position, position], color=style.TEXT, linewidth=0.6, zorder=2)
        face = "white" if method.alone else method.colour
        axes.plot([method.accuracy], [position], "o", markersize=4.5, markerfacecolor=face,
                  markeredgecolor=method.colour, markeredgewidth=1.0, zorder=3)  # fmt: skip
        if np.isfinite(method.rho):
            axes.text(1.015, position, signed(f"{method.rho:.2f}"), transform=axes.get_yaxis_transform(),
                      va="center", ha="left", fontsize=8)  # fmt: skip
    axes.axvline(0.5, color=style.METHODS["chance"], linewidth=0.8, linestyle="--", zorder=1)
    axes.axvline(reference, color=style.METHODS["reference"], linewidth=1.0, zorder=1)
    header = r"$\rho$" if ceiling is None else rf"$\rho$, ceiling {ceiling:.2f}"
    axes.text(1.015, y[0], header, transform=axes.get_yaxis_transform(), va="center", ha="left", fontsize=9)
    axes.set_yticks(y, [label for label, _ in rows])
    for label, (_, method) in zip(axes.get_yticklabels(), rows, strict=True):
        if method is None:
            label.set_fontweight("bold")
    axes.tick_params(axis="y", length=0)
    axes.spines["left"].set_visible(False)
    axes.set_ylim(-0.7, len(rows) - 0.3)
    low = min([0.45, *(method.low - 0.01 for method in methods if np.isfinite(method.low))])
    axes.set_xlim(np.floor(low * 40) / 40, max(0.85, np.ceil((reference + 0.02) * 20) / 20))
    axes.set_xlabel("Pairwise accuracy on the test judgments")
    style.light_grid(axes, "x")
    handles = [
        Line2D([], [], color=style.METHODS["reference"], linewidth=1.0, label="Reference"),
        Line2D([], [], color=style.METHODS["chance"], linewidth=0.8, linestyle="--", label="Chance"),
    ]
    if any(method.alone for method in methods):
        handles += [
            Line2D([], [], marker="o", linestyle="none", markersize=4.5, markerfacecolor=style.SECONDARY,
                   markeredgecolor=style.SECONDARY, label="Measures combined"),
            Line2D([], [], marker="o", linestyle="none", markersize=4.5, markerfacecolor="white",
                   markeredgecolor=style.SECONDARY, label="One measure alone"),
        ]  # fmt: skip
    axes.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=len(handles))
    return figure


def plot_correlation_matrix(matrix: pd.DataFrame, labels: list[str]) -> Figure:
    """The lower triangle of a matrix of Spearman correlations, each cell coloured blue (negative) to vermillion
    (positive) with its value, the rows and columns labelled by `labels` in the matrix's order; the colour bar in the
    empty upper triangle, so the cells are wide enough for a signed value at 6 pt. Full width."""
    count = len(matrix)
    figure = style.new_figure("full", 5.6 + 0.58 * count)
    axes = figure.subplots()
    values = matrix.to_numpy(dtype=float)
    lower = np.tril(np.ones_like(values, dtype=bool), k=-1)
    shown = np.where(lower, values, np.nan)
    colours = style.correlation_colormap()
    norm = Normalize(-1, 1)
    image = axes.imshow(shown, cmap=colours, norm=norm, interpolation="nearest")
    for row in range(count):
        for column in range(row):
            value = values[row, column]
            if not np.isfinite(value):
                continue
            text = signed(f"{value:.2f}".replace("0.", ".", 1))
            ink = "white" if abs(value) >= 0.6 else style.TEXT
            axes.text(column, row, text, ha="center", va="center", fontsize=6, color=ink)
    axes.set_xticks(np.arange(count - 1), labels[:-1], rotation=90)
    axes.set_yticks(np.arange(1, count), labels[1:])
    for label in [*axes.get_xticklabels(), *axes.get_yticklabels()]:
        label.set_fontproperties(style.monospace(6.5))
    axes.set_xlim(-0.5, count - 1.5)
    axes.set_ylim(count - 0.5, 0.5)
    axes.tick_params(length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)
    bar = figure.colorbar(image, cax=axes.inset_axes((0.76, 0.55, 0.03, 0.4)))
    bar.set_label(r"Spearman's $\rho$")
    bar.outline.set_linewidth(0.5)
    return figure
