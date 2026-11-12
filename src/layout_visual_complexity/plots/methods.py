"""Figures of the methods: their accuracy by Elo gap beside the consistency reference, their learning curves and
calibration, and the local ViT head's maps of complexity over pages.

Each method keeps its colour (`style.METHODS`); the consistency reference is black, chance a grey dashed line.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from layout_visual_complexity.plots import style


@dataclass
class Shown:
    """A method as a figure shows it: its label, its colour, and its table (by gap, a learning curve, a calibration)."""

    label: str
    colour: str
    table: pd.DataFrame


def gap_text(label: str) -> str:
    """A gap bin's label for a figure: `<100`, `100–200`, `≥400`, `Overall`."""
    if label == "overall":
        return "Overall"
    return label.replace(">=", "≥").replace("-", "–")


def plot_accuracy_by_gap(methods: list[Shown], consistency: pd.DataFrame, overall: bool = False) -> Figure:
    """Each method's accuracy by Elo gap (`gap`, `accuracy`, `low`, `high` in its table) as bars from 0.4 with their
    95% intervals, the consistency reference (`gap`, `consistency`) as diamonds, and chance. Full width."""
    gaps = [gap for gap in consistency["gap"] if overall or gap != "overall"]
    figure = style.new_figure("full", 7.5)
    axes = figure.subplots()
    width = 0.8 / max(len(methods), 1)
    bottom = 0.4
    x = np.arange(len(gaps))
    for number, method in enumerate(methods):
        table = method.table.set_index("gap").reindex(gaps)
        place = x - 0.4 + width * (number + 0.5)
        values = table["accuracy"].to_numpy(dtype=float)
        axes.bar(place, values - bottom, width * 0.9, bottom=bottom, color=method.colour, label=method.label)
        low, high = table["low"].to_numpy(dtype=float), table["high"].to_numpy(dtype=float)
        axes.vlines(place, low, high, color=style.TEXT, linewidth=0.6)
    reference = consistency.set_index("gap").reindex(gaps)["consistency"].to_numpy(dtype=float)
    axes.plot(x, reference, "D", markersize=4.5, markerfacecolor="white", markeredgecolor=style.METHODS["consistency"],
              markeredgewidth=0.9, linestyle="none", zorder=4)  # fmt: skip
    axes.axhline(0.5, color=style.METHODS["chance"], linewidth=0.8, linestyle="--", zorder=0)
    axes.set_xticks(x, [gap_text(gap) for gap in gaps])
    axes.set_xlim(-0.6, len(gaps) - 0.4)
    axes.set_ylim(bottom, 1.0)
    axes.set_xlabel(r"Score $\Delta$ (difference between the two pages' final Elo scores)")
    axes.set_ylabel("Pairwise accuracy")
    style.light_grid(axes)
    handles = [Patch(facecolor=method.colour, label=method.label) for method in methods]
    handles += [
        Line2D([], [], marker="D", markersize=4.5, markerfacecolor="white", markeredgecolor="black", linestyle="none",
               label="Consistency reference"),
        Line2D([], [], color=style.METHODS["chance"], linestyle="--", linewidth=0.8, label="Chance"),
    ]  # fmt: skip
    axes.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=min(5, len(handles)))
    return figure


def plot_learning_curves_and_calibration(curves: list[Shown], calibrations: list[Shown]) -> Figure:
    """(a) Accuracy as the training judgments grow and (b) as the training images grow, each method's mean over the
    folds with its 95% interval as a band (its learning curve: `by`, `share`, `accuracy`, `low`, `high`); (c) the
    observed share of wins against the predicted probability (its calibration: `predicted`, `observed`,
    `judgments`), the diagonal being perfect calibration. Full width."""
    figure = style.new_figure("full", 6.0)
    panels = figure.subplots(1, 3)
    for axes, by in zip(panels[:2], ("judgments", "images"), strict=True):
        for method in curves:
            table = method.table[method.table["by"] == by].sort_values("share")
            share = table["share"].to_numpy(dtype=float) * 100
            axes.fill_between(share, table["low"], table["high"], color=method.colour, alpha=0.15, linewidth=0)
            axes.plot(share, table["accuracy"], "o-", color=method.colour, markersize=2.5, label=method.label)
        axes.set_xlim(0, 105)
        axes.set_xticks([10, 25, 50, 75, 100], ["10", "25", "50", "75", "100"])
        axes.set_xlabel(f"Share of the training {by} (%)")
        style.light_grid(axes)
    low = min(float(method.table["low"].min()) for method in curves) if curves else 0.5
    for axes in panels[:2]:
        axes.set_ylim(min(0.5, np.floor(low * 20) / 20), max(0.8, axes.get_ylim()[1]))
    panels[0].set_ylabel("Pairwise accuracy")
    panels[1].sharey(panels[0])
    panels[1].tick_params(labelleft=False)
    axes = panels[2]
    axes.plot([0, 1], [0, 1], color=style.METHODS["chance"], linestyle="--", linewidth=0.8)
    for method in calibrations:
        table = method.table[method.table["judgments"] > 0]
        axes.plot(table["predicted"], table["observed"], "o-", color=method.colour, markersize=2.5, label=method.label)
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_aspect("equal")
    axes.set_xlabel("Predicted probability")
    axes.set_ylabel("Observed share")
    style.light_grid(axes)
    panels[0].legend(loc="lower right")
    style.panel_letters(panels)
    return figure


@dataclass
class MapPanel:
    """A page's map: its crop of the image, the patch scores on its grid (rows × columns, covering the crop), and its
    title."""

    image: np.ndarray
    scores: np.ndarray
    title: str


def greyed(image: np.ndarray, lift: float = 0.35) -> np.ndarray:
    """An RGB image in grey, lightened, so that a colour drawn over it stands out."""
    grey = image[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114]) / 255.0
    return np.clip(lift + (1 - lift) * grey, 0, 1)


def plot_local_maps(panels: list[MapPanel], scale: tuple[float, float] | None = None, columns: int = 3) -> Figure:
    """Each page greyed, its patch scores over it in vermillion on one colour scale for every page (`scale`, by
    default from the lowest to the highest patch score shown; lower scores transparent, higher ones in full colour),
    `columns` pages a row, and the scale below. Full width."""
    aspect = max(panel.image.shape[0] / panel.image.shape[1] for panel in panels)
    columns = min(columns, len(panels))
    rows = -(-len(panels) // columns)
    width = 15 / columns
    figure = style.new_figure("full", rows * (width * aspect + 0.6) + 1.4)
    grid = figure.add_gridspec(rows + 1, columns, height_ratios=[width * aspect] * rows + [0.25])
    if scale is None:
        scale = (min(float(panel.scores.min()) for panel in panels), max(float(panel.scores.max()) for panel in panels))
    norm = Normalize(*scale, clip=True)
    colours = style.map_colormap()
    drawn = []
    for number, panel in enumerate(panels):
        axes = figure.add_subplot(grid[number // columns, number % columns])
        height, width = panel.image.shape[:2]
        axes.imshow(greyed(panel.image), cmap="gray", vmin=0, vmax=1, extent=(0, width, height, 0))
        image = axes.imshow(panel.scores, cmap=colours, norm=norm, extent=(0, width, height, 0),
                            interpolation="nearest")  # fmt: skip
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_visible(False)
        drawn.append(axes)
    style.panel_letters(drawn, [panel.title for panel in panels])
    bar_axes = figure.add_subplot(grid[rows, 1 : columns - 1] if columns >= 3 else grid[rows, :])
    bar = figure.colorbar(image, cax=bar_axes, orientation="horizontal")
    bar.set_label("Patch score (local complexity)")
    bar.outline.set_linewidth(0.5)
    return figure
