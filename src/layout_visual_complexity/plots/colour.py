"""Figures of the colour level: pages whose ink is drawn by hue class, beside the classes' shares of the ink."""

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from layout_visual_complexity.plots import style

CLASSES = ("red", "gold", "green", "blue")


@dataclass
class ColourPanel:
    """A page as the figure draws it: its id, each crop pixel's class (0 paper, 1 plain ink, 2–5 red, gold, green
    and blue, −1 ink left out), the folio within the crop, and each hue class's share of the ink."""

    page: str
    classes: np.ndarray
    inside: np.ndarray
    shares: dict[str, float]


def drawn(panel: ColourPanel, long_edge: int = 700) -> np.ndarray:
    """The page's ink in its classes' colours on light parchment, white outside the folio, made smaller by area so that
    thin strokes stay visible."""
    import cv2

    colours = {0: style.PAPER, 1: style.HUE_CLASS_COLOURS["plain"], -1: style.HUE_CLASS_COLOURS["left out"]}
    colours |= {number: style.HUE_CLASS_COLOURS[name] for number, name in enumerate(CLASSES, start=2)}
    image = np.ones((*panel.classes.shape, 3), np.float32)
    for number, colour in colours.items():
        image[panel.classes == number] = to_rgb(colour)
    image[~panel.inside] = 1.0
    scale = long_edge / max(image.shape[:2])
    if scale < 1:
        size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return np.clip(image, 0, 1)


def plot_ink_colour(panels: list[ColourPanel]) -> Figure:
    """Each page's ink drawn by hue class, and below it the hue classes' shares of its ink as bars on one scale for
    every page, the pages side by side. Full width."""
    columns = len(panels)
    images = [drawn(panel) for panel in panels]
    aspect = max(image.shape[0] / image.shape[1] for image in images)
    width = 15 / columns - 1.3  # each column less the bars' labels, which narrow its image too
    figure = style.new_figure("full", width * aspect + 3.6)
    grid = figure.add_gridspec(2, columns, height_ratios=[width * aspect, 2.4])
    top = max(max(panel.shares.values()) for panel in panels)
    limit = max(1.0, np.ceil(top * 100 / 2) * 2)
    letters = []
    for number, (panel, image) in enumerate(zip(panels, images, strict=True)):
        axes = figure.add_subplot(grid[0, number])
        axes.imshow(image, interpolation="lanczos")
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_visible(False)
        letters.append(axes)
        bars = figure.add_subplot(grid[1, number])
        y = np.arange(len(CLASSES))[::-1]
        values = [100 * panel.shares[name] for name in CLASSES]
        bars.barh(y, values, height=0.65, color=[style.HUE_CLASS_COLOURS[name] for name in CLASSES])
        bars.set_yticks(y, [name.capitalize() for name in CLASSES])
        bars.set_xlim(0, limit)
        bars.set_xlabel("Share of the ink (%)")
        bars.tick_params(axis="y", length=0)
        style.light_grid(bars, "x")
    style.panel_letters(letters, [panel.page for panel in panels])
    return figure
