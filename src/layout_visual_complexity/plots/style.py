"""The figures' shared look: one colour per method, the colour scales, the font, the print sizes and panel letters.

Figures never name a colour of their own: they ask for a method's (`METHODS["geometry"]`) or a scale. The palette is
Okabe and Ito's, told apart with the common colour-vision deficiencies. Text is STIXGeneral, mathematics in STIX, at
9 pt, ticks and legends at 8 pt, file and code names in DejaVu Sans Mono: all three come with matplotlib, so a figure
looks the same on every machine. Figures are drawn at their print size, the full width of the text (15 cm) or half of
it (7.2 cm), and never scaled in LaTeX; nothing is under 6 pt, no line under 0.5 pt.
"""

from collections.abc import Iterable
from typing import Literal

import matplotlib
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

CENTIMETRE = 1 / 2.54
FULL_WIDTH = 15 * CENTIMETRE
HALF_WIDTH = 7.2 * CENTIMETRE

# One colour per method, in every figure; the consistency reference black, chance a grey dashed line.
METHODS = {
    "trivial": "#999999",
    "generic_image": "#56B4E9",
    "kyle_davidson": "#CC79A7",
    "ink": "#009E73",
    "geometry": "#E69F00",
    "colour": "#D55E00",
    "vit": "#0072B2",
    "ensemble": "#1B2A49",
    "consistency": "#000000",
    "chance": "#999999",
}
BATCHES = ("#000000", "#E69F00", "#0072B2")
NEGATIVE, MIDDLE, POSITIVE = "#0072B2", "#F7F7F7", "#D55E00"
HIGHLIGHT = "#D55E00"
TEXT, SECONDARY, GRID = "#1F2328", "#57606A", "#E6E9ED"
PAPER, PAPER_EDGE, WRITING = "#F4EBD9", "#C9B99A", "#5C4E40"
SERIF, MONOSPACE = "STIXGeneral", "DejaVu Sans Mono"
LINE = 0.75  # points

SETTINGS = {
    "font.family": SERIF,
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": TEXT,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "svg.fonttype": "none",
    "figure.dpi": 100,
}


def apply() -> None:
    """Make the style matplotlib's current settings."""
    matplotlib.rcParams.update(SETTINGS)


def width_inches(width: Literal["full", "half"] | float) -> float:
    return {"full": FULL_WIDTH, "half": HALF_WIDTH}.get(width, width)  # type: ignore[call-overload]


def new_figure(width: Literal["full", "half"] | float, height_cm: float, constrained: bool = True) -> Figure:
    """A figure in the style at its print size: `width` full or half the text's (or inches), `height_cm` tall. It
    belongs to no window or global state."""
    apply()
    return Figure(figsize=(width_inches(width), height_cm * CENTIMETRE), layout="constrained" if constrained else None)


def monospace(size: float = 7) -> FontProperties:
    """The font for file and code names."""
    return FontProperties(family=MONOSPACE, size=size)


def panel_letters(axes: Iterable[Axes], titles: Iterable[str] | None = None) -> None:
    """Letter the panels (a), (b), … in bold at their top left, before their titles if any."""
    titles = list(titles) if titles is not None else []
    for index, axis in enumerate(axes):
        letter = f"({chr(ord('a') + index)})"
        title = titles[index] if index < len(titles) else ""
        axis.set_title(f"$\\mathbf{{{letter}}}$  {title}" if title else letter, loc="left", fontweight="bold")


def light_grid(axis: Axes, which: Literal["x", "y"] = "y") -> None:
    """Light gridlines along one axis only, behind the data."""
    axis.grid(axis=which, color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)


def lighter(colour: str, amount: float) -> str:
    """`colour` mixed with white: 0 keeps it, 1 gives white."""
    return to_hex([value + (1 - value) * amount for value in to_rgb(colour)])


def correlation_colormap() -> Colormap:
    """Correlations, negative to positive: blue, a light grey middle, vermillion."""
    return LinearSegmentedColormap.from_list("correlation", [NEGATIVE, MIDDLE, POSITIVE])


def score_colormap() -> Colormap:
    """Elo and predicted scores."""
    return matplotlib.colormaps["viridis"]


def map_colormap() -> Colormap:
    """Local maps over page images: transparent to vermillion."""
    colour = to_rgb(HIGHLIGHT)
    return LinearSegmentedColormap.from_list("map", [(*colour, 0.0), (*colour, 0.8)])


def print_width_cm(figure: Figure) -> float:
    """The figure's width as printed, in centimetres."""
    return figure.get_size_inches()[0] / CENTIMETRE
