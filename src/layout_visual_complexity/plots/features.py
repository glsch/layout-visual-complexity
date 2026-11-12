"""Figures of the features: each feature's correlation with the Elo scores, the features drawn on pages, and the plate
of the feature families.

Pages are drawn in the full image's pixels, cropped to the folio: a real page's image, or a schematic page (paper and
dark line polygons). The overlays show what the families measure, with the families' own choice of zones and lines.
"""

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch, Polygon, Rectangle
from matplotlib.path import Path as DrawnPath
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from layout_visual_complexity.features import geometry, layout
from layout_visual_complexity.features.geometry import GeometrySettings
from layout_visual_complexity.features.layout import LayoutSettings
from layout_visual_complexity.features.page import Page, TextLine, Zone, shape
from layout_visual_complexity.plots import style

BLUE, VERMILLION = style.NEGATIVE, style.POSITIVE
ZONE_COLOURS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#999999")


# Each feature's correlation.


def wrapped(name: str, width: int = 46) -> str:
    """A column's formal name, broken after its feature when it's long."""
    if len(name) <= width:
        return name
    family, feature, flavour = name.split("__")
    return f"{family}__{feature}__\n    {flavour}"


def plot_feature_correlations(rows: pd.DataFrame) -> Figure:
    """Each row's Spearman ρ (`rho`) as a dot with its interval (`low`, `high`), top to bottom in the rows' order,
    labelled by its formal name (`column`). Full width."""
    names = [wrapped(name) for name in rows["column"]]
    lines = sum(name.count("\n") + 1 for name in names)
    figure = style.new_figure("full", 1.6 + 0.36 * lines + 0.12 * len(rows))
    axes = figure.subplots()
    y = np.arange(len(rows))[::-1]
    rho = rows["rho"].to_numpy(dtype=float)
    colours = np.where(rho < 0, BLUE, VERMILLION)
    low, high = rows["low"].to_numpy(dtype=float), rows["high"].to_numpy(dtype=float)
    for position, value, start, end, colour in zip(y, rho, low, high, colours, strict=True):
        axes.plot([start, end], [position, position], color=colour, linewidth=1.0, solid_capstyle="butt")
        axes.plot([value], [position], "o", color=colour, markersize=4, zorder=3)
    axes.axvline(0, color=style.TEXT, linewidth=0.6)
    limit = max(0.2, float(np.nanmax(np.abs(np.concatenate([low, high]))))) + 0.05
    axes.set_xlim(-limit, limit)
    axes.set_ylim(-0.7, len(rows) - 0.3)
    axes.set_yticks(y, names)
    for label in axes.get_yticklabels():
        label.set_fontproperties(style.monospace(6.5))
    axes.tick_params(axis="y", length=0)
    axes.spines["left"].set_visible(False)
    axes.set_xlabel(r"Spearman's $\rho$ with the Elo score")
    return figure


# Pages.


def page_frame(axes: Axes, page: Page, margin: float = 0.02) -> None:
    """The axes show the page's folio, y downwards, without ticks."""
    x0, y0, x1, y1 = page.folio_bounds
    pad = margin * max(x1 - x0, y1 - y0)
    axes.set_xlim(x0 - pad, x1 + pad)
    axes.set_ylim(y1 + pad, y0 - pad)
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)


def draw_page(axes: Axes, page: Page, image: np.ndarray | None = None) -> None:
    """A real page's image, or a schematic page: the folio as paper and the lines' polygons in ink."""
    if image is not None:
        axes.imshow(image, extent=(0, page.width, page.height, 0), interpolation="antialiased")
    else:
        axes.add_patch(Polygon(page.folio_points(), closed=True, facecolor=style.PAPER, edgecolor=style.PAPER_EDGE,
                               linewidth=0.5))  # fmt: skip
        for zone in page.regions:
            if "DropCapital" in zone.cls or "Graphic" in zone.cls:
                colour = "#B4442B" if "DropCapital" in zone.cls else "#8F9A7A"
                axes.add_patch(Polygon(zone.polygon, closed=True, facecolor=colour, edgecolor="none", alpha=0.8))
        for line in page.lines:
            axes.add_patch(Polygon(line.polygon, closed=True, facecolor=style.WRITING, edgecolor="none", alpha=0.85))
    page_frame(axes, page)


def geometry_patch(found: BaseGeometry, **options: object) -> PathPatch | None:
    """A shapely polygon or multipolygon as a matplotlib patch (holes kept)."""
    parts = [part for part in getattr(found, "geoms", [found]) if not part.is_empty and part.area > 0]
    if not parts:
        return None
    vertices, codes = [], []
    for part in parts:
        for ring in [part.exterior, *part.interiors]:
            points = np.asarray(ring.coords)
            vertices.append(points)
            codes += [DrawnPath.MOVETO] + [DrawnPath.LINETO] * (len(points) - 2) + [DrawnPath.CLOSEPOLY]
    return PathPatch(DrawnPath(np.vstack(vertices), codes), **options)


def folio_union(page: Page) -> BaseGeometry:
    shapes = [page.line_shapes[number] for number in page.folio_lines]
    return unary_union([found for found in shapes if found is not None])


def add(axes: Axes, patch: PathPatch | Polygon | None) -> None:
    if patch is not None:
        axes.add_patch(patch)


def draw_written_space(axes: Axes, page: Page) -> None:
    """The union of the line polygons, and its convex hull."""
    union = folio_union(page)
    add(axes, geometry_patch(union, facecolor=BLUE, edgecolor="none", alpha=0.35))
    add(axes, geometry_patch(union.convex_hull, facecolor="none", edgecolor=VERMILLION, linewidth=1.0, linestyle="--"))


def draw_rectangularity(axes: Axes, page: Page) -> None:
    """The written space, a rectangle of its area and the folio's proportions, and what lies outside it."""
    union = folio_union(page)
    if union.is_empty:
        return
    width, height = page.folio_size
    rectangle = geometry.ideal_rectangle((union.centroid.x, union.centroid.y), union.area, width / height)
    add(axes, geometry_patch(union, facecolor=BLUE, edgecolor="none", alpha=0.3))
    add(axes, geometry_patch(union.difference(rectangle), facecolor=VERMILLION, edgecolor="none", alpha=0.6))
    add(axes, geometry_patch(rectangle, facecolor="none", edgecolor=VERMILLION, linewidth=1.0))


def similar_pair(lines: list[TextLine]) -> tuple[TextLine, TextLine] | None:
    """Two lines to compare: the longest and the shortest of those measured."""
    if len(lines) < 2:
        return None
    lengths = [geometry.polyline_length(line.baseline) for line in lines]
    return lines[int(np.argmax(lengths))], lines[int(np.argmin(lengths))]


def draw_line_similarity(axes: Axes, page: Page, settings: GeometrySettings) -> None:
    """Two lines highlighted on the page, and an inset with their polygons centred on each other."""
    zones = geometry.measured_zones(page, settings)
    pair = similar_pair([line for zone in zones for line in zone.lines])
    if pair is None:
        return
    shapes = [shape(line.polygon) for line in pair]
    if any(found is None for found in shapes):
        return
    for found, colour in zip(shapes, (BLUE, VERMILLION), strict=True):
        add(axes, geometry_patch(found, facecolor=colour, edgecolor=colour, alpha=0.6, linewidth=0.6))
    inset = axes.inset_axes((0.08, 0.015, 0.84, 0.075))
    centred = [geometry.translate(found, -found.centroid.x, -found.centroid.y) for found in shapes]
    for found, colour in zip(centred, (BLUE, VERMILLION), strict=True):
        add(inset, geometry_patch(found, facecolor=colour, edgecolor=colour, alpha=0.5, linewidth=0.6))
    x0, y0, x1, y1 = unary_union(centred).bounds
    pad = 0.08 * (x1 - x0)
    inset.set_xlim(x0 - pad, x1 + pad)
    inset.set_ylim(y1 + 3 * pad, y0 - 3 * pad)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor(style.SECONDARY)


def draw_zones(axes: Axes, page: Page, contains: str, settings: GeometrySettings) -> None:
    """The folio's zones whose class contains `contains`, and their union's outline."""
    zones = [zone for zone in page.zones_on_folio(page.regions, settings.zone_on_folio) if contains in zone.cls]
    shapes = [found for found in (shape(zone.polygon) for zone in zones) if found is not None]
    if shapes:
        union = unary_union(shapes)
        add(axes, geometry_patch(union, facecolor=VERMILLION, edgecolor=VERMILLION, alpha=0.35, linewidth=1.0))
        ring = union.buffer(0.012 * math.hypot(*page.folio_size))
        add(axes, geometry_patch(ring, facecolor="none", edgecolor=VERMILLION, linewidth=1.0, linestyle="--"))


def draw_line_length(axes: Axes, page: Page, settings: GeometrySettings) -> None:
    """Each measured baseline, and its zone's width above the zone."""
    for zone in geometry.measured_zones(page, settings):
        top = min(line.baseline[:, 1].min() for line in zone.lines)
        gap = 0.012 * page.folio_size[1]
        axes.plot([zone.left, zone.right], [top - 3 * gap] * 2, color=VERMILLION, linewidth=1.0)
        for end in (zone.left, zone.right):
            axes.plot([end, end], [top - 4 * gap, top - 2 * gap], color=VERMILLION, linewidth=1.0)
        for line in zone.lines:
            axes.plot(line.baseline[:, 0], line.baseline[:, 1], color=BLUE, linewidth=1.2, solid_capstyle="butt")


def draw_interlinear_distance(axes: Axes, page: Page, settings: GeometrySettings) -> None:
    """Baselines, and the vertical distance between every third pair of consecutive ones, at staggered places."""
    cap = 0.012 * page.folio_size[0]
    for zone in geometry.measured_zones(page, settings):
        for line in zone.lines:
            axes.plot(line.baseline[:, 0], line.baseline[:, 1], color=BLUE, linewidth=0.7)
        pairs = list(zip(zone.lines, zone.lines[1:], strict=False))
        for number, (upper, lower) in enumerate(pairs[::3]):
            start = max(upper.baseline[:, 0].min(), lower.baseline[:, 0].min())
            end = min(upper.baseline[:, 0].max(), lower.baseline[:, 0].max())
            if end <= start:
                continue
            x = start + (0.3 + 0.2 * (number % 3)) * (end - start)
            ys = [float(np.interp(x, *baseline_xy(line))) for line in (upper, lower)]
            axes.plot([x, x], ys, color=VERMILLION, linewidth=1.0)
            for y in ys:
                axes.plot([x - cap, x + cap], [y, y], color=VERMILLION, linewidth=1.0)


def baseline_xy(line: TextLine) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(line.baseline[:, 0])
    return line.baseline[order, 0], line.baseline[order, 1]


def draw_zigzag(axes: Axes, page: Page, settings: GeometrySettings) -> None:
    """The lines' ends at each margin of their zone, joined from line to line."""
    for zone in geometry.measured_zones(page, settings):
        boxes = [geometry.bounding_box(line) for line in zone.lines]
        ys = [float(line.baseline[:, 1].mean()) for line in zone.lines]
        axes.plot([box[0] for box in boxes], ys, color=BLUE, linewidth=1.0, marker="o", markersize=1.5)
        axes.plot([box[2] for box in boxes], ys, color=VERMILLION, linewidth=1.0, marker="o", markersize=1.5)
        top, bottom = min(ys) - 0.02 * page.folio_size[1], max(ys) + 0.02 * page.folio_size[1]
        for edge in (zone.left, zone.right):
            axes.plot([edge, edge], [top, bottom], color=style.SECONDARY, linewidth=0.5, linestyle=":")


def draw_parallelism(axes: Axes, page: Page, settings: GeometrySettings) -> None:
    """Each baseline's direction, from its first point to its last, drawn across the folio."""
    x0, _, x1, _ = page.folio_bounds
    for zone in geometry.measured_zones(page, settings):
        for line in zone.lines:
            (ax, ay), (bx, by) = line.baseline[0], line.baseline[-1]
            if bx == ax:
                continue
            slope = (by - ay) / (bx - ax)
            axes.plot([x0, x1], [ay + slope * (x0 - ax), ay + slope * (x1 - ax)], color=VERMILLION, linewidth=0.5)


GROUP_DRAWINGS: dict[str, Callable[[Axes, Page, GeometrySettings], None]] = {
    "AreaFeature__WrittenSpaceRatio": lambda axes, page, settings: draw_written_space(axes, page),
    "AreaFeature__WrittenSpaceRectangularity": lambda axes, page, settings: draw_rectangularity(axes, page),
    "AreaFeature__LineSimilarity": draw_line_similarity,
    "AreaFeature__GraphicZoneArea": lambda axes, page, settings: draw_zones(
        axes, page, settings.graphic_zones, settings
    ),
    "AreaFeature__DropCapitalArea": lambda axes, page, settings: draw_zones(
        axes, page, settings.drop_capital_zones, settings
    ),
    "LineFeature__LineLength": draw_line_length,
    "LineFeature__InterlinearDistance": draw_interlinear_distance,
    "LineFeature__ZigzagScore": draw_zigzag,
    "LineFeature__LineParallelism": draw_parallelism,
}


# Layout's measures.


def cells_to_pixels(page: Page, size: tuple[int, int]) -> Callable[[float, float], tuple[float, float]]:
    x0, y0, x1, y1 = page.folio_bounds
    rows, columns = size
    return lambda row, column: (x0 + (column + 0.5) * (x1 - x0) / columns, y0 + (row + 0.5) * (y1 - y0) / rows)


def draw_clutter(axes: Axes, page: Page, settings: LayoutSettings) -> None:
    """The folio's zones, outlined by type."""
    zones = page.zones_on_folio(page.regions, settings.zone_on_folio)
    types = sorted({zone.cls for zone in zones})
    for zone in zones:
        colour = ZONE_COLOURS[types.index(zone.cls) % len(ZONE_COLOURS)]
        axes.add_patch(Polygon(zone.polygon, closed=True, facecolor="none", edgecolor=colour, linewidth=1.2))


def draw_entropy(axes: Axes, page: Page, settings: LayoutSettings) -> None:
    """The folio's zones, filled by type."""
    zones = page.zones_on_folio(page.regions, settings.zone_on_folio)
    types = sorted({zone.cls for zone in zones})
    for zone in zones:
        colour = ZONE_COLOURS[types.index(zone.cls) % len(ZONE_COLOURS)]
        axes.add_patch(Polygon(zone.polygon, closed=True, facecolor=colour, edgecolor=colour, alpha=0.35))


def draw_order(axes: Axes, page: Page, settings: LayoutSettings) -> None:
    """The written mass's text block, its axes of symmetry, the mass's centre and the folio's."""
    mass = layout.mass_layer(page, settings)
    _, size, inside = layout.folio_raster(page, settings)
    if mass.sum() <= 0:
        return
    top, bottom, left, right = layout.text_block(mass, settings.trim)
    to_pixels = cells_to_pixels(page, size)
    (bx0, by0), (bx1, by1) = to_pixels(top - 0.5, left - 0.5), to_pixels(bottom + 0.5, right + 0.5)
    axes.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, facecolor="none", edgecolor=BLUE, linewidth=0.8))
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    axes.plot([cx, cx], [by0, by1], color=BLUE, linewidth=0.8, linestyle="--")
    axes.plot([bx0, bx1], [cy, cy], color=BLUE, linewidth=0.8, linestyle="--")
    rows, columns = np.indices(mass.shape)
    centre = to_pixels((rows * mass).sum() / mass.sum(), (columns * mass).sum() / mass.sum())
    folio = to_pixels(rows[inside].mean(), columns[inside].mean())
    axes.plot(*centre, "o", color=VERMILLION, markersize=4, zorder=5)
    axes.plot(*folio, "+", color=style.TEXT, markersize=7, markeredgewidth=1.0, zorder=5)


def draw_blank_space(axes: Axes, page: Page, settings: LayoutSettings) -> None:
    """The largest empty rectangle, and the margins between the text block and the folio's edges."""
    mass = layout.mass_layer(page, settings)
    _, size, inside = layout.folio_raster(page, settings)
    to_pixels = cells_to_pixels(page, size)
    _, where = layout.largest_rectangle((mass < settings.empty_below) & inside)
    if where is not None:
        first_row, last_row, first_column, last_column = where
        (x0, y0), (x1, y1) = (
            to_pixels(first_row - 0.5, first_column - 0.5),
            to_pixels(last_row + 0.5, last_column + 0.5),
        )
        axes.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=BLUE, edgecolor=BLUE, alpha=0.3, linewidth=0.8))
    if mass.sum() <= 0:
        return
    top, bottom, left, right = layout.text_block(mass, settings.trim)
    (bx0, by0), (bx1, by1) = to_pixels(top - 0.5, left - 0.5), to_pixels(bottom + 0.5, right + 0.5)
    fx0, fy0, fx1, fy1 = page.folio_bounds
    arrow = {"arrowstyle": "<->", "color": VERMILLION, "linewidth": 0.8, "shrinkA": 0, "shrinkB": 0}
    middle_x, middle_y = (bx0 + bx1) / 2, (by0 + by1) / 2
    for start, end in [((middle_x, fy0), (middle_x, by0)), ((middle_x, by1), (middle_x, fy1)),
                       ((fx0, middle_y), (bx0, middle_y)), ((bx1, middle_y), (fx1, middle_y))]:  # fmt: skip
        axes.annotate("", xy=end, xytext=start, arrowprops=arrow)


LAYOUT_DRAWINGS: dict[str, tuple[Callable[[Axes, Page, LayoutSettings], None], tuple[str, ...]]] = {
    "Clutter": (draw_clutter, ("Layout__Clutter",)),
    "Order": (draw_order, ("Layout__Symmetry", "Layout__Regularity", "Layout__Balance")),
    "Entropy": (draw_entropy, ("Layout__Entropy",)),
    "Blank space": (draw_blank_space, ("Layout__BlankSpace",)),
}


def plot_features_on_pages(
    panels: Sequence[tuple[str, Page, np.ndarray | None]], settings: GeometrySettings, columns: int = 3
) -> Figure:
    """One panel per feature group (`Family__Feature`), each on its page: a real page's image, or a schematic page
    when the image is None. Full width, the panels lettered."""
    rows = math.ceil(len(panels) / columns)
    aspect = max(page.folio_size[1] / page.folio_size[0] for _, page, _ in panels)
    panel_width = 15 / columns
    figure = style.new_figure("full", rows * (panel_width * aspect + 0.5))
    grid = figure.subplots(rows, columns, squeeze=False)
    axes_list = list(grid.flat)
    for axes, (group, page, image) in zip(axes_list, panels, strict=False):
        draw_page(axes, page, image)
        GROUP_DRAWINGS[group](axes, page, settings)
    for axes in axes_list[len(panels) :]:
        axes.set_visible(False)
    style.panel_letters(axes_list[: len(panels)])
    return figure


def plot_feature_families(page: Page, geometry_settings: GeometrySettings, layout_settings: LayoutSettings) -> Figure:
    """The plate: the nine geometric groups and Layout's four measures, each on the same schematic page, with their
    code names below. Full width: the geometric groups three to a row, Layout's four in the last row."""
    from matplotlib.gridspec import GridSpec

    figure = style.new_figure("full", 23.5)
    grid = GridSpec(4, 12, figure=figure, height_ratios=[1, 1, 1, 0.78])
    panels: list[tuple[Axes, Callable[[Axes], None], str]] = []
    for number, group in enumerate(GROUP_DRAWINGS):
        axes = figure.add_subplot(grid[number // 3, 4 * (number % 3) : 4 * (number % 3) + 4])
        panels.append((axes, lambda axes, group=group: GROUP_DRAWINGS[group](axes, page, geometry_settings), group))
    for number, (drawing, prefixes) in enumerate(LAYOUT_DRAWINGS.values()):
        axes = figure.add_subplot(grid[3, 3 * number : 3 * number + 3])
        panels.append((axes, lambda axes, drawing=drawing: drawing(axes, page, layout_settings), ",\n".join(prefixes)))
    for axes, drawing, name in panels:
        draw_page(axes, page)
        drawing(axes)
        axes.set_xlabel(name.replace("__", "__\n") if "," not in name else name, fontproperties=style.monospace(6.5))
    style.panel_letters([axes for axes, _, _ in panels])
    return figure


# A schematic page.


def schematic_page() -> Page:
    """A made-up page with what the features describe: a main column of lines with ragged ends, uneven spacing and
    slight slants; a drop capital and a graphic zone; a marginal gloss of short lines."""
    rng = np.random.default_rng(11)
    width, height = 1000, 1450
    folio = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    regions = [
        Zone("r1", "MainZone", np.array([[170, 170], [790, 170], [790, 1290], [170, 1290]], dtype=float)),
        Zone("r2", "MarginTextZone", np.array([[820, 300], [960, 300], [960, 620], [820, 620]], dtype=float)),
        Zone("r3", "DropCapitalZone", np.array([[175, 180], [255, 180], [255, 300], [175, 300]], dtype=float)),
        Zone("r4", "GraphicZone", np.array([[180, 760], [560, 760], [560, 1010], [180, 1010]], dtype=float)),
    ]
    lines = []
    y = 210.0
    for number in range(26):
        y += 40 + rng.normal(0, 4)
        if 735 < y < 1035:
            left = 580.0
        elif y < 310:
            left = 270.0
        else:
            left = 180.0 + max(0.0, rng.normal(0, 6))
        right = 780.0 - (rng.uniform(150, 330) if number in (6, 15, 24) else abs(rng.normal(0, 12)))
        slant = math.tan(math.radians(rng.normal(0, 0.6)))
        start, end = left, right
        baseline = np.array(
            [[start, y], [(start + end) / 2, y + slant * (end - start) / 2], [end, y + slant * (end - start)]]
        )
        rise = slant * (end - start)
        polygon = np.array([[start, y - 26], [end, y - 26 + rise], [end, y + 7 + rise], [start, y + 7]])
        lines.append(TextLine(f"l{number + 1}", polygon, baseline, "r1"))
    for number in range(8):
        y = 330 + 34 * number
        end = 950 - rng.uniform(0, 50)
        baseline = np.array([[830.0, y], [end, y]])
        lines.append(TextLine(f"g{number + 1}", np.array([[830, y - 20], [end, y - 20], [end, y + 5], [830, y + 5]],
                                                         dtype=float), baseline, "r2"))  # fmt: skip
    return Page("schematic", width, height, folio, regions, lines)
