"""Double-page spreads split into their two pages, before the folio crop.

The gutter is found by image processing alone, never from the ground truth: in the middle fifth of the image's width,
the column where the leaf is thinnest against the columns on either side of it. The leaf is Otsu's bright pixels as
the folio crop finds them, with dark runs down a column shorter than a tenth of the height counted as leaf (writing
and initials are dark in short runs, a gutter's shadow down the whole leaf), counted down each column and smoothed
over a few columns. A column's depth is how far it lies below the thickest leaf on its left and on its right, within
a few hundredths of the width, so that the gutter's valley counts and the edge of a text block, a step, doesn't.
Where no column lies deep enough, the split falls back to the middle.

Each half is a page of its own: a part of the image, its ground truth with it. A zone or line goes to the half that
holds its polygon's centroid, moved to the half's coordinates; a line lying more than a share across the split marks
a split that went wrong. A half with no ground-truth line is left out, for often only one page of a spread was
transcribed.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from shapely.geometry import Polygon, box

from layout_visual_complexity.ground_truth import Page, shifted


@dataclass(frozen=True)
class SpreadSettings:
    """Which images are double-page spreads, and how they're split.

    Args:
        pages: The images split into two pages, by id; `auto` for every image wider than tall.
        work_edge: The long edge, in pixels, the image is scaled to for finding the gutter.
        middle: The share of the width, centred, in which the gutter is looked for.
        run: Dark runs down a column shorter than this share of the height (writing, initials) count as leaf.
        smoothing: The leaf's thickness is smoothed over this share of the width.
        sides: A column's depth is measured against the thickest leaf on either side of it within this share of
            the width.
        contrast: The deepest column must lie this share of the height below both sides, or the split falls back
            to the middle.
        crossing: A ground-truth line lying more than this share of its area across the split is listed.
    """

    pages: list[str] | Literal["auto"] = field(default_factory=list)
    work_edge: int = 700
    middle: float = 0.2
    run: float = 0.1
    smoothing: float = 0.01
    sides: float = 0.03
    contrast: float = 0.05
    crossing: float = 0.1


@dataclass(frozen=True)
class Gutter:
    """Where a spread is split, in the image's pixels; `found` is False where it fell back to the middle. `profile`
    is the leaf's smoothed thickness (a share of the height) across the middle, at the working size."""

    x: int
    found: bool
    depth: float
    profile: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class Split:
    """A spread split: its gutter, its halves kept, the halves left out without ground-truth lines, and the lines
    lying across the split (id and share of its area on the other side)."""

    page: str
    gutter: Gutter
    halves: list[Page]
    empty: list[str]
    crossing: list[tuple[str, float]]


def is_spread(page: Page, settings: SpreadSettings) -> bool:
    if settings.pages == "auto":
        return page.width > page.height
    return page.page in set(settings.pages)


def working_image(path: Path, work_edge: int) -> tuple[np.ndarray, float]:
    """The image as RGB at `work_edge` on its long edge (never larger), made smaller as the folio crop makes it, and
    the scale from the image to it."""
    import cv2
    from PIL import Image

    with Image.open(path) as opened:
        rgb = np.asarray(opened.convert("RGB"))
    height, width = rgb.shape[:2]
    scale = min(1.0, work_edge / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = rgb if scale == 1.0 else cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return small, size[0] / width


def find_gutter(rgb: np.ndarray, settings: SpreadSettings | None = None) -> tuple[int, bool, float, list[float]]:
    """The gutter's column in an RGB image: whether one stands out, by how much (a share of the height), and the
    smoothed thickness of the leaf across the middle."""
    import cv2

    from layout_visual_complexity.segmentation.folio import bright_mask

    settings = settings or SpreadSettings()
    height, width = rgb.shape[:2]
    run = max(3, round(settings.run * height))
    leaf = cv2.morphologyEx(
        bright_mask(rgb).astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, run))
    )
    thickness = leaf.sum(axis=0) / height
    window = max(3, round(settings.smoothing * width) | 1)
    padded = np.pad(thickness, window // 2, mode="edge")
    smooth = np.convolve(padded, np.ones(window) / window, mode="valid")
    start = round(width * (0.5 - settings.middle / 2))
    stop = max(start + 1, round(width * (0.5 + settings.middle / 2)))
    reach = max(1, round(settings.sides * width))
    depths = np.array(
        [
            min(smooth[max(0, x - reach) : x + 1].max(), smooth[x : x + reach + 1].max()) - smooth[x]
            for x in range(start, stop)
        ]
    )
    deepest = int(np.argmax(depths))
    depth = float(depths[deepest])
    profile = [float(value) for value in smooth[start:stop]]
    if depth >= settings.contrast:
        return start + deepest, True, depth, profile
    return width // 2, False, depth, profile


def gutter_of(page: Page, settings: SpreadSettings) -> Gutter:
    """A spread's gutter, in the full image's pixels."""
    rgb, scale = working_image(page.image, settings.work_edge)
    column, found, depth, profile = find_gutter(rgb, settings)
    x = round((column + 0.5) / scale) if found else page.width // 2
    return Gutter(min(max(1, x), page.width - 1), found, depth, profile)


def _shape(points: list[tuple[float, float]]):  # type: ignore[no-untyped-def]
    from shapely.validation import make_valid

    if len(points) < 3:
        return None
    polygon = Polygon(points)
    return polygon if polygon.is_valid else make_valid(polygon)


def _centre_x(points: list[tuple[float, float]]) -> float:
    shape = _shape(points)
    if shape is not None and not shape.is_empty and shape.area > 0:
        return float(shape.centroid.x)
    return float(np.mean([x for x, _ in points])) if points else 0.0


def split_page(page: Page, gutter: Gutter, crossing: float = 0.1) -> Split:
    """A spread's two halves at its gutter, with their ground truth; halves without lines left out."""
    x = gutter.x
    regions = {"left": (0, 0, x, page.height), "right": (x, 0, page.width, page.height)}
    sides = {"left": box(-1e9, -1e9, x, 1e9), "right": box(x, -1e9, 1e9, 1e9)}
    zones = {"left": [], "right": []}  # type: dict[str, list]
    lines = {"left": [], "right": []}  # type: dict[str, list]
    across = []
    for zone in page.zones:
        zones["left" if _centre_x(zone.polygon) < x else "right"].append(zone)
    for line in page.lines:
        side = "left" if _centre_x(line.polygon or line.baseline) < x else "right"
        lines[side].append(line)
        shape = _shape(line.polygon)
        if shape is not None and shape.area > 0:
            other = "right" if side == "left" else "left"
            share = shape.intersection(sides[other]).area / shape.area
            if share > crossing:
                across.append((line.id, float(share)))
    halves, empty = [], []
    for side in ("left", "right"):
        part = Page(
            page.page,
            page.document,
            page.image,
            page.width,
            page.height,
            zones[side],
            lines[side],
            page.source,
            page.checksum,
            None,
            page.label,
        )
        half = shifted(part, regions[side], side)
        (halves if lines[side] else empty).append(half if lines[side] else half.page)
    return Split(page.page, gutter, halves, empty, across)  # type: ignore[arg-type]


def split_spreads(pages: list[Page], settings: SpreadSettings) -> tuple[list[Page], list[Split]]:
    """Every page, spreads replaced by their halves in place; and each spread's split."""
    kept, splits = [], []
    for page in pages:
        if not is_spread(page, settings):
            kept.append(page)
            continue
        split = split_page(page, gutter_of(page, settings), settings.crossing)
        splits.append(split)
        kept += split.halves
    return kept, splits
