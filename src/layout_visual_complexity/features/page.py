"""A page as feature families see it: its folio, zones and lines from the layout store, in the full image's pixels,
its image on request, and its patch tokens from a token set when a family needs them. What a family asks for is
prepared once per page and shared by the families."""

import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class Zone:
    """A region (or one of YOLO's lines): its id, class and polygon, an array of (x, y) points."""

    id: str
    cls: str
    polygon: np.ndarray


@dataclass(frozen=True)
class TextLine:
    """A text line: its id, boundary polygon, baseline, and the id of its region, if any."""

    id: str
    polygon: np.ndarray
    baseline: np.ndarray
    region: str | None


def points(values: Any) -> np.ndarray:
    """A list of [x, y] points as an (n, 2) float array; empty when there are none."""
    if values is None:
        return np.zeros((0, 2))
    array = np.asarray([list(point) for point in values], dtype=float)
    return array.reshape(-1, 2)


def shape(polygon: np.ndarray) -> BaseGeometry | None:
    """A polygon as a valid shapely geometry with an area, or None: closed if needed, repaired by `buffer(0)` when
    it crosses itself, dropped when it has under three points or no area."""
    if len(polygon) < 3:
        return None
    found: BaseGeometry = Polygon(polygon)
    if not found.is_valid:
        found = found.buffer(0)
    if found.is_empty or not found.is_valid or found.area <= 0:
        return None
    return found


@dataclass
class Page:
    """One page: `width` × `height` is the full image, whose pixels the coordinates are in. Without a folio, the
    whole image is the folio."""

    page: str
    width: int
    height: int
    folio: np.ndarray | None = None
    regions: list[Zone] = field(default_factory=list)
    lines: list[TextLine] = field(default_factory=list)
    yolo_lines: list[Zone] = field(default_factory=list)
    image_path: Path | None = None
    tokens: Any = None
    _folio_images: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_layout(cls, page: str, rows: pd.DataFrame, width: int, height: int, image: Path | None = None) -> "Page":
        """A page from its rows in the layout store (`kind`, `class`, `polygon`, `baseline`, `region`, `id`), in
        their order."""
        folio = None
        regions, lines, yolo_lines = [], [], []
        columns = (rows[name] for name in ("kind", "class", "polygon", "baseline", "region", "id"))
        for kind, kind_class, polygon, baseline, region, identifier in zip(*columns, strict=True):
            if kind == "folio":
                folio = points(polygon)
            elif kind == "region":
                regions.append(Zone(str(identifier), str(kind_class), points(polygon)))
            elif kind == "yolo_line":
                yolo_lines.append(Zone(str(identifier), str(kind_class), points(polygon)))
            elif kind == "line":
                region = region if isinstance(region, str) and region else None
                lines.append(TextLine(str(identifier), points(polygon), points(baseline), region))
        return cls(page, int(width), int(height), folio, regions, lines, yolo_lines, image)

    # The folio.

    @cached_property
    def folio_shape(self) -> BaseGeometry:
        """The folio as a valid polygon; the whole image when the store has none or it has no area."""
        found = shape(self.folio) if self.folio is not None else None
        return found if found is not None else box(0, 0, self.width, self.height)

    @cached_property
    def folio_bounds(self) -> tuple[float, float, float, float]:
        """The folio's bounding box (x0, y0, x1, y1), from its points as the store gives them."""
        if self.folio is not None and len(self.folio) >= 3:
            x0, y0 = self.folio.min(axis=0)
            x1, y1 = self.folio.max(axis=0)
            return float(x0), float(y0), float(x1), float(y1)
        return self.folio_shape.bounds

    @property
    def folio_size(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.folio_bounds
        return x1 - x0, y1 - y0

    # Lines and zones on the folio.

    @cached_property
    def line_shapes(self) -> list[BaseGeometry | None]:
        """Each line's boundary as a valid polygon, or None."""
        return [shape(line.polygon) for line in self.lines]

    @cached_property
    def folio_lines(self) -> list[int]:
        """The lines on the folio, in the store's order: those whose baseline's midpoint (along it) lies inside it."""
        folio = self.folio_shape
        inside = []
        for number, line in enumerate(self.lines):
            if len(line.baseline) == 0:
                continue
            middle = (
                Point(line.baseline[0])
                if len(line.baseline) == 1
                else LineString(line.baseline).interpolate(0.5, normalized=True)
            )
            if folio.contains(middle):
                inside.append(number)
        return inside

    def zones_on_folio(self, zones: list[Zone], overlap: float) -> list[Zone]:
        """The zones whose centroid lies inside the folio and at least `overlap` of whose area does."""
        folio = self.folio_shape
        kept = []
        for zone in zones:
            found = shape(zone.polygon)
            if found is None or not folio.contains(found.centroid):
                continue
            if found.intersection(folio).area / found.area >= overlap:
                kept.append(zone)
        return kept

    # The image.

    @cached_property
    def image(self) -> np.ndarray:
        """The page image as RGB, at the size it was fetched; read once."""
        from PIL import Image

        if self.image_path is None:
            raise ValueError(f"{self.page}: no image")
        with Image.open(self.image_path) as opened:
            return np.asarray(opened.convert("RGB"))

    def folio_image(self, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """The image of the folio's bounding box resized to `size` (rows, columns), and a mask of the folio within it
        (True inside); made once per size."""
        if size not in self._folio_images:
            import cv2

            image = self.image
            scale_x, scale_y = image.shape[1] / self.width, image.shape[0] / self.height
            x0, y0, x1, y1 = self.folio_bounds
            left, top = max(0, math.floor(x0 * scale_x)), max(0, math.floor(y0 * scale_y))
            right = min(image.shape[1], max(left + 1, math.ceil(x1 * scale_x)))
            bottom = min(image.shape[0], max(top + 1, math.ceil(y1 * scale_y)))
            crop = image[top:bottom, left:right]
            resized = cv2.resize(crop, (size[1], size[0]), interpolation=cv2.INTER_AREA)
            mask = polygon_mask(self.folio_points(), (x0, y0, x1, y1), size)
            self._folio_images[size] = (resized, mask)
        return self._folio_images[size]

    def folio_points(self) -> np.ndarray:
        if self.folio is not None and len(self.folio) >= 3:
            return self.folio
        x0, y0, x1, y1 = self.folio_shape.bounds
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def raster_shape(width: float, height: float, long_edge: int) -> tuple[int, int]:
    """Rows and columns of a raster of a `width` × `height` box with `long_edge` cells along its longer side."""
    cell = max(width, height) / long_edge
    return max(1, round(height / cell)), max(1, round(width / cell))


def polygon_mask(polygon: np.ndarray, frame: tuple[float, float, float, float], size: tuple[int, int]) -> np.ndarray:
    """Which cells of a raster of `frame` (x0, y0, x1, y1) with `size` (rows, columns) have their centre inside the
    polygon."""
    return polygon_coverage([polygon], frame, size, supersample=1) >= 0.5 if len(polygon) >= 3 else np.zeros(size, bool)


def polygon_coverage(
    polygons: list[np.ndarray], frame: tuple[float, float, float, float], size: tuple[int, int], supersample: int = 4
) -> np.ndarray:
    """The share of each cell of a raster of `frame` (x0, y0, x1, y1) with `size` (rows, columns) that the union of
    the polygons covers.

    The polygons are sampled on a lattice of `supersample` + 1 points along each side of a cell, corners included,
    and each cell averages its points by the trapezoid rule. A point on a polygon's edge counts as inside on every
    side, so a mirrored layout gives a mirrored raster."""
    import cv2

    rows, columns = size
    x0, y0, x1, y1 = frame
    lattice = np.zeros((rows * supersample + 1, columns * supersample + 1), np.uint8)
    scale_x = columns * supersample / max(x1 - x0, 1e-9)
    scale_y = rows * supersample / max(y1 - y0, 1e-9)
    shift = 4
    drawn = []
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        points = np.column_stack([(polygon[:, 0] - x0) * scale_x, (polygon[:, 1] - y0) * scale_y])
        drawn.append(np.round(points * (1 << shift)).astype(np.int32))
    if drawn:
        cv2.fillPoly(lattice, drawn, 1, lineType=cv2.LINE_8, shift=shift)
    return trapezoid_cells(trapezoid_cells(lattice.astype(float), supersample, axis=1), supersample, axis=0)


def trapezoid_cells(values: np.ndarray, supersample: int, axis: int) -> np.ndarray:
    """Along `axis`, the mean of each cell's `supersample` + 1 lattice points, its two ends weighing half."""
    moved = np.moveaxis(values, axis, -1)
    cells = (moved.shape[-1] - 1) // supersample
    inner = moved[..., : cells * supersample].reshape(*moved.shape[:-1], cells, supersample).sum(axis=-1)
    ends = moved[..., : cells * supersample + 1 : supersample]
    found = (inner - 0.5 * ends[..., :-1] + 0.5 * ends[..., 1:]) / supersample
    return np.moveaxis(found, -1, axis)
