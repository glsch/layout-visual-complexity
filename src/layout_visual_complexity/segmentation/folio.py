"""The folio crop: where the leaf is on the page image.

YOLO's folio box is the seed. Otsu's threshold on CIELAB lightness separates the bright leaf from a darker ground; the
bright component overlapping the box most is the leaf, fitted with a quadrilateral (its rotated rectangle when four
corners don't fit). The box is the fallback when Otsu's region is implausible: under half or over 98% of the image, or
filling under 90% of its rotated rectangle. With no box at all, the whole image is the folio.
"""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

Method = Literal["otsu", "yolo", "image"]
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class FolioSettings:
    """How the leaf is found within YOLO's folio box.

    Args:
        work_edge: The long edge, in pixels, the image is scaled to for thresholding.
        min_area: Otsu's region is implausible under this share of the image.
        max_area: Otsu's region is implausible over this share of the image.
        min_rectangularity: Otsu's region is implausible when it fills less than this share of its rotated rectangle.
        disagreement: Pages where the IoU of YOLO's box and Otsu's region is under this are listed in the run's
            report.
        batch_size: Pages per batch sent to a worker.
        workers: Worker processes; by default a share of `--jobs`.
    """

    work_edge: int = 700
    min_area: float = 0.5
    max_area: float = 0.98
    min_rectangularity: float = 0.9
    disagreement: float = 0.85
    batch_size: int = 4
    workers: int | None = None


@dataclass(frozen=True)
class Folio:
    """The leaf: how it was found, its quadrilateral (four corners, in the image's pixels), and the IoU of YOLO's box
    with Otsu's region (None when either is missing)."""

    method: Method
    quadrilateral: list[tuple[float, float]]
    iou: float | None


def box_corners(box: Box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def fit_quadrilateral(hull: np.ndarray) -> np.ndarray:
    """Four corners whose area is within 3% of the hull's, or else the hull's rotated rectangle."""
    hull_area = cv2.contourArea(hull)
    perimeter = cv2.arcLength(hull, True)
    for epsilon in np.linspace(0.005, 0.05, 10):
        approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        close = abs(cv2.contourArea(approx) - hull_area) <= 0.03 * hull_area
        if len(approx) == 4 and cv2.isContourConvex(approx) and close:
            return ordered(approx.reshape(4, 2).astype(float))
    return ordered(cv2.boxPoints(cv2.minAreaRect(hull)).astype(float))


def ordered(points: np.ndarray) -> np.ndarray:
    """Four corners clockwise (in image coordinates), from the one nearest the top left."""
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    clockwise = points[np.argsort(angles)]
    first = int(np.argmin(clockwise.sum(axis=1)))
    return np.roll(clockwise, -first, axis=0)


def bright_mask(rgb: np.ndarray) -> np.ndarray:
    """Otsu on CIELAB lightness, smoothed: parchment and paper against a darker ground."""
    lightness = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0], (0, 0), 2)
    _, mask = cv2.threshold(lightness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask > 0


def polygon_mask(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask.astype(bool)


def iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 0.0


@dataclass(frozen=True)
class Region:
    """Otsu's leaf: the component's quadrilateral, and the share of the image and of its rotated rectangle that the
    region within its outline fills."""

    quadrilateral: np.ndarray
    area: float
    rectangularity: float


def otsu_region(rgb: np.ndarray, prefer: np.ndarray | None) -> Region | None:
    """The bright component overlapping `prefer` most (or the largest), cleaned, with its quadrilateral."""
    k = max(3, round(0.01 * max(rgb.shape[:2])))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(bright_mask(rgb).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    chosen = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if prefer is not None:
        overlap = np.bincount(labels[prefer], minlength=count)[1:]
        if overlap.max() > 0:
            chosen = 1 + int(np.argmax(overlap))
    component = (labels == chosen).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    width, height = cv2.minAreaRect(contour)[1]
    filled = cv2.contourArea(contour)  # the region within its outline: the ink inside the leaf doesn't count against it
    return Region(
        quadrilateral=fit_quadrilateral(cv2.convexHull(contour)),
        area=filled / component.size,
        rectangularity=filled / (width * height) if width * height else 0.0,
    )


def plausible(region: Region, settings: FolioSettings) -> bool:
    sized = settings.min_area <= region.area <= settings.max_area
    return sized and region.rectangularity >= settings.min_rectangularity


def find_folio(rgb: np.ndarray, box: Box | None, settings: FolioSettings | None = None) -> Folio:
    """The leaf on an RGB page image, given YOLO's folio box (in the image's pixels) or None."""
    settings = settings or FolioSettings()
    height, width = rgb.shape[:2]
    if box is None:
        corners = [(0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))]
        return Folio("image", corners, None)
    scale = min(1.0, settings.work_edge / max(height, width))
    size = (round(width * scale), round(height * scale))
    small = rgb if scale == 1.0 else cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    shape = small.shape[:2]
    small_box = box_corners(tuple(value * scale for value in box))  # type: ignore[arg-type]
    box_mask = polygon_mask(small_box, shape)
    region = otsu_region(small, box_mask)
    overlap = iou(polygon_mask(region.quadrilateral, shape), box_mask) if region is not None else None
    if region is None or not plausible(region, settings):
        return Folio("yolo", [(float(x), float(y)) for x, y in box_corners(box)], overlap)
    corners = [(float(x) / scale, float(y) / scale) for x, y in region.quadrilateral]
    return Folio("otsu", corners, overlap)
