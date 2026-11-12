"""The folio crop: where the leaf is on the page image.

Otsu's threshold on CIELAB lightness separates the bright leaf from a darker ground; the largest bright component is
the leaf, fitted with a quadrilateral (its rotated rectangle when four corners don't fit). When that quadrilateral
covers under half or over 98% of the image, the whole image is the folio: a leaf filling the image has no ground
around it.

YOLO's folio box is an option, for pages photographed on a bright ground: with it, the component overlapping the box
most is the leaf, the box is the fallback, and the IoU of the two is recorded.
"""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

Method = Literal["otsu", "yolo", "image"]
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class FolioSettings:
    """How the leaf is found.

    Args:
        work_edge: The long edge, in pixels, the image is scaled to for thresholding.
        min_area: Otsu's quadrilateral is implausible under this share of the image.
        max_area: Otsu's quadrilateral is implausible over this share of the image.
        disagreement: With YOLO's folio box, the pages where the IoU of the box and Otsu's quadrilateral is under this
            are listed in the run's report.
        batch_size: Pages per batch sent to a worker.
        workers: Worker processes; by default a share of `--jobs`.
    """

    work_edge: int = 700
    min_area: float = 0.5
    max_area: float = 0.98
    disagreement: float = 0.85
    batch_size: int = 4
    workers: int | None = None


@dataclass(frozen=True)
class Folio:
    """The leaf: how it was found (`otsu`, or the fallback: `yolo` for YOLO's box, `image` for the whole image), its
    quadrilateral (four corners, in the image's pixels), and the IoU of YOLO's box with Otsu's quadrilateral (None
    without either)."""

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
    """Otsu's leaf: the component's quadrilateral, and the share of the image it covers."""

    quadrilateral: np.ndarray
    area: float


def otsu_region(rgb: np.ndarray, prefer: np.ndarray | None = None) -> Region | None:
    """The largest bright component, or the one overlapping `prefer` most, cleaned, with its quadrilateral."""
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
    contours, _ = cv2.findContours((labels == chosen).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quadrilateral = fit_quadrilateral(cv2.convexHull(max(contours, key=cv2.contourArea)))
    return Region(quadrilateral, cv2.contourArea(quadrilateral.astype(np.float32)) / labels.size)


def plausible(region: Region, settings: FolioSettings) -> bool:
    return settings.min_area <= region.area <= settings.max_area


def corners(points: np.ndarray, scale: float = 1.0) -> list[tuple[float, float]]:
    return [(float(x) / scale, float(y) / scale) for x, y in points]


def find_folio(rgb: np.ndarray, box: Box | None = None, settings: FolioSettings | None = None) -> Folio:
    """The leaf on an RGB page image; with YOLO's folio box (in the image's pixels), the leaf it overlaps most."""
    settings = settings or FolioSettings()
    height, width = rgb.shape[:2]
    scale = min(1.0, settings.work_edge / max(height, width))
    size = (round(width * scale), round(height * scale))
    small = rgb if scale == 1.0 else cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    shape = small.shape[:2]
    box_mask = polygon_mask(box_corners(tuple(value * scale for value in box)), shape) if box else None  # type: ignore[arg-type]
    region = otsu_region(small, box_mask)
    overlap = None
    if region is not None and box_mask is not None:
        overlap = iou(polygon_mask(region.quadrilateral, shape), box_mask)
    if region is not None and plausible(region, settings):
        return Folio("otsu", corners(region.quadrilateral, scale), overlap)
    if box is not None:
        return Folio("yolo", corners(box_corners(box)), overlap)
    return Folio("image", corners(box_corners((0.0, 0.0, float(width), float(height)))), None)
