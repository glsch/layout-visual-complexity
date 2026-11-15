"""Found zones and lines against the ground truth's: one-to-one matching, F1, and the distance between baselines.

Zones are matched within each class by the IoU of boxes: the found box against the ground-truth polygon's bounding
box, the form the regions model was trained on. Lines are matched by the IoU of their polygons. Matching is
one-to-one (`scipy.optimize.linear_sum_assignment`, the pairs' IoU summed at most), a pair counting from an IoU of
0.5. A page's F1 pools its matches, misses and inventions: 2 TP / (2 TP + FP + FN).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon
from shapely.validation import make_valid

Points = Sequence[tuple[float, float]]
THRESHOLD = 0.5


def shape(points: Points):  # type: ignore[no-untyped-def]
    """A polygon from its points, made valid; None when it has no area."""
    if len(points) < 3:
        return None
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = make_valid(polygon)
        parts = [part for part in getattr(polygon, "geoms", [polygon]) if part.area > 0]
        polygon = shapely.union_all(parts) if parts else None
    return polygon if polygon is not None and not polygon.is_empty and polygon.area > 0 else None


def bounds(points: Points) -> tuple[float, float, float, float]:
    xs, ys = [x for x, _ in points], [y for _, y in points]
    return (min(xs), min(ys), max(xs), max(ys)) if xs else (0.0, 0.0, 0.0, 0.0)


def box_iou(found: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """The IoU of every found box with every ground-truth box (x0, y0, x1, y1)."""
    if not len(found) or not len(truth):
        return np.zeros((len(found), len(truth)))
    found, truth = np.asarray(found, float)[:, None, :], np.asarray(truth, float)[None, :, :]
    width = np.clip(np.minimum(found[..., 2], truth[..., 2]) - np.maximum(found[..., 0], truth[..., 0]), 0, None)
    height = np.clip(np.minimum(found[..., 3], truth[..., 3]) - np.maximum(found[..., 1], truth[..., 1]), 0, None)
    inter = width * height
    area = lambda boxes: (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])  # noqa: E731
    union = area(found) + area(truth) - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def polygon_iou(found: Sequence[Points], truth: Sequence[Points]) -> np.ndarray:
    """The IoU of every found polygon with every ground-truth polygon."""
    result = np.zeros((len(found), len(truth)))
    first = [shape(points) for points in found]
    second = [shape(points) for points in truth]
    kept_first = [i for i, item in enumerate(first) if item is not None]
    kept_second = [j for j, item in enumerate(second) if item is not None]
    if not kept_first or not kept_second:
        return result
    tree = shapely.STRtree([second[j] for j in kept_second])
    left, right = tree.query([first[i] for i in kept_first], predicate="intersects")
    if not len(left):
        return result
    a = np.array([first[kept_first[i]] for i in left], dtype=object)
    b = np.array([second[kept_second[j]] for j in right], dtype=object)
    inter = shapely.area(shapely.intersection(a, b))
    union = shapely.area(a) + shapely.area(b) - inter
    values = np.where(union > 0, inter / union, 0.0)
    result[[kept_first[i] for i in left], [kept_second[j] for j in right]] = values
    return result


def match(iou: np.ndarray, threshold: float = THRESHOLD) -> list[tuple[int, int, float]]:
    """One-to-one pairs (found, truth, IoU) with an IoU of at least `threshold`, the IoU summed at most."""
    if not iou.size:
        return []
    weights = np.where(iou >= threshold, iou, 0.0)
    rows, columns = linear_sum_assignment(weights, maximize=True)
    return [(int(i), int(j), float(iou[i, j])) for i, j in zip(rows, columns, strict=True) if iou[i, j] >= threshold]


def f1(matched: int, invented: int, missed: int) -> float:
    """2 TP / (2 TP + FP + FN); missing without anything either way."""
    total = 2 * matched + invented + missed
    return 2 * matched / total if total else math.nan


@dataclass(frozen=True)
class Counts:
    """Matched, invented and missed elements."""

    matched: int = 0
    invented: int = 0
    missed: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.matched + other.matched, self.invented + other.invented, self.missed + other.missed)

    @property
    def f1(self) -> float:
        return f1(self.matched, self.invented, self.missed)

    @property
    def precision(self) -> float:
        found = self.matched + self.invented
        return self.matched / found if found else math.nan

    @property
    def recall(self) -> float:
        truth = self.matched + self.missed
        return self.matched / truth if truth else math.nan


def baseline_distance(first: Points, second: Points, step: float = 1.0) -> float:
    """The mean of the two directed mean distances between two baselines, each sampled every `step` pixels."""
    if len(first) < 2 or len(second) < 2:
        return math.nan
    lines = LineString(first), LineString(second)

    def directed(source: LineString, target: LineString) -> float:
        at = np.append(np.arange(0.0, source.length, step), source.length)
        return float(np.mean(shapely.distance(shapely.line_interpolate_point(source, at), target)))

    return (directed(lines[0], lines[1]) + directed(lines[1], lines[0])) / 2


def line_height(polygon: Points, baseline: Points) -> float:
    """A line's height: its polygon's area over its baseline's length, the polygon's mean height along it."""
    found = shape(polygon)
    length = LineString(baseline).length if len(baseline) >= 2 else 0.0
    return found.area / length if found is not None and length > 0 else math.nan


def centroid(points: Points) -> tuple[float, float]:
    found = shape(points)
    if found is not None:
        return float(found.centroid.x), float(found.centroid.y)
    return (float(np.mean([x for x, _ in points])), float(np.mean([y for _, y in points]))) if points else (0.0, 0.0)


def inside_any(points: Sequence[tuple[float, float]], polygons: Sequence[Points]) -> np.ndarray:
    """Whether each point lies within any of the polygons."""
    shapes = [found for found in (shape(polygon) for polygon in polygons) if found is not None]
    if not shapes or not len(points):
        return np.zeros(len(points), dtype=bool)
    union = shapely.union_all(shapes)
    return np.asarray(shapely.contains_xy(union, [x for x, _ in points], [y for _, y in points]), dtype=bool)
