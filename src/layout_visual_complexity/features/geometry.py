"""The geometric features: 83 columns in nine groups, from the folio, the zones and the text lines.

Area features describe the folio's written space: the union of its lines' polygons, against the folio and its own
convex hull; against a rectangle of the same area and the folio's proportions; the similarity of the lines' polygons;
and the graphic and drop-capital zones. They are measured once per page, on the lines whose baseline's midpoint
lies on the folio.

Line features describe the lines of each text zone (`MainZone` and `MarginTextZone` by default), in the order the
segmenter gives them: their lengths, the distances between consecutive baselines, the zigzag of their ends, and their
angles. They are measured per zone and taken to the page by the median. A zone needs three lines, once the lines
whose baseline is shorter than 7.5% of the zone's width are left out; when no zone has them, the folio's lines are
measured together, in a zone around them.

Lengths and offsets are relative to the zone's width, interlinear distances to the folio's diagonal, areas to the
folio's area; angles are in degrees, counter-clockwise, and similarities, proportions and counts as they are.
Standard deviations are those of a sample, except for the line similarities'. "Range" (`max_dev`) is the maximum
minus the minimum.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.affinity import translate
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from layout_visual_complexity.features.contract import Aggregation, Family, Group, Need
from layout_visual_complexity.features.page import Page, TextLine, Zone, shape

STATISTICS = ("std", "mean", "median", "max_dev")
DISTANCES = ("min_gap", "max_gap", "mean_gap", "midpoint", "percentile25", "projection_gap")
ZONE_MEASURES = ("total_area_ratio", "num_isolated_instances", "area_ratio_mean", "area_ratio_min", "area_ratio_max")
ZONE_MEASURES += ("area_ratio_std", "area_ratio_max_dev")
SIMILARITY_STATISTICS = ("mean", "median", "std", "max_dev")
SHIFTS = ("max_possible", "coord_shift", "zigzag", "zigzag_prop", "coord_shift_prop")


@dataclass(frozen=True)
class GeometrySettings:
    """The geometric features.

    Args:
        text_zones: The zone classes whose lines the line features measure.
        graphic_zones: Zones whose class contains this are graphic zones.
        drop_capital_zones: Zones whose class contains this are drop capitals.
        min_lines: Lines a measurement needs: the folio's lines for the area features, a zone's lines (once the short
            ones are left out) for the line features.
        short_line: Lines whose baseline is shorter than this share of their zone's width are left out of the line
            features.
        zigzag_tolerance: A shift of a line's end counts when it's larger than this share of the zone's width.
        interpolation_points: Points each baseline is interpolated to for the interlinear distances.
        zone_on_folio: A zone is the folio's when its centroid lies on the folio and at least this share of its area.
        fallback_zone: When no text zone has enough lines, measure the line features on all the folio's lines, in a
            zone around them.
    """

    text_zones: tuple[str, ...] = ("MainZone", "MarginTextZone")
    graphic_zones: str = "GraphicZone"
    drop_capital_zones: str = "DropCapital"
    min_lines: int = 3
    short_line: float = 0.075
    zigzag_tolerance: float = 0.005
    interpolation_points: int = 50
    zone_on_folio: float = 0.7
    fallback_zone: bool = True


GROUPS = [
    Group(
        "Area",
        "Written space",
        "AreaFeature__WrittenSpaceRatio",
        ("folio_ratio", "convexity"),
        "Line-polygon union relative to folio area; union relative to its convex hull.",
    ),
    Group(
        "Area",
        "Rectangularity",
        "AreaFeature__WrittenSpaceRectangularity",
        ("outside_ratio", "perimeter_efficiency"),
        "Area outside an equal-area ideal rectangle; ideal-rectangle perimeter relative to convex-hull perimeter.",
    ),
    Group(
        "Area",
        "Line similarity",
        "AreaFeature__LineSimilarity",
        tuple(f"{measure}_{statistic}" for measure in ("iou", "dice") for statistic in SIMILARITY_STATISTICS),
        "Pairwise IoU and Dice similarity of centroid-aligned line polygons, each by mean, median, SD and range.",
    ),
    Group(
        "Area",
        "Graphic zones",
        "AreaFeature__GraphicZoneArea",
        tuple(f"graphic_{measure}" for measure in ZONE_MEASURES),
        "Total area ratio, component count, and mean, minimum, maximum, SD and range of component-area ratios.",
    ),
    Group(
        "Area",
        "Drop capitals",
        "AreaFeature__DropCapitalArea",
        tuple(f"drop_capital_{measure}" for measure in ZONE_MEASURES),
        "The same seven quantities for drop-capital zones.",
    ),
    Group(
        "Line",
        "Line length",
        "LineFeature__LineLength",
        ("mean", "median", "min", "max", "max_dev", "std", "var_coef"),
        "Baseline length relative to zone width: mean, median, minimum, maximum, SD, range and coefficient of "
        "variation.",
    ),
    Group(
        "Line",
        "Interlinear distance",
        "LineFeature__InterlinearDistance",
        tuple(f"{distance}_{statistic}" for distance in DISTANCES for statistic in STATISTICS),
        "Six distance definitions, each by mean, median, SD and range over consecutive lines.",
    ),
    Group(
        "Line",
        "Zigzag score",
        "LineFeature__ZigzagScore",
        tuple(f"{statistic}_{side}" for side in ("left", "right") for statistic in STATISTICS)
        + tuple(f"{shift}_{side}" for side in ("left", "right") for shift in SHIFTS),
        "Left- and right-margin offsets, significant shifts, direction reversals, and their normalised proportions.",
    ),
    Group(
        "Line",
        "Parallelism",
        "LineFeature__LineParallelism",
        tuple(f"angle_{statistic}" for statistic in ("mean", "median", "std", "max_dev"))
        + tuple(f"pairwise_angle_diff_{statistic}" for statistic in ("mean", "median", "std", "max_dev")),
        "Line angles and absolute pairwise angle differences, each by mean, median, SD and range.",
    ),
]
ADDITIVE = {
    "AreaFeature__GraphicZoneArea__graphic_total_area_ratio",
    "AreaFeature__GraphicZoneArea__graphic_num_isolated_instances",
    "AreaFeature__DropCapitalArea__drop_capital_total_area_ratio",
    "AreaFeature__DropCapitalArea__drop_capital_num_isolated_instances",
}


class Geometry(Family):
    """The geometric features of the folio's written space and of its text zones' lines."""

    name = "geometry"
    scope = "region"
    version = "1"
    Settings = GeometrySettings

    @property
    def needs(self) -> frozenset[Need]:
        return frozenset({"regions", "lines"})

    def groups(self) -> list[Group]:
        return GROUPS

    def aggregation(self, column: str) -> Aggregation:
        return "sum" if column in ADDITIVE else "median"

    def words(self, column: str) -> str:
        return column_words(column)

    def measure(self, page: Page) -> list[dict[str, float]]:
        settings: GeometrySettings = self.settings  # type: ignore[assignment]
        rows = []
        if len(page.folio_lines) >= settings.min_lines:
            zones = page.zones_on_folio(page.regions, settings.zone_on_folio)
            rows.append(area_features(page, page.folio_lines, zones, settings))
        for zone in measured_zones(page, settings):
            rows.append(line_features(zone.lines, zone.left, zone.right, folio_diagonal(page), settings))
        return rows


@dataclass
class MeasuredZone:
    """A text zone whose lines the line features measure: its left and right edges, its lines once the short ones
    are left out, and its zone (None for the zone around the folio's lines)."""

    zone: Zone | None
    left: float
    right: float
    lines: list[TextLine]


def kept_lines(lines: list[TextLine], left: float, right: float, settings: GeometrySettings) -> list[TextLine] | None:
    """A zone's lines without the short ones, or None when too few remain."""
    width = right - left
    if width <= 0:
        return None
    kept = [line for line in lines if polyline_length(line.baseline) / width >= settings.short_line]
    return kept if len(kept) >= settings.min_lines else None


def measured_zones(page: Page, settings: GeometrySettings) -> list[MeasuredZone]:
    """The text zones on the folio with enough lines, each with its lines in the store's order; when there is none,
    the folio's lines in a zone around them."""
    found = []
    for zone in page.zones_on_folio(page.regions, settings.zone_on_folio):
        if zone.cls not in settings.text_zones:
            continue
        x0, _, x1, _ = bounds(zone.polygon)
        lines = kept_lines([line for line in page.lines if line.region == zone.id], x0, x1, settings)
        if lines:
            found.append(MeasuredZone(zone, x0, x1, lines))
    if not found and settings.fallback_zone and len(page.folio_lines) >= settings.min_lines:
        everything = [page.lines[number] for number in page.folio_lines]
        x0, x1 = fallback_zone(everything, page.folio_bounds)
        lines = kept_lines(everything, x0, x1, settings)
        if lines:
            found.append(MeasuredZone(None, x0, x1, lines))
    return found


# Columns in words.

STATISTIC_WORDS = {
    "mean": "mean",
    "median": "median",
    "std": "standard deviation",
    "max_dev": "range",
    "min": "minimum",
    "max": "maximum",
    "var_coef": "coefficient of variation",
}
DISTANCE_WORDS = {
    "min_gap": "the shortest distance",
    "max_gap": "the longest distance",
    "mean_gap": "the mean vertical distance",
    "midpoint": "the vertical distance of the mean points",
    "percentile25": "the 25th percentile of the vertical distances",
    "projection_gap": "the slope-adjusted vertical distance",
}
SHIFT_WORDS = {
    "max_possible": "the number of possible shifts (lines less two) at the {side} margin",
    "coord_shift": "the number of reversals of the {side} line ends' shifts",
    "zigzag": "the number of significant shifts of the {side} line ends",
    "zigzag_prop": "the share of possible shifts of the {side} line ends that are significant",
    "coord_shift_prop": "the share of possible shifts of the {side} line ends that reverse",
}


def column_words(column: str) -> str:
    """A geometric column in plain words."""
    _, feature, flavour = column.split("__")
    if feature == "WrittenSpaceRatio":
        return {
            "folio_ratio": "the area of the union of the line polygons relative to the folio's",
            "convexity": "the area of the union of the line polygons relative to its convex hull's",
        }[flavour]
    if feature == "WrittenSpaceRectangularity":
        return {
            "outside_ratio": "the share of the written space outside a rectangle of its area and the folio's "
            "proportions",
            "perimeter_efficiency": "the perimeter of a rectangle of the written space's area and the folio's "
            "proportions relative to the written space's convex hull",
        }[flavour]
    if feature == "LineSimilarity":
        measure, _, statistic = flavour.partition("_")
        name = "IoU" if measure == "iou" else "Dice"
        return f"the {STATISTIC_WORDS[statistic]} {name} similarity of two line polygons centred on each other"
    if feature in ("GraphicZoneArea", "DropCapitalArea"):
        zones = "graphic zones" if feature == "GraphicZoneArea" else "drop capitals"
        name = flavour.split("_", 1)[1] if feature == "GraphicZoneArea" else flavour.removeprefix("drop_capital_")
        if name == "total_area_ratio":
            return f"the {zones}' area relative to the folio's"
        if name == "num_isolated_instances":
            return f"the number of separate {zones}"
        statistic = name.removeprefix("area_ratio_")
        return f"the {STATISTIC_WORDS[statistic]} area of the {zones}, each relative to the folio's"
    if feature == "LineLength":
        return f"the {STATISTIC_WORDS[flavour]} of the baselines' lengths relative to their zone's width"
    if feature == "InterlinearDistance":
        distance = next(name for name in DISTANCE_WORDS if flavour.startswith(f"{name}_"))
        statistic = flavour.removeprefix(f"{distance}_")
        return f"the {STATISTIC_WORDS[statistic]} of {DISTANCE_WORDS[distance]} between consecutive baselines"
    if feature == "ZigzagScore":
        name, _, side = flavour.rpartition("_")
        if name in SHIFT_WORDS:
            return SHIFT_WORDS[name].format(side=side)
        return f"the {STATISTIC_WORDS[name]} of the lines' {side} offsets from their zone's edge"
    if feature == "LineParallelism":
        if flavour.startswith("angle_"):
            return f"the {STATISTIC_WORDS[flavour.removeprefix('angle_')]} of the baselines' angles"
        statistic = flavour.removeprefix("pairwise_angle_diff_")
        return f"the {STATISTIC_WORDS[statistic]} of the absolute differences between two baselines' angles"
    raise ValueError(f"not a geometric column: {column}")


# Shapes and statistics.


def bounds(polygon: np.ndarray) -> tuple[float, float, float, float]:
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def folio_diagonal(page: Page) -> float:
    return math.hypot(*page.folio_size)


def fallback_zone(lines: list[TextLine], folio: tuple[float, float, float, float]) -> tuple[float, float]:
    """The left and right edges of a zone around the lines (their polygons, or baselines), within the folio."""
    found = np.vstack([line.polygon if len(line.polygon) else line.baseline for line in lines])
    left = max(float(found[:, 0].min()), float(math.floor(folio[0])))
    right = min(float(found[:, 0].max()), float(math.floor(folio[2])))
    return left, right


def sample_std(values: np.ndarray) -> float:
    """The sample standard deviation; zero for a single value."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def statistics(prefix: str, values: np.ndarray, suffix: str = "") -> dict[str, float]:
    """`std` (of a sample), `mean`, `median` and `max_dev` (the range) of the values, named `<prefix>std<suffix>`
    and so on."""
    return {
        f"{prefix}std{suffix}": sample_std(values),
        f"{prefix}mean{suffix}": float(np.mean(values)),
        f"{prefix}median{suffix}": float(np.median(values)),
        f"{prefix}max_dev{suffix}": float(np.max(values) - np.min(values)),
    }


def prefixed(group: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{group}__{name}": value for name, value in values.items()}


# Area features.


def area_features(
    page: Page, folio_lines: list[int], zones: list[Zone], settings: GeometrySettings
) -> dict[str, float]:
    """The folio's area features from its lines and zones."""
    polygons = [page.line_shapes[number] for number in folio_lines]
    polygons = [polygon for polygon in polygons if polygon is not None]
    corners = [page.lines[number].polygon for number in folio_lines]
    corners = [polygon for polygon, kept in zip(corners, page_kept(page, folio_lines), strict=True) if kept]
    folio_area = page.folio_shape.area
    width, height = page.folio_size
    row: dict[str, float] = {}
    row |= prefixed("AreaFeature__WrittenSpaceRatio", written_space(polygons, folio_area))
    row |= prefixed(
        "AreaFeature__WrittenSpaceRectangularity", rectangularity(polygons, corners, width / height if height else 1.0)
    )
    row |= prefixed("AreaFeature__LineSimilarity", line_similarity(polygons))
    graphic = [zone for zone in zones if settings.graphic_zones in zone.cls]
    capitals = [zone for zone in zones if settings.drop_capital_zones in zone.cls]
    row |= prefixed("AreaFeature__GraphicZoneArea", zone_areas("graphic_", graphic, folio_area))
    row |= prefixed("AreaFeature__DropCapitalArea", zone_areas("drop_capital_", capitals, folio_area))
    return row


def page_kept(page: Page, numbers: list[int]) -> list[bool]:
    return [page.line_shapes[number] is not None for number in numbers]


def written_space(polygons: list[BaseGeometry], folio_area: float) -> dict[str, float]:
    """The union's area against the folio's (`folio_ratio`) and against its convex hull's (`convexity`)."""
    if not polygons:
        return {"folio_ratio": math.nan, "convexity": math.nan}
    union = unary_union(polygons)
    hull = union.convex_hull.area
    return {
        "folio_ratio": union.area / folio_area if folio_area else 0.0,
        "convexity": union.area / hull if hull else 0.0,
    }


def ideal_rectangle(centre: tuple[float, float], area: float, ratio: float) -> Polygon:
    """A rectangle of `area` and width-to-height `ratio`, centred on `centre`."""
    height = math.sqrt(area / ratio)
    width = height * ratio
    x, y = centre
    return Polygon(
        [(x - width / 2, y - height / 2), (x + width / 2, y - height / 2), (x + width / 2, y + height / 2),
         (x - width / 2, y + height / 2)]
    )  # fmt: skip


def rectangularity(polygons: list[BaseGeometry], corners: list[np.ndarray], ratio: float) -> dict[str, float]:
    """How rectangular the written space is: the share of the union outside a rectangle of its area and the folio's
    proportions, centred on its centroid (`outside_ratio`); that rectangle's perimeter against the convex hull's of
    the lines' points (`perimeter_efficiency`)."""
    if not polygons:
        return {"outside_ratio": math.nan, "perimeter_efficiency": math.nan}
    union = unary_union(polygons)
    if union.area <= 0:
        return {"outside_ratio": math.nan, "perimeter_efficiency": math.nan}
    centroid = union.centroid
    rectangle = ideal_rectangle((centroid.x, centroid.y), union.area, ratio)
    hull = MultiPoint(np.vstack(corners)).convex_hull.length
    x0, y0, x1, y1 = rectangle.bounds
    return {
        "outside_ratio": union.difference(rectangle).area / union.area,
        "perimeter_efficiency": 2 * ((x1 - x0) + (y1 - y0)) / hull if hull else math.nan,
    }


def line_similarity(polygons: list[BaseGeometry]) -> dict[str, float]:
    """IoU and Dice of every pair of line polygons, each moved so their centroids meet: mean, median, range and
    standard deviation (of the whole set of pairs)."""
    if len(polygons) < 2:
        return {f"{measure}_{name}": math.nan for measure in ("iou", "dice") for name in SIMILARITY_STATISTICS}
    centred = np.array([translate(polygon, -polygon.centroid.x, -polygon.centroid.y) for polygon in polygons])
    areas = shapely.area(centred)
    first, second = np.triu_indices(len(centred), k=1)
    overlap = shapely.area(shapely.intersection(centred[first], centred[second]))
    total = areas[first] + areas[second]
    union = total - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
    dice = np.divide(2 * overlap, total, out=np.zeros_like(overlap), where=total > 0)
    row = {}
    for measure, values in (("iou", iou), ("dice", dice)):
        row[f"{measure}_mean"] = float(np.mean(values))
        row[f"{measure}_median"] = float(np.median(values))
        row[f"{measure}_max_dev"] = float(np.max(values) - np.min(values))
        row[f"{measure}_std"] = float(np.std(values))
    return row


def zone_areas(prefix: str, zones: list[Zone], folio_area: float) -> dict[str, float]:
    """The zones' union against the folio's area, its separate parts, and the parts' areas against the folio's."""
    shapes = [found for found in (shape(zone.polygon) for zone in zones) if found is not None]
    union = unary_union(shapes) if shapes else Polygon()
    parts = [part for part in getattr(union, "geoms", [union]) if part.area > 0]
    ratios = np.array([part.area / folio_area for part in parts]) if parts and folio_area else np.zeros(1)
    measures = {
        "total_area_ratio": union.area / folio_area if folio_area else 0.0,
        "num_isolated_instances": float(len(parts)),
        "area_ratio_mean": float(np.mean(ratios)),
        "area_ratio_min": float(np.min(ratios)),
        "area_ratio_max": float(np.max(ratios)),
        "area_ratio_std": sample_std(ratios),
        "area_ratio_max_dev": float(np.max(ratios) - np.min(ratios)),
    }
    return {f"{prefix}{name}": value for name, value in measures.items()}


# Line features.


def polyline_length(points: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(points, axis=0).T))) if len(points) > 1 else 0.0


def line_features(
    kept: list[TextLine], left: float, right: float, diagonal: float, settings: GeometrySettings
) -> dict[str, float]:
    """A zone's line features, from its lines once the short ones are left out."""
    width = right - left
    row: dict[str, float] = {}
    row |= prefixed("LineFeature__LineLength", line_lengths(kept, width))
    row |= prefixed(
        "LineFeature__InterlinearDistance", interlinear_distances(kept, diagonal, settings.interpolation_points)
    )
    row |= prefixed(
        "LineFeature__ZigzagScore", zigzag(kept, left, right, settings.short_line, settings.zigzag_tolerance)
    )
    row |= prefixed("LineFeature__LineParallelism", parallelism(kept))
    return row


def line_lengths(lines: list[TextLine], width: float) -> dict[str, float]:
    """The baselines' lengths against the zone's width."""
    lengths = np.array([polyline_length(line.baseline) / width for line in lines])
    mean, std = float(np.mean(lengths)), sample_std(lengths)
    return {
        "mean": mean,
        "median": float(np.median(lengths)),
        "min": float(np.min(lengths)),
        "max": float(np.max(lengths)),
        "max_dev": float(np.max(lengths) - np.min(lengths)),
        "std": std,
        "var_coef": std / mean if mean else 0.0,
    }


def interpolate_baseline(baseline: np.ndarray, count: int) -> np.ndarray:
    """`count` points along a baseline, evenly spaced in x from its leftmost to its rightmost point; a vertical one
    evenly in y. Points sharing an x are averaged."""
    x, y = baseline[:, 0], baseline[:, 1]
    if np.ptp(x) == 0:
        return np.column_stack([np.full(count, x[0]), np.linspace(y.min(), y.max(), count)])
    unique, inverse = np.unique(x, return_inverse=True)
    averaged = np.bincount(inverse, weights=y) / np.bincount(inverse)
    spread = np.linspace(unique[0], unique[-1], count)
    return np.column_stack([spread, np.interp(spread, unique, averaged)])


def slope(points: np.ndarray) -> float:
    """The least-squares slope of y on x; zero when x doesn't vary."""
    x, y = points[:, 0], points[:, 1]
    spread = np.sum((x - x.mean()) ** 2)
    return float(np.sum((x - x.mean()) * (y - y.mean())) / spread) if spread > 0 else 0.0


def gaps(upper: np.ndarray, lower: np.ndarray) -> dict[str, float]:
    """Six distances between two interpolated baselines."""
    pairwise = np.hypot(upper[:, None, 0] - lower[None, :, 0], upper[:, None, 1] - lower[None, :, 1])
    vertical = np.abs(upper[:, None, 1] - lower[None, :, 1])
    angle = math.atan((slope(upper) + slope(lower)) / 2)
    return {
        "min_gap": float(pairwise.min()),
        "max_gap": float(pairwise.max()),
        "mean_gap": float(np.mean(np.abs(upper[:, 1] - lower[:, 1]))),
        "midpoint": float(abs(upper[:, 1].mean() - lower[:, 1].mean())),
        "percentile25": float(np.percentile(vertical, 25)),
        "projection_gap": float(np.mean(np.abs((lower[:, 1] - upper[:, 1]) * math.cos(angle)))),
    }


def interlinear_distances(lines: list[TextLine], diagonal: float, count: int) -> dict[str, float]:
    """The distances between consecutive baselines, against the folio's diagonal: per definition, their standard
    deviation, mean, median and range."""
    found: dict[str, list[float]] = {distance: [] for distance in DISTANCES}
    interpolated = [interpolate_baseline(line.baseline / diagonal, count) for line in lines]
    for upper, lower in itertools.pairwise(interpolated):
        for name, value in gaps(upper, lower).items():
            found[name].append(value)
    row = {}
    for name in DISTANCES:
        row |= statistics(f"{name}_", np.array(found[name]))
    return row


def bounding_box(line: TextLine) -> tuple[float, float, float, float]:
    return bounds(line.polygon if len(line.polygon) else line.baseline)


def shifts(values: list[float], tolerance: float) -> dict[str, float]:
    """How a margin's offsets move from line to line: steps larger than `tolerance` are shifts; `zigzag` counts the
    shifts after the first, `coord_shift` those that reverse the previous shift's direction; both also against the
    most there can be (`max_possible`, the lines less two). NaN for fewer than three lines."""
    if len(values) < 3:
        return dict.fromkeys(SHIFTS, math.nan)
    reversals = significant = 0
    previous = None
    for before, after in itertools.pairwise(values):
        step = after - before
        if abs(step) <= tolerance:
            continue
        direction = 1 if step > 0 else -1
        if previous is not None:
            significant += 1
            reversals += direction != previous
        previous = direction
    possible = len(values) - 2
    return {
        "max_possible": float(possible),
        "coord_shift": float(reversals),
        "zigzag": float(significant),
        "zigzag_prop": significant / possible,
        "coord_shift_prop": reversals / possible,
    }


def zigzag(lines: list[TextLine], left: float, right: float, short: float, tolerance: float) -> dict[str, float]:
    """The lines' left and right offsets from the zone's edges, against its width (0 where a line touches the
    edge), their statistics, and their shifts. Lines whose box is narrower than `short` of the zone are left out."""
    width = right - left
    lefts, rights = [], []
    for line in lines:
        x0, _, x1, _ = bounding_box(line)
        if (x1 - x0) / width < short:
            continue
        lefts.append((x0 - left) / width)
        rights.append((right - x1) / width)
    row: dict[str, float] = {}
    for side, values in (("left", lefts), ("right", rights)):
        if values:
            row |= statistics("", np.array(values), f"_{side}")
        else:
            row |= {f"{statistic}_{side}": math.nan for statistic in STATISTICS}
        row |= {f"{name}_{side}": value for name, value in shifts(values, tolerance).items()}
    return row


def parallelism(lines: list[TextLine]) -> dict[str, float]:
    """Each baseline's angle, from its first point to its last, in degrees counter-clockwise (vertical ones left
    out), and the absolute differences between every two: their mean, median, standard deviation and range. Empty
    for fewer than two angles."""
    angles = []
    for line in lines:
        (x1, y1), (x2, y2) = line.baseline[0], line.baseline[-1]
        if x2 == x1:
            continue
        angles.append(-math.degrees(math.atan2(y2 - y1, x2 - x1)))
    if len(angles) < 2:
        return {}
    values = np.array(angles)
    first, second = np.triu_indices(len(values), k=1)
    differences = np.abs(values[first] - values[second])
    row = {}
    for prefix, found in (("angle_", values), ("pairwise_angle_diff_", differences)):
        row[f"{prefix}mean"] = float(np.mean(found))
        row[f"{prefix}median"] = float(np.median(found))
        row[f"{prefix}std"] = float(np.std(found, ddof=1)) if len(found) > 1 else math.nan
        row[f"{prefix}max_dev"] = float(np.max(found) - np.min(found))
    return row
