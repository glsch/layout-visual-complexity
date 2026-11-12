"""Layout: Kyle–Davidson's dimensions of scene complexity (clutter, order, entropy), adapted to the page, with its
blank space.

- Clutter: the number of zones on the folio, of their types, and of the types of YOLO's lines.
- Order, on a raster of the folio at a fixed resolution: mirror symmetry about the text block's vertical and horizontal
  axes; translational regularity, the highest peak of the raster's autocorrelation beyond its central lobe, along the
  vertical (the lines' rhythm) and the horizontal (columns alike); balance, the distance of the mass's centre from the
  text block's and from the folio's.
- Entropy: the Shannon entropy, in bits, of the zone types' shares of the zones' area.
- Blank space: its share of the folio, the largest empty rectangle, and the margins around the text block.

The order measures run on two layers, the flavours of their columns: `mass`, the written mass (the lines' polygons and
the decoration zones, from the segmentation), and `ink`, the pixels darker than Otsu's threshold of the folio's
lightness (from the image alone). The text block is the box holding the layer's mass, less `trim` of it at each side.
"""

import math
from dataclasses import dataclass

import numpy as np
from shapely.ops import unary_union

from layout_visual_complexity.features.contract import Family, FeatureError, Group, Need
from layout_visual_complexity.features.page import Page, polygon_coverage, raster_shape, shape

FLAVOURS = ("mass", "ink")
BLANK_SPACE = ("share", "largest_empty", "margin_top", "margin_bottom", "margin_left", "margin_right")


@dataclass(frozen=True)
class LayoutSettings:
    """The layout adaptation of Kyle–Davidson's dimensions.

    Args:
        flavours: The layers the order measures run on: `mass` (the segmentation's lines and decoration zones) and
            `ink` (Otsu's dark pixels of the folio, no segmentation needed). Clutter, entropy and blank space need
            `mass`.
        resolution: Cells of the rasters along the folio's longer side.
        supersample: Each cell is drawn from this many pixels along each side.
        mass_zones: Zone classes drawn into the written mass with the lines.
        trim: The share of the mass left outside the text block at each of its sides.
        empty_below: A cell is empty when less than this share of it is written.
        zone_on_folio: A zone is the folio's when its centroid lies on the folio and at least this share of its area.
    """

    flavours: tuple[str, ...] = FLAVOURS
    resolution: int = 256
    supersample: int = 4
    mass_zones: tuple[str, ...] = ("DropCapitalZone", "GraphicZone", "MusicZone")
    trim: float = 0.01
    empty_below: float = 0.5
    zone_on_folio: float = 0.7


class Layout(Family):
    """Clutter, order, entropy and blank space of the page's layout."""

    name = "layout"
    scope = "page"
    version = "1"
    Settings = LayoutSettings

    def __init__(self, settings: LayoutSettings | None = None) -> None:
        super().__init__(settings)
        unknown = set(self.settings.flavours) - set(FLAVOURS)
        if unknown or not self.settings.flavours:
            raise FeatureError(f"layout: flavours are {' and '.join(FLAVOURS)}, not {sorted(unknown) or 'none'}")

    @property
    def needs(self) -> frozenset[Need]:
        needs: set[Need] = set()
        if "mass" in self.settings.flavours:
            needs |= {"regions", "lines", "yolo_lines"}
        if "ink" in self.settings.flavours:
            needs |= {"image", "folio_crop"}
        return frozenset(needs)

    def groups(self) -> list[Group]:
        flavours = [flavour for flavour in FLAVOURS if flavour in self.settings.flavours]
        mass = "mass" in flavours
        groups = []
        if mass:
            groups.append(
                Group(
                    "Layout",
                    "Clutter",
                    "Layout__Clutter",
                    ("zones", "zone_types", "line_types"),
                    "Number of zones on the folio, of zone types, and of YOLO's line types.",
                )
            )
        groups += [
            Group(
                "Layout",
                "Symmetry",
                "Layout__Symmetry",
                tuple(f"{axis}_{flavour}" for flavour in flavours for axis in ("left_right", "top_bottom")),
                "Overlap of the written mass (or the ink) with its mirror image about the text block's vertical and "
                "horizontal axes.",
            ),
            Group(
                "Layout",
                "Regularity",
                "Layout__Regularity",
                tuple(f"{axis}_{flavour}" for flavour in flavours for axis in ("vertical", "horizontal")),
                "Highest autocorrelation peak of the mass beyond the central lobe, vertically and horizontally.",
            ),
            Group(
                "Layout",
                "Balance",
                "Layout__Balance",
                tuple(f"{centre}_{flavour}" for flavour in flavours for centre in ("text_block", "folio")),
                "Distance of the mass's centre from the text block's and the folio's centres, relative to the "
                "folio's diagonal.",
            ),
        ]
        if mass:
            groups += [
                Group(
                    "Layout",
                    "Entropy",
                    "Layout__Entropy",
                    ("zone_types",),
                    "Shannon entropy of the zone types' shares of the zones' area.",
                ),
                Group(
                    "Layout",
                    "Blank space",
                    "Layout__BlankSpace",
                    BLANK_SPACE,
                    "Unwritten share of the folio; largest empty rectangle; the margins around the text block.",
                ),
            ]
        return groups

    def words(self, column: str) -> str:
        _, feature, flavour = column.split("__")
        layer = "the written mass" if flavour.endswith("_mass") else "the ink"
        found = {
            ("Clutter", "zones"): "the number of zones on the folio",
            ("Clutter", "zone_types"): "the number of zone types on the folio",
            ("Clutter", "line_types"): "the number of YOLO's line types on the folio",
            ("Entropy", "zone_types"): "the entropy of the zone types' shares of the zones' area",
            ("BlankSpace", "share"): "the folio's unwritten share",
            ("BlankSpace", "largest_empty"): "the largest empty rectangle relative to the folio",
            ("BlankSpace", "margin_top"): "the top margin relative to the folio's height",
            ("BlankSpace", "margin_bottom"): "the bottom margin relative to the folio's height",
            ("BlankSpace", "margin_left"): "the left margin relative to the folio's width",
            ("BlankSpace", "margin_right"): "the right margin relative to the folio's width",
        }.get((feature, flavour))
        if found:
            return found
        axis = flavour.rsplit("_", 1)[0]
        if feature == "Symmetry":
            about = "vertical" if axis == "left_right" else "horizontal"
            return f"the mirror symmetry of {layer} about the text block's {about} axis"
        if feature == "Regularity":
            return f"the highest {axis} autocorrelation peak of {layer}"
        if feature == "Balance":
            centre = "the text block's" if axis == "text_block" else "the folio's"
            return f"the distance of {layer}'s centre from {centre} centre"
        return super().words(column)

    def measure(self, page: Page) -> list[dict[str, float]]:
        settings: LayoutSettings = self.settings  # type: ignore[assignment]
        frame, size, folio = folio_raster(page, settings)
        row: dict[str, float] = {}
        if "mass" in settings.flavours:
            zones = page.zones_on_folio(page.regions, settings.zone_on_folio)
            mass = mass_layer(page, settings)
            yolo_lines = page.zones_on_folio(page.yolo_lines, settings.zone_on_folio)
            row |= {
                "Layout__Clutter__zones": float(len(zones)),
                "Layout__Clutter__zone_types": float(len({zone.cls for zone in zones})),
                "Layout__Clutter__line_types": float(len({line.cls for line in yolo_lines})),
                "Layout__Entropy__zone_types": zone_type_entropy(zones),
            }
            row |= {f"Layout__BlankSpace__{name}": value for name, value in blank_space(mass, folio, settings).items()}
            row |= order(mass, folio, "mass", settings.trim)
        if "ink" in settings.flavours:
            row |= order(ink_layer(page, size, settings.supersample), folio, "ink", settings.trim)
        return [row]


def folio_raster(page: Page, settings: LayoutSettings) -> tuple[tuple[float, float, float, float], tuple[int, int],
                                                                np.ndarray]:  # fmt: skip
    """The rasters' frame (the folio's bounding box), their size, and which cells lie on the folio."""
    width, height = page.folio_size
    size = raster_shape(width, height, settings.resolution)
    frame = page.folio_bounds
    return frame, size, polygon_coverage([page.folio_points()], frame, size, settings.supersample) >= 0.5


def mass_layer(page: Page, settings: LayoutSettings) -> np.ndarray:
    """The share of each cell the written mass covers: the folio's lines and its decoration zones."""
    frame, size, folio = folio_raster(page, settings)
    zones = page.zones_on_folio(page.regions, settings.zone_on_folio)
    polygons = [page.lines[number].polygon for number in page.folio_lines]
    polygons += [zone.polygon for zone in zones if zone.cls in settings.mass_zones]
    return polygon_coverage(polygons, frame, size, settings.supersample) * folio


def ink_layer(page: Page, size: tuple[int, int], supersample: int) -> np.ndarray:
    """The share of each cell's pixels darker than Otsu's threshold of the folio's lightness (CIELAB L*)."""
    import cv2

    rows, columns = size
    image, inside = page.folio_image((rows * supersample, columns * supersample))
    lightness = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2LAB)[..., 0]
    if not inside.any():
        return np.zeros(size)
    threshold, _ = cv2.threshold(lightness[inside].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (lightness <= threshold) & inside
    return ink.reshape(rows, supersample, columns, supersample).mean(axis=(1, 3))


def zone_type_entropy(zones: list) -> float:
    """The Shannon entropy, in bits, of each zone type's share of the zones' area (each type's zones united); NaN
    without zones."""
    by_type: dict[str, list] = {}
    for zone in zones:
        found = shape(zone.polygon)
        if found is not None:
            by_type.setdefault(zone.cls, []).append(found)
    areas = np.array([unary_union(shapes).area for shapes in by_type.values()])
    if not len(areas) or areas.sum() <= 0:
        return math.nan
    shares = areas[areas > 0] / areas.sum()
    return float(-(shares * np.log2(shares)).sum()) + 0.0


def text_block(mass: np.ndarray, trim: float) -> tuple[int, int, int, int]:
    """The rows and columns (first and last, inclusive) holding the mass, less `trim` of it at each side; the same
    from either side, so a symmetric mass gives a symmetric box."""

    def span(profile: np.ndarray) -> tuple[int, int]:
        limit = trim * profile.sum()
        first = int(np.searchsorted(np.cumsum(profile), limit, side="right"))
        last = len(profile) - 1 - int(np.searchsorted(np.cumsum(profile[::-1]), limit, side="right"))
        return min(first, last), max(first, last)

    top, bottom = span(mass.sum(axis=1))
    left, right = span(mass.sum(axis=0))
    return top, bottom, left, right


def mirrored(raster: np.ndarray, axis: int, centre_twice: int) -> np.ndarray:
    """The raster reflected about the line at `centre_twice` / 2 (in cells) across `axis`; cells reflected from
    outside are empty."""
    count = raster.shape[axis]
    source = centre_twice - np.arange(count)
    valid = (source >= 0) & (source < count)
    reflected = np.zeros_like(raster)
    if axis == 0:
        reflected[valid] = raster[source[valid]]
    else:
        reflected[:, valid] = raster[:, source[valid]]
    return reflected


def mirror_symmetry(raster: np.ndarray, axis: int, centre_twice: int) -> float:
    """The overlap of the raster with its mirror image: the sum of their minima over the sum of their maxima, 1 for
    a mirror-symmetric raster."""
    reflected = mirrored(raster, axis, centre_twice)
    larger = np.maximum(raster, reflected).sum()
    return float(np.minimum(raster, reflected).sum() / larger) if larger > 0 else math.nan


def autocorrelation(raster: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """The raster's autocorrelation over the folio (its deviations from its mean there, zero outside), normalised to
    1 at no shift: rows are vertical shifts, columns horizontal ones."""
    rows, columns = raster.shape
    deviation = np.where(inside, raster - raster[inside].mean(), 0.0)
    spectrum = np.fft.rfft2(deviation, s=(2 * rows, 2 * columns))
    found = np.fft.irfft2(spectrum * np.conj(spectrum), s=(2 * rows, 2 * columns))[:rows, :columns]
    return found / found[0, 0] if found[0, 0] > 0 else np.full_like(found, math.nan)


def autocorrelation_peak(profile: np.ndarray) -> tuple[float, int | None]:
    """The highest point of an autocorrelation profile (from shift 0) beyond its central lobe, within half its length,
    and its shift: (0, None) when it never rises again."""
    found = profile[: len(profile) // 2 + 1]
    if not np.isfinite(found).all():
        return math.nan, None
    rising = np.flatnonzero(np.diff(found) > 0)
    if not len(rising):
        return 0.0, None
    lag = int(rising[0] + np.argmax(found[rising[0] :]))
    return max(0.0, float(found[lag])), lag


def order(raster: np.ndarray, inside: np.ndarray, flavour: str, trim: float) -> dict[str, float]:
    """Symmetry, regularity and balance of a layer."""
    names = [
        f"Layout__Symmetry__left_right_{flavour}",
        f"Layout__Symmetry__top_bottom_{flavour}",
        f"Layout__Regularity__vertical_{flavour}",
        f"Layout__Regularity__horizontal_{flavour}",
        f"Layout__Balance__text_block_{flavour}",
        f"Layout__Balance__folio_{flavour}",
    ]
    total = raster.sum()
    if total <= 0 or not inside.any():
        return dict.fromkeys(names, math.nan)
    top, bottom, left, right = text_block(raster, trim)
    correlation = autocorrelation(raster, inside)
    rows, columns = np.indices(raster.shape)
    centre = np.array([(rows * raster).sum(), (columns * raster).sum()]) / total
    block = np.array([(top + bottom) / 2, (left + right) / 2])
    folio = np.array([rows[inside].mean(), columns[inside].mean()])
    diagonal = math.hypot(*raster.shape)
    values = [
        mirror_symmetry(raster, 1, left + right),
        mirror_symmetry(raster, 0, top + bottom),
        autocorrelation_peak(correlation[:, 0])[0],
        autocorrelation_peak(correlation[0, :])[0],
        float(np.linalg.norm(centre - block) / diagonal),
        float(np.linalg.norm(centre - folio) / diagonal),
    ]
    return dict(zip(names, values, strict=True))


def largest_rectangle(empty: np.ndarray) -> tuple[int, tuple[int, int, int, int] | None]:
    """The most cells an axis-aligned rectangle of empty cells holds, and that rectangle's first and last rows and
    columns (inclusive)."""
    heights = np.zeros(empty.shape[1], dtype=int)
    best, where = 0, None
    for number, row in enumerate(empty):
        heights = np.where(row, heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for position, height in enumerate([*heights.tolist(), 0]):
            start = position
            while stack and stack[-1][1] >= height:
                start, taller = stack.pop()
                if taller * (position - start) > best:
                    best = taller * (position - start)
                    where = (number - taller + 1, number, start, position - 1)
            stack.append((start, height))
    return best, where


def blank_space(mass: np.ndarray, inside: np.ndarray, settings: LayoutSettings) -> dict[str, float]:
    """The folio's unwritten share, its largest empty rectangle as a share of it, and the margins between the text
    block and the folio's edges, each against the folio's height or width."""
    cells = int(inside.sum())
    if cells == 0:
        return dict.fromkeys(BLANK_SPACE, math.nan)
    rows = np.flatnonzero(inside.any(axis=1))
    columns = np.flatnonzero(inside.any(axis=0))
    height, width = rows[-1] - rows[0] + 1, columns[-1] - columns[0] + 1
    found = {
        "share": 1 - float(mass[inside].sum()) / cells,
        "largest_empty": largest_rectangle((mass < settings.empty_below) & inside)[0] / cells,
    }
    if mass.sum() <= 0:
        return found | dict.fromkeys(BLANK_SPACE[2:], math.nan)
    top, bottom, left, right = text_block(mass, settings.trim)
    return found | {
        "margin_top": max(0, top - rows[0]) / height,
        "margin_bottom": max(0, rows[-1] - bottom) / height,
        "margin_left": max(0, left - columns[0]) / width,
        "margin_right": max(0, columns[-1] - right) / width,
    }
