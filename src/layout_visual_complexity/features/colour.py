"""Real colour: how much of the ink is coloured, in which hues, how saturated, and where on the page it sits, on the ink
layer (`folio_image.py`) with the parchment made neutral (`ink_colour.py`).

- Chromatic share: the share of the ink whose chroma reaches `chroma`.
- Hue classes: the share of the ink in each hue class of the chromatic ink (red, gold, green, blue), and how many
  classes cover at least `class_area` square line heights.
- Hue entropy: the entropy, in bits, of the chromatic ink's hue angles in `bins` bins over 360°; 0 without chromatic
  ink.
- Saturation: the chromatic ink's median chroma and its spread (the interquartile range).
- Zones: within each kind of zone, the chromatic share of its ink: main text (`MainZone`), margins (`MarginTextZone`),
  numbering and running titles (`NumberingZone`, `RunningTitleZone`), decoration (`DropCapitalZone`, `GraphicZone`),
  heading lines (YOLO's `HeadingLine`), and the rest of the folio, outside them all.
- Lines: each text line's chromatic share of the ink within its polygon; the share of rubric lines (at least `rubric`
  of their ink chromatic) and the spread of the lines' chromatic shares (their standard deviation).
- Coloured initials: the drop capitals and graphic zones at least `coloured` of whose ink is chromatic.

The ink within stamps (`StampZone`) is left out of every measure: the libraries' stamps are modern, often red or
blue, and not the manuscript's colour. The shared measures are the ink family's, on the same ink layer; they differ
only on pages with stamps.
"""

import math
from dataclasses import dataclass

import numpy as np

from layout_visual_complexity.features.contract import Family, Group, Need
from layout_visual_complexity.features.folio_image import InkLayerSettings
from layout_visual_complexity.features.ink_colour import HUE_CLASSES, InkColour, hue_class, ink_colour
from layout_visual_complexity.features.page import Page, polygon_mask

ZONE_KINDS = {
    "main": ("MainZone",),
    "margins": ("MarginTextZone",),
    "numbering": ("NumberingZone", "RunningTitleZone"),
    "decoration": ("DropCapitalZone", "GraphicZone"),
}
HEADING_LINES = ("HeadingLine",)
INITIALS = ("DropCapitalZone", "GraphicZone")
LEFT_OUT = ("StampZone",)
KIND_WORDS = {
    "main": "main text zones",
    "margins": "margin text zones",
    "numbering": "numbering and running titles",
    "decoration": "drop capitals and graphic zones",
    "headings": "heading lines",
    "rest": "the rest of the folio",
}


@dataclass(frozen=True)
class ColourSettings:
    """The ink layer and the colour measures.

    Args:
        long_edge: The folio crop's long edge, in pixels.
        window: Sauvola's window, as a share of the crop's shorter edge.
        k: Sauvola's k.
        speck: Ink components smaller than this share of the folio's pixels are removed.
        chroma: The least chroma (CIELAB, the parchment made neutral) of chromatic ink.
        class_area: A hue class counts as present when its ink covers at least this many square line heights.
        bins: The hue entropy's bins over 360°.
        rubric: A text line is a rubric line when at least this share of its ink is chromatic.
        coloured: A drop capital or graphic zone is a coloured initial when at least this share of its ink is
            chromatic.
    """

    long_edge: int = InkLayerSettings.long_edge
    window: float = InkLayerSettings.window
    k: float = InkLayerSettings.k
    speck: float = InkLayerSettings.speck
    chroma: float = 25.0
    class_area: float = 1.0
    bins: int = 24
    rubric: float = 0.5
    coloured: float = 0.5


class Colour(Family):
    """Chromatic share, hue classes, hue entropy, saturation, and where the colour sits: by zone, across the lines,
    in the initials."""

    name = "colour"
    scope = "page"
    version = "1"
    Settings = ColourSettings

    @property
    def needs(self) -> frozenset[Need]:
        return frozenset({"image", "folio_crop", "ink", "lines", "regions", "yolo_lines"})

    def groups(self) -> list[Group]:
        settings: ColourSettings = self.settings  # type: ignore[assignment]
        bins = 360 / settings.bins
        return [
            Group("Colour", "Chromatic share", "Colour__ChromaticShare", ("share",),
                  "Share of the ink whose chroma, the parchment's colour made neutral, reaches the threshold.",
                  f"chroma at least {settings.chroma:g}", "this study"),
            Group("Colour", "Hue classes", "Colour__HueClasses", (*HUE_CLASSES, "count"),
                  "Shares of the ink in the chromatic hue classes, and the number of classes present.",
                  "red 330°–60°, gold 60°–110°, green 110°–200°, blue 200°–330°; present from "
                  f"{settings.class_area:g} square line height{'s' if settings.class_area != 1 else ''}",
                  "this study"),
            Group("Colour", "Hue entropy", "Colour__HueEntropy", ("bits",),
                  "Entropy of the chromatic ink's hue angles; 0 without chromatic ink.",
                  f"{settings.bins} bins of {bins:g}°", "this study"),
            Group("Colour", "Saturation", "Colour__Saturation", ("median", "spread"),
                  "Median chroma of the chromatic ink, and its interquartile range.",
                  "CIELAB chroma, the parchment made neutral", "this study"),
            Group("Colour", "Colour by zone", "Colour__Zones", (*ZONE_KINDS, "headings", "rest"),
                  "Chromatic share of the ink within each kind of zone: main text, margins, numbering and running "
                  "titles, decoration, heading lines, and the rest of the folio.",
                  "the regions model's zones and heading lines", "this study"),
            Group("Colour", "Colour across lines", "Colour__Lines", ("rubric_share", "spread"),
                  "Share of the text lines whose ink is mostly chromatic (rubric lines), and the standard deviation of "
                  "the lines' chromatic shares.",
                  f"a rubric line at least {settings.rubric:.0%} chromatic", "this study"),
            Group("Colour", "Coloured initials", "Colour__ColouredInitials", ("count",),
                  "Number of drop capitals and graphic zones whose ink is mostly chromatic.",
                  f"at least {settings.coloured:.0%} of the zone's ink chromatic", "this study"),
        ]  # fmt: skip

    def words(self, column: str) -> str:
        _, feature, flavour = column.split("__")
        if feature == "HueClasses":
            if flavour == "count":
                return "the number of hue classes of chromatic ink present"
            return f"the share of the ink in the {flavour} hue class"
        if feature == "Zones":
            return f"the chromatic share of the ink in {KIND_WORDS[flavour]}"
        return {
            "Colour__ChromaticShare__share": "the chromatic share of the ink against the parchment",
            "Colour__HueEntropy__bits": "the entropy of the chromatic ink's hues",
            "Colour__Saturation__median": "the chromatic ink's median chroma",
            "Colour__Saturation__spread": "the interquartile range of the chromatic ink's chroma",
            "Colour__Lines__rubric_share": "the share of rubric lines, mostly chromatic",
            "Colour__Lines__spread": "the standard deviation of the lines' chromatic shares",
            "Colour__ColouredInitials__count": "the number of coloured initials",
        }.get(column) or super().words(column)

    def prepare(self, page: Page) -> None:
        ink_colour(page, self.settings)

    def measure(self, page: Page) -> list[dict[str, float]]:
        settings: ColourSettings = self.settings  # type: ignore[assignment]
        colour, ink, chromatic = colour_layer(page, settings)
        frame, size = page.folio_bounds, ink.shape
        row = hue_measures(colour, ink, chromatic, settings)
        row |= zone_shares(page, ink, chromatic, frame, size)
        row |= line_shares(page, ink, chromatic, frame, settings.rubric)
        row["Colour__ColouredInitials__count"] = coloured_initials(page, ink, chromatic, frame, settings.coloured)
        return [row]


def hue_measures(colour: InkColour, ink: np.ndarray, chromatic: np.ndarray, settings: ColourSettings) -> dict:
    """The chromatic share, the hue classes' shares and number, the hue entropy and the saturation."""
    total = int(ink.sum())
    names = ["Colour__ChromaticShare__share", *(f"Colour__HueClasses__{name}" for name in HUE_CLASSES),
             "Colour__HueClasses__count", "Colour__HueEntropy__bits", "Colour__Saturation__median",
             "Colour__Saturation__spread"]  # fmt: skip
    if total == 0:
        return dict.fromkeys(names, math.nan)
    hue, chroma = colour.hue[chromatic], colour.chroma[chromatic]
    found = {"Colour__ChromaticShare__share": float(len(hue) / total)}
    height = colour.layer.line_height
    least = settings.class_area * height**2 if math.isfinite(height) else math.inf
    present = 0
    for name, bounds in HUE_CLASSES.items():
        pixels = int(hue_class(hue, bounds).sum())
        found[f"Colour__HueClasses__{name}"] = pixels / total
        present += pixels >= least
    found["Colour__HueClasses__count"] = float(present) if math.isfinite(least) else math.nan
    found["Colour__HueEntropy__bits"] = hue_entropy(hue, settings.bins)
    if len(chroma):
        low, middle, high = np.percentile(chroma, [25, 50, 75])
        found |= {"Colour__Saturation__median": float(middle), "Colour__Saturation__spread": float(high - low)}
    else:
        found |= {"Colour__Saturation__median": math.nan, "Colour__Saturation__spread": math.nan}
    return found


def hue_entropy(hue: np.ndarray, bins: int) -> float:
    """The entropy, in bits, of hue angles (degrees) in `bins` equal bins over 360°; 0 without any."""
    if not len(hue):
        return 0.0
    counts = np.bincount(np.minimum((np.asarray(hue) * bins / 360).astype(int), bins - 1), minlength=bins)
    shares = counts[counts > 0] / counts.sum()
    return float(-(shares * np.log2(shares)).sum())


def window(polygon: np.ndarray, frame: tuple[float, float, float, float], size: tuple[int, int]) -> tuple | None:
    """The crop's cells around a polygon (in the full image's pixels) and which of them lie in it: (rows, columns, mask)
    with the rows and columns as slices of the crop; None when it misses the crop."""
    if len(polygon) < 3:
        return None
    x0, y0, x1, y1 = frame
    rows, columns = size
    across, down = columns / max(x1 - x0, 1e-9), rows / max(y1 - y0, 1e-9)
    left = max(0, math.floor((polygon[:, 0].min() - x0) * across))
    right = min(columns, math.ceil((polygon[:, 0].max() - x0) * across) + 1)
    top = max(0, math.floor((polygon[:, 1].min() - y0) * down))
    bottom = min(rows, math.ceil((polygon[:, 1].max() - y0) * down) + 1)
    if right <= left or bottom <= top:
        return None
    local = (x0 + left / across, y0 + top / down, x0 + right / across, y0 + bottom / down)
    mask = polygon_mask(polygon, local, (bottom - top, right - left))
    return slice(top, bottom), slice(left, right), mask


def zone_mask(page: Page, classes: tuple[str, ...], frame: tuple, size: tuple[int, int]) -> np.ndarray:
    """The crop's cells within any zone of the classes (YOLO's lines too)."""
    mask = np.zeros(size, bool)
    for zone in [*page.regions, *page.yolo_lines]:
        if zone.cls in classes:
            found = window(zone.polygon, frame, size)
            if found is not None:
                rows, columns, inside = found
                mask[rows, columns] |= inside
    return mask


def zone_shares(page: Page, ink: np.ndarray, chromatic: np.ndarray, frame: tuple, size: tuple) -> dict[str, float]:
    """Within each kind of zone, the chromatic share of its ink; NaN where the page has no ink in that kind."""
    kinds = {kind: zone_mask(page, classes, frame, size) for kind, classes in ZONE_KINDS.items()}
    kinds["headings"] = zone_mask(page, HEADING_LINES, frame, size)
    kinds["rest"] = ~np.logical_or.reduce(list(kinds.values()))
    found = {}
    for kind, mask in kinds.items():
        pixels = int((ink & mask).sum())
        found[f"Colour__Zones__{kind}"] = float((chromatic & mask).sum() / pixels) if pixels else math.nan
    return found


def chromatic_share(polygon: np.ndarray, ink: np.ndarray, chromatic: np.ndarray, frame: tuple) -> float:
    """The chromatic share of the ink within a polygon; NaN where it holds no ink."""
    found = window(polygon, frame, ink.shape)
    if found is None:
        return math.nan
    rows, columns, inside = found
    pixels = int((ink[rows, columns] & inside).sum())
    return float((chromatic[rows, columns] & inside).sum() / pixels) if pixels else math.nan


def line_shares(page: Page, ink: np.ndarray, chromatic: np.ndarray, frame: tuple, rubric: float) -> dict:
    """The share of the folio's text lines whose ink is at least `rubric` chromatic, and the standard deviation of the
    lines' chromatic shares, over the lines with ink."""
    shares = np.array([chromatic_share(page.lines[number].polygon, ink, chromatic, frame) for number in
                       page.folio_lines])  # fmt: skip
    shares = shares[np.isfinite(shares)]
    if not len(shares):
        return {"Colour__Lines__rubric_share": math.nan, "Colour__Lines__spread": math.nan}
    return {"Colour__Lines__rubric_share": float((shares >= rubric).mean()),
            "Colour__Lines__spread": float(shares.std())}  # fmt: skip


def coloured_initials(page: Page, ink: np.ndarray, chromatic: np.ndarray, frame: tuple, coloured: float) -> float:
    """The drop capitals and graphic zones at least `coloured` of whose ink is chromatic."""
    shares = [chromatic_share(zone.polygon, ink, chromatic, frame) for zone in page.regions if zone.cls in INITIALS]
    return float(sum(share >= coloured for share in shares if math.isfinite(share)))


def colour_layer(page: Page, settings: ColourSettings) -> tuple[InkColour, np.ndarray, np.ndarray]:
    """The page's ink colours, its ink with the stamps left out, and its chromatic ink: what the measures and a
    drawing of them start from."""
    colour = ink_colour(page, settings)
    stamps = zone_mask(page, LEFT_OUT, page.folio_bounds, colour.layer.ink.shape)
    ink = colour.layer.ink & ~stamps
    return colour, ink, ink & (colour.chroma >= settings.chroma)
