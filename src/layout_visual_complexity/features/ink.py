"""Ink distribution: how the ink lies on the folio and what colours it has, on the ink layer (`folio_image.py`), its
scales in line heights estimated from the page itself.

- Coverage: the ink's share of the folio.
- Lacunarity at 1, 2, 4, 8 and 16 line heights: for square boxes of that side glided over the crop one pixel at a
  time (those at least 95% on the folio), the mean of the squared ink mass of a box over its squared mean, 1 for
  ink spread evenly and larger as the ink gathers in clumps with gaps between them (Plotnick et al. 1996).
- Fragmentation: the number of ink masses (8-connected) left after closing the ink by a disc one line height across,
  per area of the folio in square line heights.
- Chromatic share: the share of the ink whose chroma, once the parchment's own colour is made neutral (each linear
  RGB channel divided by the paper's), is at least `chroma`, and whose hue lies outside `excluded_hues` (60–110,
  where brown writing reaches the threshold).
- Hue classes: the share of the ink in each of three classes of the chromatic ink's hue angle (CIELAB, in degrees):
  red (330–60), green (110–200) and blue (200–330); and how many classes cover at least `class_area` square line
  heights.
- Line height: the line height as a share of the folio crop's height, and whether it came from the text lines (1)
  or from the ink's vertical profile (0); recorded, not a measure of the ink.
"""

import math
from dataclasses import dataclass

import numpy as np

from layout_visual_complexity.features.contract import Family, Group, Need
from layout_visual_complexity.features.folio_image import InkLayerSettings, ink_layer
from layout_visual_complexity.features.ink_colour import (
    EXCLUDED_HUES,
    HUE_CLASSES,
    InkColour,
    chroma_words,
    chromatic,
    hue_class,
    hue_words,
    ink_colour,
)
from layout_visual_complexity.features.page import Page


@dataclass(frozen=True)
class InkSettings:
    """The ink layer and the ink's measures.

    Args:
        long_edge: The folio crop's long edge, in pixels.
        window: Sauvola's window, as a share of the crop's shorter edge.
        k: Sauvola's k.
        speck: Ink components smaller than this share of the folio's pixels are removed.
        scales: The lacunarity's box sides, in line heights.
        inside: A lacunarity box counts when at least this share of it lies on the folio.
        chroma: The least chroma (CIELAB, the parchment made neutral) of chromatic ink.
        excluded_hues: Hue angles (degrees, from the first to the second) whose ink isn't chromatic, whatever its
            chroma; none when empty.
        class_area: A hue class counts as present when its ink covers at least this many square line heights.
    """

    long_edge: int = InkLayerSettings.long_edge
    window: float = InkLayerSettings.window
    k: float = InkLayerSettings.k
    speck: float = InkLayerSettings.speck
    scales: tuple[int, ...] = (1, 2, 4, 8, 16)
    inside: float = 0.95
    chroma: float = 25.0
    excluded_hues: tuple[float, ...] = EXCLUDED_HUES
    class_area: float = 1.0


class InkDistribution(Family):
    """Coverage, lacunarity, fragmentation, chromatic share and hue classes of the ink."""

    name = "ink"
    scope = "page"
    version = "2"
    Settings = InkSettings

    @property
    def needs(self) -> frozenset[Need]:
        return frozenset({"image", "folio_crop", "ink", "lines"})

    def groups(self) -> list[Group]:
        settings: InkSettings = self.settings  # type: ignore[assignment]
        scales = ", ".join(str(scale) for scale in settings.scales)
        return [
            Group("Ink", "Coverage", "Ink__Coverage", ("share",), "Share of the folio covered by ink.",
                  f"the ink layer at {settings.long_edge:,} px", "Sauvola and Pietikäinen 2000"),
            Group("Ink", "Lacunarity", "Ink__Lacunarity", tuple(f"lh{scale}" for scale in settings.scales),
                  "Gaps in the ink: the mean squared ink mass of gliding square boxes over its squared mean.",
                  f"boxes {scales} line heights wide", "Plotnick et al. 1996"),
            Group("Ink", "Fragmentation", "Ink__Fragmentation", ("masses",),
                  "Ink masses after closing by one line height, per square line height of the folio.",
                  "a disc one line height across", "this study"),
            Group("Ink", "Chromatic share", "Ink__ChromaticShare", ("share",),
                  "Share of the ink whose chroma, the parchment's colour made neutral, reaches the threshold, its hue "
                  "outside the excluded range.",
                  chroma_words(settings.chroma, settings.excluded_hues), "this study"),
            Group("Ink", "Hue classes", "Ink__HueClasses", (*HUE_CLASSES, "count"),
                  "Shares of the ink in the chromatic hue classes, and the number of classes present.",
                  f"{hue_words(settings.excluded_hues)}; present from "
                  f"{settings.class_area:g} square line height{'s' if settings.class_area != 1 else ''}",
                  "this study"),
            Group("Ink", "Line height", "Ink__LineHeight", ("folio_share", "from_lines"),
                  "The line height as a share of the folio's height, and whether it came from the text lines.",
                  "the text lines' median spacing, else the ink profile's period", "this study"),
        ]  # fmt: skip

    def words(self, column: str) -> str:
        _, feature, flavour = column.split("__")
        if feature == "Lacunarity":
            return f"the lacunarity of the ink at boxes {flavour[2:]} line heights wide"
        if feature == "HueClasses":
            if flavour == "count":
                return "the number of hue classes of chromatic ink present"
            return f"the share of the ink in the {flavour} hue class"
        return {
            "Ink__Coverage__share": "the share of the folio covered by ink",
            "Ink__Fragmentation__masses": "the ink masses after closing by one line height, per square line height",
            "Ink__ChromaticShare__share": "the chromatic share of the ink against the parchment",
            "Ink__LineHeight__folio_share": "the line height as a share of the folio's height",
            "Ink__LineHeight__from_lines": "whether the line height came from the text lines",
        }.get(column) or super().words(column)

    def prepare(self, page: Page) -> None:
        ink_layer(page, self.settings)

    def measure(self, page: Page) -> list[dict[str, float]]:
        settings: InkSettings = self.settings  # type: ignore[assignment]
        layer = ink_layer(page, settings)
        inside = layer.folio.inside
        height = layer.line_height
        row = {
            "Ink__Coverage__share": layer.coverage,
            "Ink__LineHeight__folio_share": height / inside.shape[0] if math.isfinite(height) else math.nan,
            "Ink__LineHeight__from_lines": {"lines": 1.0, "profile": 0.0}.get(layer.line_height_source, math.nan),
        }
        for scale in settings.scales:
            side = round(scale * height) if math.isfinite(height) else 0
            row[f"Ink__Lacunarity__lh{scale}"] = lacunarity(layer.ink, inside, side, settings.inside)
        row["Ink__Fragmentation__masses"] = fragmentation(layer.ink, inside, height)
        row |= ink_colours(ink_colour(page, settings), settings.chroma, settings.excluded_hues, settings.class_area)
        return [row]


def box_sums(values: np.ndarray, side: int) -> np.ndarray:
    """The sum of `values` in every `side` × `side` box within the array, by an integral image."""
    integral = np.zeros((values.shape[0] + 1, values.shape[1] + 1))
    integral[1:, 1:] = values.cumsum(axis=0).cumsum(axis=1)
    return integral[side:, side:] - integral[:-side, side:] - integral[side:, :-side] + integral[:-side, :-side]


def lacunarity(ink: np.ndarray, inside: np.ndarray, side: int, least_inside: float = 0.95) -> float:
    """The gliding-box lacunarity of the ink at boxes `side` pixels wide: the mean of the squared ink mass of the
    boxes at least `least_inside` on the folio, over the square of its mean; NaN without such boxes or ink."""
    if side < 1 or side > min(ink.shape):
        return math.nan
    kept = box_sums(inside.astype(np.float64), side) >= least_inside * side * side
    if not kept.any():
        return math.nan
    mass = box_sums(ink.astype(np.float64), side)[kept]
    mean = mass.mean()
    return float((mass**2).mean() / mean**2) if mean > 0 else math.nan


def fragmentation(ink: np.ndarray, inside: np.ndarray, line_height: float) -> float:
    """The ink masses (8-connected) after closing by a disc one line height across, per square line height of the
    folio; NaN without a line height."""
    import cv2

    if not math.isfinite(line_height) or line_height < 1 or not inside.any():
        return math.nan
    diameter = max(1, round(line_height))
    diameter += 1 - diameter % 2
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    closed = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, disc) & inside
    count, _ = cv2.connectedComponents(closed.astype(np.uint8), connectivity=8)
    return float((count - 1) / (inside.sum() / line_height**2))


def ink_colours(colour: InkColour, chroma: float, excluded: tuple[float, ...], class_area: float) -> dict[str, float]:
    """The ink's chromatic share against the parchment, its shares in the hue classes and how many are present."""
    names = ["Ink__ChromaticShare__share", *(f"Ink__HueClasses__{name}" for name in HUE_CLASSES),
             "Ink__HueClasses__count"]  # fmt: skip
    layer = colour.layer
    ink = layer.ink
    total = int(ink.sum())
    if total == 0:
        return dict.fromkeys(names, math.nan)
    hue = colour.hue[ink]
    is_chromatic = chromatic(colour.chroma[ink], hue, chroma, excluded)
    found = {"Ink__ChromaticShare__share": float(is_chromatic.sum() / total)}
    least = class_area * layer.line_height**2 if math.isfinite(layer.line_height) else math.inf
    present = 0
    for name, bounds in HUE_CLASSES.items():
        pixels = int((is_chromatic & hue_class(hue, bounds)).sum())
        found[f"Ink__HueClasses__{name}"] = pixels / total
        present += pixels >= least
    found["Ink__HueClasses__count"] = float(present) if math.isfinite(least) else math.nan
    return found
