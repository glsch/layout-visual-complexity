"""The colour of the ink, shared by the families that measure it (ink distribution, colour).

The parchment is made neutral: each linear RGB channel of the normalised crop is divided by the paper's (the median
of the folio's pixels that aren't ink), then converted to CIELAB. An ink pixel's chroma is the length of its (a*, b*)
and its hue the angle, in degrees from 0 to 360. Ink of chroma at least a threshold is chromatic unless its hue lies in
an excluded range, by default 60–110: yellow, gold and olive, where brown iron-gall writing reaches the threshold
against the parchment. The hue of chromatic ink puts it in one of three classes: red (330–60), green (110–200) and
blue (200–330).

The chroma and hue are made once per page and ink layer and shared by the families with the same ink layer, so their
shared measures give the same values.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from layout_visual_complexity.features.folio_image import InkLayer, InkLayerSettings, ink_layer
from layout_visual_complexity.features.page import Page

HUE_CLASSES = {"red": (330.0, 60.0), "green": (110.0, 200.0), "blue": (200.0, 330.0)}
# Hues whose ink isn't chromatic, whatever its chroma: on most pages, ink of these hues is brown writing.
EXCLUDED_HUES = (60.0, 110.0)


@dataclass
class InkColour:
    """The ink layer with each crop pixel's chroma and hue angle (degrees) against the neutral parchment."""

    layer: InkLayer
    chroma: np.ndarray
    hue: np.ndarray

    def chromatic(self, chroma: float, excluded: tuple[float, ...] = EXCLUDED_HUES) -> np.ndarray:
        """The chromatic ink: its chroma reaches `chroma`, its hue lies outside `excluded`."""
        return self.layer.ink & chromatic(self.chroma, self.hue, chroma, excluded)


def ink_colour(page: Page, settings: Any) -> InkColour:
    """The page's ink colours on the ink layer of a family's settings, made once per page and ink layer."""
    chosen = InkLayerSettings(settings.long_edge, settings.window, settings.k, settings.speck)
    return page.prepared(("ink_colour", chosen), lambda: make_ink_colour(ink_layer(page, settings)))


def make_ink_colour(layer: InkLayer) -> InkColour:
    lab = neutral_lab(layer.folio.rgb, layer.folio.inside & ~layer.ink)
    a, b = lab[..., 1], lab[..., 2]
    return InkColour(layer, np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360)


def neutral_lab(rgb: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """CIELAB of an RGB image whose paper (the pixels in `paper`) is made neutral: each linear RGB channel divided by
    the paper's median."""
    import cv2

    values = rgb.astype(np.float32) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    white = np.median(linear[paper], axis=0) if paper.any() else np.ones(3, np.float32)
    balanced = np.clip(linear / np.maximum(white, 1e-6), 0, 1).astype(np.float32)
    return cv2.cvtColor(balanced, cv2.COLOR_LRGB2Lab)


def hue_class(hue: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """Which hue angles (degrees, 0–360) lie in a class from its first bound to its second, round past 360."""
    start, end = bounds
    return (hue >= start) & (hue < end) if start < end else (hue >= start) | (hue < end)


def chromatic(chroma: np.ndarray, hue: np.ndarray, least: float, excluded: tuple[float, ...]) -> np.ndarray:
    """Which colours are chromatic: their chroma at least `least`, their hue outside the excluded range (its first
    bound to its second, round past 360; none when empty)."""
    found = chroma >= least
    return found & ~hue_class(hue, excluded) if excluded else found


def chroma_words(chroma: float, excluded: tuple[float, ...]) -> str:
    """What makes ink chromatic, in words."""
    return f"chroma at least {chroma:g}" + (f", hue outside {excluded[0]:g}°–{excluded[1]:g}°" if excluded else "")


def hue_words(excluded: tuple[float, ...]) -> str:
    """The hue classes' ranges and the excluded one, in words."""
    classes = ", ".join(f"{name} {start:g}°–{end:g}°" for name, (start, end) in HUE_CLASSES.items())
    return f"{classes}; hues {excluded[0]:g}°–{excluded[1]:g}° not chromatic" if excluded else classes
