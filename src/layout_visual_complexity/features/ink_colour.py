"""The colour of the ink, shared by the families that measure it (ink distribution, colour).

The parchment is made neutral: each linear RGB channel of the normalised crop is divided by the paper's (the median
of the folio's pixels that aren't ink), then converted to CIELAB. An ink pixel's chroma is the length of its (a*, b*)
and its hue the angle, in degrees from 0 to 360; ink of chroma at least a threshold is chromatic, and its hue puts it
in one of four classes: red (330–60), gold (60–110), green (110–200) and blue (200–330).

The chroma and hue are made once per page and ink layer and shared by the families with the same ink layer, so their
shared measures give the same values.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from layout_visual_complexity.features.folio_image import InkLayer, InkLayerSettings, ink_layer
from layout_visual_complexity.features.page import Page

HUE_CLASSES = {"red": (330.0, 60.0), "gold": (60.0, 110.0), "green": (110.0, 200.0), "blue": (200.0, 330.0)}


@dataclass
class InkColour:
    """The ink layer with each crop pixel's chroma and hue angle (degrees) against the neutral parchment."""

    layer: InkLayer
    chroma: np.ndarray
    hue: np.ndarray

    def chromatic(self, chroma: float) -> np.ndarray:
        """The ink whose chroma reaches `chroma`."""
        return self.layer.ink & (self.chroma >= chroma)


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
