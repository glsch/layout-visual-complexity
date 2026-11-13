"""The folio as the image families see it: its crop at a fixed size with the illumination flattened, and its ink
layer, with the line height that gives the ink's measures their scales.

- The crop: the folio's bounding box, resized (by area) so that its long edge is `long_edge` pixels, and the folio's
  mask within it.
- Flattening: the lightness (CIELAB L*) is divided by a smooth surface of the paper's brightness and brought back to
  the paper's median level; the colour (a*, b*) is kept. The surface is a polynomial of degree 2 fitted to the 90th
  percentile of the folio's lightness in each of 32 cells along the long edge, the cells much darker than the fit
  (pictures, shadows, dense writing) left out, three times over. Outside the folio the crop takes the paper's colour
  (the median of the brighter half of the folio), so that a measure over the whole crop sees paper there.
- The ink: Sauvola's threshold of the flattened lightness (L*/100), with a window of `window` of the crop's shorter
  edge (odd, at least 15 pixels) and `k`, inside the folio; components smaller than `speck` of the folio's pixels (at
  least 3 pixels) removed. The writing density of the article, on the flattened lightness.
- The line height: the median distance from each of the folio's text lines (their baselines) to the nearest line
  below it that overlaps it horizontally by half the shorter of the two, where the folio has at least three lines;
  otherwise the period of the ink's vertical profile, the highest peak of its autocorrelation beyond the central
  lobe. Each page records which way its line height came.

Both are made once per page and shared by the families that ask for them with the same settings.
"""

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from layout_visual_complexity.features.page import Page, raster_shape

FLATTEN_CELLS = 32
FLATTEN_DEGREE = 2
FLATTEN_PERCENTILE = 90
LineHeightSource = Literal["lines", "profile", "none"]


@dataclass
class NormalisedFolio:
    """The folio's crop at a fixed size: `rgb` (uint8) and `lab` (float32: L* 0–100, a*, b*) with the lightness
    flattened and the paper's colour outside the folio; `inside`, the folio's mask; `scale`, the crop's pixels per
    pixel of the full image; `paper`, the paper's L*, a*, b*."""

    rgb: np.ndarray
    lab: np.ndarray
    inside: np.ndarray
    scale: float
    paper: tuple[float, float, float]

    @property
    def grey(self) -> np.ndarray:
        """The flattened lightness, L*/100, in [0, 1]."""
        return self.lab[..., 0] / 100.0


@dataclass(frozen=True)
class InkLayerSettings:
    """The ink layer: the crop's long edge, Sauvola's window (a share of the crop's shorter edge) and k, and the
    smallest component kept (a share of the folio's pixels)."""

    long_edge: int = 1400
    window: float = 0.065
    k: float = 0.18
    speck: float = 5e-6


@dataclass
class InkLayer:
    """The ink inside the folio (`ink`, boolean, on the normalised crop) and the line height in the crop's pixels,
    with the way it came: from the text lines, from the ink's vertical profile, or none."""

    folio: NormalisedFolio
    ink: np.ndarray
    line_height: float
    line_height_source: LineHeightSource

    @property
    def coverage(self) -> float:
        inside = int(self.folio.inside.sum())
        return float(self.ink.sum() / inside) if inside else math.nan


def normalised_folio(page: Page, long_edge: int) -> NormalisedFolio:
    """The page's normalised folio crop at `long_edge` pixels, made once per page and size."""
    return page.prepared(("normalised_folio", long_edge), lambda: make_normalised_folio(page, long_edge))


def make_normalised_folio(page: Page, long_edge: int) -> NormalisedFolio:

    width, height = page.folio_size
    size = raster_shape(width, height, long_edge)
    image, inside = page.folio_image(size)
    return flattened(image, inside, size[1] / max(width, 1e-9))


def flattened(image: np.ndarray, inside: np.ndarray, scale: float = 1.0) -> NormalisedFolio:
    """An RGB crop with its lightness flattened and the paper's colour outside `inside`."""
    import cv2

    lab = cv2.cvtColor(np.ascontiguousarray(image).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    if not inside.any():
        inside = np.ones(inside.shape, bool)
    surface = paper_surface(lab[..., 0], inside)
    level = float(np.median(surface[inside]))
    lab[..., 0] = np.clip(lab[..., 0] * (level / np.maximum(surface, 1e-3)), 0, 100)
    lightness = lab[..., 0][inside]
    paper_pixels = inside & (lab[..., 0] >= np.median(lightness))
    paper = (level, float(np.median(lab[..., 1][paper_pixels])), float(np.median(lab[..., 2][paper_pixels])))
    lab[~inside] = paper
    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    rgb = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return NormalisedFolio(rgb, lab, inside, float(scale), paper)


def paper_surface(lightness: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """A smooth surface of the paper's lightness: a polynomial of degree 2 fitted to the 90th percentile of the
    folio's lightness in cells, the cells much darker than the fit left out."""
    rows, columns = lightness.shape
    cell = max(4, math.ceil(max(rows, columns) / FLATTEN_CELLS))
    padded_rows, padded_columns = -(-rows // cell) * cell, -(-columns // cell) * cell
    values = np.full((padded_rows, padded_columns), np.nan, np.float32)
    values[:rows, :columns] = np.where(inside, lightness, np.nan)
    blocks = values.reshape(padded_rows // cell, cell, padded_columns // cell, cell).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(blocks.shape[0], blocks.shape[1], -1)
    counts = np.isfinite(blocks).sum(axis=2)
    enough = counts >= 0.25 * cell * cell
    level = float(np.median(lightness[inside]))
    if enough.sum() < 2 * terms_count(FLATTEN_DEGREE):
        return np.full(lightness.shape, level, np.float32)
    found = np.nanpercentile(blocks[enough], FLATTEN_PERCENTILE, axis=1)
    cell_rows, cell_columns = np.nonzero(enough)
    y = (cell_rows + 0.5) * cell / rows
    x = (cell_columns + 0.5) * cell / columns
    design = polynomial_terms(x, y, FLATTEN_DEGREE)
    kept = np.ones(len(found), bool)
    for _ in range(3):
        weights, *_ = np.linalg.lstsq(design[kept], found[kept], rcond=None)
        residual = found - design @ weights
        spread = 1.4826 * np.median(np.abs(residual[kept] - np.median(residual[kept])))
        now = residual > -3 * max(spread, 0.5)
        if now.sum() < 2 * design.shape[1] or (now == kept).all():
            break
        kept = now
    x = ((np.arange(columns) + 0.5) / columns).astype(np.float32)
    y = ((np.arange(rows) + 0.5) / rows).astype(np.float32)
    powers = [(i, j) for i in range(FLATTEN_DEGREE + 1) for j in range(FLATTEN_DEGREE + 1 - i)]
    surface = np.zeros((rows, columns), np.float32)
    for weight, (i, j) in zip(weights, powers, strict=True):
        surface += np.float32(weight) * np.outer(y**j, x**i)
    return np.clip(surface, 1.0, 100.0)


def terms_count(degree: int) -> int:
    return (degree + 1) * (degree + 2) // 2


def polynomial_terms(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """The terms x^i y^j with i + j ≤ `degree`, one column each."""
    return np.column_stack([x**i * y**j for i in range(degree + 1) for j in range(degree + 1 - i)])


# The ink layer.


def ink_layer(page: Page, settings: Any) -> InkLayer:
    """The page's ink layer under a family's settings (their `long_edge`, `window`, `k` and `speck`), made once per
    page and shared by the families with the same ones."""
    chosen = InkLayerSettings(settings.long_edge, settings.window, settings.k, settings.speck)
    return page.prepared(("ink_layer", chosen), lambda: make_ink_layer(page, chosen))


def make_ink_layer(page: Page, settings: InkLayerSettings) -> InkLayer:
    folio = normalised_folio(page, settings.long_edge)
    ink = ink_mask(folio, settings)
    height, source = line_height(page, folio, ink)
    return InkLayer(folio, ink, height, source)


def ink_mask(folio: NormalisedFolio, settings: InkLayerSettings) -> np.ndarray:
    """Sauvola's dark pixels of the flattened lightness inside the folio, specks removed."""
    import cv2
    from skimage.filters import threshold_sauvola

    grey = folio.grey.astype(np.float64)
    window = max(15, round(settings.window * min(grey.shape)))
    window += 1 - window % 2
    threshold = threshold_sauvola(grey, window_size=window, k=settings.k)
    ink = (grey < threshold) & folio.inside
    smallest = max(3, int(settings.speck * folio.inside.sum()))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    large = np.zeros(count, bool)
    large[1:] = stats[1:, cv2.CC_STAT_AREA] >= smallest
    return large[labels]


def line_height(page: Page, folio: NormalisedFolio, ink: np.ndarray) -> tuple[float, LineHeightSource]:
    """The line height in the crop's pixels and where it came from: the text lines' spacing where the folio has at
    least three lines, else the period of the ink's vertical profile; NaN and `none` when neither gives one."""
    spacing = line_spacing([page.lines[number].baseline for number in page.folio_lines])
    if math.isfinite(spacing):
        return spacing * folio.scale, "lines"
    period = profile_period(ink, folio.inside)
    if math.isfinite(period):
        return period, "profile"
    return math.nan, "none"


def line_spacing(baselines: list[np.ndarray], overlap: float = 0.5) -> float:
    """The median distance from each baseline to the nearest one below it that overlaps it horizontally by
    `overlap` of the shorter of the two; NaN with fewer than three baselines or two distances."""
    kept = [baseline for baseline in baselines if len(baseline) >= 1]
    if len(kept) < 3:
        return math.nan
    left = np.array([baseline[:, 0].min() for baseline in kept])
    right = np.array([baseline[:, 0].max() for baseline in kept])
    middle = np.array([float(np.mean(baseline[:, 1])) for baseline in kept])
    length = np.maximum(right - left, 1e-9)
    shared = np.minimum(right[:, None], right[None, :]) - np.maximum(left[:, None], left[None, :])
    overlapping = shared >= overlap * np.minimum(length[:, None], length[None, :])
    below = middle[None, :] - middle[:, None]
    candidates = np.where(overlapping & (below > 0), below, np.inf)
    nearest = candidates.min(axis=1)
    distances = nearest[np.isfinite(nearest)]
    return float(np.median(distances)) if len(distances) >= 2 else math.nan


def profile_period(ink: np.ndarray, inside: np.ndarray, shortest: int = 4) -> float:
    """The period of the ink's vertical profile (the ink's share of each row of the folio): the shift of the highest
    peak of its autocorrelation beyond the central lobe, at least `shortest` pixels; NaN without one."""
    from layout_visual_complexity.features.layout import autocorrelation_peak

    rows = inside.any(axis=1)
    if rows.sum() < 4 * shortest:
        return math.nan
    profile = ink[rows].sum(axis=1) / np.maximum(inside[rows].sum(axis=1), 1)
    deviation = profile - profile.mean()
    count = len(deviation)
    spectrum = np.fft.rfft(deviation, n=2 * count)
    found = np.fft.irfft(spectrum * np.conj(spectrum), n=2 * count)[:count]
    if found[0] <= 0:
        return math.nan
    height, lag = autocorrelation_peak(found / found[0])
    if lag is None or lag < shortest or height <= 0:
        return math.nan
    return float(lag)
