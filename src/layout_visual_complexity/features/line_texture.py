"""Line texture as the article publishes it (Appendix B): 38 measures of each text line's vertical tonal transitions,
averaged over the folio's lines.

For each of the folio's text lines, on the page image as the segmentation saw it:
- the image gamma-corrected, 255 (v / 255)^1.5 on each RGB channel, truncated to whole grey levels;
- the line's crop: the axis-aligned bounds of its polygon, clamped to the image (empty or malformed crops skipped);
- a vertical Sobel derivative (dy 1, kernel 5) of each channel, as float32;
- SLIC superpixels on a min–max-normalised copy of the three responses (50 requested, compactness 10, 10
  iterations, σ 1, no Lab conversion, connectivity enforced, sizes 0.5–3 of the average);
- the 38 measures from the raw responses: the number of superpixels; their relative areas' mean, standard deviation,
  minimum, maximum and entropy; their centroids' (normalised by the crop's width and height) and radial distances'
  means and standard deviations, and the area-weighted centroids in the crop's pixels; for each channel, the mean of
  the superpixels' mean responses and of their standard deviations, the extreme responses, and the pixel-weighted mean
  and standard deviation; and the correlations of each channel's superpixel means with their horizontal and vertical
  positions. Standard deviations are a population's; correlations that are undefined are 0.

The page's value is the unweighted mean over its lines; a page without a valid line has none. The measures describe
texture, not colour: no hue or saturation enters them. Some depend on the image's size: the area-weighted centroids
are in pixels, and the responses grow with the strokes' width in pixels.
"""

import math
from dataclasses import dataclass

import numpy as np

from layout_visual_complexity.features.contract import Aggregation, Family, Group, Need
from layout_visual_complexity.features.page import Page

CHANNELS = ("r", "g", "b")
KEYS = (
    "superpixels_count",
    "superpixels_area_mean",
    "superpixels_area_std",
    "superpixels_area_min",
    "superpixels_area_max",
    "superpixels_area_entropy",
    "superpixels_centroid_x_mean",
    "superpixels_centroid_x_std",
    "superpixels_centroid_y_mean",
    "superpixels_centroid_y_std",
    "superpixels_radius_mean",
    "superpixels_radius_std",
    "superpixels_weighted_centroid_x",
    "superpixels_weighted_centroid_y",
    *(
        f"superpixels_channel_{channel}_{statistic}"
        for channel in CHANNELS
        for statistic in ("mean", "std", "min", "max", "weighted_mean", "weighted_std")
    ),
    *(f"superpixels_corr_{axis}_{channel}" for axis in ("x", "y") for channel in CHANNELS),
)
PREFIX = "LineTextureFeature__LineSuperpixels"
# Appendix B's families of the 38 measures: name, number, and what they measure.
TEXTURE_FAMILIES = (
    ("Partition size", KEYS[:1], "Number of connected superpixels."),
    ("Superpixel area", KEYS[1:6],
     "Mean, standard deviation, minimum, maximum and Shannon entropy of relative superpixel areas."),
    ("Spatial distribution", KEYS[6:14],
     "Means and standard deviations of horizontal and vertical centroid coordinates and radial distances, together "
     "with area-weighted centroids."),
    ("Channel statistics", KEYS[14:32],
     "Six summaries for each Sobel-transformed RGB channel: local mean and variation, extrema, and pixel-weighted mean "
     "and standard deviation."),
    ("Spatial–colour association", KEYS[32:],
     "Pearson correlations between horizontal or vertical centroid position and mean superpixel response in each RGB "
     "channel."),
)  # fmt: skip


@dataclass(frozen=True)
class LineTextureSettings:
    """Appendix B's settings.

    Args:
        gamma: The gamma exponent applied to each RGB channel.
        kernel: The vertical Sobel derivative's kernel size.
        segments: The superpixels SLIC is asked for.
        compactness: SLIC's compactness.
        iterations: SLIC's iterations.
        sigma: The Gaussian smoothing before SLIC.
        smallest: SLIC's smallest segment, as a share of the average segment's size.
        largest: SLIC's largest segment, as a share of the average segment's size.
    """

    gamma: float = 1.5
    kernel: int = 5
    segments: int = 50
    compactness: float = 10.0
    iterations: int = 10
    sigma: float = 1.0
    smallest: float = 0.5
    largest: float = 3.0


class LineTexture(Family):
    """Appendix B's 38 line-superpixel measures of each text line's vertical Sobel responses, averaged over the
    folio's lines."""

    name = "line_texture"
    scope = "line"
    version = "1"
    Settings = LineTextureSettings

    @property
    def needs(self) -> frozenset[Need]:
        return frozenset({"image", "lines"})

    def groups(self) -> list[Group]:
        settings: LineTextureSettings = self.settings  # type: ignore[assignment]
        return [
            Group(
                "Line texture",
                "Line superpixels",
                PREFIX,
                KEYS,
                "Each text line's vertical Sobel responses of R, G and B, partitioned into superpixels: their number, "
                "areas, positions, channel statistics and position–response correlations, averaged over the lines.",
                f"gamma {settings.gamma:g}; Sobel kernel {settings.kernel}; SLIC {settings.segments} segments, "
                f"compactness {settings.compactness:g}, σ {settings.sigma:g}",
                "Appendix B of the article",
            )
        ]

    def aggregation(self, column: str) -> Aggregation:
        return "mean"

    def words(self, column: str) -> str:
        key = column.split("__")[-1]
        family = next(name for name, keys, _ in TEXTURE_FAMILIES if key in keys)
        return f"the line texture's {key.removeprefix('superpixels_').replace('_', ' ')} ({family.lower()})"

    def measure(self, page: Page) -> list[dict[str, float]]:
        settings: LineTextureSettings = self.settings  # type: ignore[assignment]
        image = page.image
        scale_x, scale_y = image.shape[1] / page.width, image.shape[0] / page.height
        table = gamma_table(settings.gamma)
        rows = []
        for number in page.folio_lines:
            found = page.line_shapes[number]
            if found is None:
                continue
            x0, y0, x1, y1 = found.bounds
            crop = clamped_crop(image, (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y))
            if crop is None:
                continue
            values = line_measures(table[crop], settings)
            if values is not None:
                rows.append({f"{PREFIX}__{key}": value for key, value in values.items()})
        return rows


def gamma_table(gamma: float) -> np.ndarray:
    """The gamma correction of each grey level, 255 (v / 255)^gamma truncated to whole levels."""
    return np.array(255 * (np.arange(256) / 255) ** gamma, dtype=np.uint8)


def clamped_crop(image: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray | None:
    """The image within the bounds (x0, y0, x1, y1), each truncated to whole pixels and clamped to the image; None
    when that leaves nothing."""
    if not all(math.isfinite(value) for value in bounds):
        return None
    height, width = image.shape[:2]
    x0, x1 = (max(0, min(width, int(value))) for value in (bounds[0], bounds[2]))
    y0, y1 = (max(0, min(height, int(value))) for value in (bounds[1], bounds[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return image[y0:y1, x0:x1]


def vertical_sobel(crop: np.ndarray, kernel: int) -> np.ndarray:
    """Each RGB channel's vertical Sobel derivative, as float32."""
    import cv2

    rgb = crop.astype(np.float32)
    channels = [cv2.Sobel(rgb[:, :, channel], cv2.CV_64F, 0, 1, ksize=kernel) for channel in range(3)]
    return np.stack(channels, axis=-1).astype(np.float32)


def superpixels(responses: np.ndarray, settings: LineTextureSettings) -> np.ndarray:
    """SLIC's labels (from 0) of a min–max-normalised copy of the responses."""
    from skimage.segmentation import slic

    low, high = float(responses.min()), float(responses.max())
    normalised = (
        np.zeros_like(responses) if np.isclose(high, low) else (responses - np.float32(low)) / np.float32(high - low)
    )
    labels = slic(
        normalised,
        n_segments=max(2, settings.segments),
        compactness=settings.compactness,
        max_num_iter=settings.iterations,
        sigma=settings.sigma,
        convert2lab=False,
        enforce_connectivity=True,
        min_size_factor=settings.smallest,
        max_size_factor=settings.largest,
        start_label=0,
        channel_axis=-1,
    )
    return labels.astype(np.int32)


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson's correlation; 0 where it is undefined (fewer than two values, or either constant)."""
    if x.size < 2 or np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def line_measures(crop: np.ndarray, settings: LineTextureSettings) -> dict[str, float] | None:
    """The 38 measures of one gamma-corrected RGB line crop, in the article's order; None if any is undefined."""
    responses = vertical_sobel(crop, settings.kernel)
    return superpixel_measures(responses, superpixels(responses, settings))


def superpixel_measures(image: np.ndarray, labels: np.ndarray) -> dict[str, float] | None:
    """The 38 measures of a three-channel response image partitioned by `labels` (0 to n − 1)."""
    height, width = labels.shape
    flat = labels.reshape(-1)
    pixels = image.reshape(-1, 3).astype(np.float32)
    count = int(flat.max()) + 1 if flat.size else 0
    areas = np.bincount(flat, minlength=count).astype(np.float32)
    areas[areas == 0] = 1.0
    share = areas / float(height * width)
    rows, columns = np.indices((height, width), dtype=np.float32)
    centre_x = np.bincount(flat, weights=columns.reshape(-1), minlength=count).astype(np.float32) / areas
    centre_y = np.bincount(flat, weights=rows.reshape(-1), minlength=count).astype(np.float32) / areas
    x = centre_x / max(1.0, float(width - 1))
    y = centre_y / max(1.0, float(height - 1))
    radius = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    found = {
        "superpixels_count": float(count),
        "superpixels_area_mean": float(np.mean(share)),
        "superpixels_area_std": float(np.std(share)),
        "superpixels_area_min": float(np.min(share)),
        "superpixels_area_max": float(np.max(share)),
        "superpixels_area_entropy": -float(np.sum(share * np.log2(share + 1e-12))),
        "superpixels_centroid_x_mean": float(np.mean(x)),
        "superpixels_centroid_x_std": float(np.std(x)),
        "superpixels_centroid_y_mean": float(np.mean(y)),
        "superpixels_centroid_y_std": float(np.std(y)),
        "superpixels_radius_mean": float(np.mean(radius)),
        "superpixels_radius_std": float(np.std(radius)),
        "superpixels_weighted_centroid_x": float(np.sum(centre_x * share)),
        "superpixels_weighted_centroid_y": float(np.sum(centre_y * share)),
    }
    order = np.argsort(flat, kind="stable")
    starts = np.searchsorted(flat[order], np.arange(count))
    present = np.bincount(flat, minlength=count) > 0
    means = {}
    for number, channel in enumerate(CHANNELS):
        values = pixels[:, number]
        total = np.bincount(flat, weights=values, minlength=count).astype(np.float32)
        squares = np.bincount(flat, weights=values**2, minlength=count).astype(np.float32)
        mean = total / areas
        spread = np.sqrt(np.maximum(squares / areas - mean**2, 0.0))
        weighted_mean = float(np.sum(total) / np.sum(areas))
        weighted_spread = float(np.sqrt(max(float(np.sum(squares) / np.sum(areas)) - weighted_mean**2, 0.0)))
        sorted_values = values[order]
        lowest = np.minimum.reduceat(sorted_values, starts[present]) if present.any() else np.zeros(0, np.float32)
        highest = np.maximum.reduceat(sorted_values, starts[present]) if present.any() else np.zeros(0, np.float32)
        found |= {
            f"superpixels_channel_{channel}_mean": float(np.mean(mean)),
            f"superpixels_channel_{channel}_std": float(np.mean(spread)),
            f"superpixels_channel_{channel}_min": float(np.min(lowest)) if lowest.size else 0.0,
            f"superpixels_channel_{channel}_max": float(np.max(highest)) if highest.size else 0.0,
            f"superpixels_channel_{channel}_weighted_mean": weighted_mean,
            f"superpixels_channel_{channel}_weighted_std": weighted_spread,
        }
        means[channel] = mean
    for axis, positions in (("x", x), ("y", y)):
        for channel in CHANNELS:
            found[f"superpixels_corr_{axis}_{channel}"] = correlation(positions, means[channel])
    found = {key: found[key] for key in KEYS}
    return found if all(math.isfinite(value) for value in found.values()) else None
