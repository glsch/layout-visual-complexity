"""Kyle–Davidson as published: three of the four measures of scene complexity of Kyle-Davidson et al. (2023), on the
folio crop as it is, resized to their images' 700 pixels on the long edge, the folio's median colour outside it.
Openness, a GIST "open scene", means nothing for a page and is left out.

- Clutter: the number of regions left by a normalised cut (Shi and Malik 2000) of the region adjacency graph of
  SLIC superpixels, each region of one mean colour, counted on the folio. The paper names the cut but not the
  superpixels' parameters; these are scikit-image's for its normalised cut (400 superpixels, compactness 30, the
  edges' similarity exp(−d²/255) of the mean colours' distance d, a cut kept while its value is under 0.001, ten
  cuts tried at each step), the cut's random start seeded.
- Symmetry: square patches of 100, 50 and 25 pixels (a seventh, a fourteenth and a twenty-eighth of the long edge),
  stepped by half their side over the crop, those at least 90% on the folio; each patch's left half against its
  right half mirrored, and its top half against its bottom half, as one minus the root mean square difference of
  their colours (RGB in [0, 1]); the mean of the two over the patches, one value a size.
- Entropy: the Shannon entropy, in bits, of each HSV channel's values on the folio (hue in OpenCV's 180 values,
  saturation and value in 256), over a Gaussian pyramid of `levels` levels (the resolution 1, 1/2, 1/4, …), the mean
  over the levels, one value a channel.
"""

import math
from dataclasses import dataclass

import numpy as np

from layout_visual_complexity.features.contract import Family, Group, Need
from layout_visual_complexity.features.page import Page, raster_shape


@dataclass(frozen=True)
class KyleDavidsonSettings:
    """Kyle–Davidson's measures as published, and the parameters the paper leaves open.

    Args:
        long_edge: The folio crop's long edge, in pixels (the paper's images are 700 pixels square).
        superpixels: SLIC's number of superpixels.
        compactness: SLIC's compactness.
        cut_threshold: The normalised cut stops dividing a region whose best cut's value exceeds this.
        cuts: Cuts tried at each step of the normalised cut.
        seed: The seed of the normalised cut's random start.
        patch_sizes: The symmetry's patch sides, in pixels at `long_edge`.
        inside: A symmetry patch counts when at least this share of it lies on the folio.
        levels: The entropy's Gaussian pyramid levels.
    """

    long_edge: int = 700
    superpixels: int = 400
    compactness: float = 30.0
    cut_threshold: float = 0.001
    cuts: int = 10
    seed: int = 0
    patch_sizes: tuple[int, ...] = (100, 50, 25)
    inside: float = 0.9
    levels: int = 4


SIZES = ("large", "medium", "small", "smallest")


class KyleDavidson(Family):
    """Clutter, symmetry and entropy of the folio crop, as Kyle-Davidson et al. (2023) measure scenes."""

    name = "kyle_davidson"
    scope = "page"
    version = "1"
    Settings = KyleDavidsonSettings

    @property
    def needs(self) -> frozenset[Need]:
        return frozenset({"image", "folio_crop"})

    def flavours(self) -> tuple[str, ...]:
        sizes = self.settings.patch_sizes
        if len(sizes) > len(SIZES):
            return tuple(f"patch{size}" for size in sizes)
        return SIZES[: len(sizes)]

    def groups(self) -> list[Group]:
        settings: KyleDavidsonSettings = self.settings  # type: ignore[assignment]
        sizes = ", ".join(str(size) for size in settings.patch_sizes)
        return [
            Group("Kyle–Davidson", "Clutter", "KyleDavidson__Clutter", ("regions",),
                  "Regions left by a normalised cut of the region adjacency graph of SLIC superpixels.",
                  f"{settings.superpixels} superpixels, compactness {settings.compactness:g}, cut below "
                  f"{settings.cut_threshold:g}, at {settings.long_edge} px", "Kyle-Davidson et al. 2023"),
            Group("Kyle–Davidson", "Symmetry", "KyleDavidson__Symmetry", self.flavours(),
                  "Mean left–right and top–bottom mirror symmetry within patches stepped by half their side.",
                  f"patches of {sizes} px at {settings.long_edge} px", "Kyle-Davidson et al. 2023"),
            Group("Kyle–Davidson", "Entropy", "KyleDavidson__Entropy", ("hue", "saturation", "value"),
                  "Shannon entropy of each HSV channel over a Gaussian pyramid.",
                  f"{settings.levels} levels", "Kyle-Davidson et al. 2023"),
        ]  # fmt: skip

    def words(self, column: str) -> str:
        _, feature, flavour = column.split("__")
        if feature == "Symmetry":
            sizes = dict(zip(self.flavours(), self.settings.patch_sizes, strict=True))
            return f"the mirror symmetry within patches of {sizes[flavour]} pixels"
        if feature == "Entropy":
            return f"the entropy of the {flavour} channel"
        if feature == "Clutter":
            return "the regions left by the normalised cut"
        return super().words(column)

    def measure(self, page: Page) -> list[dict[str, float]]:
        settings: KyleDavidsonSettings = self.settings  # type: ignore[assignment]
        width, height = page.folio_size
        image, inside = page.folio_image(raster_shape(width, height, settings.long_edge))
        if not inside.any():
            inside = np.ones(inside.shape, bool)
        image = image.copy()
        image[~inside] = np.median(image[inside], axis=0).astype(np.uint8)
        row = {"KyleDavidson__Clutter__regions": clutter(image, inside, settings)}
        for flavour, size in zip(self.flavours(), settings.patch_sizes, strict=True):
            row[f"KyleDavidson__Symmetry__{flavour}"] = symmetry(image, inside, size, settings.inside)
        hue, saturation, value = hsv_entropy(image, inside, settings.levels)
        row |= {"KyleDavidson__Entropy__hue": hue, "KyleDavidson__Entropy__saturation": saturation,
                "KyleDavidson__Entropy__value": value}  # fmt: skip
        return [row]


def normalised_cut(image: np.ndarray, settings: KyleDavidsonSettings) -> np.ndarray:
    """The labels of the regions left by the normalised cut of the image's SLIC superpixels."""
    from skimage import graph, segmentation

    labels = segmentation.slic(image, n_segments=settings.superpixels, compactness=settings.compactness,
                               start_label=1)  # fmt: skip
    if labels.max() < 2:
        return labels
    adjacency = graph.rag_mean_color(image, labels, mode="similarity")
    return graph.cut_normalized(labels, adjacency, thresh=settings.cut_threshold, num_cuts=settings.cuts,
                                rng=settings.seed)  # fmt: skip


def clutter(image: np.ndarray, inside: np.ndarray, settings: KyleDavidsonSettings) -> float:
    """The number of the normalised cut's regions on the folio."""
    return float(len(np.unique(normalised_cut(image, settings)[inside])))


def mirror_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """The root mean square difference of each pair of arrays (patches × …), over everything but the first axis."""
    return np.sqrt(((first - second) ** 2).reshape(len(first), -1).mean(axis=1))


def patch_symmetry(patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each square patch's (patches × side × side × channels, values in [0, 1]) left–right and top–bottom symmetry:
    one minus the root mean square difference between one half and the other mirrored."""
    half = patches.shape[1] // 2
    left, right = patches[:, :, :half], patches[:, :, -half:]
    top, bottom = patches[:, :half], patches[:, -half:]
    across = 1 - mirror_difference(left[:, :, ::-1], right)
    down = 1 - mirror_difference(top[:, ::-1], bottom)
    return across, down


def symmetry(image: np.ndarray, inside: np.ndarray, size: int, least_inside: float) -> float:
    """The mean over patches of `size` pixels, stepped by half their side and at least `least_inside` on the folio,
    of the mean of their left–right and top–bottom symmetry; NaN without such patches."""
    from numpy.lib.stride_tricks import sliding_window_view

    if size < 2 or size > min(inside.shape):
        return math.nan
    step = max(1, size // 2)
    values = image.astype(np.float32) / 255.0
    windows = sliding_window_view(values, (size, size), axis=(0, 1))[::step, ::step]
    shares = sliding_window_view(inside.astype(np.float32), (size, size))[::step, ::step].mean(axis=(2, 3))
    kept = shares >= least_inside
    if not kept.any():
        return math.nan
    patches = windows[kept].transpose(0, 2, 3, 1)
    across, down = patch_symmetry(patches)
    return float(((across + down) / 2).mean())


def channel_entropy(values: np.ndarray, levels: int) -> float:
    counts = np.bincount(values, minlength=levels).astype(float)
    shares = counts[counts > 0] / counts.sum()
    return float(-(shares * np.log2(shares)).sum()) + 0.0


def hsv_entropy(image: np.ndarray, inside: np.ndarray, levels: int) -> tuple[float, float, float]:
    """The entropy, in bits, of the hue, saturation and value of the folio's pixels, each the mean over a Gaussian
    pyramid's levels."""
    import cv2

    found: list[list[float]] = [[], [], []]
    current, mask = image, inside.astype(np.float32)
    for level in range(levels):
        if level:
            current, mask = cv2.pyrDown(current), cv2.pyrDown(mask)
        kept = mask >= 0.5
        if min(current.shape[:2]) < 2 or not kept.any():
            break
        hsv = cv2.cvtColor(np.ascontiguousarray(current), cv2.COLOR_RGB2HSV)[kept]
        for channel, count in enumerate((180, 256, 256)):
            found[channel].append(channel_entropy(hsv[:, channel], count))
    hue, saturation, value = (float(np.mean(values)) if values else math.nan for values in found)
    return hue, saturation, value
