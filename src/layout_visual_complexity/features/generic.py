"""Generic image measures: one per dimension of image complexity, needing neither training on judgments nor
segmentation, on the folio crop normalised as `folio_image.py` makes it (1,400 pixels on the long edge, the
illumination flattened, the paper's colour outside the folio).

- Edge-orientation entropy: the Shannon entropy, in bits, of the orientations (modulo 180°, `orientation_bins` bins)
  of the strongest `orientation_edges` edges of the folio, each weighted by its gradient's magnitude, on the
  lightness made `orientation_pixels` pixels large (after Redies et al. 2017, with Sobel's gradients for their Gabor
  filters).
- MSSC per scale: for each of `mssc_scales` successive 2 × 2 mean poolings of the lightness (L*/100), the mean squared
  difference between the image and its pooled version brought back to its size (Kravchenko et al. 2026; the
  article's multi-scale structural complexity, one value a scale).
- Lossless compressibility: the raw size of the lightness quantised to `grey_levels` levels (a byte a pixel) over its
  size as a PNG (zlib, level 6); higher is more compressible.
- Feature congestion: the local variability of colour, luminance contrast and orientation (after Rosenholtz et al.
  2007), on a Gaussian pyramid of `congestion_levels` levels: colour, the geometric mean of the standard deviations
  of the covariance of L*, a*, b* (over 100) in a Gaussian window of σ = 3 pixels; contrast, the local standard
  deviation (σ = 3) of a difference of Gaussians (σ = 1 and 2) of the lightness; orientation, that of the opponent
  orientation energies (Jxx − Jyy, 2Jxy) of the lightness's structure tensor (σ = 1) in a window of σ = 3.5. Each map
  is the maximum over the pyramid's levels, averaged over the folio.
- Colourfulness: Hasler and Süsstrunk's (2003) measure on the folio's pixels: the spread of the opponent channels
  (R − G, (R + G)/2 − B) plus 0.3 times their mean's distance from grey.
- Patch-codebook variety: the folio's square patches (`patch` pixels, at least 90% on the folio) are each described by
  the histograms of their R, G and B values (`bins` bins each, each summing to one) and given their nearest codeword
  in one codebook learned by k-means on patches of the reference pages (`lvc codebook`), so that a codeword means the
  same on every page; the number of codewords used by at least `used` of the page's patches, and the entropy, in bits,
  of their use. Jan Hajič's patch clustering, reworked with one codebook for all pages.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from layout_visual_complexity.features.contract import Family, FeatureError, Group, Need
from layout_visual_complexity.features.folio_image import NormalisedFolio, normalised_folio
from layout_visual_complexity.features.page import Page

SHIPPED_CODEBOOK = "patch_colours.npz"


@dataclass(frozen=True)
class GenericSettings:
    """The normalisation and each measure's parameters.

    Args:
        long_edge: The folio crop's long edge, in pixels.
        orientation_bins: Bins of the edge orientations, over 180°.
        orientation_pixels: The lightness is made this many pixels large for the edge orientations.
        orientation_edges: The strongest this many edges count.
        mssc_scales: Successive 2 × 2 poolings of MSSC, one column each.
        grey_levels: Levels the lightness is quantised to before it is compressed.
        congestion_levels: Levels of feature congestion's Gaussian pyramid.
        codebook: The patch codebook (`codebook.npz` of `lvc codebook`); the one that comes with the package when
            empty.
        used: A codeword counts as used on a page when at least this share of its patches take it.
    """

    long_edge: int = 1400
    orientation_bins: int = 24
    orientation_pixels: int = 120_000
    orientation_edges: int = 10_000
    mssc_scales: int = 7
    grey_levels: int = 16
    congestion_levels: int = 3
    codebook: str = ""
    used: float = 0.01


@dataclass
class Codebook:
    """Codewords of patch colour histograms (`centroids`, codewords × 3 `bins`) and how they were learned: the patch
    size, the bins, the crop's long edge, the seed, the patches drawn per page and the pages."""

    centroids: np.ndarray
    patch: int
    bins: int
    long_edge: int
    learned: dict[str, Any]

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        about = {"patch": self.patch, "bins": self.bins, "long_edge": self.long_edge, "learned": self.learned}
        with path.open("wb") as file:
            np.savez(file, centroids=self.centroids.astype(np.float32), about=np.array(json.dumps(about)))
        return path

    @classmethod
    def load(cls, path: Path) -> "Codebook":
        with np.load(path) as stored:
            about = json.loads(str(stored["about"]))
            return cls(stored["centroids"].astype(np.float32), about["patch"], about["bins"], about["long_edge"],
                       about["learned"])  # fmt: skip

    def assign(self, histograms: np.ndarray) -> np.ndarray:
        """Each histogram's nearest codeword."""
        if not len(histograms):
            return np.zeros(0, int)
        distances = (
            (histograms**2).sum(axis=1)[:, None]
            - 2 * histograms @ self.centroids.T
            + (self.centroids**2).sum(axis=1)[None, :]
        )
        return np.argmin(distances, axis=1)


def codebook_path(given: str) -> Path:
    """The codebook a setting names, or the one that comes with the package."""
    if given:
        return Path(given)
    return Path(str(resources.files("layout_visual_complexity") / "codebooks" / SHIPPED_CODEBOOK))


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Generic(Family):
    """Edge-orientation entropy, MSSC per scale, lossless compressibility, feature congestion, colourfulness and
    patch-codebook variety."""

    name = "generic"
    scope = "page"
    version = "1"
    Settings = GenericSettings

    def __init__(self, settings: GenericSettings | None = None) -> None:
        super().__init__(settings)
        self._codebook: Codebook | None = None

    @property
    def needs(self) -> frozenset[Need]:
        return frozenset({"image", "folio_crop"})

    @property
    def codebook(self) -> Codebook:
        if self._codebook is None:
            path = codebook_path(self.settings.codebook)
            if not path.is_file():
                raise FeatureError(f"generic: no patch codebook at {path}; learn one with `lvc codebook`")
            self._codebook = Codebook.load(path)
            if self._codebook.long_edge != self.settings.long_edge:
                raise FeatureError(
                    f"generic: the codebook was learned on crops of {self._codebook.long_edge} px, not "
                    f"{self.settings.long_edge}"
                )
        return self._codebook

    def settings_hash(self) -> str:
        """The settings' hash, the codebook taken by its contents rather than its path, so that the same codebook
        anywhere gives the same hash."""
        settings = self.settings_dict()
        path = codebook_path(self.settings.codebook)
        settings["codebook"] = file_digest(path) if path.is_file() else "missing"
        payload = {"class": self.path(), "version": self.version, "settings": settings}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def groups(self) -> list[Group]:
        settings: GenericSettings = self.settings  # type: ignore[assignment]
        scales = tuple(f"scale{number}" for number in range(1, settings.mssc_scales + 1))
        return [
            Group("Generic", "Edge-orientation entropy", "Generic__EdgeOrientationEntropy", ("bits",),
                  "Shannon entropy of the edges' orientations, weighted by the gradient's magnitude.",
                  f"{settings.orientation_edges:,} strongest edges at {settings.orientation_pixels:,} px, "
                  f"{settings.orientation_bins} bins over 180°", "Redies et al. 2017"),
            Group("Generic", "MSSC", "Generic__Mssc", scales,
                  "Mean squared difference between the lightness and its 2 × 2 pooling, at each scale.",
                  f"{settings.mssc_scales} successive poolings", "Kravchenko et al. 2026"),
            Group("Generic", "Lossless compressibility", "Generic__Compressibility", ("ratio",),
                  "Raw size of the quantised lightness over its size as a PNG.",
                  f"{settings.grey_levels} grey levels, zlib level 6", "this study"),
            Group("Generic", "Feature congestion", "Generic__FeatureCongestion", ("colour", "contrast", "orientation"),
                  "Local variability of colour, luminance contrast and orientation.",
                  f"{settings.congestion_levels} pyramid levels, pooling σ 3–3.5 px", "Rosenholtz et al. 2007"),
            Group("Generic", "Colourfulness", "Generic__Colourfulness", ("opponent",),
                  "Spread and mean of the opponent colour channels.", "--", "Hasler and Süsstrunk 2003"),
            Group("Generic", "Patch-codebook variety", "Generic__CodebookVariety", ("codewords", "entropy"),
                  "Codewords used by the page's patches, and the entropy of their use, against one codebook of patch "
                  "colour histograms learned on the reference pages.",
                  self.codebook_parameters(), "after Hajič, this study"),
        ]  # fmt: skip

    def codebook_parameters(self) -> str:
        path = codebook_path(self.settings.codebook)
        if not path.is_file():
            return f"used from {self.settings.used:g} of the patches"
        codebook = self.codebook
        return (
            f"k = {len(codebook.centroids)}, {codebook.patch} px patches, {codebook.bins} bins a channel; used from "
            f"{self.settings.used:g} of the patches"
        )

    def words(self, column: str) -> str:
        _, feature, flavour = column.split("__")
        if feature == "Mssc":
            return f"MSSC at its {ordinal(int(flavour[5:]))} scale"
        if feature == "FeatureCongestion":
            return f"the feature congestion of {flavour}"
        return {
            "Generic__EdgeOrientationEntropy__bits": "the entropy of the edges' orientations",
            "Generic__Compressibility__ratio": "the lossless compressibility of the lightness",
            "Generic__Colourfulness__opponent": "the colourfulness",
            "Generic__CodebookVariety__codewords": "the number of patch codewords used",
            "Generic__CodebookVariety__entropy": "the entropy of the patch codewords' use",
        }.get(column) or super().words(column)

    def measure(self, page: Page) -> list[dict[str, float]]:
        settings: GenericSettings = self.settings  # type: ignore[assignment]
        folio = normalised_folio(page, settings.long_edge)
        grey = folio.grey
        row = {
            "Generic__EdgeOrientationEntropy__bits": orientation_entropy(
                grey, folio.inside, settings.orientation_bins, settings.orientation_pixels, settings.orientation_edges
            )
        }
        row |= {f"Generic__Mssc__scale{number}": value
                for number, value in enumerate(mssc(grey, settings.mssc_scales), start=1)}  # fmt: skip
        row["Generic__Compressibility__ratio"] = compressibility(grey, settings.grey_levels)
        colour, contrast, orientation = feature_congestion(folio.lab, folio.inside, settings.congestion_levels)
        row |= {"Generic__FeatureCongestion__colour": colour, "Generic__FeatureCongestion__contrast": contrast,
                "Generic__FeatureCongestion__orientation": orientation}  # fmt: skip
        row["Generic__Colourfulness__opponent"] = colourfulness(folio.rgb[folio.inside])
        codebook = self.codebook
        histograms = patch_histograms(folio, codebook.patch, codebook.bins)
        codewords, entropy = codebook_variety(codebook.assign(histograms), len(codebook.centroids), settings.used)
        row |= {"Generic__CodebookVariety__codewords": codewords, "Generic__CodebookVariety__entropy": entropy}
        return [row]


def ordinal(number: int) -> str:
    return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh"}.get(
        number, f"{number}th"
    )


def entropy_bits(counts: np.ndarray) -> float:
    """The Shannon entropy, in bits, of a distribution given by counts or weights."""
    total = counts.sum()
    if total <= 0:
        return math.nan
    shares = counts[counts > 0] / total
    return float(-(shares * np.log2(shares)).sum()) + 0.0


def orientation_entropy(grey: np.ndarray, inside: np.ndarray, bins: int, pixels: int, edges: int) -> float:
    """The entropy, in bits, of the orientations (modulo 180°) of the strongest `edges` gradients of the folio,
    weighted by their magnitude, on the image made `pixels` pixels large."""
    import cv2

    scale = min(1.0, math.sqrt(pixels / grey.size))
    size = (max(3, round(grey.shape[1] * scale)), max(3, round(grey.shape[0] * scale)))
    image = cv2.resize(grey.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(inside.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(dx, dy)[mask]
    angle = np.mod(np.arctan2(dy, dx)[mask], np.pi)
    strongest = np.argsort(-magnitude, kind="stable")[:edges]
    counts, _ = np.histogram(angle[strongest], bins=bins, range=(0, np.pi), weights=magnitude[strongest])
    return entropy_bits(counts)


def mssc(grey: np.ndarray, scales: int) -> list[float]:
    """For each of `scales` successive 2 × 2 mean poolings, the mean squared difference between the image and its
    pooled version repeated back to its size; NaN once the image is too small to pool."""
    found = []
    current = grey.astype(np.float64)
    for _ in range(scales):
        rows, columns = current.shape[0] // 2 * 2, current.shape[1] // 2 * 2
        if rows < 2 or columns < 2:
            found.append(math.nan)
            continue
        current = current[:rows, :columns]
        pooled = current.reshape(rows // 2, 2, columns // 2, 2).mean(axis=(1, 3))
        back = np.repeat(np.repeat(pooled, 2, axis=0), 2, axis=1)
        found.append(float(((current - back) ** 2).mean()))
        current = pooled
    return found


def compressibility(grey: np.ndarray, levels: int) -> float:
    """The raw size of the image quantised to `levels` levels, a byte a pixel, over its size as a PNG."""
    import cv2

    quantised = np.minimum(np.floor(np.clip(grey, 0, 1) * levels), levels - 1).astype(np.uint8)
    found, encoded = cv2.imencode(".png", quantised, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not found:
        return math.nan
    return float(quantised.size / len(encoded))


def colourfulness(rgb: np.ndarray) -> float:
    """Hasler and Süsstrunk's colourfulness of pixels (rows of R, G, B in 0–255)."""
    if not len(rgb):
        return math.nan
    values = rgb.reshape(-1, 3).astype(np.float64)
    red_green = values[:, 0] - values[:, 1]
    yellow_blue = 0.5 * (values[:, 0] + values[:, 1]) - values[:, 2]
    spread = math.hypot(red_green.std(), yellow_blue.std())
    mean = math.hypot(red_green.mean(), yellow_blue.mean())
    return spread + 0.3 * mean


def blurred(image: np.ndarray, sigma: float) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def local_spread(channels: list[np.ndarray], sigma: float, floor: float) -> np.ndarray:
    """The geometric mean of the standard deviations of the channels' local covariance in a Gaussian window: the
    determinant's 2n-th root, `floor` added to its diagonal."""
    means = [blurred(channel, sigma) for channel in channels]
    count = len(channels)
    c = {}
    for i in range(count):
        for j in range(i, count):
            c[i, j] = c[j, i] = blurred(channels[i] * channels[j], sigma) - means[i] * means[j] + floor * (i == j)
    if count == 2:
        determinant = c[0, 0] * c[1, 1] - c[0, 1] ** 2
    elif count == 3:
        determinant = (
            c[0, 0] * (c[1, 1] * c[2, 2] - c[1, 2] ** 2)
            - c[0, 1] * (c[0, 1] * c[2, 2] - c[1, 2] * c[0, 2])
            + c[0, 2] * (c[0, 1] * c[1, 2] - c[1, 1] * c[0, 2])
        )
    else:
        raise ValueError("local_spread takes two or three channels")
    return np.maximum(determinant, 0.0) ** (1 / (2 * count))


def feature_congestion(lab: np.ndarray, inside: np.ndarray, levels: int) -> tuple[float, float, float]:
    """Colour, contrast and orientation clutter: each map the maximum over a Gaussian pyramid's levels, averaged over
    the folio."""
    import cv2

    image = (lab / 100.0).astype(np.float32)
    rows, columns = image.shape[:2]
    names = ("colour", "contrast", "orientation")
    highest: dict[str, np.ndarray] = {}
    for level in range(levels):
        if level:
            image = cv2.pyrDown(image)
        if min(image.shape[:2]) < 8:
            break
        lightness = image[..., 0]
        maps = {"colour": local_spread([image[..., number] for number in range(3)], 3.0, 1e-4)}
        difference = blurred(lightness, 1.0) - blurred(lightness, 2.0)
        mean = blurred(difference, 3.0)
        maps["contrast"] = np.sqrt(np.maximum(blurred(difference**2, 3.0) - mean**2, 0.0))
        smooth = blurred(lightness, 1.0)
        dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        jxx, jyy, jxy = (blurred(value, 1.0) for value in (dx * dx, dy * dy, dx * dy))
        maps["orientation"] = local_spread([jxx - jyy, 2 * jxy], 3.5, 0.0)
        for name in names:
            found = maps[name].astype(np.float32)
            if level:
                found = cv2.resize(found, (columns, rows), interpolation=cv2.INTER_LINEAR)
            highest[name] = np.maximum(highest[name], found) if name in highest else found
    values = [float(highest[name][inside].mean()) if name in highest else math.nan for name in names]
    return values[0], values[1], values[2]


# The patch codebook.


def patch_histograms(folio: NormalisedFolio, patch: int, bins: int, least_inside: float = 0.9) -> np.ndarray:
    """The colour histograms of the folio's square patches (a grid of `patch` pixels from the crop's top left, those
    at least `least_inside` on the folio): `bins` bins of R, G and B each, each channel's summing to one."""
    rows, columns = folio.rgb.shape[0] // patch, folio.rgb.shape[1] // patch
    if rows == 0 or columns == 0:
        return np.zeros((0, 3 * bins), np.float32)
    rgb = folio.rgb[: rows * patch, : columns * patch]
    inside = folio.inside[: rows * patch, : columns * patch]
    share = inside.reshape(rows, patch, columns, patch).mean(axis=(1, 3)).ravel()
    blocks = rgb.reshape(rows, patch, columns, patch, 3).transpose(0, 2, 1, 3, 4).reshape(rows * columns, -1, 3)
    kept = blocks[share >= least_inside]
    if not len(kept):
        return np.zeros((0, 3 * bins), np.float32)
    values = (kept.astype(np.int64) * bins) // 256
    index = (np.arange(len(kept))[:, None, None] * 3 + np.arange(3)[None, None, :]) * bins + values
    counts = np.bincount(index.ravel(), minlength=len(kept) * 3 * bins).reshape(len(kept), 3 * bins)
    return (counts / (patch * patch)).astype(np.float32)


def codebook_variety(assigned: np.ndarray, codewords: int, used: float) -> tuple[float, float]:
    """The number of codewords taken by at least `used` of the patches, and the entropy, in bits, of their use."""
    if not len(assigned):
        return math.nan, math.nan
    counts = np.bincount(assigned, minlength=codewords).astype(float)
    return float((counts >= used * len(assigned)).sum()), entropy_bits(counts)


def page_sample(histograms: np.ndarray, page: str, count: int, seed: int) -> np.ndarray:
    """`count` of a page's patch histograms, drawn without replacement from a generator seeded by the seed and the
    page's name, so a page's draw doesn't depend on the other pages or on the order they come in."""
    if len(histograms) <= count:
        return histograms
    digest = int.from_bytes(hashlib.sha256(page.encode()).digest()[:8], "big")
    rng = np.random.default_rng([seed, digest])
    return histograms[np.sort(rng.choice(len(histograms), count, replace=False))]


def learn_codewords(samples: list[np.ndarray], codewords: int, seed: int, starts: int = 4) -> np.ndarray:
    """k-means codewords of the pages' patch histograms, taken in the order given; seeded."""
    from sklearn.cluster import KMeans

    data = np.concatenate(samples).astype(np.float64)
    if len(data) < codewords:
        raise ValueError(f"{len(data)} patches can't make {codewords} codewords")
    model = KMeans(n_clusters=codewords, n_init=starts, random_state=seed).fit(data)
    centroids = model.cluster_centers_
    order = np.lexsort(centroids.T[::-1])
    return centroids[order].astype(np.float32)
