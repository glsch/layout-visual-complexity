"""Patch tokens of a vision transformer (ViT): any `transformers` model with patch tokens, loaded from its publisher's
repository at a pinned revision. DINOv2 ViT-S/14 is the default.

A page's input is its folio crop (the folio's bounding box, as the other families take it) or its whole image, resized
so that its long edge is `long_edge` pixels, each side rounded to whole patches, and normalised with the model's image
mean and spread. Its tokens are the chosen block's patch tokens after the model's final layer norm, with the class
token (and any register tokens) dropped: one row per patch, row by row from the top left of the grid.

Tokens are kept as float16 arrays, one file per page, in the user cache, in a folder keyed by the backbone, its
revision, the layer, the input size and the crop. Each page's file also holds what its tokens were made from (its
image and its crop box), so a page whose image or folio changed is made again.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from layout_visual_complexity.features.page import Page

# facebook/dinov2-small, as published on 2023-09-06.
DINOV2_SMALL = "facebook/dinov2-small"
DINOV2_SMALL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NORMS = ("layernorm", "norm", "post_layernorm")

Crop = Literal["folio", "page"]
Device = Literal["auto", "cpu", "mps", "cuda"]


@dataclass(frozen=True)
class TokenSettings:
    """Which tokens: the backbone, its block, the input size and the crop.

    Args:
        backbone: A `transformers` model with patch tokens: its repository on Hugging Face, or a local folder.
        revision: The repository's commit; None takes the latest, which isn't reproducible.
        layer: The block whose tokens are taken, counted from 1; -1 is the last.
        long_edge: The input's long edge in pixels; each side is rounded to whole patches.
        crop: `folio`, the folio's bounding box from the segmentation; or `page`, the whole image.
    """

    backbone: str = DINOV2_SMALL
    revision: str | None = DINOV2_SMALL_REVISION
    layer: int = -1
    long_edge: int = 518
    crop: Crop = "folio"

    def key(self) -> str:
        """What decides the tokens, as 16 hexadecimal digits: the cache folder's name."""
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class TokenGrid:
    """A page's tokens: `rows` × `columns` patches, one row of `tokens` each, row by row from the top left; `box` is
    the crop they cover, in the full image's pixels (x0, y0, x1, y1)."""

    tokens: np.ndarray
    rows: int
    columns: int
    box: tuple[float, float, float, float]

    def patch_boxes(self) -> np.ndarray:
        """Each patch's box in the full image's pixels, (x0, y0, x1, y1), in the tokens' order."""
        x0, y0, x1, y1 = self.box
        width, height = (x1 - x0) / self.columns, (y1 - y0) / self.rows
        row, column = np.divmod(np.arange(self.rows * self.columns), self.columns)
        left, top = x0 + column * width, y0 + row * height
        return np.column_stack([left, top, left + width, top + height])


def crop_box(page: Page, crop: Crop) -> tuple[float, float, float, float]:
    """The part of the page the tokens cover, in the full image's pixels: the folio's bounding box within the image,
    or the whole image."""
    if crop == "page":
        return 0.0, 0.0, float(page.width), float(page.height)
    x0, y0, x1, y1 = page.folio_bounds
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(page.width), max(x1, x0 + 1)), min(float(page.height), max(y1, y0 + 1))
    return float(x0), float(y0), float(x1), float(y1)


def cropped(image: np.ndarray, box: tuple[float, float, float, float], width: int, height: int) -> np.ndarray:
    """The box of the full image (`width` × `height` pixels) cut from `image`, which may have been fetched smaller."""
    scale_x, scale_y = image.shape[1] / width, image.shape[0] / height
    x0, y0, x1, y1 = box
    left, top = max(0, int(np.floor(x0 * scale_x))), max(0, int(np.floor(y0 * scale_y)))
    right = min(image.shape[1], max(left + 1, int(np.ceil(x1 * scale_x))))
    bottom = min(image.shape[0], max(top + 1, int(np.ceil(y1 * scale_y))))
    return image[top:bottom, left:right]


def grid_shape(height: int, width: int, long_edge: int, patch_size: int) -> tuple[int, int]:
    """Patches along each side of an image resized to a long edge of `long_edge`, each side rounded to whole
    patches."""
    scale = long_edge / max(height, width)
    return max(1, round(height * scale / patch_size)), max(1, round(width * scale / patch_size))


def model_input(
    rgb: np.ndarray, long_edge: int, patch_size: int, mean: tuple[float, ...], std: tuple[float, ...]
) -> tuple[np.ndarray, int, int]:
    """An RGB image as the model's input: resized (bicubic) to whole patches with a long edge of about `long_edge`,
    scaled to [0, 1] and normalised; channels first. Returns the input and the grid's rows and columns."""
    from PIL import Image

    rows, columns = grid_shape(rgb.shape[0], rgb.shape[1], long_edge, patch_size)
    resized = Image.fromarray(rgb).resize((columns * patch_size, rows * patch_size), Image.Resampling.BICUBIC)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    values = (values - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(values.transpose(2, 0, 1)), rows, columns


def chosen_device(device: Device) -> str:
    """`auto`: Apple's GPU (MPS) or CUDA where there is one, else the CPU."""
    if device != "auto":
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class Backbone:
    """A ViT loaded once, frozen, on a device; `tokens` gives an image's patch tokens."""

    def __init__(self, settings: TokenSettings, device: str = "cpu", local_only: bool = False) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.settings = settings
        self.device = device
        options: dict[str, Any] = {"revision": settings.revision, "local_files_only": local_only}
        self.model = AutoModel.from_pretrained(settings.backbone, **options).eval().to(device)
        try:
            processor = AutoImageProcessor.from_pretrained(settings.backbone, **options)
            self.mean = tuple(float(value) for value in processor.image_mean)
            self.std = tuple(float(value) for value in processor.image_std)
        except (OSError, ValueError, AttributeError, TypeError):
            self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        config = self.model.config
        patch = getattr(config, "patch_size", None)
        if patch is None:
            raise ValueError(f"{settings.backbone} has no patch size: it isn't a ViT with patch tokens")
        self.patch_size = int(patch[0] if isinstance(patch, list | tuple) else patch)
        self.dim = int(config.hidden_size)
        self.blocks = int(config.num_hidden_layers)
        layer = settings.layer if settings.layer > 0 else self.blocks + 1 + settings.layer
        if not 1 <= layer <= self.blocks:
            raise ValueError(f"{settings.backbone} has {self.blocks} blocks; there is no block {settings.layer}")
        self.layer = layer
        self.norm = next((getattr(self.model, name) for name in NORMS if hasattr(self.model, name)), None)
        self._torch = torch

    def describe(self) -> dict[str, Any]:
        """The tokens as a token set records them."""
        return {
            **asdict(self.settings),
            "block": self.layer,
            "blocks": self.blocks,
            "patch_size": self.patch_size,
            "dim": self.dim,
            "model": type(self.model).__name__,
            "mean": list(self.mean),
            "std": list(self.std),
        }

    def tokens(self, rgb: np.ndarray) -> tuple[np.ndarray, int, int]:
        """An RGB image's patch tokens as float32 (patches × dim), and the grid's rows and columns."""
        torch = self._torch
        values, rows, columns = model_input(rgb, self.settings.long_edge, self.patch_size, self.mean, self.std)
        pixels = torch.from_numpy(values).unsqueeze(0).to(self.device)
        with torch.no_grad():
            last = self.layer == self.blocks
            output = self.model(pixel_values=pixels, output_hidden_states=not last)
            if last:
                hidden = output.last_hidden_state
            else:
                hidden = output.hidden_states[self.layer]
                hidden = self.norm(hidden) if self.norm is not None else hidden
        patches = rows * columns
        if hidden.shape[1] < patches:
            raise ValueError(f"the model gave {hidden.shape[1]} tokens for {rows} × {columns} patches")
        return hidden[0, hidden.shape[1] - patches :].float().cpu().numpy(), rows, columns


# The cache.


def page_file_name(page: str) -> str:
    """A page's file name in the cache: its id where that is safe as a file name, else a hash of it."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,150}", page):
        return f"{page}.npz"
    return f"page-{hashlib.sha256(page.encode()).hexdigest()[:24]}.npz"


def source_key(image: str | None, width: int, height: int, box: tuple[float, float, float, float]) -> str:
    """What a page's tokens are made from, besides the settings: its image, its size and the crop box."""
    payload = [image, int(width), int(height), [round(value, 2) for value in box]]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


class TokenCache:
    """Pages' tokens in `root/tokens/<key>/`, one `.npz` per page: the float16 tokens, the grid, the crop box, and
    the source key and device they were made with."""

    def __init__(self, root: Path, settings: TokenSettings) -> None:
        self.folder = Path(root).expanduser() / "tokens" / settings.key()

    def path(self, page: str) -> Path:
        return self.folder / page_file_name(page)

    def read(self, page: str, source: str | None = None) -> TokenGrid | None:
        """A page's tokens, or None when they aren't cached or were made from another source."""
        path = self.path(page)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as stored:
            if source is not None and str(stored["source"]) != source:
                return None
            rows, columns = (int(value) for value in stored["grid"])
            box = tuple(float(value) for value in stored["box"])
            return TokenGrid(stored["tokens"], rows, columns, box)  # type: ignore[arg-type]

    def write(self, page: str, grid: TokenGrid, source: str, device: str) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.path(page)
        partial = path.with_name(path.name + ".part.npz")
        np.savez(
            partial,
            tokens=grid.tokens.astype(np.float16),
            grid=np.array([grid.rows, grid.columns]),
            box=np.array(grid.box, dtype=float),
            source=np.array(source),
            device=np.array(device),
        )
        partial.replace(path)
        return path


# A token set: the tokens of a run's pages.

TOKEN_SET, TOKEN_PAGES = "tokens.json", "pages.tsv"
# What a trained head must match in the tokens it scores.
IDENTITY = ("backbone", "revision", "block", "long_edge", "crop", "patch_size", "dim")


class TokenMismatch(ValueError):
    """Tokens made otherwise than a trained head's: another backbone, revision, block, input size or crop."""


@dataclass
class TokenSet:
    """The tokens of a run's pages (`lvc embed`): what made them (`description`), each page's grid and crop box and
    source (`pages`, indexed by page), and the cache holding them."""

    description: dict[str, Any]
    pages: Any
    cache: TokenCache

    def identity(self) -> dict[str, Any]:
        """What a trained head records of its tokens and checks in those it scores."""
        return {name: self.description.get(name) for name in IDENTITY}

    def grid(self, page: str) -> TokenGrid:
        """A page's tokens, checked against the source the run recorded."""
        found = self.cache.read(page, str(self.pages.at[page, "source"]))
        if found is None:
            raise ValueError(f"{page}: its tokens aren't in the cache ({self.cache.folder}); run lvc embed again")
        return found


def read_token_set(folder: Path) -> TokenSet:
    """The token set an embed run wrote into `folder`."""
    import pandas as pd

    folder = Path(folder)
    if not (folder / TOKEN_SET).is_file():
        raise ValueError(f"{folder} holds no token set ({TOKEN_SET}); make one with lvc embed")
    description = json.loads((folder / TOKEN_SET).read_text(encoding="utf-8"))
    pages = pd.read_csv(folder / TOKEN_PAGES, sep="\t", dtype={"page": str, "source": str}).set_index("page")
    settings = TokenSettings(**{name: description[name] for name in TokenSettings.__dataclass_fields__})
    return TokenSet(description, pages, TokenCache(Path(description["cache_root"]), settings))
