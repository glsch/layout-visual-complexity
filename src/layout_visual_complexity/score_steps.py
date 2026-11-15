"""The stages `lvc score` adds to the segmentation's: the feature families on each page's layout, and the ViT head
on its patch tokens.

A page's features are measured as `lvc features` measures them from the layout store: its elements in the full
image's pixels, its lines assigned to its regions, its image the one the segmentation worked on. Its tokens are made
as `lvc embed` makes them, kept as float16 as the token cache keeps them, scored by the head and never stored: the
record keeps the page's raw score, its grid and crop box, and its patches' scores as float16.
"""

import base64
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from layout_visual_complexity.features.page import Page, points
from layout_visual_complexity.pipeline.step import Step
from layout_visual_complexity.segmentation.records import PageRecord, Status
from layout_visual_complexity.segmentation.store import layout_rows, scaled, scaler

LAYOUT_COLUMNS = ["page", "kind", "class", "polygon", "baseline", "confidence", "region", "id"]


def page_of(record: PageRecord, image: bool = True) -> Page:
    """A record as feature families see it, as the layout store would give it: in the full image's pixels, rounded
    to whole pixels, its lines assigned to its regions."""
    from layout_visual_complexity.segmentation.lines import assign_lines

    assign_lines(record.lines, record.regions)
    rows = pd.DataFrame(layout_rows(record), columns=LAYOUT_COLUMNS)
    width, height = record.full_width or record.width, record.full_height or record.height
    path = Path(record.image) if image and record.image else None
    return Page.from_layout(record.page, rows, int(width or 0), int(height or 0), path)


def folio_page(record: PageRecord) -> Page:
    """The page with its folio alone, in the full image's pixels, as the token crop takes it."""
    width, height = record.full_width or record.width, record.full_height or record.height
    folio = points(scaled(record.folio.quadrilateral, scaler(record))) if record.folio is not None else None
    return Page(record.page, int(width or 0), int(height or 0), folio)


def encode(values: np.ndarray) -> str:
    """A float16 array as text, for the record."""
    return base64.b64encode(np.asarray(values, dtype="<f2").tobytes()).decode("ascii")


def decode(text: str, rows: int, columns: int) -> np.ndarray:
    """Patch scores from the record, as a rows × columns float16 array."""
    return np.frombuffer(base64.b64decode(text), dtype="<f2").reshape(rows, columns)


@dataclass(frozen=True)
class FeatureStepSettings:
    """The feature families measured on each page.

    Args:
        families: The families, by short name or Python path.
        parameters: Each family's settings, under the name it's given by.
        scored: The families the ranker needs: a page on which one of them fails, fails.
        workers: Worker processes; by default a share of the cores.
    """

    families: list[str] = field(default_factory=lambda: ["geometry"])
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    scored: list[str] = field(default_factory=lambda: ["geometry"])
    workers: int | None = None


class FeaturesStep(Step):
    """The feature families on the page's layout and image."""

    needs = frozenset({"image", "folio", "regions", "lines"})
    produces = frozenset({"features", "feature_errors"})
    memory_mb = {"cpu": 800}
    settings: FeatureStepSettings

    def __init__(self, settings: FeatureStepSettings) -> None:
        super().__init__(settings)
        from layout_visual_complexity.features.contract import load_families

        self.families = load_families(list(settings.families), dict(settings.parameters))

    def result_settings(self) -> dict[str, Any]:
        return {
            "families": [[family.name, family.settings_hash()] for family in self.families],
            "scored": list(self.settings.scored),
        }

    def load(self, device: str) -> None:
        super().load(device)
        from layout_visual_complexity.segmentation.steps import set_threads

        set_threads(self.threads)

    def process(self, pages: list[PageRecord]) -> list[PageRecord]:
        from layout_visual_complexity.features.extraction import PREPARE, measure_page

        scored = set(self.settings.scored)
        image_scored = any(family.needs & {"image", "folio_crop", "ink"} for family in self.families
                           if family.name in scored)  # fmt: skip
        for record in pages:
            try:
                values, _, errors = measure_page(page_of(record), self.families)
            except Exception as error:  # one page failing fails alone
                record.status[self.name] = Status("failed", f"{type(error).__name__}: {error}")
                continue
            record.features = {name: float(value) for name, value in values.items()}
            record.feature_errors = errors
            failing = sorted(name for name in errors if name in scored or (name == PREPARE and image_scored))
            if failing:
                reasons = "; ".join(f"{name}: {errors[name]}" for name in failing)
                record.status[self.name] = Status("failed", reasons)
            else:
                record.status[self.name] = Status()
        return pages


@dataclass(frozen=True)
class TokenStepSettings:
    """The ViT head on the page's patch tokens.

    Args:
        head: The trained head (`head.npz`); its backbone, revision, block, input size and crop say how the tokens
            are made.
        others: Other heads taking the same tokens, whose raw scores the record keeps too, by their file's SHA-256.
        local_only: Load the backbone from the Hugging Face cache only, without asking its server.
        workers: Worker processes; one on the GPU by default.
        threads: Torch's threads in each worker.
    """

    head: Path = Path()
    others: list[Path] = field(default_factory=list)
    local_only: bool = False
    workers: int | None = None
    threads: int = 1


class TokensStep(Step):
    """The backbone's patch tokens of the folio crop, scored by the head; the tokens are dropped."""

    needs = frozenset({"image", "folio"})
    produces = frozenset({"vit"})
    gpu = True
    auto_device = "mps"
    # One worker's peak footprint with DINOv2 ViT-S/14 on Apple's GPU was 0.61 GB.
    memory_mb = {"cpu": 800, "mps": 800}
    speed_settings = frozenset({"workers", "threads", "local_only"})
    settings: TokenStepSettings

    def __init__(self, settings: TokenStepSettings) -> None:
        super().__init__(settings)
        from layout_visual_complexity.evaluation.vit_head import TrainedHead

        self.tokens = dict(TrainedHead.load(settings.head).tokens)

    def models(self) -> list[Path]:
        return [Path(self.settings.head), *(Path(other) for other in self.settings.others)]

    def version(self) -> str:
        try:
            engine = f"transformers {version('transformers')}"
        except PackageNotFoundError:
            engine = "transformers not installed"
        return f"{engine}; {super().version()}"

    def result_settings(self) -> dict[str, Any]:
        return {"tokens": self.tokens}

    def token_settings(self) -> Any:
        from layout_visual_complexity.features.vit import TokenSettings

        tokens = self.tokens
        return TokenSettings(tokens["backbone"], tokens["revision"], int(tokens["block"]), int(tokens["long_edge"]),
                             tokens["crop"])  # fmt: skip

    def make_backbone(self, device: str) -> Any:
        """The backbone the head's tokens come from, on `device`."""
        from layout_visual_complexity.features.vit import Backbone

        return Backbone(self.token_settings(), device, local_only=self.settings.local_only)

    def load(self, device: str) -> None:
        super().load(device)
        from layout_visual_complexity.evaluation.vit_head import TrainedHead
        from layout_visual_complexity.segmentation.steps import set_threads

        set_threads(self.threads)
        from layout_visual_complexity.runs import sha256

        self.head = TrainedHead.load(self.settings.head)
        self.others = {sha256(Path(path)): TrainedHead.load(path) for path in self.settings.others}
        self.backbone = self.make_backbone(device)
        described = self.backbone.describe()
        for head in [self.head, *self.others.values()]:
            head.check(described)

    def process(self, pages: list[PageRecord]) -> list[PageRecord]:
        from PIL import Image

        from layout_visual_complexity.features.vit import TokenGrid, crop_box, cropped

        crop = self.tokens["crop"]
        for record in pages:
            try:
                page = folio_page(record)
                box = crop_box(page, crop)
                with Image.open(str(record.image)) as opened:
                    rgb = np.asarray(opened.convert("RGB"))
                tokens, rows, columns = self.backbone.tokens(cropped(rgb, box, page.width, page.height))
                grid = TokenGrid(np.asarray(tokens, dtype=np.float16), rows, columns, box)
                patches = self.head.patch_scores(grid)
                others = {key: float(head.patch_scores(grid).mean()) for key, head in self.others.items()}
            except Exception as error:  # one page failing fails alone
                from layout_visual_complexity.segmentation.regions import out_of_memory

                if out_of_memory(error):
                    from layout_visual_complexity.pipeline.step import OutOfMemory

                    raise OutOfMemory(str(error)) from error
                record.status[self.name] = Status("failed", f"{type(error).__name__}: {error}")
                continue
            record.vit = {
                "score": float(patches.mean()),
                "rows": int(rows),
                "columns": int(columns),
                "box": [float(value) for value in box],
                "patches": encode(patches),
                "others": others,
            }
            record.status[self.name] = Status()
        return pages


def scoring_steps(features: FeatureStepSettings, tokens: TokenStepSettings) -> dict[str, Step]:
    """The scoring stages, by name, after the segmentation's."""
    return {"features": FeaturesStep(features), "tokens": TokensStep(tokens)}
