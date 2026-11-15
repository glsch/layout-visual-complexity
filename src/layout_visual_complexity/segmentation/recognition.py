"""Text recognition: a Kraken recognition model (CATMuS Medieval's, say) on a page's lines, read from the
full-resolution image, the ground truth's own pixels. Needs the `segment` extra.

Two sets of lines are read on each transcribed page: the ground truth's (their polygons and baselines), and Kraken's own
lines, as the segmentation found them, scaled to the full image. Lines are cut out of the image and recognised one
page at a time, in the page's order and one line at a time unless asked otherwise, so a line's text depends neither on
the workers nor on the other lines.
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from layout_visual_complexity.pipeline.step import Step
from layout_visual_complexity.segmentation.records import PageRecord, Status
from layout_visual_complexity.segmentation.store import scaled, scaler

Points = list[tuple[float, float]]


@dataclass(frozen=True)
class RecognitionSettings:
    """Kraken's text recognition.

    Args:
        model: A Kraken recognition model (`.mlmodel` or `.safetensors`).
        line_batch: Lines recognised at once. Lines of a batch are padded to the longest, which changes a few texts
            (8 of 104 on one page, from 1 to 8), so each line alone by default; it was also the fastest on the CPU
            (42 lines/s, against 22 by 8).
        batch_size: Pages per batch sent to a worker.
        workers: Worker processes; by default a share of `--jobs`.
        threads: Torch threads per worker.
    """

    model: Path | None = None
    line_batch: int = 1
    batch_size: int = 1
    workers: int | None = None
    threads: int = 1


@dataclass(frozen=True)
class PageLines:
    """What a page's recognition reads: its image and the part of it the page is, and its ground-truth lines (id,
    polygon and baseline, in the page's own pixels)."""

    image: str
    region: tuple[int, int, int, int] | None
    lines: list[tuple[str, Points, Points]]


class Recogniser:
    """A Kraken recognition model, loaded once."""

    def __init__(self, model: Path, line_batch: int) -> None:
        for name in ("kraken", "lightning", "lightning.fabric", "lightning.pytorch", "coremltools"):
            logging.getLogger(name).setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", module="coremltools")
        warnings.filterwarnings("ignore", message=".*legacy polygon extractor.*")
        from kraken.configs import RecognitionInferenceConfig
        from kraken.tasks import RecognitionTaskModel

        self.model: Any = RecognitionTaskModel.load_model(model)
        # Lines cut out in this process, in their order, so batches are always made of the same lines.
        self.config = RecognitionInferenceConfig(
            accelerator="cpu", device="auto", precision="32-true", batch_size=line_batch, num_line_workers=0
        )

    def texts(self, image: Any, lines: list[tuple[str, Points, Points]]) -> list[str]:
        """Each line's text, in order; an empty text where a line can't be cut out."""
        from kraken.containers import BaselineLine, Segmentation

        if not lines:
            return []
        found = [
            BaselineLine(
                id=identifier,
                baseline=[(round(x), round(y)) for x, y in baseline],
                boundary=[(round(x), round(y)) for x, y in polygon],
            )
            for identifier, polygon, baseline in lines
        ]
        segmentation = Segmentation(
            type="baselines", imagename="", text_direction="horizontal-lr", script_detection=False, lines=found
        )
        return [record.prediction for record in self.model.predict(image, segmentation, self.config)]


def readable(polygon: Points, baseline: Points) -> bool:
    return len(polygon) >= 3 and len(baseline) >= 2


class RecognitionStep(Step):
    """Kraken's recognition of the ground truth's lines and of its own lines, on the full-resolution image; pages
    without transcriptions are left as they are."""

    needs = frozenset({"image", "lines"})
    produces = frozenset({"recognition"})
    # One worker's peak footprint was 1.9 GB on a page alone, 3.1 GB in the stream.
    memory_mb = {"cpu": 2000}
    speed_settings = frozenset({"batch_size", "workers", "threads"})
    settings: RecognitionSettings

    def __init__(self, settings: RecognitionSettings, pages: dict[str, PageLines] | None = None) -> None:
        super().__init__(settings)
        if settings.model is None:
            raise ValueError("recognition needs a model (--recognition.model)")
        # The pages' ground truth, by page: not a setting; each page's key holds its ground truth's checksum.
        self.pages = pages or {}

    def models(self) -> list[Path]:
        return [Path(self.settings.model)] if self.settings.model else []

    def version(self) -> str:
        from layout_visual_complexity.segmentation.steps import engine_version

        return f"{engine_version('kraken')}; {super().version()}"

    def load(self, device: str) -> None:
        super().load(device)
        from layout_visual_complexity.segmentation.steps import set_threads

        set_threads(self.threads)
        self.recogniser = Recogniser(Path(self.settings.model or ""), self.settings.line_batch)

    def process(self, pages: list[PageRecord]) -> list[PageRecord]:
        from PIL import Image

        for record in pages:
            given = self.pages.get(record.page)
            if given is None or not given.lines:
                record.recognition = None
                record.status[self.name] = Status()
                continue
            try:
                with Image.open(given.image) as opened:
                    opened.load()
                    image = opened.crop(given.region) if given.region is not None else opened.copy()
                scale = scaler(record)
                own = [
                    (
                        f"l{number}",
                        [tuple(point) for point in scaled(line.boundary, scale)],
                        [tuple(point) for point in scaled(line.baseline, scale)],
                    )
                    for number, line in enumerate(record.lines, start=1)
                ]
                truth = [line for line in given.lines if readable(line[1], line[2])]
                kept = [line for line in own if readable(line[1], line[2])]
                truth_texts = self.recogniser.texts(image, truth)
                own_texts = dict(zip((line[0] for line in kept), self.recogniser.texts(image, kept), strict=True))
            except Exception as error:  # one page failing fails alone
                record.status[self.name] = Status("failed", f"{type(error).__name__}: {error}")
                continue
            record.recognition = {
                "ground_truth": {line[0]: text for line, text in zip(truth, truth_texts, strict=True)},
                "lines": [own_texts.get(f"l{number}", "") for number in range(1, len(record.lines) + 1)],
            }
            record.status[self.name] = Status()
        return pages
