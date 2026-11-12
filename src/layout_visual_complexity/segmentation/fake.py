"""Fake steps, for trying the pipeline without models.

They do cheap, deterministic work from the page image alone, and can simulate what real steps meet: slow pages,
failed pages, batches too big for a device, and a worker that dies.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from layout_visual_complexity.pipeline.step import OutOfMemory, Step
from layout_visual_complexity.segmentation.records import Detection, Line, PageRecord, Status


@dataclass(frozen=True)
class FakeSettings:
    """What a fake step simulates.

    Args:
        fail_pages: Ids of pages the step marks failed.
        fail_while: A file: the pages in `fail_pages` fail only while it exists, as a passing failure would.
        raise_pages: Ids of pages on which the step raises, failing the whole batch.
        crash_pages: Ids of pages on which the worker process dies.
        delay: Seconds spent on each page.
        max_batch: Batches larger than this raise OutOfMemory.
        batch_size: Pages per batch.
        workers: Worker processes.
    """

    fail_pages: list[str] = field(default_factory=list)
    fail_while: str | None = None
    raise_pages: list[str] = field(default_factory=list)
    crash_pages: list[str] = field(default_factory=list)
    delay: float = 0.0
    max_batch: int | None = None
    batch_size: int = 3
    workers: int | None = None


class FakeStep(Step):
    speed_settings = Step.speed_settings | {"delay", "max_batch", "crash_pages", "fail_while"}
    memory_mb = {"cpu": 100, "mps": 100}
    settings: FakeSettings

    def __init__(self, settings: FakeSettings | None = None) -> None:
        super().__init__(settings or FakeSettings())

    def simulate(self, pages: list[PageRecord]) -> list[PageRecord]:
        """Wait, crash, raise or refuse the batch as set; return the pages to work on, the failed ones marked."""
        if self.settings.max_batch is not None and len(pages) > self.settings.max_batch:
            raise OutOfMemory(f"a batch of {len(pages)} pages is more than {self.settings.max_batch}")
        working = []
        for page in pages:
            if page.page in self.settings.crash_pages:
                os._exit(3)
            if page.page in self.settings.raise_pages:
                raise RuntimeError(f"fake error on {page.page}")
            time.sleep(self.settings.delay)
            failing = self.settings.fail_while is None or Path(self.settings.fail_while).exists()
            if page.page in self.settings.fail_pages and failing:
                page.status[self.name] = Status("failed", "fake failure")
            else:
                working.append(page)
        return working

    @staticmethod
    def size(page: PageRecord) -> tuple[int, int]:
        with Image.open(str(page.image)) as image:
            return image.size


class FakeFolioBox(FakeStep):
    """A folio box over the middle 80% of the image."""

    produces = frozenset({"folio_box"})

    def process(self, pages: list[PageRecord]) -> list[PageRecord]:
        for page in self.simulate(pages):
            width, height = self.size(page)
            page.folio_box = Detection("Folio", (0.1 * width, 0.1 * height, 0.9 * width, 0.9 * height), 0.9)
            page.status[self.name] = Status()
        return pages


class FakeRegions(FakeStep):
    """Two regions: a main zone over the upper two thirds, a margin below."""

    produces = frozenset({"regions"})

    def process(self, pages: list[PageRecord]) -> list[PageRecord]:
        for page in self.simulate(pages):
            width, height = self.size(page)
            page.regions = [
                Detection("MainZone", (0.1 * width, 0.1 * height, 0.9 * width, 0.6 * height), 0.8),
                Detection("MarginTextZone", (0.1 * width, 0.65 * height, 0.9 * width, 0.9 * height), 0.6),
            ]
            page.status[self.name] = Status()
        return pages


class FakeLines(FakeStep):
    """Six horizontal lines, evenly spaced down the page."""

    produces = frozenset({"lines"})

    def process(self, pages: list[PageRecord]) -> list[PageRecord]:
        for page in self.simulate(pages):
            width, height = self.size(page)
            page.lines = []
            for n in range(1, 7):
                y = n * height / 7
                boundary = [
                    (0.15 * width, y - 10),
                    (0.85 * width, y - 10),
                    (0.85 * width, y + 3),
                    (0.15 * width, y + 3),
                ]
                page.lines.append(Line([(0.15 * width, y), (0.85 * width, y)], boundary))
            page.status[self.name] = Status()
        return pages
