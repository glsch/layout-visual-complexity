"""The segmentation steps: YOLO's folio box, the folio crop, YOLO's regions, and Kraken's lines.

The memory each worker adds was measured on Apple's M4 Pro: one worker's peak footprint (YOLO 2.2 GB on the GPU,
Kraken's network 5.5 GB, a line worker 0.95 GB) overstates it, since peaks don't come at once; twelve line workers
with the GPU steps added 11.3 GB to the memory in use. The figures below lie between the two.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar

from layout_visual_complexity.pipeline.step import Step
from layout_visual_complexity.segmentation.folio import FolioSettings, find_folio
from layout_visual_complexity.segmentation.lines import LineSettings
from layout_visual_complexity.segmentation.records import FolioRecord, PageRecord, Status
from layout_visual_complexity.segmentation.regions import FolioBoxSettings, RegionSettings

DEFAULT_CACHE = Path("~/.cache/layout-visual-complexity").expanduser()


def engine_version(distribution: str) -> str:
    """An engine's installed version, read without importing it."""
    try:
        return f"{distribution} {version(distribution)}"
    except PackageNotFoundError:
        return f"{distribution} not installed"


def set_threads(threads: int) -> None:
    """Torch's and OpenCV's threads in this worker; workers and `--jobs` govern the rest of the CPU."""
    import cv2

    cv2.setNumThreads(threads)
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(threads)


def failed(page: PageRecord, step: str, error: BaseException) -> None:
    page.status[step] = Status("failed", f"{type(error).__name__}: {error}")


class FolioBoxStep(Step):
    """YOLO's folio box: its most confident `Folio` detection, or none."""

    produces = frozenset({"folio_box"})
    speed_settings = frozenset({"workers", "threads"})
    gpu = True
    auto_device = "mps"
    memory_mb = {"cpu": 3000, "mps": 1500}
    settings: FolioBoxSettings

    def models(self) -> list[Path]:
        return [self.settings.model]

    def version(self) -> str:
        return f"{engine_version('ultralytics')}; {super().version()}"

    def load(self, device: str) -> None:
        super().load(device)
        set_threads(self.threads)
        from layout_visual_complexity.segmentation.regions import Yolo

        self.yolo = Yolo(self.settings.model, device, getattr(self, "cache", DEFAULT_CACHE))

    def process(self, pages: list[PageRecord]) -> list[PageRecord]:
        found = self.yolo.detect([page.image for page in pages], self.settings.image_size, self.settings.confidence)
        for page, detections in zip(pages, found, strict=True):
            folios = [detection for detection in detections if detection.cls == "Folio"]
            page.folio_box = folios[0] if folios else None
            page.status[self.name] = Status()
        return pages


class FolioStep(Step):
    """The folio crop: Otsu within YOLO's box, the box when Otsu's region is implausible, the image without a box."""

    needs = frozenset({"image", "folio_box"})
    produces = frozenset({"folio"})
    memory_mb = {"cpu": 300}
    settings: FolioSettings

    def load(self, device: str) -> None:
        super().load(device)
        set_threads(self.threads)

    def process(self, pages: list[PageRecord]) -> list[PageRecord]:
        import cv2

        for page in pages:
            try:
                image = cv2.imread(str(page.image), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("the image can't be read")
                box = page.folio_box.box if page.folio_box else None
                folio = find_folio(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), box, self.settings)
            except Exception as error:
                failed(page, self.name, error)
                continue
            page.folio = FolioRecord(folio.method, folio.quadrilateral, folio.iou)
            page.status[self.name] = Status()
        return pages


class RegionsStep(Step):
    """YOLO's regions: the classes the settings keep, as boxes, from the top of the page down."""

    produces = frozenset({"regions"})
    speed_settings = frozenset({"workers", "threads"})
    gpu = True
    auto_device = "mps"
    memory_mb = {"cpu": 1800, "mps": 1500}
    settings: RegionSettings

    def models(self) -> list[Path]:
        return [self.settings.model]

    def version(self) -> str:
        return f"{engine_version('ultralytics')}; {super().version()}"

    def load(self, device: str) -> None:
        super().load(device)
        set_threads(self.threads)
        from layout_visual_complexity.segmentation.regions import Yolo

        self.yolo = Yolo(self.settings.model, device, getattr(self, "cache", DEFAULT_CACHE))

    def process(self, pages: list[PageRecord]) -> list[PageRecord]:
        settings = self.settings
        found = self.yolo.detect(
            [page.image for page in pages], settings.image_size, settings.confidence, settings.classes
        )
        for page, detections in zip(pages, found, strict=True):
            page.regions = sorted(detections, key=lambda d: (round(d.box[1]), round(d.box[0]), -d.confidence))
            page.status[self.name] = Status()
        return pages


class KrakenStep(Step):
    """What Kraken's two steps share: the model, its version, and loading it."""

    settings: LineSettings
    speed_settings: ClassVar[frozenset[str]] = Step.speed_settings | {"map_workers"}

    def models(self) -> list[Path]:
        return [self.settings.model] if self.settings.model else [self.default_model()]

    @staticmethod
    def default_model() -> Path:
        from importlib import resources

        return Path(str(resources.files("kraken").joinpath("blla.mlmodel")))

    def version(self) -> str:
        return f"{engine_version('kraken')}; {super().version()}"

    def load(self, device: str) -> None:
        super().load(device)
        set_threads(self.threads)
        from layout_visual_complexity.segmentation.lines import Kraken

        self.kraken = Kraken(self.settings.model, device, self.settings.text_direction)

    def maps_folder(self) -> Path:
        return Path(getattr(self, "work", Path(".")))


class LineMapsStep(KrakenStep):
    """Kraken's network on the page, its heatmaps saved for the `lines` step."""

    produces = frozenset({"line_map"})
    gpu = True
    auto_device = "mps"
    memory_mb = {"cpu": 3500, "mps": 3500}

    @property
    def workers(self) -> int | None:
        return self.settings.map_workers

    def process(self, pages: list[PageRecord]) -> list[PageRecord]:
        for page in pages:
            try:
                target = self.maps_folder() / f"{page.page}.npz"
                page.line_map = str(self.kraken.maps(str(page.image), target))
            except Exception as error:
                from layout_visual_complexity.segmentation.regions import out_of_memory

                if out_of_memory(error):
                    from layout_visual_complexity.pipeline.step import OutOfMemory

                    raise OutOfMemory(str(error)) from error
                failed(page, self.name, error)
                continue
            page.status[self.name] = Status()
        return pages


class LinesStep(KrakenStep):
    """Kraken's lines from the saved heatmaps, in its reading order; the heatmaps are deleted once read."""

    needs = frozenset({"image", "line_map"})
    produces = frozenset({"lines"})
    memory_mb = {"cpu": 700}

    def process(self, pages: list[PageRecord]) -> list[PageRecord]:
        for page in pages:
            maps = Path(page.line_map) if page.line_map else None
            try:
                page.lines = self.kraken.lines(str(page.image), maps)
            except Exception as error:
                failed(page, self.name, error)
                continue
            finally:
                if maps is not None:
                    maps.unlink(missing_ok=True)
            page.status[self.name] = Status()
        return pages


def segmentation_steps(
    folio_box: FolioBoxSettings, folio: FolioSettings, regions: RegionSettings, lines: LineSettings
) -> dict[str, Any]:
    """The steps, by stage name, in order."""
    return {
        "folio_box": FolioBoxStep(folio_box),
        "folio": FolioStep(folio),
        "regions": RegionsStep(regions),
        "line_maps": LineMapsStep(lines),
        "lines": LinesStep(lines),
    }
