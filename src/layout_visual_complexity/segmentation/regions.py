"""YOLO detections: the folio box and the page's regions. Needs the `segment` extra.

ultralytics is kept offline and away from the user's own files: before it is imported, its settings folder is pointed
into the package's cache, and its update checks, usage events and automatic installs are turned off.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from layout_visual_complexity.segmentation.records import Detection

IOU = 0.7
MAX_DETECTIONS = 300


@dataclass(frozen=True)
class FolioBoxSettings:
    """YOLO's folio box.

    Args:
        model: The YOLO model that finds folios (one class, `Folio`).
        image_size: The size images are scaled to for the model; by default the size it was trained at.
        confidence: Detections less confident than this are left out.
        batch_size: Pages per call to the model. More than one can change detections slightly, and was no faster on
            Apple's GPU.
        workers: Worker processes; by default one on a GPU, a share of `--jobs` on the CPU.
        threads: Torch threads per worker.
    """

    model: Path = Path("models/folio_yolo.pt")
    image_size: int | None = None
    confidence: float = 0.25
    batch_size: int = 1
    workers: int | None = None
    threads: int = 1


@dataclass(frozen=True)
class RegionSettings:
    """The page's regions, found by YOLO.

    Args:
        model: The YOLO model that finds regions.
        image_size: The size images are scaled to for the model; by default the size it was trained at.
        confidence: Detections less confident than this are left out.
        classes: The classes kept, a regular expression matched against the model's class names; by default its zones
            (the lines come from Kraken).
        batch_size: Pages per call to the model. More than one can change detections slightly, and was no faster on
            Apple's GPU.
        workers: Worker processes; by default one on a GPU, a share of `--jobs` on the CPU.
        threads: Torch threads per worker.
    """

    model: Path = Path("models/medieval-yolov11x.pt")
    image_size: int | None = None
    confidence: float = 0.25
    classes: str = r".*Zone"
    batch_size: int = 1
    workers: int | None = None
    threads: int = 1


def keep_ultralytics_offline(settings_folder: Path) -> None:
    """Point ultralytics' settings into `settings_folder` and turn off its network use. Call before importing it."""
    settings_folder.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings_folder)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_VERBOSE"] = "false"


def out_of_memory(error: BaseException) -> bool:
    """Whether torch ran out of memory on its device, so a smaller batch may fit."""
    return type(error).__name__ == "OutOfMemoryError" or "out of memory" in str(error).lower()


def trained_size(detector: Any) -> int:
    """The image size a YOLO model was trained at, from its saved arguments."""
    for arguments in (getattr(detector, "overrides", None), getattr(getattr(detector, "model", None), "args", None)):
        size = arguments.get("imgsz") if isinstance(arguments, dict) else getattr(arguments, "imgsz", None)
        if isinstance(size, int):
            return size
        if isinstance(size, list | tuple) and size:
            return int(max(size))
    return 640


class Yolo:
    """A YOLO detection model, loaded once, predicting as ultralytics does by default. Batches of more than one page
    can change detections slightly (the padding and the GPU's arithmetic depend on the batch), so a batch size is a
    setting that decides results."""

    def __init__(self, model: Path, device: str, cache: Path) -> None:
        keep_ultralytics_offline(Path(cache) / "ultralytics")
        from ultralytics import YOLO

        self.device = device
        self.detector: Any = YOLO(str(model), task="detect")
        self.names: dict[int, str] = dict(self.detector.names)
        self.trained_size = trained_size(self.detector)

    def detect(
        self, images: list[str], image_size: int | None, confidence: float, classes: str | None = None
    ) -> list[list[Detection]]:
        """Each image's detections, most confident first; `classes` keeps the classes it matches."""
        keep = re.compile(classes) if classes else None
        allowed = [i for i, name in self.names.items() if keep is None or keep.fullmatch(name)]
        try:
            results = self.detector.predict(
                images,
                imgsz=image_size or self.trained_size,
                conf=confidence,
                iou=IOU,
                max_det=MAX_DETECTIONS,
                classes=allowed,
                half=False,
                device=self.device,
                batch=len(images),
                verbose=False,
            )
        except Exception as error:
            if out_of_memory(error):
                from layout_visual_complexity.pipeline.step import OutOfMemory

                raise OutOfMemory(str(error)) from error
            raise
        found = []
        for result in results:
            boxes = result.boxes
            detections = [
                Detection(self.names[int(label)], tuple(float(v) for v in box), float(score))  # type: ignore[arg-type]
                for box, label, score in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), strict=True)
            ]
            found.append(sorted(detections, key=lambda d: (-d.confidence, d.box)))
        return found
