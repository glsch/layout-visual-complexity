"""Text lines: Kraken's baseline segmenter (`blla`) on the page image, and each line's region. Kraken needs the
`segment` extra; the assignment needs only shapely.

Kraken's network takes a fifth of a second on Apple's GPU, its vectorisation four seconds on one core. So the network
runs in one step (`line_maps`, on the GPU) and the vectorisation in another (`lines`, in CPU workers), the heatmaps
passed between them on disk.
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from shapely.geometry import Polygon, box
from shapely.validation import make_valid

from layout_visual_complexity.segmentation.records import Detection, Line


@dataclass(frozen=True)
class LineSettings:
    """Kraken's baseline segmenter.

    Args:
        model: A Kraken segmentation model; by default the one Kraken ships (`blla`).
        text_direction: The page's principal text direction.
        batch_size: Pages per batch sent to a worker; Kraken segments them one by one.
        workers: Worker processes turning heatmaps into lines; by default a share of `--jobs`.
        threads: Torch threads per worker.
        map_workers: Worker processes running Kraken's network; by default one on a GPU, a share of `--jobs` on the
            CPU.
    """

    model: Path | None = None
    text_direction: Literal["horizontal-lr", "horizontal-rl", "vertical-lr", "vertical-rl"] = "horizontal-lr"
    batch_size: int = 1
    workers: int | None = None
    threads: int = 1
    map_workers: int | None = None


def kraken_device(device: str) -> tuple[str, Any]:
    """Our device name as Kraken's accelerator and devices: `cpu`, `mps`, or `gpu` with the index."""
    if device.startswith("cuda"):
        return "gpu", [int(device.partition(":")[2] or 0)]
    return device, "auto"


class Kraken:
    """Kraken's segmenter, loaded once, in two halves that run in different processes: `maps` runs the network on a
    page (on the GPU, say) and saves its heatmaps; `lines` turns saved heatmaps into lines (on the CPU), with Kraken's
    own vectorisation and reading order, so the lines are those Kraken's `predict` gives. Kraken's regions are not
    kept."""

    def __init__(self, model: Path | None, device: str, text_direction: str) -> None:
        for name in ("kraken", "lightning", "lightning.fabric", "lightning.pytorch", "coremltools"):
            logging.getLogger(name).setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", module="coremltools")
        from kraken.configs import SegmentationInferenceConfig
        from kraken.tasks import SegmentationTaskModel

        self.model: Any = SegmentationTaskModel.load_model(model)
        if len(self.model.seg_models) != 1 or self.model.ro_models:
            raise ValueError("a Kraken model with one segmentation network and no reading-order model is needed")
        self.net: Any = self.model.seg_models[0]
        if not callable(getattr(self.net, "_compute_segmentation_map", None)):
            raise RuntimeError("this Kraken version doesn't run its segmentation network apart; Kraken 7 does")
        accelerator, devices = kraken_device(device)
        self.config = SegmentationInferenceConfig(
            text_direction=text_direction, accelerator=accelerator, device=devices, precision="32-true"
        )
        self.compute = self.net._compute_segmentation_map

    def maps(self, image: str, target: Path) -> Path:
        """Run the network on a page and save its heatmaps, the scaled page and the scale into `target` (`.npz`)."""
        import numpy as np
        import torch
        from PIL import Image

        if getattr(self.net, "_inf_config", None) is not self.config:
            self.net.prepare_for_inference(self.config)
        with Image.open(image) as page:
            page.load()
            with torch.inference_mode(), self.net._fabric.init_tensor():
                found = self.compute(page)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.stem + ".part.npz")
        np.savez(partial, heatmap=found["heatmap"], scal_im=found["scal_im"], scale=found["scale"])
        partial.replace(target)
        return target

    def lines(self, image: str, maps: Path | None = None) -> list[Line]:
        """The page's lines in Kraken's reading order, each with its baseline and boundary polygon; from the heatmaps
        `maps` saved, or with the network run here."""
        import numpy as np
        from PIL import Image

        if maps is not None:
            with np.load(maps) as saved:
                found = {
                    "heatmap": saved["heatmap"],
                    "cls_map": self.net.user_metadata["class_mapping"],
                    "bounding_regions": self.net.user_metadata.get("bounding_regions", None),
                    "scale": saved["scale"],
                    "scal_im": saved["scal_im"],
                }
            self.net._compute_segmentation_map = lambda _: found
        else:
            self.net._compute_segmentation_map = self.compute
        with Image.open(image) as page:
            page.load()
            segmentation = self.model.predict(page, self.config)
        found_lines = []
        for line in segmentation.lines:
            if line.baseline and line.boundary:
                baseline = [(float(x), float(y)) for x, y in line.baseline]
                boundary = [(float(x), float(y)) for x, y in line.boundary]
                found_lines.append(Line(baseline, boundary))
        return found_lines


def _shape(points: list[tuple[float, float]]) -> Any:
    polygon = Polygon(points)
    return polygon if polygon.is_valid else make_valid(polygon)


def assign_lines(lines: list[Line], regions: list[Detection]) -> list[Line]:
    """Each line assigned to the region its boundary overlaps most, the first one on a tie, or to none."""
    shapes = [box(*region.box) for region in regions]
    for line in lines:
        if len(line.boundary) < 3:
            line.region = None
            continue
        boundary = _shape(line.boundary)
        overlaps = [boundary.intersection(shape).area for shape in shapes]
        best = max(range(len(overlaps)), key=lambda i: (overlaps[i], -i), default=None)
        line.region = best if best is not None and overlaps[best] > 0 else None
    return lines
