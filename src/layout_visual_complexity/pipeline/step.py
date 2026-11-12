"""The step contract: what a step declares, and the methods it implements."""

import dataclasses
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, ClassVar

from layout_visual_complexity.segmentation.records import PageRecord, Status

IMAGE = "image"


class OutOfMemory(Exception):
    """A batch didn't fit on its device: the machinery halves the batch and tries again."""


class Step:
    """A pipeline step. Subclasses declare what they need and produce, and implement `load` and `process`.

    `settings` is the step's settings dataclass; its fields named in `speed_settings` never change results, and are
    left out when a run decides whether earlier results can be kept. `memory_mb` is what one worker adds to the
    memory in use, by device, as measured; `auto_device` is the device `--device auto` gives the step where a GPU is
    available.
    """

    needs: ClassVar[frozenset[str]] = frozenset({IMAGE})
    produces: ClassVar[frozenset[str]] = frozenset()
    gpu: ClassVar[bool] = False
    auto_device: ClassVar[str] = "cpu"
    memory_mb: ClassVar[dict[str, int]] = {"cpu": 500, "mps": 500}
    speed_settings: ClassVar[frozenset[str]] = frozenset({"batch_size", "workers", "threads"})
    revision: ClassVar[int] = 1

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.name = ""
        self.device = "cpu"

    @property
    def batch_size(self) -> int:
        return max(1, int(getattr(self.settings, "batch_size", 1)))

    @property
    def workers(self) -> int | None:
        return getattr(self.settings, "workers", None)

    @property
    def threads(self) -> int:
        return max(1, int(getattr(self.settings, "threads", 1)))

    def memory(self, device: str) -> int:
        """Megabytes one worker on `device` adds to the memory in use."""
        return self.memory_mb.get(device, max(self.memory_mb.values()))

    def models(self) -> list[Path]:
        """The model files the step loads, for the run's checksums."""
        return []

    def version(self) -> str:
        """What else decides the results: the engine's version and the step's own revision."""
        return f"{type(self).__name__} {self.revision}"

    def result_settings(self) -> dict[str, Any]:
        """The settings that decide results."""
        values = dataclasses.asdict(self.settings) if dataclasses.is_dataclass(self.settings) else {}
        return {key: value for key, value in values.items() if key not in self.speed_settings}

    def load(self, device: str) -> None:
        """Get ready to work on `device`: load models. Called once in each worker process."""
        self.device = device

    def process(self, pages: list[PageRecord]) -> list[PageRecord]:
        """Work on a batch of pages and return them. Mark a page failed in its status rather than raising."""
        raise NotImplementedError

    def merge(self, record: PageRecord, result: PageRecord) -> None:
        """Take what this step produced from `result` into `record`."""
        for product in self.produces:
            setattr(record, product, getattr(result, product))
        record.status[self.name] = result.status.get(self.name, Status())


def peak_footprint() -> float | None:
    """This process's largest physical footprint so far, in bytes, as macOS accounts for it: memory on Apple's GPU
    included, which the resident size leaves out. None elsewhere."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        library = ctypes.CDLL("/usr/lib/libproc.dylib")
        buffer = (ctypes.c_uint64 * 64)()
        if library.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buffer)) != 0:  # RUSAGE_INFO_V4
            return None
        return float(buffer[2 + 28])  # ri_lifetime_max_phys_footprint, after the 16-byte uuid
    except (OSError, AttributeError):
        return None


def usage() -> dict[str, float]:
    """This process's CPU seconds so far and its peak memory in bytes: its physical footprint on macOS, its resident
    size elsewhere."""
    footprint = peak_footprint()
    if footprint is None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        footprint = float(peak if sys.platform == "darwin" else peak * 1024)
    return {"cpu": time.process_time(), "peak_memory": footprint}
