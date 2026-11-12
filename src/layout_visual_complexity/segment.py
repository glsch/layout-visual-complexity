"""The `lvc segment` command: pages fetched and segmented as a stream, into a layout store.

Fetching runs in threads; the folio box, the folio crop, the regions and the lines each run in worker processes, all
at once, each page moving on as soon as it's ready. A page leaves the pipeline once every step is done with it: its
lines are assigned to its regions, it's saved, and its image may leave the cache. A document's layout file is written
once all its pages are done.
"""

import hashlib
import itertools
import json
import logging
import platform
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx2

from layout_visual_complexity.download import (
    DEFAULT_CACHE,
    Cache,
    HttpClient,
    HttpSettings,
    ImageFetcher,
    ImageSettings,
    PageImage,
    cached_json,
)
from layout_visual_complexity.inputs import Document, PageSource
from layout_visual_complexity.inputs.resolve import InputSettings, Resolver
from layout_visual_complexity.logs import logger
from layout_visual_complexity.pipeline.monitor import Monitor, cores, memory_available, total_memory
from layout_visual_complexity.pipeline.scheduler import Placement, RunError, Scheduler, StageUsage
from layout_visual_complexity.pipeline.state import State
from layout_visual_complexity.pipeline.step import Step
from layout_visual_complexity.runs import Run, shown_path
from layout_visual_complexity.segmentation.folio import FolioSettings
from layout_visual_complexity.segmentation.lines import LineSettings, assign_lines
from layout_visual_complexity.segmentation.records import PageRecord
from layout_visual_complexity.segmentation.regions import FolioBoxSettings, RegionSettings
from layout_visual_complexity.segmentation.steps import segmentation_steps
from layout_visual_complexity.segmentation.store import STAGES, page_row, write_layout, write_pages

Device = Literal["auto", "cpu", "mps"]

# How the CPU's workers are shared among the steps placed on it, when the settings don't say.
CPU_SHARES = {"folio_box": 1.0, "folio": 0.5, "regions": 1.0, "line_maps": 1.5, "lines": 6.0}


@dataclass(frozen=True)
class SegmentSettings:
    """Segment pages: the folio crop, YOLO's regions and Kraken's lines, into a layout store.

    Args:
        sources: IIIF manifests or image services, image URLs, local images or folders, list files, or a dataset's
            page table (`images.tsv`); given on the command line, or here.
        output: The run's folder: `layout/<document>.parquet`, `pages.parquet`, `alto/` with `--alto`, the report,
            the configuration and the manifest.
        jobs: Worker processes on the CPU; 0 means all cores. The results don't depend on it.
        device: Where the models run: `auto` picks for each model the faster of the GPU (MPS) and the CPU; `cpu` or
            `mps` force one.
        alto: Also write one ALTO 4 file per page.
        limit: Segment only the first this many pages of the sources.
        inputs: Which pages to take from the sources.
        images: How page images are fetched and kept.
        http: How to fetch from servers.
        folio_box: YOLO's folio box.
        folio: The folio crop within it.
        regions: YOLO's regions.
        lines: Kraken's lines.
        cache: The download cache: manifests, image descriptions and page images.
        memory_gb: Memory the workers may use together; by default what the system says is available when the
            run starts, less 1 GB.
        fetch_threads: Threads fetching images at once, across hosts; each host keeps its own pace.
        ahead: Pages fetched and not yet done, at most; by default four per worker.
    """

    sources: list[str] = field(default_factory=list)
    output: Path = Path("build/segment")
    jobs: int = 0
    device: Device = "auto"
    alto: bool = False
    limit: int | None = None
    inputs: InputSettings = field(default_factory=InputSettings)
    images: ImageSettings = field(default_factory=ImageSettings)
    http: HttpSettings = field(default_factory=HttpSettings)
    folio_box: FolioBoxSettings = field(default_factory=FolioBoxSettings)
    folio: FolioSettings = field(default_factory=FolioSettings)
    regions: RegionSettings = field(default_factory=RegionSettings)
    lines: LineSettings = field(default_factory=LineSettings)
    cache: Path = DEFAULT_CACHE
    memory_gb: float | None = None
    fetch_threads: int = 8
    ahead: int | None = None


def free_memory_mb() -> float:
    """Megabytes free for the workers: what the system says is available, less 1 GB, at least 4 GB."""
    available = memory_available() or 0.75 * (total_memory() or 16 * 2**30)
    return max(4 * 1024, available / 2**20 - 1024)


def has_mps() -> bool:
    """Whether this machine has Apple's GPU (MPS): Apple silicon."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def place(steps: dict[str, Step], device: Device, jobs: int, memory_mb: float) -> list[Placement]:
    """Each step's device and number of workers.

    GPU steps go where `device` says (`auto`: the device measured faster for them), one worker each on the GPU unless
    their settings say otherwise. The CPU's `jobs` workers, less one for the main process and the GPU workers when
    there are any, are shared among the steps placed there; the largest pools shrink until every worker fits in
    `memory_mb`.
    """
    devices = {}
    for name, step in steps.items():
        chosen = "cpu" if not step.gpu or device == "cpu" else step.auto_device if device == "auto" else device
        if chosen == "mps" and not has_mps():
            chosen = "cpu" if device == "auto" else chosen
            if chosen == "mps":
                raise RunError("--device mps needs Apple's GPU, which this machine doesn't have")
        devices[name] = chosen
    on_cpu = [name for name in steps if devices[name] == "cpu" and steps[name].workers is None]
    shares = {name: CPU_SHARES.get(name, 1.0) for name in on_cpu}
    reserved = 1 if any(device != "cpu" for device in devices.values()) else 0
    given = sum(steps[name].workers or 0 for name in steps if devices[name] == "cpu")
    free = max(len(on_cpu), jobs - reserved - given)
    counts = {}
    for name, step in steps.items():
        if step.workers is not None:
            counts[name] = max(1, step.workers)
        elif devices[name] != "cpu":
            counts[name] = 1
        else:
            counts[name] = max(1, round(free * shares[name] / sum(shares.values())))

    def total() -> float:
        return sum(counts[name] * steps[name].memory(devices[name]) for name in steps)

    while total() > memory_mb:
        shrinkable = [name for name in on_cpu if counts[name] > 1]
        if not shrinkable:
            break
        largest = max(shrinkable, key=lambda name: counts[name] * steps[name].memory(devices[name]))
        counts[largest] -= 1
    return [Placement(name, devices[name], counts[name]) for name in steps]


def interleaved(documents: list[Document]) -> list[PageSource]:
    """Pages taking turns across documents, so no document waits for another and hosts overlap."""
    columns = [document.pages for document in documents]
    return [page for row in itertools.zip_longest(*columns) for page in row if page is not None]


def limited(documents: list[Document], limit: int | None) -> list[Document]:
    """The documents with only the first `limit` pages of them all."""
    if limit is None:
        return documents
    kept, left = [], limit
    for document in documents:
        if left <= 0:
            break
        kept.append(Document(document.id, document.source, document.pages[:left], document.label))
        left -= len(kept[-1].pages)
    return kept


def describe_source(page: PageSource) -> dict[str, Any]:
    return {
        "label": page.label,
        "source": shown_path(page.source) if "://" not in page.source else page.source,
        "canvas": page.canvas,
        "image_service": page.image_service,
        "image": page.image if "://" in page.image else shown_path(page.image),
    }


def settings_key(steps: dict[str, Step], checksums: dict[str, str], long_edge: int) -> str:
    """What decides a run's results: each step's class, version, settings and models, and the image size. The device
    doesn't: the CPU and Apple's GPU give the same layout, to the pixel."""
    payload: dict[str, Any] = {"long_edge": long_edge}
    for name, step in steps.items():
        settings = {key: value for key, value in step.result_settings().items() if key != "model"}
        payload[name] = {
            "class": f"{type(step).__module__}.{type(step).__qualname__}",
            "version": step.version(),
            "settings": settings,
            "models": [checksums[str(model)] for model in step.models()],
        }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def page_key(settings: str, page: PageSource) -> str:
    """The run's key with the page's own image."""
    identity = [settings, page.image, page.width, page.height]
    return hashlib.sha256(json.dumps(identity, default=str).encode()).hexdigest()


@dataclass
class Fetched:
    """A page's image as the scheduler takes it, with the seconds fetching took."""

    path: Path
    width: int
    height: int
    full_width: int
    full_height: int
    seconds: float


class Progress:
    """Pages done and failed, logged every `every` seconds."""

    def __init__(self, total: int, every: float = 15.0) -> None:
        self.total, self.every = total, every
        self.done = self.failed = 0
        self.began = self.last = time.monotonic()

    def add(self, failed: bool) -> None:
        self.done += 1
        self.failed += failed

    def tick(self) -> None:
        now = time.monotonic()
        if now - self.last < self.every:
            return
        self.last = now
        rate = self.done / max(now - self.began, 1e-9)
        left = (self.total - self.done) / rate / 60 if rate else float("nan")
        logger.info(
            "%d/%d pages, %d failed, %.2f pages/s, about %.0f min left", self.done, self.total, self.failed, rate, left
        )


def run_segment(
    settings: SegmentSettings, steps: dict[str, Step] | None = None, transport: httpx2.BaseTransport | None = None
) -> dict[str, Any]:
    """Segment the sources' pages into `output`. Running it again resumes. Returns the run's summary."""
    if not settings.sources:
        raise ValueError("no sources: give manifests, images, folders, lists or a page table")
    began = time.monotonic()
    output = Path(settings.output)
    output.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(output / "log.txt", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        return _run(settings, steps, transport, began)
    finally:
        logger.removeHandler(handler)
        handler.close()


def _run(
    settings: SegmentSettings, steps: dict[str, Step] | None, transport: httpx2.BaseTransport | None, began: float
) -> dict[str, Any]:
    output, cache = Path(settings.output), Path(settings.cache).expanduser()
    run = Run("segment", settings)
    for source in settings.sources:
        if "://" not in source and Path(source).is_file():
            run.read(Path(source))
    steps = steps or segmentation_steps(settings.folio_box, settings.folio, settings.regions, settings.lines)
    for step in steps.values():
        step.cache = cache  # type: ignore[attr-defined]
        step.work = output / ".maps"  # type: ignore[attr-defined]
    checksums = {}
    for step in steps.values():
        for model in step.models():
            if not Path(model).is_file():
                raise ValueError(f"the model {shown_path(model)} isn't there")
            if str(model) not in checksums:
                run.read(Path(model))
                checksums[str(model)] = run.inputs[-1]["sha256"]
    jobs = settings.jobs or cores()
    budget = settings.memory_gb * 1024 if settings.memory_gb else free_memory_mb()
    placements = place(steps, settings.device, jobs, budget)
    key = settings_key(steps, checksums, settings.images.long_edge)

    with HttpClient(settings.http, transport) as client:
        manifests = Cache(cache / "manifests", ".json")
        resolver = Resolver(settings.inputs, lambda url: cached_json(client, manifests, url))
        documents = limited(resolver.resolve(settings.sources), settings.limit)
        pages = interleaved(documents)
        logger.info("%d pages in %d documents", len(pages), len(documents))
        state = State(output / ".state.sqlite")
        done = state.done()
        keys = {page.page: page_key(key, page) for page in pages}
        sources = {page.page: describe_source(page) for page in pages}
        todo = [page for page in pages if done.get(page.page) != keys[page.page]]
        logger.info("%d pages to do, %d done before", len(todo), len(pages) - len(todo))
        for placement in placements:
            logger.info("%s: %d worker(s) on %s", placement.stage, placement.workers, placement.device)

        fetcher = ImageFetcher(client, cache, settings.images)
        workers = sum(placement.workers for placement in placements)
        slots = threading.Semaphore(settings.ahead or max(16, 4 * workers))
        stop = threading.Event()
        images: dict[str, PageImage] = {}
        progress = Progress(len(todo))
        alto_info = alto_steps(steps, checksums) if settings.alto else []
        todo_documents = {page.document for page in todo}

        def on_page(record: PageRecord) -> None:
            assign_lines(record.lines, record.regions)
            if record.line_map:
                Path(record.line_map).unlink(missing_ok=True)
                record.line_map = None
            image = images.pop(record.page, None)
            if image is not None:
                fetcher.release(image)
            state.save(record, keys[record.page], key, sources[record.page])
            progress.add(bool(record.failed))
            slots.release()
            if settings.alto and not record.failed:
                write_alto(output, record, sources[record.page], alto_info)

        def on_document(document: str) -> None:
            state.commit()
            write_document(output, state, key, document)

        scheduler = Scheduler(steps, placements, on_page, on_document)
        by_document: dict[str, list[PageRecord]] = {}
        for page in todo:
            by_document.setdefault(page.document, []).append(PageRecord(page.page, page.document, page.position))
        for document, records in by_document.items():
            scheduler.add_document(document, records)

        def fetch(page: PageSource) -> None:
            slots.acquire()
            if stop.is_set():
                return
            began_fetch = time.perf_counter()
            try:
                if page.image_service:
                    image = fetcher.from_service(page.image_service, page.width, page.height, page.image_api)
                elif page.image_url:
                    image = fetcher.from_url(page.image_url)
                else:
                    image = fetcher.from_file(Path(page.path or ""))
            except Exception as error:  # a page whose image fails fails alone
                scheduler.image_ready(page.page, str(error) or type(error).__name__)
                return
            images[page.page] = image
            seconds = time.perf_counter() - began_fetch
            sizes = (image.width, image.height, image.full_width, image.full_height)
            scheduler.image_ready(page.page, Fetched(image.path, *sizes, seconds))

        monitor = Monitor(lambda: scheduler.cpu_seconds() + time.process_time()).start()
        pool = ThreadPoolExecutor(max_workers=max(1, settings.fetch_threads), thread_name_prefix="fetch")
        try:
            for page in todo:
                pool.submit(fetch, page)
            finished = scheduler.run(progress.tick)
        finally:
            stop.set()
            for _ in range(len(todo) + 1):
                slots.release()
            pool.shutdown(cancel_futures=True)
            monitor.stop()
            state.commit()
            for leftover in (output / ".maps").glob("*.npz"):
                leftover.unlink(missing_ok=True)
        for document in {page.document for page in pages} - todo_documents:
            if not (output / "layout" / f"{document}.parquet").is_file():
                write_document(output, state, key, document)
        order = {page.page: number for number, page in enumerate(pages)}
        rows = [page_row(record, status, source, key) for record, status, source in state.rows(key)]
        rows.sort(key=lambda row: (order.get(row["page"], len(order)), row["document"], row["page"]))
        written = [write_pages(output / "pages.parquet", rows)]
        state.close()
        requests = list(client.records)

    summary = summarize(rows, scheduler.stage_usage(), monitor.summary(), requests, output, settings, todo, began)
    summary["stopped"] = not finished
    summary["settings_key"] = key
    summary["steps"] = {
        placement.stage: {
            "version": steps[placement.stage].version(),
            "device": placement.device,
            "workers": placement.workers,
        }
        for placement in placements
    }
    report = output / "report.md"
    report.write_text(report_text(summary), encoding="utf-8")
    written += [report, *sorted((output / "layout").glob("*.parquet"))]
    run.wrote(written)
    run.summary = summary
    run.finish(output)
    logger.info("done: %s", shown_path(report))
    return summary


def write_document(output: Path, state: State, key: str, document: str) -> Path | None:
    """A document's layout file, from every page done under the run's settings."""
    records = [record for record, status, _ in state.rows(key, document) if status == "done"]
    if not records:
        return None
    return write_layout(output / "layout" / f"{document}.parquet", records)


def alto_steps(steps: dict[str, Step], checksums: dict[str, str]) -> list[Any]:
    from layout_visual_complexity.outputs.alto import StepInfo

    return [
        StepInfo(
            name,
            step.version(),
            {k: v for k, v in step.result_settings().items() if k != "model"},
            {Path(model).name: checksums[str(model)] for model in step.models()},
        )
        for name, step in steps.items()
    ]


def write_alto(output: Path, record: PageRecord, source: dict[str, Any], steps: list[Any]) -> Path:
    from layout_visual_complexity.outputs.alto import page_alto

    path = output / "alto" / record.document / f"{record.page}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page_alto(record, source.get("label"), source.get("image"), steps), encoding="utf-8")
    return path


def summarize(
    rows: list[dict[str, Any]],
    usage: dict[str, StageUsage],
    machine: dict[str, Any],
    requests: list[Any],
    output: Path,
    settings: SegmentSettings,
    todo: list[PageSource],
    began: float,
) -> dict[str, Any]:
    """What a run reports: pages done and failed, each stage's pace, how busy the machine was, the store's size,
    and the pages where YOLO's folio box and Otsu's region disagree."""
    wall = time.monotonic() - began
    this_run = {page.page for page in todo}
    ran = [row for row in rows if row["page"] in this_run]
    done = [row for row in rows if row["status"] == "done"]
    stages = {}
    for stage in STAGES:
        seconds = [row[f"{stage}_seconds"] for row in ran if row[f"{stage}_seconds"] is not None]
        used = usage.get(stage)
        entry: dict[str, Any] = {
            "pages": len(seconds),
            "seconds_per_page": sum(seconds) / len(seconds) if seconds else None,
        }
        if used is not None:
            entry |= {
                "device": used.device,
                "workers": used.workers,
                "pages_per_second": used.workers * len(seconds) / sum(seconds) if seconds and sum(seconds) else None,
                "cpu_seconds_per_page": used.cpu / len(seconds) if seconds else None,
                "peak_memory_per_worker": used.peak_memory,
                "load_seconds": used.load,
            }
        stages[stage] = entry
    layout_bytes = sum(path.stat().st_size for path in (output / "layout").glob("*.parquet"))
    pages_bytes = (output / "pages.parquet").stat().st_size if (output / "pages.parquet").is_file() else 0
    disagreement = settings.folio.disagreement
    disagreements = sorted(
        ((row["page"], row["folio_iou"], row["folio_method"]) for row in done
         if row["folio_iou"] is not None and row["folio_iou"] < disagreement),
        key=lambda item: item[1],
    )  # fmt: skip
    methods: dict[str, int] = {}
    for row in done:
        methods[row["folio_method"]] = methods.get(row["folio_method"], 0) + 1
    fallbacks = [(row["page"], row["folio_method"]) for row in done if row["folio_method"] != "otsu"]
    statuses = [row.get("status") for row in ran]
    fetched = [request for request in requests if request.status == 200]
    return {
        "pages": len(rows),
        "pages_this_run": len(ran),
        "done": len(done),
        "done_this_run": sum(status == "done" for status in statuses),
        "failed": [(row["page"], row["status"]) for row in rows if row["status"] != "done"],
        "seconds": wall,
        "pages_per_second": len(ran) / wall if wall else None,
        "stages": stages,
        "machine": machine,
        "requests": {
            "count": len(requests),
            "ok": len(fetched),
            "refused": sum(1 for request in requests if request.status in (429, 503)),
            "bytes": sum(request.size for request in fetched),
            "median_seconds": sorted(r.seconds for r in fetched)[len(fetched) // 2] if fetched else None,
        },
        "store": {
            "layout_bytes": layout_bytes,
            "pages_bytes": pages_bytes,
            "bytes_per_page": (layout_bytes + pages_bytes) / len(done) if done else None,
        },
        "folio_methods": methods,
        "folio_fallbacks": fallbacks,
        "folio_box_used": any(row["folio_iou"] is not None for row in done),
        "folio_disagreements": disagreements,
        "disagreement_below": disagreement,
        "regions": sum(row["regions"] or 0 for row in done),
        "yolo_lines": sum(row["yolo_lines"] or 0 for row in done),
        "lines": sum(row["lines"] or 0 for row in done),
    }


def _number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "–"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def report_text(summary: dict[str, Any]) -> str:
    """The run's report, in Markdown."""
    machine = summary["machine"]
    gib = 2**30
    lines = [
        "# Segmentation run",
        "",
        f"- Pages: {summary['pages']:,} in the store, {summary['done']:,} done; this run {summary['pages_this_run']:,}"
        f" pages, {summary['done_this_run']:,} done, in {summary['seconds']:.0f} s"
        f" ({_number(summary['pages_per_second'])} pages/s)."
        + (" Stopped before the end." if summary["stopped"] else ""),
        f"- Regions {summary['regions']:,}, YOLO's lines {summary['yolo_lines']:,},"
        f" Kraken's lines {summary['lines']:,}.",
        f"- Cores busy: {_number(machine.get('cores_busy'))} of {machine.get('cores')} on average, "
        f"{_number(machine.get('cores_busy_peak'))} at the busiest interval.",
        f"- GPU busy: {_number(machine.get('gpu_busy'))} on average, {_number(machine.get('gpu_busy_peak'))} at most.",
        f"- Memory in use: {_number((machine.get('memory_peak') or 0) / gib, 1)} GB at most "
        f"({_number((machine.get('memory_before') or 0) / gib, 1)} GB before the run, "
        f"{_number((machine.get('memory_total') or 0) / gib, 0)} GB in all).",
        f"- Store: {_number(summary['store']['bytes_per_page'] and summary['store']['bytes_per_page'] / 1024, 1)} kB"
        f" per page ({summary['store']['layout_bytes']:,} bytes of layout, {summary['store']['pages_bytes']:,} of"
        " pages).",
        f"- Requests: {summary['requests']['count']:,}, {summary['requests']['refused']} refused, median"
        f" {_number(summary['requests']['median_seconds'])} s, {summary['requests']['bytes'] / 2**20:,.0f} MB.",
        "",
        "## Stages",
        "",
        "| Stage | Device | Workers | Seconds per page | Capacity (pages/s) | CPU seconds per page | Peak memory per"
        " worker (GB) | Load (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, entry in summary["stages"].items():
        lines.append(
            f"| {stage} | {entry.get('device', 'threads')} | {entry.get('workers', '–')} |"
            f" {_number(entry['seconds_per_page'], 3)} | {_number(entry.get('pages_per_second'))} |"
            f" {_number(entry.get('cpu_seconds_per_page'), 3)} |"
            f" {_number((entry.get('peak_memory_per_worker') or 0) / gib, 2)} |"
            f" {_number(entry.get('load_seconds'), 1)} |"
        )
    lines += ["", "## The folio crop", ""]
    lines.append(
        "Methods: " + ", ".join(f"{method} {count}" for method, count in sorted(summary["folio_methods"].items()))
    )
    names = {"image": "the whole image", "yolo": "YOLO's box"}
    lines += ["", "Pages where Otsu's quadrilateral was implausible, cropped to the whole image or YOLO's box:", ""]
    lines += [f"- {page}: {names.get(method, method)}" for page, method in summary["folio_fallbacks"]] or ["None."]
    if summary["folio_box_used"]:
        below = summary["disagreement_below"]
        lines += ["", f"Pages where YOLO's box and Otsu's quadrilateral disagree (IoU under {below}):", ""]
        if summary["folio_disagreements"]:
            lines += ["| Page | IoU | Method |", "|---|---:|---|"]
            lines += [f"| {page} | {iou:.3f} | {method} |" for page, iou, method in summary["folio_disagreements"]]
        else:
            lines.append("None.")
    lines += ["", "## Failures", ""]
    lines += [f"- {page}: {status}" for page, status in summary["failed"]] or ["None."]
    return "\n".join(lines) + "\n"
