"""The `lvc features` command: feature families measured on a segmentation run's pages, into one table.

Pages are measured in worker processes, a few at a time, each worker loading the families once. Every page's layout
is read once (its document's file, by the main process), its image once when a family needs it, and what the families
need is prepared once and shared among them. The results don't depend on the number of workers or the order pages
finish in. A rerun resumes: a page measured before with the same families, settings, layout and image is kept.

The table, `features.parquet`, has one row per page, `page` first, then each family's columns; its metadata records
each column's family and the family's class, settings and their hash.
"""

import hashlib
import json
import logging
import math
import os
import sqlite3
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from layout_visual_complexity.download import DEFAULT_CACHE, HttpClient, HttpSettings, ImageFetcher, ImageSettings
from layout_visual_complexity.features.contract import Family, aggregate, load_families
from layout_visual_complexity.features.page import Page
from layout_visual_complexity.inputs import PageSource
from layout_visual_complexity.logs import logger
from layout_visual_complexity.pipeline.monitor import Monitor, cores
from layout_visual_complexity.pipeline.step import usage
from layout_visual_complexity.runs import Run, shown_path
from layout_visual_complexity.segmentation.store import read_layout, read_pages

METADATA_KEY = b"layout_visual_complexity.features"
PREPARE = "prepare"


@dataclass(frozen=True)
class FeaturesSettings:
    """Measure feature families on a segmentation run's pages: one row per page.

    Args:
        layout: A segmentation run's folder: `layout/`, `pages.parquet` and the run's `config.yaml`.
        output: The run's folder: `features.parquet`, the report, the configuration and the manifest.
        families: The feature families: built-in ones by short name (`geometry`, `layout`), others by Python path
            (`package.module.Class`); comma-separated, or a list.
        parameters: Each family's settings, under the name the family is given by.
        jobs: Worker processes; 0 means all cores. The results don't depend on it.
        limit: Measure only the first this many pages.
        batch: Pages a worker measures at a time.
        cache: The download cache holding the page images; by default the segmentation run's.
        http: How to fetch page images that aren't in the cache.
    """

    layout: Path
    output: Path = Path("build/features")
    families: list[str] | str = field(default_factory=lambda: ["geometry", "layout"])
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    jobs: int = 0
    limit: int | None = None
    batch: int = 4
    cache: Path | None = None
    http: HttpSettings = field(default_factory=HttpSettings)


def family_names(families: Sequence[str] | str) -> list[str]:
    """Family names from a list, or from comma-separated text."""
    items = [families] if isinstance(families, str) else list(families)
    return [name.strip() for item in items for name in str(item).split(",") if name.strip()]


# The feature table.


@dataclass
class FeatureTable:
    """Feature values (pages × columns, indexed by page) and what computed them: each family as `describe` gives
    it, and each column's family."""

    values: pd.DataFrame
    families: list[dict[str, Any]]
    columns: dict[str, str]

    def hashes(self) -> dict[str, str]:
        """Each family's settings' hash, by name."""
        return {family["name"]: family["hash"] for family in self.families}

    def select(self, families: Sequence[str] | None) -> "FeatureTable":
        """Only the columns of these families; all when None or empty."""
        if not families:
            return self
        names = family_names(families)
        missing = sorted(set(names) - {family["name"] for family in self.families})
        if missing:
            raise ValueError(f"the feature table has no family {', '.join(missing)}")
        columns = [column for column in self.values.columns if self.columns[column] in names]
        kept = [family for family in self.families if family["name"] in names]
        return FeatureTable(self.values[columns], kept, {column: self.columns[column] for column in columns})


def write_feature_table(path: Path, values: pd.DataFrame, families: Sequence[Family]) -> Path:
    """`features.parquet`: `page` and the columns, with the families and each column's family in its metadata."""
    columns = {column: family.name for family in families for column in family.columns()}
    metadata = {"families": [family.describe() for family in families], "columns": columns}
    frame = values.reindex(columns=list(columns)).astype(float)
    frame.insert(0, "page", values.index.astype(str))
    table = pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), METADATA_KEY: json.dumps(metadata)})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    pq.write_table(table, partial, compression="zstd")
    partial.replace(path)
    return path


def read_feature_table(path: Path) -> FeatureTable:
    """A feature table and what computed it."""
    table = pq.read_table(path)
    raw = (table.schema.metadata or {}).get(METADATA_KEY)
    if raw is None:
        raise ValueError(f"{shown_path(path)} doesn't say which families computed it")
    metadata = json.loads(raw)
    values = table.to_pandas().set_index("page")
    values.index = values.index.astype(str)
    return FeatureTable(values.astype(float), metadata["families"], metadata["columns"])


# Measuring pages.

_families: list[Family] = []


def start_worker(names: list[str], parameters: dict[str, dict[str, Any]]) -> None:
    """Load the families once in a worker process."""
    global _families
    logging.getLogger("layout_visual_complexity").setLevel(logging.WARNING)
    _families = load_families(names, parameters)


def prepare(page: Page, families: Sequence[Family]) -> None:
    """Prepare once what the families need: the folio and the lines' shapes, the image."""
    needs = set().union(*(family.needs for family in families)) if families else set()
    page.folio_shape  # noqa: B018
    if needs & {"lines", "regions"}:
        page.line_shapes  # noqa: B018
        page.folio_lines  # noqa: B018
    if needs & {"image", "folio_crop"} and page.image_path is not None:
        page.image  # noqa: B018


def measure_page(page: Page, families: Sequence[Family]) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    """A page's values from every family, the seconds each took (and the preparation), and each failed family's
    error; a failed family's columns are NaN."""
    seconds: dict[str, float] = {}
    errors: dict[str, str] = {}
    values: dict[str, float] = {}
    began = time.perf_counter()
    try:
        prepare(page, families)
    except Exception as error:  # a page whose image can't be read keeps the families that don't need it
        errors[PREPARE] = f"{type(error).__name__}: {error}"
    seconds[PREPARE] = time.perf_counter() - began
    for family in families:
        began = time.perf_counter()
        try:
            values |= aggregate(family, family.measure(page))
        except Exception as error:  # a family failing on a page fails there alone
            errors[family.name] = f"{type(error).__name__}: {error}"
            values |= dict.fromkeys(family.columns(), math.nan)
        seconds[family.name] = time.perf_counter() - began
    return values, seconds, errors


def measure_batch(pages: list[Page]) -> dict[str, Any]:
    """A worker's batch: each page's values, seconds and errors, and the worker's CPU time and peak memory."""
    results = [(page.page, *measure_page(page, _families)) for page in pages]
    return {"results": results, "pid": os.getpid(), **usage()}


# Where the run's pages and images come from.


def page_key(families: Sequence[Family], page: Page, image: str | None) -> str:
    """What decides a page's values: the families' classes, versions and settings, its layout and its image."""
    digest = hashlib.sha256()
    digest.update(json.dumps([[family.name, family.settings_hash()] for family in families]).encode())
    digest.update(json.dumps([page.page, page.width, page.height, image]).encode())
    if page.folio is not None:
        digest.update(np.ascontiguousarray(page.folio).tobytes())
    for zone in [*page.regions, *page.yolo_lines]:
        digest.update(f"|{zone.id}|{zone.cls}|".encode())
        digest.update(np.ascontiguousarray(zone.polygon).tobytes())
    for line in page.lines:
        digest.update(f"|{line.id}|{line.region}|".encode())
        digest.update(np.ascontiguousarray(line.polygon).tobytes())
        digest.update(np.ascontiguousarray(line.baseline).tobytes())
    return digest.hexdigest()


class SegmentationImages:
    """The page images of a segmentation run: its sources resolved again, with its input and image settings, and
    each image fetched as the run did, from the download cache when it's there."""

    def __init__(self, folder: Path, cache: Path | None, http: HttpSettings) -> None:
        from layout_visual_complexity.inputs.resolve import InputSettings

        path = Path(folder) / "config.yaml"
        configuration = yaml.safe_load(path.read_text(encoding="utf-8")) or {} if path.is_file() else {}
        self.sources = list(configuration.get("sources") or [])
        self.inputs = InputSettings(**(configuration.get("inputs") or {}))
        self.images = ImageSettings(**(configuration.get("images") or {}))
        self.cache = Path(cache or configuration.get("cache") or DEFAULT_CACHE).expanduser()
        self.client = HttpClient(http)
        self.fetcher = ImageFetcher(self.client, self.cache, self.images)
        self._pages: dict[str, PageSource] | None = None

    def __enter__(self) -> "SegmentationImages":
        return self

    def __exit__(self, *exc: object) -> None:
        self.client.close()

    @property
    def pages(self) -> dict[str, PageSource]:
        if self._pages is None:
            from layout_visual_complexity.download import Cache, cached_json
            from layout_visual_complexity.inputs.resolve import Resolver

            manifests = Cache(self.cache / "manifests", ".json")
            resolver = Resolver(self.inputs, lambda url: cached_json(self.client, manifests, url))
            documents = resolver.resolve(self.sources) if self.sources else []
            self._pages = {page.page: page for document in documents for page in document.pages}
        return self._pages

    def identity(self, page: str) -> str | None:
        """What identifies a page's image: its service, URL or file."""
        found = self.pages.get(page)
        return found.image if found is not None else None

    def fetch(self, page: str) -> Any:
        """The page's image, pinned in the cache until released."""
        source = self.pages.get(page)
        if source is None:
            raise ValueError(f"{page} isn't among the segmentation run's sources")
        if source.image_service:
            return self.fetcher.from_service(source.image_service, source.width, source.height, source.image_api)
        if source.image_url:
            return self.fetcher.from_url(source.image_url)
        return self.fetcher.from_file(Path(source.path or ""))

    def release(self, image: Any) -> None:
        self.fetcher.release(image)


class State:
    """Pages measured so far, in `.state.sqlite` in the run's folder: each page's key, values, seconds and errors."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS pages (page TEXT PRIMARY KEY, key TEXT, values_json TEXT, seconds TEXT, "
            "errors TEXT, finished REAL)"
        )

    def keys(self) -> dict[str, str]:
        return dict(self.db.execute("SELECT page, key FROM pages"))

    def save(self, page: str, key: str, values: dict, seconds: dict, errors: dict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?, ?, ?)",
            (page, key, json.dumps(values), json.dumps(seconds), json.dumps(errors), time.time()),
        )

    def rows(self, pages: Sequence[str]) -> Iterator[tuple[str, dict, dict, dict]]:
        found = {
            page: (json.loads(values), json.loads(seconds), json.loads(errors))
            for page, values, seconds, errors in self.db.execute("SELECT page, values_json, seconds, errors FROM pages")
        }
        for page in pages:
            if page in found:
                yield page, *found[page]

    def close(self) -> None:
        self.db.commit()
        self.db.close()


def store_pages(folder: Path, limit: int | None) -> pd.DataFrame:
    """The segmentation run's pages that were done, in its order; the first `limit` of them."""
    pages = read_pages(Path(folder) / "pages.parquet")
    if "status" in pages.columns:
        pages = pages[pages["status"] == "done"]
    return pages.head(limit) if limit is not None else pages


def document_pages(folder: Path, document: str, wanted: pd.DataFrame) -> list[Page]:
    """The pages of one document's layout file, read once."""
    layout = read_layout(Path(folder) / "layout" / f"{document}.parquet")
    groups = dict(tuple(layout.groupby("page", sort=False)))
    pages = []
    for row in wanted.itertuples(index=False):
        rows = groups.get(row.page, layout.iloc[:0])
        pages.append(Page.from_layout(row.page, rows, row.width, row.height))
    return pages


def fetch_images(pages: list[Page], images: SegmentationImages, fetchers: ThreadPoolExecutor) -> dict[str, Any]:
    """The pages' images, fetched (or found in the cache) at once and pinned; each page takes its image's path. A
    page whose image can't be had keeps none, and the families needing it fail there."""

    def fetch(page: Page) -> tuple[Any, str | None]:
        try:
            return images.fetch(page.page), None
        except Exception as error:  # one page's image failing fails that page alone
            return None, f"{type(error).__name__}: {error}"

    fetched = {}
    for page, (image, error) in zip(pages, fetchers.map(fetch, pages), strict=True):
        if image is None:
            logger.warning("%s: no image: %s", page.page, error)
            continue
        page.image_path = image.path
        fetched[page.page] = image
    return fetched


# The run.


def run_features(settings: FeaturesSettings) -> dict[str, Any]:
    """Measure the families on the segmentation run's pages into `output`. Running it again resumes. Returns the
    run's summary."""
    began = time.monotonic()
    families = load_families(family_names(settings.families), settings.parameters)
    names = [family.name for family in families]
    run = Run("features", settings)
    layout_folder, output = Path(settings.layout), Path(settings.output)
    run.read(layout_folder / "pages.parquet")
    pages = store_pages(layout_folder, settings.limit)
    order = pages["page"].tolist()
    needs_image = any(family.needs & {"image", "folio_crop"} for family in families)
    jobs = settings.jobs or cores()
    state = State(output / ".state.sqlite")
    done_keys = state.keys()
    logger.info("%d pages, families %s", len(pages), ", ".join(names))

    images = SegmentationImages(layout_folder, settings.cache, settings.http)
    usage_by_worker: dict[int, dict[str, float]] = {}
    reported_cpu = [0.0]
    monitor = Monitor(lambda: reported_cpu[0] + time.process_time()).start()
    measured = skipped = 0
    try:
        with ProcessPoolExecutor(jobs, initializer=start_worker, initargs=(names, dict(settings.parameters))) as pool:
            pending: dict[Future, tuple[list[Page], list[str], list[Any]]] = {}

            def collect(finished: set[Future]) -> None:
                nonlocal measured
                for future in finished:
                    batch, keys, pinned = pending.pop(future)
                    outcome = future.result()
                    for (page, values, seconds, errors), key in zip(outcome["results"], keys, strict=True):
                        state.save(page, key, values, seconds, errors)
                        if errors:
                            logger.warning("%s: %s", page, "; ".join(f"{k}: {v}" for k, v in errors.items()))
                    measured += len(batch)
                    worker = outcome.get("pid", id(future))
                    usage_by_worker[worker] = {"cpu": outcome["cpu"], "peak_memory": outcome["peak_memory"]}
                    reported_cpu[0] = sum(used["cpu"] for used in usage_by_worker.values())
                    for image in pinned:
                        images.release(image)
                state.db.commit()

            with ThreadPoolExecutor(8, thread_name_prefix="images") as fetchers:
                for document, wanted in pages.groupby("document", sort=False):
                    document_batch = document_pages(layout_folder, str(document), wanted)
                    keys = [
                        page_key(families, page, images.identity(page.page) if needs_image else None)
                        for page in document_batch
                    ]
                    todo_pages = [
                        (page, key)
                        for page, key in zip(document_batch, keys, strict=True)
                        if done_keys.get(page.page) != key
                    ]
                    skipped += len(document_batch) - len(todo_pages)
                    fetched = fetch_images([page for page, _ in todo_pages], images, fetchers) if needs_image else {}
                    for start in range(0, len(todo_pages), max(1, settings.batch)):
                        chunk = todo_pages[start : start + max(1, settings.batch)]
                        batch = [page for page, _ in chunk]
                        pinned = [fetched[page.page] for page in batch if page.page in fetched]
                        future = pool.submit(measure_batch, batch)
                        pending[future] = (batch, [key for _, key in chunk], pinned)
                        while len(pending) > 4 * jobs:
                            done, _ = wait(pending, return_when=FIRST_COMPLETED)
                            collect(done)
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                collect(done)
    finally:
        monitor.stop()
        images.__exit__()

    rows = list(state.rows(order))
    state.close()
    values = pd.DataFrame({page: row for page, row, _, _ in rows}).T.reindex(index=order)
    written = [write_feature_table(output / "features.parquet", values, families)]
    seconds = pd.DataFrame({page: timing for page, _, timing, _ in rows}).T
    errors = {page: error for page, _, _, error in rows if error}
    wall = time.monotonic() - began
    summary = {
        "pages": len(order),
        "measured": measured,
        "kept_from_before": skipped,
        "seconds": wall,
        "pages_per_second": measured / wall if measured and wall else None,
        "workers": jobs,
        "seconds_per_page": {name: float(seconds[name].mean()) for name in [PREPARE, *names] if name in seconds},
        "machine": monitor.summary(),
        "peak_memory_per_worker": max((u["peak_memory"] for u in usage_by_worker.values()), default=None),
        "failed": errors,
        "families": [family.describe() for family in families],
        "columns": sum(len(family.columns()) for family in families),
    }
    manifest = layout_folder / "manifest.json"
    if manifest.is_file():
        summary["segmentation"] = json.loads(manifest.read_text(encoding="utf-8"))
    report = output / "report.md"
    report.write_text(report_text(summary, layout_folder), encoding="utf-8")
    written.append(report)
    run.wrote(written)
    run.summary = summary
    run.finish(output)
    print(f"{len(order):,} pages, {summary['columns']} columns ({measured:,} measured now) -> {output}")
    return summary


def report_text(summary: dict[str, Any], layout: Path) -> str:
    """The run's report, in Markdown."""
    machine = summary["machine"]
    gib = 2**30

    def number(value: Any, digits: int = 2) -> str:
        return "–" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:,.{digits}f}"

    lines = [
        "# Feature run",
        "",
        f"- Layout: `{shown_path(layout)}`; pages: {summary['pages']:,}; measured in this run: {summary['measured']:,}"
        f" ({summary['kept_from_before']:,} kept from before).",
        f"- Families: {', '.join(f['name'] for f in summary['families'])}; {summary['columns']} columns.",
        f"- Time: {summary['seconds']:.1f} s, {number(summary['pages_per_second'])} pages/s, "
        f"{summary['workers']} worker processes.",
        f"- Cores busy: {number(machine.get('cores_busy'))} of {machine.get('cores')} on average, "
        f"{number(machine.get('cores_busy_peak'))} at the busiest interval.",
        f"- Memory in use: {number((machine.get('memory_peak') or 0) / gib, 1)} GB at most "
        f"({number((machine.get('memory_before') or 0) / gib, 1)} GB before the run); "
        f"{number((summary['peak_memory_per_worker'] or 0) / gib, 2)} GB at most per worker.",
        "",
        "## Seconds per page",
        "",
        "| Step | Seconds per page |",
        "|---|---:|",
    ]
    lines += [f"| {name} | {number(value, 4)} |" for name, value in summary["seconds_per_page"].items()]
    lines += ["", "## Failures", ""]
    lines += [
        f"- {page}: " + "; ".join(f"{name}: {error}" for name, error in errors.items())
        for page, errors in summary["failed"].items()
    ] or ["None."]
    return "\n".join(lines) + "\n"
