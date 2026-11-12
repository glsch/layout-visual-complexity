"""The `lvc embed` command: the patch tokens of a segmentation run's pages, from a vision transformer, cached for
training and evaluating the local ViT head.

Images are fetched (or found in the download cache), cropped and resized in threads while the model runs on its
device, one page at a time. Tokens go to the user cache (`features/vit.py`); the run's folder holds the token set:
`tokens.json` (the backbone, its revision, block, input size, crop and device, and where the cache is) and `pages.tsv`
(each page's grid, crop box and source), with the report and the run's record. A rerun keeps the pages whose tokens
are cached from the same image and crop.
"""

import json
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from layout_visual_complexity.download import DEFAULT_CACHE, HttpSettings
from layout_visual_complexity.features.extraction import SegmentationImages, document_pages, store_pages
from layout_visual_complexity.features.page import Page
from layout_visual_complexity.features.vit import (
    TOKEN_PAGES,
    TOKEN_SET,
    Backbone,
    Device,
    TokenCache,
    TokenGrid,
    TokenSettings,
    chosen_device,
    crop_box,
    cropped,
    source_key,
)
from layout_visual_complexity.logs import logger
from layout_visual_complexity.pipeline.monitor import Monitor
from layout_visual_complexity.runs import Run, shown_path


@dataclass(frozen=True)
class EmbedSettings:
    """Patch tokens of a segmentation run's pages from a vision transformer, cached for the local ViT head.

    Args:
        layout: A segmentation run's folder: `layout/`, `pages.parquet` and the run's `config.yaml`.
        output: The token set's folder: `tokens.json`, `pages.tsv`, the report and the run's record.
        tokens: The backbone, its revision and block, the input size and the crop.
        device: Where the model runs: `auto` (Apple's GPU or CUDA where there is one), `cpu`, `mps` or `cuda`.
        token_cache: The user cache the tokens are kept in, under `tokens/`.
        limit: Embed only the first this many pages.
        threads: Threads that fetch, crop and resize the images while the model runs.
        cache: The download cache holding the page images; by default the segmentation run's.
        http: How to fetch page images that aren't in the cache.
    """

    layout: Path
    output: Path = Path("build/embed")
    tokens: TokenSettings = field(default_factory=TokenSettings)
    device: Device = "auto"
    token_cache: Path = DEFAULT_CACHE
    limit: int | None = None
    threads: int = 6
    cache: Path | None = None
    http: HttpSettings = field(default_factory=HttpSettings)


@dataclass
class Prepared:
    """A page's input, ready for the model: its crop of the image, or the error that stopped it."""

    page: str
    box: tuple[float, float, float, float]
    source: str
    image: np.ndarray | None
    seconds: float
    error: str | None = None


def prepare(page: Page, box: tuple[float, float, float, float], source: str, images: SegmentationImages) -> Prepared:
    """Fetch the page's image (from the cache when it's there) and cut out its crop."""
    from PIL import Image

    began = time.perf_counter()
    try:
        fetched = images.fetch(page.page)
        try:
            with Image.open(fetched.path) as opened:
                rgb = np.asarray(opened.convert("RGB"))
        finally:
            images.release(fetched)
        image = cropped(rgb, box, page.width, page.height)
    except Exception as error:  # one page's image failing fails that page alone
        return Prepared(page.page, box, source, None, time.perf_counter() - began, f"{type(error).__name__}: {error}")
    return Prepared(page.page, box, source, image, time.perf_counter() - began)


def run_embed(settings: EmbedSettings) -> dict[str, Any]:
    """Embed the segmentation run's pages into the token cache and write the token set into `output`. Returns the
    run's summary."""
    began = time.monotonic()
    run = Run("embed", settings)
    layout_folder, output = Path(settings.layout), Path(settings.output)
    run.read(layout_folder / "pages.parquet")
    pages = store_pages(layout_folder, settings.limit)
    cache = TokenCache(settings.token_cache, settings.tokens)
    device = chosen_device(settings.device)
    images = SegmentationImages(layout_folder, settings.cache, settings.http)
    rows: list[dict[str, Any]] = []
    todo: list[tuple[Page, tuple[float, float, float, float], str]] = []
    for document, wanted in pages.groupby("document", sort=False):
        for page in document_pages(layout_folder, str(document), wanted):
            box = crop_box(page, settings.tokens.crop)
            source = source_key(images.identity(page.page), page.width, page.height, box)
            rows.append({"page": page.page, "width": page.width, "height": page.height, "source": source})
            if cache.read(page.page, source) is None:
                todo.append((page, box, source))
    logger.info("embed: %d pages, %d to embed on %s", len(rows), len(todo), device)

    backbone, described = None, stored_description(cache)
    if todo or described is None:
        backbone = Backbone(settings.tokens, device)
        described = backbone.describe()
        (cache.folder / "backbone.json").parent.mkdir(parents=True, exist_ok=True)
        (cache.folder / "backbone.json").write_text(json.dumps(described, indent=2) + "\n", encoding="utf-8")
    prepare_seconds, model_seconds, failed = [], [], {}
    monitor = Monitor(time.process_time).start()
    try:
        with ThreadPoolExecutor(max(1, settings.threads), thread_name_prefix="embed") as threads:
            window: list[Future[Prepared]] = []
            queue = iter(todo)
            for item in queue:
                window.append(threads.submit(prepare, *item, images))
                if len(window) >= 2 * max(1, settings.threads):
                    break
            done = 0
            while window:
                prepared = window.pop(0).result()
                following = next(queue, None)
                if following is not None:
                    window.append(threads.submit(prepare, *following, images))
                prepare_seconds.append(prepared.seconds)
                if prepared.image is None:
                    failed[prepared.page] = prepared.error
                    logger.warning("%s: %s", prepared.page, prepared.error)
                    continue
                started = time.perf_counter()
                tokens, grid_rows, grid_columns = backbone.tokens(prepared.image)  # type: ignore[union-attr]
                model_seconds.append(time.perf_counter() - started)
                grid = TokenGrid(tokens, grid_rows, grid_columns, prepared.box)
                cache.write(prepared.page, grid, prepared.source, device)
                done += 1
                if done % 50 == 0:
                    logger.info("embed: %d of %d pages", done, len(todo))
    finally:
        monitor.stop()
        images.__exit__()

    table = pd.DataFrame(rows)
    grids = []
    for row in table.itertuples(index=False):
        grid = cache.read(row.page, row.source)
        if grid is None:
            grids.append((math.nan,) * 6 + ("",))
            continue
        with np.load(cache.path(row.page), allow_pickle=False) as stored:
            made_on = str(stored["device"])
        grids.append((grid.rows, grid.columns, *grid.box, made_on))
    table[["rows", "columns", "x0", "y0", "x1", "y1", "device"]] = pd.DataFrame(grids, index=table.index)
    table = table[table["device"] != ""]
    output.mkdir(parents=True, exist_ok=True)
    devices = sorted(set(table["device"]))
    token_set = {
        **described,
        "cache_root": shown_path(Path(settings.token_cache).expanduser()),
        "cache": shown_path(cache.folder),
        "key": settings.tokens.key(),
        "dtype": "float16",
        "devices": devices,
        "pages": len(table),
    }
    written = [output / TOKEN_SET, output / TOKEN_PAGES]
    written[0].write_text(json.dumps(token_set, indent=2) + "\n", encoding="utf-8")
    table.to_csv(written[1], sep="\t", index=False, lineterminator="\n", float_format="%.3f")
    wall = time.monotonic() - began
    machine = monitor.summary()
    summary = {
        "pages": len(table),
        "embedded": len(model_seconds),
        "kept_from_before": len(rows) - len(todo),
        "failed": failed,
        "device": device,
        "seconds": wall,
        "pages_per_second": len(model_seconds) / wall if model_seconds and wall else None,
        "seconds_per_page": {
            "prepare": float(np.mean(prepare_seconds)) if prepare_seconds else None,
            "model": float(np.mean(model_seconds)) if model_seconds else None,
        },
        "tokens_per_page": float((table["rows"] * table["columns"]).median()) if len(table) else None,
        "megabytes": sum(cache.path(page).stat().st_size for page in table["page"]) / 2**20,
        "machine": machine,
        "tokens": token_set,
        "settings": asdict(settings.tokens),
    }
    report = output / "report.md"
    report.write_text(report_text(summary, layout_folder), encoding="utf-8")
    written.append(report)
    run.wrote(written)
    run.summary = summary
    run.finish(output)
    print(f"{len(table):,} pages, {summary['embedded']:,} embedded now on {device} -> {output}")
    return summary


def stored_description(cache: TokenCache) -> dict[str, Any] | None:
    path = cache.folder / "backbone.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def report_text(summary: dict[str, Any], layout: Path) -> str:
    """The run's report, in Markdown."""
    tokens = summary["tokens"]
    per_page = summary["seconds_per_page"]

    def number(value: Any, digits: int = 3) -> str:
        return "–" if value is None else f"{value:,.{digits}f}"

    lines = [
        "# Token run",
        "",
        f"- Layout: `{shown_path(layout)}`; pages: {summary['pages']:,}; embedded in this run: "
        f"{summary['embedded']:,} ({summary['kept_from_before']:,} kept from before).",
        f"- Backbone: `{tokens['backbone']}` at `{tokens['revision']}`, block {tokens['block']} of {tokens['blocks']}, "
        f"{tokens['patch_size']}-pixel patches, {tokens['dim']} dimensions; input {tokens['long_edge']} px on the long "
        f"edge, the {tokens['crop']} crop.",
        f"- Device: {summary['device']}; time {summary['seconds']:.1f} s, {number(summary['pages_per_second'], 2)} "
        f"pages/s; per page, {number(per_page['prepare'])} s to prepare (in threads) and {number(per_page['model'])} s "
        "in the model.",
        f"- Tokens: {number(summary['tokens_per_page'], 0)} a page at the median; {summary['megabytes']:,.1f} MB in "
        f"`{tokens['cache']}`.",
        f"- Cores busy: {number(summary['machine'].get('cores_busy'), 2)} of {summary['machine'].get('cores')} on "
        f"average; the GPU {number(summary['machine'].get('gpu_busy'), 2)} busy on average.",
        "",
        "## Failures",
        "",
    ]
    lines += [f"- {page}: {error}" for page, error in summary["failed"].items()] or ["None."]
    return "\n".join(lines) + "\n"
