"""The `lvc codebook` command: the patch codebook of the generic image measures, learned once from a segmentation
run's pages, without any judgment.

Each page's folio crop is normalised as the generic measures take it (`features/folio_image.py`) and cut into square
patches; each patch is described by the histograms of its colours, and `samples` patches are drawn from each page, by
a generator seeded by the seed and the page's name. k-means, seeded, groups the drawn patches into `codewords`
codewords, taken in the pages' alphabetical order, so the codebook doesn't depend on the number of workers or the
order the pages finish in. The codebook (`codebook.npz`) is what the generic family's `codebook` setting names; the
package comes with the one learned on the dataset's pages.
"""

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from layout_visual_complexity.download import HttpSettings
from layout_visual_complexity.features.extraction import SegmentationImages, document_pages, fetch_images, store_pages
from layout_visual_complexity.features.folio_image import normalised_folio
from layout_visual_complexity.features.generic import Codebook, learn_codewords, page_sample, patch_histograms
from layout_visual_complexity.features.page import Page
from layout_visual_complexity.logs import logger
from layout_visual_complexity.pipeline.monitor import cores
from layout_visual_complexity.runs import Run, shown_path

CODEBOOK = "codebook.npz"


@dataclass(frozen=True)
class CodebookSettings:
    """Learn the patch codebook of the generic image measures from a segmentation run's pages: k-means over the colour
    histograms of patches drawn from every page, without any judgment.

    Args:
        layout: A segmentation run's folder: `layout/`, `pages.parquet` and the run's `config.yaml`.
        output: The codebook's folder: `codebook.npz`, the report and the run's record.
        codewords: The number of codewords (k).
        patch: The patches' side, in pixels of the folio crop.
        bins: Bins of each colour channel's histogram.
        long_edge: The folio crop's long edge, in pixels, as the generic family takes it.
        samples: Patches drawn from each page.
        seed: The seed of the draws and of k-means.
        jobs: Worker processes; 0 means all cores. The codebook doesn't depend on it.
        limit: Use only the first this many pages.
        cache: The download cache holding the page images; by default the segmentation run's.
        http: How to fetch page images that aren't in the cache.
    """

    layout: Path
    output: Path = Path("build/codebook")
    codewords: int = 32
    patch: int = 32
    bins: int = 16
    long_edge: int = 1400
    samples: int = 200
    seed: int = 1
    jobs: int = 0
    limit: int | None = None
    cache: Path | None = None
    http: HttpSettings = field(default_factory=HttpSettings)


def page_patches(page: Page, settings: CodebookSettings) -> tuple[str, np.ndarray | None, str | None]:
    """A page's drawn patch histograms, or the error that stopped it."""
    try:
        folio = normalised_folio(page, settings.long_edge)
        histograms = patch_histograms(folio, settings.patch, settings.bins)
        return page.page, page_sample(histograms, page.page, settings.samples, settings.seed), None
    except Exception as error:  # one page failing fails that page alone
        return page.page, None, f"{type(error).__name__}: {error}"


def run_codebook(settings: CodebookSettings) -> dict[str, Any]:
    """Learn the codebook from the segmentation run's pages into `output`. Returns the run's summary."""
    began = time.monotonic()
    run = Run("codebook", settings)
    layout, output = Path(settings.layout), Path(settings.output)
    run.read(layout / "pages.parquet")
    pages = store_pages(layout, settings.limit)
    jobs = settings.jobs or cores()
    samples: dict[str, np.ndarray] = {}
    failed: dict[str, str] = {}
    images = SegmentationImages(layout, settings.cache, settings.http)
    with images, ThreadPoolExecutor(8, thread_name_prefix="images") as fetchers, ProcessPoolExecutor(jobs) as pool:
        for document, wanted in pages.groupby("document", sort=False):
            batch = document_pages(layout, str(document), wanted)
            fetched = fetch_images(batch, images, fetchers)
            try:
                for page, sample, error in pool.map(page_patches, batch, [settings] * len(batch)):
                    if sample is None:
                        failed[page] = error or "no image"
                        logger.warning("%s: %s", page, error)
                    else:
                        samples[page] = sample
            finally:
                for image in fetched.values():
                    images.release(image)
    if not samples:
        raise ValueError("no page gave patches for the codebook")
    order = sorted(samples)
    learned = time.monotonic()
    centroids = learn_codewords([samples[page] for page in order], settings.codewords, settings.seed)
    learned = time.monotonic() - learned
    patches = int(sum(len(sample) for sample in samples.values()))
    about = {"pages": len(order), "patches": patches, "samples": settings.samples, "seed": settings.seed,
             "layout": shown_path(layout)}  # fmt: skip
    codebook = Codebook(centroids, settings.patch, settings.bins, settings.long_edge, about)
    written = [codebook.save(output / CODEBOOK)]
    wall = time.monotonic() - began
    summary = {
        "pages": len(order),
        "patches": patches,
        "codewords": settings.codewords,
        "failed": failed,
        "seconds": wall,
        "seconds_kmeans": learned,
        "workers": jobs,
    }
    report = output / "report.md"
    report.write_text(report_text(summary, settings), encoding="utf-8")
    written.append(report)
    run.wrote(written)
    run.summary = summary
    run.finish(output)
    print(f"{settings.codewords} codewords from {patches:,} patches of {len(order):,} pages -> {output / CODEBOOK}")
    return summary


def report_text(summary: dict[str, Any], settings: CodebookSettings) -> str:
    """The run's report, in Markdown."""
    lines = [
        "# Codebook run",
        "",
        f"- Layout: `{shown_path(settings.layout)}`; pages: {summary['pages']:,}; patches drawn: "
        f"{summary['patches']:,} ({settings.samples} at most a page, {settings.patch}-pixel patches of the folio crop "
        f"at {settings.long_edge:,} px, {settings.bins} bins a channel).",
        f"- Codewords: {summary['codewords']} (k-means, seed {settings.seed}), learned in "
        f"{summary['seconds_kmeans']:.1f} s; the run took {summary['seconds']:.1f} s with {summary['workers']} "
        "worker processes.",
        "",
        "## Failures",
        "",
    ]
    lines += [f"- {page}: {error}" for page, error in summary["failed"].items()] or ["None."]
    return "\n".join(lines) + "\n"
