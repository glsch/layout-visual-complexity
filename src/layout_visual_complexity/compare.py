"""The `lvc compare` command: the probability that one page is judged more complex than another.

Both pages go through `lvc score`'s pipeline, into a folder in the user cache by default, where they stay: comparing
either again needn't segment it again. The probability is a sigmoid of the difference of the pages' z-scores, by the
scale fitted on the evaluations' out-of-fold scores, so it holds on new pages; it is symmetric, p(a, b) = 1 − p(b, a).
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2
import pandas as pd

from layout_visual_complexity.download import DEFAULT_CACHE, Cache, HttpClient, cached_json
from layout_visual_complexity.inputs.lists import Entry
from layout_visual_complexity.inputs.resolve import Resolver
from layout_visual_complexity.pipeline.step import Step
from layout_visual_complexity.score import ScoreSettings, read_scores, run_score
from layout_visual_complexity.scoring import ScoringModels


@dataclass(frozen=True)
class CompareSettings(ScoreSettings):
    """The probability that the first page is judged more complex than the second, by the ensemble and each ranker.

    Args:
        first: The first page: an image, an image URL, a IIIF image service, or a IIIF manifest with `first_page`.
        second: The second page, as the first.
        first_page: The first page within its manifest: its place from 1 (`5`), its label (`12r`; `label:12` for a
            label of digits only), or its canvas's id.
        second_page: The second page within its manifest, as the first.
        output: The folder the pages are scored into, where they stay for later comparisons.
        sources: Not used: the pages are `first` and `second`.
        jobs: Worker processes on the CPU; 0 means all cores.
        device: Where the models run: `auto`, `cpu` or `mps`.
        alto: Also write one ALTO 4 file per page.
        limit: Not used.
        inputs: Which pages to take from the sources.
        images: How page images are fetched and kept.
        http: How to fetch from servers.
        folio_box: YOLO's folio box.
        folio: The folio crop within it.
        regions: YOLO's regions.
        lines: Kraken's lines.
        cache: The download cache: manifests, image descriptions and page images.
        memory_gb: Memory the workers may use together; by default what the system says is available, less 1 GB.
        fetch_threads: Threads fetching images at once.
        ahead: Pages fetched and not yet done, at most.
        models: The rankers (a `lvc train --ensemble` folder); the package's own by default.
        families: Feature families measured beside the ranker's.
        parameters: Those families' settings.
        maps: Keep the head's patch scores as maps.
        profiles: Not used.
        thumbnail_size: Not used.
        image_credit: Not used.
        local_only: Load the ViT backbone from the Hugging Face cache only.
        feature_workers: Worker processes measuring the features.
        other_models: Not used.
    """

    first: str = ""
    second: str = ""
    first_page: str = ""
    second_page: str = ""
    output: Path = field(default_factory=lambda: DEFAULT_CACHE.expanduser() / "compare")


def entry(source: str, page: str) -> Entry:
    """A source with the page asked for: a canvas by its id, or a selection."""
    if not page:
        return Entry(source)
    if "://" in page:
        return Entry(source, canvases=[page])
    return Entry(source, selection=page)


def page_id(resolver: Resolver, source: str, page: str) -> str:
    """The id of the one page a source and its selection give."""
    documents = resolver.entry(entry(source, page))
    pages = [found for document in documents for found in document.pages]
    if len(pages) != 1:
        hint = "; choose one with --first_page or --second_page" if len(pages) > 1 else ""
        raise ValueError(f"{source} {page}".strip() + f" gives {len(pages)} pages, not one{hint}")
    return pages[0].page


def list_file(path: Path, pages: list[tuple[str, str]]) -> Path:
    """The two pages as a list of sources the pipeline reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["source", "pages", "canvas"])
        for source, page in pages:
            canvas = page if "://" in page else ""
            writer.writerow([source, "" if canvas else page, canvas])
    return path


def run_compare(
    settings: CompareSettings,
    steps: dict[str, Step] | None = None,
    transport: httpx2.BaseTransport | None = None,
    models: ScoringModels | None = None,
) -> dict[str, Any]:
    """Score both pages and give the probability that the first is judged more complex. Returns it with the pages'
    scores."""
    if not settings.first or not settings.second:
        raise ValueError("give two pages to compare")
    models = models or ScoringModels.load(settings.models)
    output = Path(settings.output).expanduser()
    cache = Path(settings.cache).expanduser()
    given = [(_source(settings.first), settings.first_page), (_source(settings.second), settings.second_page)]
    with HttpClient(settings.http, transport) as client:
        manifests = Cache(cache / "manifests", ".json")
        resolver = Resolver(settings.inputs, lambda url: cached_json(client, manifests, url))
        ids = [page_id(resolver, source, page) for source, page in given]
    if ids[0] == ids[1]:
        raise ValueError(f"both are the same page, {ids[0]}")
    sources = list_file(output / "pages.tsv", given)
    values = {name: getattr(settings, name) for name in ScoreSettings.__dataclass_fields__}
    values |= {"sources": [str(sources)], "output": output, "limit": None, "profiles": False, "other_models": {}}
    run_score(ScoreSettings(**values), steps, transport, models)
    scores = read_scores(output).set_index("page")
    missing = [page for page in ids if page not in scores.index]
    if missing:
        raise RuntimeError(f"{', '.join(missing)} couldn't be scored; see {output / 'report.md'}")
    first, second = scores.loc[ids[0]], scores.loc[ids[1]]
    probabilities = models.probability(first, second)
    print(describe(ids, first, second, probabilities, models))
    return {"pages": ids, "probabilities": probabilities, "scores": scores.loc[ids].reset_index().to_dict("records")}


def _source(text: str) -> str:
    """A source as the pipeline reads it: URLs as they are, local paths made absolute."""
    return text if "://" in text else str(Path(text).expanduser().resolve())


def describe(ids: list[str], first: pd.Series, second: pd.Series, probabilities: dict[str, float],
             models: ScoringModels) -> str:  # fmt: skip
    """The comparison in words."""
    names = [*models.names, "ensemble"]
    lines = []
    for label, page, row in (("first", ids[0], first), ("second", ids[1], second)):
        values = ", ".join(f"{name} {row[name]:+.2f}" for name in names)
        elo = f"; expected Elo score {row['expected_elo']:.0f}" if "expected_elo" in row else ""
        lines.append(f"{label}: {page} (z-scores: {values}{elo})")
    lines.append(
        "probability that the first is judged more complex: "
        + ", ".join(f"{name} {probabilities[name]:.3f}" for name in ["ensemble", *models.names])
    )
    return "\n".join(lines)
