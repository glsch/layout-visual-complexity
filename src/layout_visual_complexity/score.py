"""The `lvc score` command: pages scored as a stream, from their sources to their complexity scores.

Each page is fetched, cropped to its folio, segmented into regions and lines, measured by the feature families, and its
folio's patch tokens scored by the ViT head; the stages run at once, each page moving on as soon as it's ready, and its
image leaves the bounded download cache once it is done. The tokens live in memory only.

The run's folder keeps the scores (`scores.tsv`: each page's id, document, label, canvas and place in its document,
each ranker's z-score, the ensemble's and its expected Elo score), the layout store (`layout/`, `pages.parquet`), the
features (`features.parquet`), and the run's record, whose manifest holds every model's checksum. On request, the
head's patch scores as small float16 maps (`maps/<document>.npz`), and each document's profile and the overview of
the documents (`figures/`). Scores are computed from the kept features and head scores at the end of every run, so
they don't depend on the number of workers or the order pages arrive in; a rerun resumes.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2
import numpy as np
import pandas as pd

from layout_visual_complexity.logs import logger
from layout_visual_complexity.pipeline.step import Step
from layout_visual_complexity.runs import sha256, shown_path
from layout_visual_complexity.score_steps import FeatureStepSettings, TokenStepSettings, decode, scoring_steps
from layout_visual_complexity.scoring import ScoringModels
from layout_visual_complexity.segment import Extension, SegmentSettings, run_segment
from layout_visual_complexity.segmentation.records import PageRecord
from layout_visual_complexity.segmentation.steps import segmentation_steps

SCORES = "scores.tsv"
FEATURES = "features.parquet"


@dataclass(frozen=True)
class ScoreSettings(SegmentSettings):
    """Score pages: fetch, segment, measure and rank them as a stream, keeping scores, layouts and features.

    Args:
        sources: IIIF manifests or image services, image URLs, local images or folders, list files, or a dataset's
            page table (`images.tsv`); given on the command line, or here.
        output: The run's folder: `scores.tsv`, `features.parquet`, `layout/` and `pages.parquet`, `maps/` and
            `figures/` on request, the report, the configuration and the manifest.
        jobs: Worker processes on the CPU; 0 means all cores. The results don't depend on it.
        device: Where the models run: `auto` picks for each model the faster of the GPU (MPS) and the CPU; `cpu` or
            `mps` force one.
        alto: Also write one ALTO 4 file per page.
        limit: Score only the first this many pages of the sources.
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
        models: The rankers (a `lvc train --ensemble` folder); the package's own by default.
        families: Feature families measured and kept beside the ranker's, which don't enter the scores (`colour`,
            to describe the pages' colour).
        parameters: Those families' settings, under the names they're given by.
        maps: Keep the head's patch scores as float16 maps, one file per document.
        profiles: Draw each document's profile and the overview of the documents.
        thumbnail_size: The profiles' thumbnails' longer edge, in pixels, as fetched from the IIIF image service.
        image_credit: A line crediting the page images, added to the captions of the profiles.
        local_only: Load the ViT backbone from the Hugging Face cache only.
        feature_workers: Worker processes measuring the features; by default a share of the cores.
        other_models: Other rankers (`lvc train --ensemble` folders) that score the same pages beside the run's, by
            name, into `scores_<name>.tsv`: their features' families and their head's tokens must be the run's.
    """

    output: Path = Path("build/score")
    models: Path | None = None
    families: list[str] = field(default_factory=list)
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    maps: bool = False
    profiles: bool = False
    thumbnail_size: int = 400
    image_credit: str = ""
    local_only: bool = False
    feature_workers: int | None = None
    other_models: dict[str, Path] = field(default_factory=dict)


def feature_families(models: ScoringModels, settings: ScoreSettings) -> FeatureStepSettings:
    """The ranker's families, with its settings, and the families asked for beside them."""
    scored = [family["name"] for family in models.families()]
    parameters = {family["name"]: dict(family["settings"]) for family in models.families()}
    extra = [name for name in settings.families if name not in scored]
    clash = sorted(set(settings.parameters) & set(scored))
    if clash:
        raise ValueError(f"the ranker's families keep its own settings: {', '.join(clash)}")
    paths = [family["class"] for family in models.families()]
    return FeatureStepSettings(
        families=[*paths, *extra],
        parameters={
            **{path: parameters[name] for path, name in zip(paths, scored, strict=True)},
            **dict(settings.parameters),
        },  # fmt: skip
        scored=scored,
        workers=settings.feature_workers,
    )


def score_table(models: ScoringModels, records: list[tuple[PageRecord, str, dict[str, Any]]],
                hashes: dict[str, str], head: str | None = None) -> pd.DataFrame:  # fmt: skip
    """Each page done: its id, document, label, canvas, place in its document, and its scores; by another head than
    the run's, by its file's SHA-256 (`head`), when given."""
    done = [(record, source) for record, status, source in records if status == "done"]
    columns = ["page", "document", "label", "canvas", "order"]
    if not done:
        return pd.DataFrame(columns=[*columns, *models.names, "ensemble", "expected_elo"])
    pages = [record.page for record, _ in done]
    features = pd.DataFrame([record.features for record, _ in done], index=pages, dtype=float)
    head_raw = pd.Series(
        [record.vit["score"] if head is None else record.vit["others"][head] for record, _ in done],  # type: ignore[index]
        index=pages,
        dtype=float,
    )
    scores = models.scores(features, hashes, head_raw)
    described = pd.DataFrame(
        [(record.page, record.document, source.get("label"), source.get("canvas"), record.position)
         for record, source in done], columns=columns,
    ).set_index("page")  # fmt: skip
    return described.join(scores).reset_index()


def write_maps(folder: Path, records: list[tuple[PageRecord, str, dict[str, Any]]]) -> list[Path]:
    """The head's patch scores, one `.npz` per document: each page's grid (float16, rows × columns) under its id, and
    its crop box in the full image's pixels under `<id>.box`."""
    by_document: dict[str, dict[str, np.ndarray]] = {}
    for record, status, _ in records:
        if status != "done" or not record.vit or not record.vit.get("patches"):
            continue
        vit = record.vit
        arrays = by_document.setdefault(record.document, {})
        arrays[record.page] = decode(vit["patches"], int(vit["rows"]), int(vit["columns"]))
        arrays[f"{record.page}.box"] = np.asarray(vit["box"], dtype=np.float32)
    written = []
    for document, arrays in by_document.items():
        path = folder / f"{document}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part.npz")
        np.savez_compressed(partial, **arrays)
        partial.replace(path)
        written.append(path)
    return written


def folder_bytes(folder: Path) -> dict[str, int]:
    """What a run's folder keeps, in bytes, by its parts."""
    found: dict[str, int] = {}
    for root, _, files in os.walk(folder):
        for name in files:
            path = Path(root) / name
            part = path.relative_to(folder).parts[0]
            found[part] = found.get(part, 0) + path.stat().st_size
    return found


class Scoring(Extension):
    """What `lvc score` adds to the segmentation: the scores, features and maps, and the profiles."""

    command = "score"
    title = "Scoring run"

    def __init__(self, settings: ScoreSettings, models: ScoringModels, features: FeatureStepSettings,
                 others: dict[str, ScoringModels] | None = None) -> None:  # fmt: skip
        self.settings, self.models, self.features = settings, models, features
        self.others = others or {}

    def inputs(self) -> list[Path]:
        found = list(self.models.files().values())
        return found + [path for other in self.others.values() for path in other.files().values()]

    def finish(self, output: Path, records: list[tuple[PageRecord, str, dict[str, Any]]],
               summary: dict[str, Any]) -> list[Path]:  # fmt: skip
        from layout_visual_complexity.features.contract import load_families
        from layout_visual_complexity.features.extraction import write_feature_table

        families = load_families(list(self.features.families), dict(self.features.parameters))
        hashes = {family.name: family.settings_hash() for family in families}
        table = score_table(self.models, records, hashes)
        written = [output / SCORES]
        table.to_csv(written[0], sep="\t", index=False, lineterminator="\n", float_format="%.6f")
        for name, other in self.others.items():
            found = score_table(other, records, hashes, sha256(other.files()[other.names[1]]))
            written.append(output / f"scores_{name}.tsv")
            found.to_csv(written[-1], sep="\t", index=False, lineterminator="\n", float_format="%.6f")
        done = [record for record, status, _ in records if status == "done"]
        values = pd.DataFrame([record.features for record in done], index=[record.page for record in done])
        written.append(write_feature_table(output / FEATURES, values, families))
        if self.settings.maps:
            written += write_maps(output / "maps", records)
        if self.settings.profiles and len(table):
            from layout_visual_complexity.manuscript_reports import score_run_figures

            written += score_run_figures(output, table, self.settings, self.models.description)
        errors = {record.page: record.feature_errors for record in done if record.feature_errors}
        kept = folder_bytes(output)
        summary["scoring"] = {
            "models": {
                name: {"path": shown_path(path), "sha256": entry["sha256"]} for name, path, entry in self.model_files()
            },  # fmt: skip
            "pages_scored": len(table),
            "documents": int(table["document"].nunique()) if len(table) else 0,
            "by_document": table.groupby("document", sort=False).size().to_dict() if len(table) else {},
            "feature_errors": errors,
            "kept_bytes": kept,
        }
        return written

    def model_files(self) -> list[tuple[str, Path, dict[str, Any]]]:
        return [(name, path, {"sha256": sha256(path)}) for name, path in self.models.files().items()]

    def report(self, summary: dict[str, Any]) -> list[str]:
        scoring = summary.get("scoring", {})
        kept = scoring.get("kept_bytes", {})
        lines = ["", "## Scores", ""]
        lines.append(f"- Pages scored: {scoring.get('pages_scored', 0):,} in {scoring.get('documents', 0)} documents.")
        lines += ["- Rankers:"] + [f"  - {name}: `{entry['path']}` (`{entry['sha256'][:16]}`)"
                                   for name, entry in scoring.get("models", {}).items()]  # fmt: skip
        total = sum(kept.values())
        parts = ", ".join(f"{part} {size / 2**20:,.1f} MB" for part, size in sorted(kept.items()))
        lines.append(f"- Kept: {total / 2**20:,.1f} MB in all, {parts}.")
        if scoring.get("pages_scored"):
            lines.append(f"  That is {total / scoring['pages_scored'] / 1024:,.1f} kB a page.")
        errors = scoring.get("feature_errors", {})
        if errors:
            lines += ["", "Families that failed on pages the ranker could still score:", ""]
            lines += [f"- {page}: " + "; ".join(f"{name}: {error}" for name, error in found.items())
                      for page, found in errors.items()]  # fmt: skip
        lines += ["", "## Pages per document", "", "| Document | Pages scored |", "|---|---:|"]
        lines += [f"| {document} | {count:,} |" for document, count in scoring.get("by_document", {}).items()]
        return lines


def run_score(
    settings: ScoreSettings, steps: dict[str, Step] | None = None, transport: httpx2.BaseTransport | None = None,
    models: ScoringModels | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Score the sources' pages into `output`. Running it again resumes. Returns the run's summary."""
    models = models or ScoringModels.load(settings.models)
    features = feature_families(models, settings)
    others = {name: ScoringModels.load(folder) for name, folder in settings.other_models.items()}
    for name, other in others.items():
        if [family["hash"] for family in other.families()] != [family["hash"] for family in models.families()]:
            raise ValueError(f"the rankers {name} take other features than the run's")
    if steps is None:
        heads = [other.files()[other.names[1]] for other in others.values()]
        tokens = TokenStepSettings(models.files()[models.names[1]], heads, local_only=settings.local_only)
        steps = segmentation_steps(
            settings.folio_box, settings.folio, settings.regions, settings.lines
        ) | scoring_steps(features, tokens)
    logger.info("score: rankers from %s", shown_path(models.folder))
    summary = run_segment(settings, steps, transport, Scoring(settings, models, features, others))
    scoring = summary.get("scoring", {})
    print(f"{scoring.get('pages_scored', 0):,} pages scored -> {Path(settings.output) / SCORES}")
    return summary


def read_scores(folder: Path) -> pd.DataFrame:
    """A score run's scores, in the run's order."""
    return pd.read_csv(Path(folder) / SCORES, sep="\t", dtype={"page": str, "document": str, "label": str})


def read_summary(folder: Path) -> dict[str, Any]:
    """A score run's summary, from its manifest."""
    return json.loads((Path(folder) / "manifest.json").read_text(encoding="utf-8")).get("summary", {})
