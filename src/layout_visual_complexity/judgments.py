"""The `lvc judgments` commands: import the annotation app's export, rank the images, report on the judgments."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import httpx2
import yaml

from layout_visual_complexity import judgment_reports
from layout_visual_complexity.download import DEFAULT_CACHE, Cache, HttpClient, HttpSettings
from layout_visual_complexity.inputs.canvases import resolve_canvases
from layout_visual_complexity.logs import logger
from layout_visual_complexity.ranking import graph
from layout_visual_complexity.ranking.annotators import read_merge_table
from layout_visual_complexity.ranking.dataset import (
    IMAGE_COLUMNS,
    Rules,
    build_dataset,
    build_judgments,
    read_dataset,
    read_image_mapping,
    read_manuscript_metadata,
    readme,
    write_dataset,
)
from layout_visual_complexity.ranking.elo import BLOCK, EloSettings, workers
from layout_visual_complexity.ranking.export import read_export
from layout_visual_complexity.ranking.reliability import DEFAULT_GAP_EDGES, spearman
from layout_visual_complexity.ranking.scores import compute_scores, read_scores, write_scores
from layout_visual_complexity.ranking.split_half import SplitHalfSettings
from layout_visual_complexity.runs import Run

Output = Literal[
    "dataset_categories",
    "reliability_by_elo_gap",
    "consistency_reference_by_fold",
    "split_half_ceiling",
    "comparison_graph",
    "elo_score_distributions",
]
ALL_OUTPUTS: list[Output] = list(get_args(Output))


@dataclass(frozen=True)
class ImportSettings:
    """Turn the annotation app's export into the anonymised dataset.

    Args:
        export: The app's export (JSON).
        mapping: The images: each file's shelfmark and IIIF manifest (JSON).
        metadata: The manuscripts' categorised metadata (JSON lines).
        output: The dataset folder to write.
        merges: The merge table (TSV with `spelling` and `person` columns). Without one, spellings are only folded.
        exclude_calibration: Drop the calibration judgments.
        exclude_sources: Drop the sessions of these sources, by name.
        canvases: Find each page's canvas in its IIIF manifest.
        unresolved: Where the pages whose canvas wasn't found are listed; by default `unresolved_canvases.tsv` in
            the output folder.
        cache: The download cache; manifests are kept in its `manifests` folder.
        http: How to fetch from servers.
    """

    export: Path
    mapping: Path
    metadata: Path
    output: Path
    merges: Path | None = None
    exclude_calibration: bool = True
    exclude_sources: list[str] = field(default_factory=list)
    canvases: bool = True
    unresolved: Path | None = None
    cache: Path = DEFAULT_CACHE
    http: HttpSettings = field(default_factory=HttpSettings)


@dataclass(frozen=True)
class RankSettings:
    """Score the images from a dataset's judgments: Elo over seeded permutations, and Bradley–Terry.

    Args:
        dataset: The dataset folder.
        output: The folder for `scores.tsv` and `score_distributions.npz`.
        permutations: How many random orders of the judgments Elo runs over.
        seed: Permutation m is drawn from `numpy.random.default_rng([seed, m])`.
        elo: Elo's parameters.
        bt_prior: Bradley–Terry's prior, in virtual matches per image against an average one, half of them won.
        anchor_factor: Anchors are the images compared more than this many times the median.
        jobs: Worker processes; 0 means all cores. The results don't depend on it.
    """

    dataset: Path
    output: Path
    permutations: int = 1000
    seed: int = 1
    elo: EloSettings = field(default_factory=EloSettings)
    bt_prior: float = 1.0
    anchor_factor: float = 3.0
    jobs: int = 0


@dataclass(frozen=True)
class FoldSettings:
    """Cross-validation folds over pages, each manuscript in one fold.

    Args:
        k: Folds per repeat.
        repeats: How many times the folds are drawn.
        seed: The seed of the folds.
    """

    k: int = 10
    repeats: int = 3
    seed: int = 1


@dataclass(frozen=True)
class ReportSettings:
    """Tables and figures about the judgments, from a dataset and the ranking run made from it.

    Args:
        dataset: The dataset folder.
        ranking: The ranking run's folder.
        output: Where `tables/`, `figures/` and the run's record (`runs/<name>/`) go.
        outputs: Which tables and figures to make.
        name: The run's name, for its record.
        gap_edges: The edges of the bins of the final Elo gap.
        folds: The folds of the consistency reference.
        split_half: The split-half ceiling.
        category_labels: Short names of the categories for the tables, by letter; by default their full names.
        position_bias_min_judgments: The annotators the position-bias table shows by name have at least this many
            judgments.
        graph_seed: The seed of the comparison graph's layout.
        thumbnails: How many page thumbnails the score distributions show.
        thumbnail_size: The thumbnails' longer edge, in pixels, as fetched from the IIIF image service.
        image_credit: A line crediting the page images, added to the captions of figures that show them.
        cache: The download cache; thumbnails are kept in its `thumbnails` folder.
        http: How to fetch from servers.
        jobs: Worker processes; 0 means all cores. The results don't depend on it.
    """

    dataset: Path
    ranking: Path
    output: Path
    outputs: list[Output] = field(default_factory=lambda: list(ALL_OUTPUTS))
    name: str = "report"
    gap_edges: list[float] = field(default_factory=lambda: list(DEFAULT_GAP_EDGES))
    folds: FoldSettings = field(default_factory=FoldSettings)
    split_half: SplitHalfSettings = field(default_factory=SplitHalfSettings)
    category_labels: dict[str, str] = field(default_factory=dict)
    position_bias_min_judgments: int = 100
    graph_seed: int = 1
    thumbnails: int = 10
    thumbnail_size: int = 200
    image_credit: str = ""
    cache: Path = DEFAULT_CACHE
    http: HttpSettings = field(default_factory=HttpSettings)
    jobs: int = 0


def run_import(settings: ImportSettings, transport: httpx2.BaseTransport | None = None) -> dict[str, object]:
    """Write the dataset: `judgments.tsv`, `images.tsv`, `manuscripts.tsv`, a README, the configuration and a
    manifest. Returns the run's summary."""
    run = Run("judgments import", settings)
    began = time.perf_counter()
    combinations = read_export(run.read(settings.export))
    merges = read_merge_table(run.read(settings.merges)) if settings.merges else {}
    rules = Rules(settings.exclude_calibration, list(settings.exclude_sources))
    judgments = build_judgments(combinations, merges, rules)
    images = read_image_mapping(run.read(settings.mapping))
    manuscripts = read_manuscript_metadata(run.read(settings.metadata))
    dataset = build_dataset(judgments.table, images, manuscripts)
    run.summary["import_seconds"] = round(time.perf_counter() - began, 3)
    notes = []
    if settings.canvases:
        began = time.perf_counter()
        cache = Cache(Path(settings.cache) / "manifests", ".json")
        with HttpClient(settings.http, transport) as client:
            filled, unresolved, counts = resolve_canvases(dataset.images, client, cache)
        dataset.images = filled[IMAGE_COLUMNS]
        unresolved_path = Path(settings.unresolved or Path(settings.output) / "unresolved_canvases.tsv")
        unresolved_path.parent.mkdir(parents=True, exist_ok=True)
        unresolved.to_csv(unresolved_path, sep="\t", index=False, lineterminator="\n")
        run.wrote(unresolved_path)
        run.summary["canvases"] = counts
        run.summary["canvas_seconds"] = round(time.perf_counter() - began, 3)
        notes.append(
            f"Canvases: {counts['canvases_found']:,} of {len(dataset.images):,} pages matched exactly one canvas "
            f"of their manifest by folio label; the other {counts['unresolved']:,} have no canvas here."
        )
        licences = "; ".join(
            f"{name or 'none given'} ({n:,} manifest{'' if n == 1 else 's'})"
            for name, n in counts["manifest_licences"].items()
        )
        notes.append(
            f"The page images are served by the libraries under their own terms. The manifests' licences: {licences}."
        )
    written = write_dataset(dataset, settings.output, readme(dataset, rules, judgments.counts, notes))
    run.wrote(written)
    run.summary.update(
        judgments=len(judgments.table),
        sessions=judgments.sessions,
        annotators=judgments.persons,
        images=len(dataset.images),
        manuscripts=len(dataset.manuscripts),
        counts_by_rule=judgments.counts.to_dict("records"),
    )
    run.wrote(run.finish(settings.output))
    print(
        f"{len(judgments.table):,} judgments in {judgments.sessions:,} sessions by {judgments.persons:,} annotators; "
        f"{len(dataset.images):,} images from {len(dataset.manuscripts):,} manuscripts -> {settings.output}"
    )
    if "canvases" in run.summary:
        counts = run.summary["canvases"]
        print(f"canvases: {counts['canvases_found']:,} found, {counts['unresolved']:,} unresolved")
    return run.summary


def run_rank(settings: RankSettings) -> dict[str, object]:
    """Write `scores.tsv` and `score_distributions.npz`, with the configuration and a manifest."""
    run = Run("judgments rank", settings)
    for name in ("judgments.tsv", "images.tsv"):
        run.read(Path(settings.dataset) / name)
    dataset = read_dataset(settings.dataset)
    images = dataset.images["image"].tolist()
    began = time.perf_counter()
    scores = compute_scores(
        dataset.judgments, images, settings.permutations, settings.seed, settings.elo, settings.bt_prior, settings.jobs
    )
    seconds = time.perf_counter() - began
    run.wrote(write_scores(scores, settings.output, settings.seed))
    described = graph.describe(dataset.judgments, images, settings.anchor_factor)
    rho = spearman(scores.table["elo_mean"].to_numpy(), scores.table["bt_score"].to_numpy())
    run.summary.update(
        judgments=len(dataset.judgments),
        images=len(images),
        permutations=settings.permutations,
        workers=min(workers(settings.jobs), -(-settings.permutations // BLOCK)),
        seconds=round(seconds, 3),
        bt_elo_spearman=rho,
        graph={key: value for key, value in vars(described).items() if key != "anchors"}
        | {"anchor_images": list(described.anchors)},
    )
    run.wrote(run.finish(settings.output))
    print(
        f"Elo over {settings.permutations:,} permutations of {len(dataset.judgments):,} judgments in {seconds:.1f} s; "
        f"Bradley-Terry against Elo: Spearman {rho:.3f} -> {settings.output}"
    )
    return run.summary


def ranking_elo_settings(folder: Path) -> EloSettings:
    """The Elo parameters a ranking run used, from its configuration."""
    configuration = yaml.safe_load((Path(folder) / "config.yaml").read_text(encoding="utf-8"))
    return EloSettings(**configuration.get("elo", {}))


def run_report(settings: ReportSettings, transport: httpx2.BaseTransport | None = None) -> dict[str, object]:
    """Write the chosen tables and figures under the output folder, and the run's record in `runs/<name>/`."""
    run = Run("judgments report", settings)
    for path in [Path(settings.dataset) / name for name in ("judgments.tsv", "images.tsv", "manuscripts.tsv")] + [
        Path(settings.ranking) / name for name in ("scores.tsv", "score_distributions.npz", "config.yaml")
    ]:
        run.read(path)
    dataset = read_dataset(settings.dataset)
    scores = read_scores(settings.ranking)
    if scores.table["image"].tolist() != dataset.images["image"].tolist():
        raise ValueError("the ranking run scored other images than the dataset lists")
    elo = ranking_elo_settings(settings.ranking)
    edges = tuple(float(edge) for edge in settings.gap_edges)
    folder = Path(settings.output)
    client = HttpClient(settings.http, transport)
    timings = {}
    try:
        for output in dict.fromkeys(settings.outputs):
            began = time.perf_counter()
            logger.info("report: %s", output)
            if output == "dataset_categories":
                written = judgment_reports.dataset_categories(dataset, dict(settings.category_labels), folder)
            elif output == "reliability_by_elo_gap":
                written = judgment_reports.reliability_by_elo_gap(
                    dataset, scores, edges, settings.position_bias_min_judgments, folder
                )
            elif output == "consistency_reference_by_fold":
                f = settings.folds
                written = judgment_reports.consistency_reference_by_fold(
                    dataset, scores, f.k, f.repeats, f.seed, edges, folder
                )
            elif output == "split_half_ceiling":
                written = judgment_reports.split_half_ceiling(dataset, settings.split_half, elo, settings.jobs, folder)
            elif output == "comparison_graph":
                anchor_factor = yaml.safe_load((Path(settings.ranking) / "config.yaml").read_text("utf-8")).get(
                    "anchor_factor", 3.0
                )
                written = judgment_reports.comparison_graph(dataset, scores, anchor_factor, settings.graph_seed, folder)
            elif output == "elo_score_distributions":
                cache = Cache(Path(settings.cache) / "thumbnails", ".jpg")
                written = judgment_reports.elo_score_distributions(
                    dataset,
                    scores,
                    settings.thumbnails,
                    settings.thumbnail_size,
                    settings.image_credit,
                    client,
                    cache,
                    folder,
                )
            else:
                raise ValueError(f"unknown output {output!r}")
            timings[output] = round(time.perf_counter() - began, 3)
            run.wrote(written.paths)
            run.summary[output] = written.summary
            print(f"{output}: {', '.join(str(path) for path in written.paths)}")
    finally:
        client.close()
    run.summary["seconds"] = timings
    run.wrote(run.finish(folder / "runs" / settings.name))
    return run.summary
