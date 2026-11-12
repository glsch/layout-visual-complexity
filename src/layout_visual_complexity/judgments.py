"""The `lvc judgments` commands: import a published dataset, rank its pages, report on the judgments."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import httpx2
import yaml

from layout_visual_complexity import judgment_reports
from layout_visual_complexity.download import DEFAULT_CACHE, Cache, HttpClient, HttpSettings
from layout_visual_complexity.evaluation.folds import FoldSettings
from layout_visual_complexity.logs import logger
from layout_visual_complexity.ranking import graph
from layout_visual_complexity.ranking.dataset import copy_dataset, read_dataset, validate_dataset
from layout_visual_complexity.ranking.elo import BLOCK, EloSettings, workers
from layout_visual_complexity.ranking.reliability import DEFAULT_GAP_EDGES, spearman
from layout_visual_complexity.ranking.scores import compute_scores, read_scores, write_scores
from layout_visual_complexity.ranking.split_half import SplitHalfSettings
from layout_visual_complexity.runs import Run
from layout_visual_complexity.zenodo import fetch_record, record_of

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
    """Read a published dataset, check it, and copy it into the working folder.

    Args:
        source: A folder holding the dataset's files (`README.md`, `judgments.tsv`, `images.tsv`,
            `manuscripts.tsv`), or the DOI of its Zenodo record (`10.5281/zenodo.…`).
        output: The folder the dataset is copied into.
        cache: The download cache; a Zenodo record's files are kept in its `zenodo` folder.
        http: How to fetch from servers.
    """

    source: str
    output: Path
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
        category_labels: Names of the categories for the tables, by letter; by default the letters alone.
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
    """Check the dataset at `source` and copy it into `output`, with the configuration and a manifest. A DOI's
    files are fetched first, each checked against the record's MD5. Returns the run's summary."""
    run = Run("judgments import", settings)
    began = time.perf_counter()
    source = str(settings.source)
    if record_of(source):
        with HttpClient(settings.http, transport) as client:
            folder = fetch_record(source, client, Path(settings.cache) / "zenodo")
        run.summary["doi"] = source
    elif Path(source).is_dir():
        folder = Path(source)
    else:
        raise ValueError(f"{source} is neither a folder nor a Zenodo DOI")
    dataset = read_dataset(folder)
    validate_dataset(dataset)
    for name in ("README.md", "judgments.tsv", "images.tsv", "manuscripts.tsv"):
        run.read(folder / name)
    run.wrote(copy_dataset(folder, settings.output))
    judgments = dataset.judgments
    run.summary.update(
        judgments=len(judgments),
        sessions=int(judgments["session"].nunique()),
        annotators=int(judgments["annotator"].nunique()),
        images=len(dataset.images),
        manuscripts=len(dataset.manuscripts),
        seconds=round(time.perf_counter() - began, 3),
    )
    run.wrote(run.finish(settings.output))
    print(
        f"{len(judgments):,} judgments in {judgments['session'].nunique():,} sessions by "
        f"{judgments['annotator'].nunique():,} annotators; {len(dataset.images):,} pages from "
        f"{len(dataset.manuscripts):,} manuscripts -> {settings.output}"
    )
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
