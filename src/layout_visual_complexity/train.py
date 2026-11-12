"""The `lvc train` command: a ranker trained on every judgment of a dataset, without the evaluation, for scoring other
pages: the pairwise logistic model on a feature table (`model.json`), or the local ViT head on a token set
(`head.npz`). Each keeps what its inputs must be, and the scale that turns a difference of its z-scored page scores
into the log-odds that the first page is chosen.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from layout_visual_complexity.evaluate import FeatureMethod, HeadMethod, Method
from layout_visual_complexity.evaluation.vit_head import HeadSettings
from layout_visual_complexity.ranking.dataset import read_dataset
from layout_visual_complexity.runs import Run


@dataclass(frozen=True)
class TrainSettings:
    """Train a ranker on every judgment: the pairwise logistic model on a feature table, or the local ViT head on a
    token set.

    Args:
        dataset: The dataset's folder: its judgments and images.
        output: The folder for the trained ranker (`model.json` or `head.npz`) and the run's record.
        features: A feature table (`features.parquet` of `lvc features`), for the pairwise logistic model.
        families: Only these families' columns: comma-separated, or a list; all by default.
        tokens: A token set (the folder of `lvc embed`), for the local ViT head.
        head: How the local ViT head is trained.
        regularisation: The pairwise logistic model's C: smaller is stronger.
    """

    dataset: Path
    output: Path
    features: Path | None = None
    families: list[str] | str = field(default_factory=list)
    tokens: Path | None = None
    head: HeadSettings = field(default_factory=HeadSettings)
    regularisation: float = 1.0


def run_train(settings: TrainSettings) -> dict[str, Any]:
    """Train the ranker and write it into `output`, with the run's record. Returns the run's summary."""
    from layout_visual_complexity.evaluate import EvaluateSettings

    began = time.perf_counter()
    run = Run("train", settings)
    for name in ("judgments.tsv", "images.tsv"):
        run.read(Path(settings.dataset) / name)
    if bool(settings.features) == bool(settings.tokens):
        raise ValueError("give one of --features or --tokens")
    dataset = read_dataset(settings.dataset)
    pages = dataset.images["image"].tolist()
    judgments = dataset.judgments.reset_index(drop=True)
    shared = EvaluateSettings(
        settings.dataset,
        Path(),
        settings.output,
        features=settings.features,
        families=settings.families,
        tokens=settings.tokens,
        head=settings.head,
        regularisation=settings.regularisation,
        jobs=1,
    )
    kind = FeatureMethod if settings.features else HeadMethod
    method: Method = kind(shared, run, pages, judgments)
    output = Path(settings.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        path, z, summary = method.train(output)
    finally:
        method.close()
    run.wrote(path)
    run.summary = {"method": method.kind, **summary, "pages": len(pages), "judgments": len(judgments)}
    run.summary["seconds"] = round(time.perf_counter() - began, 2)
    run.finish(output)
    print(f"trained on {len(judgments):,} judgments of {len(pages):,} pages -> {path}")
    return run.summary
