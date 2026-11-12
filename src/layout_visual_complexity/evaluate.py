"""The `lvc evaluate` command: a feature table judged by the evaluation protocol, and the model trained on it.

The pairwise logistic model is cross-validated on folds by manuscript: accuracy by final Elo gap and overall (the mean
over folds, as the article reports it, and pooled), SROCC against the Elo scores beside the split-half ceiling,
log-loss and calibration, with 95% intervals by bootstrap over manuscripts; the consistency of the same test
judgments with the final Elo scores is the reference. Then the model is trained on every judgment and kept, with the
families and settings its features came from.
"""

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from layout_visual_complexity.evaluation import metrics
from layout_visual_complexity.evaluation.folds import FoldSettings, manuscript_folds
from layout_visual_complexity.evaluation.model import TrainedModel, fit_pairwise_model
from layout_visual_complexity.evaluation.protocol import (
    Evaluation,
    ProtocolSettings,
    bootstrap,
    evaluate,
    fold_mean_intervals,
)
from layout_visual_complexity.features.extraction import family_names, read_feature_table
from layout_visual_complexity.logs import logger
from layout_visual_complexity.ranking.dataset import read_dataset
from layout_visual_complexity.ranking.reliability import DEFAULT_GAP_EDGES, other_image
from layout_visual_complexity.runs import Run, shown_path

CORRECTED = "Spearman-Brown corrected"


@dataclass(frozen=True)
class EvaluateSettings:
    """Evaluate a feature table with the pairwise logistic model, cross-validated by manuscript, and train it.

    Args:
        features: A feature table (`features.parquet` of `lvc features`).
        dataset: The dataset's folder: its judgments, images and manuscripts.
        scores: The ranking's scores (`scores.tsv`), whose final Elo scores are the reference.
        output: The folder for the out-of-fold scores, the tables, the trained model (`model.json`) and the run's
            record.
        families: Only these families' columns: comma-separated, or a list; all by default.
        folds: The folds by manuscript.
        gap_edges: The edges of the bins of the final Elo gap.
        regularisation: The logistic regression's C: smaller is stronger.
        bootstrap: Resamples of the manuscripts for the 95% intervals.
        seed: The seed of the bootstrap.
        reliability_bins: Bins of the predicted probability in the calibration table.
        ceiling: A split-half table (`split_half_ceiling.tsv` of `lvc judgments report`), whose corrected median is
            the ceiling for SROCC.
    """

    features: Path
    dataset: Path
    scores: Path
    output: Path
    families: list[str] | str = field(default_factory=list)
    folds: FoldSettings = field(default_factory=FoldSettings)
    gap_edges: list[float] = field(default_factory=lambda: list(DEFAULT_GAP_EDGES))
    regularisation: float = 1.0
    bootstrap: int = 1000
    seed: int = 1
    reliability_bins: int = 10
    ceiling: Path | None = None


def read_elo(path: Path) -> pd.Series:
    """The final Elo scores (`elo_mean`) of a ranking's `scores.tsv`, by page."""
    table = pd.read_csv(path, sep="\t", dtype={"image": str})
    return table.set_index("image")["elo_mean"].astype(float)


def read_ceiling(path: Path) -> float:
    """The corrected median of a split-half table."""
    table = pd.read_csv(path, sep="\t")
    return float(table.set_index("measure").at[CORRECTED, "median"])


def train(features: pd.DataFrame, judgments: pd.DataFrame, regularisation: float) -> tuple:
    """The model on every judgment, its scores' mean and spread over the pages, and the scale of win probabilities."""
    model = fit_pairwise_model(features, judgments, regularisation)
    raw = model.scores(features)
    mean, sd = float(raw.mean()), float(raw.std(ddof=0))
    z = (raw - mean) / (sd if sd > 0 else 1.0)
    scale = metrics.fit_scale(z[judgments["chosen"]].to_numpy() - z[other_image(judgments)].to_numpy())
    return model, mean, sd, scale


def run_evaluate(settings: EvaluateSettings) -> dict[str, object]:
    """Write the evaluation's tables, the out-of-fold scores and the trained model into `output`, with the run's
    configuration and manifest. Returns the run's summary."""
    began = time.perf_counter()
    run = Run("evaluate", settings)
    for path in [Path(settings.features), Path(settings.scores)] + [
        Path(settings.dataset) / name for name in ("judgments.tsv", "images.tsv", "manuscripts.tsv")
    ]:
        run.read(path)
    table = read_feature_table(settings.features).select(family_names(settings.families))
    dataset = read_dataset(settings.dataset)
    elo = read_elo(settings.scores)
    pages = dataset.images["image"].tolist()
    missing = [page for page in pages if page not in table.values.index]
    if len(missing) == len(pages):
        raise ValueError("the feature table has none of the dataset's pages")
    if missing:
        logger.warning("%d pages have no features; the model fills them with the median", len(missing))
    features = table.values.reindex(pages)
    judgments = dataset.judgments.reset_index(drop=True)
    manuscripts = dataset.images.set_index("image")["manuscript"]
    edges = tuple(float(edge) for edge in settings.gap_edges)
    protocol = ProtocolSettings(
        edges, settings.regularisation, settings.bootstrap, settings.reliability_bins, settings.seed
    )
    fold = settings.folds
    folds = manuscript_folds(pages, manuscripts, judgments, fold.k, fold.repeats, fold.seed)
    logger.info("evaluate: %d columns, %d pages, %d folds", features.shape[1], len(pages), len(folds))
    evaluation = evaluate(features, judgments, elo, folds, protocol)
    tested = evaluation.tested
    agreement = metrics.hits(judgments, elo)
    method_values = tested.assign(value=tested["hit"])
    reference_values = tested.assign(value=agreement[tested["row"].to_numpy()])
    method_intervals = fold_mean_intervals(method_values, judgments, manuscripts, edges, protocol)
    reference_intervals = fold_mean_intervals(reference_values, judgments, manuscripts, edges, protocol)
    intervals = bootstrap(evaluation, judgments, elo, manuscripts, protocol)

    by_gap = by_elo_gap(evaluation, reference_values, method_intervals, reference_intervals, edges, len(folds))
    ceiling = read_ceiling(settings.ceiling) if settings.ceiling else math.nan
    summary_metrics = pd.DataFrame(
        [
            ("accuracy", evaluation.summary["accuracy"], *method_intervals.iloc[-1][["low", "high"]]),
            ("accuracy_pooled", evaluation.summary["accuracy_pooled"], intervals["accuracy_low"],
             intervals["accuracy_high"]),
            ("consistency", by_gap.iloc[-1]["consistency"], *reference_intervals.iloc[-1][["low", "high"]]),
            ("srocc", evaluation.summary["srocc"], intervals["srocc_low"], intervals["srocc_high"]),
            ("srocc_ceiling", ceiling, math.nan, math.nan),
            ("log_loss", evaluation.summary["log_loss"], math.nan, math.nan),
        ],
        columns=["metric", "value", "low", "high"],
    )  # fmt: skip

    model, mean, sd, scale = train(features, judgments, settings.regularisation)
    trained = TrainedModel(
        model,
        table.families,
        mean,
        sd,
        scale,
        settings.regularisation,
        {"dataset": shown_path(settings.dataset), "pages": len(pages), "judgments": len(judgments)},
    )

    output = Path(settings.output)
    output.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(
        {
            "page": pages,
            "manuscript": manuscripts.reindex(pages).to_numpy(),
            "score": evaluation.scores.reindex(pages).to_numpy(),
            "elo": elo.reindex(pages).to_numpy(),
        }
    )
    written = []
    for name, frame in [
        ("scores", scores),
        ("by_elo_gap", by_gap),
        ("metrics", summary_metrics),
        ("folds", evaluation.folds),
        ("calibration", evaluation.reliability),
    ]:
        path = output / f"{name}.tsv"
        frame.to_csv(path, sep="\t", index=False, lineterminator="\n", float_format="%.6f")
        written.append(path)
    written.append(trained.save(output / "model.json"))
    report = output / "report.md"
    report.write_text(report_text(settings, table.families, by_gap, summary_metrics, features.shape[1]), "utf-8")
    written.append(report)
    run.wrote(written)
    seconds = time.perf_counter() - began
    run.summary = {
        "columns": int(features.shape[1]),
        "families": [family["name"] for family in table.families],
        "pages": len(pages),
        "pages_without_features": len(missing),
        "judgments": len(judgments),
        "folds": len(folds),
        "tested_judgments": len(tested),
        "metrics": summary_metrics.to_dict("records"),
        "by_elo_gap": by_gap.to_dict("records"),
        "seconds": round(seconds, 2),
    }
    run.finish(output)
    accuracy, srocc = evaluation.summary["accuracy"], evaluation.summary["srocc"]
    print(f"accuracy {accuracy:.3f}, SROCC {srocc:.3f} over {len(folds)} folds -> {output}")
    return run.summary


def by_elo_gap(
    evaluation: Evaluation,
    reference_values: pd.DataFrame,
    method: pd.DataFrame,
    reference: pd.DataFrame,
    edges: tuple,
    folds: int,
) -> pd.DataFrame:
    """Per gap bin and overall: judgments per fold, the method's accuracy (mean over folds, its SD, pooled, and its
    interval) and the consistency reference on the same test judgments (likewise)."""
    per_fold = evaluation.by_gap.set_index("gap")
    rows = []
    reference_bins = metrics.gap_bin(reference_values["gap"].to_numpy(), edges)
    labels = [metrics.gap_label(low, high) for low, high in metrics.gap_bins(edges)]
    for number, label in enumerate([*labels, "overall"]):
        chosen = reference_bins == number if label != "overall" else np.ones(len(reference_values), dtype=bool)
        by_fold = reference_values[chosen].groupby("fold_number")["value"].mean()
        rows.append(
            {
                "gap": label,
                "judgments_per_fold": float(chosen.sum()) / max(folds, 1),
                "accuracy": per_fold.at[label, "fold_mean"],
                "accuracy_sd": per_fold.at[label, "fold_sd"],
                "accuracy_pooled": per_fold.at[label, "pooled"],
                "accuracy_low": method.set_index("gap").at[label, "low"],
                "accuracy_high": method.set_index("gap").at[label, "high"],
                "consistency": float(by_fold.mean()) if len(by_fold) else math.nan,
                "consistency_sd": float(by_fold.std(ddof=0)) if len(by_fold) else math.nan,
                "consistency_low": reference.set_index("gap").at[label, "low"],
                "consistency_high": reference.set_index("gap").at[label, "high"],
            }
        )
    return pd.DataFrame(rows)


def report_text(
    settings: EvaluateSettings, families: list[dict], by_gap: pd.DataFrame, summary: pd.DataFrame, columns: int
) -> str:
    """The evaluation in Markdown."""

    def interval(low: float, high: float) -> str:
        return "" if not (math.isfinite(low) and math.isfinite(high)) else f" [{low:.3f}, {high:.3f}]"

    values = summary.set_index("metric")
    lines = [
        "# Evaluation",
        "",
        f"- Features: `{shown_path(settings.features)}`, {columns} columns of "
        + ", ".join(f"{family['name']} (`{family['hash']}`)" for family in families)
        + ".",
        f"- Folds: {settings.folds.k} × {settings.folds.repeats} by manuscript (seed {settings.folds.seed}); "
        f"intervals: 95%, {settings.bootstrap:,} resamples of the manuscripts (seed {settings.seed}).",
        "",
        "| Metric | Value | 95% interval |",
        "|---|---:|---|",
    ]
    for metric, row in values.iterrows():
        if math.isfinite(row["value"]):
            lines.append(f"| {metric} | {row['value']:.3f} |{interval(row['low'], row['high'])} |")
    lines += [
        "",
        "| Elo gap | Judgments per fold | Accuracy | 95% interval | Consistency | 95% interval |",
        "|---|---:|---:|---|---:|---|",
    ]
    for row in by_gap.itertuples(index=False):
        lines.append(
            f"| {row.gap} | {row.judgments_per_fold:.1f} | {row.accuracy:.3f} |"
            f"{interval(row.accuracy_low, row.accuracy_high)} | {row.consistency:.3f} |"
            f"{interval(row.consistency_low, row.consistency_high)} |"
        )
    return "\n".join(lines) + "\n"


def read_metrics(folder: Path) -> dict[str, dict[str, float]]:
    """An evaluation's metrics, by name: value, low, high."""
    table = pd.read_csv(Path(folder) / "metrics.tsv", sep="\t").set_index("metric")
    return {metric: row.to_dict() for metric, row in table.iterrows()}
