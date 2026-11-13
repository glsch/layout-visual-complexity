"""The `lvc evaluate` command: a method judged by the evaluation protocol, and the method trained on every judgment.

A method is the pairwise logistic model on a feature table, the local ViT head on a token set, or an ensemble of
earlier evaluations, which combines their z-scores fold by fold with weights from the configuration (equal by
default) or fitted on each fold's training judgments. It is cross-validated on folds by manuscript: accuracy by final
Elo gap and overall (the mean over folds, as the article reports it, and pooled), SROCC against the Elo scores beside
its ceiling, log-loss and calibration, with 95% intervals by bootstrap over manuscripts; the consistency of
the same test judgments with the final Elo scores is the reference. Each fold's hits and z-scores are kept, so that
two evaluations can be compared on the same resamples and combined into ensembles. On request, learning curves: the
method trained on shares of each fold's training judgments or images. Then the method is trained on every judgment
and kept, with what its inputs must be.
"""

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from layout_visual_complexity.evaluation import metrics
from layout_visual_complexity.evaluation.folds import Fold, FoldSettings, manuscript_folds
from layout_visual_complexity.evaluation.model import TrainedModel, fit_pairwise_model
from layout_visual_complexity.evaluation.protocol import (
    Evaluation,
    FoldScores,
    ProtocolSettings,
    bootstrap,
    ensemble_fold_scores,
    evaluate_fold_scores,
    fitted_weights,
    fold_mean_intervals,
    paired_difference,
    pairwise_fold_scores,
    reduced_folds,
)
from layout_visual_complexity.evaluation.vit_head import HeadSettings
from layout_visual_complexity.features.extraction import family_names, read_feature_table
from layout_visual_complexity.logs import logger
from layout_visual_complexity.ranking.dataset import read_dataset
from layout_visual_complexity.ranking.reliability import DEFAULT_GAP_EDGES, other_image, spearman
from layout_visual_complexity.runs import Run, sha256, shown_path

HITS, FOLD_SCORES = "hits.tsv", "fold_scores.tsv"
CURVE, CURVE_SCORES = "learning_curve.tsv", "learning_curve_fold_scores.tsv"


@dataclass(frozen=True)
class LearningCurveSettings:
    """Learning curves: the method trained on shares of each fold's training judgments, or of its training images
    (then on the judgments between the images kept), and tested on the fold's test judgments.

    Args:
        shares: The shares, below one; the whole is the evaluation itself. None, the default, draws no curves.
        by: What is shared out: `judgments`, `images`, or both.
        repeats: The curves use the folds of the first this many repeats; all by default.
    """

    shares: list[float] = field(default_factory=list)
    by: list[Literal["judgments", "images"]] = field(default_factory=lambda: ["judgments", "images"])
    repeats: int | None = None


@dataclass(frozen=True)
class RestatingSettings:
    """A check against a reference measure: a column whose Spearman correlation with the reference reaches
    `threshold` in absolute value, and which adds nothing to the reference (the pairwise model on both against the
    reference alone, on the same folds: neither the accuracy's nor SROCC's difference beyond chance above zero), only
    restates it and is left out of the model.

    Args:
        reference: The reference column (ink coverage, `Trivial__InkCoverage__share`); empty, the default, checks
            nothing.
        threshold: The least absolute Spearman correlation with the reference that calls for the paired comparison.
    """

    reference: str = ""
    threshold: float = 0.9


@dataclass(frozen=True)
class EvaluateSettings:
    """Evaluate a method by the protocol, cross-validated by manuscript, and train it on every judgment: the pairwise
    logistic model on a feature table, the local ViT head on a token set, or an ensemble of earlier evaluations.

    Args:
        dataset: The dataset's folder: its judgments, images and manuscripts.
        scores: The ranking's scores (`scores.tsv`), whose final Elo scores are the reference.
        output: The folder for the out-of-fold scores, the tables, each fold's hits and z-scores, the trained method
            (`model.json`, `head.npz` or `ensemble.json`) and the run's record.
        features: A feature table (`features.parquet` of `lvc features`), for the pairwise logistic model.
        families: Only these families' columns: comma-separated, or a list; all by default.
        columns: Only these columns, each by its name or its measure's prefix (`Family__Feature`): comma-separated, or
            a list; all the families' by default.
        restating: Columns that only restate a reference measure, left out of the model.
        tokens: A token set (the folder of `lvc embed`), for the local ViT head.
        head: How the local ViT head is trained.
        ensemble: Earlier evaluations (`lvc evaluate` folders, on the same folds) to combine, by the names their
            weights take.
        weights: The ensemble's weights, by member; equal when none are given.
        fitted_weights: Fit the ensemble's weights on each fold's training judgments (and, for the trained ensemble,
            on every judgment) instead.
        folds: The folds by manuscript.
        gap_edges: The edges of the bins of the final Elo gap.
        regularisation: The logistic regressions' C (the pairwise model's, the fitted weights'): smaller is stronger.
        bootstrap: Resamples of the manuscripts for the 95% intervals.
        seed: The seed of the bootstrap and of the learning curves' shares.
        reliability_bins: Bins of the predicted probability in the calibration table.
        ceiling: A split-half table (`split_half_ceiling.tsv` of `lvc judgments report`): the Elo scores' reliability
            (its corrected median) and the ceiling for SROCC, the reliability's square root.
        learning_curves: Which learning curves to draw, if any.
        jobs: Worker processes training the head's folds; 0 means all cores, as many as fit in memory. The results
            don't depend on it.
    """

    dataset: Path
    scores: Path
    output: Path
    features: Path | None = None
    families: list[str] | str = field(default_factory=list)
    columns: list[str] | str = field(default_factory=list)
    restating: RestatingSettings = field(default_factory=RestatingSettings)
    tokens: Path | None = None
    head: HeadSettings = field(default_factory=HeadSettings)
    ensemble: dict[str, Path] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    fitted_weights: bool = False
    folds: FoldSettings = field(default_factory=FoldSettings)
    gap_edges: list[float] = field(default_factory=lambda: list(DEFAULT_GAP_EDGES))
    regularisation: float = 1.0
    bootstrap: int = 1000
    seed: int = 1
    reliability_bins: int = 10
    ceiling: Path | None = None
    learning_curves: LearningCurveSettings = field(default_factory=LearningCurveSettings)
    jobs: int = 0


def read_elo(path: Path) -> pd.Series:
    """The final Elo scores (`elo_mean`) of a ranking's `scores.tsv`, by page."""
    table = pd.read_csv(path, sep="\t", dtype={"image": str})
    return table.set_index("image")["elo_mean"].astype(float)


def read_ceiling(path: Path) -> tuple[float, float]:
    """A split-half table's reliability (the corrected median) and the ceiling for SROCC, its square root."""
    from layout_visual_complexity.ranking.split_half import RELIABILITY, ceiling

    table = pd.read_csv(path, sep="\t").set_index("measure")
    reliability = float(table.at[RELIABILITY, "median"])
    return reliability, ceiling(reliability)


def train(features: pd.DataFrame, judgments: pd.DataFrame, regularisation: float) -> tuple:
    """The model on every judgment, its scores' mean and spread over the pages, and the scale of win probabilities."""
    model = fit_pairwise_model(features, judgments, regularisation)
    raw = model.scores(features)
    mean, sd = float(raw.mean()), float(raw.std(ddof=0))
    z = (raw - mean) / (sd if sd > 0 else 1.0)
    scale = metrics.fit_scale(z[judgments["chosen"]].to_numpy() - z[other_image(judgments)].to_numpy())
    return model, mean, sd, scale


# The methods.

# A learning curve's draw: what is shared out, and the share's number (from 1).
Curve = tuple[str, int]


class Method:
    """What the command evaluates: its raw scores on folds (the evaluation's own, or a learning curve's), and the
    method trained on every judgment, written into the output folder, with each page's z-score under it."""

    kind = ""

    def describe(self) -> str:
        raise NotImplementedError

    def fold_scores(self, folds: list[Fold], curve: Curve | None) -> FoldScores:
        raise NotImplementedError

    def train(self, output: Path) -> tuple[Path, pd.Series, dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class FeatureMethod(Method):
    """The pairwise logistic model on a feature table."""

    kind = "features"

    def __init__(self, settings: EvaluateSettings, run: Run, pages: list[str], judgments: pd.DataFrame) -> None:
        whole = read_feature_table(run.read(Path(settings.features)))  # type: ignore[arg-type]
        self.table = whole.select(family_names(settings.families)).select_columns(settings.columns)
        missing = [page for page in pages if page not in self.table.values.index]
        if len(missing) == len(pages):
            raise ValueError("the feature table has none of the dataset's pages")
        if missing:
            logger.warning("%d pages have no features; the model fills them with the median", len(missing))
        self.settings, self.judgments, self.pages = settings, judgments, pages
        self.features = self.table.values.reindex(pages)
        self.missing = len(missing)
        self.reference = None
        if settings.restating.reference:
            if settings.restating.reference not in whole.values.columns:
                raise ValueError(f"the feature table has no reference column {settings.restating.reference!r}")
            self.reference = whole.values[settings.restating.reference].reindex(pages)
        self.restating: pd.DataFrame | None = None

    def drop_restating(
        self, folds: list[Fold], elo: pd.Series, manuscripts: pd.Series, protocol: ProtocolSettings
    ) -> pd.DataFrame:
        """Leave out the columns that only restate the reference: each column's Spearman correlation with it and,
        where it reaches the threshold, the pairwise model on the reference and the column against the reference
        alone; the table of every column, and whether it was dropped."""
        check, reference = self.settings.restating, self.reference
        assert reference is not None
        base = None
        rows = []
        for column in self.features.columns:
            values = self.features[column]
            both = np.isfinite(values.to_numpy(dtype=float)) & np.isfinite(reference.to_numpy(dtype=float))
            rho = spearman(values[both].to_numpy(), reference[both].to_numpy()) if both.sum() >= 3 else math.nan
            row: dict[str, Any] = {"column": column, "rho": rho, "compared": False, "dropped": False}
            if column != check.reference and math.isfinite(rho) and abs(rho) >= check.threshold:
                if base is None:
                    base = self.evaluate(reference.to_frame("reference"), folds, elo, protocol)
                paired = pd.concat([reference.rename("reference"), values], axis=1)
                found = paired_difference(self.evaluate(paired, folds, elo, protocol), base, self.judgments,
                                          manuscripts, protocol, elo)  # fmt: skip
                row |= found | {"compared": True, "dropped": restates(found)}
            rows.append(row)
        table = pd.DataFrame(rows)
        dropped = table.loc[table["dropped"], "column"].tolist()
        if len(dropped) == self.features.shape[1]:
            raise ValueError("every column only restates the reference; nothing is left to evaluate")
        if dropped:
            logger.info("evaluate: left out, restating %s: %s", check.reference, ", ".join(dropped))
        self.features = self.features.drop(columns=dropped)
        self.table = self.table.only(list(self.features.columns))
        self.restating = table
        return table

    def evaluate(
        self, features: pd.DataFrame, folds: list[Fold], elo: pd.Series, protocol: ProtocolSettings
    ) -> Evaluation:
        fold_scores = pairwise_fold_scores(features, self.judgments, folds, self.settings.regularisation)
        return evaluate_fold_scores(fold_scores, self.judgments, elo, folds, protocol)

    def describe(self) -> str:
        families = ", ".join(f"{family['name']} (`{family['hash']}`)" for family in self.table.families)
        return f"Features: `{shown_path(self.settings.features)}`, {self.features.shape[1]} columns of {families}."  # type: ignore[arg-type]

    def fold_scores(self, folds: list[Fold], curve: Curve | None) -> FoldScores:
        return pairwise_fold_scores(self.features, self.judgments, folds, self.settings.regularisation)

    def train(self, output: Path) -> tuple[Path, pd.Series, dict[str, Any]]:
        model, mean, sd, scale = train(self.features, self.judgments, self.settings.regularisation)
        trained = TrainedModel(model, self.table.families, mean, sd, scale, self.settings.regularisation,
                               trained_on(self.settings, self.pages, self.judgments))  # fmt: skip
        z = (model.scores(self.features) - mean) / (sd if sd > 0 else 1.0)
        summary = {"columns": int(self.features.shape[1]), "families": [f["name"] for f in self.table.families]}
        summary["pages_without_features"] = self.missing
        summary["measures"] = measures(self.table)
        if self.settings.columns:
            summary["selected"] = list(self.features.columns)
        if self.restating is not None:
            summary["dropped"] = self.restating.loc[self.restating["dropped"], "column"].tolist()
            summary["restating"] = self.settings.restating.reference
        return trained.save(output / "model.json"), z, summary


class HeadMethod(Method):
    """The local ViT head on a token set, its folds trained in worker processes."""

    kind = "tokens"

    def __init__(self, settings: EvaluateSettings, run: Run, pages: list[str], judgments: pd.DataFrame) -> None:
        from layout_visual_complexity.evaluation.vit_head import HeadRanker
        from layout_visual_complexity.features.vit import TOKEN_PAGES, TOKEN_SET, read_token_set

        folder = Path(settings.tokens)  # type: ignore[arg-type]
        for name in (TOKEN_SET, TOKEN_PAGES):
            run.read(folder / name)
        self.token_set = read_token_set(folder)
        missing = [page for page in pages if page not in self.token_set.pages.index]
        if missing:
            raise ValueError(f"{len(missing)} of the dataset's pages have no tokens, {missing[0]} among them")
        self.settings, self.judgments, self.pages = settings, judgments, pages
        self.ranker = HeadRanker(self.token_set, pages, settings.head, settings.jobs)

    def describe(self) -> str:
        tokens = self.token_set.description
        return (
            f"Tokens: `{shown_path(self.settings.tokens)}` (`{tokens['backbone']}` at `{tokens['revision']}`, block "  # type: ignore[arg-type]
            f"{tokens['block']}, {tokens['long_edge']} px, the {tokens['crop']} crop); the head: "
            + ", ".join(f"{name} {value}" for name, value in asdict(self.settings.head).items())
            + f"; {self.ranker.workers} workers."
        )

    def fold_scores(self, folds: list[Fold], curve: Curve | None) -> FoldScores:
        if curve is None:
            return self.ranker.fold_scores(folds, self.judgments)
        kinds = list(self.settings.learning_curves.by)
        share = kinds.index(curve[0]) * 100 + curve[1]
        return self.ranker.fold_scores(folds, self.judgments, share=share)

    def train(self, output: Path) -> tuple[Path, pd.Series, dict[str, Any]]:
        trained = self.ranker.train(self.judgments, trained_on(self.settings, self.pages, self.judgments))
        z = trained.scores(self.token_set, self.pages)
        timings = self.ranker.timings
        summary = {
            "tokens": self.token_set.identity(),
            "head": asdict(self.settings.head),
            "workers": self.ranker.workers,
            "trainings": len(timings),
            "seconds_per_training": float(np.mean(timings)) if timings else None,
            "peak_memory_per_worker": max(self.ranker.peak_memory, default=None),
        }
        return trained.save(output / "head.npz"), z, summary

    def close(self) -> None:
        self.ranker.close()


class EnsembleMethod(Method):
    """An ensemble of earlier evaluations: their fold z-scores combined with given or fitted weights."""

    kind = "ensemble"

    def __init__(self, settings: EvaluateSettings, run: Run, pages: list[str], judgments: pd.DataFrame) -> None:
        if len(settings.ensemble) < 2:
            raise ValueError("an ensemble needs at least two evaluations")
        unknown = sorted(set(settings.weights) - set(settings.ensemble))
        if unknown:
            raise ValueError(f"weights given for evaluations not in the ensemble: {', '.join(unknown)}")
        if settings.weights and set(settings.weights) != set(settings.ensemble):
            raise ValueError("give every member's weight, or none for equal weights")
        self.settings, self.judgments, self.pages, self.run = settings, judgments, pages, run
        self.members = {name: Path(folder) for name, folder in settings.ensemble.items()}
        self.fold_z = {name: read_fold_scores(run.read(folder / FOLD_SCORES)) for name, folder in self.members.items()}
        self.fold_weights: list[pd.DataFrame] = []
        self.curve_tables: dict[str, pd.DataFrame] = {}

    def weights(self) -> dict[str, float] | None:
        if self.settings.fitted_weights:
            return None
        if self.settings.weights:
            return {name: float(value) for name, value in self.settings.weights.items()}
        return dict.fromkeys(self.members, 1 / len(self.members))

    def describe(self) -> str:
        members = ", ".join(f"{name} (`{shown_path(folder)}`)" for name, folder in self.members.items())
        weights = self.weights()
        how = (
            "fitted on each fold's training judgments"
            if weights is None
            else ", ".join(f"{name} {value:g}" for name, value in weights.items())
        )
        return f"Ensemble of {members}; weights: {how}."

    def member_curves(self, curve: Curve) -> dict[str, pd.DataFrame]:
        found = {}
        for name, folder in self.members.items():
            path = folder / CURVE_SCORES
            if not path.is_file():
                raise ValueError(f"the member {name} has no learning curves; evaluate it with the same ones")
            if name not in self.curve_tables:
                self.curve_tables[name] = read_fold_scores(self.run.read(path))
            table = self.curve_tables[name]
            chosen = table[(table["by"] == curve[0]) & (table["share_number"] == curve[1])]
            if chosen.empty:
                raise ValueError(f"the member {name} has no learning curve by {curve[0]}, share {curve[1]}")
            found[name] = chosen
        return found

    def fold_scores(self, folds: list[Fold], curve: Curve | None) -> FoldScores:
        members = self.fold_z if curve is None else self.member_curves(curve)
        found, weights = ensemble_fold_scores(members, folds, self.judgments, self.weights(),
                                              self.settings.regularisation)  # fmt: skip
        if curve is None:
            self.fold_weights.append(weights)
        return found

    def train(self, output: Path) -> tuple[Path, pd.Series, dict[str, Any]]:
        trained = {}
        for name, folder in self.members.items():
            table = pd.read_csv(self.run.read(folder / "scores.tsv"), sep="\t", dtype={"page": str})
            trained[name] = table.set_index("page")["trained"].reindex(self.pages)
        z = pd.DataFrame(trained)
        weights = self.weights()
        values = (
            fitted_weights(z, self.judgments, self.settings.regularisation)
            if weights is None
            else np.array([weights[name] for name in self.members])
        )
        combined = pd.Series(z.to_numpy(dtype=float) @ values, index=self.pages)
        combined = (combined - combined.mean()) / (combined.std(ddof=0) or 1.0)
        chosen = combined[self.judgments["chosen"]].to_numpy() - combined[other_image(self.judgments)].to_numpy()
        members = {}
        for name, folder in self.members.items():
            models = [folder / file for file in ("model.json", "head.npz", "ensemble.json")]
            model = next(path for path in models if path.is_file())
            members[name] = {"evaluation": shown_path(folder), "model": model.name, "sha256": sha256(model)}
        description = {
            "model": "ensemble",
            "members": members,
            "weights": dict(zip(self.members, values.tolist(), strict=True)),
            "fitted": weights is None,
            "scale": metrics.fit_scale(chosen),
            "trained_on": trained_on(self.settings, self.pages, self.judgments),
        }
        path = output / "ensemble.json"
        path.write_text(json.dumps(description, indent=2) + "\n", encoding="utf-8")
        return path, combined, {"members": list(self.members), "weights": description["weights"]}


def restates(difference: dict[str, float]) -> bool:
    """Whether a measure adds nothing to the reference it is set against: neither its accuracy's nor its SROCC's
    difference lies beyond chance above zero."""
    return not (difference["accuracy_low"] > 0 or difference["srocc_low"] > 0)


def measures(table: Any) -> list[dict[str, Any]]:
    """The measures a feature table's columns belong to, in their order: each one's prefix, family and feature in
    words, and its number of columns; by prefix alone where its family can't be loaded."""
    from layout_visual_complexity.features.contract import FeatureError, load_family

    groups: dict[str, dict[str, str]] = {}
    for described in table.families:
        try:
            family = load_family(described["class"], described["settings"])
        except (FeatureError, TypeError):
            continue
        groups |= {group.prefix: {"family": group.family, "feature": group.feature} for group in family.groups()}
    found: dict[str, dict[str, Any]] = {}
    for column in table.values.columns:
        prefix = column.rsplit("__", 1)[0]
        entry = found.setdefault(prefix, {"prefix": prefix, **groups.get(prefix, {}), "columns": 0})
        entry["columns"] += 1
    return list(found.values())


def read_fold_scores(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", dtype={"page": str})
    table["test"] = table["test"].astype(bool)
    return table


def trained_on(settings: EvaluateSettings, pages: list[str], judgments: pd.DataFrame) -> dict[str, Any]:
    return {"dataset": shown_path(settings.dataset), "pages": len(pages), "judgments": len(judgments)}


def chosen_method(settings: EvaluateSettings, run: Run, pages: list[str], judgments: pd.DataFrame) -> Method:
    given = [name for name in ("features", "tokens", "ensemble") if getattr(settings, name)]
    if len(given) != 1:
        raise ValueError("give one of --features, --tokens or --ensemble")
    kind = {"features": FeatureMethod, "tokens": HeadMethod, "ensemble": EnsembleMethod}[given[0]]
    return kind(settings, run, pages, judgments)


# The run.


def run_evaluate(settings: EvaluateSettings) -> dict[str, object]:
    """Write the evaluation's tables, the out-of-fold scores, each fold's hits and z-scores, the learning curves asked
    for and the trained method into `output`, with the run's configuration and manifest. Returns the run's summary."""
    began = time.perf_counter()
    run = Run("evaluate", settings)
    for path in [Path(settings.scores)] + [
        Path(settings.dataset) / name for name in ("judgments.tsv", "images.tsv", "manuscripts.tsv")
    ]:
        run.read(path)
    dataset = read_dataset(settings.dataset)
    elo = read_elo(settings.scores)
    pages = dataset.images["image"].tolist()
    judgments = dataset.judgments.reset_index(drop=True)
    manuscripts = dataset.images.set_index("image")["manuscript"]
    edges = tuple(float(edge) for edge in settings.gap_edges)
    protocol = ProtocolSettings(
        edges, settings.regularisation, settings.bootstrap, settings.reliability_bins, settings.seed
    )
    fold = settings.folds
    folds = manuscript_folds(pages, manuscripts, judgments, fold.k, fold.repeats, fold.seed)
    method = chosen_method(settings, run, pages, judgments)
    output = Path(settings.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("evaluate: %s, %d pages, %d folds", method.kind, len(pages), len(folds))
        restating = None
        if isinstance(method, FeatureMethod) and method.reference is not None:
            restating = method.drop_restating(folds, elo, manuscripts, protocol)
        evaluated = time.perf_counter()
        evaluation = evaluate_fold_scores(method.fold_scores(folds, None), judgments, elo, folds, protocol)
        evaluated = time.perf_counter() - evaluated
        tested = evaluation.tested
        agreement = metrics.hits(judgments, elo)
        method_values = tested.assign(value=tested["hit"])
        reference_values = tested.assign(value=agreement[tested["row"].to_numpy()])
        method_intervals = fold_mean_intervals(method_values, judgments, manuscripts, edges, protocol)
        reference_intervals = fold_mean_intervals(reference_values, judgments, manuscripts, edges, protocol)
        intervals = bootstrap(evaluation, judgments, elo, manuscripts, protocol)
        by_gap = by_elo_gap(evaluation, reference_values, method_intervals, reference_intervals, edges, len(folds))
        reliability, ceiling = read_ceiling(settings.ceiling) if settings.ceiling else (math.nan, math.nan)
        summary_metrics = pd.DataFrame(
            [
                ("accuracy", evaluation.summary["accuracy"], *method_intervals.iloc[-1][["low", "high"]]),
                ("accuracy_pooled", evaluation.summary["accuracy_pooled"], intervals["accuracy_low"],
                 intervals["accuracy_high"]),
                ("consistency", by_gap.iloc[-1]["consistency"], *reference_intervals.iloc[-1][["low", "high"]]),
                ("srocc", evaluation.summary["srocc"], intervals["srocc_low"], intervals["srocc_high"]),
                ("srocc_ceiling", ceiling, math.nan, math.nan),
                ("elo_reliability", reliability, math.nan, math.nan),
                ("log_loss", evaluation.summary["log_loss"], math.nan, math.nan),
            ],
            columns=["metric", "value", "low", "high"],
        )  # fmt: skip
        curve, curve_scores = learning_curves(method, settings, judgments, elo, manuscripts, folds, protocol)
        if curve is not None:
            whole = whole_share(evaluation, folds, settings, judgments, elo, manuscripts, protocol)
            curve = pd.concat([curve, pd.DataFrame([whole | {"by": by} for by in settings.learning_curves.by])])
            curve = curve.sort_values(["by", "share"], kind="stable").reset_index(drop=True)
        trained_path, trained_z, method_summary = method.train(output)
    finally:
        method.close()

    scores = pd.DataFrame(
        {
            "page": pages,
            "manuscript": manuscripts.reindex(pages).to_numpy(),
            "score": evaluation.scores.reindex(pages).to_numpy(),
            "elo": elo.reindex(pages).to_numpy(),
            "trained": trained_z.reindex(pages).to_numpy(),
        }
    )
    folds_table = evaluation.folds
    if isinstance(method, EnsembleMethod) and method.fold_weights:
        weights = method.fold_weights[0].set_index("fold_number").add_prefix("weight_")
        folds_table = folds_table.join(weights.reset_index(drop=True))
    hits = tested.assign(judgment=judgments["judgment"].iloc[tested["row"].to_numpy()].to_numpy())
    hits = hits[["fold_number", "judgment", "gap", "hit", "chosen_probability", "left_probability", "left_chosen"]]
    written = []
    tables = [
        ("scores", scores),
        ("by_elo_gap", by_gap),
        ("metrics", summary_metrics),
        ("folds", folds_table),
        ("calibration", evaluation.reliability),
        ("hits", hits),
        ("fold_scores", evaluation.fold_scores),
    ]
    if curve is not None:
        tables += [("learning_curve", curve), ("learning_curve_fold_scores", curve_scores)]
    if restating is not None:
        tables.append(("restating", restating))
    for name, frame in tables:
        path = output / f"{name}.tsv"
        frame.to_csv(path, sep="\t", index=False, lineterminator="\n", float_format="%.6f")
        written.append(path)
    written.append(trained_path)
    report = output / "report.md"
    report.write_text(report_text(settings, method.describe(), by_gap, summary_metrics, curve, restating), "utf-8")
    written.append(report)
    run.wrote(written)
    seconds = time.perf_counter() - began
    run.summary = {
        "method": method.kind,
        **method_summary,
        "pages": len(pages),
        "judgments": len(judgments),
        "folds": len(folds),
        "tested_judgments": len(tested),
        "metrics": summary_metrics.to_dict("records"),
        "by_elo_gap": by_gap.to_dict("records"),
        "learning_curve": curve.to_dict("records") if curve is not None else None,
        "seconds_folds": round(evaluated, 2),
        "seconds": round(seconds, 2),
    }
    run.finish(output)
    accuracy, srocc = evaluation.summary["accuracy"], evaluation.summary["srocc"]
    print(f"accuracy {accuracy:.3f}, SROCC {srocc:.3f} over {len(folds)} folds -> {output}")
    return run.summary


def learning_curves(
    method: Method,
    settings: EvaluateSettings,
    judgments: pd.DataFrame,
    elo: pd.Series,
    manuscripts: pd.Series,
    folds: list[Fold],
    protocol: ProtocolSettings,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """The method on each learning curve's shares: per share, accuracy (the mean over folds, its SD and interval)
    and SROCC; and each fold's z-scores, for ensembles."""
    shares = [float(share) for share in settings.learning_curves.shares]
    if not shares:
        return None, None
    if any(not 0 < share < 1 for share in shares):
        raise ValueError("learning curves' shares lie between 0 and 1; the whole is the evaluation itself")
    repeats = settings.learning_curves.repeats
    folds = [fold for fold in folds if repeats is None or fold.repeat < repeats]
    rows, scores = [], []
    for by in settings.learning_curves.by:
        for number, share in enumerate(shares, start=1):
            logger.info("evaluate: learning curve by %s, share %g", by, share)
            reduced = reduced_folds(folds, judgments, share, by, protocol.seed, number)
            result = evaluate_fold_scores(method.fold_scores(reduced, (by, number)), judgments, elo, reduced, protocol)
            values = result.tested.assign(value=result.tested["hit"])
            overall = fold_mean_intervals(values, judgments, manuscripts, protocol.gap_edges, protocol).iloc[-1]
            rows.append(
                {
                    "by": by,
                    "share": share,
                    "share_number": number,
                    "train_per_fold": float(np.mean([len(item.train) for item in reduced])),
                    "train_pages_per_fold": float(np.mean([len(item.train_pages) for item in reduced])),
                    "accuracy": result.summary["accuracy"],
                    "accuracy_sd": result.summary["accuracy_sd"],
                    "low": overall["low"],
                    "high": overall["high"],
                    "srocc": result.summary["srocc"],
                }
            )
            scores.append(result.fold_scores.assign(by=by, share_number=number))
    return pd.DataFrame(rows), pd.concat(scores, ignore_index=True)


def whole_share(
    evaluation: Evaluation,
    folds: list[Fold],
    settings: EvaluateSettings,
    judgments: pd.DataFrame,
    elo: pd.Series,
    manuscripts: pd.Series,
    protocol: ProtocolSettings,
) -> dict[str, Any]:
    """The learning curves' whole share: the evaluation itself, on the folds the curves use."""
    repeats = settings.learning_curves.repeats
    kept = [number for number, fold in enumerate(folds) if repeats is None or fold.repeat < repeats]
    if len(kept) == len(folds):
        result, chosen = evaluation, folds
    else:
        chosen = [folds[number] for number in kept]
        fold_scores = evaluation.fold_scores[evaluation.fold_scores["fold_number"].isin(kept)]
        raw = [fold_scores[fold_scores["fold_number"] == number].set_index("page")["z"] for number in kept]
        result = evaluate_fold_scores(raw, judgments, elo, chosen, protocol)  # type: ignore[arg-type]
    values = result.tested.assign(value=result.tested["hit"])
    overall = fold_mean_intervals(values, judgments, manuscripts, protocol.gap_edges, protocol).iloc[-1]
    return {
        "share": 1.0,
        "share_number": 0,
        "train_per_fold": float(np.mean([len(fold.train) for fold in chosen])),
        "train_pages_per_fold": float(np.mean([len(fold.train_pages) for fold in chosen])),
        "accuracy": result.summary["accuracy"],
        "accuracy_sd": result.summary["accuracy_sd"],
        "low": overall["low"],
        "high": overall["high"],
        "srocc": result.summary["srocc"],
    }


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
    settings: EvaluateSettings,
    described: str,
    by_gap: pd.DataFrame,
    summary: pd.DataFrame,
    curve: pd.DataFrame | None = None,
    restating: pd.DataFrame | None = None,
) -> str:
    """The evaluation in Markdown."""

    def interval(low: float, high: float) -> str:
        return "" if not (math.isfinite(low) and math.isfinite(high)) else f" [{low:.3f}, {high:.3f}]"

    values = summary.set_index("metric")
    lines = [
        "# Evaluation",
        "",
        f"- {described}",
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
    if curve is not None:
        lines += [
            "",
            "| Learning curve | Share | Training judgments per fold | Accuracy | 95% interval | SROCC |",
            "|---|---:|---:|---:|---|---:|",
        ]
        for row in curve.itertuples(index=False):
            lines.append(
                f"| {row.by} | {row.share:g} | {row.train_per_fold:.0f} | {row.accuracy:.3f} |"
                f"{interval(row.low, row.high)} | {row.srocc:.3f} |"
            )
    if restating is not None:
        lines += [
            "",
            f"Columns set against `{settings.restating.reference}` (left out when |ρ| ≥ "
            f"{settings.restating.threshold:g} and neither difference is beyond chance above zero):",
            "",
            "| Column | ρ | Accuracy difference | 95% interval | SROCC difference | 95% interval | Left out |",
            "|---|---:|---:|---|---:|---|---|",
        ]
        for row in restating.itertuples(index=False):
            values = row._asdict()
            if values["compared"]:
                accuracy = interval(values["accuracy_low"], values["accuracy_high"])
                srocc = interval(values["srocc_low"], values["srocc_high"])
                compared = (
                    f" {values['accuracy_difference']:+.3f} |{accuracy} | {values['srocc_difference']:+.3f} |{srocc} |"
                )
            else:
                compared = " | | | |"
            lines.append(f"| `{row.column}` | {row.rho:+.3f} |{compared} {'yes' if row.dropped else 'no'} |")
    return "\n".join(lines) + "\n"


def read_metrics(folder: Path) -> dict[str, dict[str, float]]:
    """An evaluation's metrics, by name: value, low, high."""
    table = pd.read_csv(Path(folder) / "metrics.tsv", sep="\t").set_index("metric")
    return {metric: row.to_dict() for metric, row in table.iterrows()}


def read_evaluation(folder: Path, judgments: pd.DataFrame) -> Evaluation:
    """An evaluation as `lvc evaluate` wrote it into `folder`: per fold, by gap, each fold's hits (their judgments
    placed by their ids in `judgments`), the out-of-fold scores, the calibration and the fold z-scores; enough to
    compare it with another on the same resamples (`paired_comparison`)."""
    folder = Path(folder)
    if not (folder / HITS).is_file():
        raise ValueError(f"{folder} holds no per-fold hits; evaluate it again")
    tested = pd.read_csv(folder / HITS, sep="\t", dtype={"judgment": str})
    rows = pd.Series(np.arange(len(judgments)), index=judgments["judgment"].astype(str))
    missing = ~tested["judgment"].isin(rows.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} of the evaluation's judgments aren't in the dataset")
    tested["row"] = rows.reindex(tested["judgment"]).to_numpy()
    scores = pd.read_csv(folder / "scores.tsv", sep="\t", dtype={"page": str}).set_index("page")["score"]
    found = {name: pd.read_csv(folder / f"{name}.tsv", sep="\t") for name in ("folds", "by_elo_gap", "calibration")}
    fold_scores = read_fold_scores(folder / FOLD_SCORES) if (folder / FOLD_SCORES).is_file() else pd.DataFrame()
    summary = {name: row["value"] for name, row in read_metrics(folder).items()}
    return Evaluation(found["folds"], found["by_elo_gap"], tested, scores, found["calibration"], summary, fold_scores)
