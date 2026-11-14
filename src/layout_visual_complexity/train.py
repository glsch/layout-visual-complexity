"""The `lvc train` command: a ranker trained on every judgment of a dataset, without the evaluation, for scoring other
pages: the pairwise logistic model on a feature table (`model.json`), the local ViT head on a token set (`head.npz`),
or the ensemble of rankers trained before, which is what `lvc score` and `lvc compare` use.

Each ranker keeps what its inputs must be, and the scale that turns a difference of its z-scored page scores into the
log-odds that the first page is chosen; each writes its training pages' z-scores (`scores.tsv`), from which an
ensemble is built. Manuscripts can be left out: their pages, and every judgment that touches them.

An ensemble's folder holds its members' files beside `ensemble.json`: the members' weights, the mean and spread of
their weighted sum over the training pages, the scales of win probabilities (fitted on evaluations' out-of-fold scores
when they are given, so that the probabilities hold on new pages), and the Elo scale of its scores (the regression of
the Elo scores on the ensemble evaluation's out-of-fold scores).
"""

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from layout_visual_complexity.evaluate import FeatureMethod, HeadMethod, Method
from layout_visual_complexity.evaluation import calibration, metrics
from layout_visual_complexity.evaluation.vit_head import HeadSettings
from layout_visual_complexity.logs import logger
from layout_visual_complexity.ranking.dataset import Dataset, read_dataset
from layout_visual_complexity.ranking.reliability import other_image
from layout_visual_complexity.runs import Run, sha256, shown_path

ENSEMBLE = "ensemble.json"
SCORES = "scores.tsv"
CALIBRATION = "calibration.tsv"
MODEL_FILES = ("model.json", "head.npz", ENSEMBLE)


@dataclass(frozen=True)
class TrainSettings:
    """Train a ranker on every judgment: the pairwise logistic model on a feature table, the local ViT head on a token
    set, or an ensemble of rankers trained before.

    Args:
        dataset: The dataset's folder: its judgments and images.
        output: The folder for the trained ranker (`model.json`, `head.npz`, or `ensemble.json` with its members'
            files), its training pages' z-scores (`scores.tsv`) and the run's record.
        features: A feature table (`features.parquet` of `lvc features`), for the pairwise logistic model.
        families: Only these families' columns: comma-separated, or a list; all by default.
        tokens: A token set (the folder of `lvc embed`), for the local ViT head.
        head: How the local ViT head is trained.
        regularisation: The pairwise logistic model's C: smaller is stronger.
        leave_out: Manuscripts left out, by id: their pages, and every judgment that touches them.
        ensemble: Rankers trained before (`lvc train` folders), by the names the ensemble gives them; their training
            pages must be the same.
        files: Each member's file name in the ensemble's folder; by default its name with its file's suffix.
        weights: The ensemble's weights, by member; equal when none are given.
        calibration: Evaluations (`lvc evaluate` folders) whose out-of-fold scores fit the scales of win
            probabilities, by member name, and `ensemble` for the ensemble's own, which also gives the Elo scale.
            Without them, the scales are fitted on the training judgments and there is no Elo scale.
        bins: Bins of the predicted probability in the calibration table.
        fit_bound: Pages whose out-of-fold z-score lies beyond this, in absolute value, in any of the calibration's
            evaluations, are left out of the fits of the scales and the Elo scale, with every judgment touching
            them: a linear model extrapolates on a page whose features lie far outside those it was trained on (a
            diagram whose lines run every way), and one such page would set a scale alone.
    """

    dataset: Path
    output: Path
    features: Path | None = None
    families: list[str] | str = field(default_factory=list)
    tokens: Path | None = None
    head: HeadSettings = field(default_factory=HeadSettings)
    regularisation: float = 1.0
    leave_out: list[str] = field(default_factory=list)
    ensemble: dict[str, Path] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    calibration: dict[str, Path] = field(default_factory=dict)
    bins: int = 10
    fit_bound: float = 5.0


def without_manuscripts(dataset: Dataset, manuscripts: list[str]) -> Dataset:
    """The dataset without these manuscripts' pages and every judgment that touches them."""
    if not manuscripts:
        return dataset
    unknown = sorted(set(manuscripts) - set(dataset.images["manuscript"]))
    if unknown:
        raise ValueError(f"no pages of {', '.join(unknown)} in the dataset")
    images = dataset.images[~dataset.images["manuscript"].isin(manuscripts)].reset_index(drop=True)
    kept = set(images["image"])
    touching = dataset.judgments["left"].isin(kept) & dataset.judgments["right"].isin(kept)
    judgments = dataset.judgments[touching].reset_index(drop=True)
    others = dataset.manuscripts[~dataset.manuscripts["manuscript"].isin(manuscripts)].reset_index(drop=True)
    return Dataset(judgments, images, others)


def write_scores(path: Path, z: pd.Series, manuscripts: pd.Series) -> Path:
    """The training pages' z-scores under the trained ranker."""
    table = pd.DataFrame({"page": z.index, "manuscript": manuscripts.reindex(z.index).to_numpy(), "trained": z})
    table.to_csv(path, sep="\t", index=False, lineterminator="\n", float_format="%.6f")
    return path


def run_train(settings: TrainSettings) -> dict[str, Any]:
    """Train the ranker and write it into `output`, with the run's record. Returns the run's summary."""
    began = time.perf_counter()
    run = Run("train", settings)
    for name in ("judgments.tsv", "images.tsv"):
        run.read(Path(settings.dataset) / name)
    given = [name for name in ("features", "tokens", "ensemble") if getattr(settings, name)]
    if len(given) != 1:
        raise ValueError("give one of --features, --tokens or --ensemble")
    dataset = without_manuscripts(read_dataset(settings.dataset), list(settings.leave_out))
    pages = dataset.images["image"].tolist()
    judgments = dataset.judgments.reset_index(drop=True)
    manuscripts = dataset.images.set_index("image")["manuscript"]
    output = Path(settings.output)
    output.mkdir(parents=True, exist_ok=True)
    if settings.ensemble:
        path, z, summary = train_ensemble(settings, run, pages, judgments, output)
    else:
        path, z, summary = train_member(settings, run, pages, judgments, output)
    written = [path, write_scores(output / SCORES, z, manuscripts)]
    if settings.ensemble:
        written += [output / name for name in summary.pop("files")]
    run.wrote(written)
    run.summary = {**summary, "pages": len(pages), "judgments": len(judgments)}
    if settings.leave_out:
        run.summary["left_out"] = list(settings.leave_out)
    run.summary["seconds"] = round(time.perf_counter() - began, 2)
    run.finish(output)
    print(f"trained on {len(judgments):,} judgments of {len(pages):,} pages -> {path}")
    return run.summary


def train_member(
    settings: TrainSettings, run: Run, pages: list[str], judgments: pd.DataFrame, output: Path
) -> tuple[Path, pd.Series, dict[str, Any]]:
    """The pairwise logistic model or the local ViT head, on every judgment between the pages kept."""
    from layout_visual_complexity.evaluate import EvaluateSettings

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
    try:
        path, z, summary = method.train(output)
    finally:
        method.close()
    return path, z, {"method": method.kind, **summary}


def member_model(folder: Path) -> Path:
    """The trained ranker in a `lvc train` (or `lvc evaluate`) folder."""
    found = [Path(folder) / name for name in MODEL_FILES if (Path(folder) / name).is_file()]
    if not found:
        raise ValueError(f"{shown_path(folder)} holds no trained ranker ({', '.join(MODEL_FILES)})")
    return found[0]


def member_scale(model: Path) -> float:
    """The scale a trained ranker was given on its training judgments."""
    if model.suffix == ".npz":
        from layout_visual_complexity.evaluation.vit_head import TrainedHead

        return TrainedHead.load(model).scale
    return float(json.loads(model.read_text(encoding="utf-8"))["scale"])


def train_ensemble(
    settings: TrainSettings, run: Run, pages: list[str], judgments: pd.DataFrame, output: Path
) -> tuple[Path, pd.Series, dict[str, Any]]:
    """The ensemble of rankers trained before: their training pages' z-scores combined with the weights, its spread
    over those pages, the scales of win probabilities and the Elo scale."""
    names = list(settings.ensemble)
    if len(names) < 2:
        raise ValueError("an ensemble needs at least two rankers")
    if settings.weights and set(settings.weights) != set(names):
        raise ValueError("give every member's weight, or none for equal weights")
    unknown = sorted(set(settings.calibration) - {*names, "ensemble"})
    if unknown:
        raise ValueError(f"calibration given for evaluations not in the ensemble: {', '.join(unknown)}")
    weights = np.array([float(settings.weights[name]) if settings.weights else 1 / len(names) for name in names])
    z = {}
    for name, folder in settings.ensemble.items():
        table = pd.read_csv(run.read(Path(folder) / SCORES), sep="\t", dtype={"page": str}).set_index("page")
        if set(table.index) != set(pages):
            raise ValueError(f"the member {name} was trained on other pages than the dataset's (left out: "
                             f"{', '.join(settings.leave_out) or 'none'})")  # fmt: skip
        z[name] = table["trained"].reindex(pages)
    frame = pd.DataFrame(z)
    combined = pd.Series(frame.to_numpy(dtype=float) @ weights, index=pages)
    mean, sd = float(combined.mean()), float(combined.std(ddof=0))
    ensemble_z = (combined - mean) / (sd if sd > 0 else 1.0)
    difference = ensemble_z[judgments["chosen"]].to_numpy() - ensemble_z[other_image(judgments)].to_numpy()
    training_scale = metrics.fit_scale(difference)

    members, files = {}, []
    for (name, folder), weight in zip(settings.ensemble.items(), weights, strict=True):
        model = member_model(Path(folder))
        file = settings.files.get(name, f"{name}{model.suffix}")
        target = output / file
        if model.resolve() != target.resolve():
            shutil.copyfile(run.read(model), target)
        files.append(file)
        members[name] = {"file": file, "sha256": sha256(target), "weight": float(weight),
                         "scale_training": member_scale(model)}  # fmt: skip

    given_weights = dict(zip(names, weights.tolist(), strict=True))
    summary: dict[str, Any] = {"method": "ensemble", "members": names, "weights": given_weights}
    description: dict[str, Any] = {
        "model": "ensemble",
        "members": members,
        "combined": {"mean": mean, "sd": sd},
        "scale_training": training_scale,
        "scale": training_scale,
        "scales_from": "the training judgments",
        "trained_on": {
            "dataset": shown_path(settings.dataset),
            "pages": len(pages),
            "judgments": len(judgments),
            "left_out": list(settings.leave_out),
        },
    }
    if settings.calibration:
        missing = sorted({*names, "ensemble"} - set(settings.calibration))
        if missing:
            raise ValueError(f"calibration needs an evaluation for every member and the ensemble: {', '.join(missing)}")
        full = read_dataset(settings.dataset).judgments.reset_index(drop=True)
        evaluations = {}
        for name, folder in settings.calibration.items():
            run.read(Path(folder) / SCORES)
            evaluations[name] = calibration.read_out_of_fold_scores(Path(folder))
        beyond = sorted({page for table in evaluations.values()
                         for page in table.index[table["score"].abs() > settings.fit_bound]})  # fmt: skip
        if beyond:
            logger.warning("train: left out of the fits, out-of-fold z-scores beyond %g: %s", settings.fit_bound,
                           ", ".join(beyond))  # fmt: skip
        kept = full[~(full["left"].isin(beyond) | full["right"].isin(beyond))].reset_index(drop=True)
        rows, curves = [], []
        for name, evaluation in evaluations.items():
            folder = Path(settings.calibration[name])
            scores = evaluation["score"].drop(index=beyond, errors="ignore")
            scale = calibration.out_of_fold_scale(scores, kept)
            entry = members.get(name, description)
            entry["scale"] = scale
            tested = calibration.test_differences(folder, full)
            tested = tested[~(tested["chosen"].isin(beyond) | tested["other"].isin(beyond))]
            held = calibration.held_out_scale_probabilities(scores, kept, evaluation["manuscript"])
            training = members[name]["scale_training"] if name in members else training_scale
            given = {
                "each fold's training fit": tested["training_probability"].to_numpy(),
                "trained ranker's scale": metrics.sigmoid(training * tested["difference"].to_numpy()),
                "out-of-fold scale": metrics.sigmoid(scale * tested["difference"].to_numpy()),
                "out-of-fold scale, by manuscript held out": held[np.isfinite(held)],
            }
            for how, probabilities in given.items():
                error = calibration.calibration_error(probabilities, settings.bins)
                rows.append({"method": name, "probabilities": how,
                             "scale": {"trained ranker's scale": training, "out-of-fold scale": scale}.get(how),
                             "judgments": len(probabilities), "log_loss": metrics.log_loss(probabilities),
                             "calibration_error": error})  # fmt: skip
                curve = calibration.calibration_curve(probabilities, settings.bins)
                curves.append(curve.assign(method=name, probabilities=how))
            if name == "ensemble":
                elo = evaluation["elo"].drop(index=beyond, errors="ignore")
                description["elo"] = calibration.fit_elo_scale(scores, elo).to_dict()
        description["left_out_of_fits"] = {"bound": settings.fit_bound, "pages": beyond,
                                           "judgments": len(full) - len(kept)}  # fmt: skip
        description["scales_from"] = "the evaluations' out-of-fold scores"
        description["calibration"] = {
            name: {"evaluation": shown_path(folder), "scores_sha256": sha256(Path(folder) / SCORES)}
            for name, folder in settings.calibration.items()
        }
        table = pd.DataFrame(rows)
        curve_table = pd.concat(curves, ignore_index=True)
        table.to_csv(output / CALIBRATION, sep="\t", index=False, lineterminator="\n", float_format="%.6f")
        curve_table.to_csv(output / "calibration_curves.tsv", sep="\t", index=False, lineterminator="\n",
                           float_format="%.6f")  # fmt: skip
        files += [CALIBRATION, "calibration_curves.tsv"]
        summary["calibration"] = table.to_dict("records")
        summary["elo"] = description.get("elo")
    path = output / ENSEMBLE
    path.write_text(json.dumps(description, indent=2) + "\n", encoding="utf-8")
    summary["scales"] = {name: members[name].get("scale", members[name]["scale_training"]) for name in names}
    summary["scales"]["ensemble"] = description["scale"]
    summary["files"] = files
    return path, ensemble_z, summary
