"""The tables and figures of the methods, for `lvc report`: any number of evaluations (`lvc evaluate` folders) set side
by side, each defined in the captions from what its run recorded.

- `methods_by_elo_gap` (and `ensembles_by_elo_gap`, the same for the appendix's ensembles): accuracy by Elo gap and
  SROCC beside the Reference (for accuracy, the consistency of the same test judgments with the final Elo scores; for
  ρ, its ceiling); intervals in the Overall and ρ rows (every row's in the TSV).
- `accuracy_by_elo_gap`: the same as a figure, and as a table with every row's interval.
- `paired_differences`: the differences between pairs of methods, in accuracy and SROCC, overall and within
  manuscripts, on the same resamples.
- `within_and_across_manuscripts`: every method's accuracy on the test judgments within one manuscript and across
  manuscripts, and its ρ overall and within manuscripts.
- `learning_curves_and_calibration`: accuracy as the training judgments and images grow, and calibration.
- `local_complexity_maps`: the local ViT head's patch scores over pages.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from layout_visual_complexity.evaluate import read_elo, read_evaluation
from layout_visual_complexity.evaluation.protocol import ProtocolSettings, paired_difference, within_and_across
from layout_visual_complexity.judgment_reports import Written, decimal
from layout_visual_complexity.outputs import LatexTable, escape, gap_latex, save_figure, write_table
from layout_visual_complexity.ranking.dataset import read_dataset
from layout_visual_complexity.runs import Run

BACKBONES = {
    "facebook/dinov2-small": "DINOv2 ViT-S/14",
    "facebook/dinov3-vits16-pretrain-lvd1689m": "DINOv3 ViT-S/16",
}
ADJECTIVES = {
    "geometry": "geometric",
    "layout": "Layout",
    "trivial": "trivial",
    "generic": "generic image",
    "kyle_davidson": "Kyle--Davidson",
    "ink": "ink distribution",
    "line_texture": "line-texture",
    "colour": "real colour",
}
NUMBERS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
# Families described by their measures rather than by their number of features.
BY_MEASURE = {"trivial", "generic", "kyle_davidson", "ink", "mean_token", "colour"}
KIND_COLOURS = {"features": "geometry", "tokens": "vit", "ensemble": "ensemble"}


def needed(settings: Any, name: str, output: str) -> Any:
    value = getattr(settings, name)
    if value is None or value == {} or value == []:
        raise ValueError(f"{output} needs --{name}")
    return value


class Evaluations:
    """The evaluations a table or figure sets side by side: each one's tables, metrics and run record, its label and
    its colour; those the settings name as `shown` are its columns (all by default), the others serve comparisons."""

    def __init__(self, settings: Any, run: Run, output: str) -> None:
        self.settings, self.run, self.output = settings, run, output
        self.members: dict[str, Path] = {}
        self.folders = {key: Path(path) for key, path in needed(settings, "evaluations", output).items()}
        unknown = [key for key in getattr(settings, "shown", None) or [] if key not in self.folders]
        if unknown:
            raise ValueError(f"{output}: {', '.join(unknown)} isn't among the evaluations")
        self.splits: dict[str, dict[str, float]] = {}
        self.loaded: dict[str, Any] = {}
        self.inputs: tuple | None = None
        self.by_gap, self.metrics, self.manifests = {}, {}, {}
        for key, folder in self.folders.items():
            for name in ("by_elo_gap.tsv", "metrics.tsv", "manifest.json"):
                run.read(folder / name)
            self.by_gap[key] = pd.read_csv(folder / "by_elo_gap.tsv", sep="\t").set_index("gap")
            self.metrics[key] = pd.read_csv(folder / "metrics.tsv", sep="\t").set_index("metric")
            self.manifests[key] = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        first = next(iter(self.by_gap.values()))
        for key, table in self.by_gap.items():
            if not np.allclose(table["consistency"], first["consistency"], equal_nan=True):
                raise ValueError(f"the evaluation {key} ran on other folds than the first")

    @property
    def keys(self) -> list[str]:
        """The evaluations shown, in the order given."""
        return list(getattr(self.settings, "shown", None) or self.folders)

    def ceiling(self) -> float:
        """The ceiling for SROCC, the Reference's ρ."""
        return float(next(iter(self.metrics.values())).at["srocc_ceiling", "value"])

    def judged(self) -> tuple[pd.DataFrame, pd.Series, pd.Series, ProtocolSettings]:
        """The judgments, the final Elo scores, each page's manuscript and the protocol's resamples, as the
        evaluations used them."""
        if self.inputs is None:
            dataset = read_dataset(needed(self.settings, "dataset", self.output))
            elo = read_elo(self.run.read(Path(needed(self.settings, "scores", self.output))))
            configuration = self.configuration()
            protocol = ProtocolSettings(bootstrap=int(configuration["bootstrap"]), seed=int(configuration["seed"]))
            manuscripts = dataset.images.set_index("image")["manuscript"]
            self.inputs = (dataset.judgments.reset_index(drop=True), elo, manuscripts, protocol)
        return self.inputs

    def evaluation(self, key: str) -> Any:
        """An evaluation read back from its hits and scores."""
        if key not in self.loaded:
            for name in ("hits.tsv", "scores.tsv"):
                self.run.read(self.folders[key] / name)
            self.loaded[key] = read_evaluation(self.folders[key], self.judged()[0])
        return self.loaded[key]

    def split(self, key: str) -> dict[str, float]:
        """The evaluation's accuracy within and across manuscripts and its ρ within manuscripts, from its hits and
        scores (`protocol.within_and_across`)."""
        if key not in self.splits:
            judgments, elo, manuscripts, protocol = self.judged()
            self.splits[key] = within_and_across(self.evaluation(key), judgments, elo, manuscripts, protocol)
        return self.splits[key]

    def label(self, key: str) -> str:
        return self.settings.labels.get(key, key)

    def kind(self, key: str) -> str:
        return self.manifests[key]["summary"].get("method", "features")

    def colour(self, key: str) -> str:
        from layout_visual_complexity.plots import style

        chosen = self.settings.colours.get(key) or (key if key in style.METHODS else KIND_COLOURS[self.kind(key)])
        return style.METHODS.get(chosen, chosen)

    def configuration(self) -> dict[str, Any]:
        return next(iter(self.manifests.values()))["configuration"]

    def member_label(self, name: str, folder: str) -> str:
        """An ensemble member's label: that of the evaluation in this table from the same folder, if any; else the
        label given for its name, and the member is defined after the table's own evaluations."""
        for key, path in self.folders.items():
            if Path(folder) == path:
                return self.label(key)
        self.members.setdefault(self.label(name), Path(folder))
        return self.label(name)

    def definition(self, key: str) -> str:
        """What the evaluation's method is, in words, from its run's record."""
        return self.define(self.manifests[key])

    def define(self, manifest: dict[str, Any]) -> str:
        """What a method is, in words, from its evaluation's run record."""
        summary, configuration = manifest["summary"], manifest["configuration"]
        kind = summary.get("method", "features")
        if kind == "tokens":
            tokens, head = summary["tokens"], summary["head"]
            backbone = BACKBONES.get(tokens["backbone"], tokens["backbone"])
            return (
                f"the local scoring head on the patch tokens of {escape(backbone)} (block {tokens['block']}, "
                f"{tokens['long_edge']} pixels on the long edge, the {tokens['crop']} crop): an MLP "
                f"({tokens['dim']}--{head['hidden']}--1, ReLU, softplus) scoring each patch, the page's score the mean "
                f"of its patches', trained with RankNet's logistic loss ({head['epochs']} epochs of Adam)"
            )
        if kind == "ensemble":
            members = [self.member_label(name, folder) for name, folder in configuration["ensemble"].items()]
            named = " and ".join(f"\\emph{{{escape(member)}}}" for member in members)
            if configuration.get("fitted_weights"):
                return f"{named} combined, their z-scored page scores weighted as each fold's training judgments fit"
            weights = configuration.get("weights") or {}
            if not weights or len(set(weights.values())) == 1:
                return f"the mean of the z-scored page scores of {named}"
            given = ", ".join(f"{escape(name)} {value:g}" for name, value in weights.items())
            return f"{named} combined, their z-scored page scores weighted {given}"
        names = summary.get("families", [])
        if summary.get("measures") and (configuration.get("columns") or set(names) & BY_MEASURE):
            return f"the pairwise logistic model on {measures_described(summary, names)}"
        families = [ADJECTIVES.get(name, name) for name in names]
        return f"the pairwise logistic model on the {summary.get('columns')} {' and '.join(families)} features"

    def definitions(self, keys: list[str] | None = None) -> str:
        """Each evaluation's definition (those shown, or those given), then those of ensembles' members that aren't
        among the evaluations."""
        found = [f"\\emph{{{escape(self.label(key))}}}, {self.definition(key)}" for key in keys or self.keys]
        for label, folder in self.members.items():
            manifest = json.loads(self.run.read(folder / "manifest.json").read_text(encoding="utf-8"))
            found.append(f"\\emph{{{escape(label)}}}, {self.define(manifest)}")
        return "; ".join(found)

    def protocol(self) -> str:
        configuration = self.configuration()
        folds = configuration["folds"]
        return (
            f"Each method is evaluated by {folds['k']}-fold cross-validation repeated {folds['repeats']} times, each "
            "manuscript's pages in one fold; a fold tests the judgments between two of its pages."
        )

    def resamples(self) -> str:
        return f"{self.configuration()['bootstrap']:,} resamples of the manuscripts"


def measure_name(feature: str) -> str:
    """A measure's name within a sentence: lower case, except for acronyms."""
    return feature if feature[:2].isupper() else feature[0].lower() + feature[1:]


def counted(count: int, noun: str) -> str:
    word = NUMBERS[count] if count < len(NUMBERS) else f"{count:,}"
    return f"{word} {noun}{'' if count == 1 else 's'}"


def measures_described(summary: dict[str, Any], families: list[str]) -> str:
    """The measures an evaluation's model took, in words: one alone, or a family's with their columns."""
    found = summary["measures"]
    columns = int(summary.get("columns", sum(measure["columns"] for measure in found)))
    if families == ["mean_token"]:
        return f"the mean of the page's patch tokens ({columns:,} dimensions)"
    names = [escape(measure_name(measure.get("feature", measure["prefix"]))) for measure in found]
    dropped = summary.get("dropped") or []
    left_out = f", {counted(len(dropped), 'column')} restating ink coverage left out" if dropped else ""
    if len(found) == 1:
        alone = f"{names[0]} alone"
        return alone if columns == 1 else f"{alone} ({counted(columns, 'column')}{left_out})"
    kind = " and ".join(ADJECTIVES.get(name, name) for name in families)
    listed = ", ".join(names[:-1]) + f" and {names[-1]}"
    return f"the {counted(len(found), kind + ' measure')}, {listed} ({columns} columns{left_out})"


def with_interval(value: float, low: float, high: float) -> str:
    if not math.isfinite(value):
        return "--"
    if not (math.isfinite(low) and math.isfinite(high)):
        return decimal(value)
    return f"{decimal(value)} [{decimal(low)}, {decimal(high)}]"


REFERENCE = (
    r"\emph{Reference} is one limit, measured for accuracy and for $\rho$: for accuracy, the consistency of the same "
    r"test judgments with the final Elo scores (the share that agree with them); for $\rho$, its ceiling, the square "
    "root of the Elo scores' reliability (their split-half correlation with its Spearman--Brown correction), the most "
    "that scores free of the annotation's noise can reach against them."
)


def by_gap_table(evaluations: Evaluations) -> pd.DataFrame:
    """Per gap bin and overall, then SROCC: the Reference (consistency for accuracy, the ceiling for SROCC) and each
    evaluation's value, with intervals."""
    first = next(iter(evaluations.by_gap.values()))
    rows = []
    for gap in first.index:
        row = {"gap": gap, "judgments_per_fold": first.at[gap, "judgments_per_fold"]}
        row |= {"reference": first.at[gap, "consistency"], "reference_low": first.at[gap, "consistency_low"],
                "reference_high": first.at[gap, "consistency_high"]}  # fmt: skip
        for key in evaluations.keys:
            table = evaluations.by_gap[key]
            row |= {key: table.at[gap, "accuracy"], f"{key}_low": table.at[gap, "accuracy_low"],
                    f"{key}_high": table.at[gap, "accuracy_high"]}  # fmt: skip
        rows.append(row)
    srocc = {"gap": "srocc", "judgments_per_fold": math.nan, "reference": evaluations.ceiling()}
    srocc |= {"reference_low": math.nan, "reference_high": math.nan}
    for key in evaluations.keys:
        found = evaluations.metrics[key]
        srocc |= {key: found.at["srocc", "value"], f"{key}_low": found.at["srocc", "low"],
                  f"{key}_high": found.at["srocc", "high"]}  # fmt: skip
    rows.append(srocc)
    return pd.DataFrame(rows)


def methods_by_elo_gap(settings: Any, run: Run, folder: Path, name: str, every_row: bool = False) -> Written:
    """The evaluations' accuracy by Elo gap and their SROCC beside the Reference: as a table under `name`, intervals in
    the Overall and ρ rows (in every row when `every_row`; always in the TSV)."""
    evaluations = Evaluations(settings, run, name)
    table = by_gap_table(evaluations)
    latex_rows = []
    for row in table.itertuples(index=False):
        values = row._asdict()
        shown = every_row or row.gap in {"overall", "srocc"}
        if row.gap == "srocc":
            cells = [r"$\rho$ (SROCC)", decimal(values["reference"])]
        else:
            cells = [gap_latex(row.gap)]
            cells.append(with_interval(row.reference, row.reference_low, row.reference_high) if shown
                         else decimal(row.reference))  # fmt: skip
        for key in evaluations.keys:
            value = values[key]
            cells.append(with_interval(value, values[f"{key}_low"], values[f"{key}_high"]) if shown else decimal(value))
        latex_rows.append(cells)
    where = "" if every_row else " in the Overall and $\\rho$ rows"
    caption = (
        r"Accuracy of pairwise outcome prediction by the absolute difference ($\Delta$) between the two pages' final "
        r"Elo scores, and Spearman's correlation ($\rho$, SROCC) between the out-of-fold page scores and the Elo "
        f"scores. {evaluations.definitions()}. {REFERENCE} {evaluations.protocol()} "
        r"\emph{Overall} is the mean over the folds of the share of test judgments whose chosen page the method "
        f"scores higher. In brackets{where}, 95\\% intervals from {evaluations.resamples()}."
    )
    latex = LatexTable(
        headers=[r"Score $\Delta$", "Reference", *[escape(evaluations.label(key)) for key in evaluations.keys]],
        rows=latex_rows,
        caption=caption,
        label=f"tab:{name}",
        align="l" + "c" * (len(latex_rows[0]) - 1),
        rules_before=[len(latex_rows) - 2, len(latex_rows) - 1],
    )
    paths = write_table(table, folder / "tables" / name, latex)
    return Written(paths, {"by_elo_gap": table.to_dict("records")})


def accuracy_by_elo_gap(settings: Any, run: Run, folder: Path, name: str = "accuracy_by_elo_gap") -> Written:
    """The evaluations' accuracy by Elo gap as a figure, each method in its colour with its interval, beside the
    Reference; and the table with every row's interval."""
    from layout_visual_complexity.plots.methods import Shown, plot_accuracy_by_gap

    written = methods_by_elo_gap(settings, run, folder, name, every_row=True)
    evaluations = Evaluations(settings, run, name)
    shown = []
    for key in evaluations.keys:
        table = evaluations.by_gap[key].reset_index()
        table = table.rename(columns={"accuracy_low": "low", "accuracy_high": "high"})
        shown.append(Shown(evaluations.label(key), evaluations.colour(key), table))
    first = next(iter(evaluations.by_gap.values())).reset_index()
    reference = first[["gap", "consistency"]].rename(columns={"consistency": "reference"})
    figure = plot_accuracy_by_gap(shown, reference)
    caption = (
        r"Pairwise accuracy on the test judgments by the absolute difference ($\Delta$) between the two pages' final "
        f"Elo scores: {evaluations.definitions()}. Bars are the mean over the folds, whiskers 95\\% intervals from "
        f"{evaluations.resamples()}; diamonds the Reference, the consistency of the same test judgments with the "
        "final Elo scores (the share that agree with them), which is the limit for accuracy as the ceiling for SROCC "
        f"({decimal(evaluations.ceiling())}) is for $\\rho$; the dashed line chance. {evaluations.protocol()}"
    )
    paths = save_figure(figure, folder / "figures" / name, caption, f"fig:{name}")
    return Written(written.paths + paths, written.summary)


def paired_differences(
    settings: Any, run: Run, folder: Path, name: str = "paired_differences", within: bool = False
) -> Written:
    """For each pair of evaluations, the first's accuracy and SROCC minus the second's, overall and within
    manuscripts, with 95% intervals from the same resamples of the manuscripts for both, as a table under `name`; its
    LaTeX shows the differences within manuscripts too when `within`."""
    evaluations = Evaluations(settings, run, name)
    pairs = needed(settings, "comparisons", name)
    judgments, elo, manuscripts, protocol = evaluations.judged()
    rows = []
    for first, second in pairs:
        if first not in evaluations.folders or second not in evaluations.folders:
            raise ValueError(f"{name}: {first} or {second} isn't among the evaluations")
        try:
            found = paired_difference(evaluations.evaluation(first), evaluations.evaluation(second), judgments,
                                      manuscripts, protocol, elo)  # fmt: skip
        except ValueError as error:
            raise ValueError(f"{first} and {second}: {error}") from None
        rows.append({"first": first, "second": second, **found})
    table = pd.DataFrame(rows)
    measures = ["accuracy", "srocc", *(["accuracy_within", "srocc_within"] if within else [])]
    latex_rows = [
        [f"{escape(evaluations.label(row['first']))} -- {escape(evaluations.label(row['second']))}"]
        + [signed_interval(row[f"{measure}_difference"], row[f"{measure}_low"], row[f"{measure}_high"])
           for measure in measures]
        for row in table.to_dict("records")
    ]  # fmt: skip
    where = (
        " overall and within manuscripts (on the test judgments between two pages of one manuscript, and Spearman's "
        "$\\rho$ with each manuscript's mean removed from both scores)"
        if within
        else ""
    )
    defined = evaluations.definitions(list(dict.fromkeys(key for pair in pairs for key in pair)))
    caption = (
        r"Differences between methods on the same folds: the first's accuracy (the mean over the folds) and SROCC "
        f"minus the second's{where}, with 95\\% intervals from {evaluations.resamples()}, the same for both methods; "
        f"an interval that excludes zero marks a difference beyond chance. {defined}."
    )
    headers = ["Methods", "Accuracy", r"$\rho$ (SROCC)"]
    if within:
        headers += ["Accuracy within", r"$\rho$ within"]
    latex = LatexTable(headers, latex_rows, caption, f"tab:{name}", align="l" + "c" * len(measures))
    paths = write_table(table, folder / "tables" / name, latex)
    return Written(paths, {"differences": table.to_dict("records")})


def signed_interval(value: float, low: float, high: float) -> str:
    return f"{value:+.3f} [{low:+.3f}, {high:+.3f}]"


def within_and_across_manuscripts(settings: Any, run: Run, folder: Path) -> Written:
    """Every evaluation's accuracy on the test judgments across manuscripts and within one, and its ρ overall and
    within manuscripts, with intervals, below the Reference's (consistency, and the ceiling for ρ overall)."""
    name = "within_and_across_manuscripts"
    evaluations = Evaluations(settings, run, name)
    first = evaluations.split(evaluations.keys[0])
    rows = [
        {
            "method": "reference",
            "label": "Reference",
            **{
                f"accuracy_{kind}{end}": first[f"consistency_{kind}{end}"]
                for kind in ("across", "within")
                for end in ("", "_low", "_high")
            },  # fmt: skip
            "srocc": evaluations.ceiling(),
            "srocc_low": math.nan,
            "srocc_high": math.nan,
            "srocc_within": math.nan,
            "srocc_within_low": math.nan,
            "srocc_within_high": math.nan,
        }
    ]
    for key in evaluations.keys:
        split, srocc = evaluations.split(key), evaluations.metrics[key].loc["srocc"]
        row = {"method": key, "label": evaluations.label(key)}
        row |= {f"accuracy_{kind}{end}": split[f"accuracy_{kind}{end}"] for kind in ("across", "within")
                for end in ("", "_low", "_high")}  # fmt: skip
        row |= {"srocc": srocc["value"], "srocc_low": srocc["low"], "srocc_high": srocc["high"]}
        row |= {f"srocc_within{end}": split[f"srocc_within{end}"] for end in ("", "_low", "_high")}
        rows.append(row)
    table = pd.DataFrame(rows)
    measures = ["accuracy_across", "accuracy_within", "srocc", "srocc_within"]
    latex_rows = [
        [escape(row["label"])] + [with_interval(row[measure], row[f"{measure}_low"], row[f"{measure}_high"])
                                  for measure in measures]
        for row in table.to_dict("records")
    ]  # fmt: skip
    across, within = (first[f"judgments_{kind}_per_fold"] for kind in ("across", "within"))
    caption = (
        "Each method across and within manuscripts: accuracy (the mean over the folds) on the test judgments between "
        f"pages of two manuscripts ({across:.1f} a fold on average) and between two pages of one manuscript "
        f"({within:.1f}); Spearman's $\\rho$ between the out-of-fold page scores and the Elo scores over all pages, "
        "and within manuscripts, with each manuscript's mean removed from both, over the manuscripts with at least "
        f"two pages. In brackets, 95\\% intervals from {evaluations.resamples()}. {REFERENCE} Within manuscripts it "
        f"has no ceiling for $\\rho$. {evaluations.definitions()}. {evaluations.protocol()}"
    )
    latex = LatexTable(
        ["Method", "Accuracy across", "Accuracy within", r"$\rho$", r"$\rho$ within"],
        latex_rows,
        caption,
        f"tab:{name}",
        align="lcccc",
        rules_before=[1],
    )
    paths = write_table(table, folder / "tables" / name, latex)
    return Written(paths, {"methods": table.to_dict("records"), "judgments_per_fold": [across, within]})


def learning_curves_and_calibration(settings: Any, run: Run, folder: Path) -> Written:
    """The evaluations' learning curves by judgments and by images, and their calibration."""
    from layout_visual_complexity.plots.methods import Shown, plot_learning_curves_and_calibration

    name = "learning_curves_and_calibration"
    evaluations = Evaluations(settings, run, name)
    curves, calibrations, rows = [], [], []
    for key in evaluations.keys:
        path = evaluations.folders[key] / "learning_curve.tsv"
        if not path.is_file():
            raise ValueError(f"{name}: the evaluation {key} has no learning curves")
        curve = pd.read_csv(run.read(path), sep="\t")
        calibration = pd.read_csv(run.read(evaluations.folders[key] / "calibration.tsv"), sep="\t")
        curves.append(Shown(evaluations.label(key), evaluations.colour(key), curve))
        calibrations.append(Shown(evaluations.label(key), evaluations.colour(key), calibration))
        rows.append(curve.assign(method=key))
    figure = plot_learning_curves_and_calibration(curves, calibrations)
    whole = rows[0][rows[0]["share"] == 1.0]
    judgments = float(whole["train_per_fold"].iloc[0]) if len(whole) else math.nan
    pages = float(whole["train_pages_per_fold"].iloc[0]) if len(whole) else math.nan
    caption = (
        "Learning curves and calibration. (a) Accuracy on each fold's test judgments when the methods are trained on "
        f"a share of the fold's training judgments (all of them, {judgments:,.0f} on average) between all its "
        f"training images; (b) on a share of its training images ({pages:,.0f} on average), with the judgments "
        f"between them; the mean over the folds, with 95\\% intervals from {evaluations.resamples()} as bands. "
        "(c) Calibration of the predicted win probabilities: per tenth of the predicted probability that the left "
        "page is chosen, the share of the test judgments that chose it; the dashed diagonal is perfect "
        f"calibration. {evaluations.definitions()}. {evaluations.protocol()}"
    )
    paths = save_figure(figure, folder / "figures" / name, caption, f"fig:{name}")
    table = pd.concat(rows, ignore_index=True)
    columns = ["method", "by", "share", "train_per_fold", "train_pages_per_fold", "accuracy", "accuracy_sd", "low",
               "high", "srocc"]  # fmt: skip
    table = table[columns]
    latex_rows = [
        [escape(evaluations.label(row.method)), row.by, f"{row.share:.0%}".replace("%", r"\%"),
         f"{row.train_per_fold:,.0f}", with_interval(row.accuracy, row.low, row.high), decimal(row.srocc)]
        for row in table.itertuples(index=False)
    ]  # fmt: skip
    latex = LatexTable(
        ["Method", "Shared out", "Share", "Training judgments", "Accuracy", r"$\rho$"],
        latex_rows,
        "Learning curves: accuracy (the mean over the folds, with its 95\\% interval) and SROCC when each method is "
        "trained on a share of each fold's training judgments or images.",
        "tab:learning_curves",
        align="llcccc",
    )
    paths += write_table(table, folder / "tables" / "learning_curves", latex)
    return Written(paths, {"learning_curves": table.to_dict("records")})


def local_complexity_maps(settings: Any, run: Run, folder: Path) -> Written:
    """The local ViT head's patch scores over the pages the settings name, on one colour scale."""
    from PIL import Image

    from layout_visual_complexity.evaluation.vit_head import TrainedHead, local_map, shipped_head
    from layout_visual_complexity.features.extraction import SegmentationImages
    from layout_visual_complexity.features.vit import TOKEN_PAGES, TOKEN_SET, cropped, read_token_set
    from layout_visual_complexity.plots.methods import MapPanel, plot_local_maps

    name = "local_complexity_maps"
    tokens = Path(needed(settings, "tokens", name))
    for file in (TOKEN_SET, TOKEN_PAGES):
        run.read(tokens / file)
    token_set = read_token_set(tokens)
    head = TrainedHead.load(run.read(Path(settings.head))) if settings.head else shipped_head()
    head.check(token_set.identity())
    pages = needed(settings, "map_pages", name)
    scores = head.scores(token_set)
    layout = Path(needed(settings, "layout", name))
    panels, described = [], []
    with SegmentationImages(layout, settings.cache, settings.http) as images:
        for page in pages:
            grid = token_set.grid(page)
            fetched = images.fetch(page)
            try:
                with Image.open(fetched.path) as opened:
                    rgb = np.asarray(opened.convert("RGB"))
            finally:
                images.release(fetched)
            width, height = (int(token_set.pages.at[page, column]) for column in ("width", "height"))
            crop = Image.fromarray(cropped(rgb, grid.box, width, height))
            crop.thumbnail((settings.map_long_edge, settings.map_long_edge), Image.Resampling.LANCZOS)
            value = float(scores[page])
            panels.append(MapPanel(np.asarray(crop), local_map(head, grid), f"{page}, $z = {value:+.2f}$"))
            described.append(f"{escape(page)} ($z = {value:+.2f}$)")
    every = np.concatenate([head.patch_scores(token_set.grid(page)) for page in token_set.pages.index])
    low, high = (float(value) for value in np.quantile(every, settings.map_range))
    figure = plot_local_maps(panels, (low, high))
    letters = ", ".join(f"({chr(ord('a') + number)}) {text}" for number, text in enumerate(described))
    above = ", ".join(f"{100 * (scores < scores[page]).mean():.0f}\\%" for page in pages)
    tokens_described = head.tokens
    backbone = BACKBONES.get(tokens_described["backbone"], tokens_described["backbone"])
    quantiles = [f"{100 * value:g}" for value in settings.map_range]
    scale = (
        f"from the {quantiles[0]}th to the {quantiles[1]}th percentile of the patch scores of all the dataset's pages "
        f"({low:.1f} to {high:.1f}), lower scores transparent and higher ones in full colour"
    )
    caption = (
        "Local complexity maps: where the ViT head finds complexity on pages of low, middle and high predicted "
        f"complexity, {letters}, scored above {above} of the dataset's pages. Each "
        f"{tokens_described['patch_size']}-pixel patch of the folio crop, resized to {tokens_described['long_edge']} "
        "pixels on its long edge, is drawn with its score under the head trained on every judgment (on the patch "
        f"tokens of {escape(backbone)}, block {tokens_described['block']}), in vermillion over the greyed page, on "
        f"one colour scale for every page, {scale}; the page's score is the mean of its patches'. $z$: the "
        "page's score, z-scored over the dataset's pages."
    )
    if settings.image_credit:
        caption += f" {escape(settings.image_credit)}"
    paths = save_figure(figure, folder / "figures" / name, caption, f"fig:{name}")
    summary = {"pages": {page: float(scores[page]) for page in pages}, "scale": [low, high]}
    return Written(paths, summary)
