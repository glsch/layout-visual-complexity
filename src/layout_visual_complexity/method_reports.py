"""The tables and figures of the methods, for `lvc report`: any number of evaluations (`lvc evaluate` folders) set side
by side, each defined in the captions from what its run recorded.

- `methods_by_elo_gap` (and `ensembles_by_elo_gap`, the same for the appendix's ensembles): accuracy by Elo gap beside
  the consistency reference, with SROCC beside its ceiling; intervals in the Overall and ρ rows (every row's in the
  TSV).
- `accuracy_by_elo_gap`: the same as a figure, and as a table with every row's interval.
- `paired_differences`: the differences between pairs of methods, in accuracy and SROCC, on the same resamples.
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
from layout_visual_complexity.evaluation.protocol import ProtocolSettings, fold_mean_intervals, paired_comparison
from layout_visual_complexity.judgment_reports import Written, decimal
from layout_visual_complexity.outputs import LatexTable, escape, gap_latex, save_figure, write_table
from layout_visual_complexity.ranking.dataset import read_dataset
from layout_visual_complexity.runs import Run

BACKBONES = {
    "facebook/dinov2-small": "DINOv2 ViT-S/14",
    "facebook/dinov3-vits16-pretrain-lvd1689m": "DINOv3 ViT-S/16",
}
ADJECTIVES = {"geometry": "geometric", "layout": "Layout"}
KIND_COLOURS = {"features": "geometry", "tokens": "vit", "ensemble": "ensemble"}


def needed(settings: Any, name: str, output: str) -> Any:
    value = getattr(settings, name)
    if value is None or value == {} or value == []:
        raise ValueError(f"{output} needs --{name}")
    return value


class Evaluations:
    """The evaluations a table or figure sets side by side: each one's tables, metrics and run record, its label and
    its colour."""

    def __init__(self, settings: Any, run: Run, output: str) -> None:
        self.settings = settings
        self.folders = {key: Path(path) for key, path in needed(settings, "evaluations", output).items()}
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
        return list(self.folders)

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
        label given for its name."""
        for key, path in self.folders.items():
            if Path(folder) == path:
                return self.label(key)
        return self.label(name)

    def definition(self, key: str) -> str:
        """What the evaluation's method is, in words, from its run's record."""
        summary, configuration = self.manifests[key]["summary"], self.manifests[key]["configuration"]
        kind = self.kind(key)
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
        families = [ADJECTIVES.get(name, name) for name in summary.get("families", [])]
        return f"the pairwise logistic model on the {summary.get('columns')} {' and '.join(families)} features"

    def definitions(self) -> str:
        return "; ".join(f"\\emph{{{escape(self.label(key))}}}, {self.definition(key)}" for key in self.keys)

    def protocol(self) -> str:
        configuration = self.configuration()
        folds = configuration["folds"]
        return (
            f"Each method is evaluated by {folds['k']}-fold cross-validation repeated {folds['repeats']} times, each "
            "manuscript's pages in one fold; a fold tests the judgments between two of its pages."
        )

    def resamples(self) -> str:
        return f"{self.configuration()['bootstrap']:,} resamples of the manuscripts"


def with_interval(value: float, low: float, high: float) -> str:
    if not math.isfinite(value):
        return "--"
    if not (math.isfinite(low) and math.isfinite(high)):
        return decimal(value)
    return f"{decimal(value)} [{decimal(low)}, {decimal(high)}]"


def by_gap_table(evaluations: Evaluations) -> pd.DataFrame:
    """Per gap bin and overall, then SROCC: the consistency reference and each evaluation's value, with intervals."""
    first = next(iter(evaluations.by_gap.values()))
    rows = []
    for gap in first.index:
        row = {"gap": gap, "judgments_per_fold": first.at[gap, "judgments_per_fold"]}
        row |= {name: first.at[gap, name] for name in ("consistency", "consistency_low", "consistency_high")}
        for key, table in evaluations.by_gap.items():
            row |= {key: table.at[gap, "accuracy"], f"{key}_low": table.at[gap, "accuracy_low"],
                    f"{key}_high": table.at[gap, "accuracy_high"]}  # fmt: skip
        rows.append(row)
    first_metrics = next(iter(evaluations.metrics.values()))
    srocc = {"gap": "srocc", "judgments_per_fold": math.nan, "consistency": first_metrics.at["srocc_ceiling", "value"]}
    srocc |= {"consistency_low": math.nan, "consistency_high": math.nan}
    for key, found in evaluations.metrics.items():
        srocc |= {key: found.at["srocc", "value"], f"{key}_low": found.at["srocc", "low"],
                  f"{key}_high": found.at["srocc", "high"]}  # fmt: skip
    rows.append(srocc)
    return pd.DataFrame(rows)


def methods_by_elo_gap(settings: Any, run: Run, folder: Path, name: str, every_row: bool = False) -> Written:
    """The evaluations' accuracy by Elo gap beside the consistency reference, and their SROCC beside its ceiling: as
    a table under `name`, intervals in the Overall and ρ rows (in every row when `every_row`; always in the TSV)."""
    evaluations = Evaluations(settings, run, name)
    table = by_gap_table(evaluations)
    latex_rows = []
    for row in table.itertuples(index=False):
        values = row._asdict()
        shown = every_row or row.gap in {"overall", "srocc"}
        if row.gap == "srocc":
            cells = [r"$\rho$ (SROCC)", decimal(values["consistency"])]
        else:
            cells = [gap_latex(row.gap)]
            cells.append(with_interval(row.consistency, row.consistency_low, row.consistency_high) if shown
                         else decimal(row.consistency))  # fmt: skip
        for key in evaluations.keys:
            value = values[key]
            cells.append(with_interval(value, values[f"{key}_low"], values[f"{key}_high"]) if shown else decimal(value))
        latex_rows.append(cells)
    where = "" if every_row else " in the Overall and $\\rho$ rows"
    caption = (
        r"Accuracy of pairwise outcome prediction by the absolute difference ($\Delta$) between the two pages' final "
        r"Elo scores, and Spearman's correlation ($\rho$, SROCC) between the out-of-fold page scores and the Elo "
        f"scores. {evaluations.definitions()}. \\emph{{Consistency}} is the share of the same test judgments that "
        r"agree with the final Elo scores, the reference for accuracy; its $\rho$ is the ceiling for SROCC, the "
        f"split-half correlation of the Elo scores with its Spearman--Brown correction. {evaluations.protocol()} "
        r"\emph{Overall} is the mean over the folds of the share of test judgments whose chosen page the method "
        f"scores higher. In brackets{where}, 95\\% intervals from {evaluations.resamples()}."
    )
    latex = LatexTable(
        headers=[r"Score $\Delta$", "Consistency", *[escape(evaluations.label(key)) for key in evaluations.keys]],
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
    consistency reference; and the table with every row's interval."""
    from layout_visual_complexity.plots.methods import Shown, plot_accuracy_by_gap

    written = methods_by_elo_gap(settings, run, folder, name, every_row=True)
    evaluations = Evaluations(settings, run, name)
    shown = []
    for key in evaluations.keys:
        table = evaluations.by_gap[key].reset_index()
        table = table.rename(columns={"accuracy_low": "low", "accuracy_high": "high"})
        shown.append(Shown(evaluations.label(key), evaluations.colour(key), table))
    reference = next(iter(evaluations.by_gap.values())).reset_index()[["gap", "consistency"]]
    figure = plot_accuracy_by_gap(shown, reference)
    caption = (
        r"Pairwise accuracy on the test judgments by the absolute difference ($\Delta$) between the two pages' final "
        f"Elo scores: {evaluations.definitions()}. Bars are the mean over the folds, whiskers 95\\% intervals from "
        f"{evaluations.resamples()}; diamonds the consistency reference, the share of the same test judgments that "
        f"agree with the final Elo scores; the dashed line chance. {evaluations.protocol()}"
    )
    paths = save_figure(figure, folder / "figures" / name, caption, f"fig:{name}")
    return Written(written.paths + paths, written.summary)


def paired_differences(settings: Any, run: Run, folder: Path) -> Written:
    """For each pair of evaluations, the first's accuracy and SROCC minus the second's, with 95% intervals from the
    same resamples of the manuscripts for both."""
    evaluations = Evaluations(settings, run, "paired_differences")
    pairs = needed(settings, "comparisons", "paired_differences")
    dataset = read_dataset(needed(settings, "dataset", "paired_differences"))
    elo = read_elo(run.read(Path(needed(settings, "scores", "paired_differences"))))
    judgments = dataset.judgments.reset_index(drop=True)
    manuscripts = dataset.images.set_index("image")["manuscript"]
    configuration = evaluations.configuration()
    protocol = ProtocolSettings(bootstrap=int(configuration["bootstrap"]), seed=int(configuration["seed"]))
    loaded = {key: read_evaluation(evaluations.folders[key], judgments) for key in evaluations.keys}
    rows = []
    for first, second in pairs:
        if first not in loaded or second not in loaded:
            raise ValueError(f"paired_differences: {first} or {second} isn't among the evaluations")
        a, b = loaded[first], loaded[second]
        merged = a.tested.merge(b.tested, on=["fold_number", "row"], suffixes=("_a", "_b"))
        if len(merged) != len(a.tested) or len(merged) != len(b.tested):
            raise ValueError(f"{first} and {second} ran on other folds")
        values = merged.assign(value=merged["hit_a"] - merged["hit_b"], gap=merged["gap_a"])
        interval = fold_mean_intervals(values, judgments, manuscripts, settings=protocol).iloc[-1]
        found = paired_comparison(a, b, judgments, manuscripts, protocol, reference=elo)
        rows.append(
            {
                "first": first,
                "second": second,
                "accuracy_difference": found["difference"],
                "accuracy_low": interval["low"],
                "accuracy_high": interval["high"],
                "srocc_difference": found["srocc_difference"],
                "srocc_low": found["srocc_low"],
                "srocc_high": found["srocc_high"],
            }
        )
    table = pd.DataFrame(rows)
    latex_rows = [
        [
            f"{escape(evaluations.label(row.first))} -- {escape(evaluations.label(row.second))}",
            signed_interval(row.accuracy_difference, row.accuracy_low, row.accuracy_high),
            signed_interval(row.srocc_difference, row.srocc_low, row.srocc_high),
        ]
        for row in table.itertuples(index=False)
    ]
    caption = (
        r"Differences between methods on the same folds: the first's accuracy (the mean over the folds) and SROCC "
        f"minus the second's, with 95\\% intervals from {evaluations.resamples()}, the same for both methods; an "
        f"interval that excludes zero marks a difference beyond chance. {evaluations.definitions()}."
    )
    latex = LatexTable(["Methods", "Accuracy", r"$\rho$ (SROCC)"], latex_rows, caption, "tab:paired_differences",
                       align="lcc")  # fmt: skip
    paths = write_table(table, folder / "tables" / "paired_differences", latex)
    return Written(paths, {"differences": table.to_dict("records")})


def signed_interval(value: float, low: float, high: float) -> str:
    return f"{value:+.3f} [{low:+.3f}, {high:+.3f}]"


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
            panels.append(MapPanel(np.asarray(crop), local_map(head, grid), f"{page}, $z$ = {value:+.2f}"))
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
