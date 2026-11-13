"""The `lvc report` command: tables and figures of the features and of the rankers evaluated on them.

- `geometry_by_elo_gap`: the accuracy of evaluations by final Elo gap, beside the consistency reference, with SROCC
  beside its ceiling, each with its 95% interval.
- `geometry_feature_hierarchy`: the families' features: family, feature, code name, number of flavours, definition.
- `geometry_feature_correlations`: each feature's best flavour's Spearman correlation with the Elo scores, with its
  95% interval over manuscripts; the table holds every column's.
- `geometry_features_on_page`: the geometric features drawn on real pages, one group a panel; and
  `geometry_features_on_schematic_page`, the same on a made-up page.
- `geometry_feature_families`: the plate of the geometric groups and Layout's measures on a made-up page, with their
  code names.
- The methods' tables and figures (`method_reports.py`): `methods_by_elo_gap`, `ensembles_by_elo_gap`,
  `accuracy_by_elo_gap`, `paired_differences`, `learning_curves_and_calibration`, `local_complexity_maps`.
- The baselines' (`method_reports.py`, `baseline_reports.py`): `baselines_by_elo_gap`, `baseline_differences`,
  `baseline_measures`, `baselines_by_information`, `baselines_by_information_each_measure`,
  `generic_measure_correlations`.
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import pandas as pd

from layout_visual_complexity import baseline_reports, method_reports
from layout_visual_complexity.download import HttpSettings
from layout_visual_complexity.evaluate import read_elo
from layout_visual_complexity.evaluation import metrics
from layout_visual_complexity.features.contract import Family, aggregate, load_families, load_family
from layout_visual_complexity.features.extraction import FeatureTable, SegmentationImages, read_feature_table
from layout_visual_complexity.features.geometry import Geometry, GeometrySettings
from layout_visual_complexity.features.layout import Layout, LayoutSettings
from layout_visual_complexity.features.page import Page
from layout_visual_complexity.judgment_reports import Written, decimal
from layout_visual_complexity.logs import logger
from layout_visual_complexity.outputs import LatexTable, escape, gap_latex, save_figure, write_table
from layout_visual_complexity.ranking.dataset import read_dataset
from layout_visual_complexity.ranking.reliability import spearman
from layout_visual_complexity.runs import Run
from layout_visual_complexity.segmentation.store import read_layout, read_pages

Output = Literal[
    "geometry_by_elo_gap",
    "geometry_feature_hierarchy",
    "geometry_feature_correlations",
    "geometry_features_on_page",
    "geometry_features_on_schematic_page",
    "geometry_feature_families",
    "methods_by_elo_gap",
    "ensembles_by_elo_gap",
    "accuracy_by_elo_gap",
    "paired_differences",
    "learning_curves_and_calibration",
    "local_complexity_maps",
    "baselines_by_elo_gap",
    "baseline_differences",
    "baseline_measures",
    "baselines_by_information",
    "baselines_by_information_each_measure",
    "generic_measure_correlations",
]
ALL_OUTPUTS: list[Output] = list(get_args(Output))

# The flavour a drawing of each geometric group reports.
HEADLINES = {
    "AreaFeature__WrittenSpaceRatio": "convexity",
    "AreaFeature__WrittenSpaceRectangularity": "outside_ratio",
    "AreaFeature__LineSimilarity": "dice_median",
    "AreaFeature__GraphicZoneArea": "graphic_total_area_ratio",
    "AreaFeature__DropCapitalArea": "drop_capital_num_isolated_instances",
    "LineFeature__LineLength": "var_coef",
    "LineFeature__InterlinearDistance": "max_gap_std",
    "LineFeature__ZigzagScore": "std_right",
    "LineFeature__LineParallelism": "pairwise_angle_diff_mean",
}
DRAWN = {
    "AreaFeature__WrittenSpaceRatio": "the union of the line polygons (blue) and its convex hull (dashed)",
    "AreaFeature__WrittenSpaceRectangularity": "the written space (blue), the rectangle of its area and the folio's "
    "proportions centred on it, and the written space outside it (vermillion)",
    "AreaFeature__LineSimilarity": "each line coloured by its median Dice similarity with the other lines, their "
    "polygons centred on each other: blue where it is like most others, vermillion where it is unlike them",
    "AreaFeature__GraphicZoneArea": "the graphic zones (vermillion; drop capitals dotted grey)",
    "AreaFeature__DropCapitalArea": "the drop capitals (vermillion; graphic zones dotted grey)",
    "LineFeature__LineLength": "the baselines measured (blue) and their zones' widths",
    "LineFeature__InterlinearDistance": "the baselines and, for every third pair, the distance between them",
    "LineFeature__ZigzagScore": "the lines' left (blue) and right (vermillion) ends, from line to line",
    "LineFeature__LineParallelism": "each baseline's direction, drawn across the folio",
}
LAYOUT_DRAWN = {
    "Clutter": "the zones, outlined by type",
    "Order": "the written mass's text block with its axes of symmetry, the mass's centre (dot) and the folio's (cross)",
    "Entropy": "the zones, filled by type",
    "Blank space": "the largest empty rectangle and the margins around the text block",
}


@dataclass(frozen=True)
class ReportSettings:
    """Tables and figures of the features and of the rankers evaluated on them.

    Args:
        output: Where `tables/`, `figures/` and the run's record (`runs/<name>/`) go.
        outputs: Which tables and figures to make.
        name: The run's name, for its record.
        dataset: The dataset's folder: its pages' manuscripts, for the intervals.
        scores: The ranking's scores (`scores.tsv`), whose final Elo scores the features are correlated with.
        features: A feature table (`features.parquet` of `lvc features`).
        layout: The segmentation run the features were measured on, for the pages drawn.
        evaluations: Evaluations (`lvc evaluate` folders) to set side by side, under the short names their columns
            take.
        labels: Each evaluation's name in the tables.
        colours: Each evaluation's colour in the figures: a method's (`geometry`, `vit`, `ensemble`, …) or a hex
            code; by default its short name's, if it is a method's, else its kind's.
        comparisons: Pairs of evaluations (by short name) whose differences `paired_differences` gives, the first
            minus the second.
        families: The families the hierarchy table shows (short names or Python paths); those whose features'
            correlations with the Elo scores are drawn, those the measures' table defines, and those whose columns
            the generic measures' correlations show (short names).
        pages: The real page each geometric group is drawn on, by its code name (`Family__Feature`).
        bootstrap: Resamples of the manuscripts for the correlations' 95% intervals.
        seed: The seed of the resamples.
        image_credit: A line crediting the page images, added to the captions of figures that show them.
        tokens: A token set (the folder of `lvc embed`), for the local complexity maps.
        head: The trained local ViT head (`head.npz`) the maps come from; the package's own by default.
        map_pages: The pages the local complexity maps are drawn on.
        map_long_edge: The long edge, in pixels, the maps' page images are drawn at.
        map_range: The maps' colour scale runs between these quantiles of the patch scores of every page of the
            token set, lower scores transparent, higher ones in full colour.
        cache: The download cache holding the page images; by default the segmentation run's.
        http: How to fetch page images that aren't in the cache.
        levels: Each evaluation's level of information (its group in the figures of the baselines), by short name;
            the levels in the order they first come.
        alone: The evaluations of one measure alone, drawn as open dots.
        reference_column: The column the generic measures are set against (ink coverage).
    """

    output: Path
    outputs: list[Output] = field(default_factory=lambda: list(ALL_OUTPUTS))
    name: str = "report"
    dataset: Path | None = None
    scores: Path | None = None
    features: Path | None = None
    layout: Path | None = None
    evaluations: dict[str, Path] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    colours: dict[str, str] = field(default_factory=dict)
    comparisons: list[list[str]] = field(default_factory=list)
    families: list[str] = field(default_factory=lambda: ["geometry", "layout"])
    pages: dict[str, str] = field(default_factory=dict)
    bootstrap: int = 1000
    seed: int = 1
    image_credit: str = ""
    tokens: Path | None = None
    head: Path | None = None
    map_pages: list[str] = field(default_factory=list)
    map_long_edge: int = 900
    map_range: list[float] = field(default_factory=lambda: [0.05, 0.95])
    cache: Path | None = None
    http: HttpSettings = field(default_factory=HttpSettings)
    levels: dict[str, str] = field(default_factory=dict)
    alone: list[str] = field(default_factory=list)
    reference_column: str = ""


def needed(settings: ReportSettings, name: str, output: str) -> object:
    value = getattr(settings, name)
    if value is None or value == {}:
        raise ValueError(f"{output} needs --{name}")
    return value


def run_report(settings: ReportSettings) -> dict[str, object]:
    """Write the chosen tables and figures under the output folder, and the run's record in `runs/<name>/`."""
    run = Run("report", settings)
    folder = Path(settings.output)
    timings = {}
    for output in dict.fromkeys(settings.outputs):
        began = time.perf_counter()
        logger.info("report: %s", output)
        if output == "geometry_by_elo_gap":
            written = by_elo_gap(settings, run, folder)
        elif output == "geometry_feature_hierarchy":
            written = feature_hierarchy(load_families(settings.families), folder)
        elif output == "geometry_feature_correlations":
            written = feature_correlations(settings, run, folder)
        elif output == "geometry_features_on_page":
            written = features_on_page(settings, run, folder)
        elif output == "geometry_features_on_schematic_page":
            written = features_on_schematic_page(folder)
        elif output == "geometry_feature_families":
            written = feature_families(folder)
        elif output in METHOD_OUTPUTS:
            written = METHOD_OUTPUTS[output](settings, run, folder)
        else:
            raise ValueError(f"unknown output {output!r}")
        timings[output] = round(time.perf_counter() - began, 3)
        run.wrote(written.paths)
        run.summary[output] = written.summary
        print(f"{output}: {', '.join(str(path) for path in written.paths)}")
    run.summary["seconds"] = timings
    run.wrote(run.finish(folder / "runs" / settings.name))
    return run.summary


METHOD_OUTPUTS = {
    "methods_by_elo_gap": lambda settings, run, folder: method_reports.methods_by_elo_gap(
        settings, run, folder, "methods_by_elo_gap"
    ),
    "ensembles_by_elo_gap": lambda settings, run, folder: method_reports.methods_by_elo_gap(
        settings, run, folder, "ensembles_by_elo_gap"
    ),
    "accuracy_by_elo_gap": method_reports.accuracy_by_elo_gap,
    "paired_differences": method_reports.paired_differences,
    "learning_curves_and_calibration": method_reports.learning_curves_and_calibration,
    "local_complexity_maps": method_reports.local_complexity_maps,
    "baselines_by_elo_gap": lambda settings, run, folder: method_reports.methods_by_elo_gap(
        settings, run, folder, "baselines_by_elo_gap"
    ),
    "baseline_differences": lambda settings, run, folder: method_reports.paired_differences(
        settings, run, folder, "baseline_differences"
    ),
    "baseline_measures": baseline_reports.baseline_measures,
    "baselines_by_information": lambda settings, run, folder: baseline_reports.baselines_by_information(
        settings, run, folder, "baselines_by_information"
    ),
    "baselines_by_information_each_measure": lambda settings, run, folder: baseline_reports.baselines_by_information(
        settings, run, folder, "baselines_by_information_each_measure"
    ),
    "generic_measure_correlations": baseline_reports.generic_measure_correlations,
}


# Accuracy by Elo gap.


def with_interval(value: float, low: float, high: float) -> str:
    if not math.isfinite(value):
        return "--"
    if not (math.isfinite(low) and math.isfinite(high)):
        return decimal(value)
    return f"{decimal(value)} [{decimal(low)}, {decimal(high)}]"


def by_elo_gap(settings: ReportSettings, run: Run, folder: Path) -> Written:
    """The evaluations' accuracy by final Elo gap beside the consistency reference, and their SROCC beside its
    ceiling."""
    evaluations = needed(settings, "evaluations", "geometry_by_elo_gap")
    tables, measures = {}, {}
    for key, path in evaluations.items():  # type: ignore[union-attr]
        for name in ("by_elo_gap.tsv", "metrics.tsv", "manifest.json"):
            run.read(Path(path) / name)
        tables[key] = pd.read_csv(Path(path) / "by_elo_gap.tsv", sep="\t").set_index("gap")
        measures[key] = pd.read_csv(Path(path) / "metrics.tsv", sep="\t").set_index("metric")
    first = next(iter(tables.values()))
    for key, table in tables.items():
        if not np.allclose(table["consistency"], first["consistency"], equal_nan=True):
            raise ValueError(f"the evaluation {key} ran on other folds than the first")
    rows = []
    for gap in first.index:
        row = {"gap": gap, "judgments_per_fold": first.at[gap, "judgments_per_fold"]}
        row |= {name: first.at[gap, name] for name in ("consistency", "consistency_low", "consistency_high")}
        for key, table in tables.items():
            row |= {key: table.at[gap, "accuracy"], f"{key}_low": table.at[gap, "accuracy_low"],
                    f"{key}_high": table.at[gap, "accuracy_high"]}  # fmt: skip
        rows.append(row)
    first_metrics = next(iter(measures.values()))
    srocc = {"gap": "srocc", "judgments_per_fold": math.nan, "consistency": first_metrics.at["srocc_ceiling", "value"]}
    srocc |= {"consistency_low": math.nan, "consistency_high": math.nan}
    for key, found in measures.items():
        srocc |= {key: found.at["srocc", "value"], f"{key}_low": found.at["srocc", "low"],
                  f"{key}_high": found.at["srocc", "high"]}  # fmt: skip
    rows.append(srocc)
    table = pd.DataFrame(rows)
    labels = [settings.labels.get(key, key) for key in tables]
    latex_rows = []
    for row in table.itertuples(index=False):
        values = row._asdict()
        if row.gap == "srocc":
            cells = [r"$\rho$ (SROCC)", "", decimal(values["consistency"])]
        else:
            cells = [gap_latex(row.gap), f"{row.judgments_per_fold:.1f}"]
            cells.append(with_interval(row.consistency, row.consistency_low, row.consistency_high))
        cells += [with_interval(values[key], values[f"{key}_low"], values[f"{key}_high"]) for key in tables]
        latex_rows.append(cells)
    first_folder = Path(next(iter(evaluations.values())))  # type: ignore[union-attr]
    configuration = json.loads((first_folder / "manifest.json").read_text(encoding="utf-8"))["configuration"]
    folds = configuration["folds"]
    latex = LatexTable(
        headers=[r"Score $\Delta$", "Judgments per fold", "Consistency", *[escape(label) for label in labels]],
        rows=latex_rows,
        caption=r"Accuracy of pairwise outcome prediction by the absolute difference ($\Delta$) between the two pages' "
        "final Elo scores, and Spearman's correlation ($\\rho$, SROCC) between the out-of-fold page scores and the "
        f"Elo scores. Each method is the pairwise logistic model fitted to a feature set, evaluated by "
        f"{folds['k']}-fold cross-validation repeated {folds['repeats']} times, each manuscript's pages in one fold; "
        "a fold tests the judgments between two of its pages. Accuracy is the mean over the folds of the share of "
        "test judgments whose chosen page the model scores higher; \\emph{Consistency} is the same share for the "
        "final Elo scores, on the same test judgments, and the reference for accuracy. Its $\\rho$ is the ceiling "
        "for SROCC: the square root of the Elo scores' reliability (their split-half correlation with its "
        "Spearman--Brown correction), the most that scores free of the annotation's noise can reach against them. "
        f"\\emph{{Judgments per fold}} counts the test judgments. In brackets, 95\\% intervals from "
        f"{configuration['bootstrap']:,} resamples of the manuscripts.",
        label="tab:geometry_by_elo_gap",
        align="l" + "c" * (len(latex_rows[0]) - 1),
        rules_before=[len(latex_rows) - 2, len(latex_rows) - 1],
    )
    paths = write_table(table, folder / "tables" / "geometry_by_elo_gap", latex)
    return Written(paths, {"by_elo_gap": table.to_dict("records")})


# The families' features.


def code(name: str) -> str:
    """A code name in LaTeX's typewriter type, breakable after its double underscores."""
    return r"\texttt{" + escape(name).replace(r"\_\_", r"\_\_\allowbreak{}") + "}"


def feature_hierarchy(families: list[Family], folder: Path) -> Written:
    """The families' features: family, feature, code name, number of flavours, definition."""
    rows = [
        {
            "family": group.family,
            "feature": group.feature,
            "code_name": group.prefix,
            "flavours": len(group.flavours),
            "definition": group.definition,
            "columns": " ".join(group.flavours),
            "family_name": family.name,
        }
        for family in families
        for group in family.groups()
    ]
    table = pd.DataFrame(rows)
    counts = [str(sum(len(group.flavours) for group in family.groups())) for family in families]
    latex_rows = [[escape(row.family), escape(row.feature), code(row.code_name), str(row.flavours),
                   escape(row.definition)] for row in table.itertuples(index=False)]  # fmt: skip
    latex_rows.append([r"\textbf{Total}", "", "", r"\textbf{" + " + ".join(counts) + "}", ""])
    kinds = table["family"].tolist()
    breaks = [number for number in range(1, len(kinds)) if kinds[number] != kinds[number - 1]]
    latex = LatexTable(
        headers=["Family", "Feature", "Code name", "$n$", "Definition and flavours"],
        rows=latex_rows,
        caption="The features: each feature's family, its code name (columns are named "
        r"\texttt{Family\_\_Feature\_\_Flavour}), its number of flavours $n$, and their definitions. ``Range'' "
        "(\\texttt{max\\_dev}) is the maximum minus the minimum. Lengths and offsets are relative to the zone's width, "
        "interlinear distances to the folio's diagonal, areas to the folio's area. The Layout family adapts "
        "Kyle--Davidson's dimensions of scene complexity to the page.",
        label="tab:geometry_feature_hierarchy",
        align=r"lcccp{0.38\linewidth}",
        rules_before=[*breaks, len(latex_rows) - 1],
    )
    paths = write_table(table, folder / "tables" / "geometry_feature_hierarchy", latex)
    names = [family.name for family in families]
    return Written(paths, {"features": len(table), "columns": dict(zip(names, counts, strict=True))})


# Each feature's correlation with the Elo scores.


def table_families(table: FeatureTable) -> list[Family]:
    """The families that computed a table, with their settings."""
    return [load_family(described["class"], described["settings"]) for described in table.families]


def correlations(
    values: pd.DataFrame, elo: pd.Series, manuscripts: pd.Series, resamples: int, seed: int
) -> pd.DataFrame:
    """Each column's Spearman ρ with the Elo scores over the pages with a value, and its 95% interval from
    resamples of the manuscripts (a page as often as its manuscript is drawn)."""
    pages = values.index.intersection(elo.index)
    names = sorted(manuscripts.reindex(pages).dropna().unique())
    number = {name: index for index, name in enumerate(names)}
    multiplicities = metrics.manuscript_multiplicities(names, resamples, seed)
    page_manuscript = manuscripts.reindex(pages).map(number).to_numpy()
    reference = elo[pages].to_numpy(dtype=float)
    rows = []
    for column in values.columns:
        found = values.loc[pages, column].to_numpy(dtype=float)
        kept = np.isfinite(found)
        if kept.sum() < 3 or np.ptp(found[kept]) == 0:
            rows.append({"column": column, "rho": math.nan, "low": math.nan, "high": math.nan})
            rows[-1]["pages"] = int(kept.sum())
            continue
        rho = spearman(found[kept], reference[kept])
        resampled = metrics.bootstrap_srocc(found[kept], reference[kept], page_manuscript[kept], multiplicities)
        low, high = metrics.interval(resampled)
        rows.append({"column": column, "rho": rho, "low": low, "high": high, "pages": int(kept.sum())})
    return pd.DataFrame(rows)


def feature_correlations(settings: ReportSettings, run: Run, folder: Path) -> Written:
    """Each feature's best flavour's correlation with the Elo scores, as a figure; every column's in the table."""
    from layout_visual_complexity.plots.features import plot_feature_correlations

    path = Path(needed(settings, "features", "geometry_feature_correlations"))  # type: ignore[arg-type]
    table = read_feature_table(run.read(path)).select(settings.families)
    elo = read_elo(run.read(Path(needed(settings, "scores", "geometry_feature_correlations"))))  # type: ignore[arg-type]
    dataset = read_dataset(needed(settings, "dataset", "geometry_feature_correlations"))  # type: ignore[arg-type]
    manuscripts = dataset.images.set_index("image")["manuscript"]
    families = {family.name: family for family in table_families(table)}
    found = correlations(table.values, elo, manuscripts, settings.bootstrap, settings.seed)
    found["family"] = found["column"].map(table.columns)
    found["feature"] = found["column"].str.rsplit("__", n=1).str[0]
    found["words"] = [families[family].words(column) for column, family in zip(found["column"], found["family"],
                                                                                strict=True)]  # fmt: skip
    strength = found["rho"].abs().fillna(-1)
    found["best"] = strength == strength.groupby(found["feature"]).transform("max")
    found.loc[found.duplicated(["feature", "best"]) & found["best"], "best"] = False
    best = found[found["best"]].sort_values("rho", key=lambda values: -values.abs()).reset_index(drop=True)
    figure = plot_feature_correlations(best)
    pages = int(found["pages"].max())
    described = "; ".join(f"\\texttt{{{escape(row.column)}}}, {row.words}" for row in best.itertuples(index=False))
    caption = (
        "Spearman's correlation ($\\rho$) between each feature's best flavour (the one whose correlation is "
        f"strongest) and the pages' Elo scores, over the {pages:,} pages, with 95\\% intervals from "
        f"{settings.bootstrap:,} resamples of the manuscripts; blue where negative, vermillion where positive. From "
        f"top to bottom: {described}."
    )
    paths = save_figure(figure, folder / "figures" / "geometry_feature_correlations", caption,
                        "fig:geometry_feature_correlations")  # fmt: skip
    ordered = found.sort_values(["family", "feature", "column"], kind="stable")
    columns = ["column", "family", "feature", "rho", "low", "high", "pages", "best", "words"]
    latex = LatexTable(
        headers=["Feature", r"$\rho$", r"95\% interval"],
        rows=[
            [code(row.column), decimal(row.rho), f"[{decimal(row.low)}, {decimal(row.high)}]"]
            for row in best.itertuples(index=False)
        ],  # fmt: skip
        caption="Spearman's correlation between each feature's best flavour and the pages' Elo scores, with 95\\% "
        f"intervals from {settings.bootstrap:,} resamples of the manuscripts. The TSV gives every flavour's.",
        label="tab:geometry_feature_correlations",
        align="lcc",
    )
    paths += write_table(ordered[columns], folder / "tables" / "geometry_feature_correlations", latex)
    return Written(paths, {"best": best[["column", "rho", "low", "high"]].to_dict("records")})


# The features drawn on pages.


def read_page(layout: Path, page: str) -> Page:
    """A page of a segmentation run, from its layout store."""
    pages = read_pages(layout / "pages.parquet").set_index("page")
    if page not in pages.index:
        raise ValueError(f"the segmentation run has no page {page}")
    rows = read_layout(layout / "layout" / f"{pages.at[page, 'document']}.parquet")
    return Page.from_layout(page, rows[rows["page"] == page], pages.at[page, "width"], pages.at[page, "height"])


def page_image(images: SegmentationImages, page: str, long_edge: int = 900) -> np.ndarray:
    """A page's image, made smaller for a figure."""
    from PIL import Image

    fetched = images.fetch(page)
    try:
        with Image.open(fetched.path) as opened:
            image = opened.convert("RGB")
            image.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
            return np.asarray(image)
    finally:
        images.release(fetched)


def headline(group: str, value: float) -> str:
    flavour = HEADLINES[group]
    shown = f"{value:.0f}" if flavour.endswith("instances") else f"{value:.3f}"
    return f"\\texttt{{{escape(flavour)}}} {shown}"


def features_on_page(settings: ReportSettings, run: Run, folder: Path) -> Written:
    """The geometric groups drawn on the real pages the settings name, one panel each."""
    from layout_visual_complexity.plots.features import GROUP_DRAWINGS, plot_features_on_pages

    layout = Path(needed(settings, "layout", "geometry_features_on_page"))  # type: ignore[arg-type]
    chosen = needed(settings, "pages", "geometry_features_on_page")
    unknown = sorted(set(chosen) - set(GROUP_DRAWINGS))  # type: ignore[arg-type]
    if unknown:
        raise ValueError(f"no drawing for {', '.join(unknown)}: give pages for {', '.join(GROUP_DRAWINGS)}")
    family = Geometry()
    panels, described = [], []
    with SegmentationImages(layout, settings.cache, settings.http) as images:
        for group in GROUP_DRAWINGS:
            if group not in chosen:  # type: ignore[operator]
                continue
            page = read_page(layout, chosen[group])  # type: ignore[index]
            values = aggregate(family, family.measure(page))
            panels.append((group, page, page_image(images, page.page)))
            value = values[f"{group}__{HEADLINES[group]}"]
            described.append(f"{DRAWN[group]}, on {escape(page.page)} ({headline(group, value)})")
    figure = plot_features_on_pages(panels, family.settings)
    letters = [f"({chr(ord('a') + number)}) {text}" for number, text in enumerate(described)]
    caption = "The geometric features drawn on pages of the dataset: " + "; ".join(letters) + "."
    if settings.image_credit:
        caption += f" {escape(settings.image_credit)}"
    stem = folder / "figures" / "geometry_features_on_page"
    paths = save_figure(figure, stem, caption, "fig:geometry_features_on_page")
    return Written(paths, {"pages": dict(chosen)})  # type: ignore[arg-type]


def features_on_schematic_page(folder: Path) -> Written:
    """The geometric groups drawn on a made-up page, with its values."""
    from layout_visual_complexity.plots.features import GROUP_DRAWINGS, plot_features_on_pages, schematic_page

    page = schematic_page()
    family = Geometry()
    values = aggregate(family, family.measure(page))
    figure = plot_features_on_pages([(group, page, None) for group in GROUP_DRAWINGS], family.settings)
    described = []
    for number, group in enumerate(GROUP_DRAWINGS):
        shown = headline(group, values[f"{group}__{HEADLINES[group]}"])
        described.append(f"({chr(ord('a') + number)}) {DRAWN[group]} ({shown})")
    caption = "The geometric features drawn on a made-up page: " + "; ".join(described) + "."
    paths = save_figure(figure, folder / "figures" / "geometry_features_on_schematic_page", caption,
                        "fig:geometry_features_on_schematic_page")  # fmt: skip
    return Written(paths, {"values": {group: values[f"{group}__{HEADLINES[group]}"] for group in GROUP_DRAWINGS}})


def feature_families(folder: Path) -> Written:
    """The plate: the nine geometric groups and Layout's four measures on a made-up page, with their code names."""
    from layout_visual_complexity.plots.features import GROUP_DRAWINGS, plot_feature_families, schematic_page

    figure = plot_feature_families(schematic_page(), GeometrySettings(), LayoutSettings())
    drawn = [DRAWN[group] for group in GROUP_DRAWINGS] + list(LAYOUT_DRAWN.values())
    names = [group.split("__")[1] for group in GROUP_DRAWINGS] + list(LAYOUT_DRAWN)
    described = [f"({chr(ord('a') + number)}) {escape(name)}: {text}" for number, (name, text) in
                 enumerate(zip(names, drawn, strict=True))]  # fmt: skip
    caption = (
        "The features, each drawn on the same made-up page, with its code name below: the nine groups of geometric "
        "features, then the four measures of the Layout family. " + "; ".join(described) + "."
    )
    paths = save_figure(figure, folder / "figures" / "geometry_feature_families", caption,
                        "fig:geometry_feature_families")  # fmt: skip
    return Written(paths, {"panels": len(drawn), "layout": Layout().name})
