"""The tables and figures of the baselines, for `lvc report`.

- `baselines_by_information`: what each level of information buys: each evaluation's accuracy with its interval and
  its ρ, grouped by the information it needs; `baselines_by_information_each_measure`, the same with each measure
  alone beside its group combined.
- `baseline_measures`: each measure defined: its name, what it measures, its parameters and its source.
- `generic_measure_correlations`: the generic image measures' Spearman correlations with each other and with ink
  coverage, and the check of each against ink coverage.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from layout_visual_complexity.features.extraction import read_feature_table
from layout_visual_complexity.judgment_reports import Written, decimal
from layout_visual_complexity.method_reports import Evaluations, needed, with_interval
from layout_visual_complexity.outputs import LatexTable, escape, save_figure, write_table
from layout_visual_complexity.runs import Run


def baselines_by_information(settings: Any, run: Run, folder: Path, name: str) -> Written:
    """The evaluations' accuracy and ρ grouped by the information they need (`levels`, in the order of first
    appearance), the measures alone (`alone`) as open dots, as a figure and a table."""
    from layout_visual_complexity.plots.baselines import Placed, plot_by_information

    evaluations = Evaluations(settings, run, name)
    levels = needed(settings, "levels", name)
    missing = [key for key in evaluations.keys if key not in levels]
    if missing:
        raise ValueError(f"{name}: no level of information for {', '.join(missing)}")
    alone = set(settings.alone)
    rows, placed = [], []
    for key in evaluations.keys:
        overall = evaluations.by_gap[key].loc["overall"]
        srocc = evaluations.metrics[key].loc["srocc"]
        row = {
            "level": levels[key],
            "evaluation": key,
            "label": evaluations.label(key),
            "alone": key in alone,
            "accuracy": overall["accuracy"],
            "low": overall["accuracy_low"],
            "high": overall["accuracy_high"],
            "srocc": srocc["value"],
            "srocc_low": srocc["low"],
            "srocc_high": srocc["high"],
        }
        rows.append(row)
        placed.append(Placed(row["level"], row["label"], evaluations.colour(key), row["accuracy"], row["low"],
                             row["high"], row["srocc"], row["alone"]))  # fmt: skip
    table = pd.DataFrame(rows)
    consistency = float(next(iter(evaluations.by_gap.values())).loc["overall", "consistency"])
    chance = levels[evaluations.keys[0]]
    figure = plot_by_information(placed, consistency, chance)
    combined = [key for key in evaluations.keys if key not in alone]
    defined = "; ".join(
        f"\\emph{{{escape(evaluations.label(key))}}}, {evaluations.definition(key)}" for key in combined
    )
    order = ", ".join(escape(level.lower()) for level in dict.fromkeys(table["level"]))
    each = (
        " Filled dots: measures combined in one model; open dots: one measure alone, the pairwise logistic model on "
        "that measure's columns (the measures are defined in Table~\\ref{tab:baseline_measures})."
        if alone
        else ""
    )
    caption = (
        "What each level of information buys: pairwise accuracy on the test judgments (dots, the mean over the "
        f"folds, with 95\\% intervals from {evaluations.resamples()}) and Spearman's $\\rho$ between the out-of-fold "
        f"page scores and the Elo scores (right), the methods grouped by the information they need ({order}).{each} "
        f"The solid line is the consistency reference ({decimal(consistency)}), the share of the same test judgments "
        f"that agree with the final Elo scores; the dashed line is chance. {defined}. {evaluations.protocol()}"
    )
    paths = save_figure(figure, folder / "figures" / name, caption, f"fig:{name}")
    latex_rows = [
        [escape(row.level), escape(row.label), "alone" if row.alone else "combined",
         with_interval(row.accuracy, row.low, row.high), with_interval(row.srocc, row.srocc_low, row.srocc_high)]
        for row in table.itertuples(index=False)
    ]  # fmt: skip
    latex = LatexTable(
        ["Information", "Method", "Measures", "Accuracy", r"$\rho$ (SROCC)"],
        latex_rows,
        f"The methods grouped by the information they need: accuracy (the mean over the folds) and SROCC, with 95\\% "
        f"intervals from {evaluations.resamples()}. {defined}.",
        f"tab:{name}",
        align="lllcc",
    )
    paths += write_table(table, folder / "tables" / name, latex)
    return Written(paths, {"methods": table.to_dict("records"), "consistency": consistency})


def table_families(settings: Any, run: Run, output: str) -> list:
    """The families the settings name, with the settings the feature table recorded for them."""
    from layout_visual_complexity.features.contract import load_family

    table = read_feature_table(run.read(Path(needed(settings, "features", output))))
    recorded = {family["name"]: family for family in table.families}
    missing = [name for name in settings.families if name not in recorded]
    if missing:
        raise ValueError(f"{output}: the feature table has no family {', '.join(missing)}")
    return [load_family(recorded[name]["class"], recorded[name]["settings"]) for name in settings.families]


def baseline_measures(settings: Any, run: Run, folder: Path) -> Written:
    """Each measure of the families named: its family, name, code name, definition, parameters and source."""
    families = table_families(settings, run, "baseline_measures")
    rows = [
        {
            "family": group.family,
            "measure": group.feature,
            "code_name": group.prefix,
            "columns": len(group.flavours),
            "definition": group.definition,
            "parameters": group.parameters,
            "source": group.source,
            "family_name": family.name,
        }
        for family in families
        for group in family.groups()
    ]
    table = pd.DataFrame(rows)
    latex_rows, breaks = [], []
    for family, measures in table.groupby("family", sort=False):
        breaks.append(len(latex_rows))
        latex_rows.append([f"\\emph{{{escape(family)}}}", "", "", ""])
        for row in measures.itertuples(index=False):
            latex_rows.append([escape(row.measure), escape(row.definition), escape(row.parameters),
                               escape(row.source)])  # fmt: skip
    latex = LatexTable(
        headers=["Measure", "Definition", "Parameters", "Source"],
        rows=latex_rows,
        caption="The baseline measures: what each measures, its parameters and its source. The image measures work "
        "on the folio crop resized to a fixed long edge: the generic measures and the ink's after the lightness is "
        "flattened (divided by a smooth surface of the paper's brightness), the ink's on one ink layer (Sauvola's "
        "threshold of the flattened lightness, specks removed) with scales in line heights, the median spacing of "
        "the page's text lines; Kyle--Davidson's on the crop as it is, at their images' size.",
        label="tab:baseline_measures",
        align=r"lp{0.36\linewidth}p{0.24\linewidth}p{0.14\linewidth}",
        rules_before=breaks[1:],
    )
    paths = write_table(table, folder / "tables" / "baseline_measures", latex)
    return Written(paths, {"measures": len(table)})


def generic_measure_correlations(settings: Any, run: Run, folder: Path) -> Written:
    """The Spearman correlations of the families' columns with each other and with the reference column, as a
    figure; each column's correlation with the reference with its interval, and the check of the evaluations given
    (their `restating.tsv`), as a table."""
    from layout_visual_complexity.plots.baselines import plot_correlation_matrix
    from layout_visual_complexity.ranking.dataset import read_dataset
    from layout_visual_complexity.reports import correlations

    name = "generic_measure_correlations"
    table = read_feature_table(run.read(Path(needed(settings, "features", name))))
    reference = needed(settings, "reference_column", name)
    if reference not in table.values.columns:
        raise ValueError(f"{name}: the feature table has no column {reference!r}")
    columns = [column for column in table.values.columns if table.columns[column] in settings.families]
    if not columns:
        raise ValueError(f"{name}: the feature table has no column of {', '.join(settings.families)}")
    values = table.values[[reference, *columns]]
    matrix = values.corr(method="spearman")
    labels = [column.split("__", 1)[1] for column in matrix.columns]
    figure = plot_correlation_matrix(matrix, labels)
    dataset = read_dataset(needed(settings, "dataset", name))
    manuscripts = dataset.images.set_index("image")["manuscript"]
    found = correlations(values[columns], values[reference], manuscripts, settings.bootstrap, settings.seed)
    found = found.rename(columns={"rho": "rho_reference", "low": "rho_low", "high": "rho_high"})
    checked = pd.DataFrame()
    for key, path in settings.evaluations.items():
        file = Path(path) / "restating.tsv"
        if file.is_file():
            restating = pd.read_csv(run.read(file), sep="\t").assign(evaluation=key)
            checked = pd.concat([checked, restating.drop(columns=["rho"])])
    if len(checked):
        found = found.merge(checked, on="column", how="left")
    found.insert(1, "reference", reference)
    long = matrix.stack().rename("rho").rename_axis(["first", "second"]).reset_index()
    caption = (
        f"Spearman's correlations among the {len(columns)} columns of the generic image measures and with ink "
        f"coverage (\\texttt{{{escape(reference)}}}, first), over the {len(values):,} pages; blue where negative, "
        "vermillion where positive. Code names without their family's prefix (\\texttt{Generic\\_\\_})."
    )
    paths = save_figure(figure, folder / "figures" / name, caption, f"fig:{name}")
    threshold = None
    latex_rows = []
    for row in found.itertuples(index=False):
        values_row = row._asdict()
        compared = values_row.get("compared")
        difference = (
            f"{values_row['accuracy_difference']:+.3f} [{values_row['accuracy_low']:+.3f}, "
            f"{values_row['accuracy_high']:+.3f}]"
            if compared is True
            else "--"
        )
        dropped = "yes" if values_row.get("dropped") is True else "no"
        latex_rows.append([f"\\texttt{{{escape(row.column)}}}", with_interval(row.rho_reference, row.rho_low,
                           row.rho_high), difference, dropped])  # fmt: skip
    if len(checked):
        threshold = "a column whose $|\\rho|$ reaches 0.9 is set against ink coverage alone in a paired comparison"
    latex = LatexTable(
        ["Column", r"$\rho$ with ink coverage", "Accuracy added", "Left out"],
        latex_rows,
        "Each generic image measure's column against ink coverage: its Spearman correlation with it, with a 95\\% "
        f"interval from {settings.bootstrap:,} resamples of the manuscripts; "
        + (threshold or "no paired comparison")
        + " (the pairwise logistic model on both against ink coverage alone, on the same folds), and is left out of "
        "the combined generic measures when neither its accuracy's nor its SROCC's difference is beyond chance above "
        "zero.",
        f"tab:{name}",
        align="lccc",
    )
    paths += write_table(found, folder / "tables" / name, latex)
    long_path = folder / "tables" / f"{name}_matrix.tsv"
    long.to_csv(long_path, sep="\t", index=False, lineterminator="\n", float_format="%.6f")
    paths.append(long_path)
    summary = {"against_reference": found[["column", "rho_reference"]].to_dict("records")}
    return Written(paths, summary)
