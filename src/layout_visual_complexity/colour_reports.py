"""The tables and figures of the colour level, for `lvc report`.

- `colour_by_elo_gap`: Appendix B's successor: line texture, real colour and geometry with real colour by Elo gap
  beside the Reference, with ρ, accuracy and ρ within manuscripts, and colour's value added to geometry.
- `colour_differences`: what colour adds, as paired differences in accuracy and ρ, overall and within manuscripts
  (`method_reports.paired_differences`).
- `line_texture_families`: Appendix B's table of the line-texture measures' families.
- `colour_measures`: each real colour measure defined: its name, what it measures, its parameters and its source.
- `ink_colour_on_page`: pages whose ink is drawn by hue class, beside the classes' shares of the ink.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from layout_visual_complexity.baseline_reports import baseline_measures
from layout_visual_complexity.evaluation.protocol import paired_difference
from layout_visual_complexity.features.line_texture import PREFIX, TEXTURE_FAMILIES
from layout_visual_complexity.judgment_reports import Written, decimal
from layout_visual_complexity.method_reports import (
    REFERENCE,
    Evaluations,
    by_gap_table,
    needed,
    signed_interval,
    with_interval,
)
from layout_visual_complexity.outputs import LatexTable, escape, gap_latex, save_figure, write_table
from layout_visual_complexity.runs import Run

COLOUR_MEASURES = (
    "The real colour measures: what each measures, its parameters and its source. They work on the ink layer of the "
    "baseline measures (the folio crop at a fixed long edge, its lightness flattened, Sauvola's threshold, specks "
    "removed), each linear RGB channel divided by the paper's so that the parchment is neutral, then in CIELAB: an "
    "ink pixel's chroma is the length of its $(a^*, b^*)$ and its hue the angle. Zones and lines are the "
    "segmentation's, mapped onto the crop. The ink within stamps is left out: the libraries' stamps are modern."
)


def colour_by_elo_gap(settings: Any, run: Run, folder: Path) -> Written:
    """The shown evaluations' accuracy by Elo gap and ρ beside the Reference, their accuracy and ρ within manuscripts,
    and the value each comparison's first adds over its second, in the first's column."""
    name = "colour_by_elo_gap"
    evaluations = Evaluations(settings, run, name)
    judgments, elo, manuscripts, protocol = evaluations.judged()
    table = by_gap_table(evaluations).rename(columns={"gap": "row"})
    first = evaluations.split(evaluations.keys[0])
    within = []
    for measure in ("accuracy_within", "srocc_within"):
        reference = "consistency_within" if measure == "accuracy_within" else None
        row: dict[str, Any] = {"row": measure, "judgments_per_fold": math.nan}
        row |= {f"reference{end}": first[f"{reference}{end}"] if reference else math.nan
                for end in ("", "_low", "_high")}  # fmt: skip
        for key in evaluations.keys:
            row |= {f"{key}{end}": evaluations.split(key)[f"{measure}{end}"] for end in ("", "_low", "_high")}
        within.append(row)
    added, compared = [], []
    for measure in ("accuracy", "srocc", "accuracy_within", "srocc_within"):
        row = {"row": f"added_{measure}", "judgments_per_fold": math.nan}
        row |= dict.fromkeys(("reference", "reference_low", "reference_high"), math.nan)
        for key in evaluations.keys:
            row |= dict.fromkeys((key, f"{key}_low", f"{key}_high"), math.nan)
        added.append(row)
    for better, base in needed(settings, "comparisons", name):
        if better not in evaluations.keys or base not in evaluations.folders:
            raise ValueError(f"{name}: compare a shown evaluation with another evaluation, not {better} and {base}")
        found = paired_difference(evaluations.evaluation(better), evaluations.evaluation(base), judgments,
                                  manuscripts, protocol, elo)  # fmt: skip
        compared.append((better, base))
        for row, measure in zip(added, ("accuracy", "srocc", "accuracy_within", "srocc_within"), strict=True):
            row |= {better: found[f"{measure}_difference"], f"{better}_low": found[f"{measure}_low"],
                    f"{better}_high": found[f"{measure}_high"]}  # fmt: skip
    table = pd.concat([table, pd.DataFrame(within + added)], ignore_index=True)
    labels = {
        "overall": "Overall",
        "srocc": r"$\rho$ (SROCC)",
        "accuracy_within": "Accuracy within",
        "srocc_within": r"$\rho$ within",
        "added_accuracy": "Added: accuracy",
        "added_srocc": r"Added: $\rho$",
        "added_accuracy_within": "Added: accuracy within",
        "added_srocc_within": r"Added: $\rho$ within",
    }
    latex_rows = []
    for values in table.to_dict("records"):
        shown = values["row"] in {"overall", "srocc", "accuracy_within", "srocc_within"}
        added_row = values["row"].startswith("added_")
        cells = [labels.get(values["row"]) or gap_latex(values["row"])]
        for key in ("reference", *evaluations.keys):
            value, low, high = values[key], values[f"{key}_low"], values[f"{key}_high"]
            if added_row:
                cells.append(signed_interval(value, low, high) if math.isfinite(value) else "")
            elif shown:
                cells.append(with_interval(value, low, high))
            else:
                cells.append(decimal(value))
        latex_rows.append(cells)
    rows = table["row"].tolist()
    comparisons = "; ".join(
        f"\\emph{{{escape(evaluations.label(better))}}} minus \\emph{{{escape(evaluations.label(base))}}}"
        for better, base in compared
    )
    bases = [base for _, base in compared if base not in evaluations.keys]
    caption = (
        r"Accuracy of pairwise outcome prediction by the absolute difference ($\Delta$) between the two pages' final "
        r"Elo scores, Spearman's correlation ($\rho$) between the out-of-fold page scores and the Elo scores, and "
        "both within manuscripts: the accuracy on the test judgments between two pages of one manuscript, and $\\rho$ "
        "with each manuscript's mean removed from both scores, over the manuscripts with at least two pages. "
        f"{evaluations.definitions([*evaluations.keys, *bases])}. {REFERENCE} Within manuscripts the Reference for "
        r"accuracy is the consistency of those judgments; $\rho$ has none. \emph{Added}: what colour adds, "
        f"{comparisons}, on the same folds, with 95\\% intervals from the same resamples for both. "
        f"{evaluations.protocol()} In brackets, 95\\% intervals from {evaluations.resamples()}."
    )
    latex = LatexTable(
        headers=[r"Score $\Delta$", "Reference", *[escape(evaluations.label(key)) for key in evaluations.keys]],
        rows=latex_rows,
        caption=caption,
        label=f"tab:{name}",
        align="l" + "c" * (len(latex_rows[0]) - 1),
        rules_before=[rows.index("overall"), rows.index("srocc"), rows.index("accuracy_within"),
                      rows.index("added_accuracy")],
    )  # fmt: skip
    paths = write_table(table, folder / "tables" / name, latex)
    return Written(paths, {"rows": table.to_dict("records")})


def colour_differences(settings: Any, run: Run, folder: Path) -> Written:
    """What colour adds: the comparisons' paired differences, overall and within manuscripts."""
    from layout_visual_complexity.method_reports import paired_differences

    return paired_differences(settings, run, folder, "colour_differences", within=True)


def line_texture_families(folder: Path) -> Written:
    """Appendix B's families of line-texture measures: family, number of measures, what they measure, and their
    columns."""
    rows = [
        {"family": family, "count": len(keys), "measurements": text, "columns": " ".join(keys)}
        for family, keys, text in TEXTURE_FAMILIES
    ]
    table = pd.DataFrame(rows)
    latex_rows = [[escape(row["family"]), str(row["count"]), escape(row["measurements"])] for row in rows]
    latex_rows.append(["Total", str(int(table["count"].sum())), ""])
    caption = (
        "The families of line-texture measures. For each text line, an axis-aligned crop of the gamma-corrected page "
        "image, the vertical Sobel response of its red, green and blue channels, partitioned into superpixels by SLIC "
        "on a min--max-normalised copy; the measures come from the raw responses and are averaged over the folio's "
        "lines. They describe tonal transitions around the strokes, not hue or saturation. Columns are named "
        f"\\texttt{{{escape(PREFIX)}\\_\\_<measure>}}."
    )
    latex = LatexTable(["Family", "$N$", "Measurements"], latex_rows, caption, "tab:line_texture_families",
                       align=r"lrp{0.6\linewidth}", rules_before=[len(latex_rows) - 1])  # fmt: skip
    paths = write_table(table, folder / "tables" / "line_texture_families", latex)
    return Written(paths, {"families": len(rows), "measures": int(table["count"].sum())})


def colour_measures(settings: Any, run: Run, folder: Path) -> Written:
    """Each real colour measure defined, as `baseline_measures` defines the baselines'."""
    return baseline_measures(settings, run, folder, "colour_measures", COLOUR_MEASURES)


def ink_colour_on_page(settings: Any, run: Run, folder: Path) -> Written:
    """The pages the settings name, each one's ink drawn by hue class against the neutral parchment, beside the
    classes' shares of its ink."""
    from layout_visual_complexity.features.colour import Colour, ColourSettings, colour_layer
    from layout_visual_complexity.features.contract import aggregate, load_family
    from layout_visual_complexity.features.extraction import SegmentationImages, read_feature_table
    from layout_visual_complexity.features.ink_colour import HUE_CLASSES, hue_class
    from layout_visual_complexity.plots.colour import ColourPanel, plot_ink_colour
    from layout_visual_complexity.reports import read_page

    name = "ink_colour_on_page"
    layout = Path(needed(settings, "layout", name))
    pages = needed(settings, "colour_pages", name)
    family: Any = Colour()
    if settings.features:
        recorded = {item["name"]: item for item in read_feature_table(run.read(Path(settings.features))).families}
        if "colour" in recorded:
            family = load_family(recorded["colour"]["class"], recorded["colour"]["settings"])
    colour_settings: ColourSettings = family.settings
    panels, described = [], []
    with SegmentationImages(layout, settings.cache, settings.http) as images:
        for page_id in pages:
            page = read_page(layout, page_id)
            fetched = images.fetch(page_id)
            try:
                page.image_path = fetched.path
                colour, ink, chromatic = colour_layer(page, colour_settings)
                values = aggregate(family, family.measure(page))
            finally:
                images.release(fetched)
            classes = np.zeros(ink.shape, np.int8)
            classes[ink] = 1
            for number, bounds in enumerate(HUE_CLASSES.values(), start=2):
                classes[chromatic & hue_class(colour.hue, bounds)] = number
            classes[colour.layer.ink & ~ink] = -1
            shares = {cls: values[f"Colour__HueClasses__{cls}"] for cls in HUE_CLASSES}
            panels.append(ColourPanel(page_id, classes, colour.layer.folio.inside, shares))
            listed = ", ".join(f"{cls} {100 * share:.1f}\\%" for cls, share in shares.items() if share >= 0.0005)
            described.append(
                f"{escape(page_id)}: {100 * values['Colour__ChromaticShare__share']:.1f}\\% of the ink chromatic "
                f"({listed or 'none'}), {values['Colour__HueEntropy__bits']:.2f} bits of hue entropy, "
                f"{values['Colour__Lines__rubric_share'] * 100:.0f}\\% rubric lines, "
                f"{int(values['Colour__ColouredInitials__count'])} coloured initials"
            )
    figure = plot_ink_colour(panels)
    letters = "; ".join(f"({chr(ord('a') + number)}) {text}" for number, text in enumerate(described))
    caption = (
        "Ink colour on three pages: each page's ink as the colour measures see it, the parchment made neutral "
        f"(plain ink brown, chromatic ink, of chroma at least {colour_settings.chroma:g}, in the colour of its hue "
        "class: red, gold, green or blue), beside the classes' shares of the ink. "
        f"{letters}. Ink within stamps, left out of the measures, is light grey."
    )
    if settings.image_credit:
        caption += f" {escape(settings.image_credit)}"
    paths = save_figure(figure, folder / "figures" / name, caption, f"fig:{name}")
    rows = [{"page": panel.page, **panel.shares} for panel in panels]
    table = pd.DataFrame(rows)
    latex = LatexTable(
        ["Page", *[cls.capitalize() for cls in HUE_CLASSES]],
        [[escape(row["page"]), *[f"{100 * row[cls]:.1f}" for cls in HUE_CLASSES]] for row in rows],
        "The hue classes' shares of the ink (\\%) on the pages of Figure~\\ref{fig:ink_colour_on_page}.",
        "tab:ink_colour_on_page",
        align="lcccc",
    )
    paths += write_table(table, folder / "tables" / name, latex)
    return Written(paths, {"pages": rows})
