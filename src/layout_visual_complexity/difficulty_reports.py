"""The difficulty experiment's tables and figures, for `lvc report`, from `lvc difficulty` runs:

- `difficulty_against_complexity`: the four measures against predicted complexity, a dot per page
  (`figures/difficulty_against_complexity`), and the same with the CER on a log axis when the CER is skewed
  (`figures/difficulty_against_complexity_log_cer`);
- `difficulty_regression`: the mixed model on the main run's pages, a column per measure;
- `difficulty_regression_across_and_within`: least squares across manuscripts and within them;
- `difficulty_held_out`: the held-out comparison of the segmentation measures, and the held-out pages of the main run
  beside its seen pages of similar complexity (`tables/difficulty_held_out_pages`, a TSV);
- `difficulty_correlations`: Spearman's ρ of each measure with complexity, overall and within manuscripts;
- `difficulty_main_text_and_margins`: line F1 and the CERs for the lines in main text zones and in margins apart;
- `difficulty_material`: each manuscript's source, century, language, pages and whether it is held out;
- `difficulty_pages`: each run's per-page rows (`difficulty/<run>.tsv`).
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from layout_visual_complexity import difficulty_analysis as analysis
from layout_visual_complexity.difficulty import ROWS, read_manuscript_table, read_rows
from layout_visual_complexity.judgment_reports import Written
from layout_visual_complexity.outputs import LatexTable, escape, save_figure, write_table
from layout_visual_complexity.runs import Run

OUTPUTS = (
    "difficulty_against_complexity",
    "difficulty_regression",
    "difficulty_regression_across_and_within",
    "difficulty_held_out",
    "difficulty_correlations",
    "difficulty_main_text_and_margins",
    "difficulty_material",
    "difficulty_pages",
)
TERM_LABELS = {
    "complexity": "Complexity (per SD)",
    "held_out": "Held out",
    "complexity:held_out": "Complexity $\\times$ held out",
    "log_lines": "Lines (log)",
    "century": "Century",
    "latin": "Latin",
    "log_resolution": "Resolution (log)",
}
SKEWED = 1.0


@dataclass(frozen=True)
class DifficultySettings:
    """What the difficulty outputs need, from `lvc report`'s settings."""

    runs: dict[str, Path]
    main: str
    manuscripts: Path | None = None
    resamples: int = 1000
    seed: int = 1
    neighbours: float = 0.25


@dataclass
class Material:
    """Every run's pages, each run's own, and the main run's and the held-out comparison's pages, prepared."""

    rows: pd.DataFrame
    by_run: dict[str, pd.DataFrame]
    main: pd.DataFrame
    comparison: pd.DataFrame
    skew: dict[str, float] = field(default_factory=dict)

    @property
    def log_cer(self) -> bool:
        return any(value > SKEWED for value in self.skew.values())


def analysed(rows: pd.DataFrame) -> pd.DataFrame:
    """The measured pages whose ground truth has text lines: on a page whose lines are all drop capitals' or music's,
    line F1 would count only the found lines matched to none of them."""
    return rows[(rows["status"] == "done") & rows["line_f1"].notna() & (rows["lines"] > 0)]


def load(settings: DifficultySettings, run: Run | None = None) -> Material:
    """The runs' rows, with the table of manuscripts joined anew when given, and the pages analysed."""
    facts = read_manuscript_table(settings.manuscripts)
    if run is not None and settings.manuscripts is not None:
        run.read(Path(settings.manuscripts))
    by_run = {}
    for name, folder in settings.runs.items():
        if run is not None:
            run.read(Path(folder) / ROWS)
        rows = read_rows(folder)
        if len(facts):
            rows = rows.drop(columns=[column for column in facts.columns if column in rows.columns])
            rows = rows.join(facts, on="document")
        by_run[name] = rows.assign(run=name)
    rows = pd.concat(by_run.values(), ignore_index=True)
    measured = analysed(rows)
    main = analysis.prepared(measured[measured["run"] == settings.main])
    comparison = analysis.prepared(measured[measured["comparison"].astype(str) == "yes"])
    skew = {measure: analysis.skewness(main[measure]) for measure in ("cer_lines", "cer_end_to_end")}
    return Material(rows, by_run, main, comparison, skew)


def cell(estimate: float, low: float = math.nan, high: float = math.nan, digits: int = 3) -> str:
    if estimate is None or not math.isfinite(estimate):
        return "--"
    found = f"{estimate:+.{digits}f}".replace("-", "−")
    if math.isfinite(low) and math.isfinite(high):
        found += f" [{low:+.{digits}f}, {high:+.{digits}f}]".replace("-", "−")
    return found


def plain(estimate: float, low: float = math.nan, high: float = math.nan, digits: int = 3) -> str:
    if estimate is None or not math.isfinite(estimate):
        return "--"
    found = f"{estimate:.{digits}f}"
    return found + (f" [{low:.{digits}f}, {high:.{digits}f}]" if math.isfinite(low) and math.isfinite(high) else "")


def measures(material: Material, segmentation_only: bool = False) -> dict[str, str]:
    """The measures a table shows, with the log CER beside the CER when it is skewed."""
    found = {
        key: label for key, label in analysis.MEASURES.items() if not segmentation_only or key in analysis.SEGMENTATION
    }
    if material.log_cer and not segmentation_only:
        found |= {"log_cer_lines": "log CER, ground-truth lines", "log_cer_end_to_end": "log CER, end to end"}
    return found


DEFINITIONS = (
    "Zone F1 is the F1 score of the regions model's zones against the ground truth's, matched one to one within each "
    "class by the IoU of boxes (from 0.5); line F1 that of Kraken's lines against the ground truth's text lines "
    "(default, heading and interlinear), matched by the IoU of polygons, a found line matching a drop capital's or "
    "music's counting neither way; the baseline distance, the mean distance between matched baselines in line "
    "heights; the CER, the character error rate of Kraken with CATMuS Medieval's model, on the ground-truth lines and "
    "end to end (on Kraken's lines, missed lines' characters and invented lines' added), texts in NFC with "
    "whitespace runs as one space."
)
COMPLEXITY = (
    "Complexity is the expected Elo score of the shipped ensemble, standardised over the pages analysed, so its "
    "coefficient is per standard deviation; these are predictions, for nobody judged these pages."
)
CONTROLS = (
    "The controls: the log of the ground truth's text lines, the century, Latin against the other languages, and the "
    "log of the median height of the ground-truth line polygons in pixels (the resolution)."
)


def seen_note(material: Material) -> str:
    held = material.main[material.main["held_out"] == 1]
    return (
        f"CATMuS's models trained on {material.main.loc[material.main['held_out'] == 0, 'document'].nunique()} "
        f"of CREMMA's {material.main['document'].nunique()} manuscripts; {len(held)} of its "
        f"{len(material.main)} pages, from {held['document'].nunique()} manuscripts, are in CATMuS's test split."
    )


def regression_rows(fits: dict[str, analysis.Fit], terms: list[str], keys: list[str]) -> list[list[str]]:
    rows = []
    for term in terms:
        rows.append(
            [TERM_LABELS[term]]
            + [cell(*fits[key].coefficients[term]) if term in fits[key].coefficients else "--" for key in keys]
        )
    return rows


def write_regression_table(
    stem: Path,
    blocks: list[tuple[str, dict[str, analysis.Fit]]],
    terms: list[str],
    columns: dict[str, str],
    caption: str,
    label: str,
    r2: str,
) -> tuple[list[Path], dict]:
    """A table of regressions: a block of rows per model, a column per measure."""
    keys = list(columns)
    tsv, latex, rules = [], [], []
    for title, fits in blocks:
        if len(blocks) > 1:
            if latex:
                rules.append(len(latex))
            latex.append([f"\\textit{{{escape(title)}}}"] + [""] * len(keys))
        for row in regression_rows(fits, terms, keys):
            latex.append([row[0], *[escape(value) for value in row[1:]]])
        latex.append(["Pages; manuscripts"] + [f"{fits[key].pages:,}; {fits[key].manuscripts}" for key in keys])
        latex.append([r2] + [plain(fits[key].r2) for key in keys])
        for term in terms:
            for key in keys:
                found = fits[key].coefficients.get(term, (math.nan, math.nan, math.nan))
                tsv.append(
                    {
                        "model": title,
                        "term": term,
                        "measure": key,
                        "estimate": found[0],
                        "low": found[1],
                        "high": found[2],
                        "pages": fits[key].pages,
                        "manuscripts": fits[key].manuscripts,
                        "r2": fits[key].r2,
                        "notes": "; ".join(fits[key].notes),
                    }
                )
    headers = [""] + [escape(columns[key]) for key in keys]
    table = LatexTable(headers, latex, caption, label, rules_before=rules)
    summary = {
        f"{title}: {key}": {term: fits[key].coefficients.get(term) for term in terms} | {"r2": fits[key].r2}
        for title, fits in blocks
        for key in keys
    }
    return write_table(pd.DataFrame(tsv), stem, table), summary


def regression(settings: DifficultySettings, material: Material, folder: Path) -> Written:
    data = material.main
    columns = measures(material)
    fits = {key: analysis.mixed_model(data, key) for key in columns}
    log_note = (
        (
            " The CER is skewed (skewness "
            + ", ".join(f"{value:.1f}" for value in material.skew.values())
            + "), so it is also modelled as log(CER + 0.001)."
        )
        if material.log_cer
        else ""
    )
    caption = (
        "Automatic difficulty against predicted complexity on CREMMA's pages: each measure regressed on complexity and "
        "the controls in a mixed model with a random intercept for the manuscript (REML), coefficients with their "
        f"95\\% Wald intervals. {COMPLEXITY} {CONTROLS} {DEFINITIONS}{log_note} The marginal R$^2$ is the share of the "
        f"variance the fixed effects explain (Nakagawa and Schielzeth). {seen_note(material)}"
    )
    paths, summary = write_regression_table(
        folder / "tables" / "difficulty_regression",
        [("Mixed model", fits)],
        analysis.TERMS,
        columns,
        caption,
        "tab:difficulty_regression",
        "Marginal R$^2$",
    )
    return Written(paths, summary)


def across_and_within(settings: DifficultySettings, material: Material, folder: Path) -> Written:
    data = material.main
    columns = measures(material)
    blocks = [
        (
            "Across manuscripts",
            {
                key: analysis.least_squares(data, key, resamples=settings.resamples, seed=settings.seed)
                for key in columns
            },
        ),
        (
            "Within manuscripts",
            {
                key: analysis.least_squares(data, key, within=True, resamples=settings.resamples, seed=settings.seed)
                for key in columns
            },
        ),
    ]
    caption = (
        "Automatic difficulty against predicted complexity on CREMMA's pages, by least squares across manuscripts "
        "(every page pooled) and within them (each manuscript's means removed from the measure and the covariates, so "
        "the century and Latin drop out), coefficients with their 95\\% intervals from "
        f"{settings.resamples:,} resamples of the manuscripts. {COMPLEXITY} {CONTROLS} {DEFINITIONS} "
        f"{seen_note(material)}"
    )
    paths, summary = write_regression_table(
        folder / "tables" / "difficulty_regression_across_and_within",
        blocks,
        analysis.TERMS,
        columns,
        caption,
        "tab:difficulty_regression_across_and_within",
        "R$^2$",
    )
    return Written(paths, summary)


def held_out(settings: DifficultySettings, material: Material, folder: Path) -> Written:
    data = material.comparison
    columns = measures(material, segmentation_only=True)
    terms = analysis.HELD_OUT_TERMS
    blocks = [
        ("Mixed model", {key: analysis.mixed_model(data, key, terms) for key in columns}),
        (
            "Across manuscripts",
            {
                key: analysis.least_squares(data, key, terms, resamples=settings.resamples, seed=settings.seed)
                for key in columns
            },
        ),
        (
            "Within manuscripts",
            {
                key: analysis.least_squares(
                    data, key, terms, within=True, resamples=settings.resamples, seed=settings.seed
                )
                for key in columns
            },
        ),
    ]
    held = data[data["held_out"] == 1]
    seen = data[data["held_out"] == 0]
    caption = (
        "The segmentation measures on pages CATMuS's models never saw against pages they trained on, all handwritten "
        f"and of the 14th to 16th century: {len(held):,} pages from {held['document'].nunique()} manuscripts of "
        f"CATMuS's test split (CREMMA's two among them) against {len(seen):,} of CREMMA's pages from "
        f"{seen['document'].nunique()} manuscripts, each measure regressed on complexity, held out and their "
        "interaction, with the controls: in a mixed model with a random intercept for the manuscript (Wald "
        "intervals), and by least squares across and within manuscripts (intervals from "
        f"{settings.resamples:,} resamples of the manuscripts; within, held out and the manuscripts' own controls "
        f"drop out). {COMPLEXITY} {CONTROLS} {DEFINITIONS}"
    )
    paths, summary = write_regression_table(
        folder / "tables" / "difficulty_held_out", blocks, terms, columns, caption, "tab:difficulty_held_out", "R$^2$"
    )
    similar = analysis.similar_pages(material.main, list(analysis.MEASURES), settings.neighbours)
    paths += write_table(similar, folder / "tables" / "difficulty_held_out_pages", None)
    summary["held_out_pages"] = similar.to_dict(orient="records")
    return Written(paths, summary)


def correlations(settings: DifficultySettings, material: Material, folder: Path) -> Written:
    held = material.comparison[material.comparison["held_out"] == 1]
    rows, latex, summary = [], [], {}
    for key, label in analysis.MEASURES.items():
        main = analysis.correlation(material.main, key, settings.resamples, settings.seed)
        line = [escape(label), plain(*main.overall), plain(*main.within)]
        record = {
            "measure": key,
            "rho": main.overall[0],
            "rho_low": main.overall[1],
            "rho_high": main.overall[2],
            "rho_within": main.within[0],
            "rho_within_low": main.within[1],
            "rho_within_high": main.within[2],
            "pages": main.pages,
            "manuscripts": main.manuscripts,
        }
        if key in analysis.SEGMENTATION and len(held):
            other = analysis.correlation(held, key, settings.resamples, settings.seed)
            line += [plain(*other.overall), plain(*other.within)]
            record |= {
                "held_out_rho": other.overall[0],
                "held_out_rho_low": other.overall[1],
                "held_out_rho_high": other.overall[2],
                "held_out_rho_within": other.within[0],
                "held_out_rho_within_low": other.within[1],
                "held_out_rho_within_high": other.within[2],
                "held_out_pages": other.pages,
                "held_out_manuscripts": other.manuscripts,
            }
        else:
            line += ["--", "--"]
        latex.append(line)
        rows.append(record)
        summary[key] = record
    caption = (
        "Spearman's $\\rho$ of each measure of automatic difficulty with predicted complexity, over all pages and "
        "within manuscripts (both centred on each manuscript's mean, over the manuscripts with at least two pages), "
        f"with 95\\% intervals from {settings.resamples:,} resamples of the manuscripts: on CREMMA's "
        f"{len(material.main):,} pages, "
        f"and on the {len(held):,} held-out pages of the comparison ({held['document'].nunique()} manuscripts of "
        f"CATMuS's test split, of the 14th to 16th century), for segmentation alone: they have no transcriptions. "
        f"{DEFINITIONS} Higher is better for F1, lower for the baseline distance and the CER."
    )
    headers = ["", "$\\rho$, CREMMA", "$\\rho$ within", "$\\rho$, held out", "$\\rho$ within"]
    table = LatexTable(headers, latex, caption, "tab:difficulty_correlations")
    return Written(write_table(pd.DataFrame(rows), folder / "tables" / "difficulty_correlations", table), summary)


def main_text_and_margins(settings: DifficultySettings, material: Material, folder: Path) -> Written:
    data = material.main.copy()
    rows, summary = [], {}
    found: dict[str, list[str]] = {part: [] for part in ("main", "margin")}
    labels = []
    for part in ("main", "margin"):
        data[f"twice_matched_{part}"] = 2 * data[f"{part}_lines_matched"]
        data[f"f1_total_{part}"] = (
            2 * data[f"{part}_lines_matched"] + data[f"{part}_lines_invented"] + data[f"{part}_lines_missed"]
        )
    for part in ("main", "margin"):
        with_lines = data[data[f"{part}_lines"] > 0]
        found[part] += [f"{len(with_lines):,}", f"{int(with_lines[f'{part}_lines'].sum()):,}"]
        for name, (top, bottom) in {
            "line_f1": (f"twice_matched_{part}", f"f1_total_{part}"),
            "cer_lines": (f"errors_lines_{part}", f"characters_{part}"),
            "cer_end_to_end": (f"errors_end_to_end_{part}", f"characters_{part}"),
        }.items():
            value = analysis.pooled(with_lines, top, bottom, settings.resamples, settings.seed)
            found[part].append(plain(*value))
            rows.append(
                {
                    "part": part,
                    "measure": name,
                    "pooled": value[0],
                    "low": value[1],
                    "high": value[2],
                    "pages": len(with_lines),
                }
            )
            summary[f"{part}_{name}"] = value
        for name, column in {
            "line_f1": f"{part}_line_f1",
            "cer_lines": f"cer_lines_{part}",
            "cer_end_to_end": f"cer_end_to_end_{part}",
        }.items():
            rho = analysis.correlation(with_lines, column, settings.resamples, settings.seed)
            found[part].append(plain(*rho.overall))
            rows.append(
                {
                    "part": part,
                    "measure": f"rho_{name}",
                    "pooled": rho.overall[0],
                    "low": rho.overall[1],
                    "high": rho.overall[2],
                    "pages": rho.pages,
                }
            )
    labels = [
        "Pages",
        "Ground-truth text lines",
        "Line F1",
        "CER, ground-truth lines",
        "CER, end to end",
        "$\\rho$: line F1",
        "$\\rho$: CER, ground-truth lines",
        "$\\rho$: CER, end to end",
    ]
    latex = [[label, found["main"][i], found["margin"][i]] for i, label in enumerate(labels)]
    caption = (
        "Main text against margins on CREMMA's pages: the ground-truth text lines in main text zones and in margin "
        "text zones apart, and Kraken's lines unmatched within those zones as the invented ones. Line F1 and the CERs "
        "pooled "
        "over the pages (all their matched, invented and missed lines; all their errors over all their reference "
        "characters), and Spearman's $\\rho$ of each page's value with predicted complexity, with 95\\% intervals from "
        f"{settings.resamples:,} resamples of the manuscripts. {DEFINITIONS}"
    )
    table = LatexTable(
        ["", "Main text", "Margins"], latex, caption, "tab:difficulty_main_text_and_margins", rules_before=[2, 5]
    )
    return Written(
        write_table(pd.DataFrame(rows), folder / "tables" / "difficulty_main_text_and_margins", table),
        {key: list(value) for key, value in summary.items()},
    )


def material_table(settings: DifficultySettings, material: Material, folder: Path) -> Written:
    rows = material.rows[(material.rows["status"] == "done") & material.rows["line_f1"].notna()]
    grouped = rows.groupby("document", sort=False)
    table = grouped.agg(
        source=("source", "first"),
        shelfmark=("shelfmark", "first"),
        century=("century", "first"),
        language=("language", "first"),
        split=("split", "first"),
        comparison=("comparison", "first"),
        pages=("page", "size"),
    ).reset_index()
    table["held_out"] = np.where(table["split"] == "test", "yes", "no")
    table = table.sort_values(["source", "century", "shelfmark"], kind="stable")
    latex = [
        [
            escape(row.shelfmark),
            escape(row.source),
            str(row.century),
            escape(row.language),
            f"{row.pages:,}",
            row.held_out,
            row.comparison,
        ]
        for row in table.itertuples()
    ]
    caption = (
        "The difficulty experiment's manuscripts: CREMMA Medieval's and CREMMA Medii Aevi's, read from their ALTO "
        "files, and the handwritten manuscripts of the 14th to 16th century in CATMuS Medieval Segmentation's test "
        "split, read "
        "from its metadata. Shelfmarks as the sources give them; the century as CREMMA's tables and CATMuS's metadata "
        "give it; pages after double-page spreads are split, each half a page, halves without ground truth left out. "
        "Held out: in CATMuS's test split, so neither the regions model nor the recognition model trained on it. "
        "Compared: in the held-out comparison of the segmentation measures."
    )
    headers = ["Shelfmark", "Source", "Century", "Language", "Pages", "Held out", "Compared"]
    latex_table = LatexTable(headers, latex, caption, "tab:difficulty_material")
    columns = ["document", "source", "shelfmark", "century", "language", "split", "held_out", "comparison", "pages"]
    paths = write_table(table[columns], folder / "tables" / "difficulty_material", latex_table)
    return Written(paths, {"manuscripts": len(table), "pages": int(table["pages"].sum())})


def pages_tables(settings: DifficultySettings, material: Material, folder: Path) -> Written:
    paths = []
    for name, rows in material.by_run.items():
        path = folder / "difficulty" / f"{name}.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.drop(columns=["image", "run"], errors="ignore").to_csv(
            path, sep="\t", index=False, lineterminator="\n", float_format="%.6f"
        )
        paths.append(path)
    return Written(paths, {name: len(rows) for name, rows in material.by_run.items()})


def figure(settings: DifficultySettings, material: Material, folder: Path) -> Written:
    from layout_visual_complexity.plots.difficulty import Panel, draw

    main = material.main
    measured = material.rows[(material.rows["status"] == "done") & material.rows["line_f1"].notna()]
    others = analysis.prepared(analysed(measured[measured["run"] != settings.main]))
    without = len(measured) - len(main) - len(others)
    without_note = (
        f" Left out: {without} {'page' if without == 1 else 'pages'} whose ground truth has no text line (every line "
        "a drop capital's or music's)."
        if without
        else ""
    )
    both = pd.concat([main, others[others["comparison"].astype(str) == "yes"]], ignore_index=True)
    in_comparison = both["comparison"].astype(str) == "yes"
    everything = pd.Series(True, index=main.index)
    paths, summary = [], {}
    for log in [False, True] if material.log_cer else [False]:
        panels = [
            Panel("cer_lines", "CER, ground-truth lines", main, everything, log),
            Panel("cer_end_to_end", "CER, end to end", main, everything, log),
            Panel("line_f1", "Line F1", both, in_comparison),
            Panel("zone_f1", "Zone F1", both, in_comparison),
        ]
        drawn, windows = draw(panels)
        held = main[main["held_out"] == 1]
        compared = both[in_comparison]
        log_note = " The CER is drawn on a log axis, pages read without error at 0.001." if log else ""
        caption = (
            "Automatic difficulty against predicted complexity, each dot a page: (a) the character error rate (CER) of "
            "Kraken with CATMuS Medieval's model on the ground-truth lines and (b) end to end, on Kraken's own lines; "
            "(c) line F1 of Kraken's lines and (d) zone F1 of the regions model's zones, against the ground truth. "
            f"(a) and (b) show CREMMA's {len(main):,} pages, (c) and (d) also the "
            f"{len(compared[compared['run'] != settings.main]):,} "
            f"pages of CATMuS's test split from its handwritten manuscripts of the 14th to 16th century.{without_note} "
            "Grey: pages "
            "CATMuS's models trained on; vermillion: held out, in CATMuS's test split "
            f"({len(held)} of CREMMA's, from {held['document'].nunique()} manuscripts). The lines are running medians "
            "over about an eighth of each group's pages, ordered by complexity; in (c) and (d) over the pages of the "
            "14th to 16th century, both sides, CREMMA's 12th- and 13th-century pages lighter and outside them. "
            "Predicted complexity is the expected Elo score of the shipped ensemble: a prediction, for nobody judged "
            f"these pages. The regions model and the recognition model trained on {(main['held_out'] == 0).sum():,} "
            f"of CREMMA's {len(main):,} pages.{log_note} {DEFINITIONS}"
        )
        stem = (
            folder / "figures" / ("difficulty_against_complexity_log_cer" if log else "difficulty_against_complexity")
        )
        paths += save_figure(drawn, stem, caption, f"fig:{stem.name}")
        summary[stem.name] = windows
    summary["skewness"] = material.skew
    return Written(paths, summary)


MAKERS = {
    "difficulty_against_complexity": figure,
    "difficulty_regression": regression,
    "difficulty_regression_across_and_within": across_and_within,
    "difficulty_held_out": held_out,
    "difficulty_correlations": correlations,
    "difficulty_main_text_and_margins": main_text_and_margins,
    "difficulty_material": material_table,
    "difficulty_pages": pages_tables,
}


def make(output: str, settings: DifficultySettings, material: Material, folder: Path) -> Written:
    return MAKERS[output](settings, material, folder)
