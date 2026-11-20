"""The difficulty experiment's tables and figures, for `lvc report`, from `lvc difficulty` runs:

- `difficulty_against_complexity`: the end-to-end CER, on a log axis, and line F1 against predicted complexity on
  the main run's pages, a dot per page, the held-out pages marked (`figures/difficulty_against_complexity`);
- `difficulty_regression`: the mixed model on the main run's pages, a column per measure;
- `difficulty_regression_across_and_within`: least squares across manuscripts and within them;
- `difficulty_held_out`: the held-out comparison of the segmentation measures, and the held-out pages of the main run
  beside its seen pages of similar complexity (`tables/difficulty_held_out_pages`);
- `difficulty_correlations`: Spearman's ρ of each measure with complexity, overall and within manuscripts;
- `difficulty_main_text_and_margins`: line F1 and the CERs for the lines in main text zones and in margins apart;
- `difficulty_ground_truth_coverage`: the mixed model with every page and without the pages whose ground truth may
  leave text out, with each run's pages flagged and line and zone precision with and without them;
- `difficulty_by_ranker`: ρ and the mixed model for each ranker alone, the geometry ranker, the ViT head and the
  ensemble;
- `difficulty_by_quarter`: the medians of each measure in four quarters of predicted complexity;
- `difficulty_material`: each manuscript's source, century, language, pages and whether it is held out;
- `difficulty_pages`: each run's per-page rows (`difficulty/<run>.tsv`).

Every character error rate leaves out the pages whose ground-truth texts don't fit their lines, as `lvc difficulty`
flags them; the captions count them.
"""

import math
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from layout_visual_complexity import difficulty_analysis as analysis
from layout_visual_complexity.difficulty import (
    ROWS,
    GroundTruthChecks,
    leaves_text_out,
    read_manuscript_table,
    read_rows,
    texts_misplaced,
)
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
    "difficulty_ground_truth_coverage",
    "difficulty_by_ranker",
    "difficulty_by_quarter",
    "difficulty_material",
    "difficulty_pages",
)
# The rankers whose scores the runs keep, with their names in tables.
RANKERS = {"geometry": "Layout (geometry)", "vit": "ViT head", "ensemble": "Ensemble"}
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
    outside_lines: int = GroundTruthChecks.outside_lines
    outside_share: float = GroundTruthChecks.outside_share


@dataclass
class Material:
    """Every run's pages, each run's own, and the main run's and the held-out comparison's pages, prepared; the main
    run's pages whose ground-truth texts don't fit their lines, left out of the CERs, and the checks that found them."""

    rows: pd.DataFrame
    by_run: dict[str, pd.DataFrame]
    main: pd.DataFrame
    comparison: pd.DataFrame
    skew: dict[str, float] = field(default_factory=dict)
    misplaced: pd.DataFrame = field(default_factory=pd.DataFrame)
    checks: GroundTruthChecks = field(default_factory=GroundTruthChecks)

    @property
    def log_cer(self) -> bool:
        return any(value > SKEWED for value in self.skew.values())


def analysed(rows: pd.DataFrame) -> pd.DataFrame:
    """The measured pages whose ground truth has text lines: on a page whose lines are all drop capitals' or music's,
    line F1 would count only the found lines matched to none of them."""
    return rows[(rows["status"] == "done") & rows["line_f1"].notna() & (rows["lines"] > 0)]


def run_checks(folder: Path, run: Run | None = None) -> GroundTruthChecks:
    """The checks of the ground truth a `lvc difficulty` run used, from its configuration; the defaults without it."""
    path = Path(folder) / "config.yaml"
    if not path.is_file():
        return GroundTruthChecks()
    if run is not None:
        run.read(path)
    given = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("checks") or {}
    known = {item.name for item in fields(GroundTruthChecks)}
    return GroundTruthChecks(**{key: value for key, value in given.items() if key in known})


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
    in_main = measured[measured["run"] == settings.main]
    misplaced = in_main[texts_misplaced(in_main)]
    main = analysis.prepared(in_main)
    comparison = analysis.prepared(measured[measured["comparison"].astype(str) == "yes"])
    skew = {measure: analysis.skewness(main[measure]) for measure in ("cer_lines", "cer_end_to_end")}
    checks = run_checks(settings.runs[settings.main], run)
    return Material(rows, by_run, main, comparison, skew, misplaced, checks)


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
COMPLEXITY_SCORE = (
    "Predicted complexity is the expected Elo score of the shipped ensemble: a prediction, for nobody judged these "
    "pages."
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


def pages(count: int) -> str:
    return f"{count:,} {'page' if count == 1 else 'pages'}"


def misplaced_note(material: Material, subject: str = "The CERs leave") -> str:
    """A caption's sentence counting the pages the CERs leave out, `subject` its first words; empty when there are
    none."""
    count = len(material.misplaced)
    if not count:
        return ""
    checks = material.checks
    return (
        f" {subject} out {pages(count)} whose ground-truth texts don't fit their lines: at least {checks.lines} "
        "text lines read, on Kraken's line matched to them, as the text of the line just before or after (a CER "
        f"over {checks.shifted_own:.0%} against their own text, under {checks.shifted_neighbour:.0%} against the "
        "neighbour's), or read as empty through their own polygon while Kraken's line reads text."
    ).replace("%", "\\%")


def note_left_out(fits: dict[str, analysis.Fit], material: Material) -> None:
    """The CER fits' notes count the pages left out."""
    count = len(material.misplaced)
    for key, fit in fits.items():
        if count and "cer" in key:
            fit.notes.append(f"{pages(count)} whose ground-truth texts don't fit their lines left out")


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
    note_left_out(fits, material)
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
        f"variance the fixed effects explain (Nakagawa and Schielzeth). {seen_note(material)}{misplaced_note(material)}"
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
    for _, fits in blocks:
        note_left_out(fits, material)
    caption = (
        "Automatic difficulty against predicted complexity on CREMMA's pages, by least squares across manuscripts "
        "(every page pooled) and within them (each manuscript's means removed from the measure and the covariates, so "
        "the century and Latin drop out), coefficients with their 95\\% intervals from "
        f"{settings.resamples:,} resamples of the manuscripts. {COMPLEXITY} {CONTROLS} {DEFINITIONS} "
        f"{seen_note(material)}{misplaced_note(material)}"
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
    pages_table = held_out_pages_table(similar, settings, material)
    paths += write_table(similar, folder / "tables" / "difficulty_held_out_pages", pages_table)
    summary["held_out_pages"] = similar.to_dict(orient="records")
    return Written(paths, summary)


def held_out_pages_table(similar: pd.DataFrame, settings: DifficultySettings, material: Material) -> LatexTable:
    held = material.main[material.main["held_out"] == 1]
    latex = [
        [
            escape(analysis.MEASURES[row.measure]),
            f"{row.held_out_pages:,}",
            plain(row.held_out_median),
            plain(row.similar_seen_median),
            cell(row.median_difference),
            f"{row.pages_above:,}",
        ]
        for row in similar.itertuples()
    ]
    caption = (
        f"CREMMA's {len(held):,} held-out pages, from the {held['document'].nunique()} manuscripts of CATMuS's test "
        "split, beside its seen pages of similar predicted complexity (within "
        f"{settings.neighbours:g} SD): for each measure, the held-out pages' median, the median over them of their "
        "similar seen pages' medians, the median of each held-out page's difference from its similar seen pages' "
        "median, and how many held-out pages lie above it. Neither the regions model nor the recognition model "
        f"trained on the held-out pages. {DEFINITIONS}{misplaced_note(material)}"
    )
    headers = ["", "Held-out pages", "Their median", "Similar seen pages", "Median difference", "Pages above"]
    return LatexTable(headers, latex, caption, "tab:difficulty_held_out_pages")


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
        f"{DEFINITIONS} Higher is better for F1, lower for the baseline distance and the CER.{misplaced_note(material)}"
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
        f"{settings.resamples:,} resamples of the manuscripts. {DEFINITIONS}{misplaced_note(material)}"
    )
    table = LatexTable(
        ["", "Main text", "Margins"], latex, caption, "tab:difficulty_main_text_and_margins", rules_before=[2, 5]
    )
    return Written(
        write_table(pd.DataFrame(rows), folder / "tables" / "difficulty_main_text_and_margins", table),
        {key: list(value) for key, value in summary.items()},
    )


def precision(rows: pd.DataFrame, kind: str) -> float:
    """Matched over matched and invented, pooled over the pages: of zones or lines."""
    found = rows[f"{kind}_matched"].sum() + rows[f"{kind}_invented"].sum()
    return float(rows[f"{kind}_matched"].sum() / found) if found else math.nan


def run_label(material: Material, name: str) -> str:
    """A run's name in captions: its source, or its sources, and CATMuS's split when all its pages share one."""
    rows = material.by_run[name]
    sources = sorted(rows["source"].dropna().astype(str).unique()) if "source" in rows else []
    if sources and all(source.startswith("CREMMA") for source in sources):
        return "CREMMA"
    label = " and ".join(sources) or name
    splits = rows["split"].dropna().astype(str).unique() if "split" in rows else []
    return f"{label}'s {splits[0]} split" if len(splits) == 1 else label


def ground_truth_coverage(settings: DifficultySettings, material: Material, folder: Path) -> Written:
    measured = analysed(material.rows)
    flagged = leaves_text_out(measured, settings.outside_lines, settings.outside_share)
    notes, summary = [], {}
    for name, part in measured.groupby("run", sort=False):
        here = flagged[part.index]
        kept = part[~here]
        record = {
            "pages": len(part),
            "pages_flagged": int(here.sum()),
            "lines_outside": int(part["lines_outside"].sum()),
            "lines_invented": int(part["lines_invented"].sum()),
            "zones_outside": int(part["zones_outside"].sum()),
            "zones_invented": int(part["zones_invented"].sum()),
            "line_precision": precision(part, "lines"),
            "line_precision_without": precision(kept, "lines"),
            "zone_precision": precision(part, "zones"),
            "zone_precision_without": precision(kept, "zones"),
            "flagged": part.loc[here, "page"].tolist(),
        }
        summary[name] = record
        notes.append(
            f"{run_label(material, str(name))}: {record['pages_flagged']:,} of {record['pages']:,} pages flagged; "
            f"{record['lines_outside']:,} of the {record['lines_invented']:,} lines Kraken invented lie outside every "
            f"zone, and {record['zones_outside']:,} of the {record['zones_invented']:,} zones the regions model "
            f"invented; line precision {record['line_precision']:.3f}, {record['line_precision_without']:.3f} "
            f"without the flagged pages; zone precision {record['zone_precision']:.3f}, "
            f"{record['zone_precision_without']:.3f} without."
        )
    main = material.main
    without = main[~leaves_text_out(main, settings.outside_lines, settings.outside_share)]
    rows, latex = [], []
    missing = (math.nan, math.nan, math.nan)
    for key, label in measures(material).items():
        every, fewer = analysis.mixed_model(main, key), analysis.mixed_model(without, key)
        first, second = every.coefficients.get("complexity", missing), fewer.coefficients.get("complexity", missing)
        latex.append([escape(label), escape(cell(*first)), escape(cell(*second))])
        rows.append(
            {
                "measure": key,
                "estimate": first[0],
                "low": first[1],
                "high": first[2],
                "pages": every.pages,
                "without_estimate": second[0],
                "without_low": second[1],
                "without_high": second[2],
                "without_pages": fewer.pages,
                "notes": "; ".join(dict.fromkeys(every.notes + fewer.notes)),
            }
        )
        summary[key] = {"every": first, "without": second}
    caption = (
        "Ground truth that leaves text out: the mixed model's coefficient of complexity for each measure, with its "
        "95\\% Wald interval and the controls as in the regression table, on CREMMA's pages, with every page and "
        f"without the pages where at least {settings.outside_lines} of Kraken's lines, matched to none, lie outside "
        f"every ground-truth zone, making at least {settings.outside_share * 100:g}\\% of the lines found: the "
        "neighbouring leaf in the photograph, decoration taken for lines, or text the ground truth doesn't hold. "
        "Complexity is "
        f"standardised over all {len(main):,} pages in both. Every page stays in the main analysis. "
        + " ".join(notes)
        + f" {DEFINITIONS}{misplaced_note(material)}"
    )
    headers = ["", "Every page", "Without the flagged pages"]
    table = LatexTable(headers, latex, caption, "tab:difficulty_ground_truth_coverage")
    paths = write_table(pd.DataFrame(rows), folder / "tables" / "difficulty_ground_truth_coverage", table)
    return Written(paths, summary)


def by_ranker(settings: DifficultySettings, material: Material, folder: Path) -> Written:
    rankers = {column: label for column, label in RANKERS.items() if column in material.main}
    rows, latex, rules, summary = [], [], [], {}
    missing = (math.nan, math.nan, math.nan)
    for key, label in measures(material).items():
        if latex:
            rules.append(len(latex))
        latex.append([f"\\textit{{{escape(label)}}}", "", "", ""])
        for column, name in rankers.items():
            data = analysis.with_ranker(material.main, column)
            rho = analysis.correlation(data, key, settings.resamples, settings.seed)
            fit = analysis.mixed_model(data, key)
            slope = fit.coefficients.get("complexity", missing)
            latex.append([name, plain(*rho.overall), plain(*rho.within), escape(cell(*slope))])
            record = {
                "measure": key,
                "ranker": column,
                "rho": rho.overall[0],
                "rho_low": rho.overall[1],
                "rho_high": rho.overall[2],
                "rho_within": rho.within[0],
                "rho_within_low": rho.within[1],
                "rho_within_high": rho.within[2],
                "estimate": slope[0],
                "low": slope[1],
                "high": slope[2],
                "pages": fit.pages,
                "manuscripts": fit.manuscripts,
                "notes": "; ".join(fit.notes),
            }
            rows.append(record)
            summary[f"{key}: {column}"] = record
    caption = (
        f"Each ranker alone, on CREMMA's {len(material.main):,} pages: Spearman's $\\rho$ of each measure with each "
        "ranker's predicted complexity, over all pages and within manuscripts (both centred on each manuscript's "
        f"mean, over the manuscripts with at least two pages), with 95\\% intervals from {settings.resamples:,} "
        "resamples of the manuscripts, and the mixed model's coefficient of complexity with its 95\\% Wald interval "
        "(the controls and the manuscripts' random intercept as in the regression table). Each ranker's score is "
        "standardised over these pages. The features of Layout (geometry) are measured on the same lines of Kraken's "
        "and zones of the regions model's that line F1 and zone F1 judge, so a badly segmented page could look more "
        "complex and score worse for one reason; the ViT head sees only the image. The ensemble combines the two: "
        f"its rows are the main analysis's. {DEFINITIONS}{misplaced_note(material)}"
    )
    headers = ["", "$\\rho$", "$\\rho$ within", "Complexity (per SD), mixed model"]
    table = LatexTable(headers, latex, caption, "tab:difficulty_by_ranker", rules_before=rules)
    return Written(write_table(pd.DataFrame(rows), folder / "tables" / "difficulty_by_ranker", table), summary)


def median_count(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


def by_quarter(settings: DifficultySettings, material: Material, folder: Path) -> Written:
    table = analysis.by_quarter(material.main)

    def share(value: float) -> str:
        return f"{value:.1%}".replace("%", "\\%") if math.isfinite(value) else "--"

    formats = {
        "line_f1": ("Line F1", "{:.2f}".format),
        "line_recall": ("Line recall", "{:.2f}".format),
        "line_precision": ("Line precision", "{:.2f}".format),
        "cer_end_to_end": ("CER, end to end", share),
        "cer_lines": ("CER, ground-truth lines", share),
        "zone_f1": ("Zone F1", "{:.2f}".format),
        "baseline_distance": ("Baseline distance", "{:.3f}".format),
        "lines": ("Ground-truth text lines", median_count),
    }
    latex = [["Pages"] + [f"{value:,}" for value in table["pages"]]]
    ranges = zip(table["elo_low"], table["elo_high"], strict=True)
    latex.append(["Expected Elo score"] + [f"{low:,.0f}--{high:,.0f}" for low, high in ranges])
    for measure, (label, shown) in formats.items():
        latex.append([label] + [shown(value) for value in table[measure]])
    names = ["First (least complex)", "Second", "Third", "Fourth (most complex)"]
    headers = [""] + (names if len(table) == 4 else [f"Quarter {number}" for number in table["quarter"]])
    with_cer = table["cer_pages"].tolist()
    cer_note = (
        f" The CERs' medians are over {', '.join(f'{count:,}' for count in with_cer)} pages."
        if with_cer != table["pages"].tolist()
        else ""
    )
    caption = (
        f"CREMMA's {len(material.main):,} pages in {len(table)} quarters of equal size by predicted complexity, from "
        "the least to the most complex: each quarter's pages and its range of expected Elo scores, and the median "
        "over its pages of each measure. Line recall is a page's matched lines over its matched and missed ones, line "
        f"precision its matched lines over its matched and invented ones. {COMPLEXITY_SCORE} {DEFINITIONS}"
        f"{misplaced_note(material)}{cer_note}"
    )
    latex_table = LatexTable(headers, latex, caption, "tab:difficulty_by_quarter", rules_before=[2])
    paths = write_table(table, folder / "tables" / "difficulty_by_quarter", latex_table)
    return Written(paths, {"quarters": table.to_dict(orient="records")})


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


FIGURE_DEFINITIONS = (
    "Line F1 is the F1 score of Kraken's lines against the ground truth's text lines (default, heading and "
    "interlinear), matched one to one by the IoU of polygons (from 0.5), a found line matching a drop capital's or "
    "music's counting neither way. The end-to-end CER adds the missed lines' characters and the invented lines'; texts "
    "in NFC, with whitespace runs as one space."
)
SHARES = {1 / 2: "half", 1 / 3: "a third", 2 / 3: "two thirds", 1 / 4: "a quarter"}


def share_words(share: float) -> str:
    return next((words for value, words in SHARES.items() if math.isclose(share, value)), f"{share:.0%}")


def figure(settings: DifficultySettings, material: Material, folder: Path) -> Written:
    """The end-to-end CER (on a log axis) and line F1 against predicted complexity on the main run's pages, two
    panels in a row, the held-out pages marked, one trend over all of them."""
    from layout_visual_complexity.plots.difficulty import FRAC, Panel, draw

    main = material.main
    measured = material.rows[(material.rows["run"] == settings.main) & (material.rows["status"] == "done")]
    without = int(measured["line_f1"].notna().sum()) - len(main)
    without_note = (
        f" Left out: {without} {'page' if without == 1 else 'pages'} whose ground truth has no text line (every line "
        "a drop capital's or music's)."
        if without
        else ""
    )
    panels = [
        Panel("cer_end_to_end", "CER, end to end", main, log=True),
        Panel("line_f1", "Line F1", main),
    ]
    drawn, counts = draw(panels)
    held = main[main["held_out"] == 1]
    with_cer = int(main["cer_end_to_end"].notna().sum())
    name = run_label(material, settings.main)
    caption = (
        f"Automatic difficulty against predicted complexity on {name}'s pages, each dot a page: (a) the character "
        "error rate (CER) of Kraken with CATMuS Medieval's model end to end, on Kraken's own lines, on a log axis; "
        "(b) line F1 of Kraken's lines against the ground truth's text lines. "
        f"(a) shows the {with_cer:,} pages with a CER, (b) all {len(main):,}.{without_note}"
        f"{misplaced_note(material, 'The CER leaves')} Each panel's line is a robust LOWESS trend over all its "
        f"pages, each local fit over {share_words(FRAC)} of them with three robustifying iterations; in (a) it is "
        "fitted to the log of the CER, pages read without error drawn at 0.001. Vermillion: the "
        f"{len(held)} pages, from {held['document'].nunique()} manuscripts, in CATMuS's test split, which CATMuS "
        f"Medieval's model didn't train on; it trained on the other {(main['held_out'] == 0).sum():,}. "
        f"{COMPLEXITY_SCORE} {FIGURE_DEFINITIONS}"
    )
    paths = save_figure(drawn, folder / "figures" / "difficulty_against_complexity", caption,
                        "fig:difficulty_against_complexity")  # fmt: skip
    summary = {"pages": counts, "skewness": material.skew}
    summary["texts_misplaced"] = material.misplaced["page"].tolist() if len(material.misplaced) else []
    return Written(paths, summary)


MAKERS = {
    "difficulty_against_complexity": figure,
    "difficulty_regression": regression,
    "difficulty_regression_across_and_within": across_and_within,
    "difficulty_held_out": held_out,
    "difficulty_correlations": correlations,
    "difficulty_main_text_and_margins": main_text_and_margins,
    "difficulty_ground_truth_coverage": ground_truth_coverage,
    "difficulty_by_ranker": by_ranker,
    "difficulty_by_quarter": by_quarter,
    "difficulty_material": material_table,
    "difficulty_pages": pages_tables,
}


def make(output: str, settings: DifficultySettings, material: Material, folder: Path) -> Written:
    return MAKERS[output](settings, material, folder)
