"""The `lvc difficulty` command: how hard pages are for the automatic tools, beside their predicted complexity.

Pages with ground truth (ALTO files, or a CATMuS-style `metadata.jsonl` with its images beside it) go through
`lvc score`'s stream, double-page spreads split into their pages first, and through Kraken's text recognition on the
pages that are transcribed. Each page then gets one row:

- its predicted complexity: the rankers' z-scores, the ensemble's expected Elo score, and how many of the geometry
  ranker's features lie beyond the bounds it clips them to, a sign of how far the page lies from its training pages;
- zone F1: the regions model's zones against the ground truth's, matched within each class by the IoU of boxes;
- line F1 and baseline distance: Kraken's lines against the ground truth's text lines (`DefaultLine`, `HeadingLine`,
  `InterlinearLine`), matched by the IoU of polygons; a found line matching another kind of ground-truth line (a
  drop capital's, music) counts neither way. The distance between matched baselines is in line heights;
- the character error rate of the recognition model on the ground truth's lines, and end to end, on Kraken's lines
  matched as for line F1, with the missed lines' characters and the invented lines' added;
- the same for the lines in main text zones and in margins apart;
- checks of the ground truth: how many of its text lines read as a neighbouring line's text (a shifted text), how
  many read as empty through their own polygon while the found line matched to them reads text, and whether either
  count flags the page's texts as misplaced; how many unmatched found lines and zones lie outside every ground-truth
  zone, where the ground truth may leave text out;
- the controls: the ground truth's lines, the median height of its line polygons (the resolution), and the columns
  of a table of manuscripts, when one is given.

The run keeps the rows (`difficulty.tsv`), the zones by class (`zone_classes.tsv`), every zone and line with its
match (`zones.parquet`, `lines.parquet`), each spread's split (`spreads.tsv`), and what `lvc score` keeps; never an
image or a token. Its report lists the pages the checks flag. The measures are computed from the run's records at
the end of every run, so they don't depend on the workers or the order pages arrive in; a rerun resumes.
"""

import math
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from layout_visual_complexity import ground_truth as truth
from layout_visual_complexity import layout_matching as matching
from layout_visual_complexity.inputs import Document, PageSource
from layout_visual_complexity.inputs.resolve import Resolver
from layout_visual_complexity.logs import logger
from layout_visual_complexity.pipeline.step import Step
from layout_visual_complexity.runs import shown_path
from layout_visual_complexity.score import SCORES, ScoreSettings, Scoring, feature_families
from layout_visual_complexity.score_steps import TokenStepSettings, scoring_steps
from layout_visual_complexity.scoring import ScoringModels
from layout_visual_complexity.segment import run_segment
from layout_visual_complexity.segmentation.recognition import PageLines, RecognitionSettings, RecognitionStep
from layout_visual_complexity.segmentation.records import PageRecord
from layout_visual_complexity.segmentation.steps import segmentation_steps
from layout_visual_complexity.segmentation.store import box_points, scaled, scaler
from layout_visual_complexity.spreads import Split, SpreadSettings, split_spreads
from layout_visual_complexity.text_errors import edit_distance, normalise, rate

ROWS = "difficulty.tsv"
ZONE_CLASSES = "zone_classes.tsv"
LINES = "lines.parquet"
ZONES = "zones.parquet"
SPREADS = "spreads.tsv"

# The regions model's 13 zones, the classes zone F1 is measured on.
MODEL_ZONES = [
    "DamageZone",
    "DigitizationArtefactZone",
    "DropCapitalZone",
    "GraphicZone",
    "MainZone",
    "MarginTextZone",
    "MusicZone",
    "NumberingZone",
    "QuireMarksZone",
    "RunningTitleZone",
    "SealZone",
    "StampZone",
    "TitlePageZone",
]
TEXT_LINES = ("DefaultLine", "HeadingLine", "InterlinearLine")
# The zones whose lines are measured apart.
PARTS = {"main": "MainZone", "margin": "MarginTextZone"}


@dataclass(frozen=True)
class GroundTruthChecks:
    """Checks of the ground truth, whose flagged pages the run's report lists.

    A ground-truth text line with a reference and a found line matched to it, texts normalised as for the CER, has a
    shifted text when the found line reads, against the line's own text, a CER over `shifted_own`, and against the
    text of the text line just before or after it (in the ground truth's order), a CER under `shifted_neighbour`; it
    reads empty when its own polygon reads as empty and the found line reads text. A page with at least `lines`
    lines of either kind has its texts flagged as misplaced, and `lvc report` leaves it out of the CERs.

    A page's ground truth may leave text out when at least `outside_lines` unmatched found lines lie outside every
    ground-truth zone, making at least `outside_share` of the lines found.

    Args:
        shifted_own: A shifted text's CER against the line's own text is over this.
        shifted_neighbour: A shifted text's CER against a neighbouring line's text is under this.
        lines: A page's texts are misplaced with at least this many shifted texts, or this many lines read empty.
        outside_lines: Unmatched found lines outside every zone, at least, for a page that may leave text out.
        outside_share: Their share of the lines found, at least.
    """

    shifted_own: float = 0.5
    shifted_neighbour: float = 0.25
    lines: int = 3
    outside_lines: int = 5
    outside_share: float = 0.1


@dataclass(frozen=True)
class DifficultySettings(ScoreSettings):
    """Measure how hard pages with ground truth are for the automatic tools, beside their predicted complexity.

    Args:
        sources: ALTO files, folders holding them (at any depth), lists of these, or a `metadata.jsonl` with its
            images beside it; given on the command line, or here.
        output: The run's folder: `difficulty.tsv`, `zone_classes.tsv`, `lines.parquet`, `zones.parquet`,
            `spreads.tsv`, and what `lvc score` keeps.
        alto_files: In folders, the ALTO files read: a pattern of their names (`*.chocomufin.xml`, say).
        manuscripts: In a `metadata.jsonl`, the manuscripts read, by their folders; all by default.
        spreads: The double-page spreads, split into their two pages before the folio crop.
        recognition: Kraken's text recognition on the transcribed pages.
        manuscript_table: A table of the manuscripts (TSV, by `folder`), whose columns join each page's row.
        zone_classes: The classes zone F1 is measured on: the regions model's zones.
        threshold: The IoU from which a found zone or line matches a ground-truth one.
        checks: The checks of the ground truth: texts that don't fit their lines, and text the ground truth leaves out.
    """

    output: Path = Path("build/difficulty")
    alto_files: str = "*.xml"
    manuscripts: list[str] = field(default_factory=list)
    spreads: SpreadSettings = field(default_factory=SpreadSettings)
    recognition: RecognitionSettings = field(default_factory=RecognitionSettings)
    manuscript_table: Path | None = None
    zone_classes: list[str] = field(default_factory=lambda: list(MODEL_ZONES))
    threshold: float = matching.THRESHOLD
    checks: GroundTruthChecks = field(default_factory=GroundTruthChecks)


# The measures of one page.


@dataclass
class Found:
    """What the tools found on a page, in its own full-resolution pixels: zones (class, box) and lines (polygon,
    baseline), with the texts read on the lines when the page was recognised."""

    zones: list[tuple[str, tuple[float, float, float, float]]]
    lines: list[tuple[list[tuple[float, float]], list[tuple[float, float]]]]
    texts: list[str] | None = None


def found_layout(record: PageRecord) -> Found:
    """A record's zones and lines in the page's full-resolution pixels."""
    scale = scaler(record)
    zones = []
    for region in record.regions:
        x0, y0, x1, y1 = region.box
        zones.append((region.cls, (x0 * scale[0], y0 * scale[1], x1 * scale[0], y1 * scale[1])))
    lines = [
        (
            [tuple(point) for point in scaled(line.boundary, scale)],
            [tuple(point) for point in scaled(line.baseline, scale)],
        )
        for line in record.lines
    ]
    texts = list(record.recognition["lines"]) if record.recognition else None
    return Found(zones, lines, texts)  # type: ignore[arg-type]


@dataclass
class Measured:
    """A page's measures, and its zones and lines with their matches, as rows."""

    row: dict[str, Any]
    zone_classes: list[dict[str, Any]]
    zones: list[dict[str, Any]]
    lines: list[dict[str, Any]]


def errors(pairs: list[tuple[str, str]], spaces: bool = True) -> tuple[int, int]:
    """Total edit distance and reference characters over (reference, recognised) pairs."""
    total = characters = 0
    for reference, recognised in pairs:
        first, second = normalise(reference, spaces), normalise(recognised, spaces)
        total += edit_distance(first, second)
        characters += len(first)
    return total, characters


def misfits(
    counted: list[truth.Line],
    matched: dict[int, tuple[int, float]],
    recognised: dict[str, str],
    texts: list[str],
    checks: GroundTruthChecks,
) -> tuple[int, int]:
    """How many of a page's text lines (in the ground truth's order) have a shifted text, and how many read empty,
    as `GroundTruthChecks` defines them; `matched` maps a text line to the found line matched to it."""
    references = [normalise(line.text) for line in counted]

    def cer(reference: str, reading: str) -> float:
        return rate(edit_distance(reference, reading), len(reference))

    shifted = empty = 0
    for j, line in enumerate(counted):
        hit = matched.get(j)
        if not references[j] or hit is None:
            continue
        reading = normalise(texts[hit[0]])
        if reading and not normalise(recognised.get(line.id)):
            empty += 1
        neighbours = [references[k] for k in (j - 1, j + 1) if 0 <= k < len(counted) and references[k]]
        if cer(references[j], reading) > checks.shifted_own and any(
            cer(neighbour, reading) < checks.shifted_neighbour for neighbour in neighbours
        ):
            shifted += 1
    return shifted, empty


def measure_page(
    page: truth.Page,
    found: Found,
    recognised: dict[str, str] | None = None,
    zone_classes: list[str] | None = None,
    threshold: float = matching.THRESHOLD,
    checks: GroundTruthChecks | None = None,
) -> Measured:
    """A page's zone F1, line F1, baseline distance and character error rates against its ground truth, and the
    checks of its texts.

    `recognised` holds the texts read on its ground-truth lines, by id; `found.texts` those read on the found lines.
    Without them the character error rates and the checks of the texts are missing."""
    checks = checks or GroundTruthChecks()
    classes = list(zone_classes or MODEL_ZONES)
    row: dict[str, Any] = {}
    zone_rows, zone_class_rows, line_rows = [], [], []

    # Zones, class by class.
    truth_zones = [zone for zone in page.zones if zone.cls in classes and len(zone.polygon) >= 3]
    zone_match: dict[int, tuple[int, float]] = {}
    total = matching.Counts()
    for cls in classes:
        mine = [i for i, (name, _) in enumerate(found.zones) if name == cls]
        theirs = [j for j, zone in enumerate(truth_zones) if zone.cls == cls]
        if not mine and not theirs:
            continue
        iou = matching.box_iou(
            np.array([found.zones[i][1] for i in mine]).reshape(-1, 4),
            np.array([matching.bounds(truth_zones[j].polygon) for j in theirs]).reshape(-1, 4),
        )
        pairs = matching.match(iou, threshold)
        for i, j, value in pairs:
            zone_match[mine[i]] = (theirs[j], value)
        counts = matching.Counts(len(pairs), len(mine) - len(pairs), len(theirs) - len(pairs))
        total += counts
        zone_class_rows.append(
            {
                "page": page.page,
                "class": cls,
                "matched": counts.matched,
                "invented": counts.invented,
                "missed": counts.missed,
            }
        )
    row |= {
        "zones_found": len(found.zones),
        "zones_truth": len(truth_zones),
        "zones_matched": total.matched,
        "zones_invented": total.invented,
        "zones_missed": total.missed,
        "zone_f1": total.f1,
    }
    matched_truth_zones = {j: (i, value) for i, (j, value) in zone_match.items()}

    # Lines: every ground-truth line is a candidate, only the text lines count.
    counted = [line for line in page.lines if line.cls in TEXT_LINES]
    others = [line for line in page.lines if line.cls not in TEXT_LINES]
    candidates = counted + others
    iou = matching.polygon_iou([polygon for polygon, _ in found.lines], [line.polygon for line in candidates])
    pairs = matching.match(iou, threshold)
    to_counted = {i: (j, value) for i, j, value in pairs if j < len(counted)}
    to_other = {i for i, j, _ in pairs if j >= len(counted)}
    unmatched = [i for i in range(len(found.lines)) if i not in to_counted and i not in to_other]
    counted_match = {j: (i, value) for i, (j, value) in to_counted.items()}
    lines = matching.Counts(len(to_counted), len(unmatched), len(counted) - len(to_counted))
    heights = [matching.line_height(line.polygon, line.baseline) for line in counted]
    finite = [value for value in heights if math.isfinite(value)]
    height = statistics.median(finite) if finite else math.nan
    distances = (
        {
            j: matching.baseline_distance(found.lines[i][1], counted[j].baseline) / height
            for j, (i, _) in counted_match.items()
        }
        if height and math.isfinite(height)
        else {}
    )
    finite_distances = [value for value in distances.values() if math.isfinite(value)]
    row |= {
        "lines": len(counted),
        "line_height": height,
        "lines_found": len(found.lines),
        "lines_matched": lines.matched,
        "lines_invented": lines.invented,
        "lines_missed": lines.missed,
        "line_f1": lines.f1,
        "baseline_distance": float(np.mean(finite_distances)) if finite_distances else math.nan,
        "baseline_pairs": len(finite_distances),
    }

    # Found zones and lines outside every ground-truth zone, unmatched: where the ground truth may leave text out.
    all_zones = [zone.polygon for zone in page.zones if len(zone.polygon) >= 3]
    line_centres = [matching.centroid(polygon) for polygon, _ in found.lines]
    outside_lines = ~matching.inside_any(line_centres, all_zones)
    zone_centres = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for _, box in found.zones]
    outside_zones = ~matching.inside_any(zone_centres, all_zones)
    row["lines_outside"] = int(sum(bool(outside_lines[i]) for i in unmatched))
    row["zones_outside"] = int(sum(bool(outside_zones[i]) for i in range(len(found.zones)) if i not in zone_match))

    # Character errors.
    texts = found.texts
    with_text = [line for line in counted if normalise(line.text)]
    rates = recognised is not None and texts is not None and page.transcribed

    def cer_of(chosen: list[truth.Line], invented: list[int], suffix: str) -> dict[str, float]:
        if not rates:
            return {
                f"characters{suffix}": math.nan,
                f"cer_lines{suffix}": math.nan,
                f"cer_lines_no_spaces{suffix}": math.nan,
                f"cer_end_to_end{suffix}": math.nan,
                f"cer_end_to_end_no_spaces{suffix}": math.nan,
            }
        found_rates = {}
        references = [line for line in chosen if normalise(line.text)]
        for spaces, name in ((True, ""), (False, "_no_spaces")):
            on_lines = errors([(line.text or "", recognised.get(line.id, "")) for line in references], spaces)
            ended, characters = 0, 0
            index = {id(line): j for j, line in enumerate(counted)}
            for line in references:
                reference = normalise(line.text, spaces)
                characters += len(reference)
                hit = counted_match.get(index[id(line)])
                ended += edit_distance(reference, normalise(texts[hit[0]], spaces)) if hit else len(reference)
            ended += sum(len(normalise(texts[i], spaces)) for i in invented)
            found_rates[f"cer_lines{name}{suffix}"] = rate(*on_lines)
            found_rates[f"cer_end_to_end{name}{suffix}"] = rate(ended, characters)
            if spaces:
                found_rates[f"characters{suffix}"] = characters
                found_rates[f"errors_lines{suffix}"] = on_lines[0]
                found_rates[f"errors_end_to_end{suffix}"] = ended
        return found_rates

    # A found line matched to a text line without a reference counts neither way end to end: it is neither among
    # the references nor invented.
    row |= cer_of(with_text, unmatched, "")

    # Ground-truth texts that don't fit their lines: read on the found lines matched to them, or read empty.
    if rates:
        shifted, empty = misfits(counted, counted_match, recognised or {}, texts or [], checks)
        row |= {
            "lines_shifted": shifted,
            "lines_read_empty": empty,
            "texts_misplaced": shifted >= checks.lines or empty >= checks.lines,
        }
    else:
        row |= {"lines_shifted": math.nan, "lines_read_empty": math.nan, "texts_misplaced": False}

    # Main text and margins apart.
    for part, cls in PARTS.items():
        chosen = [line for line in counted if line.zone_cls == cls]
        zones_of = [zone.polygon for zone in page.zones if zone.cls == cls and len(zone.polygon) >= 3]
        inside = matching.inside_any(line_centres, zones_of)
        invented = [i for i in unmatched if inside[i]]
        chosen_ids = {id(line) for line in chosen}
        matched_here = sum(1 for j in counted_match if id(counted[j]) in chosen_ids)
        part_counts = matching.Counts(matched_here, len(invented), len(chosen) - matched_here)
        row |= {
            f"{part}_lines": len(chosen),
            f"{part}_lines_matched": part_counts.matched,
            f"{part}_lines_invented": part_counts.invented,
            f"{part}_lines_missed": part_counts.missed,
            f"{part}_line_f1": part_counts.f1,
        }
        row |= cer_of(chosen, invented, f"_{part}")

    # The zones and lines, with their matches.
    for j, zone in enumerate(truth_zones):
        hit = matched_truth_zones.get(j)
        zone_rows.append(
            {
                "page": page.page,
                "source": "truth",
                "id": zone.id,
                "class": zone.cls,
                "polygon": zone.polygon,
                "match": f"r{hit[0] + 1}" if hit else None,
                "iou": hit[1] if hit else None,
            }
        )
    for i, (cls, box) in enumerate(found.zones):
        hit = zone_match.get(i)
        zone_rows.append(
            {
                "page": page.page,
                "source": "found",
                "id": f"r{i + 1}",
                "class": cls,
                "polygon": box_points(box),
                "match": truth_zones[hit[0]].id if hit else None,
                "iou": hit[1] if hit else None,
            }
        )
    by_candidate = {j: (i, value) for i, j, value in pairs}
    for j, line in enumerate(candidates):
        hit = by_candidate.get(j)
        line_rows.append(
            {
                "page": page.page,
                "source": "truth",
                "id": line.id,
                "class": line.cls,
                "zone_class": line.zone_cls,
                "polygon": line.polygon,
                "baseline": line.baseline,
                "text": line.text,
                "recognised": (recognised or {}).get(line.id),
                "match": f"l{hit[0] + 1}" if hit else None,
                "iou": hit[1] if hit else None,
                "baseline_distance": distances.get(j),
            }
        )
    for i, (polygon, baseline) in enumerate(found.lines):
        hit = by_candidate_of(pairs, i)
        line_rows.append(
            {
                "page": page.page,
                "source": "found",
                "id": f"l{i + 1}",
                "class": None,
                "zone_class": None,
                "polygon": polygon,
                "baseline": baseline,
                "text": None,
                "recognised": texts[i] if texts else None,
                "match": candidates[hit[0]].id if hit else None,
                "iou": hit[1] if hit else None,
                "baseline_distance": distances.get(hit[0]) if hit else None,
            }
        )
    return Measured(row, zone_class_rows, zone_rows, line_rows)


def by_candidate_of(pairs: list[tuple[int, int, float]], found: int) -> tuple[int, float] | None:
    for i, j, value in pairs:
        if i == found:
            return j, value
    return None


# The pages and their spreads.


def page_source(page: truth.Page, position: int) -> PageSource:
    width, height = page.size
    return PageSource(
        page.page,
        page.document,
        position,
        str(page.source),
        page.label,
        path=str(page.image),
        width=width,
        height=height,
        region=page.region,
    )


def read_pages(settings: DifficultySettings, jobs: int) -> tuple[list[truth.Page], list[Split]]:
    """The ground truth's pages, the spreads split, in the sources' order."""
    pages = truth.read_sources(list(settings.sources), settings.alto_files, list(settings.manuscripts))
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        found = list(pool.map(lambda page: split_spreads([page], settings.spreads), pages))
    kept = [half for halves, _ in found for half in halves]
    return kept, [split for _, splits in found for split in splits]


def read_manuscript_table(path: Path | None) -> pd.DataFrame:
    """A table of manuscripts by folder, with its folder made a document id."""
    if path is None:
        return pd.DataFrame()
    from layout_visual_complexity.inputs.selection import document_id

    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    table["document"] = table["folder"].map(document_id)
    return table.set_index("document")


class Difficulty(Scoring):
    """What `lvc difficulty` adds to scoring: the pages from their ground truth, and the measures against it."""

    command = "difficulty"
    title = "Difficulty run"

    def __init__(
        self,
        settings: DifficultySettings,
        models: ScoringModels,
        features: Any,
        pages: list[truth.Page],
        splits: list[Split],
        others: dict | None = None,
    ) -> None:
        super().__init__(settings, models, features, others)
        self.pages = {page.page: page for page in pages}
        self.splits = splits

    def inputs(self) -> list[Path]:
        given = {Path(source).resolve() for source in self.settings.sources}
        sources = [
            source
            for source in dict.fromkeys(page.source for page in self.pages.values())
            if source.resolve() not in given
        ]
        return super().inputs() + sources

    def documents(self, resolver: Resolver, sources: list[str]) -> list[Document]:
        documents: dict[str, list[PageSource]] = {}
        for page in self.pages.values():
            found = documents.setdefault(page.document, [])
            found.append(page_source(page, len(found) + 1))
        return [Document(name, name, pages) for name, pages in documents.items()]

    def page_identity(self, page: PageSource) -> list[Any]:
        return [self.pages[page.page].checksum]

    def finish(
        self, output: Path, records: list[tuple[PageRecord, str, dict[str, Any]]], summary: dict[str, Any]
    ) -> list[Path]:
        written = super().finish(output, records, summary)
        settings: DifficultySettings = self.settings  # type: ignore[assignment]
        scores = pd.read_csv(output / SCORES, sep="\t", dtype={"page": str}).set_index("page")
        clipped = self.clipped(output)
        rows, classes, zones, lines = [], [], [], []
        place = {name: number for number, name in enumerate(self.pages)}
        for record, status, _ in sorted(records, key=lambda item: place.get(item[0].page, len(place))):
            name = record.page
            page = self.pages[name]
            row: dict[str, Any] = {
                "page": name,
                "document": page.document,
                "label": page.label,
                "image": shown_path(page.image),
                "half": name.rsplit("_", 1)[-1] if page.region is not None else "",
            }
            row["status"] = status
            if name in scores.index:
                row |= {
                    column: scores.at[name, column]
                    for column in [*self.models.names, "ensemble", "expected_elo"]
                    if column in scores.columns
                }
            row["clipped_features"] = clipped.get(name, math.nan)
            if status == "done":
                recognised = (record.recognition or {}).get("ground_truth") if record.recognition else None
                measured = measure_page(
                    page, found_layout(record), recognised, settings.zone_classes, settings.threshold, settings.checks
                )
                row |= measured.row
                classes += measured.zone_classes
                zones += measured.zones
                lines += measured.lines
            rows.append(row)
        table = pd.DataFrame(rows)
        facts = read_manuscript_table(settings.manuscript_table)
        if len(facts):
            joined = facts.drop(
                columns=[c for c in facts.columns if c in table.columns and c != "document"], errors="ignore"
            )
            table = table.join(joined, on="document")
        written.append(output / ROWS)
        table.to_csv(written[-1], sep="\t", index=False, lineterminator="\n", float_format="%.6f")
        written.append(output / ZONE_CLASSES)
        pd.DataFrame(classes, columns=["page", "class", "matched", "invented", "missed"]).to_csv(
            written[-1], sep="\t", index=False, lineterminator="\n"
        )
        written.append(write_elements(output / ZONES, zones, line=False))
        written.append(write_elements(output / LINES, lines, line=True))
        written.append(write_splits(output / SPREADS, self.splits))
        summary["difficulty"] = self.summarise(table, records)
        return written

    def clipped(self, output: Path) -> dict[str, int]:
        """How many of the geometry ranker's features lie beyond its clipping bounds, by page."""
        from layout_visual_complexity.features.extraction import read_feature_table
        from layout_visual_complexity.score import FEATURES

        model = self.models.features.model
        if model.low is None or model.high is None or not (output / FEATURES).is_file():
            return {}
        values = read_feature_table(output / FEATURES).values.reindex(columns=model.columns)
        array = values.to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            beyond = (array < model.low) | (array > model.high)
        return dict(zip(values.index, beyond.sum(axis=1).astype(int).tolist(), strict=True))

    def summarise(self, table: pd.DataFrame, records: list[tuple[PageRecord, str, dict[str, Any]]]) -> dict[str, Any]:
        recognised = [record for record, status, _ in records if status == "done" and record.recognition]
        seconds = sum(record.seconds.get("recognition", 0.0) for record in recognised)
        count = sum(
            len(record.recognition["ground_truth"]) + len(record.recognition["lines"])  # type: ignore[index]
            for record in recognised
        )
        checks: GroundTruthChecks = self.settings.checks  # type: ignore[attr-defined]
        misplaced = table[texts_misplaced(table)]
        outside = table[leaves_text_out(table, checks.outside_lines, checks.outside_share)]
        if len(outside):
            outside = outside.assign(share=outside["lines_outside"] / outside["lines_found"])
            outside = outside.sort_values("share", ascending=False, kind="stable")
        return {
            "pages": len(table),
            "pages_measured": int(table["line_f1"].notna().sum()) if "line_f1" in table else 0,
            "texts_misplaced": {
                row.page: [int(row.lines_shifted), int(row.lines_read_empty)] for row in misplaced.itertuples()
            },
            "text_left_out": {row.page: [int(row.lines_found), int(row.lines_outside)] for row in outside.itertuples()},
            "spreads": len(self.splits),
            "gutters_found": sum(split.gutter.found for split in self.splits),
            "middle": [split.page for split in self.splits if not split.gutter.found],
            "halves_left_out": [half for split in self.splits for half in split.empty],
            "lines_across": {split.page: split.crossing for split in self.splits if split.crossing},
            "lines_recognised": count,
            "recognition_seconds": seconds,
            "lines_per_second": count / seconds if seconds else None,
        }

    def report(self, summary: dict[str, Any]) -> list[str]:
        lines = super().report(summary)
        found = summary.get("difficulty", {})
        lines += [
            "",
            "## Ground truth",
            "",
            f"- Pages: {found.get('pages', 0):,}, measured {found.get('pages_measured', 0):,}.",
        ]
        lines.append(
            f"- Spreads split: {found.get('spreads', 0)}, the gutter found on {found.get('gutters_found', 0)};"
            " split in the middle: " + (", ".join(found.get("middle", [])) or "none") + "."
        )
        lines.append(
            "- Halves left out, without ground-truth lines: "
            + (", ".join(found.get("halves_left_out", [])) or "none")
            + "."
        )
        across = found.get("lines_across", {})
        lines.append(
            "- Ground-truth lines lying across a split: "
            + (
                "; ".join(
                    f"{page}: " + ", ".join(f"{line} ({share:.0%})" for line, share in items)
                    for page, items in across.items()
                )
                or "none"
            )
            + "."
        )
        if found.get("lines_per_second"):
            lines.append(
                f"- Recognised: {found['lines_recognised']:,} lines, {found['lines_per_second']:.1f} lines/s"
                " in one worker."
            )
        checks: GroundTruthChecks = self.settings.checks  # type: ignore[attr-defined]
        misplaced = found.get("texts_misplaced", {})
        lines += [
            "",
            "## Ground-truth texts that don't fit their lines",
            "",
            f"Pages with at least {checks.lines} text lines whose found line reads another text line's text (a CER "
            f"over {checks.shifted_own:.0%} against their own text, under {checks.shifted_neighbour:.0%} against the "
            "text line's just before or after), or that read as empty through their own polygon while the found "
            "line reads text. `lvc report` leaves them out of the character error rates.",
            "",
        ]
        if misplaced:
            lines += ["| Page | Shifted texts | Read empty |", "|---|---:|---:|"]
            lines += [f"| {page} | {shifted} | {empty} |" for page, (shifted, empty) in misplaced.items()]
        else:
            lines.append("None.")
        outside = found.get("text_left_out", {})
        lines += [
            "",
            "## Ground truth that may leave text out",
            "",
            f"Pages where at least {checks.outside_lines} found lines, matched to none, lie outside every "
            f"ground-truth zone, making at least {checks.outside_share:.0%} of the lines found: the neighbouring "
            "leaf, decoration, or text the ground truth doesn't hold.",
            "",
        ]
        if outside:
            lines += ["| Page | Lines found | Outside every zone | Share |", "|---|---:|---:|---:|"]
            lines += [
                f"| {page} | {total} | {count} | {count / total:.0%} |" for page, (total, count) in outside.items()
            ]
        else:
            lines.append("None.")
        return lines


POINTS = pa.list_(pa.list_(pa.int32()))


def _points(points: Any) -> list[list[int]] | None:
    if points is None:
        return None
    return [[round(float(x)), round(float(y))] for x, y in points]


def write_elements(path: Path, rows: list[dict[str, Any]], line: bool) -> Path:
    """Zones or lines with their matches, as Parquet: polygons (and baselines) as lists of `[x, y]` points in each
    page's own pixels."""
    fields = [("page", pa.string()), ("source", pa.string()), ("id", pa.string()), ("class", pa.string())]
    if line:
        fields += [("zone_class", pa.string())]
    fields += [("polygon", POINTS)]
    if line:
        fields += [("baseline", POINTS), ("text", pa.string()), ("recognised", pa.string())]
    fields += [("match", pa.string()), ("iou", pa.float32())]
    if line:
        fields += [("baseline_distance", pa.float32())]
    schema = pa.schema(fields)
    cleaned = []
    for row in rows:
        item = {name: row.get(name) for name in schema.names}
        item["polygon"] = _points(item["polygon"])
        if line:
            item["baseline"] = _points(item["baseline"])
            distance = item["baseline_distance"]
            item["baseline_distance"] = None if distance is None or not math.isfinite(distance) else distance
        cleaned.append(item)
    partial = path.with_name(path.name + ".part")
    pq.write_table(pa.Table.from_pylist(cleaned, schema=schema), partial, compression="zstd")
    partial.replace(path)
    return path


def write_splits(path: Path, splits: list[Split]) -> Path:
    """Each spread's split: where, whether its gutter was found, the halves kept and left out, the lines across."""
    rows = [
        {
            "page": split.page,
            "gutter": split.gutter.x,
            "found": split.gutter.found,
            "depth": round(split.gutter.depth, 4),
            "halves": " ".join(half.page for half in split.halves),
            "left_out": " ".join(split.empty),
            "lines_across": " ".join(f"{line}:{share:.3f}" for line, share in split.crossing),
        }
        for split in splits
    ]
    columns = ["page", "gutter", "found", "depth", "halves", "left_out", "lines_across"]
    pd.DataFrame(rows, columns=columns).to_csv(path, sep="\t", index=False, lineterminator="\n")
    return path


def recognition_pages(pages: list[truth.Page]) -> dict[str, PageLines]:
    """What recognition reads on each transcribed page: its image and part, and its text lines with a reference."""
    found = {}
    for page in pages:
        lines = [
            (line.id, line.polygon, line.baseline)
            for line in page.lines
            if line.cls in TEXT_LINES and normalise(line.text)
        ]
        if page.transcribed and lines:
            found[page.page] = PageLines(str(page.image), page.region, lines)
    return found


def run_difficulty(
    settings: DifficultySettings, steps: dict[str, Step] | None = None, models: ScoringModels | None = None
) -> dict[str, Any]:
    """Measure the sources' pages into `output`. Running it again resumes. Returns the run's summary."""
    from layout_visual_complexity.pipeline.monitor import cores

    if not settings.sources:
        raise ValueError("no sources: give ALTO files, folders, lists or a metadata.jsonl")
    models = models or ScoringModels.load(settings.models)
    features = feature_families(models, settings)
    pages, splits = read_pages(settings, settings.jobs or cores())
    logger.info("difficulty: %d pages, %d spreads split", len(pages), len(splits))
    if steps is None:
        tokens = TokenStepSettings(models.files()[models.names[1]], local_only=settings.local_only)
        steps = segmentation_steps(settings.folio_box, settings.folio, settings.regions, settings.lines)
        steps |= scoring_steps(features, tokens)
        steps["recognition"] = RecognitionStep(settings.recognition, recognition_pages(pages))
    extension = Difficulty(settings, models, features, pages, splits)
    summary = run_segment(settings, steps, None, extension)
    found = summary.get("difficulty", {})
    print(f"{found.get('pages_measured', 0):,} pages measured -> {Path(settings.output) / ROWS}")
    return summary


def read_rows(folder: Path) -> pd.DataFrame:
    """A difficulty run's rows."""
    return pd.read_csv(Path(folder) / ROWS, sep="\t", dtype={"page": str, "document": str, "label": str})


def texts_misplaced(rows: pd.DataFrame) -> pd.Series:
    """The pages whose ground-truth texts the run's checks flagged as not fitting their lines."""
    if "texts_misplaced" not in rows:
        return pd.Series(False, index=rows.index)
    return rows["texts_misplaced"].map(lambda value: str(value) == "True")


def leaves_text_out(
    rows: pd.DataFrame,
    lines: int = GroundTruthChecks.outside_lines,
    share: float = GroundTruthChecks.outside_share,
) -> pd.Series:
    """The pages whose ground truth may leave text out: at least `lines` found lines, matched to none, outside every
    ground-truth zone, making at least `share` of the lines found."""
    if "lines_outside" not in rows:
        return pd.Series(False, index=rows.index)
    found = rows["lines_found"].astype(float)
    outside = rows["lines_outside"].astype(float)
    return (outside >= lines) & (outside / found.where(found > 0) >= share)
